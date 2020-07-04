// -*- C++ -*-
//
//
// dummy producer for a TkEm
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"

#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"

#include "DataFormats/Math/interface/LorentzVector.h"

// for L1Tracks:
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <string>
#include <cmath>

static constexpr float EtaECal = 1.479;
static constexpr float REcal = 129.;
static constexpr float ZEcal = 315.4;
using namespace l1t;
//
// class declaration
//

class L1TkEmParticleProducer : public edm::EDProducer {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;

  explicit L1TkEmParticleProducer(const edm::ParameterSet&);
  ~L1TkEmParticleProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  float DeltaPhi(float phi1, float phi2);
  float deltaR(float eta1, float eta2, float phi1, float phi2);
  float CorrectedEta(float eta, float zv);

private:
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  //virtual void beginRun(edm::Run&, edm::EventSetup const&);
  //virtual void endRun(edm::Run&, edm::EventSetup const&);
  //virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  //virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

  // ----------member data ---------------------------

  std::string label;

  float etMin_;  // min ET in GeV of L1EG objects

  float zMax_;  // |z_track| < zMax_ in cm
  float chi2Max_;
  float dRMin_;
  float dRMax_;
  float pTMinTra_;
  bool primaryVtxConstrain_;  // use the primary vertex (default = false)
                              //bool DeltaZConstrain;	// use z = z of the leading track within DR < dRMax_;
  float deltaZMax_;           // | z_track - z_primaryvtx | < deltaZMax_ in cm.
                              // Used only when primaryVtxConstrain_ = True.
  float isoCut_;
  bool relativeIsolation_;

  const edm::EDGetTokenT<EGammaBxCollection> egToken;
  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > trackToken;
  const edm::EDGetTokenT<TkPrimaryVertexCollection> vertexToken;
};

//
// constructors and destructor
//
L1TkEmParticleProducer::L1TkEmParticleProducer(const edm::ParameterSet& iConfig)
    : egToken(consumes<EGammaBxCollection>(iConfig.getParameter<edm::InputTag>("L1EGammaInputTag"))),
      trackToken(consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > >(
          iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
      vertexToken(consumes<TkPrimaryVertexCollection>(iConfig.getParameter<edm::InputTag>("L1VertexInputTag"))) {
  label = iConfig.getParameter<std::string>("label");  // label of the collection produced
  // e.g. EG or IsoEG if all objects are kept
  // EGIsoTrk or IsoEGIsoTrk if only the EG or IsoEG
  // objects that pass a cut RelIso < isoCut_ are written
  // in the new collection.

  etMin_ = (float)iConfig.getParameter<double>("ETmin");

  // parameters for the calculation of the isolation :
  zMax_ = (float)iConfig.getParameter<double>("ZMAX");
  chi2Max_ = (float)iConfig.getParameter<double>("CHI2MAX");
  pTMinTra_ = (float)iConfig.getParameter<double>("PTMINTRA");
  dRMin_ = (float)iConfig.getParameter<double>("DRmin");
  dRMax_ = (float)iConfig.getParameter<double>("DRmax");
  primaryVtxConstrain_ = iConfig.getParameter<bool>("PrimaryVtxConstrain");
  //DeltaZConstrain = iConfig.getParameter<bool>("DeltaZConstrain");
  deltaZMax_ = (float)iConfig.getParameter<double>("DeltaZMax");
  // cut applied on the isolation (if this number is <= 0, no cut is applied)
  isoCut_ = (float)iConfig.getParameter<double>("IsoCut");
  relativeIsolation_ = iConfig.getParameter<bool>("RelativeIsolation");

  produces<TkEmCollection>(label);
}

L1TkEmParticleProducer::~L1TkEmParticleProducer() {}

// ------------ method called to produce the data  ------------
void L1TkEmParticleProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::unique_ptr<TkEmCollection> result(new TkEmCollection);

  // the L1EGamma objects
  edm::Handle<EGammaBxCollection> eGammaHandle;
  iEvent.getByToken(egToken, eGammaHandle);
  EGammaBxCollection eGammaCollection = (*eGammaHandle.product());
  EGammaBxCollection::const_iterator egIter;

  // the L1Tracks
  edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > L1TTTrackHandle;
  iEvent.getByToken(trackToken, L1TTTrackHandle);
  L1TTTrackCollectionType::const_iterator trackIter;

  // the primary vertex (used only if primaryVtxConstrain_ = true)
  float zvtxL1tk = -999;
  //if (primaryVtxConstrain_) {
  edm::Handle<TkPrimaryVertexCollection> L1VertexHandle;
  iEvent.getByToken(vertexToken, L1VertexHandle);
  if (!L1VertexHandle.isValid()) {
    LogWarning("L1TkEmParticleProducer")
        << "Warning: TkPrimaryVertexCollection not found in the event. Won't use any PrimaryVertex constraint."
        << std::endl;
    primaryVtxConstrain_ = false;
  } else {
    std::vector<TkPrimaryVertex>::const_iterator vtxIter = L1VertexHandle->begin();
    // by convention, the first vertex in the collection is the one that should
    // be used by default
    zvtxL1tk = vtxIter->zvertex();
  }
  //}

  if (!L1TTTrackHandle.isValid()) {
    LogError("L1TkEmParticleProducer") << "\nWarning: L1TTTrackCollectionType not found in the event. Exit."
                                       << std::endl;
    return;
  }

  // Now loop over the L1EGamma objects

  if (!eGammaHandle.isValid()) {
    LogError("L1TkEmParticleProducer") << "\nWarning: L1EmCollection not found in the event. Exit." << std::endl;
    return;
  }

  int ieg = 0;
  for (egIter = eGammaCollection.begin(0); egIter != eGammaCollection.end(0); ++egIter)  // considering BX = only
  {
    edm::Ref<EGammaBxCollection> EGammaRef(eGammaHandle, ieg);
    ieg++;

    float eta = egIter->eta();
    // The eta of the L1EG object is seen from (0,0,0).
    // if primaryVtxConstrain_ = true, and for the PV constrained iso, use the zvtxL1tk to correct the eta(L1EG)
    // that is used in the calculation of DeltaR.
    float etaPV = CorrectedEta((float)eta, zvtxL1tk);

    float phi = egIter->phi();
    float et = egIter->et();

    if (et < etMin_)
      continue;

    // calculate the isolation of the L1EG object with
    // respect to L1Tracks.

    float trkisol = -999;
    float sumPt = 0;
    float sumPtPV = 0;
    float trkisolPV = -999;

    //std::cout << " here an EG w et = " << et << std::endl;

    //float z_leadingTrack = -999;
    //float Pt_leadingTrack = -999;

    /*
	if (DeltaZConstrain) {
	// first loop over the tracks to find the leading one in DR < dRMax_
	for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {
	float Pt = trackIter->momentum().perp();
	float Eta = trackIter->momentum().eta();
	float Phi = trackIter->momentum().phi();
	float z  = trackIter->POCA().z();
	if (fabs(z) > zMax_) continue;
	if (Pt < pTMinTra_) continue;
	float chi2 = trackIter->chi2();
	if (chi2 > chi2Max_) continue;
	float dr = deltaR(Eta, eta, Phi,phi);
	if (dr < dRMax_) {
	if (Pt > Pt_leadingTrack) {
	Pt_leadingTrack = Pt;
	z_leadingTrack = z;
	}
	}
	} // end loop over the tracks
	} // endif DeltaZConstrain
      */

    for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {
      float Pt = trackIter->momentum().perp();
      float Eta = trackIter->momentum().eta();
      float Phi = trackIter->momentum().phi();
      float z = trackIter->POCA().z();
      if (fabs(z) > zMax_)
        continue;
      if (Pt < pTMinTra_)
        continue;
      float chi2 = trackIter->chi2();
      if (chi2 > chi2Max_)
        continue;

      float dr = deltaR(Eta, eta, Phi, phi);
      if (dr < dRMax_ && dr >= dRMin_) {
        //std::cout << " a track in the cone, z Pt = " << z << " " << Pt << std::endl;
        sumPt += Pt;
      }

      if (zvtxL1tk > -999 && fabs(z - zvtxL1tk) >= deltaZMax_)
        continue;  // Now, PV constrained trackSum:

      dr = deltaR(Eta, etaPV, Phi, phi);  // recompute using the corrected eta

      if (dr < dRMax_ && dr >= dRMin_) {
        sumPtPV += Pt;
      }
    }  // end loop over tracks

    if (relativeIsolation_) {
      if (et > 0) {
        trkisol = sumPt / et;
        trkisolPV = sumPtPV / et;
      }       // relative isolation
    } else {  // absolute isolation
      trkisol = sumPt;
      trkisolPV = sumPtPV;
    }

    float isolation = trkisol;
    if (primaryVtxConstrain_) {
      isolation = trkisolPV;
    }

    const math::XYZTLorentzVector P4 = egIter->p4();
    TkEm trkEm(P4, EGammaRef, trkisol, trkisolPV);

    if (isoCut_ <= 0) {
      // write the L1TkEm particle to the collection,
      // irrespective of its relative isolation
      result->push_back(trkEm);
    } else {
      // the object is written to the collection only
      // if it passes the isolation cut
      if (isolation <= isoCut_)
        result->push_back(trkEm);
    }

  }  // end loop over EGamma objects

  iEvent.put(std::move(result), label);
}

// --------------------------------------------------------------------------------------

float L1TkEmParticleProducer::CorrectedEta(float eta, float zv) {
  // Correct the eta of the L1EG object once we know the zvertex

  bool IsBarrel = (fabs(eta) < EtaECal);

  float theta = 2. * atan(exp(-eta));
  if (theta < 0)
    theta = theta + M_PI;
  float tantheta = tan(theta);

  float delta;
  if (IsBarrel) {
    delta = REcal / tantheta;
  } else {
    if (theta > 0)
      delta = ZEcal;
    if (theta < 0)
      delta = -ZEcal;
  }

  float tanthetaprime = delta * tantheta / (delta - zv);

  float thetaprime = atan(tanthetaprime);
  if (thetaprime < 0)
    thetaprime = thetaprime + M_PI;

  float etaprime = -log(tan(thetaprime / 2.));
  return etaprime;
}

// --------------------------------------------------------------------------------------

float L1TkEmParticleProducer::DeltaPhi(float phi1, float phi2) {
  // dPhi between 0 and Pi
  float dphi = phi1 - phi2;
  if (dphi < 0)
    dphi = dphi + 2. * M_PI;
  if (dphi > M_PI)
    dphi = 2. * M_PI - dphi;
  return dphi;
}

// --------------------------------------------------------------------------------------

float L1TkEmParticleProducer::deltaR(float eta1, float eta2, float phi1, float phi2) {
  float deta = eta1 - eta2;
  float dphi = DeltaPhi(phi1, phi2);
  float DR = sqrt(deta * deta + dphi * dphi);
  return DR;
}

// ------------ method called once each job just before starting event loop  ------------
void L1TkEmParticleProducer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void L1TkEmParticleProducer::endJob() {}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TkEmParticleProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkEmParticleProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkEmParticleProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkEmParticleProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TkEmParticleProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkEmParticleProducer);
