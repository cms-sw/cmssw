// -*- C++ -*-
//
//
// dummy producer for a TkEm
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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

#include "DataFormats/Math/interface/deltaR.h"

#include <string>
#include <cmath>

static constexpr float EtaECal = 1.479;
static constexpr float REcal = 129.;
static constexpr float ZEcal = 315.4;
using namespace l1t;
//
// class declaration
//

class L1TkEmParticleProducer : public edm::global::EDProducer<> {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;

  explicit L1TkEmParticleProducer(const edm::ParameterSet&);
  ~L1TkEmParticleProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  float CorrectedEta(float eta, float zv) const;

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  std::string label_;

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

  const edm::EDGetTokenT<EGammaBxCollection> egToken_;
  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > trackToken_;
  const edm::EDGetTokenT<TkPrimaryVertexCollection> vertexToken_;
};

//
// constructors and destructor
//
L1TkEmParticleProducer::L1TkEmParticleProducer(const edm::ParameterSet& iConfig)
    : egToken_(consumes<EGammaBxCollection>(iConfig.getParameter<edm::InputTag>("L1EGammaInputTag"))),
      trackToken_(consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > >(
          iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
      vertexToken_(consumes<TkPrimaryVertexCollection>(iConfig.getParameter<edm::InputTag>("L1VertexInputTag"))) {
  label_ = iConfig.getParameter<std::string>("label");  // label of the collection produced
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
  deltaZMax_ = (float)iConfig.getParameter<double>("DeltaZMax");
  // cut applied on the isolation (if this number is <= 0, no cut is applied)
  isoCut_ = (float)iConfig.getParameter<double>("IsoCut");
  relativeIsolation_ = iConfig.getParameter<bool>("RelativeIsolation");

  produces<TkEmCollection>(label_);
}

L1TkEmParticleProducer::~L1TkEmParticleProducer() {}

// ------------ method called to produce the data  ------------
void L1TkEmParticleProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  auto result = std::make_unique<TkEmCollection>();

  // the L1EGamma objects
  edm::Handle<EGammaBxCollection> eGammaHandle;
  iEvent.getByToken(egToken_, eGammaHandle);
  EGammaBxCollection eGammaCollection = (*eGammaHandle.product());
  EGammaBxCollection::const_iterator egIter;

  // the L1Tracks
  edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > L1TTTrackHandle;
  iEvent.getByToken(trackToken_, L1TTTrackHandle);

  // the primary vertex (used only if primaryVtxConstrain_ = true)
  float zvtxL1tk = -999;
  edm::Handle<TkPrimaryVertexCollection> L1VertexHandle;
  iEvent.getByToken(vertexToken_, L1VertexHandle);
  bool primaryVtxConstrain = primaryVtxConstrain_;
  if (!L1VertexHandle.isValid()) {
    LogWarning("L1TkEmParticleProducer")
        << "Warning: TkPrimaryVertexCollection not found in the event. Won't use any PrimaryVertex constraint."
        << std::endl;
    primaryVtxConstrain = false;
  } else {
    std::vector<TkPrimaryVertex>::const_iterator vtxIter = L1VertexHandle->begin();
    // by convention, the first vertex in the collection is the one that should
    // be used by default
    zvtxL1tk = vtxIter->zvertex();
  }

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

    for (const auto& track : *L1TTTrackHandle) {
      float Pt = track.momentum().perp();
      float Eta = track.momentum().eta();
      float Phi = track.momentum().phi();
      float z = track.POCA().z();
      if (fabs(z) > zMax_)
        continue;
      if (Pt < pTMinTra_)
        continue;
      float chi2 = track.chi2();
      if (chi2 > chi2Max_)
        continue;

      float dr = reco::deltaR(Eta, Phi, eta, phi);
      if (dr < dRMax_ && dr >= dRMin_) {
        sumPt += Pt;
      }

      if (zvtxL1tk > -999 && std::abs(z - zvtxL1tk) >= deltaZMax_)
        continue;  // Now, PV constrained trackSum:

      dr = reco::deltaR(Eta, Phi, etaPV, phi);  // recompute using the corrected eta

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
    if (primaryVtxConstrain) {
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

  iEvent.put(std::move(result), label_);
}

// --------------------------------------------------------------------------------------

float L1TkEmParticleProducer::CorrectedEta(float eta, float zv) const {
  // Correct the eta of the L1EG object once we know the zvertex

  bool IsBarrel = (std::abs(eta) < EtaECal);

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
