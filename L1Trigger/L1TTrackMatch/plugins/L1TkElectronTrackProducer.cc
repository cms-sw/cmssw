// -*- C++ -*-
//
/**\class L1TkElectronTrackMatchAlgo

 Description: Producer of a TkElectron, for the algorithm matching a L1Track to the L1EG object

 Implementation:
     [Notes on implementation]
*/
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// Matching Algorithm
#include "L1Trigger/L1TTrackMatch/interface/L1TkElectronTrackMatchAlgo.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkElectronEtComparator.h"
#include "L1Trigger/L1TTrackMatch/interface/pTFrom2Stubs.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"

#include <string>

static constexpr float EB_MaxEta = 0.9;

using namespace l1t;

//
// class declaration
//

class L1TkElectronTrackProducer : public edm::stream::EDProducer<> {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;

  explicit L1TkElectronTrackProducer(const edm::ParameterSet&);
  ~L1TkElectronTrackProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  //void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  //void endJob() override;

  float isolation(const edm::Handle<L1TTTrackCollectionType>& trkHandle, int match_index);
  double getPtScaledCut(double pt, std::vector<double>& parameters);
  bool selectMatchedTrack(double& d_r, double& d_phi, double& d_eta, double& tk_pt, float& eg_eta);

  // ----------member data ---------------------------
  std::string label;

  const float etMin_;  // min ET in GeV of L1EG objects
  const float pTMinTra_;
  const float dRMin_;
  const float dRMax_;
  float deltaZ_;  // | z_track - z_ref_track | < deltaZ_ in cm.
                  // Used only when primaryVtxConstrain_ = True.
  const float maxChi2IsoTracks_;
  const unsigned int minNStubsIsoTracks_;
  float isoCut_;
  bool relativeIsolation_;
  bool primaryVtxConstrain_;  // use the primary vertex (default = false)
  float trkQualityChi2_;
  float trkQualityPtMin_;
  bool useTwoStubsPT_;
  bool useClusterET_;  // use cluster et to extrapolate tracks
  std::vector<double> dPhiCutoff_;
  std::vector<double> dRCutoff_;
  std::vector<double> dEtaCutoff_;
  std::string matchType_;

  const edm::EDGetTokenT<EGammaBxCollection> egToken_;
  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> trackToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken_;
};

//
// constructors and destructor
//
L1TkElectronTrackProducer::L1TkElectronTrackProducer(const edm::ParameterSet& iConfig)
    : label(iConfig.getParameter<std::string>("label")),
      etMin_((float)iConfig.getParameter<double>("ETmin")),
      pTMinTra_((float)iConfig.getParameter<double>("PTMINTRA")),
      dRMin_((float)iConfig.getParameter<double>("DRmin")),
      dRMax_((float)iConfig.getParameter<double>("DRmax")),
      deltaZ_((float)iConfig.getParameter<double>("DeltaZ")),
      maxChi2IsoTracks_(iConfig.getParameter<double>("maxChi2IsoTracks")),
      minNStubsIsoTracks_(iConfig.getParameter<int>("minNStubsIsoTracks")),
      isoCut_((float)iConfig.getParameter<double>("IsoCut")),
      relativeIsolation_(iConfig.getParameter<bool>("RelativeIsolation")),
      trkQualityChi2_((float)iConfig.getParameter<double>("TrackChi2")),
      trkQualityPtMin_((float)iConfig.getParameter<double>("TrackMinPt")),
      useTwoStubsPT_(iConfig.getParameter<bool>("useTwoStubsPT")),
      useClusterET_(iConfig.getParameter<bool>("useClusterET")),
      dPhiCutoff_(iConfig.getParameter<std::vector<double>>("TrackEGammaDeltaPhi")),
      dRCutoff_(iConfig.getParameter<std::vector<double>>("TrackEGammaDeltaR")),
      dEtaCutoff_(iConfig.getParameter<std::vector<double>>("TrackEGammaDeltaEta")),
      matchType_(iConfig.getParameter<std::string>("TrackEGammaMatchType")),
      egToken_(consumes<EGammaBxCollection>(iConfig.getParameter<edm::InputTag>("L1EGammaInputTag"))),
      trackToken_(consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>(
          iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
      tGeomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()) {
  produces<TkElectronCollection>(label);
}

L1TkElectronTrackProducer::~L1TkElectronTrackProducer() {}

// ------------ method called to produce the data  ------------
void L1TkElectronTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::unique_ptr<TkElectronCollection> result(new TkElectronCollection);

  // geometry needed to call pTFrom2Stubs
  const TrackerGeometry& tGeom = iSetup.getData(tGeomToken_);
  const TrackerGeometry* tGeometry = &tGeom;

  // the L1EGamma objects
  edm::Handle<EGammaBxCollection> eGammaHandle;
  iEvent.getByToken(egToken_, eGammaHandle);
  EGammaBxCollection eGammaCollection = (*eGammaHandle.product());
  EGammaBxCollection::const_iterator egIter;

  // the L1Tracks
  edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken_, L1TTTrackHandle);
  L1TTTrackCollectionType::const_iterator trackIter;

  if (!eGammaHandle.isValid()) {
    throw cms::Exception("L1TkElectronTrackProducer")
        << "\nWarning: L1EmCollection not found in the event. Exit" << std::endl;
    return;
  }
  if (!L1TTTrackHandle.isValid()) {
    throw cms::Exception("L1TkElectronTrackProducer")
        << "\nWarning: L1TTTrackCollectionType not found in the event. Exit." << std::endl;
    return;
  }

  int ieg = 0;
  for (egIter = eGammaCollection.begin(0); egIter != eGammaCollection.end(0); ++egIter) {  // considering BX = only
    edm::Ref<EGammaBxCollection> EGammaRef(eGammaHandle, ieg);
    ieg++;

    float e_ele = egIter->energy();
    float eta_ele = egIter->eta();
    float et_ele = 0;
    float cosh_eta_ele = cosh(eta_ele);
    if (cosh_eta_ele > 0.0)
      et_ele = e_ele / cosh_eta_ele;
    else
      et_ele = -1.0;
    if (etMin_ > 0.0 && et_ele <= etMin_)
      continue;
    // match the L1EG object with a L1Track
    float drmin = 999;
    int itr = 0;
    int itrack = -1;
    for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {
      edm::Ptr<L1TTTrackType> L1TrackPtr(L1TTTrackHandle, itr);
      double trkPt_fit = trackIter->momentum().perp();
      double trkPt_stubs = pTFrom2Stubs::pTFrom2(trackIter, tGeometry);
      double trkPt = trkPt_fit;
      if (useTwoStubsPT_)
        trkPt = trkPt_stubs;

      if (trkPt > trkQualityPtMin_ && trackIter->chi2() < trkQualityChi2_) {
        double dPhi = 99.;
        double dR = 99.;
        double dEta = 99.;
        if (useClusterET_)
          L1TkElectronTrackMatchAlgo::doMatchClusterET(egIter, L1TrackPtr, dPhi, dR, dEta);
        else
          L1TkElectronTrackMatchAlgo::doMatch(egIter, L1TrackPtr, dPhi, dR, dEta);
        if (dR < drmin && selectMatchedTrack(dR, dPhi, dEta, trkPt, eta_ele)) {
          drmin = dR;
          itrack = itr;
        }
      }
      itr++;
    }
    if (itrack >= 0) {
      edm::Ptr<L1TTTrackType> matchedL1TrackPtr(L1TTTrackHandle, itrack);

      const math::XYZTLorentzVector P4 = egIter->p4();
      float trkisol = isolation(L1TTTrackHandle, itrack);
      if (relativeIsolation_ && et_ele > 0.0) {  // relative isolation
        trkisol = trkisol / et_ele;
      }

      TkElectron trkEm(P4, EGammaRef, matchedL1TrackPtr, trkisol);

      trkEm.setTrackCurvature(matchedL1TrackPtr->rInv());  // should this have npars? 4? 5?

      //std::cout<<matchedL1TrackPtr->rInv()<<"  "<<matchedL1TrackPtr->rInv(4)<<"   "<<matchedL1TrackPtr->rInv()<<std::endl;

      if (isoCut_ <= 0) {
        // write the L1TkEm particle to the collection,
        // irrespective of its relative isolation
        result->push_back(trkEm);
      } else {
        // the object is written to the collection only
        // if it passes the isolation cut
        if (trkisol <= isoCut_)
          result->push_back(trkEm);
      }
    }

  }  // end loop over EGamma objects

  iEvent.put(std::move(result), label);
}

// ------------ method called once each job just before starting event loop  ------------
//void L1TkElectronTrackProducer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
//void L1TkElectronTrackProducer::endJob() {}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TkElectronTrackProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkElectronTrackProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkElectronTrackProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkElectronTrackProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TkElectronTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  {
    // L1TkElectrons
    edm::ParameterSetDescription desc;
    desc.add<std::string>("label", "EG");
    desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag("simCaloStage2Digis"));
    desc.add<double>("ETmin", -1.0);
    desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
    desc.add<double>("TrackChi2", 10000000000.0);
    desc.add<double>("TrackMinPt", 10.0);
    desc.add<bool>("useTwoStubsPT", false);
    desc.add<bool>("useClusterET", false);
    desc.add<std::string>("TrackEGammaMatchType", "PtDependentCut");
    desc.add<std::vector<double>>("TrackEGammaDeltaPhi",
                                  {
                                      0.07,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaR",
                                  {
                                      0.08,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaEta",
                                  {
                                      10000000000.0,
                                      0.0,
                                      0.0,
                                  });
    desc.add<bool>("RelativeIsolation", true);
    desc.add<double>("IsoCut", -0.1);
    desc.add<double>("PTMINTRA", 2.0);
    desc.add<double>("DRmin", 0.03);
    desc.add<double>("DRmax", 0.2);
    desc.add<double>("maxChi2IsoTracks", 10000000000.0);
    desc.add<int>("minNStubsIsoTracks", 0);
    desc.add<double>("DeltaZ", 0.6);
    descriptions.add("L1TkElectrons", desc);
    // or use the following to generate the label from the module's C++ type
    //descriptions.addWithDefaultLabel(desc);
  }
  {
    // L1TkIsoElectrons
    edm::ParameterSetDescription desc;
    desc.add<std::string>("label", "EG");
    desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag("simCaloStage2Digis"));
    desc.add<double>("ETmin", -1.0);
    desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
    desc.add<double>("TrackChi2", 10000000000.0);
    desc.add<double>("TrackMinPt", 10.0);
    desc.add<bool>("useTwoStubsPT", false);
    desc.add<bool>("useClusterET", false);
    desc.add<std::string>("TrackEGammaMatchType", "PtDependentCut");
    desc.add<std::vector<double>>("TrackEGammaDeltaPhi",
                                  {
                                      0.07,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaR",
                                  {
                                      0.08,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaEta",
                                  {
                                      10000000000.0,
                                      0.0,
                                      0.0,
                                  });
    desc.add<bool>("RelativeIsolation", true);
    desc.add<double>("IsoCut", 0.1);
    desc.add<double>("PTMINTRA", 2.0);
    desc.add<double>("DRmin", 0.03);
    desc.add<double>("DRmax", 0.2);
    desc.add<double>("maxChi2IsoTracks", 10000000000.0);
    desc.add<int>("minNStubsIsoTracks", 0);
    desc.add<double>("DeltaZ", 0.6);
    descriptions.add("L1TkIsoElectrons", desc);
    // or use the following to generate the label from the module's C++ type
    //descriptions.addWithDefaultLabel(desc);
  }
  {
    // L1TkElectronsLoose
    edm::ParameterSetDescription desc;
    desc.add<std::string>("label", "EG");
    desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag("simCaloStage2Digis"));
    desc.add<double>("ETmin", -1.0);
    desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
    desc.add<double>("TrackChi2", 10000000000.0);
    desc.add<double>("TrackMinPt", 3.0);
    desc.add<bool>("useTwoStubsPT", false);
    desc.add<bool>("useClusterET", false);
    desc.add<std::string>("TrackEGammaMatchType", "PtDependentCut");
    desc.add<std::vector<double>>("TrackEGammaDeltaPhi",
                                  {
                                      0.07,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaR",
                                  {
                                      0.12,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaEta",
                                  {
                                      10000000000.0,
                                      0.0,
                                      0.0,
                                  });
    desc.add<bool>("RelativeIsolation", true);
    desc.add<double>("IsoCut", -0.1);
    desc.add<double>("PTMINTRA", 2.0);
    desc.add<double>("DRmin", 0.03);
    desc.add<double>("DRmax", 0.2);
    desc.add<double>("maxChi2IsoTracks", 10000000000.0);
    desc.add<int>("minNStubsIsoTracks", 0);
    desc.add<double>("DeltaZ", 0.6);
    descriptions.add("L1TkElectronsLoose", desc);
    // or use the following to generate the label from the module's C++ type
    //descriptions.addWithDefaultLabel(desc);
  }
  {
    // L1TkElectronsCrystal
    edm::ParameterSetDescription desc;
    desc.add<std::string>("label", "EG");
    desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag("L1EGammaClusterEmuProducer"));
    desc.add<double>("ETmin", -1.0);
    desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
    desc.add<double>("TrackChi2", 10000000000.0);
    desc.add<double>("TrackMinPt", 10.0);
    desc.add<bool>("useTwoStubsPT", false);
    desc.add<bool>("useClusterET", false);
    desc.add<std::string>("TrackEGammaMatchType", "PtDependentCut");
    desc.add<std::vector<double>>("TrackEGammaDeltaPhi",
                                  {
                                      0.07,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaR",
                                  {
                                      0.08,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaEta",
                                  {
                                      10000000000.0,
                                      0.0,
                                      0.0,
                                  });
    desc.add<bool>("RelativeIsolation", true);
    desc.add<double>("IsoCut", -0.1);
    desc.add<double>("PTMINTRA", 2.0);
    desc.add<double>("DRmin", 0.03);
    desc.add<double>("DRmax", 0.2);
    desc.add<double>("maxChi2IsoTracks", 10000000000.0);
    desc.add<int>("minNStubsIsoTracks", 0);
    desc.add<double>("DeltaZ", 0.6);
    descriptions.add("L1TkElectronsCrystal", desc);
    // or use the following to generate the label from the module's C++ type
    //descriptions.addWithDefaultLabel(desc);
  }
  {
    // L1TkIsoElectronsCrystal
    edm::ParameterSetDescription desc;
    desc.add<std::string>("label", "EG");
    desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag("L1EGammaClusterEmuProducer"));
    desc.add<double>("ETmin", -1.0);
    desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
    desc.add<double>("TrackChi2", 10000000000.0);
    desc.add<double>("TrackMinPt", 10.0);
    desc.add<bool>("useTwoStubsPT", false);
    desc.add<bool>("useClusterET", false);
    desc.add<std::string>("TrackEGammaMatchType", "PtDependentCut");
    desc.add<std::vector<double>>("TrackEGammaDeltaPhi",
                                  {
                                      0.07,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaR",
                                  {
                                      0.08,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaEta",
                                  {
                                      10000000000.0,
                                      0.0,
                                      0.0,
                                  });
    desc.add<bool>("RelativeIsolation", true);
    desc.add<double>("IsoCut", 0.1);
    desc.add<double>("PTMINTRA", 2.0);
    desc.add<double>("DRmin", 0.03);
    desc.add<double>("DRmax", 0.2);
    desc.add<double>("maxChi2IsoTracks", 10000000000.0);
    desc.add<int>("minNStubsIsoTracks", 0);
    desc.add<double>("DeltaZ", 0.6);
    descriptions.add("L1TkIsoElectronsCrystal", desc);
    // or use the following to generate the label from the module's C++ type
    //descriptions.addWithDefaultLabel(desc);
  }
  {
    // L1TkElectronsLooseCrystal
    edm::ParameterSetDescription desc;
    desc.add<std::string>("label", "EG");
    desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag("L1EGammaClusterEmuProducer"));
    desc.add<double>("ETmin", -1.0);
    desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
    desc.add<double>("TrackChi2", 10000000000.0);
    desc.add<double>("TrackMinPt", 3.0);
    desc.add<bool>("useTwoStubsPT", false);
    desc.add<bool>("useClusterET", false);
    desc.add<std::string>("TrackEGammaMatchType", "PtDependentCut");
    desc.add<std::vector<double>>("TrackEGammaDeltaPhi",
                                  {
                                      0.07,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaR",
                                  {
                                      0.12,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaEta",
                                  {
                                      10000000000.0,
                                      0.0,
                                      0.0,
                                  });
    desc.add<bool>("RelativeIsolation", true);
    desc.add<double>("IsoCut", -0.1);
    desc.add<double>("PTMINTRA", 2.0);
    desc.add<double>("DRmin", 0.03);
    desc.add<double>("DRmax", 0.2);
    desc.add<double>("maxChi2IsoTracks", 10000000000.0);
    desc.add<int>("minNStubsIsoTracks", 0);
    desc.add<double>("DeltaZ", 0.6);
    descriptions.add("L1TkElectronsLooseCrystal", desc);
    // or use the following to generate the label from the module's C++ type
    //descriptions.addWithDefaultLabel(desc);
  }
  {
    // L1TkElectronsEllipticMatchCrystal
    edm::ParameterSetDescription desc;
    desc.add<std::string>("label", "EG");
    desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag("L1EGammaClusterEmuProducer"));
    desc.add<double>("ETmin", -1.0);
    desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
    desc.add<double>("TrackChi2", 10000000000.0);
    desc.add<double>("TrackMinPt", 10.0);
    desc.add<bool>("useTwoStubsPT", false);
    desc.add<bool>("useClusterET", false);
    desc.add<std::string>("TrackEGammaMatchType", "EllipticalCut");
    desc.add<std::vector<double>>("TrackEGammaDeltaPhi",
                                  {
                                      0.07,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaR",
                                  {
                                      0.08,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaEta",
                                  {
                                      0.015,
                                      0.025,
                                      10000000000.0,
                                  });
    desc.add<bool>("RelativeIsolation", true);
    desc.add<double>("IsoCut", -0.1);
    desc.add<double>("PTMINTRA", 2.0);
    desc.add<double>("DRmin", 0.03);
    desc.add<double>("DRmax", 0.2);
    desc.add<double>("maxChi2IsoTracks", 10000000000.0);
    desc.add<int>("minNStubsIsoTracks", 0);
    desc.add<double>("DeltaZ", 0.6);
    descriptions.add("L1TkElectronsEllipticMatchCrystal", desc);
    // or use the following to generate the label from the module's C++ type
    //descriptions.addWithDefaultLabel(desc);
  }
  {
    // L1TkElectronsHGC
    edm::ParameterSetDescription desc;
    desc.add<std::string>("label", "EG");
    desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag("l1EGammaEEProducer", "L1EGammaCollectionBXVWithCuts"));
    desc.add<double>("ETmin", -1.0);
    desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
    desc.add<double>("TrackChi2", 10000000000.0);
    desc.add<double>("TrackMinPt", 10.0);
    desc.add<bool>("useTwoStubsPT", false);
    desc.add<bool>("useClusterET", false);
    desc.add<std::string>("TrackEGammaMatchType", "PtDependentCut");
    desc.add<std::vector<double>>("TrackEGammaDeltaPhi",
                                  {
                                      0.07,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaR",
                                  {
                                      0.08,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaEta",
                                  {
                                      10000000000.0,
                                      0.0,
                                      0.0,
                                  });
    desc.add<bool>("RelativeIsolation", true);
    desc.add<double>("IsoCut", -0.1);
    desc.add<double>("PTMINTRA", 2.0);
    desc.add<double>("DRmin", 0.03);
    desc.add<double>("DRmax", 0.2);
    desc.add<double>("maxChi2IsoTracks", 10000000000.0);
    desc.add<int>("minNStubsIsoTracks", 0);
    desc.add<double>("DeltaZ", 0.6);
    descriptions.add("L1TkElectronsHGC", desc);
    // or use the following to generate the label from the module's C++ type
    //descriptions.addWithDefaultLabel(desc);
  }
  {
    // L1TkElectronsEllipticMatchHGC
    edm::ParameterSetDescription desc;
    desc.add<std::string>("label", "EG");
    desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag("l1EGammaEEProducer", "L1EGammaCollectionBXVWithCuts"));
    desc.add<double>("ETmin", -1.0);
    desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
    desc.add<double>("TrackChi2", 10000000000.0);
    desc.add<double>("TrackMinPt", 10.0);
    desc.add<bool>("useTwoStubsPT", false);
    desc.add<bool>("useClusterET", false);
    desc.add<std::string>("TrackEGammaMatchType", "EllipticalCut");
    desc.add<std::vector<double>>("TrackEGammaDeltaPhi",
                                  {
                                      0.07,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaR",
                                  {
                                      0.08,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaEta",
                                  {
                                      0.01,
                                      0.01,
                                      10000000000.0,
                                  });
    desc.add<bool>("RelativeIsolation", true);
    desc.add<double>("IsoCut", -0.1);
    desc.add<double>("PTMINTRA", 2.0);
    desc.add<double>("DRmin", 0.03);
    desc.add<double>("DRmax", 0.2);
    desc.add<double>("maxChi2IsoTracks", 100);
    desc.add<int>("minNStubsIsoTracks", 4);
    desc.add<double>("DeltaZ", 0.6);
    descriptions.add("L1TkElectronsEllipticMatchHGC", desc);
    // or use the following to generate the label from the module's C++ type
    //descriptions.addWithDefaultLabel(desc);
  }
  {
    // L1TkIsoElectronsHGC
    edm::ParameterSetDescription desc;
    desc.add<std::string>("label", "EG");
    desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag("l1EGammaEEProducer", "L1EGammaCollectionBXVWithCuts"));
    desc.add<double>("ETmin", -1.0);
    desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
    desc.add<double>("TrackChi2", 10000000000.0);
    desc.add<double>("TrackMinPt", 10.0);
    desc.add<bool>("useTwoStubsPT", false);
    desc.add<bool>("useClusterET", false);
    desc.add<std::string>("TrackEGammaMatchType", "PtDependentCut");
    desc.add<std::vector<double>>("TrackEGammaDeltaPhi",
                                  {
                                      0.07,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaR",
                                  {
                                      0.08,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaEta",
                                  {
                                      10000000000.0,
                                      0.0,
                                      0.0,
                                  });
    desc.add<bool>("RelativeIsolation", true);
    desc.add<double>("IsoCut", 0.1);
    desc.add<double>("PTMINTRA", 2.0);
    desc.add<double>("DRmin", 0.03);
    desc.add<double>("DRmax", 0.4);
    desc.add<double>("maxChi2IsoTracks", 100);
    desc.add<int>("minNStubsIsoTracks", 4);
    desc.add<double>("DeltaZ", 1.0);
    descriptions.add("L1TkIsoElectronsHGC", desc);
    // or use the following to generate the label from the module's C++ type
    //descriptions.addWithDefaultLabel(desc);
  }
  {
    // L1TkElectronsLooseHGC
    edm::ParameterSetDescription desc;
    desc.add<std::string>("label", "EG");
    desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag("l1EGammaEEProducer", "L1EGammaCollectionBXVWithCuts"));
    desc.add<double>("ETmin", -1.0);
    desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
    desc.add<double>("TrackChi2", 10000000000.0);
    desc.add<double>("TrackMinPt", 3.0);
    desc.add<bool>("useTwoStubsPT", false);
    desc.add<bool>("useClusterET", false);
    desc.add<std::string>("TrackEGammaMatchType", "PtDependentCut");
    desc.add<std::vector<double>>("TrackEGammaDeltaPhi",
                                  {
                                      0.07,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaR",
                                  {
                                      0.12,
                                      0.0,
                                      0.0,
                                  });
    desc.add<std::vector<double>>("TrackEGammaDeltaEta",
                                  {
                                      10000000000.0,
                                      0.0,
                                      0.0,
                                  });
    desc.add<bool>("RelativeIsolation", true);
    desc.add<double>("IsoCut", -0.1);
    desc.add<double>("PTMINTRA", 2.0);
    desc.add<double>("DRmin", 0.03);
    desc.add<double>("DRmax", 0.2);
    desc.add<double>("maxChi2IsoTracks", 10000000000.0);
    desc.add<int>("minNStubsIsoTracks", 0);
    desc.add<double>("DeltaZ", 0.6);
    descriptions.add("L1TkElectronsLooseHGC", desc);
    // or use the following to generate the label from the module's C++ type
    //descriptions.addWithDefaultLabel(desc);
  }
}

// method to calculate isolation
float L1TkElectronTrackProducer::isolation(const edm::Handle<L1TTTrackCollectionType>& trkHandle, int match_index) {
  edm::Ptr<L1TTTrackType> matchedTrkPtr(trkHandle, match_index);
  L1TTTrackCollectionType::const_iterator trackIter;

  float sumPt = 0.0;
  int itr = 0;

  float dRMin2 = dRMin_ * dRMin_;
  float dRMax2 = dRMax_ * dRMax_;

  for (trackIter = trkHandle->begin(); trackIter != trkHandle->end(); ++trackIter) {
    if (itr++ != match_index) {
      if (trackIter->chi2() > maxChi2IsoTracks_ || trackIter->momentum().perp() <= pTMinTra_ ||
          trackIter->getStubRefs().size() < minNStubsIsoTracks_) {
        continue;
      }

      float dZ = std::abs(trackIter->POCA().z() - matchedTrkPtr->POCA().z());

      float phi1 = trackIter->momentum().phi();
      float phi2 = matchedTrkPtr->momentum().phi();
      float dPhi = reco::deltaPhi(phi1, phi2);
      float dEta = (trackIter->momentum().eta() - matchedTrkPtr->momentum().eta());
      float dR2 = (dPhi * dPhi + dEta * dEta);

      if (dR2 > dRMin2 && dR2 < dRMax2 && dZ < deltaZ_ && trackIter->momentum().perp() > pTMinTra_) {
        sumPt += trackIter->momentum().perp();
      }
    }
  }

  return sumPt;
}
double L1TkElectronTrackProducer::getPtScaledCut(double pt, std::vector<double>& parameters) {
  return (parameters[0] + parameters[1] * exp(parameters[2] * pt));
}
bool L1TkElectronTrackProducer::selectMatchedTrack(
    double& d_r, double& d_phi, double& d_eta, double& tk_pt, float& eg_eta) {
  if (matchType_ == "PtDependentCut") {
    if (std::abs(d_phi) < getPtScaledCut(tk_pt, dPhiCutoff_) && d_r < getPtScaledCut(tk_pt, dRCutoff_))
      return true;
  } else {
    double deta_max = dEtaCutoff_[0];
    if (std::abs(eg_eta) < EB_MaxEta)
      deta_max = dEtaCutoff_[1];
    double dphi_max = dPhiCutoff_[0];
    if ((d_eta / deta_max) * (d_eta / deta_max) + (d_phi / dphi_max) * (d_phi / dphi_max) < 1)
      return true;
  }
  return false;
}
//define this as a plug-in
DEFINE_FWK_MODULE(L1TkElectronTrackProducer);
