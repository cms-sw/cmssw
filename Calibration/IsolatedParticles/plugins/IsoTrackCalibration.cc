// system include files
#include <memory>
#include <string>
#include <vector>

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TH1F.h"
#include "TLorentzVector.h"
#include "TInterpreter.h"

//Tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//Triggers
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "DataFormats/JetReco/interface/PFJet.h"

class IsoTrackCalibration : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit IsoTrackCalibration(const edm::ParameterSet &);
  ~IsoTrackCalibration() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void analyze(edm::Event const &, edm::EventSetup const &) override;
  void beginJob() override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void endRun(edm::Run const &, edm::EventSetup const &) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &);
  virtual void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &);

  edm::Service<TFileService> fs_;
  HLTConfigProvider hltConfig_;
  std::vector<std::string> trigNames_, HLTNames_;
  int verbosity_;
  spr::trackSelectionParameters selectionParameters_;
  std::string theTrackQuality_;
  double a_mipR_, a_coneR_, a_charIsoR_;
  bool isMC_, isQCD_, isAOD_;
  double constTrackPt_, slopeTrackPt_;
  double maxEcalEnr_;
  double maxNeighborTrackEnr_;
  int nRun_;
  edm::InputTag triggerEvent_, theTriggerResultsLabel_;
  edm::EDGetTokenT<trigger::TriggerEvent> tok_trigEvt_;
  edm::EDGetTokenT<edm::TriggerResults> tok_trigRes_;

  edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot> tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<GenEventInfoProduct> tok_ew_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;

  TTree *tree;
  int t_Run, t_Event, t_nVtx, t_nTrk, t_ieta;
  double t_EventWeight, t_p, t_pt, t_phi;
  double t_eHcal, t_eHcal10, t_eHcal30;
  double t_eMipDR, t_hmaxNearP;
  bool t_selectTk, t_qltyMissFlag, t_qltyPVFlag;
  std::vector<unsigned int> *t_DetIds, *t_DetIds1, *t_DetIds3;
  std::vector<double> *t_HitEnergies, *t_HitEnergies1, *t_HitEnergies3;

  TH1F *h_nTrk, *h_nVtx;
  TProfile *h_RecHit_iEta, *h_RecHit_num;
  TH1I *h_iEta, *h_tketa0[5], *h_tketa1[5], *h_tketa2[5];
  TH1I *h_tketa3[5], *h_tketa4[5], *h_tketa5[5];
  TH1F *h_Rechit_E, *h_jetp;
  TH1F *h_jetpt[4];
  TH1I *h_tketav1[5][6], *h_tketav2[5][6];
};

IsoTrackCalibration::IsoTrackCalibration(const edm::ParameterSet &iConfig) : nRun_(0) {
  usesResource("TFileService");
  //now do whatever initialization is needed
  const double isolationRadius(28.9);
  verbosity_ = iConfig.getUntrackedParameter<int>("Verbosity", 0);
  trigNames_ = iConfig.getUntrackedParameter<std::vector<std::string> >("Triggers");
  theTrackQuality_ = iConfig.getUntrackedParameter<std::string>("TrackQuality", "highPurity");
  reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality_);
  constTrackPt_ = iConfig.getUntrackedParameter<double>("ConstTrackPt", 10.0);
  slopeTrackPt_ = iConfig.getUntrackedParameter<double>("SlopeTrackPt", 0.16);
  selectionParameters_.minPt = constTrackPt_;
  selectionParameters_.minQuality = trackQuality_;
  selectionParameters_.maxDxyPV = iConfig.getUntrackedParameter<double>("MaxDxyPV", 0.02);
  selectionParameters_.maxDzPV = iConfig.getUntrackedParameter<double>("MaxDzPV", 0.02);
  selectionParameters_.maxChi2 = iConfig.getUntrackedParameter<double>("MaxChi2", 5.0);
  selectionParameters_.maxDpOverP = iConfig.getUntrackedParameter<double>("MaxDpOverP", 0.1);
  selectionParameters_.minOuterHit = iConfig.getUntrackedParameter<int>("MinOuterHit", 4);
  selectionParameters_.minLayerCrossed = iConfig.getUntrackedParameter<int>("MinLayerCrossed", 8);
  selectionParameters_.maxInMiss = iConfig.getUntrackedParameter<int>("MaxInMiss", 0);
  selectionParameters_.maxOutMiss = iConfig.getUntrackedParameter<int>("MaxOutMiss", 0);
  a_coneR_ = iConfig.getUntrackedParameter<double>("ConeRadius", 34.98);
  a_charIsoR_ = a_coneR_ + isolationRadius;
  a_mipR_ = iConfig.getUntrackedParameter<double>("ConeRadiusMIP", 14.0);
  maxEcalEnr_ = iConfig.getUntrackedParameter<double>("MaxEcalEnergyInCone", 2.5);
  maxNeighborTrackEnr_ = iConfig.getUntrackedParameter<double>("MaxNeighborTrackEnergy", 40.0);
  isMC_ = iConfig.getUntrackedParameter<bool>("IsMC", false);
  isQCD_ = iConfig.getUntrackedParameter<bool>("IsQCD", false);
  isAOD_ = iConfig.getUntrackedParameter<bool>("IsAOD", true);
  triggerEvent_ = edm::InputTag("hltTriggerSummaryAOD", "", "HLT");
  theTriggerResultsLabel_ = edm::InputTag("TriggerResults", "", "HLT");

  // define tokens for access
  tok_trigEvt_ = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes_ = consumes<edm::TriggerResults>(theTriggerResultsLabel_);
  tok_genTrack_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  tok_recVtx_ = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  tok_bs_ = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  tok_ew_ = consumes<GenEventInfoProduct>(edm::InputTag("generator"));

  if (isAOD_) {
    tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"));
    tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"));
    tok_hbhe_ = consumes<HBHERecHitCollection>(edm::InputTag("reducedHcalRecHits", "hbhereco"));
  } else {
    tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
    tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
    tok_hbhe_ = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  }

  if (verbosity_ >= 0) {
    edm::LogVerbatim("IsoTrack") << "Parameters read from config file \n"
                                 << "\t minPt " << selectionParameters_.minPt << "\t theTrackQuality "
                                 << theTrackQuality_ << "\t minQuality " << selectionParameters_.minQuality
                                 << "\t maxDxyPV " << selectionParameters_.maxDxyPV << "\t maxDzPV "
                                 << selectionParameters_.maxDzPV << "\t maxChi2 " << selectionParameters_.maxChi2
                                 << "\t maxDpOverP " << selectionParameters_.maxDpOverP << "\t minOuterHit "
                                 << selectionParameters_.minOuterHit << "\t minLayerCrossed "
                                 << selectionParameters_.minLayerCrossed << "\t maxInMiss "
                                 << selectionParameters_.maxInMiss << "\t maxOutMiss "
                                 << selectionParameters_.maxOutMiss << "\t a_coneR " << a_coneR_ << "\t a_charIsoR "
                                 << a_charIsoR_ << "\t a_mipR " << a_mipR_ << "\t isMC " << isMC_ << "\t isQCD "
                                 << isQCD_ << "\t isAOD " << isAOD_;
    edm::LogVerbatim("IsoTrack") << trigNames_.size() << " triggers to be studied:";
    for (unsigned int k = 0; k < trigNames_.size(); ++k)
      edm::LogVerbatim("IsoTrack") << "[" << k << "] " << trigNames_[k];
  }

  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_magField_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
}

IsoTrackCalibration::~IsoTrackCalibration() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void IsoTrackCalibration::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  t_Run = iEvent.id().run();
  t_Event = iEvent.id().event();
  if (verbosity_ % 10 > 0)
    edm::LogVerbatim("IsoTrack") << "Run " << t_Run << " Event " << t_Event << " Luminosity "
                                 << iEvent.luminosityBlock() << " Bunch " << iEvent.bunchCrossing()
                                 << " starts ==========";
  //Get magnetic field
  const MagneticField *bField = &iSetup.getData(tok_magField_);

  // get handles to calogeometry
  const CaloGeometry *geo = &iSetup.getData(tok_geom_);

  //Get track collection
  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);

  //event weight for FLAT sample and PU information
  t_EventWeight = 1.0;
  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(tok_ew_, genEventInfo);
  if (genEventInfo.isValid())
    t_EventWeight = genEventInfo->weight();

  //Define the best vertex and the beamspot
  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_, recVtxs);
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);
  math::XYZPoint leadPV(0, 0, 0);

  t_nVtx = recVtxs->size();
  h_nVtx->Fill(t_nVtx);

  if (!recVtxs->empty() && !((*recVtxs)[0].isFake())) {
    leadPV = math::XYZPoint((*recVtxs)[0].x(), (*recVtxs)[0].y(), (*recVtxs)[0].z());
  } else if (beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }
  if ((verbosity_ / 100) % 10 > 0) {
    edm::LogVerbatim("IsoTrack") << "Primary Vertex " << leadPV;
    if (beamSpotH.isValid())
      edm::LogVerbatim("IsoTrack") << " Beam Spot " << beamSpotH->position();
  }

  // RecHits
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);

  //Trigger
  bool triggerOK = false;
  if (isMC_ && !isQCD_) {
    triggerOK = true;  // ignore HLT for single pion MC
  } else {
    trigger::TriggerEvent triggerEvent;
    edm::Handle<trigger::TriggerEvent> triggerEventHandle;
    iEvent.getByToken(tok_trigEvt_, triggerEventHandle);
    if (!triggerEventHandle.isValid()) {
      edm::LogVerbatim("IsoTrack") << "Error! Can't get the product " << triggerEvent_.label();
    } else {
      /////////////////////////////TriggerResults
      edm::Handle<edm::TriggerResults> triggerResults;
      iEvent.getByToken(tok_trigRes_, triggerResults);
      if (triggerResults.isValid()) {
        const edm::TriggerNames &triggerNames = iEvent.triggerNames(*triggerResults);
        const std::vector<std::string> &triggerNames_ = triggerNames.triggerNames();
        for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
          int hlt = triggerResults->accept(iHLT);
          if (hlt > 0) {
            for (unsigned int i = 0; i < trigNames_.size(); ++i) {
              if (triggerNames_[iHLT].find(trigNames_[i]) != std::string::npos) {
                triggerOK = true;
                if (verbosity_ % 10 > 0)
                  edm::LogVerbatim("IsoTrack")
                      << "This is the trigger we are looking for " << triggerNames_[iHLT] << " Flag " << hlt;
              }
            }
          }
        }
      }
    }
  }

  if (triggerOK) {
    //Propagate tracks to calorimeter surface)
    std::vector<spr::propagatedTrackDirection> trkCaloDirections;
    spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_, trkCaloDirections, ((verbosity_ / 100) % 10 > 2));
    //Loop over tracks
    std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
    unsigned int nTracks(0);
    t_nTrk = trkCaloDirections.size();

    for (trkDetItr = trkCaloDirections.begin(), nTracks = 0; trkDetItr != trkCaloDirections.end();
         trkDetItr++, nTracks++) {
      const reco::Track *pTrack = &(*(trkDetItr->trkItr));
      if (verbosity_ % 10 > 0)
        edm::LogVerbatim("IsoTrack") << "This track : " << nTracks << " (pt/eta/phi/p) :" << pTrack->pt() << "/"
                                     << pTrack->eta() << "/" << pTrack->phi() << "/" << pTrack->p();

      t_ieta = 0;
      if (trkDetItr->okHCAL) {
        HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
        t_ieta = detId.ieta();
      }
      // ---------- eta-dependent restriction on Pt ----------------------------
      selectionParameters_.minPt = constTrackPt_ - std::abs((double)t_ieta) * slopeTrackPt_;
      // -----------------------------------------------------------------------

      //Selection of good track
      t_selectTk = spr::goodTrack(pTrack, leadPV, selectionParameters_, ((verbosity_ / 100) % 10 > 2));
      spr::trackSelectionParameters oneCutParameters = selectionParameters_;
      oneCutParameters.maxDxyPV = 10;
      oneCutParameters.maxDzPV = 100;
      oneCutParameters.maxInMiss = 2;
      oneCutParameters.maxOutMiss = 2;
      bool qltyFlag = spr::goodTrack(pTrack, leadPV, oneCutParameters, ((verbosity_ / 100) % 10 > 2));
      oneCutParameters = selectionParameters_;
      oneCutParameters.maxDxyPV = 10;
      oneCutParameters.maxDzPV = 100;
      t_qltyMissFlag = spr::goodTrack(pTrack, leadPV, oneCutParameters, ((verbosity_ / 100) % 10 > 2));
      oneCutParameters = selectionParameters_;
      oneCutParameters.maxInMiss = 2;
      oneCutParameters.maxOutMiss = 2;
      t_qltyPVFlag = spr::goodTrack(pTrack, leadPV, oneCutParameters, ((verbosity_ / 100) % 10 > 2));

      if (verbosity_ % 10 > 0)
        edm::LogVerbatim("IsoTrack") << "qltyFlag|okECAL|okHCAL : " << qltyFlag << "|" << trkDetItr->okECAL << "/"
                                     << trkDetItr->okHCAL;
      if (qltyFlag && trkDetItr->okECAL && trkDetItr->okHCAL) {
        int nRH_eMipDR(0), nNearTRKs(0);
        //------ ecal energy around track -------------------------------
        t_eMipDR = spr::eCone_ecal(geo,
                                   barrelRecHitsHandle,
                                   endcapRecHitsHandle,
                                   trkDetItr->pointHCAL,
                                   trkDetItr->pointECAL,
                                   a_mipR_,
                                   trkDetItr->directionECAL,
                                   nRH_eMipDR);
        //---- isolation criteria ----------------------------------------------
        t_hmaxNearP =
            spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR_, nNearTRKs, ((verbosity_ / 100) % 10 > 2));

        if (t_eMipDR < maxEcalEnr_ && t_hmaxNearP < maxNeighborTrackEnr_) {
          //------------ HCAL --------------------------------------------------
          //------ initialize arrays of DetID and hit energies -----------------
          t_DetIds->clear();
          t_DetIds1->clear();
          t_DetIds3->clear();
          t_HitEnergies->clear();
          t_HitEnergies1->clear();
          t_HitEnergies3->clear();
          int nRecHits(-999), nRecHits1(-999), nRecHits3(-999);
          std::vector<DetId> ids, ids1, ids3;
          //------ hcal energy in the main cone -------------------------------
          t_eHcal = spr::eCone_hcal(geo,
                                    hbhe,
                                    trkDetItr->pointHCAL,
                                    trkDetItr->pointECAL,
                                    a_coneR_,
                                    trkDetItr->directionHCAL,
                                    nRecHits,
                                    ids,
                                    *t_HitEnergies);
          t_DetIds->reserve(ids.size());
          for (unsigned int k = 0; k < ids.size(); ++k) {
            t_DetIds->push_back(ids[k].rawId());
          }
          //----- hcal energy in the extended cone 1 (a_coneR+10) --------------
          t_eHcal10 = spr::eCone_hcal(geo,
                                      hbhe,
                                      trkDetItr->pointHCAL,
                                      trkDetItr->pointECAL,
                                      a_coneR_ + 10,
                                      trkDetItr->directionHCAL,
                                      nRecHits1,
                                      ids1,
                                      *t_HitEnergies1);
          t_DetIds1->reserve(ids1.size());
          for (unsigned int k = 0; k < ids1.size(); ++k) {
            t_DetIds1->push_back(ids1[k].rawId());
          }
          //----- hcal energy in the extended cone 3 (a_coneR+30) --------------
          t_eHcal30 = spr::eCone_hcal(geo,
                                      hbhe,
                                      trkDetItr->pointHCAL,
                                      trkDetItr->pointECAL,
                                      a_coneR_ + 30,
                                      trkDetItr->directionHCAL,
                                      nRecHits3,
                                      ids3,
                                      *t_HitEnergies3);
          t_DetIds3->reserve(ids3.size());
          for (unsigned int k = 0; k < ids3.size(); ++k) {
            t_DetIds3->push_back(ids3[k].rawId());
          }

          t_p = pTrack->p();
          t_pt = pTrack->pt();
          t_phi = pTrack->phi();
          if (verbosity_ % 10 > 0) {
            edm::LogVerbatim("IsoTrack") << "This track : " << nTracks << " (pt/eta/phi/p) :" << pTrack->pt() << "/"
                                         << pTrack->eta() << "/" << pTrack->phi() << "/" << t_p;
            edm::LogVerbatim("IsoTrack") << "e_MIP " << t_eMipDR << " Chg Isolation " << t_hmaxNearP << " eHcal"
                                         << t_eHcal << " ieta " << t_ieta << " Quality " << t_qltyMissFlag << ":"
                                         << t_qltyPVFlag << ":" << t_selectTk;
            for (unsigned int lll = 0; lll < t_DetIds->size(); lll++) {
              edm::LogVerbatim("IsoTrack")
                  << "det id is = " << t_DetIds->at(lll) << "   hit enery is  = " << t_HitEnergies->at(lll);
            }
          }
          tree->Fill();
        }  // end of conditions on t_eMipDR and t_hmaxNearP
      }    // end of loose check of track quality
    }      // end of loop over tracks

    h_nTrk->Fill(nTracks);
  }  // end of triggerOK
}

void IsoTrackCalibration::beginJob() {
  h_nVtx = fs_->make<TH1F>("h_nVtx", "h_nVtx", 100, 0, 100);
  h_nTrk = fs_->make<TH1F>("h_nTrk", "h_nTrk", 100, 0, 2000);

  tree = fs_->make<TTree>("CalibTree", "CalibTree");

  tree->Branch("t_Run", &t_Run, "t_Run/I");
  tree->Branch("t_Event", &t_Event, "t_Event/I");
  tree->Branch("t_nVtx", &t_nVtx, "t_nVtx/I");
  tree->Branch("t_nTrk", &t_nTrk, "t_nTrk/I");
  tree->Branch("t_EventWeight", &t_EventWeight, "t_EventWeight/D");
  tree->Branch("t_p", &t_p, "t_p/D");
  tree->Branch("t_pt", &t_pt, "t_pt/D");
  tree->Branch("t_ieta", &t_ieta, "t_ieta/I");
  tree->Branch("t_phi", &t_phi, "t_phi/D");
  tree->Branch("t_eMipDR", &t_eMipDR, "t_eMipDR/D");
  tree->Branch("t_eHcal", &t_eHcal, "t_eHcal/D");
  tree->Branch("t_eHcal10", &t_eHcal10, "t_eHcal10/D");
  tree->Branch("t_eHcal30", &t_eHcal30, "t_eHcal30/D");
  tree->Branch("t_hmaxNearP", &t_hmaxNearP, "t_hmaxNearP/D");
  tree->Branch("t_selectTk", &t_selectTk, "t_selectTk/O");
  tree->Branch("t_qltyMissFlag", &t_qltyMissFlag, "t_qltyMissFlag/O");
  tree->Branch("t_qltyPVFlag", &t_qltyPVFlag, "t_qltyPVFlag/O)");

  t_DetIds = new std::vector<unsigned int>();
  t_DetIds1 = new std::vector<unsigned int>();
  t_DetIds3 = new std::vector<unsigned int>();
  t_HitEnergies = new std::vector<double>();
  t_HitEnergies1 = new std::vector<double>();
  t_HitEnergies3 = new std::vector<double>();

  tree->Branch("t_DetIds", "std::vector<unsigned int>", &t_DetIds);
  //tree->Branch("t_DetIds1",      "std::vector<unsigned int>", &t_DetIds1);
  //tree->Branch("t_DetIds3",      "std::vector<unsigned int>", &t_DetIds3);
  tree->Branch("t_HitEnergies", "std::vector<double>", &t_HitEnergies);
  //tree->Branch("t_HitEnergies1", "std::vector<double>",       &t_HitEnergies1);
  //tree->Branch("t_HitEnergies3", "std::vector<double>",       &t_HitEnergies3);
}

// ------------ method called when starting to processes a run  ------------
void IsoTrackCalibration::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  bool changed_(true);
  bool flag = hltConfig_.init(iRun, iSetup, "HLT", changed_);
  edm::LogInfo("HcalIsoTrack") << "Run[" << nRun_ << "] " << iRun.run() << " process HLT init flag " << flag
                               << " change flag " << changed_;

  // check if trigger names in (new) config
  if (changed_) {
    edm::LogInfo("HcalIsoTrack") << "New trigger menu found !!!";
    const unsigned int n(hltConfig_.size());
    for (unsigned itrig = 0; itrig < trigNames_.size(); itrig++) {
      unsigned int triggerindx = hltConfig_.triggerIndex(trigNames_[itrig]);
      if (triggerindx >= n) {
        edm::LogWarning("HcalIsoTrack") << trigNames_[itrig] << " " << triggerindx << " does not exist in "
                                        << "the current menu";
      } else {
        edm::LogInfo("HcalIsoTrack") << trigNames_[itrig] << " " << triggerindx << " exists";
      }
    }
  }
}

// ------------ method called when ending the processing of a run  ------------
void IsoTrackCalibration::endRun(edm::Run const &iRun, edm::EventSetup const &) {
  ++nRun_;
  edm::LogWarning("HcalIsoTrack") << "endRun[" << nRun_ << "] " << iRun.run();
}

// ------------ method called when starting to processes a luminosity block  ------------
void IsoTrackCalibration::beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}
// ------------ method called when ending the processing of a luminosity block  ------------
void IsoTrackCalibration::endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void IsoTrackCalibration::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsoTrackCalibration);
