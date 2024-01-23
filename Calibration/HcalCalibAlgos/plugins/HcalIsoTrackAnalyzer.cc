// system include files
#include <string>
#include <vector>

// Root objects
#include "TTree.h"

#include "DataFormats/HcalCalibObjects/interface/HcalIsoTrkCalibVariables.h"
#include "DataFormats/HcalCalibObjects/interface/HcalIsoTrkEventVariables.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//#define EDM_ML_DEBUG

class HcalIsoTrackAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HcalIsoTrackAnalyzer(edm::ParameterSet const&);
  ~HcalIsoTrackAnalyzer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  const double pTrackLow_, pTrackHigh_;
  const int useRaw_, dataType_;
  const edm::InputTag labelIsoTkVar_, labelIsoTkEvt_;
  const std::vector<int> debEvents_;
  edm::EDGetTokenT<HcalIsoTrkCalibVariablesCollection> tokIsoTrkVar_;
  edm::EDGetTokenT<HcalIsoTrkEventVariablesCollection> tokIsoTrkEvt_;
  unsigned int nRun_, nRange_, nLow_, nHigh_;

  TTree *tree, *tree2;
  int t_Run, t_Event, t_DataType, t_ieta, t_iphi;
  int t_goodPV, t_nVtx, t_nTrk;
  double t_EventWeight, t_p, t_pt, t_phi;
  double t_l1pt, t_l1eta, t_l1phi;
  double t_l3pt, t_l3eta, t_l3phi;
  double t_mindR1, t_mindR2;
  double t_eMipDR, t_eMipDR2, t_eMipDR3, t_eMipDR4;
  double t_eMipDR5, t_hmaxNearP, t_gentrackP;
  double t_emaxNearP, t_eAnnular, t_hAnnular;
  double t_eHcal, t_eHcal10, t_eHcal30, t_rhoh;
  bool t_selectTk, t_qltyFlag, t_qltyMissFlag, t_qltyPVFlag;
  std::vector<unsigned int> t_DetIds, t_DetIds1, t_DetIds3;
  std::vector<double> t_HitEnergies, t_HitEnergies1, t_HitEnergies3;
  std::vector<bool> t_trgbits;

  unsigned int t_RunNo, t_EventNo;
  bool t_TrigPass, t_TrigPassSel, t_L1Bit;
  int t_Tracks, t_TracksProp, t_TracksSaved;
  int t_TracksLoose, t_TracksTight, t_allvertex;
  std::vector<int> t_ietaAll, t_ietaGood, t_trackType;
  std::vector<bool> t_hltbits;
};

HcalIsoTrackAnalyzer::HcalIsoTrackAnalyzer(const edm::ParameterSet& iConfig)
    : pTrackLow_(iConfig.getParameter<double>("momentumLow")),
      pTrackHigh_(iConfig.getParameter<double>("momentumHigh")),
      useRaw_(iConfig.getUntrackedParameter<int>("useRaw", 0)),
      dataType_(iConfig.getUntrackedParameter<int>("dataType", 0)),
      labelIsoTkVar_(iConfig.getParameter<edm::InputTag>("isoTrackVarLabel")),
      labelIsoTkEvt_(iConfig.getParameter<edm::InputTag>("isoTrackEvtLabel")),
      debEvents_(iConfig.getParameter<std::vector<int>>("debugEvents")),
      tokIsoTrkVar_(consumes<HcalIsoTrkCalibVariablesCollection>(labelIsoTkVar_)),
      tokIsoTrkEvt_(consumes<HcalIsoTrkEventVariablesCollection>(labelIsoTkEvt_)),
      nRun_(0),
      nRange_(0),
      nLow_(0),
      nHigh_(0) {
  usesResource(TFileService::kSharedResource);

  //now do whatever initialization is needed
  edm::LogVerbatim("HcalIsoTrack") << "Labels used " << labelIsoTkVar_ << " " << labelIsoTkEvt_;

  edm::LogVerbatim("HcalIsoTrack") << "Parameters read from config file \n\t momentumLow_ " << pTrackLow_
                                   << "\t momentumHigh_ " << pTrackHigh_ << "\t useRaw_ " << useRaw_
                                   << "\t dataType_      " << dataType_ << " and " << debEvents_.size()
                                   << " events to be debugged";
}

void HcalIsoTrackAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  t_Run = iEvent.id().run();
  t_Event = iEvent.id().event();
  t_DataType = dataType_;
#ifdef EDM_ML_DEBUG
  bool debug = (debEvents_.empty())
                   ? true
                   : (std::find(debEvents_.begin(), debEvents_.end(), iEvent.id().event()) != debEvents_.end());
  if (debug)
    edm::LogVerbatim("HcalIsoTrack") << "Run " << t_Run << " Event " << t_Event << " type " << t_DataType
                                     << " Luminosity " << iEvent.luminosityBlock() << " Bunch "
                                     << iEvent.bunchCrossing();
#endif

  // Fill from IsoTrkCalibVariables collection
  auto const& isotrkCalibColl = iEvent.getHandle(tokIsoTrkVar_);
  if (isotrkCalibColl.isValid()) {
    auto isotrkCalib = *isotrkCalibColl.product();
#ifdef EDM_ML_DEBUG
    if (debug)
      edm::LogVerbatim("HcalIsoTrack") << "Finds HcalIsoTrkCalibVariablesCollection with " << isotrkCalib.size()
                                       << " entries";
    int k(0);
#endif
    for (const auto& itr : isotrkCalib) {
      t_ieta = itr.ieta_;
      t_iphi = itr.iphi_;
      t_goodPV = itr.goodPV_;
      t_nVtx = itr.nVtx_;
      t_nTrk = itr.nTrk_;
      t_EventWeight = itr.eventWeight_;
      t_p = itr.p_;
      t_pt = itr.pt_;
      t_phi = itr.phi_;
#ifdef EDM_ML_DEBUG
      ++k;
      if (debug)
        edm::LogVerbatim("HcalIsoTrack") << "Track " << k << " p:pt:phi " << t_p << ":" << t_pt << ":" << t_phi
                                         << " nvtx:ntrk:goodPV:wt " << t_nVtx << ":" << t_nTrk << ":" << t_goodPV << ":"
                                         << t_EventWeight << " ieta:iphi " << t_ieta << ":" << t_iphi;
#endif
      t_l1pt = itr.l1pt_;
      t_l1eta = itr.l1eta_;
      t_l1phi = itr.l1phi_;
      t_l3pt = itr.l3pt_;
      t_l3eta = itr.l3eta_;
      t_l3phi = itr.l3phi_;
      t_mindR1 = itr.mindR1_;
      t_mindR2 = itr.mindR2_;
#ifdef EDM_ML_DEBUG
      if (debug)
        edm::LogVerbatim("HcalIsoTrack") << "L1 pt:eta:phi " << t_l1pt << ":" << t_l1eta << ":" << t_l1phi
                                         << " L3 pt:eta:phi " << t_l3pt << ":" << t_l3eta << ":" << t_l3phi << " R1:R2 "
                                         << t_mindR1 << ":" << t_mindR2;
#endif
      t_eMipDR = itr.eMipDR_[0];
      t_eMipDR2 = itr.eMipDR_[1];
      t_eMipDR3 = itr.eMipDR_[2];
      t_eMipDR4 = itr.eMipDR_[3];
      t_eMipDR5 = itr.eMipDR_[4];
#ifdef EDM_ML_DEBUG
      if (debug)
        edm::LogVerbatim("HcalIsoTrack") << "eMIPDR 1:2:3:4:5 " << t_eMipDR << ":" << t_eMipDR2 << ":" << t_eMipDR3
                                         << ":" << t_eMipDR4 << ":" << t_eMipDR5;
#endif
      t_hmaxNearP = itr.hmaxNearP_;
      t_emaxNearP = itr.emaxNearP_;
      t_eAnnular = itr.eAnnular_;
      t_hAnnular = itr.hAnnular_;
#ifdef EDM_ML_DEBUG
      if (debug)
        edm::LogVerbatim("HcalIsoTrack") << "emaxNearP:hmaxNearP " << t_emaxNearP << ":" << t_hmaxNearP
                                         << " eAnnlar:hAnnular" << t_eAnnular << ":" << t_hAnnular;
#endif
      t_gentrackP = itr.gentrackP_;
      t_rhoh = itr.rhoh_;
      t_selectTk = itr.selectTk_;
      t_qltyFlag = itr.qltyFlag_;
      t_qltyMissFlag = itr.qltyMissFlag_;
      t_qltyPVFlag = itr.qltyPVFlag_;
#ifdef EDM_ML_DEBUG
      if (debug)
        edm::LogVerbatim("HcalIsoTrack") << "gentrackP " << t_gentrackP << " rhoh " << t_rhoh
                                         << " qltyFlag:qltyMissFlag:qltyPVFlag:selectTk " << t_qltyFlag << ":"
                                         << t_qltyMissFlag << ":" << t_qltyPVFlag << ":" << t_selectTk;
#endif
      t_trgbits = itr.trgbits_;
      t_DetIds = itr.detIds_;
      t_DetIds1 = itr.detIds1_;
      t_DetIds3 = itr.detIds3_;
      if (useRaw_ == 1) {
        t_eHcal = itr.eHcalAux_;
        t_eHcal10 = itr.eHcal10Aux_;
        t_eHcal30 = itr.eHcal30Aux_;
        t_HitEnergies = itr.hitEnergiesAux_;
        t_HitEnergies1 = itr.hitEnergies1Aux_;
        t_HitEnergies3 = itr.hitEnergies3Aux_;
      } else if (useRaw_ == 2) {
        t_eHcal = itr.eHcalRaw_;
        t_eHcal10 = itr.eHcal10Raw_;
        t_eHcal30 = itr.eHcal30Raw_;
        t_HitEnergies = itr.hitEnergiesRaw_;
        t_HitEnergies1 = itr.hitEnergies1Raw_;
        t_HitEnergies3 = itr.hitEnergies3Raw_;
      } else {
        t_eHcal = itr.eHcal_;
        t_eHcal10 = itr.eHcal10_;
        t_eHcal30 = itr.eHcal30_;
        t_HitEnergies = itr.hitEnergies_;
        t_HitEnergies1 = itr.hitEnergies1_;
        t_HitEnergies3 = itr.hitEnergies3_;
      }
#ifdef EDM_ML_DEBUG
      if (debug)
        edm::LogVerbatim("HcalIsoTrack") << "eHcal:eHcal10:eHCal30 " << t_eHcal << ":" << t_eHcal10 << t_eHcal30;
#endif
      tree->Fill();
      edm::LogVerbatim("HcalIsoTrackX") << "Run " << t_Run << " Event " << t_Event << " p " << t_p;

      if (t_p < pTrackLow_) {
        ++nLow_;
      } else if (t_p < pTrackHigh_) {
        ++nHigh_;
      } else {
        ++nRange_;
      }
    }
  } else {
    edm::LogVerbatim("HcalIsoTrack") << "Cannot find HcalIsoTrkCalibVariablesCollection";
  }

  // Fill from IsoTrkEventVariables collection
  auto const& isotrkEventColl = iEvent.getHandle(tokIsoTrkEvt_);
  if (isotrkEventColl.isValid()) {
    auto isotrkEvent = isotrkEventColl.product();
#ifdef EDM_ML_DEBUG
    if (debug)
      edm::LogVerbatim("HcalIsoTrack") << "Finds HcalIsoTrkEventVariablesCollection with " << isotrkEvent->size()
                                       << " entries";
#endif
    auto itr = isotrkEvent->begin();
    if (itr != isotrkEvent->end()) {
      t_RunNo = iEvent.id().run();
      t_EventNo = iEvent.id().event();
      t_TrigPass = itr->trigPass_;
      t_TrigPassSel = itr->trigPassSel_;
      t_L1Bit = itr->l1Bit_;
      t_Tracks = itr->tracks_;
      t_TracksProp = itr->tracksProp_;
      t_TracksSaved = itr->tracksSaved_;
      t_TracksLoose = itr->tracksLoose_;
      t_TracksTight = itr->tracksTight_;
      t_allvertex = itr->allvertex_;
      t_ietaAll = itr->ietaAll_;
      t_ietaGood = itr->ietaGood_;
      t_trackType = itr->trackType_;
      t_hltbits = itr->hltbits_;
      tree2->Fill();
    }
  } else {
    edm::LogVerbatim("HcalIsoTrack") << "Cannot find HcalIsoTrkEventVariablesCollections";
  }
}

void HcalIsoTrackAnalyzer::beginJob() {
  edm::Service<TFileService> fs;
  tree = fs->make<TTree>("CalibTree", "CalibTree");

  tree->Branch("t_Run", &t_Run, "t_Run/I");
  tree->Branch("t_Event", &t_Event, "t_Event/I");
  tree->Branch("t_DataType", &t_DataType, "t_DataType/I");
  tree->Branch("t_ieta", &t_ieta, "t_ieta/I");
  tree->Branch("t_iphi", &t_iphi, "t_iphi/I");
  tree->Branch("t_EventWeight", &t_EventWeight, "t_EventWeight/D");
  tree->Branch("t_nVtx", &t_nVtx, "t_nVtx/I");
  tree->Branch("t_nTrk", &t_nTrk, "t_nTrk/I");
  tree->Branch("t_goodPV", &t_goodPV, "t_goodPV/I");
  tree->Branch("t_l1pt", &t_l1pt, "t_l1pt/D");
  tree->Branch("t_l1eta", &t_l1eta, "t_l1eta/D");
  tree->Branch("t_l1phi", &t_l1phi, "t_l1phi/D");
  tree->Branch("t_l3pt", &t_l3pt, "t_l3pt/D");
  tree->Branch("t_l3eta", &t_l3eta, "t_l3eta/D");
  tree->Branch("t_l3phi", &t_l3phi, "t_l3phi/D");
  tree->Branch("t_p", &t_p, "t_p/D");
  tree->Branch("t_pt", &t_pt, "t_pt/D");
  tree->Branch("t_phi", &t_phi, "t_phi/D");
  tree->Branch("t_mindR1", &t_mindR1, "t_mindR1/D");
  tree->Branch("t_mindR2", &t_mindR2, "t_mindR2/D");
  tree->Branch("t_eMipDR", &t_eMipDR, "t_eMipDR/D");
  tree->Branch("t_eMipDR2", &t_eMipDR2, "t_eMipDR2/D");
  tree->Branch("t_eMipDR3", &t_eMipDR3, "t_eMipDR3/D");
  tree->Branch("t_eMipDR4", &t_eMipDR4, "t_eMipDR4/D");
  tree->Branch("t_eMipDR5", &t_eMipDR5, "t_eMipDR5/D");
  tree->Branch("t_eHcal", &t_eHcal, "t_eHcal/D");
  tree->Branch("t_eHcal10", &t_eHcal10, "t_eHcal10/D");
  tree->Branch("t_eHcal30", &t_eHcal30, "t_eHcal30/D");
  tree->Branch("t_hmaxNearP", &t_hmaxNearP, "t_hmaxNearP/D");
  tree->Branch("t_emaxNearP", &t_emaxNearP, "t_emaxNearP/D");
  tree->Branch("t_eAnnular", &t_eAnnular, "t_eAnnular/D");
  tree->Branch("t_hAnnular", &t_hAnnular, "t_hAnnular/D");
  tree->Branch("t_rhoh", &t_rhoh, "t_rhoh/D");
  tree->Branch("t_selectTk", &t_selectTk, "t_selectTk/O");
  tree->Branch("t_qltyFlag", &t_qltyFlag, "t_qltyFlag/O");
  tree->Branch("t_qltyMissFlag", &t_qltyMissFlag, "t_qltyMissFlag/O");
  tree->Branch("t_qltyPVFlag", &t_qltyPVFlag, "t_qltyPVFlag/O");
  tree->Branch("t_gentrackP", &t_gentrackP, "t_gentrackP/D");

  tree->Branch("t_DetIds", &t_DetIds);
  tree->Branch("t_HitEnergies", &t_HitEnergies);
  tree->Branch("t_trgbits", &t_trgbits);
  tree->Branch("t_DetIds1", &t_DetIds1);
  tree->Branch("t_DetIds3", &t_DetIds3);
  tree->Branch("t_HitEnergies1", &t_HitEnergies1);
  tree->Branch("t_HitEnergies3", &t_HitEnergies3);

  tree2 = fs->make<TTree>("EventInfo", "Event Information");

  tree2->Branch("t_RunNo", &t_RunNo, "t_RunNo/i");
  tree2->Branch("t_EventNo", &t_EventNo, "t_EventNo/i");
  tree2->Branch("t_Tracks", &t_Tracks, "t_Tracks/I");
  tree2->Branch("t_TracksProp", &t_TracksProp, "t_TracksProp/I");
  tree2->Branch("t_TracksSaved", &t_TracksSaved, "t_TracksSaved/I");
  tree2->Branch("t_TracksLoose", &t_TracksLoose, "t_TracksLoose/I");
  tree2->Branch("t_TracksTight", &t_TracksTight, "t_TracksTight/I");
  tree2->Branch("t_TrigPass", &t_TrigPass, "t_TrigPass/O");
  tree2->Branch("t_TrigPassSel", &t_TrigPassSel, "t_TrigPassSel/O");
  tree2->Branch("t_L1Bit", &t_L1Bit, "t_L1Bit/O");
  tree2->Branch("t_allvertex", &t_allvertex, "t_allvertex/I");
  tree2->Branch("t_ietaAll", &t_ietaAll);
  tree2->Branch("t_ietaGood", &t_ietaGood);
  tree2->Branch("t_trackType", &t_trackType);
  tree2->Branch("t_hltbits", &t_hltbits);
}

// ------------ method called when starting to processes a run  ------------

// ------------ method called when ending the processing of a run  ------------
void HcalIsoTrackAnalyzer::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  nRun_++;
  edm::LogVerbatim("HcalIsoTrack") << "endRun[" << nRun_ << "] " << iRun.run() << " with " << nLow_
                                   << " events with p < " << pTrackLow_ << ", " << nHigh_ << " events with p > "
                                   << pTrackHigh_ << ", and " << nRange_ << " events in the right momentum range";
}

void HcalIsoTrackAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("momentumLow", 40.0);
  desc.add<double>("momentumHigh", 60.0);
  desc.addUntracked<int>("useRaw", 0);
  desc.addUntracked<int>("dataType", 0);
  desc.add<edm::InputTag>("isoTrackVarLabel", edm::InputTag("alcaHcalIsotrkProducer", "HcalIsoTrack"));
  desc.add<edm::InputTag>("isoTrackEvtLabel", edm::InputTag("alcaHcalIsotrkProducer", "HcalIsoTrackEvent"));
  std::vector<int> events;
  desc.add<std::vector<int>>("debugEvents", events);
  descriptions.add("hcalIsoTrackAnalyzer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalIsoTrackAnalyzer);
