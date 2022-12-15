/** \class METAnalyzer
 *
 *  DQM MET analysis monitoring
 *
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
 *
 *          Jan. '14: modified by
 *
 *          M. Artur Weber
 *          R. Schoefbeck
 *          V. Sordini
 */

#include "DQMOffline/JetMET/interface/METAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include <cmath>
#include "TH2F.h"
#include "TH2.h"

#include <string>

using namespace edm;
using namespace reco;
using namespace math;

// ***********************************************************
METAnalyzer::METAnalyzer(const edm::ParameterSet& pSet) {
  parameters = pSet;

  m_l1algoname_ = pSet.getParameter<std::string>("l1algoname");
  m_bitAlgTechTrig_ = -1;

  miniaodfilterdec = -1;

  LSBegin_ = pSet.getParameter<int>("LSBegin");
  LSEnd_ = pSet.getParameter<int>("LSEnd");
  // Smallest track pt
  ptMinCand_ = pSet.getParameter<double>("ptMinCand");

  MetType_ = parameters.getUntrackedParameter<std::string>("METType");

  triggerResultsLabel_ = parameters.getParameter<edm::InputTag>("TriggerResultsLabel");
  triggerResultsToken_ = consumes<edm::TriggerResults>(edm::InputTag(triggerResultsLabel_));

  isCaloMet_ = (std::string("calo") == MetType_);
  //isTCMet_ = (std::string("tc") ==MetType_);
  isPFMet_ = (std::string("pf") == MetType_);
  isMiniAODMet_ = (std::string("miniaod") == MetType_);
  if (!isMiniAODMet_) {
    jetCorrectorToken_ = consumes<reco::JetCorrector>(pSet.getParameter<edm::InputTag>("JetCorrections"));
  }

  // MET information
  metCollectionLabel_ = parameters.getParameter<edm::InputTag>("METCollectionLabel");

  if (/*isTCMet_ || */ isCaloMet_) {
    inputJetIDValueMap = pSet.getParameter<edm::InputTag>("InputJetIDValueMap");
    jetID_ValueMapToken_ = consumes<edm::ValueMap<reco::JetID> >(inputJetIDValueMap);
    jetIDFunctorLoose = JetIDSelectionFunctor(JetIDSelectionFunctor::PURE09, JetIDSelectionFunctor::LOOSE);
  }

  if (isPFMet_) {
    pflowToken_ = consumes<std::vector<reco::PFCandidate> >(pSet.getParameter<edm::InputTag>("srcPFlow"));
    pfjetIDFunctorLoose = PFJetIDSelectionFunctor(PFJetIDSelectionFunctor::WINTER16, PFJetIDSelectionFunctor::LOOSE);
  }
  if (isMiniAODMet_) {
    pflowPackedToken_ = consumes<std::vector<pat::PackedCandidate> >(pSet.getParameter<edm::InputTag>("srcPFlow"));
    pfjetIDFunctorLoose = PFJetIDSelectionFunctor(PFJetIDSelectionFunctor::WINTER16, PFJetIDSelectionFunctor::LOOSE);
  }
  MuonsToken_ = consumes<reco::MuonCollection>(pSet.getParameter<edm::InputTag>("muonsrc"));

  ptThreshold_ = parameters.getParameter<double>("ptThreshold");

  if (isPFMet_) {
    pfMetToken_ = consumes<reco::PFMETCollection>(edm::InputTag(metCollectionLabel_));
  }
  if (isCaloMet_) {
    caloMetToken_ = consumes<reco::CaloMETCollection>(edm::InputTag(metCollectionLabel_));
  }
  if (isMiniAODMet_) {
    patMetToken_ = consumes<pat::METCollection>(edm::InputTag(metCollectionLabel_));
  }
  //if(isTCMet_){
  // tcMetToken_= consumes<reco::METCollection>(edm::InputTag(metCollectionLabel_));
  //}

  fill_met_high_level_histo = parameters.getParameter<bool>("fillMetHighLevel");
  fillCandidateMap_histos = parameters.getParameter<bool>("fillCandidateMaps");

  hTriggerLabelsIsSet_ = false;
  //jet cleanup parameters
  cleaningParameters_ = pSet.getParameter<ParameterSet>("CleaningParameters");

  diagnosticsParameters_ = pSet.getParameter<std::vector<edm::ParameterSet> >("METDiagonisticsParameters");

  edm::ConsumesCollector iC = consumesCollector();
  //DCS
  DCSFilter_ = new JetMETDQMDCSFilter(parameters.getParameter<ParameterSet>("DCSFilter"), iC);

  //Vertex requirements
  bypassAllPVChecks_ = cleaningParameters_.getParameter<bool>("bypassAllPVChecks");
  bypassAllDCSChecks_ = cleaningParameters_.getParameter<bool>("bypassAllDCSChecks");
  runcosmics_ = parameters.getUntrackedParameter<bool>("runcosmics");
  onlyCleaned_ = parameters.getUntrackedParameter<bool>("onlyCleaned");
  vertexTag_ = cleaningParameters_.getParameter<edm::InputTag>("vertexCollection");
  vertexToken_ = consumes<std::vector<reco::Vertex> >(edm::InputTag(vertexTag_));

  //Trigger parameters
  gtTag_ = cleaningParameters_.getParameter<edm::InputTag>("gtLabel");
  gtToken_ = consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag(gtTag_));

  // Other data collections
  jetCollectionLabel_ = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  if (isCaloMet_)
    caloJetsToken_ = consumes<reco::CaloJetCollection>(jetCollectionLabel_);
  //if (isTCMet_)   jptJetsToken_ = consumes<reco::JPTJetCollection>(jetCollectionLabel_);
  if (isPFMet_)
    pfJetsToken_ = consumes<reco::PFJetCollection>(jetCollectionLabel_);
  if (isMiniAODMet_)
    patJetsToken_ = consumes<pat::JetCollection>(jetCollectionLabel_);

  HBHENoiseStringMiniAOD = parameters.getParameter<std::string>("HBHENoiseLabelMiniAOD");
  HBHEIsoNoiseStringMiniAOD = parameters.getParameter<std::string>("HBHEIsoNoiseLabelMiniAOD");

  hbheNoiseFilterResultTag_ = parameters.getParameter<edm::InputTag>("HBHENoiseFilterResultLabel");
  hbheNoiseFilterResultToken_ = consumes<bool>(hbheNoiseFilterResultTag_);
  hbheNoiseIsoFilterResultTag_ = parameters.getParameter<edm::InputTag>("HBHENoiseIsoFilterResultLabel");
  hbheIsoNoiseFilterResultToken_ = consumes<bool>(hbheNoiseIsoFilterResultTag_);
  CSCHaloResultTag_ = parameters.getParameter<edm::InputTag>("CSCHaloResultLabel");
  CSCHaloResultToken_ = consumes<bool>(CSCHaloResultTag_);
  CSCHalo2015ResultTag_ = parameters.getParameter<edm::InputTag>("CSCHalo2015ResultLabel");
  CSCHalo2015ResultToken_ = consumes<bool>(CSCHalo2015ResultTag_);
  EcalDeadCellTriggerTag_ = parameters.getParameter<edm::InputTag>("EcalDeadCellTriggerPrimitiveFilterLabel");
  EcalDeadCellTriggerToken_ = consumes<bool>(EcalDeadCellTriggerTag_);
  EcalDeadCellBoundaryTag_ = parameters.getParameter<edm::InputTag>("EcalDeadCellBoundaryEnergyFilterLabel");
  EcalDeadCellBoundaryToken_ = consumes<bool>(EcalDeadCellBoundaryTag_);
  eeBadScFilterTag_ = parameters.getParameter<edm::InputTag>("eeBadScFilterLabel");
  eeBadScFilterToken_ = consumes<bool>(eeBadScFilterTag_);
  HcalStripHaloTag_ = parameters.getParameter<edm::InputTag>("HcalStripHaloFilterLabel");
  HcalStripHaloToken_ = consumes<bool>(HcalStripHaloTag_);

  if (isMiniAODMet_) {
    METFilterMiniAODLabel_ = parameters.getParameter<edm::InputTag>("FilterResultsLabelMiniAOD");
    METFilterMiniAODToken_ = consumes<edm::TriggerResults>(METFilterMiniAODLabel_);

    METFilterMiniAODLabel2_ = parameters.getParameter<edm::InputTag>("FilterResultsLabelMiniAOD2");
    METFilterMiniAODToken2_ = consumes<edm::TriggerResults>(METFilterMiniAODLabel2_);
  }

  //
  nbinsPV_ = parameters.getParameter<int>("pVBin");
  nPVMin_ = parameters.getParameter<double>("pVMin");
  nPVMax_ = parameters.getParameter<double>("pVMax");

  triggerSelectedSubFolders_ = parameters.getParameter<edm::VParameterSet>("triggerSelectedSubFolders");
  for (edm::VParameterSet::const_iterator it = triggerSelectedSubFolders_.begin();
       it != triggerSelectedSubFolders_.end();
       it++) {
    triggerFolderEventFlag_.push_back(new GenericTriggerEventFlag(*it, consumesCollector(), *this));
    triggerFolderExpr_.push_back(it->getParameter<std::vector<std::string> >("hltPaths"));
    triggerFolderLabels_.push_back(it->getParameter<std::string>("label"));
  }

  cleaningParameters_ = parameters.getParameter<ParameterSet>("CleaningParameters");

  verbose_ = parameters.getParameter<int>("verbose");

  FolderName_ = parameters.getUntrackedParameter<std::string>("FolderName");

  l1gtTrigMenuToken_ = esConsumes<edm::Transition::BeginRun>();
}

// ***********************************************************
METAnalyzer::~METAnalyzer() {
  for (std::vector<GenericTriggerEventFlag*>::const_iterator it = triggerFolderEventFlag_.begin();
       it != triggerFolderEventFlag_.end();
       it++) {
    delete *it;
  }
  delete DCSFilter_;
}

void METAnalyzer::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const&) {
  std::string DirName = FolderName_ + metCollectionLabel_.label();
  ibooker.setCurrentFolder(DirName);
  // since this module does things in dqmEndRun, we need to make sure to have
  // per-run histograms.
  ibooker.setScope(MonitorElementData::Scope::RUN);

  if (!folderNames_.empty()) {
    folderNames_.clear();
  }
  if (runcosmics_) {
    folderNames_.push_back("Uncleaned");
  } else {
    if (!onlyCleaned_) {
      folderNames_.push_back("Uncleaned");
    }
    folderNames_.push_back("Cleaned");
    folderNames_.push_back("DiJet");
    if (!isMiniAODMet_) {
      folderNames_.push_back("ZJets");
    }
  }
  for (std::vector<std::string>::const_iterator ic = folderNames_.begin(); ic != folderNames_.end(); ic++) {
    bookMESet(DirName + "/" + *ic, ibooker, map_dijet_MEs);
  }
}

// ***********************************************************
void METAnalyzer::bookMESet(std::string DirName,
                            DQMStore::IBooker& ibooker,
                            std::map<std::string, MonitorElement*>& map_of_MEs) {
  bool bLumiSecPlot = fill_met_high_level_histo;
  //bool inTriggerPathPlots=false;
  bool fillPFCandidatePlots = false;
  bool fillZPlots = false;

  if (DirName.find("Cleaned") != std::string::npos) {
    fillPFCandidatePlots = true;
    bookMonitorElement(DirName, ibooker, map_of_MEs, bLumiSecPlot, fillPFCandidatePlots, fillZPlots);
    //for (unsigned int i = 0; i<triggerFolderEventFlag_.size(); i++) {
    //fillPFCandidatePlots=false;
    //if (triggerFolderEventFlag_[i]->on()) {
    //bookMonitorElement(DirName+"/"+triggerFolderLabels_[i],ibooker,map_of_MEs,bLumiSecPlot,fillPFCandidatePlots,fillZPlots);
    //}
    //}
  } else if (DirName.find("ZJets") != std::string::npos) {
    fillPFCandidatePlots = false;
    fillZPlots = true;
    bookMonitorElement(DirName, ibooker, map_of_MEs, bLumiSecPlot, fillPFCandidatePlots, fillZPlots);
  } else {
    bookMonitorElement(DirName, ibooker, map_of_MEs, bLumiSecPlot, fillPFCandidatePlots, fillZPlots);
  }
}

// ***********************************************************
void METAnalyzer::bookMonitorElement(std::string DirName,
                                     DQMStore::IBooker& ibooker,
                                     std::map<std::string, MonitorElement*>& map_of_MEs,
                                     bool bLumiSecPlot = false,
                                     bool fillPFCandPlots = false,
                                     bool fillZPlots = false) {
  if (verbose_)
    std::cout << "bookMonitorElement " << DirName << std::endl;

  ibooker.setCurrentFolder(DirName);
  if (fillZPlots) {
    if (isCaloMet_) {
      meZJets_u_par = ibooker.book1D("u_parallel_Z_inc", "u_parallel_Z_inc", 50, -1000., 75);
    } else {
      meZJets_u_par = ibooker.book1D("u_parallel_Z_inc", "u_parallel_Z_inc", 50, -800., 75);
    }
    meZJets_u_par_ZPt_0_15 = ibooker.book1D("u_parallel_ZPt_0_15", "u_parallel_ZPt_0_15", 50, -100, 75);
    meZJets_u_par_ZPt_15_30 = ibooker.book1D("u_parallel_ZPt_15_30", "u_parallel_ZPt_15_30", 50, -100, 50);
    meZJets_u_par_ZPt_30_55 = ibooker.book1D("u_parallel_ZPt_30_55", "u_parallel_ZPt_30_55", 50, -175, 50);
    meZJets_u_par_ZPt_55_75 = ibooker.book1D("u_parallel_ZPt_55_75", "u_parallel_ZPt_55_75", 50, -175, 0);
    meZJets_u_par_ZPt_75_150 = ibooker.book1D("u_parallel_ZPt_75_150", "u_parallel_ZPt_75_150", 50, -300, 0);
    if (isCaloMet_) {
      meZJets_u_par_ZPt_150_290 = ibooker.book1D("u_parallel_ZPt_150_290", "u_parallel_ZPt_150_290", 50, -750, -100);
    } else {
      meZJets_u_par_ZPt_150_290 = ibooker.book1D("u_parallel_ZPt_150_290", "u_parallel_ZPt_150_290", 50, -450, -50);
    }
    if (isCaloMet_) {
      meZJets_u_par_ZPt_290 = ibooker.book1D("u_parallel_ZPt_290", "u_parallel_ZPt_290", 50, -1000., -350.);
    } else {
      meZJets_u_par_ZPt_290 = ibooker.book1D("u_parallel_ZPt_290", "u_parallel_ZPt_290", 50, -750., -150.);
    }
    meZJets_u_perp = ibooker.book1D("u_perp_Z_inc", "u_perp_Z_inc", 50, -85., 85.);
    meZJets_u_perp_ZPt_0_15 = ibooker.book1D("u_perp_ZPt_0_15", "u_perp_ZPt_0_15", 50, -85., 85.);
    meZJets_u_perp_ZPt_15_30 = ibooker.book1D("u_perp_ZPt_15_30", "u_perp_ZPt_15_30", 50, -85., 85.);
    meZJets_u_perp_ZPt_30_55 = ibooker.book1D("u_perp_ZPt_30_55", "u_perp_ZPt_30_55", 50, -85., 85.);
    meZJets_u_perp_ZPt_55_75 = ibooker.book1D("u_perp_ZPt_55_75", "u_perp_ZPt_55_75", 50, -85., 85.);
    meZJets_u_perp_ZPt_75_150 = ibooker.book1D("u_perp_ZPt_75_150", "u_perp_ZPt_75_150", 50, -85., 85.);
    meZJets_u_perp_ZPt_150_290 = ibooker.book1D("u_perp_ZPt_150_290", "u_perp_ZPt_150_290", 50, -85., 85.);
    meZJets_u_perp_ZPt_290 = ibooker.book1D("u_perp_ZPt_290", "u_perp_ZPt_290", 50, -85., 85.);

    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "u_parallel_Z_inc", meZJets_u_par));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_parallel_ZPt_0_15", meZJets_u_par_ZPt_0_15));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_parallel_ZPt_15_30", meZJets_u_par_ZPt_15_30));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_parallel_ZPt_30_55", meZJets_u_par_ZPt_30_55));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_parallel_ZPt_55_75", meZJets_u_par_ZPt_55_75));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_parallel_ZPt_75_150", meZJets_u_par_ZPt_75_150));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_parallel_ZPt_150_290", meZJets_u_par_ZPt_150_290));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_parallel_ZPt_290", meZJets_u_par_ZPt_290));

    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "u_perp_Z_inc", meZJets_u_perp));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_perp_ZPt_0_15", meZJets_u_perp_ZPt_0_15));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_perp_ZPt_15_30", meZJets_u_perp_ZPt_15_30));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_perp_ZPt_30_55", meZJets_u_perp_ZPt_30_55));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_perp_ZPt_55_75", meZJets_u_perp_ZPt_55_75));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_perp_ZPt_75_150", meZJets_u_perp_ZPt_75_150));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_perp_ZPt_150_290", meZJets_u_perp_ZPt_150_290));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "u_perp_ZPt_290", meZJets_u_perp_ZPt_290));
  }

  if (!fillZPlots) {
    hTrigger = ibooker.book1D("triggerResults", "triggerResults", 500, 0, 500);
    for (unsigned int i = 0; i < allTriggerNames_.size(); i++) {
      if (i < (unsigned int)hTrigger->getNbinsX()) {
        if (!hTriggerLabelsIsSet_) {
          hTrigger->setBinLabel(i + 1, allTriggerNames_[i]);
        }
      }
    }
    hTriggerLabelsIsSet_ = true;

    hMEx = ibooker.book1D("MEx", "MEx", 200, -500, 500);
    hMEy = ibooker.book1D("MEy", "MEy", 200, -500, 500);
    hMET = ibooker.book1D("MET", "MET", 200, 0, 1000);

    {
      auto scope = DQMStore::IBooker::UseLumiScope(ibooker);
      hMET_2 = ibooker.book1D("MET_2", "MET Range 2", 200, 0, 2000);
      hSumET = ibooker.book1D("SumET", "SumET", 400, 0, 4000);
      hMETSig = ibooker.book1D("METSig", "METSig", 51, 0, 51);
      hMETPhi = ibooker.book1D("METPhi", "METPhi", 60, -M_PI, M_PI);
    }

    hMET_logx = ibooker.book1D("MET_logx", "MET_logx", 40, -1, 9);
    hSumET_logx = ibooker.book1D("SumET_logx", "SumET_logx", 40, -1, 9);

    hMEx->setAxisTitle("MEx [GeV]", 1);
    hMEy->setAxisTitle("MEy [GeV]", 1);
    hMET->setAxisTitle("MET [GeV]", 1);
    hMET_2->setAxisTitle("MET [GeV]", 1);
    hSumET->setAxisTitle("SumET [GeV]", 1);
    hMETSig->setAxisTitle("METSig", 1);
    hMETPhi->setAxisTitle("METPhi [rad]", 1);
    hMET_logx->setAxisTitle("log(MET) [GeV]", 1);
    hSumET_logx->setAxisTitle("log(SumET) [GeV]", 1);

    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "triggerResults", hTrigger));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MEx", hMEx));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MEy", hMEy));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MET", hMET));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_2", hMET_2));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "SumET", hSumET));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METSig", hMETSig));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhi", hMETPhi));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_logx", hMET_logx));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "SumET_logx", hSumET_logx));

    hMET_HBHENoiseFilter = ibooker.book1D("MET_HBHENoiseFilter", "MET_HBHENoiseFiltered", 200, 0, 1000);
    hMET_CSCTightHaloFilter = ibooker.book1D("MET_CSCTightHaloFilter", "MET_CSCTightHaloFiltered", 200, 0, 1000);
    hMET_eeBadScFilter = ibooker.book1D("MET_eeBadScFilter", "MET_eeBadScFiltered", 200, 0, 1000);
    hMET_HBHEIsoNoiseFilter = ibooker.book1D("MET_HBHEIsoNoiseFilter", "MET_HBHEIsoNoiseFiltered", 200, 0, 1000);
    hMET_CSCTightHalo2015Filter =
        ibooker.book1D("MET_CSCTightHalo2015Filter", "MET_CSCTightHalo2015Filtered", 200, 0, 1000);
    hMET_EcalDeadCellTriggerFilter =
        ibooker.book1D("MET_EcalDeadCellTriggerFilter", "MET_EcalDeadCellTriggerFiltered", 200, 0, 1000);
    hMET_EcalDeadCellBoundaryFilter =
        ibooker.book1D("MET_EcalDeadCellBoundaryFilter", "MET_EcalDeadCellBoundaryFiltered", 200, 0, 1000);
    hMET_HcalStripHaloFilter = ibooker.book1D("MET_HcalStripHaloFilter", "MET_HcalStripHaloFiltered", 200, 0, 1000);

    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_HBHENoiseFilter", hMET_HBHENoiseFilter));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_CSCTightHaloFilter", hMET_CSCTightHaloFilter));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_eeBadScFilter", hMET_eeBadScFilter));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_HBHEIsoNoiseFilter", hMET_HBHEIsoNoiseFilter));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_CSCTightHalo2015Filter",
                                                              hMET_CSCTightHalo2015Filter));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_EcalDeadCellTriggerFilter",
                                                              hMET_EcalDeadCellTriggerFilter));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_EcalDeadCellBoundaryFilter",
                                                              hMET_EcalDeadCellBoundaryFilter));
    map_of_MEs.insert(
        std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_HcalStripHaloFilter", hMET_HcalStripHaloFilter));

    // Book NPV profiles --> would some of these profiles be interesting for other MET types too
    //----------------------------------------------------------------------------
    meMEx_profile = ibooker.bookProfile("MEx_profile", "met.px()", nbinsPV_, nPVMin_, nPVMax_, 200, -500, 500);
    meMEy_profile = ibooker.bookProfile("MEy_profile", "met.py()", nbinsPV_, nPVMin_, nPVMax_, 200, -500, 500);
    meMET_profile = ibooker.bookProfile("MET_profile", "met.pt()", nbinsPV_, nPVMin_, nPVMax_, 200, 0, 1000);
    meSumET_profile = ibooker.bookProfile("SumET_profile", "met.sumEt()", nbinsPV_, nPVMin_, nPVMax_, 400, 0, 4000);
    // Set NPV profiles x-axis title
    //----------------------------------------------------------------------------
    meMEx_profile->setAxisTitle("nvtx", 1);
    meMEy_profile->setAxisTitle("nvtx", 1);
    meMET_profile->setAxisTitle("nvtx", 1);
    meSumET_profile->setAxisTitle("nvtx", 1);

    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MEx_profile", meMEx_profile));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MEy_profile", meMEy_profile));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_profile", meMET_profile));
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "SumET_profile", meSumET_profile));

    if (isCaloMet_) {
      hCaloHadEtInHB = ibooker.book1D("CaloHadEtInHB", "CaloHadEtInHB", 50, 0, 2000);
      hCaloHadEtInHB->setAxisTitle("Had Et [GeV]", 1);
      hCaloHadEtInHO = ibooker.book1D("CaloHadEtInHO", "CaloHadEtInHO", 25, 0, 500);
      hCaloHadEtInHO->setAxisTitle("Had Et [GeV]", 1);
      hCaloHadEtInHE = ibooker.book1D("CaloHadEtInHE", "CaloHadEtInHE", 50, 0, 2000);
      hCaloHadEtInHE->setAxisTitle("Had Et [GeV]", 1);
      hCaloHadEtInHF = ibooker.book1D("CaloHadEtInHF", "CaloHadEtInHF", 50, 0, 1000);
      hCaloHadEtInHF->setAxisTitle("Had Et [GeV]", 1);
      hCaloEmEtInHF = ibooker.book1D("CaloEmEtInHF", "CaloEmEtInHF", 25, 0, 500);
      hCaloEmEtInHF->setAxisTitle("EM Et [GeV]", 1);
      hCaloEmEtInEE = ibooker.book1D("CaloEmEtInEE", "CaloEmEtInEE", 50, 0, 1000);
      hCaloEmEtInEE->setAxisTitle("EM Et [GeV]", 1);
      hCaloEmEtInEB = ibooker.book1D("CaloEmEtInEB", "CaloEmEtInEB", 50, 0, 2000);
      hCaloEmEtInEB->setAxisTitle("EM Et [GeV]", 1);

      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "CaloHadEtInHO", hCaloHadEtInHO));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "CaloHadEtInHF", hCaloHadEtInHF));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "CaloHadEtInHE", hCaloHadEtInHE));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "CaloHadEtInHB", hCaloHadEtInHB));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "CaloEmEtInHF", hCaloEmEtInHF));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "CaloEmEtInEE", hCaloEmEtInEE));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "CaloEmEtInEB", hCaloEmEtInEB));

      hCaloMETPhi020 = ibooker.book1D("CaloMETPhi020", "CaloMETPhi020", 60, -M_PI, M_PI);
      hCaloMETPhi020->setAxisTitle("METPhi [rad] (MET>20 GeV)", 1);

      hCaloEtFractionHadronic = ibooker.book1D("CaloEtFractionHadronic", "CaloEtFractionHadronic", 50, 0, 1);
      hCaloEtFractionHadronic->setAxisTitle("Hadronic Et Fraction", 1);
      hCaloEmEtFraction = ibooker.book1D("CaloEmEtFraction", "CaloEmEtFraction", 50, 0, 1);
      hCaloEmEtFraction->setAxisTitle("EM Et Fraction", 1);

      hCaloEmEtFraction020 = ibooker.book1D("CaloEmEtFraction020", "CaloEmEtFraction020", 50, 0, 1);
      hCaloEmEtFraction020->setAxisTitle("EM Et Fraction (MET>20 GeV)", 1);

      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "CaloMETPhi020", hCaloMETPhi020));
      map_of_MEs.insert(
          std::pair<std::string, MonitorElement*>(DirName + "/" + "CaloEtFractionHadronic", hCaloEtFractionHadronic));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "CaloEmEtFraction", hCaloEmEtFraction));
      map_of_MEs.insert(
          std::pair<std::string, MonitorElement*>(DirName + "/" + "CaloEmEtFraction020", hCaloEmEtFraction020));
    }

    if (isPFMet_) {
      if (fillPFCandPlots &&
          fillCandidateMap_histos) {  //first bool internal checks for subdirectory filling, second bool given in cfg file, checks that we fill maps only in one module in total

        meCHF_Barrel = ibooker.book1D("PfChargedHadronEtFractionBarrel", "chargedHadronEtFractionBarrel", 50, 0, 1);
        meCHF_EndcapPlus =
            ibooker.book1D("PfChargedHadronEtFractionEndcapPlus", "chargedHadronEtFractionEndcapPlus", 50, 0, 1);
        meCHF_EndcapMinus =
            ibooker.book1D("PfChargedHadronEtFractionEndcapMinus", "chargedHadronEtFractionEndcapMinus", 50, 0, 1);
        meCHF_Barrel_BXm1Empty = ibooker.book1D(
            "PfChargedHadronEtFractionBarrel_BXm1Empty", "chargedHadronEtFractionBarrel prev empty bunch", 50, 0, 1);
        meCHF_EndcapPlus_BXm1Empty = ibooker.book1D("PfChargedHadronEtFractionEndcapPlus_BXm1Empty",
                                                    "chargedHadronEtFractionEndcapPlus prev empty bunch",
                                                    50,
                                                    0,
                                                    1);
        meCHF_EndcapMinus_BXm1Empty = ibooker.book1D("PfChargedHadronEtFractionEndcapMinus_BXm1Empty",
                                                     "chargedHadronEtFractionEndcapMinus prev empty bunch",
                                                     50,
                                                     0,
                                                     1);
        meCHF_Barrel_BXm1Filled = ibooker.book1D("PfChargedHadronEtFractionBarrel_BXm1Filled",
                                                 "chargedHadronEtFractionBarrel prev filled 2 bunches",
                                                 50,
                                                 0,
                                                 1);
        meCHF_EndcapPlus_BXm1Filled = ibooker.book1D("PfChargedHadronEtFractionEndcapPlus_BXm1Filled",
                                                     "chargedHadronEtFractionEndcapPlus prev filled bunch",
                                                     50,
                                                     0,
                                                     1);
        meCHF_EndcapMinus_BXm1Filled = ibooker.book1D("PfChargedHadronEtFractionEndcapMinus_BXm1Filled",
                                                      "chargedHadronEtFractionEndcapMinus prev filled bunch",
                                                      50,
                                                      0,
                                                      1);

        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "PfChargedHadronEtFractionBarrel", meCHF_Barrel));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfChargedHadronEtFractionEndcapPlus",
                                                                  meCHF_EndcapPlus));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfChargedHadronEtFractionEndcapMinus", meCHF_EndcapMinus));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfChargedHadronEtFractionBarrel_BXm1Empty", meCHF_Barrel_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfChargedHadronEtFractionEndcapPlus_BXm1Empty", meCHF_EndcapPlus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfChargedHadronEtFractionEndcapMinus_BXm1Empty", meCHF_EndcapMinus_BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFractionBarrel_BXm2BXm1Empty",         meCHF_Barrel_BXm2BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFractionEndcapPlus_BXm2BXm1Empty",     meCHF_EndcapPlus_BXm2BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFractionEndcapMinus_BXm2BXm1Empty",    meCHF_EndcapMinus_BXm2BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfChargedHadronEtFractionBarrel_BXm1Filled", meCHF_Barrel_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfChargedHadronEtFractionEndcapPlus_BXm1Filled", meCHF_EndcapPlus_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfChargedHadronEtFractionEndcapMinus_BXm1Filled", meCHF_EndcapMinus_BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFractionBarrel_BXm2BXm1Filled",        meCHF_Barrel_BXm2BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFractionEndcapPlus_BXm2BXm1Filled",    meCHF_EndcapPlus_BXm2BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFractionEndcapMinus_BXm2BXm1Filled",   meCHF_EndcapMinus_BXm2BXm1Filled));

        meNHF_Barrel = ibooker.book1D("PfNeutralHadronEtFractionBarrel", "neutralHadronEtFractionBarrel", 50, 0, 1);
        meNHF_EndcapPlus =
            ibooker.book1D("PfNeutralHadronEtFractionEndcapPlus", "neutralHadronEtFractionEndcapPlus", 50, 0, 1);
        meNHF_EndcapMinus =
            ibooker.book1D("PfNeutralHadronEtFractionEndcapMinus", "neutralHadronEtFractionEndcapMinus", 50, 0, 1);
        meNHF_Barrel_BXm1Empty = ibooker.book1D(
            "PfNeutralHadronEtFractionBarrel_BXm1Empty", "neutralHadronEtFractionBarrel prev empty bunch", 50, 0, 1);
        meNHF_EndcapPlus_BXm1Empty = ibooker.book1D("PfNeutralHadronEtFractionEndcapPlus_BXm1Empty",
                                                    "neutralHadronEtFractionEndcapPlus prev empty bunch",
                                                    50,
                                                    0,
                                                    1);
        meNHF_EndcapMinus_BXm1Empty = ibooker.book1D("PfNeutralHadronEtFractionEndcapMinus_BXm1Empty",
                                                     "neutralHadronEtFractionEndcapMinus prev empty bunch",
                                                     50,
                                                     0,
                                                     1);
        //meNHF_Barrel_BXm2BXm1Empty         = ibooker.book1D("PfNeutralHadronEtFractionBarrel_BXm2BXm1Empty",         "neutralHadronEtFractionBarrel prev empty 2 bunches",         50, 0,    1);
        //meNHF_EndcapPlus_BXm2BXm1Empty     = ibooker.book1D("PfNeutralHadronEtFractionEndcapPlus_BXm2BXm1Empty",     "neutralHadronEtFractionEndcapPlus prev empty 2 bunches",     50, 0,    1);
        //meNHF_EndcapMinus_BXm2BXm1Empty    = ibooker.book1D("PfNeutralHadronEtFractionEndcapMinus_BXm2BXm1Empty",    "neutralHadronEtFractionEndcapMinus prev empty 2 bunches",    50, 0,    1);
        meNHF_Barrel_BXm1Filled = ibooker.book1D("PfNeutralHadronEtFractionBarrel_BXm1Filled",
                                                 "neutralHadronEtFractionBarrel prev filled 2 bunches",
                                                 50,
                                                 0,
                                                 1);
        meNHF_EndcapPlus_BXm1Filled = ibooker.book1D("PfNeutralHadronEtFractionEndcapPlus_BXm1Filled",
                                                     "neutralHadronEtFractionEndcapPlus prev filled bunch",
                                                     50,
                                                     0,
                                                     1);
        meNHF_EndcapMinus_BXm1Filled = ibooker.book1D("PfNeutralHadronEtFractionEndcapMinus_BXm1Filled",
                                                      "neutralHadronEtFractionEndcapMinus prev filled bunch",
                                                      50,
                                                      0,
                                                      1);
        //meNHF_Barrel_BXm2BXm1Filled        = ibooker.book1D("PfNeutralHadronEtFractionBarrel_BXm2BXm1Filled",        "neutralHadronEtFractionBarrel prev filled 2 bunches",        50, 0,    1);
        //meNHF_EndcapPlus_BXm2BXm1Filled    = ibooker.book1D("PfNeutralHadronEtFractionEndcapPlus_BXm2BXm1Filled",    "neutralHadronEtFractionEndcapPlus prev filled 2 bunches",    50, 0,    1);
        //meNHF_EndcapMinus_BXm2BXm1Filled   = ibooker.book1D("PfNeutralHadronEtFractionEndcapMinus_BXm2BXm1Filled",   "neutralHadronEtFractionEndcapMinus prev filled 2 bunches",   50, 0,    1);

        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "PfNeutralHadronEtFractionBarrel", meNHF_Barrel));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfNeutralHadronEtFractionEndcapPlus",
                                                                  meNHF_EndcapPlus));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfNeutralHadronEtFractionEndcapMinus", meNHF_EndcapMinus));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfNeutralHadronEtFractionBarrel_BXm1Empty", meNHF_Barrel_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfNeutralHadronEtFractionEndcapPlus_BXm1Empty", meNHF_EndcapPlus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfNeutralHadronEtFractionEndcapMinus_BXm1Empty", meNHF_EndcapMinus_BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFractionBarrel_BXm2BXm1Empty",         meNHF_Barrel_BXm2BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFractionEndcapPlus_BXm2BXm1Empty",     meNHF_EndcapPlus_BXm2BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFractionEndcapMinus_BXm2BXm1Empty",    meNHF_EndcapMinus_BXm2BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfNeutralHadronEtFractionBarrel_BXm1Filled", meNHF_Barrel_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfNeutralHadronEtFractionEndcapPlus_BXm1Filled", meNHF_EndcapPlus_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfNeutralHadronEtFractionEndcapMinus_BXm1Filled", meNHF_EndcapMinus_BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFractionBarrel_BXm2BXm1Filled",        meNHF_Barrel_BXm2BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFractionEndcapPlus_BXm2BXm1Filled",    meNHF_EndcapPlus_BXm2BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFractionEndcapMinus_BXm2BXm1Filled",   meNHF_EndcapMinus_BXm2BXm1Filled));

        mePhF_Barrel = ibooker.book1D("PfPhotonEtFractionBarrel", "photonEtFractionBarrel", 50, 0, 1);
        mePhF_EndcapPlus = ibooker.book1D("PfPhotonEtFractionEndcapPlus", "photonEtFractionEndcapPlus", 50, 0, 1);
        mePhF_EndcapMinus = ibooker.book1D("PfPhotonEtFractionEndcapMinus", "photonEtFractionEndcapMinus", 50, 0, 1);
        mePhF_Barrel_BXm1Empty =
            ibooker.book1D("PfPhotonEtFractionBarrel_BXm1Empty", "photonEtFractionBarrel prev empty bunch", 50, 0, 1);
        mePhF_EndcapPlus_BXm1Empty = ibooker.book1D(
            "PfPhotonEtFractionEndcapPlus_BXm1Empty", "photonEtFractionEndcapPlus prev empty bunch", 50, 0, 1);
        mePhF_EndcapMinus_BXm1Empty = ibooker.book1D(
            "PfPhotonEtFractionEndcapMinus_BXm1Empty", "photonEtFractionEndcapMinus prev empty bunch", 50, 0, 1);
        //mePhF_Barrel_BXm2BXm1Empty         = ibooker.book1D("PfPhotonEtFractionBarrel_BXm2BXm1Empty",         "photonEtFractionBarrel prev empty 2 bunches",         50, 0,    1);
        //mePhF_EndcapPlus_BXm2BXm1Empty     = ibooker.book1D("PfPhotonEtFractionEndcapPlus_BXm2BXm1Empty",     "photonEtFractionEndcapPlus prev empty 2 bunches",     50, 0,    1);
        //mePhF_EndcapMinus_BXm2BXm1Empty    = ibooker.book1D("PfPhotonEtFractionEndcapMinus_BXm2BXm1Empty",    "photonEtFractionEndcapMinus prev empty 2 bunches",    50, 0,    1);
        mePhF_Barrel_BXm1Filled = ibooker.book1D(
            "PfPhotonEtFractionBarrel_BXm1Filled", "photonEtFractionBarrel prev filled 2 bunches", 50, 0, 1);
        mePhF_EndcapPlus_BXm1Filled = ibooker.book1D(
            "PfPhotonEtFractionEndcapPlus_BXm1Filled", "photonEtFractionEndcapPlus prev filled bunch", 50, 0, 1);
        mePhF_EndcapMinus_BXm1Filled = ibooker.book1D(
            "PfPhotonEtFractionEndcapMinus_BXm1Filled", "photonEtFractionEndcapMinus prev filled bunch", 50, 0, 1);
        //mePhF_Barrel_BXm2BXm1Filled        = ibooker.book1D("PfPhotonEtFractionBarrel_BXm2BXm1Filled",        "photonEtFractionBarrel prev filled 2 bunches",        50, 0,    1);
        //mePhF_EndcapPlus_BXm2BXm1Filled    = ibooker.book1D("PfPhotonEtFractionEndcapPlus_BXm2BXm1Filled",    "photonEtFractionEndcapPlus prev filled 2 bunches",    50, 0,    1);
        //mePhF_EndcapMinus_BXm2BXm1Filled   = ibooker.book1D("PfPhotonEtFractionEndcapMinus_BXm2BXm1Filled",   "photonEtFractionEndcapMinus prev filled 2 bunches",   50, 0,    1);

        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "PfPhotonEtFractionBarrel", mePhF_Barrel));
        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "PfPhotonEtFractionEndcapPlus", mePhF_EndcapPlus));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfPhotonEtFractionEndcapMinus",
                                                                  mePhF_EndcapMinus));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfPhotonEtFractionBarrel_BXm1Empty",
                                                                  mePhF_Barrel_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfPhotonEtFractionEndcapPlus_BXm1Empty", mePhF_EndcapPlus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfPhotonEtFractionEndcapMinus_BXm1Empty", mePhF_EndcapMinus_BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFractionBarrel_BXm2BXm1Empty",         mePhF_Barrel_BXm2BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFractionEndcapPlus_BXm2BXm1Empty",     mePhF_EndcapPlus_BXm2BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFractionEndcapMinus_BXm2BXm1Empty",    mePhF_EndcapMinus_BXm2BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfPhotonEtFractionBarrel_BXm1Filled",
                                                                  mePhF_Barrel_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfPhotonEtFractionEndcapPlus_BXm1Filled", mePhF_EndcapPlus_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfPhotonEtFractionEndcapMinus_BXm1Filled", mePhF_EndcapMinus_BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFractionBarrel_BXm2BXm1Filled",        mePhF_Barrel_BXm2BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFractionEndcapPlus_BXm2BXm1Filled",    mePhF_EndcapPlus_BXm2BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFractionEndcapMinus_BXm2BXm1Filled",   mePhF_EndcapMinus_BXm2BXm1Filled));

        meHFHadF_Plus = ibooker.book1D("PfHFHadronEtFractionPlus", "HFHadronEtFractionPlus", 50, 0, 1);
        meHFHadF_Minus = ibooker.book1D("PfHFHadronEtFractionMinus", "HFHadronEtFractionMinus", 50, 0, 1);
        meHFHadF_Plus_BXm1Empty =
            ibooker.book1D("PfHFHadronEtFractionPlus_BXm1Empty", "HFHadronEtFractionPlus prev empty bunch", 50, 0, 1);
        meHFHadF_Minus_BXm1Empty =
            ibooker.book1D("PfHFHadronEtFractionMinus_BXm1Empty", "HFHadronEtFractionMinus prev empty bunch", 50, 0, 1);
        //meHFHadF_Plus_BXm2BXm1Empty     = ibooker.book1D("PfHFHadronEtFractionPlus_BXm2BXm1Empty",     "HFHadronEtFractionPlus prev empty 2 bunches",     50, 0,    1);
        //meHFHadF_Minus_BXm2BXm1Empty    = ibooker.book1D("PfHFHadronEtFractionMinus_BXm2BXm1Empty",    "HFHadronEtFractionMinus prev empty 2 bunches",    50, 0,    1);
        meHFHadF_Plus_BXm1Filled =
            ibooker.book1D("PfHFHadronEtFractionPlus_BXm1Filled", "HFHadronEtFractionPlus prev filled bunch", 50, 0, 1);
        meHFHadF_Minus_BXm1Filled = ibooker.book1D(
            "PfHFHadronEtFractionMinus_BXm1Filled", "HFHadronEtFractionMinus prev filled bunch", 50, 0, 1);
        //meHFHadF_Plus_BXm2BXm1Filled    = ibooker.book1D("PfHFHadronEtFractionPlus_BXm2BXm1Filled",    "HFHadronEtFractionPlus prev filled 2 bunches",    50, 0,    1);
        //meHFHadF_Minus_BXm2BXm1Filled   = ibooker.book1D("PfHFHadronEtFractionMinus_BXm2BXm1Filled",   "HFHadronEtFractionMinus prev filled 2 bunches",   50, 0,    1);

        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFHadronEtFractionPlus", meHFHadF_Plus));
        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFHadronEtFractionMinus", meHFHadF_Minus));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFHadronEtFractionPlus_BXm1Empty",
                                                                  meHFHadF_Plus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFHadronEtFractionMinus_BXm1Empty",
                                                                  meHFHadF_Minus_BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEtFractionPlus_BXm2BXm1Empty",     meHFHadF_Plus_BXm2BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEtFractionMinus_BXm2BXm1Empty",    meHFHadF_Minus_BXm2BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFHadronEtFractionPlus_BXm1Filled",
                                                                  meHFHadF_Plus_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "PfHFHadronEtFractionMinus_BXm1Filled", meHFHadF_Minus_BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEtFractionPlus_BXm2BXm1Filled",    meHFHadF_Plus_BXm2BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEtFractionMinus_BXm2BXm1Filled",   meHFHadF_Minus_BXm2BXm1Filled));

        meHFEMF_Plus = ibooker.book1D("PfHFEMEtFractionPlus", "HFEMEtFractionPlus", 50, 0, 1);
        meHFEMF_Minus = ibooker.book1D("PfHFEMEtFractionMinus", "HFEMEtFractionMinus", 50, 0, 1);
        meHFEMF_Plus_BXm1Empty =
            ibooker.book1D("PfHFEMEtFractionPlus_BXm1Empty", "HFEMEtFractionPlus prev empty bunch", 50, 0, 1);
        meHFEMF_Minus_BXm1Empty =
            ibooker.book1D("PfHFEMEtFractionMinus_BXm1Empty", "HFEMEtFractionMinus prev empty bunch", 50, 0, 1);
        //meHFEMF_Plus_BXm2BXm1Empty     = ibooker.book1D("PfHFEMEtFractionPlus_BXm2BXm1Empty",     "HFEMEtFractionPlus prev empty 2 bunches",     50, 0,    1);
        //meHFEMF_Minus_BXm2BXm1Empty    = ibooker.book1D("PfHFEMEtFractionMinus_BXm2BXm1Empty",    "HFEMEtFractionMinus prev empty 2 bunches",    50, 0,    1);
        meHFEMF_Plus_BXm1Filled =
            ibooker.book1D("PfHFEMEtFractionPlus_BXm1Filled", "HFEMEtFractionPlus prev filled bunch", 50, 0, 1);
        meHFEMF_Minus_BXm1Filled =
            ibooker.book1D("PfHFEMEtFractionMinus_BXm1Filled", "HFEMEtFractionMinus prev filled bunch", 50, 0, 1);
        //meHFEMF_Plus_BXm2BXm1Filled    = ibooker.book1D("PfHFEMEtFractionPlus_BXm2BXm1Filled",    "HFEMEtFractionPlus prev filled 2 bunches",    50, 0,    1);
        //meHFEMF_Minus_BXm2BXm1Filled   = ibooker.book1D("PfHFEMEtFractionMinus_BXm2BXm1Filled",   "HFEMEtFractionMinus prev filled 2 bunches",   50, 0,    1);

        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFEMEtFractionPlus", meHFEMF_Plus));
        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFEMEtFractionMinus", meHFEMF_Minus));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFEMEtFractionPlus_BXm1Empty",
                                                                  meHFEMF_Plus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFEMEtFractionMinus_BXm1Empty",
                                                                  meHFEMF_Minus_BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEtFractionPlus_BXm2BXm1Empty",     meHFEMF_Plus_BXm2BXm1Empty));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEtFractionMinus_BXm2BXm1Empty",    meHFEMF_Minus_BXm2BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFEMEtFractionPlus_BXm1Filled",
                                                                  meHFEMF_Plus_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFEMEtFractionMinus_BXm1Filled",
                                                                  meHFEMF_Minus_BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEtFractionPlus_BXm2BXm1Filled",    meHFEMF_Plus_BXm2BXm1Filled));
        //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEtFractionMinus_BXm2BXm1Filled",   meHFEMF_Minus_BXm2BXm1Filled));
        /*
	meMETPhiChargedHadronsBarrel_BXm2BXm1Filled       = ibooker.book1D("METPhiChargedHadronsBarrel_BXm2BXm1Filled",     "METPhi_PFChargedHadronsBarrel prev two bunches filled",       50, -M_PI,M_PI);
	meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Filled   = ibooker.book1D("METPhiChargedHadronsEndcapPlus_BXm2BXm1Filled", "METPhi_PFChargedHadronsEndcapPlus prev two bunches filled",   50, -M_PI,M_PI);
	meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Filled  = ibooker.book1D("METPhiChargedHadronsEndcapMinus_BXm2BXm1Filled","METPhi_PFChargedHadronsEndcapMinus prev two bunches filled",  50, -M_PI,M_PI);
	meMETPhiNeutralHadronsBarrel_BXm2BXm1Filled       = ibooker.book1D("METPhiNeutralHadronsBarrel_BXm2BXm1Filled",     "METPhi_PFNeutralHadronsBarrel prev two bunches filled",       50, -M_PI,M_PI);
	meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Filled   = ibooker.book1D("METPhiNeutralHadronsEndcapPlus_BXm2BXm1Filled", "METPhi_PFNeutralHadronsEndcapPlus prev two bunches filled",   50, -M_PI,M_PI);
	meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Filled  = ibooker.book1D("METPhiNeutralHadronsEndcapMinus_BXm2BXm1Filled","METPhi_PFNeutralHadronsEndcapMinus prev two bunches filled",  50, -M_PI,M_PI);
	meMETPhiPhotonsBarrel_BXm2BXm1Filled              = ibooker.book1D("METPhiPhotonsBarrel_BXm2BXm1Filled",            "METPhi_PFPhotonsBarrel prev two bunches filled",              50, -M_PI,M_PI);
	meMETPhiPhotonsEndcapPlus_BXm2BXm1Filled          = ibooker.book1D("METPhiPhotonsEndcapPlus_BXm2BXm1Filled",        "METPhi_PFPhotonsEndcapPlus prev two bunches filled",          50, -M_PI,M_PI);
	meMETPhiPhotonsEndcapMinus_BXm2BXm1Filled         = ibooker.book1D("METPhiPhotonsEndcapMinus_BXm2BXm1Filled",       "METPhi_PFPhotonsEndcapMinus prev two bunches filled",         50, -M_PI,M_PI);
	meMETPhiHFHadronsPlus_BXm2BXm1Filled              = ibooker.book1D("METPhiHFHadronsPlus_BXm2BXm1Filled",            "METPhi_PFHFHadronsPlus prev two bunches filled",              50, -M_PI,M_PI);
	meMETPhiHFHadronsMinus_BXm2BXm1Filled             = ibooker.book1D("METPhiHFHadronsMinus_BXm2BXm1Filled",           "METPhi_PFHFHadronsMinus prev two bunches filled",             50, -M_PI,M_PI);
	meMETPhiHFEGammasPlus_BXm2BXm1Filled              = ibooker.book1D("METPhiHFEGammasPlus_BXm2BXm1Filled",            "METPhi_PFHFEGammasPlus prev two bunches filled",              50, -M_PI,M_PI);
	meMETPhiHFEGammasMinus_BXm2BXm1Filled             = ibooker.book1D("METPhiHFEGammasMinus_BXm2BXm1Filled",           "METPhi_PFHFEGammasMinus prev two bunches filled",             50, -M_PI,M_PI);
	
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiChargedHadronsBarrel_BXm2BXm1Filled"         ,meMETPhiChargedHadronsBarrel_BXm2BXm1Filled));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiChargedHadronsEndcapPlus_BXm2BXm1Filled"     ,meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Filled));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiChargedHadronsEndcapMinus_BXm2BXm1Filled"    ,meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Filled));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiNeutralHadronsBarrel_BXm2BXm1Filled"         ,meMETPhiNeutralHadronsBarrel_BXm2BXm1Filled));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiNeutralHadronsEndcapPlus_BXm2BXm1Filled"     ,meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Filled));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiNeutralHadronsEndcapMinus_BXm2BXm1Filled"    ,meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Filled));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiPhotonsBarrel_BXm2BXm1Filled"                ,meMETPhiPhotonsBarrel_BXm2BXm1Filled));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiPhotonsEndcapPlus_BXm2BXm1Filled"            ,meMETPhiPhotonsEndcapPlus_BXm2BXm1Filled));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiPhotonsEndcapMinus_BXm2BXm1Filled"           ,meMETPhiPhotonsEndcapMinus_BXm2BXm1Filled));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiHFHadronsPlus_BXm2BXm1Filled"                ,meMETPhiHFHadronsPlus_BXm2BXm1Filled));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiHFHadronsMinus_BXm2BXm1Filled"               ,meMETPhiHFHadronsMinus_BXm2BXm1Filled));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiHFEGammasPlus_BXm2BXm1Filled"                ,meMETPhiHFEGammasPlus_BXm2BXm1Filled));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiHFEGammasMinus_BXm2BXm1Filled"               ,meMETPhiHFEGammasMinus_BXm2BXm1Filled));
	
	meMETPhiChargedHadronsBarrel_BXm2BXm1Empty       = ibooker.book1D("METPhiChargedHadronsBarrel_BXm2BXm1Empty",     "METPhi_PFChargedHadronsBarrel prev two bunches empty",       50, -M_PI,M_PI);
	meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Empty   = ibooker.book1D("METPhiChargedHadronsEndcapPlus_BXm2BXm1Empty", "METPhi_PFChargedHadronsEndcapPlus prev two bunches empty",   50, -M_PI,M_PI);
	meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Empty  = ibooker.book1D("METPhiChargedHadronsEndcapMinus_BXm2BXm1Empty","METPhi_PFChargedHadronsEndcapMinus prev two bunches empty",  50, -M_PI,M_PI);
	meMETPhiNeutralHadronsBarrel_BXm2BXm1Empty       = ibooker.book1D("METPhiNeutralHadronsBarrel_BXm2BXm1Empty",     "METPhi_PFNeutralHadronsBarrel prev two bunches empty",       50, -M_PI,M_PI);
	meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Empty   = ibooker.book1D("METPhiNeutralHadronsEndcapPlus_BXm2BXm1Empty", "METPhi_PFNeutralHadronsEndcapPlus prev two bunches empty",   50, -M_PI,M_PI);
	meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Empty  = ibooker.book1D("METPhiNeutralHadronsEndcapMinus_BXm2BXm1Empty","METPhi_PFNeutralHadronsEndcapMinus prev two bunches empty",  50, -M_PI,M_PI);
	meMETPhiPhotonsBarrel_BXm2BXm1Empty              = ibooker.book1D("METPhiPhotonsBarrel_BXm2BXm1Empty",            "METPhi_PFPhotonsBarrel prev two bunches empty",              50, -M_PI,M_PI);
	meMETPhiPhotonsEndcapPlus_BXm2BXm1Empty          = ibooker.book1D("METPhiPhotonsEndcapPlus_BXm2BXm1Empty",        "METPhi_PFPhotonsEndcapPlus prev two bunches empty",          50, -M_PI,M_PI);
	meMETPhiPhotonsEndcapMinus_BXm2BXm1Empty         = ibooker.book1D("METPhiPhotonsEndcapMinus_BXm2BXm1Empty",       "METPhi_PFPhotonsEndcapMinus prev two bunches empty",         50, -M_PI,M_PI);
	meMETPhiHFHadronsPlus_BXm2BXm1Empty              = ibooker.book1D("METPhiHFHadronsPlus_BXm2BXm1Empty",            "METPhi_PFHFHadronsPlus prev two bunches empty",              50, -M_PI,M_PI);
	meMETPhiHFHadronsMinus_BXm2BXm1Empty             = ibooker.book1D("METPhiHFHadronsMinus_BXm2BXm1Empty",           "METPhi_PFHFHadronsMinus prev two bunches empty",             50, -M_PI,M_PI);
	meMETPhiHFEGammasPlus_BXm2BXm1Empty              = ibooker.book1D("METPhiHFEGammasPlus_BXm2BXm1Empty",            "METPhi_PFHFEGammasPlus prev two bunches empty",              50, -M_PI,M_PI);
	meMETPhiHFEGammasMinus_BXm2BXm1Empty             = ibooker.book1D("METPhiHFEGammasMinus_BXm2BXm1Empty",           "METPhi_PFHFEGammasMinus prev two bunches empty",             50, -M_PI,M_PI);
	
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiChargedHadronsBarrel_BXm2BXm1Empty"         ,meMETPhiChargedHadronsBarrel_BXm2BXm1Empty));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiChargedHadronsEndcapPlus_BXm2BXm1Empty"     ,meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Empty));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiChargedHadronsEndcapMinus_BXm2BXm1Empty"    ,meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Empty));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiNeutralHadronsBarrel_BXm2BXm1Empty"         ,meMETPhiNeutralHadronsBarrel_BXm2BXm1Empty));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiNeutralHadronsEndcapPlus_BXm2BXm1Empty"     ,meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Empty));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiNeutralHadronsEndcapMinus_BXm2BXm1Empty"    ,meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Empty));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiPhotonsBarrel_BXm2BXm1Empty"                ,meMETPhiPhotonsBarrel_BXm2BXm1Empty));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiPhotonsEndcapPlus_BXm2BXm1Empty"            ,meMETPhiPhotonsEndcapPlus_BXm2BXm1Empty));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiPhotonsEndcapMinus_BXm2BXm1Empty"           ,meMETPhiPhotonsEndcapMinus_BXm2BXm1Empty));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiHFHadronsPlus_BXm2BXm1Empty"                ,meMETPhiHFHadronsPlus_BXm2BXm1Empty));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiHFHadronsMinus_BXm2BXm1Empty"               ,meMETPhiHFHadronsMinus_BXm2BXm1Empty));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiHFEGammasPlus_BXm2BXm1Empty"                ,meMETPhiHFEGammasPlus_BXm2BXm1Empty));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhiHFEGammasMinus_BXm2BXm1Empty"               ,meMETPhiHFEGammasMinus_BXm2BXm1Empty));
	*/
        //histos where one previous bunch was empty/filled
        mePhotonEtFraction_BXm1Empty =
            ibooker.book1D("PfPhotonEtFraction_BXm1Empty", "photonEtFraction() prev empty bunch", 50, 0, 1);
        mePhotonEtFraction_BXm1Filled =
            ibooker.book1D("PfPhotonEtFraction_BXm1Filled", "photonEtFraction() prev filled bunch", 50, 0, 1);
        meNeutralHadronEtFraction_BXm1Empty = ibooker.book1D(
            "PfNeutralHadronEtFraction_BXm1Empty", "neutralHadronEtFraction() prev empty bunch", 50, 0, 1);
        meNeutralHadronEtFraction_BXm1Filled = ibooker.book1D(
            "PfNeutralHadronEtFraction_BXm1Filled", "neutralHadronEtFraction() prev filled bunch", 50, 0, 1);
        meChargedHadronEtFraction_BXm1Empty = ibooker.book1D(
            "PfChargedHadronEtFraction_BXm1Empty", "chargedHadronEtFraction() prev empty bunch", 50, 0, 1);
        meChargedHadronEtFraction_BXm1Filled = ibooker.book1D(
            "PfChargedHadronEtFraction_BXm1Filled", "chargedHadronEtFraction() prev filled bunch", 50, 0, 1);
        meMET_BXm1Empty = ibooker.book1D("MET_BXm1Empty", "MET prev empty bunch", 200, 0, 1000);
        meMET_BXm1Filled = ibooker.book1D("MET_BXm1Filled", "MET prev filled bunch", 200, 0, 1000);
        meSumET_BXm1Empty = ibooker.book1D("SumET_BXm1Empty", "SumET prev empty bunch", 400, 0, 4000);
        meSumET_BXm1Filled = ibooker.book1D("SumET_BXm1Filled", "SumET prev filled bunch", 400, 0, 4000);

        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfPhotonEtFraction_BXm1Empty",
                                                                  mePhotonEtFraction_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfPhotonEtFraction_BXm1Filled",
                                                                  mePhotonEtFraction_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfNeutralHadronEtFraction_BXm1Empty",
                                                                  meNeutralHadronEtFraction_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfNeutralEtFraction_BXm1Filled",
                                                                  meNeutralHadronEtFraction_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfChargedHadronEtFraction_BXm1Empty",
                                                                  meChargedHadronEtFraction_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfChargedEtFraction_BXm1Filled",
                                                                  meChargedHadronEtFraction_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_BXm1Empty", meMET_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MET_BXm1Filled", meMET_BXm1Filled));
        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "SumET_BXm1Empty", meSumET_BXm1Empty));
        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "SumET_BXm1Filled", meSumET_BXm1Filled));

        meMETPhiChargedHadronsBarrel_BXm1Filled = ibooker.book1D(
            "METPhiChargedHadronsBarrel_BXm1Filled", "METPhi_PFChargedHadronsBarrel prev bunch filled", 50, -M_PI, M_PI);
        meMETPhiChargedHadronsEndcapPlus_BXm1Filled =
            ibooker.book1D("METPhiChargedHadronsEndcapPlus_BXm1Filled",
                           "METPhi_PFChargedHadronsEndcapPlus prev bunch filled",
                           50,
                           -M_PI,
                           M_PI);
        meMETPhiChargedHadronsEndcapMinus_BXm1Filled =
            ibooker.book1D("METPhiChargedHadronsEndcapMinus_BXm1Filled",
                           "METPhi_PFChargedHadronsEndcapMinus prev bunch filled",
                           50,
                           -M_PI,
                           M_PI);
        meMETPhiNeutralHadronsBarrel_BXm1Filled = ibooker.book1D(
            "METPhiNeutralHadronsBarrel_BXm1Filled", "METPhi_PFNeutralHadronsBarrel prev bunch filled", 50, -M_PI, M_PI);
        meMETPhiNeutralHadronsEndcapPlus_BXm1Filled =
            ibooker.book1D("METPhiNeutralHadronsEndcapPlus_BXm1Filled",
                           "METPhi_PFNeutralHadronsEndcapPlus prev bunch filled",
                           50,
                           -M_PI,
                           M_PI);
        meMETPhiNeutralHadronsEndcapMinus_BXm1Filled =
            ibooker.book1D("METPhiNeutralHadronsEndcapMinus_BXm1Filled",
                           "METPhi_PFNeutralHadronsEndcapMinus prev bunch filled",
                           50,
                           -M_PI,
                           M_PI);
        meMETPhiPhotonsBarrel_BXm1Filled = ibooker.book1D(
            "METPhiPhotonsBarrel_BXm1Filled", "METPhi_PFPhotonsBarrel prev bunch filled", 50, -M_PI, M_PI);
        meMETPhiPhotonsEndcapPlus_BXm1Filled = ibooker.book1D(
            "METPhiPhotonsEndcapPlus_BXm1Filled", "METPhi_PFPhotonsEndcapPlus prev bunch filled", 50, -M_PI, M_PI);
        meMETPhiPhotonsEndcapMinus_BXm1Filled = ibooker.book1D(
            "METPhiPhotonsEndcapMinus_BXm1Filled", "METPhi_PFPhotonsEndcapMinus prev bunch filled", 50, -M_PI, M_PI);
        meMETPhiHFHadronsPlus_BXm1Filled = ibooker.book1D(
            "METPhiHFHadronsPlus_BXm1Filled", "METPhi_PFHFHadronsPlus prev bunch filled", 50, -M_PI, M_PI);
        meMETPhiHFHadronsMinus_BXm1Filled = ibooker.book1D(
            "METPhiHFHadronsMinus_BXm1Filled", "METPhi_PFHFHadronsMinus prev bunch filled", 50, -M_PI, M_PI);
        meMETPhiHFEGammasPlus_BXm1Filled = ibooker.book1D(
            "METPhiHFEGammasPlus_BXm1Filled", "METPhi_PFHFEGammasPlus prev bunch filled", 50, -M_PI, M_PI);
        meMETPhiHFEGammasMinus_BXm1Filled = ibooker.book1D(
            "METPhiHFEGammasMinus_BXm1Filled", "METPhi_PFHFEGammasMinus prev bunch filled", 50, -M_PI, M_PI);

        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "METPhiChargedHadronsBarrel_BXm1Filled", meMETPhiChargedHadronsBarrel_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "METPhiChargedHadronsEndcapPlus_BXm1Filled", meMETPhiChargedHadronsEndcapPlus_BXm1Filled));
        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiChargedHadronsEndcapMinus_BXm1Filled",
                                                    meMETPhiChargedHadronsEndcapMinus_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "METPhiNeutralHadronsBarrel_BXm1Filled", meMETPhiNeutralHadronsBarrel_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "METPhiNeutralHadronsEndcapPlus_BXm1Filled", meMETPhiNeutralHadronsEndcapPlus_BXm1Filled));
        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiNeutralHadronsEndcapMinus_BXm1Filled",
                                                    meMETPhiNeutralHadronsEndcapMinus_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiPhotonsBarrel_BXm1Filled",
                                                                  meMETPhiPhotonsBarrel_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiPhotonsEndcapPlus_BXm1Filled",
                                                                  meMETPhiPhotonsEndcapPlus_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiPhotonsEndcapMinus_BXm1Filled",
                                                                  meMETPhiPhotonsEndcapMinus_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiHFHadronsPlus_BXm1Filled",
                                                                  meMETPhiHFHadronsPlus_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiHFHadronsMinus_BXm1Filled",
                                                                  meMETPhiHFHadronsMinus_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiHFEGammasPlus_BXm1Filled",
                                                                  meMETPhiHFEGammasPlus_BXm1Filled));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiHFEGammasMinus_BXm1Filled",
                                                                  meMETPhiHFEGammasMinus_BXm1Filled));

        meMETPhiChargedHadronsBarrel_BXm1Empty = ibooker.book1D(
            "METPhiChargedHadronsBarrel_BXm1Empty", "METPhi_PFChargedHadronsBarrel prev bunch empty", 50, -M_PI, M_PI);
        meMETPhiChargedHadronsEndcapPlus_BXm1Empty =
            ibooker.book1D("METPhiChargedHadronsEndcapPlus_BXm1Empty",
                           "METPhi_PFChargedHadronsEndcapPlus prev bunch empty",
                           50,
                           -M_PI,
                           M_PI);
        meMETPhiChargedHadronsEndcapMinus_BXm1Empty =
            ibooker.book1D("METPhiChargedHadronsEndcapMinus_BXm1Empty",
                           "METPhi_PFChargedHadronsEndcapMinus prev bunch empty",
                           50,
                           -M_PI,
                           M_PI);
        meMETPhiNeutralHadronsBarrel_BXm1Empty = ibooker.book1D(
            "METPhiNeutralHadronsBarrel_BXm1Empty", "METPhi_PFNeutralHadronsBarrel prev bunch empty", 50, -M_PI, M_PI);
        meMETPhiNeutralHadronsEndcapPlus_BXm1Empty =
            ibooker.book1D("METPhiNeutralHadronsEndcapPlus_BXm1Empty",
                           "METPhi_PFNeutralHadronsEndcapPlus prev bunch empty",
                           50,
                           -M_PI,
                           M_PI);
        meMETPhiNeutralHadronsEndcapMinus_BXm1Empty =
            ibooker.book1D("METPhiNeutralHadronsEndcapMinus_BXm1Empty",
                           "METPhi_PFNeutralHadronsEndcapMinus prev bunch empty",
                           50,
                           -M_PI,
                           M_PI);
        meMETPhiPhotonsBarrel_BXm1Empty =
            ibooker.book1D("METPhiPhotonsBarrel_BXm1Empty", "METPhi_PFPhotonsBarrel prev bunch empty", 50, -M_PI, M_PI);
        meMETPhiPhotonsEndcapPlus_BXm1Empty = ibooker.book1D(
            "METPhiPhotonsEndcapPlus_BXm1Empty", "METPhi_PFPhotonsEndcapPlus prev bunch empty", 50, -M_PI, M_PI);
        meMETPhiPhotonsEndcapMinus_BXm1Empty = ibooker.book1D(
            "METPhiPhotonsEndcapMinus_BXm1Empty", "METPhi_PFPhotonsEndcapMinus prev bunch empty", 50, -M_PI, M_PI);
        meMETPhiHFHadronsPlus_BXm1Empty =
            ibooker.book1D("METPhiHFHadronsPlus_BXm1Empty", "METPhi_PFHFHadronsPlus prev bunch empty", 50, -M_PI, M_PI);
        meMETPhiHFHadronsMinus_BXm1Empty = ibooker.book1D(
            "METPhiHFHadronsMinus_BXm1Empty", "METPhi_PFHFHadronsMinus prev bunch empty", 50, -M_PI, M_PI);
        meMETPhiHFEGammasPlus_BXm1Empty =
            ibooker.book1D("METPhiHFEGammasPlus_BXm1Empty", "METPhi_PFHFEGammasPlus prev bunch empty", 50, -M_PI, M_PI);
        meMETPhiHFEGammasMinus_BXm1Empty = ibooker.book1D(
            "METPhiHFEGammasMinus_BXm1Empty", "METPhi_PFHFEGammasMinus prev bunch empty", 50, -M_PI, M_PI);

        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "METPhiChargedHadronsBarrel_BXm1Empty", meMETPhiChargedHadronsBarrel_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "METPhiChargedHadronsEndcapPlus_BXm1Empty", meMETPhiChargedHadronsEndcapPlus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "METPhiChargedHadronsEndcapMinus_BXm1Empty", meMETPhiChargedHadronsEndcapMinus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "METPhiNeutralHadronsBarrel_BXm1Empty", meMETPhiNeutralHadronsBarrel_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "METPhiNeutralHadronsEndcapPlus_BXm1Empty", meMETPhiNeutralHadronsEndcapPlus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
            DirName + "/" + "METPhiNeutralHadronsEndcapMinus_BXm1Empty", meMETPhiNeutralHadronsEndcapMinus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiPhotonsBarrel_BXm1Empty",
                                                                  meMETPhiPhotonsBarrel_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiPhotonsEndcapPlus_BXm1Empty",
                                                                  meMETPhiPhotonsEndcapPlus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiPhotonsEndcapMinus_BXm1Empty",
                                                                  meMETPhiPhotonsEndcapMinus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiHFHadronsPlus_BXm1Empty",
                                                                  meMETPhiHFHadronsPlus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiHFHadronsMinus_BXm1Empty",
                                                                  meMETPhiHFHadronsMinus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiHFEGammasPlus_BXm1Empty",
                                                                  meMETPhiHFEGammasPlus_BXm1Empty));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiHFEGammasMinus_BXm1Empty",
                                                                  meMETPhiHFEGammasMinus_BXm1Empty));

        meMETPhiChargedHadronsBarrel =
            ibooker.book1D("METPhiChargedHadronsBarrel", "METPhi_PFChargedHadronsBarrel", 50, -M_PI, M_PI);
        meMETPhiChargedHadronsEndcapPlus =
            ibooker.book1D("METPhiChargedHadronsEndcapPlus", "METPhi_PFChargedHadronsEndcapPlus", 50, -M_PI, M_PI);
        meMETPhiChargedHadronsEndcapMinus =
            ibooker.book1D("METPhiChargedHadronsEndcapMinus", "METPhi_PFChargedHadronsEndcapMinus", 50, -M_PI, M_PI);
        meMETPhiNeutralHadronsBarrel =
            ibooker.book1D("METPhiNeutralHadronsBarrel", "METPhi_PFNeutralHadronsBarrel", 50, -M_PI, M_PI);
        meMETPhiNeutralHadronsEndcapPlus =
            ibooker.book1D("METPhiNeutralHadronsEndcapPlus", "METPhi_PFNeutralHadronsEndcapPlus", 50, -M_PI, M_PI);
        meMETPhiNeutralHadronsEndcapMinus =
            ibooker.book1D("METPhiNeutralHadronsEndcapMinus", "METPhi_PFNeutralHadronsEndcapMinus", 50, -M_PI, M_PI);
        meMETPhiPhotonsBarrel = ibooker.book1D("METPhiPhotonsBarrel", "METPhi_PFPhotonsBarrel", 50, -M_PI, M_PI);
        meMETPhiPhotonsEndcapPlus =
            ibooker.book1D("METPhiPhotonsEndcapPlus", "METPhi_PFPhotonsEndcapPlus", 50, -M_PI, M_PI);
        meMETPhiPhotonsEndcapMinus =
            ibooker.book1D("METPhiPhotonsEndcapMinus", "METPhi_PFPhotonsEndcapMinus", 50, -M_PI, M_PI);
        meMETPhiHFHadronsPlus = ibooker.book1D("METPhiHFHadronsPlus", "METPhi_PFHFHadronsPlus", 50, -M_PI, M_PI);
        meMETPhiHFHadronsMinus = ibooker.book1D("METPhiHFHadronsMinus", "METPhi_PFHFHadronsMinus", 50, -M_PI, M_PI);
        meMETPhiHFEGammasPlus = ibooker.book1D("METPhiHFEGammasPlus", "METPhi_PFHFEGammasPlus", 50, -M_PI, M_PI);
        meMETPhiHFEGammasMinus = ibooker.book1D("METPhiHFEGammasMinus", "METPhi_PFHFEGammasMinus", 50, -M_PI, M_PI);

        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiChargedHadronsBarrel",
                                                                  meMETPhiChargedHadronsBarrel));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiChargedHadronsEndcapPlus",
                                                                  meMETPhiChargedHadronsEndcapPlus));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiChargedHadronsEndcapMinus",
                                                                  meMETPhiChargedHadronsEndcapMinus));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiNeutralHadronsBarrel",
                                                                  meMETPhiNeutralHadronsBarrel));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiNeutralHadronsEndcapPlus",
                                                                  meMETPhiNeutralHadronsEndcapPlus));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiNeutralHadronsEndcapMinus",
                                                                  meMETPhiNeutralHadronsEndcapMinus));
        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiPhotonsBarrel", meMETPhiPhotonsBarrel));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiPhotonsEndcapPlus",
                                                                  meMETPhiPhotonsEndcapPlus));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiPhotonsEndcapMinus",
                                                                  meMETPhiPhotonsEndcapMinus));
        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiHFHadronsPlus", meMETPhiHFHadronsPlus));
        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiHFHadronsMinus", meMETPhiHFHadronsMinus));
        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiHFEGammasPlus", meMETPhiHFEGammasPlus));
        map_of_MEs.insert(
            std::pair<std::string, MonitorElement*>(DirName + "/" + "METPhiHFEGammasMinus", meMETPhiHFEGammasMinus));
      }

      if (fillPFCandPlots && fillCandidateMap_histos) {
        if (!profilePFCand_x_.empty()) {
          etaMinPFCand_.clear();
          etaMaxPFCand_.clear();
          typePFCand_.clear();
          countsPFCand_.clear();
          MExPFCand_.clear();
          MEyPFCand_.clear();
          profilePFCand_x_.clear();
          profilePFCand_y_.clear();
          profilePFCand_x_name_.clear();
          profilePFCand_y_name_.clear();
        }
        for (std::vector<edm::ParameterSet>::const_iterator v = diagnosticsParameters_.begin();
             v != diagnosticsParameters_.end();
             v++) {
          double etaMinPFCand = v->getParameter<double>("etaMin");
          double etaMaxPFCand = v->getParameter<double>("etaMax");
          int nMinPFCand = v->getParameter<int>("nMin");
          int nMaxPFCand = v->getParameter<int>("nMax");
          int nbinsPFCand = v->getParameter<double>("nbins");

          // etaNBins_.push_back(etaNBins);
          etaMinPFCand_.push_back(etaMinPFCand);
          etaMaxPFCand_.push_back(etaMaxPFCand);
          typePFCand_.push_back(v->getParameter<int>("type"));
          countsPFCand_.push_back(0);
          MExPFCand_.push_back(0.);
          MEyPFCand_.push_back(0.);

          profilePFCand_x_.push_back(
              ibooker.bookProfile(std::string(v->getParameter<std::string>("name")).append("_Px_").c_str(),
                                  std::string(v->getParameter<std::string>("name")) + "Px",
                                  nbinsPFCand,
                                  nMinPFCand,
                                  nMaxPFCand,
                                  -300,
                                  300));
          profilePFCand_x_name_.push_back(std::string(v->getParameter<std::string>("name")).append("_Px_"));
          map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
              DirName + "/" + profilePFCand_x_name_[profilePFCand_x_name_.size() - 1],
              profilePFCand_x_[profilePFCand_x_.size() - 1]));
          profilePFCand_y_.push_back(
              ibooker.bookProfile(std::string(v->getParameter<std::string>("name")).append("_Py_").c_str(),
                                  std::string(v->getParameter<std::string>("name")) + "Py",
                                  nbinsPFCand,
                                  nMinPFCand,
                                  nMaxPFCand,
                                  -300,
                                  300));
          profilePFCand_y_name_.push_back(std::string(v->getParameter<std::string>("name")).append("_Py_"));
          map_of_MEs.insert(std::pair<std::string, MonitorElement*>(
              DirName + "/" + profilePFCand_y_name_[profilePFCand_y_name_.size() - 1],
              profilePFCand_y_[profilePFCand_y_.size() - 1]));
        }
      }
    }
    if (isMiniAODMet_) {
      if (fillPFCandPlots &&
          fillCandidateMap_histos) {  //first bool internal checks for subdirectory filling, second bool given in cfg file, checks that we fill maps only in one module in total
        if (!profilePFCand_x_.empty()) {
          etaMinPFCand_.clear();
          etaMaxPFCand_.clear();
          typePFCand_.clear();
          countsPFCand_.clear();
          profilePFCand_x_.clear();
          profilePFCand_y_.clear();
        }
        for (std::vector<edm::ParameterSet>::const_iterator v = diagnosticsParameters_.begin();
             v != diagnosticsParameters_.end();
             v++) {
          double etaMinPFCand = v->getParameter<double>("etaMin");
          double etaMaxPFCand = v->getParameter<double>("etaMax");

          etaMinPFCand_.push_back(etaMinPFCand);
          etaMaxPFCand_.push_back(etaMaxPFCand);
          typePFCand_.push_back(v->getParameter<int>("type"));
          countsPFCand_.push_back(0);
          MExPFCand_.push_back(0.);
          MEyPFCand_.push_back(0.);
        }
      }
    }

    if (isPFMet_ || isMiniAODMet_) {
      mePhotonEtFraction = ibooker.book1D("PfPhotonEtFraction", "photonEtFraction()", 50, 0, 1);
      meNeutralHadronEtFraction = ibooker.book1D("PfNeutralHadronEtFraction", "neutralHadronEtFraction()", 50, 0, 1);
      meChargedHadronEtFraction = ibooker.book1D("PfChargedHadronEtFraction", "chargedHadronEtFraction()", 50, 0, 1);
      meHFHadronEtFraction = ibooker.book1D("PfHFHadronEtFraction", "HFHadronEtFraction()", 50, 0, 1);
      meHFEMEtFraction = ibooker.book1D("PfHFEMEtFraction", "HFEMEtFraction()", 50, 0, 1);

      map_of_MEs.insert(
          std::pair<std::string, MonitorElement*>(DirName + "/" + "PfPhotonEtFraction", mePhotonEtFraction));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfNeutralHadronEtFraction",
                                                                meNeutralHadronEtFraction));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfChargedHadronEtFraction",
                                                                meChargedHadronEtFraction));
      map_of_MEs.insert(
          std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFHadronEtFraction", meHFHadronEtFraction));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFEMEtFraction", meHFEMEtFraction));

      mePhotonEtFraction_profile =
          ibooker.bookProfile("PfPhotonEtFraction_profile", "photonEtFraction()", nbinsPV_, nPVMin_, nPVMax_, 50, 0, 1);
      meNeutralHadronEtFraction_profile = ibooker.bookProfile(
          "PfNeutralHadronEtFraction_profile", "neutralHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_, 50, 0, 1);
      meChargedHadronEtFraction_profile = ibooker.bookProfile(
          "PfChargedHadronEtFraction_profile", "chargedHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_, 50, 0, 1);
      meHFHadronEtFraction_profile = ibooker.bookProfile(
          "PfHFHadronEtFraction_profile", "HFHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_, 50, 0, 1);
      meHFEMEtFraction_profile =
          ibooker.bookProfile("PfHFEMEtFraction_profile", "HFEMEtFraction()", nbinsPV_, nPVMin_, nPVMax_, 50, 0, 1);
      mePhotonEtFraction_profile->setAxisTitle("nvtx", 1);
      meNeutralHadronEtFraction_profile->setAxisTitle("nvtx", 1);
      meChargedHadronEtFraction_profile->setAxisTitle("nvtx", 1);
      meHFHadronEtFraction_profile->setAxisTitle("nvtx", 1);
      meHFEMEtFraction_profile->setAxisTitle("nvtx", 1);

      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfPhotonEtFraction_profile",
                                                                mePhotonEtFraction_profile));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfNeutralHadronEtFraction_profile",
                                                                meNeutralHadronEtFraction_profile));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfChargedHadronEtFraction_profile",
                                                                meChargedHadronEtFraction_profile));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFHadronEtFraction_profile",
                                                                meHFHadronEtFraction_profile));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFEMEtFraction_profile",
                                                                meHFEMEtFraction_profile));

      mePhotonEt = ibooker.book1D("PfPhotonEt", "photonEt()", 50, 0, 1000);
      meNeutralHadronEt = ibooker.book1D("PfNeutralHadronEt", "neutralHadronEt()", 50, 0, 1000);
      meElectronEt = ibooker.book1D("PfElectronEt", "electronEt()", 50, 0, 100);
      meChargedHadronEt = ibooker.book1D("PfChargedHadronEt", "chargedHadronEt()", 50, 0, 2000);
      meMuonEt = ibooker.book1D("PfMuonEt", "muonEt()", 50, 0, 100);
      meHFHadronEt = ibooker.book1D("PfHFHadronEt", "HFHadronEt()", 50, 0, 2000);
      meHFEMEt = ibooker.book1D("PfHFEMEt", "HFEMEt()", 50, 0, 1000);

      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfPhotonEt", mePhotonEt));
      map_of_MEs.insert(
          std::pair<std::string, MonitorElement*>(DirName + "/" + "PfNeutralHadronEt", meNeutralHadronEt));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfElectronEt", meElectronEt));
      map_of_MEs.insert(
          std::pair<std::string, MonitorElement*>(DirName + "/" + "PfChargedHadronEt", meChargedHadronEt));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfMuonEt", meMuonEt));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFHadronEt", meHFHadronEt));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFEMEt", meHFEMEt));

      mePhotonEt_profile =
          ibooker.bookProfile("PfPhotonEt_profile", "photonEt()", nbinsPV_, nPVMin_, nPVMax_, 50, 0, 1000);
      meNeutralHadronEt_profile = ibooker.bookProfile(
          "PfNeutralHadronEt_profile", "neutralHadronEt()", nbinsPV_, nPVMin_, nPVMax_, 50, 0, 1000);
      meChargedHadronEt_profile = ibooker.bookProfile(
          "PfChargedHadronEt_profile", "chargedHadronEt()", nbinsPV_, nPVMin_, nPVMax_, 50, 0, 1000);
      meHFHadronEt_profile =
          ibooker.bookProfile("PfHFHadronEt_profile", "HFHadronEt()", nbinsPV_, nPVMin_, nPVMax_, 50, 0, 1000);
      meHFEMEt_profile = ibooker.bookProfile("PfHFEMEt_profile", "HFEMEt()", nbinsPV_, nPVMin_, nPVMax_, 50, 0, 1000);

      mePhotonEt_profile->setAxisTitle("nvtx", 1);
      meNeutralHadronEt_profile->setAxisTitle("nvtx", 1);
      meChargedHadronEt_profile->setAxisTitle("nvtx", 1);
      meHFHadronEt_profile->setAxisTitle("nvtx", 1);
      meHFEMEt_profile->setAxisTitle("nvtx", 1);

      map_of_MEs.insert(
          std::pair<std::string, MonitorElement*>(DirName + "/" + "PfPhotonEt_profile", mePhotonEt_profile));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfNeutralHadronEt_profile",
                                                                meNeutralHadronEt_profile));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfChargedHadronEt_profile",
                                                                meChargedHadronEt_profile));
      map_of_MEs.insert(
          std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFHadronEt_profile", meHFHadronEt_profile));
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "PfHFEMEt_profile", meHFEMEt_profile));
    }

    if (isCaloMet_) {
      if (fill_met_high_level_histo) {  //now configurable in python file
        hMExLS = ibooker.book2D("MExLS", "MEx_LS", 200, -200, 200, 250, 0., 2500.);
        hMExLS->setAxisTitle("MEx [GeV]", 1);
        hMExLS->setAxisTitle("Lumi Section", 2);
        hMExLS->setOption("colz");
        hMEyLS = ibooker.book2D("MEyLS", "MEy_LS", 200, -200, 200, 250, 0., 2500.);
        hMEyLS->setAxisTitle("MEy [GeV]", 1);
        hMEyLS->setAxisTitle("Lumi Section", 2);
        hMEyLS->setOption("colz");
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MExLS", hMExLS));
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "MEyLS", hMEyLS));
      }
    }

    hMETRate = ibooker.book1D("METRate", "METRate", 200, 0, 1000);
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(DirName + "/" + "METRate", hMETRate));

    ibooker.setCurrentFolder("JetMET");
    lumisecME = ibooker.book1D("lumisec", "lumisec", 2501, -1., 2500.);
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>("JetMET/lumisec", lumisecME));
  }  //all non Z plots (restrict Z Plots only for resolution study)
}

// ***********************************************************
void METAnalyzer::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  const L1GtTriggerMenu* menu = &iSetup.getData(l1gtTrigMenuToken_);
  for (CItAlgo techTrig = menu->gtTechnicalTriggerMap().begin(); techTrig != menu->gtTechnicalTriggerMap().end();
       ++techTrig) {
    if ((techTrig->second).algoName() == m_l1algoname_) {
      m_bitAlgTechTrig_ = (techTrig->second).algoBitNumber();
      break;
    }
  }

  //  std::cout  << "Run " << iRun.run() << " hltconfig.init "
  //             << hltConfig_.init(iRun,iSetup,triggerResultsLabel_.process(),changed_) << " length: "<<hltConfig_.triggerNames().size()<<" changed "<<changed_<<std::endl;
  bool changed(true);
  if (hltConfig_.init(iRun, iSetup, triggerResultsLabel_.process(), changed)) {
    if (changed) {
      //hltConfig_.dump("ProcessName");
      //hltConfig_.dump("GlobalTag");
      //hltConfig_.dump("TableName");
      //      hltConfig_.dump("Streams");
      //      hltConfig_.dump("Datasets");
      //      hltConfig_.dump("PrescaleTable");
      //      hltConfig_.dump("ProcessPSet");
    }
  } else {
    if (verbose_)
      std::cout << "HLTEventAnalyzerAOD::analyze:"
                << " config extraction failure with process name " << triggerResultsLabel_.process() << std::endl;
  }

  allTriggerNames_.clear();
  for (unsigned int i = 0; i < hltConfig_.size(); i++) {
    allTriggerNames_.push_back(hltConfig_.triggerName(i));
  }
  //  std::cout<<"Length: "<<allTriggerNames_.size()<<std::endl;

  triggerSelectedSubFolders_ = parameters.getParameter<edm::VParameterSet>("triggerSelectedSubFolders");
  for (std::vector<GenericTriggerEventFlag*>::const_iterator it = triggerFolderEventFlag_.begin();
       it != triggerFolderEventFlag_.end();
       it++) {
    int pos = it - triggerFolderEventFlag_.begin();
    if ((*it)->on()) {
      (*it)->initRun(iRun, iSetup);
      if (triggerSelectedSubFolders_[pos].exists(std::string("hltDBKey"))) {
        if ((*it)->expressionsFromDB((*it)->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
          triggerFolderExpr_[pos] = (*it)->expressionsFromDB((*it)->hltDBKey(), iSetup);
      }
    }
  }
  if (isMiniAODMet_) {
    bool changed_filter = true;
    std::vector<int> initializeFilter(8, -1);  //we have 8 filters at the moment
    miniaodFilterIndex_ = initializeFilter;
    if (FilterhltConfig_.init(iRun, iSetup, METFilterMiniAODLabel_.process(), changed_filter)) {
      miniaodfilterdec = 0;
      for (unsigned int i = 0; i < FilterhltConfig_.size(); i++) {
        std::string search = FilterhltConfig_.triggerName(i).substr(
            5);  //actual label of filter, the first 5 items are Flag_, so stripped off
        std::string search2 =
            HBHENoiseStringMiniAOD;  //all filters end with DQM, which is not in the flag --> ONLY not for HBHEFilters
        std::size_t found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[0] = i;
        }
        search2 = CSCHaloResultTag_.label().substr(0, CSCHaloResultTag_.label().size() - 3);
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[1] = i;
        }
        search2 = eeBadScFilterTag_.label().substr(0, eeBadScFilterTag_.label().size() - 3);
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[2] = i;
        }
        search2 = HBHEIsoNoiseStringMiniAOD;
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[3] = i;
        }
        search2 = CSCHalo2015ResultTag_.label().substr(0, CSCHalo2015ResultTag_.label().size() - 3);
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[4] = i;
        }
        search2 = EcalDeadCellTriggerTag_.label().substr(0, EcalDeadCellTriggerTag_.label().size() - 3);
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[5] = i;
        }
        search2 = EcalDeadCellBoundaryTag_.label().substr(0, EcalDeadCellBoundaryTag_.label().size() - 3);
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[6] = i;
        }
        search2 = HcalStripHaloTag_.label().substr(0, HcalStripHaloTag_.label().size() - 3);
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[7] = i;
        }
      }
    } else if (FilterhltConfig_.init(iRun, iSetup, METFilterMiniAODLabel2_.process(), changed_filter)) {
      miniaodfilterdec = 1;
      for (unsigned int i = 0; i < FilterhltConfig_.size(); i++) {
        std::string search = FilterhltConfig_.triggerName(i).substr(
            5);  //actual label of filter, the first 5 items are Flag_, so stripped off
        std::string search2 =
            HBHENoiseStringMiniAOD;  //all filters end with DQM, which is not in the flag --> ONLY not for HBHEFilters
        std::size_t found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[0] = i;
        }
        search2 = CSCHaloResultTag_.label().substr(0, CSCHaloResultTag_.label().size() - 3);
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[1] = i;
        }
        search2 = eeBadScFilterTag_.label().substr(0, eeBadScFilterTag_.label().size() - 3);
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[2] = i;
        }
        search2 = HBHEIsoNoiseStringMiniAOD;
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[3] = i;
        }
        search2 = CSCHalo2015ResultTag_.label().substr(0, CSCHalo2015ResultTag_.label().size() - 3);
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[4] = i;
        }
        search2 = EcalDeadCellTriggerTag_.label().substr(0, EcalDeadCellTriggerTag_.label().size() - 3);
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[5] = i;
        }
        search2 = EcalDeadCellBoundaryTag_.label().substr(0, EcalDeadCellBoundaryTag_.label().size() - 3);
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[6] = i;
        }
        search2 = HcalStripHaloTag_.label().substr(0, HcalStripHaloTag_.label().size() - 3);
        found = search2.find(search);
        if (found != std::string::npos) {
          miniaodFilterIndex_[7] = i;
        }
      }
    } else {
      edm::LogWarning("MiniAOD METAN Filter HLT OBject version")
          << "nothing found with both RECO and reRECO label" << std::endl;
    }
  }
}

// ***********************************************************
void METAnalyzer::dqmEndRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  //
  //--- Check the time length of the Run from the lumi section plots

  TH1F* tlumisec;

  MonitorElement* meLumiSec = map_dijet_MEs["aaa"];
  meLumiSec = map_dijet_MEs["JetMET/lumisec"];

  int totlsec = 0;
  int totlssecsum = 0;
  double totltime = 0.;
  if (meLumiSec && meLumiSec->getRootObject()) {
    tlumisec = meLumiSec->getTH1F();
    //check overflow bin (if we have more than 2500 LS in a run)
    //lumisec is filled every time the analyze section is processed
    //we know an LS is present only once in a run: normalize how many events we had on average
    //if lumi fluctuates strongly might be unreliable for overflow bin though
    for (int i = 0; i < (tlumisec->GetNbinsX()); i++) {
      if (tlumisec->GetBinContent(i) != 0) {
        totlsec += 1;
        totlssecsum += tlumisec->GetBinContent(i);
      }
    }
    int num_per_ls = (double)totlssecsum / (double)totlsec;
    totlsec = totlsec + tlumisec->GetBinContent(tlumisec->GetNbinsX() + 1) / (double)num_per_ls;
    totltime = double(totlsec * 90);  // one lumi sec ~ 90 (sec)
  }

  if (totltime == 0.)
    totltime = 1.;

  std::string dirName = FolderName_ + metCollectionLabel_.label() + "/";
  //dbe_->setCurrentFolder(dirName);

  //below is the original METAnalyzer formulation

  for (std::vector<std::string>::const_iterator ic = folderNames_.begin(); ic != folderNames_.end(); ic++) {
    std::string DirName;
    DirName = dirName + *ic;
    makeRatePlot(DirName, totltime);
    for (std::vector<GenericTriggerEventFlag*>::const_iterator it = triggerFolderEventFlag_.begin();
         it != triggerFolderEventFlag_.end();
         it++) {
      int pos = it - triggerFolderEventFlag_.begin();
      if ((*it)->on()) {
        makeRatePlot(DirName + "/" + triggerFolderLabels_[pos], totltime);
      }
    }
  }
}

// ***********************************************************
void METAnalyzer::makeRatePlot(std::string DirName, double totltime) {
  //dbe_->setCurrentFolder(DirName);
  MonitorElement* meMET = map_dijet_MEs[DirName + "/" + "MET"];
  MonitorElement* mMETRate = map_dijet_MEs[DirName + "/" + "METRate"];

  TH1F* tMET;
  TH1F* tMETRate;

  if (meMET && mMETRate) {
    if (meMET->getRootObject() && mMETRate->getRootObject()) {
      tMET = meMET->getTH1F();

      // Integral plot & convert number of events to rate (hz)
      tMETRate = (TH1F*)tMET->Clone("METRateHist");
      for (int i = tMETRate->GetNbinsX() - 1; i >= 0; i--) {
        tMETRate->SetBinContent(i + 1, tMETRate->GetBinContent(i + 2) + tMET->GetBinContent(i + 1));
      }
      for (int i = 0; i < tMETRate->GetNbinsX(); i++) {
        tMETRate->SetBinContent(i + 1, tMETRate->GetBinContent(i + 1) / double(totltime));
        mMETRate->setBinContent(i + 1, tMETRate->GetBinContent(i + 1));
      }
    }
  }
}

// ***********************************************************
void METAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // *** Fill lumisection ME
  int myLuminosityBlock;
  myLuminosityBlock = iEvent.luminosityBlock();
  if (fill_met_high_level_histo) {
    lumisecME = map_dijet_MEs["JetMET/lumisec"];
    if (lumisecME && lumisecME->getRootObject())
      lumisecME->Fill(myLuminosityBlock);
  }

  if (myLuminosityBlock < LSBegin_)
    return;
  if (myLuminosityBlock > LSEnd_ && LSEnd_ > 0)
    return;

  if (verbose_)
    std::cout << "METAnalyzer analyze" << std::endl;

  std::string DirName = FolderName_ + metCollectionLabel_.label();

  // ==========================================================
  // Trigger information
  //
  //  trigJetMB_=0;
  //  trigHighPtJet_=0;
  //  trigLowPtJet_=0;
  //  trigMinBias_=0;
  //  trigHighMET_=0;
  //  //  _trig_LowMET=0;
  //  trigEle_=0;
  //  trigMuon_=0;
  //  trigPhysDec_=0;
  std::vector<int> triggerFolderDecisions;
  triggerFolderDecisions_ = std::vector<int>(triggerFolderEventFlag_.size(), 0);
  // **** Get the TriggerResults container
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);

  if (triggerResults.isValid()) {
    /////////// Analyzing HLT Trigger Results (TriggerResults) //////////
    // Check how many HLT triggers are in triggerResults
    int ntrigs = (*triggerResults).size();
    if (verbose_)
      std::cout << "ntrigs=" << ntrigs << std::endl;
    // If index=ntrigs, this HLT trigger doesn't exist in the HLT table for this data.
    for (std::vector<GenericTriggerEventFlag*>::const_iterator it = triggerFolderEventFlag_.begin();
         it != triggerFolderEventFlag_.end();
         it++) {
      unsigned int pos = it - triggerFolderEventFlag_.begin();
      bool fd = (*it)->accept(iEvent, iSetup);
      triggerFolderDecisions_[pos] = fd;
    }
    allTriggerDecisions_.clear();
    for (unsigned int i = 0; i < allTriggerNames_.size(); ++i) {
      allTriggerDecisions_.push_back((*triggerResults).accept(i));
      //std::cout<<"TR "<<(*triggerResults).size()<<" "<<(*triggerResults).accept(i)<<" "<<allTriggerNames_[i]<<std::endl;
    }
  }

  // ==========================================================
  // MET information

  // **** Get the MET container
  edm::Handle<reco::METCollection> tcmetcoll;
  edm::Handle<reco::CaloMETCollection> calometcoll;
  edm::Handle<reco::PFMETCollection> pfmetcoll;
  edm::Handle<pat::METCollection> patmetcoll;

  //if(isTCMet_){
  //iEvent.getByToken(tcMetToken_, tcmetcoll);
  //if(!tcmetcoll.isValid()) return;
  //}
  if (isCaloMet_) {
    iEvent.getByToken(caloMetToken_, calometcoll);
    if (!calometcoll.isValid())
      return;
  }
  if (isPFMet_) {
    iEvent.getByToken(pfMetToken_, pfmetcoll);
    if (!pfmetcoll.isValid())
      return;
  }
  if (isMiniAODMet_) {
    iEvent.getByToken(patMetToken_, patmetcoll);
    if (!patmetcoll.isValid())
      return;
  }

  const MET* met = nullptr;
  const pat::MET* patmet = nullptr;
  const PFMET* pfmet = nullptr;
  const CaloMET* calomet = nullptr;
  //if(isTCMet_){
  //met=&(tcmetcoll->front());
  //}
  if (isPFMet_) {
    assert(!pfmetcoll->empty());
    met = &(pfmetcoll->front());
    pfmet = &(pfmetcoll->front());
  }
  if (isCaloMet_) {
    assert(!calometcoll->empty());
    met = &(calometcoll->front());
    calomet = &(calometcoll->front());
  }
  if (isMiniAODMet_) {
    assert(!patmetcoll->empty());
    met = &(patmetcoll->front());
    patmet = &(patmetcoll->front());
  }

  LogTrace("METAnalyzer") << "[METAnalyzer] Call to the MET analyzer";

  // ==========================================================
  // TCMET

  //if (/*isTCMet_ || */(isCaloMet_ && metCollectionLabel_.label() == "corMetGlobalMuons")) {

  //iEvent.getByToken(MuonToken_, muonHandle_);
  //iEvent.getByToken(TrackToken_, trackHandle_);
  //iEvent.getByToken(ElectronToken_, electronHandle_);
  //iEvent.getByToken(BeamspotToken_, beamSpotHandle_);
  //iEvent.getByToken(tcMETValueMapToken_,tcMetValueMapHandle_);

  //if(!muonHandle_.isValid())     edm::LogInfo("OutputInfo") << "falied to retrieve muon data require by MET Task";
  //if(!trackHandle_.isValid())    edm::LogInfo("OutputInfo") << "falied to retrieve track data require by MET Task";
  //if(!electronHandle_.isValid()) edm::LogInfo("OutputInfo") << "falied to retrieve electron data require by MET Task";
  //if(!beamSpotHandle_.isValid()) edm::LogInfo("OutputInfo") << "falied to retrieve beam spot data require by MET Task";

  //beamSpot_ = ( beamSpotHandle_.isValid() ) ? beamSpotHandle_->position() : math::XYZPoint(0, 0, 0);
  //}

  // ==========================================================
  bool bJetID = false;
  bool bDiJetID = false;
  // Jet ID -------------------------------------------------------
  //

  edm::Handle<CaloJetCollection> caloJets;
  edm::Handle<JPTJetCollection> jptJets;
  edm::Handle<PFJetCollection> pfJets;
  edm::Handle<pat::JetCollection> patJets;

  int collsize = -1;

  if (isCaloMet_) {
    iEvent.getByToken(caloJetsToken_, caloJets);
    if (!caloJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find calojet product" << std::endl;
      if (verbose_)
        std::cout << "METAnalyzer: Could not find calojet product" << std::endl;
    }
    collsize = caloJets->size();
  }
  ///*
  //if (isTCMet_){
  //iEvent.getByToken(jptJetsToken_, jptJets);
  //if (!jptJets.isValid()) {
  //  LogDebug("") << "METAnalyzer: Could not find jptjet product" << std::endl;
  //  if (verbose_) std::cout << "METAnalyzer: Could not find jptjet product" << std::endl;
  //}
  //collsize=jptJets->size();
  //}*/

  edm::Handle<edm::ValueMap<reco::JetID> > jetID_ValueMap_Handle;
  if (/*isTCMet_ || */ isCaloMet_) {
    if (!runcosmics_) {
      iEvent.getByToken(jetID_ValueMapToken_, jetID_ValueMap_Handle);
    }
  }

  if (isMiniAODMet_) {
    iEvent.getByToken(patJetsToken_, patJets);
    if (!patJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find patjet product" << std::endl;
      if (verbose_)
        std::cout << "METAnalyzer: Could not find patjet product" << std::endl;
    }
    collsize = patJets->size();
  }

  if (isPFMet_) {
    iEvent.getByToken(pfJetsToken_, pfJets);
    if (!pfJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find pfjet product" << std::endl;
      if (verbose_)
        std::cout << "METAnalyzer: Could not find pfjet product" << std::endl;
    }
    collsize = pfJets->size();
  }

  unsigned int ind1 = -1;
  double pt1 = -1;
  bool pass_jetID1 = false;
  unsigned int ind2 = -1;
  double pt2 = -1;
  bool pass_jetID2 = false;

  edm::Handle<reco::JetCorrector> jetCorr;
  bool pass_correction_flag = false;
  if (!isMiniAODMet_) {
    iEvent.getByToken(jetCorrectorToken_, jetCorr);
    if (jetCorr.isValid()) {
      pass_correction_flag = true;
    }
  } else {
    pass_correction_flag = true;
  }
  //do loose jet ID-> check threshold on corrected jets
  for (int ijet = 0; ijet < collsize; ijet++) {
    double pt_jet = -10;
    double scale = 1.;
    bool iscleaned = false;
    if (pass_correction_flag) {
      if (isCaloMet_) {
        scale = jetCorr->correction((*caloJets)[ijet]);
      }
      //if(isTCMet_){
      //scale = jetCorr->correction((*jptJets)[ijet]);
      //}
      if (isPFMet_) {
        scale = jetCorr->correction((*pfJets)[ijet]);
      }
    }
    if (isCaloMet_) {
      pt_jet = scale * (*caloJets)[ijet].pt();
      if (pt_jet > ptThreshold_) {
        reco::CaloJetRef calojetref(caloJets, ijet);
        if (!runcosmics_) {
          reco::JetID jetID = (*jetID_ValueMap_Handle)[calojetref];
          iscleaned = jetIDFunctorLoose((*caloJets)[ijet], jetID);
        } else {
          iscleaned = true;
        }
      }
    }
    ///*
    //if(isTCMet_){
    //pt_jet=scale*(*jptJets)[ijet].pt();
    //if(pt_jet> ptThreshold_){
    //	const edm::RefToBase<reco::Jet>&  rawJet = (*jptJets)[ijet].getCaloJetRef();
    //	const reco::CaloJet *rawCaloJet = dynamic_cast<const reco::CaloJet*>(&*rawJet);
    //	reco::CaloJetRef const theCaloJetRef = (rawJet).castTo<reco::CaloJetRef>();
    //	if(!runcosmics_){
    //	  reco::JetID jetID = (*jetID_ValueMap_Handle)[theCaloJetRef];
    //	  iscleaned = jetIDFunctorLoose(*rawCaloJet, jetID);
    //	}else{
    //	  iscleaned=true;
    //	}
    //}
    //}*/
    if (isPFMet_) {
      pt_jet = scale * (*pfJets)[ijet].pt();
      if (pt_jet > ptThreshold_) {
        iscleaned = pfjetIDFunctorLoose((*pfJets)[ijet]);
      }
    }
    if (isMiniAODMet_) {
      pt_jet = (*patJets)[ijet].pt();
      if (pt_jet > ptThreshold_) {
        pat::strbitset stringbitset = pfjetIDFunctorLoose.getBitTemplate();
        iscleaned = pfjetIDFunctorLoose((*patJets)[ijet], stringbitset);
      }
    }
    if (iscleaned) {
      bJetID = true;
    }
    if (pt_jet > pt1) {
      pt2 = pt1;
      ind2 = ind1;
      pass_jetID2 = pass_jetID1;
      pt1 = pt_jet;
      ind1 = ijet;
      pass_jetID1 = iscleaned;
    } else if (pt_jet > pt2) {
      pt2 = pt_jet;
      ind2 = ijet;
      pass_jetID2 = iscleaned;
    }
  }
  if (pass_jetID1 && pass_jetID2) {
    double dphi = -1.0;
    if (isCaloMet_) {
      dphi = fabs((*caloJets)[ind1].phi() - (*caloJets)[ind2].phi());
    }
    ///* if(isTCMet_){
    //dphi=fabs((*jptJets)[ind1].phi()-(*jptJets)[ind2].phi());
    //}*/
    if (isPFMet_) {
      dphi = fabs((*pfJets)[ind1].phi() - (*pfJets)[ind2].phi());
    }
    if (isMiniAODMet_) {
      dphi = fabs((*patJets)[0].phi() - (*patJets)[1].phi());
    }
    if (dphi > acos(-1.)) {
      dphi = 2 * acos(-1.) - dphi;
    }
    if (dphi > 2.7) {
      bDiJetID = true;
    }
  }
  // ==========================================================
  // ==========================================================
  //Vertex information
  Handle<VertexCollection> vertexHandle;
  iEvent.getByToken(vertexToken_, vertexHandle);

  if (!vertexHandle.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find vertex collection" << std::endl;
    if (verbose_)
      std::cout << "CaloMETAnalyzer: Could not find vertex collection" << std::endl;
  }
  numPV_ = 0;
  if (vertexHandle.isValid()) {
    VertexCollection vertexCollection = *(vertexHandle.product());
    numPV_ = vertexCollection.size();
  }
  bool bPrimaryVertex = (bypassAllPVChecks_ || (numPV_ > 0));

  bool bZJets = false;

  edm::Handle<MuonCollection> Muons;
  iEvent.getByToken(MuonsToken_, Muons);

  reco::Candidate::PolarLorentzVector zCand;

  double pt_muon0 = -1;
  double pt_muon1 = -1;
  int mu_index0 = -1;
  int mu_index1 = -1;
  //fill it only for cleaned jets
  if (Muons.isValid() && Muons->size() > 1) {
    for (unsigned int i = 0; i < Muons->size(); i++) {
      bool pass_muon_id = false;
      bool pass_muon_iso = false;
      double dxy = fabs((*Muons)[i].muonBestTrack()->dxy());
      double dz = fabs((*Muons)[i].muonBestTrack()->dz());
      if (numPV_ > 0) {
        dxy = fabs((*Muons)[i].muonBestTrack()->dxy((*vertexHandle)[0].position()));
        dz = fabs((*Muons)[i].muonBestTrack()->dz((*vertexHandle)[0].position()));
      }
      if ((*Muons)[i].pt() > 20 && fabs((*Muons)[i].eta()) < 2.3) {
        if ((*Muons)[i].isGlobalMuon() && (*Muons)[i].isPFMuon() &&
            (*Muons)[i].globalTrack()->hitPattern().numberOfValidMuonHits() > 0 &&
            (*Muons)[i].numberOfMatchedStations() > 1 && dxy < 0.2 && (*Muons)[i].numberOfMatchedStations() > 1 &&
            dz < 0.5 && (*Muons)[i].innerTrack()->hitPattern().numberOfValidPixelHits() > 0 &&
            (*Muons)[i].innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5) {
          pass_muon_id = true;
        }
        // Muon pf isolation DB corrected
        float muonIsoPFdb =
            ((*Muons)[i].pfIsolationR04().sumChargedHadronPt +
             std::max(0.,
                      (*Muons)[i].pfIsolationR04().sumNeutralHadronEt + (*Muons)[i].pfIsolationR04().sumPhotonEt -
                          0.5 * (*Muons)[i].pfIsolationR04().sumPUPt)) /
            (*Muons)[i].pt();
        if (muonIsoPFdb < 0.12) {
          pass_muon_iso = true;
        }

        if (pass_muon_id && pass_muon_iso) {
          if ((*Muons)[i].pt() > pt_muon0) {
            mu_index1 = mu_index0;
            pt_muon1 = pt_muon0;
            mu_index0 = i;
            pt_muon0 = (*Muons)[i].pt();
          } else if ((*Muons)[i].pt() > pt_muon1) {
            mu_index1 = i;
            pt_muon1 = (*Muons)[i].pt();
          }
        }
      }
    }
    if (mu_index0 >= 0 && mu_index1 >= 0) {
      if ((*Muons)[mu_index0].charge() * (*Muons)[mu_index1].charge() < 0) {
        zCand = (*Muons)[mu_index0].polarP4() + (*Muons)[mu_index1].polarP4();
        if (fabs(zCand.M() - 91.) < 20) {
          bZJets = true;
        }
      }
    }
  }

  // ==========================================================

  edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
  iEvent.getByToken(gtToken_, gtReadoutRecord);

  bool techTriggerResultBxM2 = false;
  bool techTriggerResultBxM1 = false;
  bool techTriggerResultBx0 = false;

  if (!gtReadoutRecord.isValid()) {
    LogDebug("") << "METAnalyzer: Could not find GT readout record" << std::endl;
    if (verbose_)
      std::cout << "METAnalyzer: Could not find GT readout record product" << std::endl;
  } else {
    // trigger results before mask for BxInEvent -2 (E), -1 (F), 0 (L1A), 1, 2
    const TechnicalTriggerWord& technicalTriggerWordBeforeMaskBxM2 = gtReadoutRecord->technicalTriggerWord(-2);
    const TechnicalTriggerWord& technicalTriggerWordBeforeMaskBxM1 = gtReadoutRecord->technicalTriggerWord(-1);
    const TechnicalTriggerWord& technicalTriggerWordBeforeMaskBx0 = gtReadoutRecord->technicalTriggerWord();
    //const TechnicalTriggerWord&  technicalTriggerWordBeforeMaskBxG = gtReadoutRecord->technicalTriggerWord(1);
    //const TechnicalTriggerWord&  technicalTriggerWordBeforeMaskBxH = gtReadoutRecord->technicalTriggerWord(2);
    if (m_bitAlgTechTrig_ > -1 && !technicalTriggerWordBeforeMaskBx0.empty()) {
      techTriggerResultBx0 = technicalTriggerWordBeforeMaskBx0.at(m_bitAlgTechTrig_);
      if (techTriggerResultBx0 != 0) {
        techTriggerResultBxM2 = technicalTriggerWordBeforeMaskBxM2.at(m_bitAlgTechTrig_);
        techTriggerResultBxM1 = technicalTriggerWordBeforeMaskBxM1.at(m_bitAlgTechTrig_);
      }
    }
  }

  std::vector<bool> trigger_flag(4, false);
  if (techTriggerResultBx0 && techTriggerResultBxM2 &&
      techTriggerResultBxM1) {  //current and previous two bunches filled
    trigger_flag[0] = true;
  }
  if (techTriggerResultBx0 && techTriggerResultBxM1) {  //current and previous bunch filled
    trigger_flag[1] = true;
  }
  if (techTriggerResultBx0 && !techTriggerResultBxM1) {  //current bunch filled, but  previous bunch emtpy
    trigger_flag[2] = true;
  }
  if (techTriggerResultBx0 && !techTriggerResultBxM2 &&
      !techTriggerResultBxM1) {  //current bunch filled, but previous two bunches emtpy
    trigger_flag[3] = true;
  }
  std::vector<bool> filter_decisions(
      8, false);  //include all recommended filters, old filters in MiniAOD, and 2 new filters in testing phase
  if (!isMiniAODMet_ &&
      !runcosmics_) {  //not checked for MiniAOD -> for miniaod decision filled as "triggerResults" bool
    edm::Handle<bool> HBHENoiseFilterResultHandle;
    iEvent.getByToken(hbheNoiseFilterResultToken_, HBHENoiseFilterResultHandle);
    if (!HBHENoiseFilterResultHandle.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
      if (verbose_)
        std::cout << "METAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
    }
    filter_decisions[0] = *HBHENoiseFilterResultHandle;
    edm::Handle<bool> CSCTightHaloFilterResultHandle;
    iEvent.getByToken(CSCHaloResultToken_, CSCTightHaloFilterResultHandle);
    if (!CSCTightHaloFilterResultHandle.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find CSCTightHaloFilterResultHandle" << std::endl;
      if (verbose_)
        std::cout << "METAnalyzer: CSCTightHaloFilterResultHandle" << std::endl;
    }
    filter_decisions[1] = *CSCTightHaloFilterResultHandle;
    edm::Handle<bool> eeBadScFilterResultHandle;
    iEvent.getByToken(eeBadScFilterToken_, eeBadScFilterResultHandle);
    if (!eeBadScFilterResultHandle.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find eeBadScFilterResultHandle" << std::endl;
      if (verbose_)
        std::cout << "METAnalyzer: eeBadScFilterResultHandle" << std::endl;
    }
    filter_decisions[2] = *eeBadScFilterResultHandle;
    edm::Handle<bool> HBHENoiseIsoFilterResultHandle;
    iEvent.getByToken(hbheIsoNoiseFilterResultToken_, HBHENoiseIsoFilterResultHandle);
    if (!HBHENoiseIsoFilterResultHandle.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find HBHENoiseIsoFilterResult" << std::endl;
      if (verbose_)
        std::cout << "METAnalyzer: Could not find HBHENoiseIsoFilterResult" << std::endl;
    }
    filter_decisions[3] = *HBHENoiseIsoFilterResultHandle;
    edm::Handle<bool> CSCTightHalo2015FilterResultHandle;
    iEvent.getByToken(CSCHalo2015ResultToken_, CSCTightHalo2015FilterResultHandle);
    if (!CSCTightHalo2015FilterResultHandle.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find CSCTightHalo2015FilterResultHandle" << std::endl;
      if (verbose_)
        std::cout << "METAnalyzer: CSCTightHalo2015FilterResultHandle" << std::endl;
    }
    filter_decisions[4] = *CSCTightHalo2015FilterResultHandle;
    edm::Handle<bool> EcalDeadCellTriggerFilterResultHandle;
    iEvent.getByToken(EcalDeadCellTriggerToken_, EcalDeadCellTriggerFilterResultHandle);
    if (!EcalDeadCellTriggerFilterResultHandle.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find EcalDeadCellTriggerFilterResultHandle" << std::endl;
      if (verbose_)
        std::cout << "METAnalyzer: EcalDeadCellTriggerFilterResultHandle" << std::endl;
    }
    filter_decisions[5] = *EcalDeadCellTriggerFilterResultHandle;
    edm::Handle<bool> EcalDeadCellBoundaryHandle;
    iEvent.getByToken(EcalDeadCellBoundaryToken_, EcalDeadCellBoundaryHandle);
    if (!EcalDeadCellBoundaryHandle.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find EcalDeadCellBoundaryHandle" << std::endl;
      if (verbose_)
        std::cout << "METAnalyzer: EcalDeadCellBoundaryHandle" << std::endl;
    }
    filter_decisions[6] = *EcalDeadCellBoundaryHandle;
    edm::Handle<bool> HcalStripHaloFilterHandle;
    iEvent.getByToken(HcalStripHaloToken_, HcalStripHaloFilterHandle);
    if (!HcalStripHaloFilterHandle.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find CSCTightHalo2015FilterResultHandle" << std::endl;
      if (verbose_)
        std::cout << "METAnalyzer: CSCTightHalo2015FilterResultHandle" << std::endl;
    }
    filter_decisions[7] = *HcalStripHaloFilterHandle;
  } else if (isMiniAODMet_) {
    //miniaodFilterIndex_ is only filled in dqmBeginRun if isMiniAODMet_ true
    edm::Handle<edm::TriggerResults> metFilterResults;
    iEvent.getByToken(METFilterMiniAODToken_, metFilterResults);
    if (metFilterResults.isValid()) {
      if (miniaodFilterIndex_[0] != -1) {
        filter_decisions[0] = metFilterResults->accept(miniaodFilterIndex_[0]);
      }
      if (miniaodFilterIndex_[1] != -1) {
        filter_decisions[1] = metFilterResults->accept(miniaodFilterIndex_[1]);
      }
      if (miniaodFilterIndex_[2] != -1) {
        filter_decisions[2] = metFilterResults->accept(miniaodFilterIndex_[2]);
      }
      if (miniaodFilterIndex_[3] != -1) {
        filter_decisions[3] = metFilterResults->accept(miniaodFilterIndex_[3]);
      }
      if (miniaodFilterIndex_[4] != -1) {
        filter_decisions[4] = metFilterResults->accept(miniaodFilterIndex_[4]);
      }
      if (miniaodFilterIndex_[5] != -1) {
        filter_decisions[5] = metFilterResults->accept(miniaodFilterIndex_[5]);
      }
      if (miniaodFilterIndex_[6] != -1) {
        filter_decisions[6] = metFilterResults->accept(miniaodFilterIndex_[6]);
      }
      if (miniaodFilterIndex_[7] != -1) {
        filter_decisions[7] = metFilterResults->accept(miniaodFilterIndex_[7]);
      }
    } else {
      iEvent.getByToken(METFilterMiniAODToken2_, metFilterResults);
      if (metFilterResults.isValid()) {
        if (miniaodFilterIndex_[0] != -1) {
          filter_decisions[0] = metFilterResults->accept(miniaodFilterIndex_[0]);
        }
        if (miniaodFilterIndex_[1] != -1) {
          filter_decisions[1] = metFilterResults->accept(miniaodFilterIndex_[1]);
        }
        if (miniaodFilterIndex_[2] != -1) {
          filter_decisions[2] = metFilterResults->accept(miniaodFilterIndex_[2]);
        }
        if (miniaodFilterIndex_[3] != -1) {
          filter_decisions[3] = metFilterResults->accept(miniaodFilterIndex_[3]);
        }
        if (miniaodFilterIndex_[4] != -1) {
          filter_decisions[4] = metFilterResults->accept(miniaodFilterIndex_[4]);
        }
        if (miniaodFilterIndex_[5] != -1) {
          filter_decisions[5] = metFilterResults->accept(miniaodFilterIndex_[5]);
        }
        if (miniaodFilterIndex_[6] != -1) {
          filter_decisions[6] = metFilterResults->accept(miniaodFilterIndex_[6]);
        }
        if (miniaodFilterIndex_[7] != -1) {
          filter_decisions[7] = metFilterResults->accept(miniaodFilterIndex_[7]);
        }
      }
    }
  }
  bool HBHENoiseFilterResultFlag = filter_decisions[0];  //setup for RECO and MINIAOD
  // ==========================================================
  // HCAL Noise filter
  bool bHBHENoiseFilter = HBHENoiseFilterResultFlag;

  // DCS Filter
  bool bDCSFilter = (bypassAllDCSChecks_ || DCSFilter_->filter(iEvent, iSetup));
  // ==========================================================
  // Reconstructed MET Information - fill MonitorElements
  std::string DirName_old = DirName;
  for (std::vector<std::string>::const_iterator ic = folderNames_.begin(); ic != folderNames_.end(); ic++) {
    bool pass_selection = false;
    if ((*ic == "Uncleaned") && (isCaloMet_ || bPrimaryVertex)) {
      fillMESet(iEvent,
                DirName_old + "/" + *ic,
                *met,
                patmet,
                pfmet,
                calomet,
                zCand,
                map_dijet_MEs,
                trigger_flag,
                filter_decisions);
      pass_selection = true;
    }
    //take two lines out for first check
    if ((*ic == "Cleaned") && bDCSFilter && bHBHENoiseFilter && bPrimaryVertex && bJetID) {
      fillMESet(iEvent,
                DirName_old + "/" + *ic,
                *met,
                patmet,
                pfmet,
                calomet,
                zCand,
                map_dijet_MEs,
                trigger_flag,
                filter_decisions);
      pass_selection = true;
    }
    if ((*ic == "DiJet") && bDCSFilter && bHBHENoiseFilter && bPrimaryVertex && bDiJetID) {
      fillMESet(iEvent,
                DirName_old + "/" + *ic,
                *met,
                patmet,
                pfmet,
                calomet,
                zCand,
                map_dijet_MEs,
                trigger_flag,
                filter_decisions);
      pass_selection = true;
    }
    if ((*ic == "ZJets") && bDCSFilter && bHBHENoiseFilter && bPrimaryVertex && bZJets) {
      fillMESet(iEvent,
                DirName_old + "/" + *ic,
                *met,
                patmet,
                pfmet,
                calomet,
                zCand,
                map_dijet_MEs,
                trigger_flag,
                filter_decisions);
      pass_selection = true;
    }
    if (pass_selection && isPFMet_) {
      DirName = DirName_old + "/" + *ic;
    }
  }
}

// ***********************************************************
void METAnalyzer::fillMESet(const edm::Event& iEvent,
                            std::string DirName,
                            const reco::MET& met,
                            const pat::MET* patmet,
                            const reco::PFMET* pfmet,
                            const reco::CaloMET* calomet,
                            const reco::Candidate::PolarLorentzVector& zCand,
                            std::map<std::string, MonitorElement*>& map_of_MEs,
                            std::vector<bool> techTriggerCase,
                            std::vector<bool> METFilterDecision) {
  bool bLumiSecPlot = fill_met_high_level_histo;
  bool fillPFCandidatePlots = false;
  if (DirName.find("Cleaned") != std::string::npos) {
    fillPFCandidatePlots = true;
    fillMonitorElement(iEvent,
                       DirName,
                       std::string(""),
                       met,
                       patmet,
                       pfmet,
                       calomet,
                       zCand,
                       map_of_MEs,
                       bLumiSecPlot,
                       fillPFCandidatePlots,
                       techTriggerCase,
                       METFilterDecision);
    for (unsigned int i = 0; i < triggerFolderLabels_.size(); i++) {
      fillPFCandidatePlots = false;
      if (triggerFolderDecisions_[i]) {
        fillMonitorElement(iEvent,
                           DirName,
                           triggerFolderLabels_[i],
                           met,
                           patmet,
                           pfmet,
                           calomet,
                           zCand,
                           map_of_MEs,
                           bLumiSecPlot,
                           fillPFCandidatePlots,
                           techTriggerCase,
                           METFilterDecision);
      }
    }
  } else if (DirName.find("DiJet") != std::string::npos) {
    fillMonitorElement(iEvent,
                       DirName,
                       std::string(""),
                       met,
                       patmet,
                       pfmet,
                       calomet,
                       zCand,
                       map_of_MEs,
                       bLumiSecPlot,
                       fillPFCandidatePlots,
                       techTriggerCase,
                       METFilterDecision);
    for (unsigned int i = 0; i < triggerFolderLabels_.size(); i++) {
      if (triggerFolderDecisions_[i])
        fillMonitorElement(iEvent,
                           DirName,
                           triggerFolderLabels_[i],
                           met,
                           patmet,
                           pfmet,
                           calomet,
                           zCand,
                           map_of_MEs,
                           bLumiSecPlot,
                           fillPFCandidatePlots,
                           techTriggerCase,
                           METFilterDecision);
    }
  } else if (DirName.find("ZJets") != std::string::npos) {
    fillMonitorElement(iEvent,
                       DirName,
                       std::string(""),
                       met,
                       patmet,
                       pfmet,
                       calomet,
                       zCand,
                       map_of_MEs,
                       bLumiSecPlot,
                       fillPFCandidatePlots,
                       techTriggerCase,
                       METFilterDecision);
  } else {
    fillMonitorElement(iEvent,
                       DirName,
                       std::string(""),
                       met,
                       patmet,
                       pfmet,
                       calomet,
                       zCand,
                       map_of_MEs,
                       bLumiSecPlot,
                       fillPFCandidatePlots,
                       techTriggerCase,
                       METFilterDecision);
  }
}

// ***********************************************************
void METAnalyzer::fillMonitorElement(const edm::Event& iEvent,
                                     std::string DirName,
                                     std::string subFolderName,
                                     const reco::MET& met,
                                     const pat::MET* patmet,
                                     const reco::PFMET* pfmet,
                                     const reco::CaloMET* calomet,
                                     const reco::Candidate::PolarLorentzVector& zCand,
                                     std::map<std::string, MonitorElement*>& map_of_MEs,
                                     bool bLumiSecPlot,
                                     bool fillPFCandidatePlots,
                                     std::vector<bool> techTriggerCase,
                                     std::vector<bool> METFilterDecision) {
  bool do_only_Z_histograms = false;
  if (DirName.find("ZJets") != std::string::npos) {  //do Z plots only
    do_only_Z_histograms = true;
    //\vec{p}_{T}^{Z}+vec{u}_{T}+\vec{MET}=0

    double u_x = -met.px() - zCand.Px();
    double u_y = -met.py() - zCand.Py();

    //protection for VERY special case where Z-Pt==0
    double u_par = 0;
    double u_perp = sqrt(u_x * u_x + u_y * u_y);
    double e_Z_x = 0;
    double e_Z_y = 0;
    if (zCand.Pt() != 0) {
      e_Z_x = zCand.Px() / zCand.Pt();
      e_Z_y = zCand.Py() / zCand.Pt();
    }
    u_par = u_x * e_Z_x + u_y * e_Z_y;
    u_perp = -e_Z_y * u_x + e_Z_x * u_y;

    meZJets_u_par = map_of_MEs[DirName + "/" + "u_parallel_Z_inc"];
    if (meZJets_u_par && meZJets_u_par->getRootObject())
      meZJets_u_par->Fill(u_par);
    if (zCand.Pt() < 15) {
      meZJets_u_par_ZPt_0_15 = map_of_MEs[DirName + "/" + "u_parallel_ZPt_0_15"];
      if (meZJets_u_par_ZPt_0_15 && meZJets_u_par_ZPt_0_15->getRootObject())
        meZJets_u_par_ZPt_0_15->Fill(u_par);
    } else if (zCand.Pt() < 30) {
      meZJets_u_par_ZPt_15_30 = map_of_MEs[DirName + "/" + "u_parallel_ZPt_15_30"];
      if (meZJets_u_par_ZPt_15_30 && meZJets_u_par_ZPt_15_30->getRootObject())
        meZJets_u_par_ZPt_15_30->Fill(u_par);
    } else if (zCand.Pt() < 55) {
      meZJets_u_par_ZPt_30_55 = map_of_MEs[DirName + "/" + "u_parallel_ZPt_30_55"];
      if (meZJets_u_par_ZPt_30_55 && meZJets_u_par_ZPt_30_55->getRootObject())
        meZJets_u_par_ZPt_30_55->Fill(u_par);
    } else if (zCand.Pt() < 75) {
      meZJets_u_par_ZPt_55_75 = map_of_MEs[DirName + "/" + "u_parallel_ZPt_55_75"];
      if (meZJets_u_par_ZPt_55_75 && meZJets_u_par_ZPt_55_75->getRootObject())
        meZJets_u_par_ZPt_55_75->Fill(u_par);
    } else if (zCand.Pt() < 150) {
      meZJets_u_par_ZPt_75_150 = map_of_MEs[DirName + "/" + "u_parallel_ZPt_75_150"];
      if (meZJets_u_par_ZPt_75_150 && meZJets_u_par_ZPt_75_150->getRootObject())
        meZJets_u_par_ZPt_75_150->Fill(u_par);
    } else if (zCand.Pt() < 290) {
      meZJets_u_par_ZPt_150_290 = map_of_MEs[DirName + "/" + "u_parallel_ZPt_150_290"];
      if (meZJets_u_par_ZPt_150_290 && meZJets_u_par_ZPt_150_290->getRootObject())
        meZJets_u_par_ZPt_150_290->Fill(u_par);
    } else {
      meZJets_u_par_ZPt_290 = map_of_MEs[DirName + "/" + "u_parallel_ZPt_290"];
      if (meZJets_u_par_ZPt_290 && meZJets_u_par_ZPt_290->getRootObject())
        meZJets_u_par_ZPt_290->Fill(u_par);
    }

    meZJets_u_perp = map_of_MEs[DirName + "/" + "u_perp_Z_inc"];
    if (meZJets_u_perp && meZJets_u_perp->getRootObject())
      meZJets_u_perp->Fill(u_perp);
    if (zCand.Pt() < 15) {
      meZJets_u_perp_ZPt_0_15 = map_of_MEs[DirName + "/" + "u_perp_ZPt_0_15"];
      if (meZJets_u_perp_ZPt_0_15 && meZJets_u_perp_ZPt_0_15->getRootObject())
        meZJets_u_perp_ZPt_0_15->Fill(u_perp);
    } else if (zCand.Pt() < 30) {
      meZJets_u_perp_ZPt_15_30 = map_of_MEs[DirName + "/" + "u_perp_ZPt_15_30"];
      if (meZJets_u_perp_ZPt_15_30 && meZJets_u_perp_ZPt_15_30->getRootObject())
        meZJets_u_perp_ZPt_15_30->Fill(u_perp);
    } else if (zCand.Pt() < 55) {
      meZJets_u_perp_ZPt_30_55 = map_of_MEs[DirName + "/" + "u_perp_ZPt_30_55"];
      if (meZJets_u_perp_ZPt_30_55 && meZJets_u_perp_ZPt_30_55->getRootObject())
        meZJets_u_perp_ZPt_30_55->Fill(u_perp);
    } else if (zCand.Pt() < 75) {
      meZJets_u_perp_ZPt_55_75 = map_of_MEs[DirName + "/" + "u_perp_ZPt_55_75"];
      if (meZJets_u_perp_ZPt_55_75 && meZJets_u_perp_ZPt_55_75->getRootObject())
        meZJets_u_perp_ZPt_55_75->Fill(u_perp);
    } else if (zCand.Pt() < 150) {
      meZJets_u_perp_ZPt_75_150 = map_of_MEs[DirName + "/" + "u_perp_ZPt_75_150"];
      if (meZJets_u_perp_ZPt_75_150 && meZJets_u_perp_ZPt_75_150->getRootObject())
        meZJets_u_perp_ZPt_75_150->Fill(u_perp);
    } else if (zCand.Pt() < 290) {
      meZJets_u_perp_ZPt_150_290 = map_of_MEs[DirName + "/" + "u_perp_ZPt_150_290"];
      if (meZJets_u_perp_ZPt_150_290 && meZJets_u_perp_ZPt_150_290->getRootObject())
        meZJets_u_perp_ZPt_150_290->Fill(u_perp);
    } else {
      meZJets_u_perp_ZPt_290 = map_of_MEs[DirName + "/" + "u_perp_ZPt_290"];
      if (meZJets_u_perp_ZPt_290 && meZJets_u_perp_ZPt_290->getRootObject())
        meZJets_u_perp_ZPt_290->Fill(u_perp);
    }
  }
  if (!do_only_Z_histograms) {
    // Reconstructed MET Information
    double SumET = met.sumEt();
    double METSig = met.mEtSig();
    //double Ez     = met.e_longitudinal();
    double MET = met.pt();
    double MEx = met.px();
    double MEy = met.py();
    double METPhi = met.phi();
    //
    int myLuminosityBlock;
    myLuminosityBlock = iEvent.luminosityBlock();
    //

    if (!subFolderName.empty()) {
      DirName = DirName + "/" + subFolderName;
    }

    hTrigger = map_of_MEs[DirName + "/triggerResults"];
    if (hTrigger && hTrigger->getRootObject()) {
      for (unsigned int i = 0; i < allTriggerDecisions_.size(); i++) {
        if (i < (unsigned int)hTrigger->getNbinsX()) {
          hTrigger->Fill(i + .5, allTriggerDecisions_[i]);
        }
      }
    }

    hMEx = map_of_MEs[DirName + "/" + "MEx"];
    if (hMEx && hMEx->getRootObject())
      hMEx->Fill(MEx);
    hMEy = map_of_MEs[DirName + "/" + "MEy"];
    if (hMEy && hMEy->getRootObject())
      hMEy->Fill(MEy);
    hMET = map_of_MEs[DirName + "/" + "MET"];
    if (hMET && hMET->getRootObject())
      hMET->Fill(MET);
    hMET_2 = map_of_MEs[DirName + "/" + "MET_2"];
    if (hMET_2 && hMET_2->getRootObject())
      hMET_2->Fill(MET);

    //hMET_HBHENoiseFilter        = ibooker.book1D("MET_HBHENoiseFilter",        "MET_HBHENoiseFiltered",        200,    0, 1000);
    //hMET_CSCTightHaloFilter    = ibooker.book1D("MET_CSCTightHaloFilter",        "MET_CSCTightHaloFiltered",        200,    0, 1000);
    //hMET_eeBadScFilter    = ibooker.book1D("MET_eeBadScFilter",        "MET_eeBadScFiltered",        200,    0, 1000);
    //hMET_HBHEIsoNoiseFilter        = ibooker.book1D("MET_HBHEIsoNoiseFilter",        "MET_HBHEIsoNoiseFiltered",        200,    0, 1000);
    //hMET_CSCTightHalo2015Filter    = ibooker.book1D("MET_CSCTightHalo2015Filter",        "MET_CSCTightHalo2015Filtered",        200,    0, 1000);
    //hMET_EcalDeadCellTriggerFilter    = ibooker.book1D("MET_EcalDeadCellTriggerFilter",        "MET_EcalDeadCellTriggerFiltered",        200,    0, 1000);
    //hMET_EcalDeadCellBoundaryFilter    = ibooker.book1D("MET_EcalDeadCellBoundaryFilter",        "MET_EcalDeadCellBoundaryFiltered",        200,    0, 1000);
    //hMET_HcalStripHaloFilter    = ibooker.book1D("MET_HcalStripHaloFilter",        "MET_HcalStripHaloFiltered",        200,    0, 1000);

    bool HBHENoiseFilterResult = false;
    bool CSCTightHaloFilterResult = false;
    bool eeBadScFilterResult = false;
    bool HBHEIsoNoiseFilterResult = false;
    bool CSCTightHalo2015FilterResult = false;
    bool EcalDeadCellTriggerFilterResult = false;
    bool EcalDeadCellBoundaryFilterResult = false;
    bool HcalStripHaloFilterResult = false;
    HBHENoiseFilterResult = METFilterDecision[0];
    if (HBHENoiseFilterResult) {
      hMET_HBHENoiseFilter = map_of_MEs[DirName + "/" + "MET_HBHENoiseFilter"];
      if (hMET_HBHENoiseFilter && hMET_HBHENoiseFilter->getRootObject())
        hMET_HBHENoiseFilter->Fill(MET);
    }
    CSCTightHaloFilterResult = METFilterDecision[1];
    if (CSCTightHaloFilterResult) {
      hMET_CSCTightHaloFilter = map_of_MEs[DirName + "/" + "MET_CSCTightHaloFilter"];
      if (hMET_CSCTightHaloFilter && hMET_CSCTightHaloFilter->getRootObject())
        hMET_CSCTightHaloFilter->Fill(MET);
    }
    eeBadScFilterResult = METFilterDecision[2];
    if (eeBadScFilterResult) {
      hMET_eeBadScFilter = map_of_MEs[DirName + "/" + "MET_eeBadScFilter"];
      if (hMET_eeBadScFilter && hMET_eeBadScFilter->getRootObject())
        hMET_eeBadScFilter->Fill(MET);
    }
    HBHEIsoNoiseFilterResult = METFilterDecision[3];
    if (HBHEIsoNoiseFilterResult) {
      hMET_HBHEIsoNoiseFilter = map_of_MEs[DirName + "/" + "MET_HBHEIsoNoiseFilter"];
      if (hMET_HBHEIsoNoiseFilter && hMET_HBHEIsoNoiseFilter->getRootObject())
        hMET_HBHEIsoNoiseFilter->Fill(MET);
    }
    CSCTightHalo2015FilterResult = METFilterDecision[4];
    if (CSCTightHalo2015FilterResult) {
      hMET_CSCTightHalo2015Filter = map_of_MEs[DirName + "/" + "MET_CSCTightHalo2015Filter"];
      if (hMET_CSCTightHalo2015Filter && hMET_CSCTightHalo2015Filter->getRootObject())
        hMET_CSCTightHalo2015Filter->Fill(MET);
    }
    EcalDeadCellTriggerFilterResult = METFilterDecision[5];
    if (EcalDeadCellTriggerFilterResult) {
      hMET_EcalDeadCellTriggerFilter = map_of_MEs[DirName + "/" + "MET_EcalDeadCellTriggerFilter"];
      if (hMET_EcalDeadCellTriggerFilter && hMET_EcalDeadCellTriggerFilter->getRootObject())
        hMET_EcalDeadCellTriggerFilter->Fill(MET);
    }
    EcalDeadCellBoundaryFilterResult = METFilterDecision[6];
    if (EcalDeadCellBoundaryFilterResult) {
      hMET_EcalDeadCellBoundaryFilter = map_of_MEs[DirName + "/" + "MET_EcalDeadCellBoundaryFilter"];
      if (hMET_EcalDeadCellBoundaryFilter && hMET_EcalDeadCellBoundaryFilter->getRootObject())
        hMET_EcalDeadCellBoundaryFilter->Fill(MET);
    }
    HcalStripHaloFilterResult = METFilterDecision[7];
    if (HcalStripHaloFilterResult) {
      hMET_HcalStripHaloFilter = map_of_MEs[DirName + "/" + "MET_HcalStripHaloFilter"];
      if (hMET_HcalStripHaloFilter && hMET_HcalStripHaloFilter->getRootObject())
        hMET_HcalStripHaloFilter->Fill(MET);
    }
    hMETPhi = map_of_MEs[DirName + "/" + "METPhi"];
    if (hMETPhi && hMETPhi->getRootObject())
      hMETPhi->Fill(METPhi);
    hSumET = map_of_MEs[DirName + "/" + "SumET"];
    if (hSumET && hSumET->getRootObject())
      hSumET->Fill(SumET);
    hMETSig = map_of_MEs[DirName + "/" + "METSig"];
    if (hMETSig && hMETSig->getRootObject())
      hMETSig->Fill(METSig);
    hMET_logx = map_of_MEs[DirName + "/" + "MET_logx"];
    if (hMET_logx && hMET_logx->getRootObject())
      hMET_logx->Fill(log10(MET));
    hSumET_logx = map_of_MEs[DirName + "/" + "SumET_logx"];
    if (hSumET_logx && hSumET_logx->getRootObject())
      hSumET_logx->Fill(log10(SumET));

    // Fill NPV profiles
    //--------------------------------------------------------------------------
    meMEx_profile = map_of_MEs[DirName + "/MEx_profile"];
    meMEy_profile = map_of_MEs[DirName + "/MEy_profile"];
    meMET_profile = map_of_MEs[DirName + "/MET_profile"];
    meSumET_profile = map_of_MEs[DirName + "/SumET_profile"];

    if (meMEx_profile && meMEx_profile->getRootObject())
      meMEx_profile->Fill(numPV_, MEx);
    if (meMEy_profile && meMEy_profile->getRootObject())
      meMEy_profile->Fill(numPV_, MEy);
    if (meMET_profile && meMET_profile->getRootObject())
      meMET_profile->Fill(numPV_, MET);
    if (meSumET_profile && meSumET_profile->getRootObject())
      meSumET_profile->Fill(numPV_, SumET);

    if (isCaloMet_) {
      //const reco::CaloMETCollection *calometcol = calometcoll.product();
      //const reco::CaloMET *calomet;
      //calomet = &(calometcol->front());

      double caloEtFractionHadronic = calomet->etFractionHadronic();
      double caloEmEtFraction = calomet->emEtFraction();

      double caloHadEtInHB = calomet->hadEtInHB();
      double caloHadEtInHO = calomet->hadEtInHO();
      double caloHadEtInHE = calomet->hadEtInHE();
      double caloHadEtInHF = calomet->hadEtInHF();
      double caloEmEtInEB = calomet->emEtInEB();
      double caloEmEtInEE = calomet->emEtInEE();
      double caloEmEtInHF = calomet->emEtInHF();

      hCaloHadEtInHB = map_of_MEs[DirName + "/" + "CaloHadEtInHB"];
      if (hCaloHadEtInHB && hCaloHadEtInHB->getRootObject())
        hCaloHadEtInHB->Fill(caloHadEtInHB);
      hCaloHadEtInHO = map_of_MEs[DirName + "/" + "CaloHadEtInHO"];
      if (hCaloHadEtInHO && hCaloHadEtInHO->getRootObject())
        hCaloHadEtInHO->Fill(caloHadEtInHO);
      hCaloHadEtInHE = map_of_MEs[DirName + "/" + "CaloHadEtInHE"];
      if (hCaloHadEtInHE && hCaloHadEtInHE->getRootObject())
        hCaloHadEtInHE->Fill(caloHadEtInHE);
      hCaloHadEtInHF = map_of_MEs[DirName + "/" + "CaloHadEtInHF"];
      if (hCaloHadEtInHF && hCaloHadEtInHF->getRootObject())
        hCaloHadEtInHF->Fill(caloHadEtInHF);
      hCaloEmEtInEB = map_of_MEs[DirName + "/" + "CaloEmEtInEB"];
      if (hCaloEmEtInEB && hCaloEmEtInEB->getRootObject())
        hCaloEmEtInEB->Fill(caloEmEtInEB);
      hCaloEmEtInEE = map_of_MEs[DirName + "/" + "CaloEmEtInEE"];
      if (hCaloEmEtInEE && hCaloEmEtInEE->getRootObject())
        hCaloEmEtInEE->Fill(caloEmEtInEE);
      hCaloEmEtInHF = map_of_MEs[DirName + "/" + "CaloEmEtInHF"];
      if (hCaloEmEtInHF && hCaloEmEtInHF->getRootObject())
        hCaloEmEtInHF->Fill(caloEmEtInHF);

      hCaloMETPhi020 = map_of_MEs[DirName + "/" + "CaloMETPhi020"];
      if (MET > 20. && hCaloMETPhi020 && hCaloMETPhi020->getRootObject()) {
        hCaloMETPhi020->Fill(METPhi);
      }

      hCaloEtFractionHadronic = map_of_MEs[DirName + "/" + "CaloEtFractionHadronic"];
      if (hCaloEtFractionHadronic && hCaloEtFractionHadronic->getRootObject())
        hCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
      hCaloEmEtFraction = map_of_MEs[DirName + "/" + "CaloEmEtFraction"];
      if (hCaloEmEtFraction && hCaloEmEtFraction->getRootObject())
        hCaloEmEtFraction->Fill(caloEmEtFraction);
      hCaloEmEtFraction020 = map_of_MEs[DirName + "/" + "CaloEmEtFraction020"];
      if (MET > 20. && hCaloEmEtFraction020 && hCaloEmEtFraction020->getRootObject())
        hCaloEmEtFraction020->Fill(caloEmEtFraction);
    }
    if (isPFMet_) {
      if (fillPFCandidatePlots && fillCandidateMap_histos) {
        for (unsigned int i = 0; i < countsPFCand_.size(); i++) {
          countsPFCand_[i] = 0;
          MExPFCand_[i] = 0.;
          MEyPFCand_[i] = 0.;
        }

        // typedef std::vector<reco::PFCandidate> pfCand;
        edm::Handle<std::vector<reco::PFCandidate> > particleFlow;
        iEvent.getByToken(pflowToken_, particleFlow);

        float pt_sum_CHF_Barrel = 0;
        float pt_sum_CHF_Endcap_plus = 0;
        float pt_sum_CHF_Endcap_minus = 0;
        float pt_sum_NHF_Barrel = 0;
        float pt_sum_NHF_Endcap_plus = 0;
        float pt_sum_NHF_Endcap_minus = 0;
        float pt_sum_PhF_Barrel = 0;
        float pt_sum_PhF_Endcap_plus = 0;
        float pt_sum_PhF_Endcap_minus = 0;
        float pt_sum_HFH_plus = 0;
        float pt_sum_HFH_minus = 0;
        float pt_sum_HFE_plus = 0;
        float pt_sum_HFE_minus = 0;

        float px_chargedHadronsBarrel = 0;
        float py_chargedHadronsBarrel = 0;
        float px_chargedHadronsEndcapPlus = 0;
        float py_chargedHadronsEndcapPlus = 0;
        float px_chargedHadronsEndcapMinus = 0;
        float py_chargedHadronsEndcapMinus = 0;
        float px_neutralHadronsBarrel = 0;
        float py_neutralHadronsBarrel = 0;
        float px_neutralHadronsEndcapPlus = 0;
        float py_neutralHadronsEndcapPlus = 0;
        float px_neutralHadronsEndcapMinus = 0;
        float py_neutralHadronsEndcapMinus = 0;
        float px_PhotonsBarrel = 0;
        float py_PhotonsBarrel = 0;
        float px_PhotonsEndcapPlus = 0;
        float py_PhotonsEndcapPlus = 0;
        float px_PhotonsEndcapMinus = 0;
        float py_PhotonsEndcapMinus = 0;
        float px_HFHadronsPlus = 0;
        float py_HFHadronsPlus = 0;
        float px_HFHadronsMinus = 0;
        float py_HFHadronsMinus = 0;
        float px_HFEGammasPlus = 0;
        float py_HFEGammasPlus = 0;
        float px_HFEGammasMinus = 0;
        float py_HFEGammasMinus = 0;
        for (unsigned int i = 0; i < particleFlow->size(); i++) {
          const reco::PFCandidate& c = particleFlow->at(i);
          if (c.particleId() == 1) {  //charged hadrons
            //endcap minus
            if (c.eta() > (-3.0) && c.eta() < (-1.392)) {
              px_chargedHadronsEndcapMinus -= c.px();
              py_chargedHadronsEndcapMinus -= c.py();
              pt_sum_CHF_Endcap_minus += c.et();
            } else if (c.eta() >= (-1.392) && c.eta() <= 1.392) {  //barrel
              px_chargedHadronsBarrel -= c.px();
              py_chargedHadronsBarrel -= c.py();
              pt_sum_CHF_Barrel += c.et();
            } else if (c.eta() > 1.392 && c.eta() < 3.0) {  //endcap plus
              px_chargedHadronsEndcapPlus -= c.px();
              py_chargedHadronsEndcapPlus -= c.py();
              pt_sum_CHF_Endcap_plus += c.et();
            }
          }
          if (c.particleId() == 5) {  //neutral hadrons
            //endcap minus
            if (c.eta() > (-3.0) && c.eta() < (-1.392)) {
              px_neutralHadronsEndcapMinus -= c.px();
              py_neutralHadronsEndcapMinus -= c.py();
              pt_sum_NHF_Endcap_minus += c.et();
            } else if (c.eta() >= (-1.392) && c.eta() <= 1.392) {
              px_neutralHadronsBarrel -= c.px();
              py_neutralHadronsBarrel -= c.py();
              pt_sum_NHF_Barrel += c.et();
            } else if (c.eta() > 1.392 && c.eta() < 3.0) {
              px_neutralHadronsEndcapPlus -= c.px();
              py_neutralHadronsEndcapPlus -= c.py();
              pt_sum_NHF_Endcap_plus += c.et();
            }
          }
          if (c.particleId() == 4) {  //photons
            //endcap minus
            if (c.eta() > (-3.0) && c.eta() < (-1.479)) {
              px_PhotonsEndcapMinus -= c.px();
              py_PhotonsEndcapMinus -= c.py();
              pt_sum_PhF_Endcap_minus += c.et();
            } else if (c.eta() >= (-1.479) && c.eta() <= 1.479) {
              px_PhotonsBarrel -= c.px();
              py_PhotonsBarrel -= c.py();
              pt_sum_PhF_Barrel += c.et();
            } else if (c.eta() > 1.479 && c.eta() < 3.0) {
              px_PhotonsEndcapPlus -= c.px();
              py_PhotonsEndcapPlus -= c.py();
              pt_sum_PhF_Endcap_plus += c.et();
            }
          }
          if (c.particleId() == 6) {  //HFHadrons
            //forward minus
            if (c.eta() > (-5.20) && c.eta() < -2.901376) {
              pt_sum_HFH_minus += c.et();
              px_HFHadronsMinus -= c.px();
              py_HFHadronsMinus -= c.py();
            } else if (c.eta() > 2.901376 && c.eta() < 5.20) {  //forward plus
              px_HFHadronsPlus -= c.px();
              py_HFHadronsPlus -= c.py();
              pt_sum_HFH_plus += c.et();
            }
          }
          if (c.particleId() == 7) {  //HFEGammas
            //forward minus
            if (c.eta() > (-5.20) && c.eta() < -2.901376) {
              pt_sum_HFE_minus += c.et();
              px_HFEGammasMinus -= c.px();
              py_HFEGammasMinus -= c.py();
            } else if (c.eta() > 2.901376 && c.eta() < 5.20) {  //forward plus
              px_HFEGammasPlus -= c.px();
              py_HFEGammasPlus -= c.py();
              pt_sum_HFE_plus += c.et();
            }
          }
          for (unsigned int j = 0; j < typePFCand_.size(); j++) {
            if (c.particleId() == typePFCand_[j]) {
              //second check for endcap, if inside barrel Max and Min symmetric around 0
              if (((c.eta() > etaMinPFCand_[j]) && (c.eta() < etaMaxPFCand_[j])) ||
                  ((c.eta() > (-etaMaxPFCand_[j])) && (c.eta() < (-etaMinPFCand_[j])))) {
                countsPFCand_[j] += 1;
                MExPFCand_[j] -= c.px();
                MEyPFCand_[j] -= c.py();
              }
            }
          }
        }

        for (unsigned int j = 0; j < countsPFCand_.size(); j++) {
          profilePFCand_x_[j] = map_of_MEs[DirName + "/" + profilePFCand_x_name_[j]];
          if (profilePFCand_x_[j] && profilePFCand_x_[j]->getRootObject())
            profilePFCand_x_[j]->Fill(countsPFCand_[j], MExPFCand_[j]);
          profilePFCand_y_[j] = map_of_MEs[DirName + "/" + profilePFCand_y_name_[j]];
          if (profilePFCand_y_[j] && profilePFCand_y_[j]->getRootObject())
            profilePFCand_y_[j]->Fill(countsPFCand_[j], MEyPFCand_[j]);
        }
        meCHF_Barrel = map_of_MEs[DirName + "/" + "PfChargedHadronEtFractionBarrel"];
        if (meCHF_Barrel && meCHF_Barrel->getRootObject())
          meCHF_Barrel->Fill(pt_sum_CHF_Barrel / pfmet->sumEt());
        meCHF_EndcapPlus = map_of_MEs[DirName + "/" + "PfChargedHadronEtFractionEndcapPlus"];
        if (meCHF_EndcapPlus && meCHF_EndcapPlus->getRootObject())
          meCHF_EndcapPlus->Fill(pt_sum_CHF_Endcap_plus / pfmet->sumEt());
        meCHF_EndcapMinus = map_of_MEs[DirName + "/" + "PfChargedHadronEtFractionEndcapMinus"];
        if (meCHF_EndcapMinus && meCHF_EndcapMinus->getRootObject())
          meCHF_EndcapMinus->Fill(pt_sum_CHF_Endcap_minus / pfmet->sumEt());
        meNHF_Barrel = map_of_MEs[DirName + "/" + "PfNeutralHadronEtFractionBarrel"];
        if (meNHF_Barrel && meNHF_Barrel->getRootObject())
          meNHF_Barrel->Fill(pt_sum_NHF_Barrel / pfmet->sumEt());
        meNHF_EndcapPlus = map_of_MEs[DirName + "/" + "PfNeutralHadronEtFractionEndcapPlus"];
        if (meNHF_EndcapPlus && meNHF_EndcapPlus->getRootObject())
          meNHF_EndcapPlus->Fill(pt_sum_NHF_Endcap_plus / pfmet->sumEt());
        meNHF_EndcapMinus = map_of_MEs[DirName + "/" + "PfNeutralHadronEtFractionEndcapMinus"];
        if (meNHF_EndcapMinus && meNHF_EndcapMinus->getRootObject())
          meNHF_EndcapMinus->Fill(pt_sum_NHF_Endcap_minus / pfmet->sumEt());
        mePhF_Barrel = map_of_MEs[DirName + "/" + "PfPhotonEtFractionBarrel"];
        if (mePhF_Barrel && mePhF_Barrel->getRootObject())
          mePhF_Barrel->Fill(pt_sum_PhF_Barrel / pfmet->sumEt());
        mePhF_EndcapPlus = map_of_MEs[DirName + "/" + "PfPhotonEtFractionEndcapPlus"];
        if (mePhF_EndcapPlus && mePhF_EndcapPlus->getRootObject())
          mePhF_EndcapPlus->Fill(pt_sum_PhF_Endcap_plus / pfmet->sumEt());
        mePhF_EndcapMinus = map_of_MEs[DirName + "/" + "PfPhotonEtFractionEndcapMinus"];
        if (mePhF_EndcapMinus && mePhF_EndcapMinus->getRootObject())
          mePhF_EndcapMinus->Fill(pt_sum_PhF_Endcap_minus / pfmet->sumEt());
        meHFHadF_Plus = map_of_MEs[DirName + "/" + "PfHFHadronEtFractionPlus"];
        if (meHFHadF_Plus && meHFHadF_Plus->getRootObject())
          meHFHadF_Plus->Fill(pt_sum_HFH_plus / pfmet->sumEt());
        meHFHadF_Minus = map_of_MEs[DirName + "/" + "PfHFHadronEtFractionMinus"];
        if (meHFHadF_Minus && meHFHadF_Minus->getRootObject())
          meHFHadF_Minus->Fill(pt_sum_HFH_minus / pfmet->sumEt());
        meHFEMF_Plus = map_of_MEs[DirName + "/" + "PfHFEMEtFractionPlus"];
        if (meHFEMF_Plus && meHFEMF_Plus->getRootObject())
          meHFEMF_Plus->Fill(pt_sum_HFE_plus / pfmet->sumEt());
        meHFEMF_Minus = map_of_MEs[DirName + "/" + "PfHFEMEtFractionMinus"];
        if (meHFEMF_Minus && meHFEMF_Minus->getRootObject())
          meHFEMF_Minus->Fill(pt_sum_HFE_minus / pfmet->sumEt());
        //sanity check if we have any type of the respective species in the events
        //else don't fill phi, as else we have a results of a biased peak at 0
        //if pt_sum of species part is 0, obviously that would be the case
        if (pt_sum_CHF_Barrel) {
          meMETPhiChargedHadronsBarrel = map_of_MEs[DirName + "/" + "METPhiChargedHadronsBarrel"];
          if (meMETPhiChargedHadronsBarrel && meMETPhiChargedHadronsBarrel->getRootObject())
            meMETPhiChargedHadronsBarrel->Fill(atan2(py_chargedHadronsBarrel, px_chargedHadronsBarrel));
        }
        if (pt_sum_CHF_Endcap_plus) {
          meMETPhiChargedHadronsEndcapPlus = map_of_MEs[DirName + "/" + "METPhiChargedHadronsEndcapPlus"];
          if (meMETPhiChargedHadronsEndcapPlus && meMETPhiChargedHadronsEndcapPlus->getRootObject())
            meMETPhiChargedHadronsEndcapPlus->Fill(atan2(py_chargedHadronsEndcapPlus, px_chargedHadronsEndcapPlus));
        }
        if (pt_sum_CHF_Endcap_minus) {
          meMETPhiChargedHadronsEndcapMinus = map_of_MEs[DirName + "/" + "METPhiChargedHadronsEndcapMinus"];
          if (meMETPhiChargedHadronsEndcapMinus && meMETPhiChargedHadronsEndcapMinus->getRootObject())
            meMETPhiChargedHadronsEndcapMinus->Fill(atan2(py_chargedHadronsEndcapMinus, px_chargedHadronsEndcapMinus));
        }
        if (pt_sum_NHF_Barrel) {
          meMETPhiNeutralHadronsBarrel = map_of_MEs[DirName + "/" + "METPhiNeutralHadronsBarrel"];
          if (meMETPhiNeutralHadronsBarrel && meMETPhiNeutralHadronsBarrel->getRootObject())
            meMETPhiNeutralHadronsBarrel->Fill(atan2(py_neutralHadronsBarrel, px_neutralHadronsBarrel));
        }
        if (pt_sum_NHF_Endcap_plus) {
          meMETPhiNeutralHadronsEndcapPlus = map_of_MEs[DirName + "/" + "METPhiNeutralHadronsEndcapPlus"];
          if (meMETPhiNeutralHadronsEndcapPlus && meMETPhiNeutralHadronsEndcapPlus->getRootObject())
            meMETPhiNeutralHadronsEndcapPlus->Fill(atan2(py_neutralHadronsEndcapPlus, px_neutralHadronsEndcapPlus));
        }
        if (pt_sum_NHF_Endcap_minus) {
          meMETPhiNeutralHadronsEndcapMinus = map_of_MEs[DirName + "/" + "METPhiNeutralHadronsEndcapMinus"];
          if (meMETPhiNeutralHadronsEndcapMinus && meMETPhiNeutralHadronsEndcapMinus->getRootObject())
            meMETPhiNeutralHadronsEndcapMinus->Fill(atan2(py_neutralHadronsEndcapMinus, px_neutralHadronsEndcapMinus));
        }
        if (pt_sum_PhF_Barrel) {
          meMETPhiPhotonsBarrel = map_of_MEs[DirName + "/" + "METPhiPhotonsBarrel"];
          if (meMETPhiPhotonsBarrel && meMETPhiPhotonsBarrel->getRootObject())
            meMETPhiPhotonsBarrel->Fill(atan2(py_PhotonsBarrel, px_PhotonsBarrel));
        }
        if (pt_sum_PhF_Endcap_plus) {
          meMETPhiPhotonsEndcapPlus = map_of_MEs[DirName + "/" + "METPhiPhotonsEndcapPlus"];
          if (meMETPhiPhotonsEndcapPlus && meMETPhiPhotonsEndcapPlus->getRootObject())
            meMETPhiPhotonsEndcapPlus->Fill(atan2(py_PhotonsEndcapPlus, px_PhotonsEndcapPlus));
        }
        if (pt_sum_PhF_Endcap_minus) {
          meMETPhiPhotonsEndcapMinus = map_of_MEs[DirName + "/" + "METPhiPhotonsEndcapMinus"];
          if (meMETPhiPhotonsEndcapMinus && meMETPhiPhotonsEndcapMinus->getRootObject())
            meMETPhiPhotonsEndcapMinus->Fill(atan2(py_PhotonsEndcapMinus, px_PhotonsEndcapMinus));
        }
        if (pt_sum_HFH_plus) {
          meMETPhiHFHadronsPlus = map_of_MEs[DirName + "/" + "METPhiHFHadronsPlus"];
          if (meMETPhiHFHadronsPlus && meMETPhiHFHadronsPlus->getRootObject())
            meMETPhiHFHadronsPlus->Fill(atan2(py_HFHadronsPlus, px_HFHadronsPlus));
        }
        if (pt_sum_HFH_minus) {
          meMETPhiHFHadronsMinus = map_of_MEs[DirName + "/" + "METPhiHFHadronsMinus"];
          if (meMETPhiHFHadronsMinus && meMETPhiHFHadronsMinus->getRootObject())
            meMETPhiHFHadronsMinus->Fill(atan2(py_HFHadronsMinus, px_HFHadronsMinus));
        }
        if (pt_sum_HFE_plus) {
          meMETPhiHFEGammasPlus = map_of_MEs[DirName + "/" + "METPhiHFEGammasPlus"];
          if (meMETPhiHFEGammasPlus && meMETPhiHFEGammasPlus->getRootObject())
            meMETPhiHFEGammasPlus->Fill(atan2(py_HFEGammasPlus, px_HFEGammasPlus));
        }
        if (pt_sum_HFE_minus) {
          meMETPhiHFEGammasMinus = map_of_MEs[DirName + "/" + "METPhiHFEGammasMinus"];
          if (meMETPhiHFEGammasMinus && meMETPhiHFEGammasMinus->getRootObject())
            meMETPhiHFEGammasMinus->Fill(atan2(py_HFEGammasMinus, px_HFEGammasMinus));
        }
        //fill other diagnostic plots based on trigger decision
        /*if(techTriggerCase[0]){//techTriggerResultBx0 && techTriggerResultBxM2 && techTriggerResultBxM1 -> both previous bunches filled
	  meCHF_Barrel_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfChargedHadronEtFractionBarrel_BXm2BXm1Filled"]; if(meCHF_Barrel_BXm2BXm1Filled && meCHF_Barrel_BXm2BXm1Filled->getRootObject()) meCHF_Barrel_BXm2BXm1Filled->Fill(pt_sum_CHF_Barrel/pfmet->sumEt()); 
	  meCHF_EndcapPlus_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfChargedHadronEtFractionEndcapPlus_BXm2BXm1Filled"]; if(meCHF_EndcapPlus_BXm2BXm1Filled && meCHF_EndcapPlus_BXm2BXm1Filled->getRootObject()) meCHF_EndcapPlus_BXm2BXm1Filled->Fill(pt_sum_CHF_Endcap_plus/pfmet->sumEt()); 
	  meCHF_EndcapMinus_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfChargedHadronEtFractionEndcapMinus_BXm2BXm1Filled"]; if(meCHF_EndcapMinus_BXm2BXm1Filled && meCHF_EndcapMinus_BXm2BXm1Filled->getRootObject()) meCHF_EndcapMinus_BXm2BXm1Filled->Fill(pt_sum_CHF_Endcap_minus/pfmet->sumEt()); 
	  meNHF_Barrel_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfNeutralHadronEtFractionBarrel_BXm2BXm1Filled"]; if(meNHF_Barrel_BXm2BXm1Filled && meNHF_Barrel_BXm2BXm1Filled->getRootObject()) meNHF_Barrel_BXm2BXm1Filled->Fill(pt_sum_NHF_Barrel/pfmet->sumEt()); 
	  meNHF_EndcapPlus_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfNeutralHadronEtFractionEndcapPlus_BXm2BXm1Filled"]; if(meNHF_EndcapPlus_BXm2BXm1Filled && meNHF_EndcapPlus_BXm2BXm1Filled->getRootObject()) meNHF_EndcapPlus_BXm2BXm1Filled->Fill(pt_sum_NHF_Endcap_plus/pfmet->sumEt()); 
	  meNHF_EndcapMinus_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfNeutralHadronEtFractionEndcapMinus_BXm2BXm1Filled"]; if(meNHF_EndcapMinus_BXm2BXm1Filled && meNHF_EndcapMinus_BXm2BXm1Filled->getRootObject()) meNHF_EndcapMinus_BXm2BXm1Filled->Fill(pt_sum_NHF_Endcap_minus/pfmet->sumEt()); 
	  mePhF_Barrel_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfPhotonEtFractionBarrel_BXm2BXm1Filled"]; if(mePhF_Barrel_BXm2BXm1Filled && mePhF_Barrel_BXm2BXm1Filled->getRootObject()) mePhF_Barrel_BXm2BXm1Filled->Fill(pt_sum_PhF_Barrel/pfmet->sumEt()); 
	  mePhF_EndcapPlus_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfPhotonEtFractionEndcapPlus_BXm2BXm1Filled"]; if(mePhF_EndcapPlus_BXm2BXm1Filled && mePhF_EndcapPlus_BXm2BXm1Filled->getRootObject()) mePhF_EndcapPlus_BXm2BXm1Filled->Fill(pt_sum_PhF_Endcap_plus/pfmet->sumEt()); 
	  mePhF_EndcapMinus_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfPhotonEtFractionEndcapMinus_BXm2BXm1Filled"]; if(mePhF_EndcapMinus_BXm2BXm1Filled && mePhF_EndcapMinus_BXm2BXm1Filled->getRootObject()) mePhF_EndcapMinus_BXm2BXm1Filled->Fill(pt_sum_PhF_Endcap_minus/pfmet->sumEt()); 
     	  meHFHadF_Plus_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfHFHadronEtFractionPlus_BXm2BXm1Filled"]; if(meHFHadF_Plus_BXm2BXm1Filled && meHFHadF_Plus_BXm2BXm1Filled->getRootObject()) meHFHadF_Plus_BXm2BXm1Filled->Fill(pt_sum_HFH_plus/pfmet->sumEt()); 
	  meHFHadF_Minus_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfHFHadronEtFractionMinus_BXm2BXm1Filled"]; if(meHFHadF_Minus_BXm2BXm1Filled && meHFHadF_Minus_BXm2BXm1Filled->getRootObject()) meHFHadF_Minus_BXm2BXm1Filled->Fill(pt_sum_HFH_minus/pfmet->sumEt()); 
	  meHFEMF_Plus_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfHFEMEtFractionPlus_BXm2BXm1Filled"]; if(meHFEMF_Plus_BXm2BXm1Filled && meHFEMF_Plus_BXm2BXm1Filled->getRootObject()) meHFEMF_Plus_BXm2BXm1Filled->Fill(pt_sum_HFE_plus/pfmet->sumEt()); 
	  meHFEMF_Minus_BXm2BXm1Filled=map_of_MEs[DirName+"/"+"PfHFEMEtFractionMinus_BXm2BXm1Filled"]; if(meHFEMF_Minus_BXm2BXm1Filled && meHFEMF_Minus_BXm2BXm1Filled->getRootObject()) meHFEMF_Minus_BXm2BXm1Filled->Fill(pt_sum_HFE_minus/pfmet->sumEt());
	  mePhotonEtFraction_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"PfPhotonEtFraction_BXm2BXm1Filled"];     if (  mePhotonEtFraction_BXm2BXm1Filled  && mePhotonEtFraction_BXm2BXm1Filled ->getRootObject())  mePhotonEtFraction_BXm2BXm1Filled  ->Fill(pfmet->photonEtFraction());
	  meNeutralHadronEtFraction_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"PfNeutralHadronEtFraction_BXm2BXm1Filled"];     if (  meNeutralHadronEtFraction_BXm2BXm1Filled  && meNeutralHadronEtFraction_BXm2BXm1Filled ->getRootObject())  meNeutralHadronEtFraction_BXm2BXm1Filled  ->Fill(pfmet->neutralHadronEtFraction());
	  meChargedHadronEtFraction_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"PfChargedHadronEtFraction_BXm2BXm1Filled"];     if (  meChargedHadronEtFraction_BXm2BXm1Filled  && meChargedHadronEtFraction_BXm2BXm1Filled ->getRootObject())  meChargedHadronEtFraction_BXm2BXm1Filled  ->Fill(pfmet->chargedHadronEtFraction());
	  meMET_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"MET_BXm2BXm1Filled"];     if (  meMET_BXm2BXm1Filled  && meMET_BXm2BXm1Filled ->getRootObject())  meMET_BXm2BXm1Filled  ->Fill(pfmet->pt());
	  meSumET_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"SumET_BXm2BXm1Filled"];     if (  meSumET_BXm2BXm1Filled  && meSumET_BXm2BXm1Filled ->getRootObject())  meSumET_BXm2BXm1Filled  ->Fill(pfmet->sumEt());
	  if(pt_sum_CHF_Barrel){
	    meMETPhiChargedHadronsBarrel_BXm2BXm1Filled     = map_of_MEs[DirName+"/"+"METPhiChargedHadronsBarrel_BXm2BXm1Filled"];if(meMETPhiChargedHadronsBarrel_BXm2BXm1Filled  && meMETPhiChargedHadronsBarrel_BXm2BXm1Filled ->getRootObject())meMETPhiChargedHadronsBarrel_BXm2BXm1Filled->Fill(atan2(py_chargedHadronsBarrel,px_chargedHadronsBarrel));
	  }
	  if(pt_sum_CHF_Endcap_plus){
	    meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Filled  = map_of_MEs[DirName+"/"+"METPhiChargedHadronsEndcapPlus_BXm2BXm1Filled"];if(meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Filled  && meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Filled ->getRootObject())meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Filled->Fill(atan2(py_chargedHadronsEndcapPlus,px_chargedHadronsEndcapPlus));
	  }
	  if(pt_sum_CHF_Endcap_minus){
	    meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Filled     = map_of_MEs[DirName+"/"+"METPhiChargedHadronsEndcapMinus_BXm2BXm1Filled"];if(meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Filled  && meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Filled ->getRootObject())meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Filled->Fill(atan2(py_chargedHadronsEndcapMinus,px_chargedHadronsEndcapMinus));
	  }
	  if(pt_sum_NHF_Barrel){
	    meMETPhiNeutralHadronsBarrel_BXm2BXm1Filled     = map_of_MEs[DirName+"/"+"METPhiNeutralHadronsBarrel_BXm2BXm1Filled"];if(meMETPhiNeutralHadronsBarrel_BXm2BXm1Filled  && meMETPhiNeutralHadronsBarrel_BXm2BXm1Filled ->getRootObject())meMETPhiNeutralHadronsBarrel_BXm2BXm1Filled->Fill(atan2(py_neutralHadronsBarrel,px_neutralHadronsBarrel));
	  }
	  if(pt_sum_NHF_Endcap_plus){
	    meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Filled     = map_of_MEs[DirName+"/"+"METPhiNeutralHadronsEndcapPlus_BXm2BXm1Filled"];if(meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Filled  && meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Filled ->getRootObject())meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Filled->Fill(atan2(py_neutralHadronsEndcapPlus,px_neutralHadronsEndcapPlus));
	  }
	  if(pt_sum_NHF_Endcap_minus){
	    meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Filled     = map_of_MEs[DirName+"/"+"METPhiNeutralHadronsEndcapMinus_BXm2BXm1Filled"];if(meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Filled  && meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Filled ->getRootObject())meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Filled->Fill(atan2(py_neutralHadronsEndcapMinus,px_neutralHadronsEndcapMinus));
	  }
	  if(pt_sum_PhF_Barrel){
	    meMETPhiPhotonsBarrel_BXm2BXm1Filled     = map_of_MEs[DirName+"/"+"METPhiPhotonsBarrel_BXm2BXm1Filled"];if(meMETPhiPhotonsBarrel_BXm2BXm1Filled  && meMETPhiPhotonsBarrel_BXm2BXm1Filled ->getRootObject())meMETPhiPhotonsBarrel_BXm2BXm1Filled->Fill(atan2(py_PhotonsBarrel,px_PhotonsBarrel));
	  }
	  if(pt_sum_PhF_Endcap_plus){
	    meMETPhiPhotonsEndcapPlus_BXm2BXm1Filled     = map_of_MEs[DirName+"/"+"METPhiPhotonsEndcapPlus_BXm2BXm1Filled"];if(meMETPhiPhotonsEndcapPlus_BXm2BXm1Filled  && meMETPhiPhotonsEndcapPlus_BXm2BXm1Filled ->getRootObject())meMETPhiPhotonsEndcapPlus_BXm2BXm1Filled->Fill(atan2(py_PhotonsEndcapPlus,px_PhotonsEndcapPlus));
	  }
	  if(pt_sum_PhF_Endcap_minus){
	    meMETPhiPhotonsEndcapMinus_BXm2BXm1Filled     = map_of_MEs[DirName+"/"+"METPhiPhotonsEndcapMinus_BXm2BXm1Filled"];if(meMETPhiPhotonsEndcapMinus_BXm2BXm1Filled  && meMETPhiPhotonsEndcapMinus_BXm2BXm1Filled ->getRootObject())meMETPhiPhotonsEndcapMinus_BXm2BXm1Filled->Fill(atan2(py_PhotonsEndcapMinus,px_PhotonsEndcapMinus));
	  }
	  if(pt_sum_HFH_plus){
	    meMETPhiHFHadronsPlus_BXm2BXm1Filled     = map_of_MEs[DirName+"/"+"METPhiHFHadronsPlus_BXm2BXm1Filled"];if(meMETPhiHFHadronsPlus_BXm2BXm1Filled  && meMETPhiHFHadronsPlus_BXm2BXm1Filled ->getRootObject())meMETPhiHFHadronsPlus_BXm2BXm1Filled->Fill(atan2(py_HFHadronsPlus,px_HFHadronsPlus));
	  }
	  if(pt_sum_HFH_minus){
	    meMETPhiHFHadronsMinus_BXm2BXm1Filled     = map_of_MEs[DirName+"/"+"METPhiHFHadronsMinus_BXm2BXm1Filled"];if(meMETPhiHFHadronsMinus_BXm2BXm1Filled  && meMETPhiHFHadronsMinus_BXm2BXm1Filled ->getRootObject())meMETPhiHFHadronsMinus_BXm2BXm1Filled->Fill(atan2(py_HFHadronsMinus,px_HFHadronsMinus));
	  }
	  if(pt_sum_HFE_plus){
	    meMETPhiHFEGammasPlus_BXm2BXm1Filled     = map_of_MEs[DirName+"/"+"METPhiHFEGammasPlus_BXm2BXm1Filled"];if(meMETPhiHFEGammasPlus_BXm2BXm1Filled  && meMETPhiHFEGammasPlus_BXm2BXm1Filled ->getRootObject())meMETPhiHFEGammasPlus_BXm2BXm1Filled->Fill(atan2(py_HFEGammasPlus,px_HFEGammasPlus));
	  }
	  if(pt_sum_HFE_minus){
	    meMETPhiHFEGammasMinus_BXm2BXm1Filled     = map_of_MEs[DirName+"/"+"METPhiHFEGammasMinus_BXm2BXm1Filled"];if(meMETPhiHFEGammasMinus_BXm2BXm1Filled  && meMETPhiHFEGammasMinus_BXm2BXm1Filled ->getRootObject())meMETPhiHFEGammasMinus_BXm2BXm1Filled->Fill(atan2(py_HFEGammasMinus,px_HFEGammasMinus));
	  }
	  }*/
        if (techTriggerCase[1]) {  //techTriggerResultBx0 && techTriggerResultBxM1 -> previous bunch filled
          meCHF_Barrel_BXm1Filled = map_of_MEs[DirName + "/" + "PfChargedHadronEtFractionBarrel_BXm1Filled"];
          if (meCHF_Barrel_BXm1Filled && meCHF_Barrel_BXm1Filled->getRootObject())
            meCHF_Barrel_BXm1Filled->Fill(pt_sum_CHF_Barrel / pfmet->sumEt());
          meCHF_EndcapPlus_BXm1Filled = map_of_MEs[DirName + "/" + "PfChargedHadronEtFractionEndcapPlus_BXm1Filled"];
          if (meCHF_EndcapPlus_BXm1Filled && meCHF_EndcapPlus_BXm1Filled->getRootObject())
            meCHF_EndcapPlus_BXm1Filled->Fill(pt_sum_CHF_Endcap_plus / pfmet->sumEt());
          meCHF_EndcapMinus_BXm1Filled = map_of_MEs[DirName + "/" + "PfChargedHadronEtFractionEndcapMinus_BXm1Filled"];
          if (meCHF_EndcapMinus_BXm1Filled && meCHF_EndcapMinus_BXm1Filled->getRootObject())
            meCHF_EndcapMinus_BXm1Filled->Fill(pt_sum_CHF_Endcap_minus / pfmet->sumEt());
          meNHF_Barrel_BXm1Filled = map_of_MEs[DirName + "/" + "PfNeutralHadronEtFractionBarrel_BXm1Filled"];
          if (meNHF_Barrel_BXm1Filled && meNHF_Barrel_BXm1Filled->getRootObject())
            meNHF_Barrel_BXm1Filled->Fill(pt_sum_NHF_Barrel / pfmet->sumEt());
          meNHF_EndcapPlus_BXm1Filled = map_of_MEs[DirName + "/" + "PfNeutralHadronEtFractionEndcapPlus_BXm1Filled"];
          if (meNHF_EndcapPlus_BXm1Filled && meNHF_EndcapPlus_BXm1Filled->getRootObject())
            meNHF_EndcapPlus_BXm1Filled->Fill(pt_sum_NHF_Endcap_plus / pfmet->sumEt());
          meNHF_EndcapMinus_BXm1Filled = map_of_MEs[DirName + "/" + "PfNeutralHadronEtFractionEndcapMinus_BXm1Filled"];
          if (meNHF_EndcapMinus_BXm1Filled && meNHF_EndcapMinus_BXm1Filled->getRootObject())
            meNHF_EndcapMinus_BXm1Filled->Fill(pt_sum_NHF_Endcap_minus / pfmet->sumEt());
          mePhF_Barrel_BXm1Filled = map_of_MEs[DirName + "/" + "PfPhotonEtFractionBarrel_BXm1Filled"];
          if (mePhF_Barrel_BXm1Filled && mePhF_Barrel_BXm1Filled->getRootObject())
            mePhF_Barrel_BXm1Filled->Fill(pt_sum_PhF_Barrel / pfmet->sumEt());
          mePhF_EndcapPlus_BXm1Filled = map_of_MEs[DirName + "/" + "PfPhotonEtFractionEndcapPlus_BXm1Filled"];
          if (mePhF_EndcapPlus_BXm1Filled && mePhF_EndcapPlus_BXm1Filled->getRootObject())
            mePhF_EndcapPlus_BXm1Filled->Fill(pt_sum_PhF_Endcap_plus / pfmet->sumEt());
          mePhF_EndcapMinus_BXm1Filled = map_of_MEs[DirName + "/" + "PfPhotonEtFractionEndcapMinus_BXm1Filled"];
          if (mePhF_EndcapMinus_BXm1Filled && mePhF_EndcapMinus_BXm1Filled->getRootObject())
            mePhF_EndcapMinus_BXm1Filled->Fill(pt_sum_PhF_Endcap_minus / pfmet->sumEt());
          meHFHadF_Plus_BXm1Filled = map_of_MEs[DirName + "/" + "PfHFHadronEtFractionPlus_BXm1Filled"];
          if (meHFHadF_Plus_BXm1Filled && meHFHadF_Plus_BXm1Filled->getRootObject())
            meHFHadF_Plus_BXm1Filled->Fill(pt_sum_HFH_plus / pfmet->sumEt());
          meHFHadF_Minus_BXm1Filled = map_of_MEs[DirName + "/" + "PfHFHadronEtFractionMinus_BXm1Filled"];
          if (meHFHadF_Minus_BXm1Filled && meHFHadF_Minus_BXm1Filled->getRootObject())
            meHFHadF_Minus_BXm1Filled->Fill(pt_sum_HFH_minus / pfmet->sumEt());
          meHFEMF_Plus_BXm1Filled = map_of_MEs[DirName + "/" + "PfHFEMEtFractionPlus_BXm1Filled"];
          if (meHFEMF_Plus_BXm1Filled && meHFEMF_Plus_BXm1Filled->getRootObject())
            meHFEMF_Plus_BXm1Filled->Fill(pt_sum_HFE_plus / pfmet->sumEt());
          meHFEMF_Minus_BXm1Filled = map_of_MEs[DirName + "/" + "PfHFEMEtFractionMinus_BXm1Filled"];
          if (meHFEMF_Minus_BXm1Filled && meHFEMF_Minus_BXm1Filled->getRootObject())
            meHFEMF_Minus_BXm1Filled->Fill(pt_sum_HFE_minus / pfmet->sumEt());
          mePhotonEtFraction_BXm1Filled = map_of_MEs[DirName + "/" + "PfPhotonEtFraction_BXm1Filled"];
          if (mePhotonEtFraction_BXm1Filled && mePhotonEtFraction_BXm1Filled->getRootObject())
            mePhotonEtFraction_BXm1Filled->Fill(pfmet->photonEtFraction());
          meNeutralHadronEtFraction_BXm1Filled = map_of_MEs[DirName + "/" + "PfNeutralHadronEtFraction_BXm1Filled"];
          if (meNeutralHadronEtFraction_BXm1Filled && meNeutralHadronEtFraction_BXm1Filled->getRootObject())
            meNeutralHadronEtFraction_BXm1Filled->Fill(pfmet->neutralHadronEtFraction());
          meChargedHadronEtFraction_BXm1Filled = map_of_MEs[DirName + "/" + "PfChargedHadronEtFraction_BXm1Filled"];
          if (meChargedHadronEtFraction_BXm1Filled && meChargedHadronEtFraction_BXm1Filled->getRootObject())
            meChargedHadronEtFraction_BXm1Filled->Fill(pfmet->chargedHadronEtFraction());
          meMET_BXm1Filled = map_of_MEs[DirName + "/" + "MET_BXm1Filled"];
          if (meMET_BXm1Filled && meMET_BXm1Filled->getRootObject())
            meMET_BXm1Filled->Fill(pfmet->pt());
          meSumET_BXm1Filled = map_of_MEs[DirName + "/" + "SumET_BXm1Filled"];
          if (meSumET_BXm1Filled && meSumET_BXm1Filled->getRootObject())
            meSumET_BXm1Filled->Fill(pfmet->sumEt());
          if (pt_sum_CHF_Barrel) {
            meMETPhiChargedHadronsBarrel_BXm1Filled =
                map_of_MEs[DirName + "/" + "METPhiChargedHadronsBarrel_BXm1Filled"];
            if (meMETPhiChargedHadronsBarrel_BXm1Filled && meMETPhiChargedHadronsBarrel_BXm1Filled->getRootObject())
              meMETPhiChargedHadronsBarrel_BXm1Filled->Fill(atan2(py_chargedHadronsBarrel, px_chargedHadronsBarrel));
          }
          if (pt_sum_CHF_Endcap_plus) {
            meMETPhiChargedHadronsEndcapPlus_BXm1Filled =
                map_of_MEs[DirName + "/" + "METPhiChargedHadronsEndcapPlus_BXm1Filled"];
            if (meMETPhiChargedHadronsEndcapPlus_BXm1Filled &&
                meMETPhiChargedHadronsEndcapPlus_BXm1Filled->getRootObject())
              meMETPhiChargedHadronsEndcapPlus_BXm1Filled->Fill(
                  atan2(py_chargedHadronsEndcapPlus, px_chargedHadronsEndcapPlus));
          }
          if (pt_sum_CHF_Endcap_minus) {
            meMETPhiChargedHadronsEndcapMinus_BXm1Filled =
                map_of_MEs[DirName + "/" + "METPhiChargedHadronsEndcapMinus_BXm1Filled"];
            if (meMETPhiChargedHadronsEndcapMinus_BXm1Filled &&
                meMETPhiChargedHadronsEndcapMinus_BXm1Filled->getRootObject())
              meMETPhiChargedHadronsEndcapMinus_BXm1Filled->Fill(
                  atan2(py_chargedHadronsEndcapMinus, px_chargedHadronsEndcapMinus));
          }
          if (pt_sum_NHF_Barrel) {
            meMETPhiNeutralHadronsBarrel_BXm1Filled =
                map_of_MEs[DirName + "/" + "METPhiNeutralHadronsBarrel_BXm1Filled"];
            if (meMETPhiNeutralHadronsBarrel_BXm1Filled && meMETPhiNeutralHadronsBarrel_BXm1Filled->getRootObject())
              meMETPhiNeutralHadronsBarrel_BXm1Filled->Fill(atan2(py_neutralHadronsBarrel, px_neutralHadronsBarrel));
          }
          if (pt_sum_NHF_Endcap_plus) {
            meMETPhiNeutralHadronsEndcapPlus_BXm1Filled =
                map_of_MEs[DirName + "/" + "METPhiNeutralHadronsEndcapPlus_BXm1Filled"];
            if (meMETPhiNeutralHadronsEndcapPlus_BXm1Filled &&
                meMETPhiNeutralHadronsEndcapPlus_BXm1Filled->getRootObject())
              meMETPhiNeutralHadronsEndcapPlus_BXm1Filled->Fill(
                  atan2(py_neutralHadronsEndcapPlus, px_neutralHadronsEndcapPlus));
          }
          if (pt_sum_NHF_Endcap_minus) {
            meMETPhiNeutralHadronsEndcapMinus_BXm1Filled =
                map_of_MEs[DirName + "/" + "METPhiNeutralHadronsEndcapMinus_BXm1Filled"];
            if (meMETPhiNeutralHadronsEndcapMinus_BXm1Filled &&
                meMETPhiNeutralHadronsEndcapMinus_BXm1Filled->getRootObject())
              meMETPhiNeutralHadronsEndcapMinus_BXm1Filled->Fill(
                  atan2(py_neutralHadronsEndcapMinus, px_neutralHadronsEndcapMinus));
          }
          if (pt_sum_PhF_Barrel) {
            meMETPhiPhotonsBarrel_BXm1Filled = map_of_MEs[DirName + "/" + "METPhiPhotonsBarrel_BXm1Filled"];
            if (meMETPhiPhotonsBarrel_BXm1Filled && meMETPhiPhotonsBarrel_BXm1Filled->getRootObject())
              meMETPhiPhotonsBarrel_BXm1Filled->Fill(atan2(py_PhotonsBarrel, px_PhotonsBarrel));
          }
          if (pt_sum_PhF_Endcap_plus) {
            meMETPhiPhotonsEndcapPlus_BXm1Filled = map_of_MEs[DirName + "/" + "METPhiPhotonsEndcapPlus_BXm1Filled"];
            if (meMETPhiPhotonsEndcapPlus_BXm1Filled && meMETPhiPhotonsEndcapPlus_BXm1Filled->getRootObject())
              meMETPhiPhotonsEndcapPlus_BXm1Filled->Fill(atan2(py_PhotonsEndcapPlus, px_PhotonsEndcapPlus));
          }
          if (pt_sum_PhF_Endcap_minus) {
            meMETPhiPhotonsEndcapMinus_BXm1Filled = map_of_MEs[DirName + "/" + "METPhiPhotonsEndcapMinus_BXm1Filled"];
            if (meMETPhiPhotonsEndcapMinus_BXm1Filled && meMETPhiPhotonsEndcapMinus_BXm1Filled->getRootObject())
              meMETPhiPhotonsEndcapMinus_BXm1Filled->Fill(atan2(py_PhotonsEndcapMinus, px_PhotonsEndcapMinus));
          }
          if (pt_sum_HFH_plus) {
            meMETPhiHFHadronsPlus_BXm1Filled = map_of_MEs[DirName + "/" + "METPhiHFHadronsPlus_BXm1Filled"];
            if (meMETPhiHFHadronsPlus_BXm1Filled && meMETPhiHFHadronsPlus_BXm1Filled->getRootObject())
              meMETPhiHFHadronsPlus_BXm1Filled->Fill(atan2(py_HFHadronsPlus, px_HFHadronsPlus));
          }
          if (pt_sum_HFH_minus) {
            meMETPhiHFHadronsMinus_BXm1Filled = map_of_MEs[DirName + "/" + "METPhiHFHadronsMinus_BXm1Filled"];
            if (meMETPhiHFHadronsMinus_BXm1Filled && meMETPhiHFHadronsMinus_BXm1Filled->getRootObject())
              meMETPhiHFHadronsMinus_BXm1Filled->Fill(atan2(py_HFHadronsMinus, px_HFHadronsMinus));
          }
          if (pt_sum_HFE_plus) {
            meMETPhiHFEGammasPlus_BXm1Filled = map_of_MEs[DirName + "/" + "METPhiHFEGammasPlus_BXm1Filled"];
            if (meMETPhiHFEGammasPlus_BXm1Filled && meMETPhiHFEGammasPlus_BXm1Filled->getRootObject())
              meMETPhiHFEGammasPlus_BXm1Filled->Fill(atan2(py_HFEGammasPlus, px_HFEGammasPlus));
          }
          if (pt_sum_HFE_minus) {
            meMETPhiHFEGammasMinus_BXm1Filled = map_of_MEs[DirName + "/" + "METPhiHFEGammasMinus_BXm1Filled"];
            if (meMETPhiHFEGammasMinus_BXm1Filled && meMETPhiHFEGammasMinus_BXm1Filled->getRootObject())
              meMETPhiHFEGammasMinus_BXm1Filled->Fill(atan2(py_HFEGammasMinus, px_HFEGammasMinus));
          }
        }
        /*if(techTriggerCase[3]){//techTriggerResultBx0 && !techTriggerResultBxM2 && !techTriggerResultBxM1 ->previous two bunches empty
	  meCHF_Barrel_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfChargedHadronEtFractionBarrel_BXm2BXm1Empty"]; if(meCHF_Barrel_BXm2BXm1Empty && meCHF_Barrel_BXm2BXm1Empty->getRootObject()) meCHF_Barrel_BXm2BXm1Empty->Fill(pt_sum_CHF_Barrel/pfmet->sumEt()); 
	  meCHF_EndcapPlus_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfChargedHadronEtFractionEndcapPlus_BXm2BXm1Empty"]; if(meCHF_EndcapPlus_BXm2BXm1Empty && meCHF_EndcapPlus_BXm2BXm1Empty->getRootObject()) meCHF_EndcapPlus_BXm2BXm1Empty->Fill(pt_sum_CHF_Endcap_plus/pfmet->sumEt()); 
	  meCHF_EndcapMinus_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfChargedHadronEtFractionEndcapMinus_BXm2BXm1Empty"]; if(meCHF_EndcapMinus_BXm2BXm1Empty && meCHF_EndcapMinus_BXm2BXm1Empty->getRootObject()) meCHF_EndcapMinus_BXm2BXm1Empty->Fill(pt_sum_CHF_Endcap_minus/pfmet->sumEt()); 
	  meNHF_Barrel_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfNeutralHadronEtFractionBarrel_BXm2BXm1Empty"]; if(meNHF_Barrel_BXm2BXm1Empty && meNHF_Barrel_BXm2BXm1Empty->getRootObject()) meNHF_Barrel_BXm2BXm1Empty->Fill(pt_sum_NHF_Barrel/pfmet->sumEt()); 
	  meNHF_EndcapPlus_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfNeutralHadronEtFractionEndcapPlus_BXm2BXm1Empty"]; if(meNHF_EndcapPlus_BXm2BXm1Empty && meNHF_EndcapPlus_BXm2BXm1Empty->getRootObject()) meNHF_EndcapPlus_BXm2BXm1Empty->Fill(pt_sum_NHF_Endcap_plus/pfmet->sumEt()); 
	  meNHF_EndcapMinus_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfNeutralHadronEtFractionEndcapMinus_BXm2BXm1Empty"]; if(meNHF_EndcapMinus_BXm2BXm1Empty && meNHF_EndcapMinus_BXm2BXm1Empty->getRootObject()) meNHF_EndcapMinus_BXm2BXm1Empty->Fill(pt_sum_NHF_Endcap_minus/pfmet->sumEt()); 
	  mePhF_Barrel_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfPhotonEtFractionBarrel_BXm2BXm1Empty"]; if(mePhF_Barrel_BXm2BXm1Empty && mePhF_Barrel_BXm2BXm1Empty->getRootObject()) mePhF_Barrel_BXm2BXm1Empty->Fill(pt_sum_PhF_Barrel/pfmet->sumEt()); 
	  mePhF_EndcapPlus_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfPhotonEtFractionEndcapPlus_BXm2BXm1Empty"]; if(mePhF_EndcapPlus_BXm2BXm1Empty && mePhF_EndcapPlus_BXm2BXm1Empty->getRootObject()) mePhF_EndcapPlus_BXm2BXm1Empty->Fill(pt_sum_PhF_Endcap_plus/pfmet->sumEt()); 
	  mePhF_EndcapMinus_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfPhotonEtFractionEndcapMinus_BXm2BXm1Empty"]; if(mePhF_EndcapMinus_BXm2BXm1Empty && mePhF_EndcapMinus_BXm2BXm1Empty->getRootObject()) mePhF_EndcapMinus_BXm2BXm1Empty->Fill(pt_sum_PhF_Endcap_minus/pfmet->sumEt()); 
	  meHFHadF_Plus_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfHFHadronEtFractionPlus_BXm2BXm1Empty"]; if(meHFHadF_Plus_BXm2BXm1Empty && meHFHadF_Plus_BXm2BXm1Empty->getRootObject()) meHFHadF_Plus_BXm2BXm1Empty->Fill(pt_sum_HFH_plus/pfmet->sumEt()); 
	  meHFHadF_Minus_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfHFHadronEtFractionMinus_BXm2BXm1Empty"]; if(meHFHadF_Minus_BXm2BXm1Empty && meHFHadF_Minus_BXm2BXm1Empty->getRootObject()) meHFHadF_Minus_BXm2BXm1Empty->Fill(pt_sum_HFH_minus/pfmet->sumEt()); 
	  meHFEMF_Plus_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfHFEMEtFractionPlus_BXm2BXm1Empty"]; if(meHFEMF_Plus_BXm2BXm1Empty && meHFEMF_Plus_BXm2BXm1Empty->getRootObject()) meHFEMF_Plus_BXm2BXm1Empty->Fill(pt_sum_HFE_plus/pfmet->sumEt()); 
	  meHFEMF_Minus_BXm2BXm1Empty=map_of_MEs[DirName+"/"+"PfHFEMEtFractionMinus_BXm2BXm1Empty"]; if(meHFEMF_Minus_BXm2BXm1Empty && meHFEMF_Minus_BXm2BXm1Empty->getRootObject()) meHFEMF_Minus_BXm2BXm1Empty->Fill(pt_sum_HFE_minus/pfmet->sumEt());
	  mePhotonEtFraction_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"PfPhotonEtFraction_BXm2BXm1Empty"];     if (  mePhotonEtFraction_BXm2BXm1Empty  && mePhotonEtFraction_BXm2BXm1Empty ->getRootObject())  mePhotonEtFraction_BXm2BXm1Empty  ->Fill(pfmet->photonEtFraction());
	  meNeutralHadronEtFraction_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"PfNeutralHadronEtFraction_BXm2BXm1Empty"];     if (  meNeutralHadronEtFraction_BXm2BXm1Empty  && meNeutralHadronEtFraction_BXm2BXm1Empty ->getRootObject())  meNeutralHadronEtFraction_BXm2BXm1Empty  ->Fill(pfmet->neutralHadronEtFraction());
	  meChargedHadronEtFraction_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"PfChargedHadronEtFraction_BXm2BXm1Empty"];     if (  meChargedHadronEtFraction_BXm2BXm1Empty  && meChargedHadronEtFraction_BXm2BXm1Empty ->getRootObject())  meChargedHadronEtFraction_BXm2BXm1Empty  ->Fill(pfmet->chargedHadronEtFraction());
	  meMET_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"MET_BXm2BXm1Empty"];     if (  meMET_BXm2BXm1Empty  && meMET_BXm2BXm1Empty ->getRootObject())  meMET_BXm2BXm1Empty  ->Fill(pfmet->pt());
	  meSumET_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"SumET_BXm2BXm1Empty"];     if (  meSumET_BXm2BXm1Empty  && meSumET_BXm2BXm1Empty ->getRootObject())  meSumET_BXm2BXm1Empty  ->Fill(pfmet->sumEt());
	  if(pt_sum_CHF_Barrel){
	    meMETPhiChargedHadronsBarrel_BXm2BXm1Empty     = map_of_MEs[DirName+"/"+"METPhiChargedHadronsBarrel_BXm2BXm1Empty"];if(meMETPhiChargedHadronsBarrel_BXm2BXm1Empty  && meMETPhiChargedHadronsBarrel_BXm2BXm1Empty ->getRootObject())meMETPhiChargedHadronsBarrel_BXm2BXm1Empty->Fill(atan2(py_chargedHadronsBarrel,px_chargedHadronsBarrel));
	  }
	  if(pt_sum_CHF_Endcap_plus){
	    meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Empty  = map_of_MEs[DirName+"/"+"METPhiChargedHadronsEndcapPlus_BXm2BXm1Empty"];if(meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Empty  && meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Empty ->getRootObject())meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Empty->Fill(atan2(py_chargedHadronsEndcapPlus,px_chargedHadronsEndcapPlus));
	  }
	  if(pt_sum_CHF_Endcap_minus){
	    meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Empty     = map_of_MEs[DirName+"/"+"METPhiChargedHadronsEndcapMinus_BXm2BXm1Empty"];if(meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Empty  && meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Empty ->getRootObject())meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Empty->Fill(atan2(py_chargedHadronsEndcapMinus,px_chargedHadronsEndcapMinus));
	  }
	  if(pt_sum_NHF_Barrel){
	    meMETPhiNeutralHadronsBarrel_BXm2BXm1Empty     = map_of_MEs[DirName+"/"+"METPhiNeutralHadronsBarrel_BXm2BXm1Empty"];if(meMETPhiNeutralHadronsBarrel_BXm2BXm1Empty  && meMETPhiNeutralHadronsBarrel_BXm2BXm1Empty ->getRootObject())meMETPhiNeutralHadronsBarrel_BXm2BXm1Empty->Fill(atan2(py_neutralHadronsBarrel,px_neutralHadronsBarrel));
	  }
	  if(pt_sum_NHF_Endcap_plus){
	    meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Empty     = map_of_MEs[DirName+"/"+"METPhiNeutralHadronsEndcapPlus_BXm2BXm1Empty"];if(meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Empty  && meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Empty ->getRootObject())meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Empty->Fill(atan2(py_neutralHadronsEndcapPlus,px_neutralHadronsEndcapPlus));
	  }
	  if(pt_sum_NHF_Endcap_minus){
	    meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Empty     = map_of_MEs[DirName+"/"+"METPhiNeutralHadronsEndcapMinus_BXm2BXm1Empty"];if(meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Empty  && meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Empty ->getRootObject())meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Empty->Fill(atan2(py_neutralHadronsEndcapMinus,px_neutralHadronsEndcapMinus));
	  }
	  if(pt_sum_PhF_Barrel){
	    meMETPhiPhotonsBarrel_BXm2BXm1Empty     = map_of_MEs[DirName+"/"+"METPhiPhotonsBarrel_BXm2BXm1Empty"];if(meMETPhiPhotonsBarrel_BXm2BXm1Empty  && meMETPhiPhotonsBarrel_BXm2BXm1Empty ->getRootObject())meMETPhiPhotonsBarrel_BXm2BXm1Empty->Fill(atan2(py_PhotonsBarrel,px_PhotonsBarrel));
	  }
	  if(pt_sum_PhF_Endcap_plus){
	    meMETPhiPhotonsEndcapPlus_BXm2BXm1Empty     = map_of_MEs[DirName+"/"+"METPhiPhotonsEndcapPlus_BXm2BXm1Empty"];if(meMETPhiPhotonsEndcapPlus_BXm2BXm1Empty  && meMETPhiPhotonsEndcapPlus_BXm2BXm1Empty ->getRootObject())meMETPhiPhotonsEndcapPlus_BXm2BXm1Empty->Fill(atan2(py_PhotonsEndcapPlus,px_PhotonsEndcapPlus));
	  }
	  if(pt_sum_PhF_Endcap_minus){
	    meMETPhiPhotonsEndcapMinus_BXm2BXm1Empty     = map_of_MEs[DirName+"/"+"METPhiPhotonsEndcapMinus_BXm2BXm1Empty"];if(meMETPhiPhotonsEndcapMinus_BXm2BXm1Empty  && meMETPhiPhotonsEndcapMinus_BXm2BXm1Empty ->getRootObject())meMETPhiPhotonsEndcapMinus_BXm2BXm1Empty->Fill(atan2(py_PhotonsEndcapMinus,px_PhotonsEndcapMinus));
	  }
	  if(pt_sum_HFH_plus){
	    meMETPhiHFHadronsPlus_BXm2BXm1Empty     = map_of_MEs[DirName+"/"+"METPhiHFHadronsPlus_BXm2BXm1Empty"];if(meMETPhiHFHadronsPlus_BXm2BXm1Empty  && meMETPhiHFHadronsPlus_BXm2BXm1Empty ->getRootObject())meMETPhiHFHadronsPlus_BXm2BXm1Empty->Fill(atan2(py_HFHadronsPlus,px_HFHadronsPlus));
	  }
	  if(pt_sum_HFH_minus){
	    meMETPhiHFHadronsMinus_BXm2BXm1Empty     = map_of_MEs[DirName+"/"+"METPhiHFHadronsMinus_BXm2BXm1Empty"];if(meMETPhiHFHadronsMinus_BXm2BXm1Empty  && meMETPhiHFHadronsMinus_BXm2BXm1Empty ->getRootObject())meMETPhiHFHadronsMinus_BXm2BXm1Empty->Fill(atan2(py_HFHadronsMinus,px_HFHadronsMinus));
	  }
	  if(pt_sum_HFE_plus){
	    meMETPhiHFEGammasPlus_BXm2BXm1Empty     = map_of_MEs[DirName+"/"+"METPhiHFEGammasPlus_BXm2BXm1Empty"];if(meMETPhiHFEGammasPlus_BXm2BXm1Empty  && meMETPhiHFEGammasPlus_BXm2BXm1Empty ->getRootObject())meMETPhiHFEGammasPlus_BXm2BXm1Empty->Fill(atan2(py_HFEGammasPlus,px_HFEGammasPlus));
	  }
	  if(pt_sum_HFE_minus){
	    meMETPhiHFEGammasMinus_BXm2BXm1Empty     = map_of_MEs[DirName+"/"+"METPhiHFEGammasMinus_BXm2BXm1Empty"];if(meMETPhiHFEGammasMinus_BXm2BXm1Empty  && meMETPhiHFEGammasMinus_BXm2BXm1Empty ->getRootObject())meMETPhiHFEGammasMinus_BXm2BXm1Empty->Fill(atan2(py_HFEGammasMinus,px_HFEGammasMinus));
	  }
	  }*/
        if (techTriggerCase[2]) {  //techTriggerResultBx0 && !techTriggerResultBxM1 --> previous bunch empty
          meCHF_Barrel_BXm1Empty = map_of_MEs[DirName + "/" + "PfChargedHadronEtFractionBarrel_BXm1Empty"];
          if (meCHF_Barrel_BXm1Empty && meCHF_Barrel_BXm1Empty->getRootObject())
            meCHF_Barrel_BXm1Empty->Fill(pt_sum_CHF_Barrel / pfmet->sumEt());
          meCHF_EndcapPlus_BXm1Empty = map_of_MEs[DirName + "/" + "PfChargedHadronEtFractionEndcapPlus_BXm1Empty"];
          if (meCHF_EndcapPlus_BXm1Empty && meCHF_EndcapPlus_BXm1Empty->getRootObject())
            meCHF_EndcapPlus_BXm1Empty->Fill(pt_sum_CHF_Endcap_plus / pfmet->sumEt());
          meCHF_EndcapMinus_BXm1Empty = map_of_MEs[DirName + "/" + "PfChargedHadronEtFractionEndcapMinus_BXm1Empty"];
          if (meCHF_EndcapMinus_BXm1Empty && meCHF_EndcapMinus_BXm1Empty->getRootObject())
            meCHF_EndcapMinus_BXm1Empty->Fill(pt_sum_CHF_Endcap_minus / pfmet->sumEt());
          meNHF_Barrel_BXm1Empty = map_of_MEs[DirName + "/" + "PfNeutralHadronEtFractionBarrel_BXm1Empty"];
          if (meNHF_Barrel_BXm1Empty && meNHF_Barrel_BXm1Empty->getRootObject())
            meNHF_Barrel_BXm1Empty->Fill(pt_sum_NHF_Barrel / pfmet->sumEt());
          meNHF_EndcapPlus_BXm1Empty = map_of_MEs[DirName + "/" + "PfNeutralHadronEtFractionEndcapPlus_BXm1Empty"];
          if (meNHF_EndcapPlus_BXm1Empty && meNHF_EndcapPlus_BXm1Empty->getRootObject())
            meNHF_EndcapPlus_BXm1Empty->Fill(pt_sum_NHF_Endcap_plus / pfmet->sumEt());
          meNHF_EndcapMinus_BXm1Empty = map_of_MEs[DirName + "/" + "PfNeutralHadronEtFractionEndcapMinus_BXm1Empty"];
          if (meNHF_EndcapMinus_BXm1Empty && meNHF_EndcapMinus_BXm1Empty->getRootObject())
            meNHF_EndcapMinus_BXm1Empty->Fill(pt_sum_NHF_Endcap_minus / pfmet->sumEt());
          mePhF_Barrel_BXm1Empty = map_of_MEs[DirName + "/" + "PfPhotonEtFractionBarrel_BXm1Empty"];
          if (mePhF_Barrel_BXm1Empty && mePhF_Barrel_BXm1Empty->getRootObject())
            mePhF_Barrel_BXm1Empty->Fill(pt_sum_PhF_Barrel / pfmet->sumEt());
          mePhF_EndcapPlus_BXm1Empty = map_of_MEs[DirName + "/" + "PfPhotonEtFractionEndcapPlus_BXm1Empty"];
          if (mePhF_EndcapPlus_BXm1Empty && mePhF_EndcapPlus_BXm1Empty->getRootObject())
            mePhF_EndcapPlus_BXm1Empty->Fill(pt_sum_PhF_Endcap_plus / pfmet->sumEt());
          mePhF_EndcapMinus_BXm1Empty = map_of_MEs[DirName + "/" + "PfPhotonEtFractionEndcapMinus_BXm1Empty"];
          if (mePhF_EndcapMinus_BXm1Empty && mePhF_EndcapMinus_BXm1Empty->getRootObject())
            mePhF_EndcapMinus_BXm1Empty->Fill(pt_sum_PhF_Endcap_minus / pfmet->sumEt());
          meHFHadF_Plus_BXm1Empty = map_of_MEs[DirName + "/" + "PfHFHadronEtFractionPlus_BXm1Empty"];
          if (meHFHadF_Plus_BXm1Empty && meHFHadF_Plus_BXm1Empty->getRootObject())
            meHFHadF_Plus_BXm1Empty->Fill(pt_sum_HFH_plus / pfmet->sumEt());
          meHFHadF_Minus_BXm1Empty = map_of_MEs[DirName + "/" + "PfHFHadronEtFractionMinus_BXm1Empty"];
          if (meHFHadF_Minus_BXm1Empty && meHFHadF_Minus_BXm1Empty->getRootObject())
            meHFHadF_Minus_BXm1Empty->Fill(pt_sum_HFH_minus / pfmet->sumEt());
          meHFEMF_Plus_BXm1Empty = map_of_MEs[DirName + "/" + "PfHFEMEtFractionPlus_BXm1Empty"];
          if (meHFEMF_Plus_BXm1Empty && meHFEMF_Plus_BXm1Empty->getRootObject())
            meHFEMF_Plus_BXm1Empty->Fill(pt_sum_HFE_plus / pfmet->sumEt());
          meHFEMF_Minus_BXm1Empty = map_of_MEs[DirName + "/" + "PfHFEMEtFractionMinus_BXm1Empty"];
          if (meHFEMF_Minus_BXm1Empty && meHFEMF_Minus_BXm1Empty->getRootObject())
            meHFEMF_Minus_BXm1Empty->Fill(pt_sum_HFE_minus / pfmet->sumEt());
          mePhotonEtFraction_BXm1Empty = map_of_MEs[DirName + "/" + "PfPhotonEtFraction_BXm1Empty"];
          if (mePhotonEtFraction_BXm1Empty && mePhotonEtFraction_BXm1Empty->getRootObject())
            mePhotonEtFraction_BXm1Empty->Fill(pfmet->photonEtFraction());
          meNeutralHadronEtFraction_BXm1Empty = map_of_MEs[DirName + "/" + "PfNeutralHadronEtFraction_BXm1Empty"];
          if (meNeutralHadronEtFraction_BXm1Empty && meNeutralHadronEtFraction_BXm1Empty->getRootObject())
            meNeutralHadronEtFraction_BXm1Empty->Fill(pfmet->neutralHadronEtFraction());
          meChargedHadronEtFraction_BXm1Empty = map_of_MEs[DirName + "/" + "PfChargedHadronEtFraction_BXm1Empty"];
          if (meChargedHadronEtFraction_BXm1Empty && meChargedHadronEtFraction_BXm1Empty->getRootObject())
            meChargedHadronEtFraction_BXm1Empty->Fill(pfmet->chargedHadronEtFraction());
          meMET_BXm1Empty = map_of_MEs[DirName + "/" + "MET_BXm1Empty"];
          if (meMET_BXm1Empty && meMET_BXm1Empty->getRootObject())
            meMET_BXm1Empty->Fill(pfmet->pt());
          meSumET_BXm1Empty = map_of_MEs[DirName + "/" + "SumET_BXm1Empty"];
          if (meSumET_BXm1Empty && meSumET_BXm1Empty->getRootObject())
            meSumET_BXm1Empty->Fill(pfmet->sumEt());
          if (pt_sum_CHF_Barrel) {
            meMETPhiChargedHadronsBarrel_BXm1Empty = map_of_MEs[DirName + "/" + "METPhiChargedHadronsBarrel_BXm1Empty"];
            if (meMETPhiChargedHadronsBarrel_BXm1Empty && meMETPhiChargedHadronsBarrel_BXm1Empty->getRootObject())
              meMETPhiChargedHadronsBarrel_BXm1Empty->Fill(atan2(py_chargedHadronsBarrel, px_chargedHadronsBarrel));
          }
          if (pt_sum_CHF_Endcap_plus) {
            meMETPhiChargedHadronsEndcapPlus_BXm1Empty =
                map_of_MEs[DirName + "/" + "METPhiChargedHadronsEndcapPlus_BXm1Empty"];
            if (meMETPhiChargedHadronsEndcapPlus_BXm1Empty &&
                meMETPhiChargedHadronsEndcapPlus_BXm1Empty->getRootObject())
              meMETPhiChargedHadronsEndcapPlus_BXm1Empty->Fill(
                  atan2(py_chargedHadronsEndcapPlus, px_chargedHadronsEndcapPlus));
          }
          if (pt_sum_CHF_Endcap_minus) {
            meMETPhiChargedHadronsEndcapMinus_BXm1Empty =
                map_of_MEs[DirName + "/" + "METPhiChargedHadronsEndcapMinus_BXm1Empty"];
            if (meMETPhiChargedHadronsEndcapMinus_BXm1Empty &&
                meMETPhiChargedHadronsEndcapMinus_BXm1Empty->getRootObject())
              meMETPhiChargedHadronsEndcapMinus_BXm1Empty->Fill(
                  atan2(py_chargedHadronsEndcapMinus, px_chargedHadronsEndcapMinus));
          }
          if (pt_sum_NHF_Barrel) {
            meMETPhiNeutralHadronsBarrel_BXm1Empty = map_of_MEs[DirName + "/" + "METPhiNeutralHadronsBarrel_BXm1Empty"];
            if (meMETPhiNeutralHadronsBarrel_BXm1Empty && meMETPhiNeutralHadronsBarrel_BXm1Empty->getRootObject())
              meMETPhiNeutralHadronsBarrel_BXm1Empty->Fill(atan2(py_neutralHadronsBarrel, px_neutralHadronsBarrel));
          }
          if (pt_sum_NHF_Endcap_plus) {
            meMETPhiNeutralHadronsEndcapPlus_BXm1Empty =
                map_of_MEs[DirName + "/" + "METPhiNeutralHadronsEndcapPlus_BXm1Empty"];
            if (meMETPhiNeutralHadronsEndcapPlus_BXm1Empty &&
                meMETPhiNeutralHadronsEndcapPlus_BXm1Empty->getRootObject())
              meMETPhiNeutralHadronsEndcapPlus_BXm1Empty->Fill(
                  atan2(py_neutralHadronsEndcapPlus, px_neutralHadronsEndcapPlus));
          }
          if (pt_sum_NHF_Endcap_minus) {
            meMETPhiNeutralHadronsEndcapMinus_BXm1Empty =
                map_of_MEs[DirName + "/" + "METPhiNeutralHadronsEndcapMinus_BXm1Empty"];
            if (meMETPhiNeutralHadronsEndcapMinus_BXm1Empty &&
                meMETPhiNeutralHadronsEndcapMinus_BXm1Empty->getRootObject())
              meMETPhiNeutralHadronsEndcapMinus_BXm1Empty->Fill(
                  atan2(py_neutralHadronsEndcapMinus, px_neutralHadronsEndcapMinus));
          }
          if (pt_sum_PhF_Barrel) {
            meMETPhiPhotonsBarrel_BXm1Empty = map_of_MEs[DirName + "/" + "METPhiPhotonsBarrel_BXm1Empty"];
            if (meMETPhiPhotonsBarrel_BXm1Empty && meMETPhiPhotonsBarrel_BXm1Empty->getRootObject())
              meMETPhiPhotonsBarrel_BXm1Empty->Fill(atan2(py_PhotonsBarrel, px_PhotonsBarrel));
          }
          if (pt_sum_PhF_Endcap_plus) {
            meMETPhiPhotonsEndcapPlus_BXm1Empty = map_of_MEs[DirName + "/" + "METPhiPhotonsEndcapPlus_BXm1Empty"];
            if (meMETPhiPhotonsEndcapPlus_BXm1Empty && meMETPhiPhotonsEndcapPlus_BXm1Empty->getRootObject())
              meMETPhiPhotonsEndcapPlus_BXm1Empty->Fill(atan2(py_PhotonsEndcapPlus, px_PhotonsEndcapPlus));
          }
          if (pt_sum_PhF_Endcap_minus) {
            meMETPhiPhotonsEndcapMinus_BXm1Empty = map_of_MEs[DirName + "/" + "METPhiPhotonsEndcapMinus_BXm1Empty"];
            if (meMETPhiPhotonsEndcapMinus_BXm1Empty && meMETPhiPhotonsEndcapMinus_BXm1Empty->getRootObject())
              meMETPhiPhotonsEndcapMinus_BXm1Empty->Fill(atan2(py_PhotonsEndcapMinus, px_PhotonsEndcapMinus));
          }
          if (pt_sum_HFH_plus) {
            meMETPhiHFHadronsPlus_BXm1Empty = map_of_MEs[DirName + "/" + "METPhiHFHadronsPlus_BXm1Empty"];
            if (meMETPhiHFHadronsPlus_BXm1Empty && meMETPhiHFHadronsPlus_BXm1Empty->getRootObject())
              meMETPhiHFHadronsPlus_BXm1Empty->Fill(atan2(py_HFHadronsPlus, px_HFHadronsPlus));
          }
          if (pt_sum_HFH_minus) {
            meMETPhiHFHadronsMinus_BXm1Empty = map_of_MEs[DirName + "/" + "METPhiHFHadronsMinus_BXm1Empty"];
            if (meMETPhiHFHadronsMinus_BXm1Empty && meMETPhiHFHadronsMinus_BXm1Empty->getRootObject())
              meMETPhiHFHadronsMinus_BXm1Empty->Fill(atan2(py_HFHadronsMinus, px_HFHadronsMinus));
          }
          if (pt_sum_HFE_plus) {
            meMETPhiHFEGammasPlus_BXm1Empty = map_of_MEs[DirName + "/" + "METPhiHFEGammasPlus_BXm1Empty"];
            if (meMETPhiHFEGammasPlus_BXm1Empty && meMETPhiHFEGammasPlus_BXm1Empty->getRootObject())
              meMETPhiHFEGammasPlus_BXm1Empty->Fill(atan2(py_HFEGammasPlus, px_HFEGammasPlus));
          }
          if (pt_sum_HFE_minus) {
            meMETPhiHFEGammasMinus_BXm1Empty = map_of_MEs[DirName + "/" + "METPhiHFEGammasMinus_BXm1Empty"];
            if (meMETPhiHFEGammasMinus_BXm1Empty && meMETPhiHFEGammasMinus_BXm1Empty->getRootObject())
              meMETPhiHFEGammasMinus_BXm1Empty->Fill(atan2(py_HFEGammasMinus, px_HFEGammasMinus));
          }
        }

      }  //fill candidate plots only then

      // PFMET getters
      //----------------------------------------------------------------------------
      double pfPhotonEtFraction = pfmet->photonEtFraction();
      double pfPhotonEt = pfmet->photonEt();
      double pfNeutralHadronEtFraction = pfmet->neutralHadronEtFraction();
      double pfNeutralHadronEt = pfmet->neutralHadronEt();
      double pfElectronEt = pfmet->electronEt();
      double pfChargedHadronEtFraction = pfmet->chargedHadronEtFraction();
      double pfChargedHadronEt = pfmet->chargedHadronEt();
      double pfMuonEt = pfmet->muonEt();
      double pfHFHadronEtFraction = pfmet->HFHadronEtFraction();
      double pfHFHadronEt = pfmet->HFHadronEt();
      double pfHFEMEtFraction = pfmet->HFEMEtFraction();
      double pfHFEMEt = pfmet->HFEMEt();
      mePhotonEtFraction = map_of_MEs[DirName + "/PfPhotonEtFraction"];
      mePhotonEt = map_of_MEs[DirName + "/PfPhotonEt"];
      meNeutralHadronEtFraction = map_of_MEs[DirName + "/PfNeutralHadronEtFraction"];
      meNeutralHadronEt = map_of_MEs[DirName + "/PfNeutralHadronEt"];
      meElectronEt = map_of_MEs[DirName + "/PfElectronEt"];
      meChargedHadronEtFraction = map_of_MEs[DirName + "/PfChargedHadronEtFraction"];
      meChargedHadronEt = map_of_MEs[DirName + "/PfChargedHadronEt"];
      meMuonEt = map_of_MEs[DirName + "/PfMuonEt"];
      meHFHadronEtFraction = map_of_MEs[DirName + "/PfHFHadronEtFraction"];
      meHFHadronEt = map_of_MEs[DirName + "/PfHFHadronEt"];
      meHFEMEtFraction = map_of_MEs[DirName + "/PfHFEMEtFraction"];
      meHFEMEt = map_of_MEs[DirName + "/PfHFEMEt"];

      if (mePhotonEtFraction && mePhotonEtFraction->getRootObject())
        mePhotonEtFraction->Fill(pfPhotonEtFraction);
      if (mePhotonEt && mePhotonEt->getRootObject())
        mePhotonEt->Fill(pfPhotonEt);
      if (meNeutralHadronEtFraction && meNeutralHadronEtFraction->getRootObject())
        meNeutralHadronEtFraction->Fill(pfNeutralHadronEtFraction);
      if (meNeutralHadronEt && meNeutralHadronEt->getRootObject())
        meNeutralHadronEt->Fill(pfNeutralHadronEt);
      if (meElectronEt && meElectronEt->getRootObject())
        meElectronEt->Fill(pfElectronEt);
      if (meChargedHadronEtFraction && meChargedHadronEtFraction->getRootObject())
        meChargedHadronEtFraction->Fill(pfChargedHadronEtFraction);
      if (meChargedHadronEt && meChargedHadronEt->getRootObject())
        meChargedHadronEt->Fill(pfChargedHadronEt);
      if (meMuonEt && meMuonEt->getRootObject())
        meMuonEt->Fill(pfMuonEt);
      if (meHFHadronEtFraction && meHFHadronEtFraction->getRootObject())
        meHFHadronEtFraction->Fill(pfHFHadronEtFraction);
      if (meHFHadronEt && meHFHadronEt->getRootObject())
        meHFHadronEt->Fill(pfHFHadronEt);
      if (meHFEMEtFraction && meHFEMEtFraction->getRootObject())
        meHFEMEtFraction->Fill(pfHFEMEtFraction);
      if (meHFEMEt && meHFEMEt->getRootObject())
        meHFEMEt->Fill(pfHFEMEt);

      //NPV profiles

      mePhotonEtFraction_profile = map_of_MEs[DirName + "/PfPhotonEtFraction_profile"];
      mePhotonEt_profile = map_of_MEs[DirName + "/PfPhotonEt_profile"];
      meNeutralHadronEtFraction_profile = map_of_MEs[DirName + "/PfNeutralHadronEtFraction_profile"];
      meNeutralHadronEt_profile = map_of_MEs[DirName + "/PfNeutralHadronEt_profile"];
      meChargedHadronEtFraction_profile = map_of_MEs[DirName + "/PfChargedHadronEtFraction_profile"];
      meChargedHadronEt_profile = map_of_MEs[DirName + "/PfChargedHadronEt_profile"];
      meHFHadronEtFraction_profile = map_of_MEs[DirName + "/PfHFHadronEtFraction_profile"];
      meHFHadronEt_profile = map_of_MEs[DirName + "/PfHFHadronEt_profile"];
      meHFEMEtFraction_profile = map_of_MEs[DirName + "/PfHFEMEtFraction_profile"];
      meHFEMEt_profile = map_of_MEs[DirName + "/PfHFEMEt_profile"];

      if (mePhotonEtFraction_profile && mePhotonEtFraction_profile->getRootObject())
        mePhotonEtFraction_profile->Fill(numPV_, pfPhotonEtFraction);
      if (mePhotonEt_profile && mePhotonEt_profile->getRootObject())
        mePhotonEt_profile->Fill(numPV_, pfPhotonEt);
      if (meNeutralHadronEtFraction_profile && meNeutralHadronEtFraction_profile->getRootObject())
        meNeutralHadronEtFraction_profile->Fill(numPV_, pfNeutralHadronEtFraction);
      if (meNeutralHadronEt_profile && meNeutralHadronEt_profile->getRootObject())
        meNeutralHadronEt_profile->Fill(numPV_, pfNeutralHadronEt);
      if (meChargedHadronEtFraction_profile && meChargedHadronEtFraction_profile->getRootObject())
        meChargedHadronEtFraction_profile->Fill(numPV_, pfChargedHadronEtFraction);
      if (meChargedHadronEt_profile && meChargedHadronEt_profile->getRootObject())
        meChargedHadronEt_profile->Fill(numPV_, pfChargedHadronEt);
      if (meHFHadronEtFraction_profile && meHFHadronEtFraction_profile->getRootObject())
        meHFHadronEtFraction_profile->Fill(numPV_, pfHFHadronEtFraction);
      if (meHFHadronEt_profile && meHFHadronEt_profile->getRootObject())
        meHFHadronEt_profile->Fill(numPV_, pfHFHadronEt);
      if (meHFEMEtFraction_profile && meHFEMEtFraction_profile->getRootObject())
        meHFEMEtFraction_profile->Fill(numPV_, pfHFEMEtFraction);
      if (meHFEMEt_profile && meHFEMEt_profile->getRootObject())
        meHFEMEt_profile->Fill(numPV_, pfHFEMEt);
    }

    if (isMiniAODMet_) {
      mePhotonEtFraction = map_of_MEs[DirName + "/PfPhotonEtFraction"];
      meNeutralHadronEtFraction = map_of_MEs[DirName + "/PfNeutralHadronEtFraction"];
      meChargedHadronEtFraction = map_of_MEs[DirName + "/PfChargedHadronEtFraction"];
      meHFHadronEtFraction = map_of_MEs[DirName + "/PfHFHadronEtFraction"];
      meHFEMEtFraction = map_of_MEs[DirName + "/PfHFEMEtFraction"];

      if (mePhotonEtFraction && mePhotonEtFraction->getRootObject())
        mePhotonEtFraction->Fill(patmet->NeutralEMFraction());
      if (meNeutralHadronEtFraction && meNeutralHadronEtFraction->getRootObject())
        meNeutralHadronEtFraction->Fill(patmet->NeutralHadEtFraction());
      if (meChargedHadronEtFraction && meChargedHadronEtFraction->getRootObject())
        meChargedHadronEtFraction->Fill(patmet->ChargedHadEtFraction());
      if (meHFHadronEtFraction && meHFHadronEtFraction->getRootObject())
        meHFHadronEtFraction->Fill(patmet->Type6EtFraction());  //HFHadrons
      if (meHFEMEtFraction && meHFEMEtFraction->getRootObject())
        meHFEMEtFraction->Fill(patmet->Type7EtFraction());

      //NPV profiles
      mePhotonEtFraction_profile = map_of_MEs[DirName + "/PfPhotonEtFraction_profile"];
      meNeutralHadronEtFraction_profile = map_of_MEs[DirName + "/PfNeutralHadronEtFraction_profile"];
      meChargedHadronEtFraction_profile = map_of_MEs[DirName + "/PfChargedHadronEtFraction_profile"];
      meHFHadronEtFraction_profile = map_of_MEs[DirName + "/PfHFHadronEtFraction_profile"];
      meHFEMEtFraction_profile = map_of_MEs[DirName + "/PfHFEMEtFraction_profile"];

      if (mePhotonEtFraction_profile && mePhotonEtFraction_profile->getRootObject())
        mePhotonEtFraction_profile->Fill(numPV_, patmet->NeutralEMFraction());
      if (meNeutralHadronEtFraction_profile && meNeutralHadronEtFraction_profile->getRootObject())
        meNeutralHadronEtFraction_profile->Fill(numPV_, patmet->NeutralHadEtFraction());
      if (meChargedHadronEtFraction_profile && meChargedHadronEtFraction_profile->getRootObject())
        meChargedHadronEtFraction_profile->Fill(numPV_, patmet->ChargedHadEtFraction());
      if (meHFHadronEtFraction_profile && meHFHadronEtFraction_profile->getRootObject())
        meHFHadronEtFraction_profile->Fill(numPV_, patmet->Type6EtFraction());
      if (meHFEMEtFraction_profile && meHFEMEtFraction_profile->getRootObject())
        meHFEMEtFraction_profile->Fill(numPV_, patmet->Type7EtFraction());

      mePhotonEt = map_of_MEs[DirName + "/PfPhotonEt"];
      meNeutralHadronEt = map_of_MEs[DirName + "/PfNeutralHadronEt"];
      meChargedHadronEt = map_of_MEs[DirName + "/PfChargedHadronEt"];
      meHFHadronEt = map_of_MEs[DirName + "/PfHFHadronEt"];
      meHFEMEt = map_of_MEs[DirName + "/PfHFEMEt"];
      meMuonEt = map_of_MEs[DirName + "/PfMuonEt"];
      meElectronEt = map_of_MEs[DirName + "/PfElectronEt"];

      if (mePhotonEt && mePhotonEt->getRootObject())
        mePhotonEt->Fill(patmet->NeutralEMFraction() * patmet->sumEt());
      if (meNeutralHadronEt && meNeutralHadronEt->getRootObject())
        meNeutralHadronEt->Fill(patmet->NeutralHadEtFraction() * patmet->sumEt());
      if (meChargedHadronEt && meChargedHadronEt->getRootObject())
        meChargedHadronEt->Fill(patmet->ChargedHadEtFraction() * patmet->sumEt());
      if (meHFHadronEt && meHFHadronEt->getRootObject())
        meHFHadronEt->Fill(patmet->Type6EtFraction() * patmet->sumEt());  //HFHadrons
      if (meHFEMEt && meHFEMEt->getRootObject())
        meHFEMEt->Fill(patmet->Type7EtFraction() * patmet->sumEt());
      if (meMuonEt && meMuonEt->getRootObject())
        meMuonEt->Fill(patmet->MuonEtFraction() * patmet->sumEt());
      if (meElectronEt && meElectronEt->getRootObject())
        meElectronEt->Fill(patmet->ChargedEMEtFraction() * patmet->sumEt());

      //NPV profiles
      mePhotonEt_profile = map_of_MEs[DirName + "/PfPhotonEt_profile"];
      meNeutralHadronEt_profile = map_of_MEs[DirName + "/PfNeutralHadronEt_profile"];
      meChargedHadronEt_profile = map_of_MEs[DirName + "/PfChargedHadronEt_profile"];
      meHFHadronEt_profile = map_of_MEs[DirName + "/PfHFHadronEt_profile"];
      meHFEMEt_profile = map_of_MEs[DirName + "/PfHFEMEt_profile"];

      if (mePhotonEt_profile && mePhotonEt_profile->getRootObject())
        mePhotonEt_profile->Fill(numPV_, patmet->NeutralEMFraction() * patmet->sumEt());
      if (meNeutralHadronEt_profile && meNeutralHadronEt_profile->getRootObject())
        meNeutralHadronEt_profile->Fill(numPV_, patmet->NeutralHadEtFraction() * patmet->sumEt());
      if (meChargedHadronEt_profile && meChargedHadronEt_profile->getRootObject())
        meChargedHadronEt_profile->Fill(numPV_, patmet->ChargedHadEtFraction() * patmet->sumEt());
      if (meHFHadronEt_profile && meHFHadronEt_profile->getRootObject())
        meHFHadronEt_profile->Fill(numPV_, patmet->Type6EtFraction() * patmet->sumEt());
      if (meHFEMEt_profile && meHFEMEt_profile->getRootObject())
        meHFEMEt_profile->Fill(numPV_, patmet->Type7EtFraction() * patmet->sumEt());
    }

    if (isCaloMet_) {
      //if (bLumiSecPlot){//get from config level
      if (fill_met_high_level_histo) {
        hMExLS = map_of_MEs[DirName + "/" + "MExLS"];
        if (hMExLS && hMExLS->getRootObject())
          hMExLS->Fill(MEx, myLuminosityBlock);
        hMEyLS = map_of_MEs[DirName + "/" + "MEyLS"];
        if (hMEyLS && hMEyLS->getRootObject())
          hMEyLS->Fill(MEy, myLuminosityBlock);
      }
    }
  }  //check if we only wanna do Z plots
}
