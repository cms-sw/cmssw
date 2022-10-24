// -*- C++ -*-
//
// Package: 	GctErrorAnalyzer
// Class: 	GctErrorAnalyzer
//
/**\class GctErrorAnalyzer GctErrorAnalyzer.cc L1Trigger/L1GctAnalyzer/src/GctErrorAnalyzer.cc

Description: Tool to debug the GCT with useful output

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Jad Marrouche
//         Created:  Wed May 20 14:19:23 CEST 2009
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//TFile maker include
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//ROOT includes
#include "TH1.h"
#include "TH2.h"
#include "TAxis.h"
//RCT and GCT DataFormat Collections
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
//GctErrorAnalyzer includes
#include "L1Trigger/L1GctAnalyzer/interface/GctErrorAnalyzerDefinitions.h"
#include "L1Trigger/L1GctAnalyzer/interface/compareCands.h"
#include "L1Trigger/L1GctAnalyzer/interface/compareRingSums.h"
#include "L1Trigger/L1GctAnalyzer/interface/compareBitCounts.h"
#include "L1Trigger/L1GctAnalyzer/interface/compareTotalEnergySums.h"
#include "L1Trigger/L1GctAnalyzer/interface/compareMissingEnergySums.h"
//STL includes
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

//
// class declaration
//

class GctErrorAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  GctErrorAnalyzer() = delete;
  GctErrorAnalyzer(const GctErrorAnalyzer &) = delete;
  GctErrorAnalyzer operator=(const GctErrorAnalyzer &) = delete;

private:
  void plotRCTRegions(const edm::Handle<L1CaloRegionCollection> &caloRegions);
  void plotIsoEm(const edm::Handle<L1GctEmCandCollection> &data, const edm::Handle<L1GctEmCandCollection> &emu);
  void plotNonIsoEm(const edm::Handle<L1GctEmCandCollection> &data, const edm::Handle<L1GctEmCandCollection> &emu);
  void plotEGErrors(const edm::Handle<L1GctEmCandCollection> &dataiso,
                    const edm::Handle<L1GctEmCandCollection> &emuiso,
                    const edm::Handle<L1GctEmCandCollection> &datanoniso,
                    const edm::Handle<L1GctEmCandCollection> &emunoniso,
                    const edm::Handle<L1CaloEmCollection> &regions);
  void plotCenJets(const edm::Handle<L1GctJetCandCollection> &data, const edm::Handle<L1GctJetCandCollection> &emu);
  void plotTauJets(const edm::Handle<L1GctJetCandCollection> &data, const edm::Handle<L1GctJetCandCollection> &emu);
  void plotForJets(const edm::Handle<L1GctJetCandCollection> &data, const edm::Handle<L1GctJetCandCollection> &emu);
  void plotIntJets(const edm::Handle<L1GctInternJetDataCollection> &emu);
  static bool sortJets(const jetData &jet1,
                       const jetData &jet2);  //define this as static as it doesn't need a GctErrorAnalyzer object
  void plotJetErrors(const edm::Handle<L1GctJetCandCollection> &cendata,
                     const edm::Handle<L1GctJetCandCollection> &cenemu,
                     const edm::Handle<L1GctJetCandCollection> &taudata,
                     const edm::Handle<L1GctJetCandCollection> &tauemu,
                     const edm::Handle<L1GctJetCandCollection> &fordata,
                     const edm::Handle<L1GctJetCandCollection> &foremu,
                     const edm::Handle<L1CaloRegionCollection> &regions);
  void plotHFRingSums(const edm::Handle<L1GctHFRingEtSumsCollection> &data,
                      const edm::Handle<L1GctHFRingEtSumsCollection> &emu);
  void plotHFBitCounts(const edm::Handle<L1GctHFBitCountsCollection> &hfBitCountsD,
                       const edm::Handle<L1GctHFBitCountsCollection> &hfBitCountsE);
  void plotHFErrors(const edm::Handle<L1GctHFRingEtSumsCollection> &hfRingSumsD,
                    const edm::Handle<L1GctHFRingEtSumsCollection> &hfRingSumsE,
                    const edm::Handle<L1GctHFBitCountsCollection> &hfBitCountsD,
                    const edm::Handle<L1GctHFBitCountsCollection> &hfBitCountsE,
                    const edm::Handle<L1CaloRegionCollection> &caloRegions);
  void plotTotalE(const edm::Handle<L1GctEtTotalCollection> &totalEtD,
                  const edm::Handle<L1GctEtTotalCollection> &totalEtE);
  void plotTotalH(const edm::Handle<L1GctEtHadCollection> &totalHtD, const edm::Handle<L1GctEtHadCollection> &totalHtE);
  void plotTotalEErrors(const edm::Handle<L1GctEtTotalCollection> &totalEtD,
                        const edm::Handle<L1GctEtTotalCollection> &totalEtE,
                        const edm::Handle<L1GctEtHadCollection> &totalHtD,
                        const edm::Handle<L1GctEtHadCollection> &totalHtE,
                        const edm::Handle<L1CaloRegionCollection> &caloRegions);
  void plotMissingEt(const edm::Handle<L1GctEtMissCollection> &missingEtD,
                     const edm::Handle<L1GctEtMissCollection> &missingEtE);
  void plotMissingHt(const edm::Handle<L1GctHtMissCollection> &missingHtD,
                     const edm::Handle<L1GctHtMissCollection> &missingHtE);
  void plotMissingEErrors(const edm::Handle<L1GctEtMissCollection> &missingEtD,
                          const edm::Handle<L1GctEtMissCollection> &missingEtE,
                          const edm::Handle<L1GctHtMissCollection> &missingHtD,
                          const edm::Handle<L1GctHtMissCollection> &missingHtE,
                          edm::Handle<L1CaloRegionCollection> &caloRegions,
                          const edm::Handle<L1GctInternJetDataCollection> &intjetsemu,
                          const edm::Handle<L1GctInternHtMissCollection> intMissingHtD);
  template <class T>
  bool checkCollections(const T &collection, const unsigned int &constraint, const std::string &label);

public:
  explicit GctErrorAnalyzer(const edm::ParameterSet &);
  ~GctErrorAnalyzer() override;

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  // ----------member data ---------------------------
  //the following flags select what we'd like to plot and whether or not we want error information
  bool doRCT_;
  bool doEg_;
  bool doIsoDebug_;
  bool doNonIsoDebug_;
  bool doJets_;
  bool doCenJetsDebug_;
  bool doTauJetsDebug_;
  bool doForJetsDebug_;
  bool doHF_;
  bool doRingSumDebug_;
  bool doBitCountDebug_;
  bool doTotalEnergySums_;
  bool doTotalEtDebug_;
  bool doTotalHtDebug_;
  bool doMissingEnergySums_;
  bool doMissingETDebug_;
  bool doMissingHTDebug_;
  bool doExtraMissingHTDebug_;
  //the following flags configure whether or not we want multiple BX behaviour for
  //1. RCT regions
  //2. Emulator output
  //3. Hardware output
  bool doRCTMBx_;
  bool doEmuMBx_;
  bool doGCTMBx_;
  //the following values select the definition of the "triggered" Bx i.e. where to define Bx=0
  int RCTTrigBx_;
  int EmuTrigBx_;
  int GCTTrigBx_;
  //the following flags contain the location of the hardware and emulator digis
  edm::InputTag dataTag_;
  edm::InputTag emuTag_;
  //the following is a string which dictates whether or not we want to use the lab or full system parameters
  std::string useSys_;

  //the following declares a struct to hold the MBX Info to make it easy to pass the information around
  GctErrorAnalyzerMBxInfo MBxInfo;

  // histograms
  //RCT Regions
  TH2I *RCT_EtEtaPhi_, *RCT_TvEtaPhi_, *RCT_FgEtaPhi_, *RCT_OfEtaPhi_;
  //isoEg
  TH1I *isoEgD_Rank_, *isoEgE_Rank_;
  TH2I *isoEgD_EtEtaPhi_, *isoEgE_EtEtaPhi_;
  TH2I *isoEgD_OccEtaPhi_, *isoEgE_OccEtaPhi_;
  TH1I *isoEg_errorFlag_;
  //Global Error Histograms
  TH1I *isoEgD_GlobalError_Rank_;
  TH1I *isoEgE_GlobalError_Rank_;
  TH2I *isoEgD_GlobalError_EtEtaPhi_;
  TH2I *isoEgE_GlobalError_EtEtaPhi_;
  //nonIsoEg
  TH1I *nonIsoEgD_Rank_, *nonIsoEgE_Rank_;
  TH2I *nonIsoEgD_EtEtaPhi_, *nonIsoEgE_EtEtaPhi_;
  TH2I *nonIsoEgD_OccEtaPhi_, *nonIsoEgE_OccEtaPhi_;
  TH1I *nonIsoEg_errorFlag_;
  //Global Error Histograms
  TH1I *nonIsoEgD_GlobalError_Rank_;
  TH1I *nonIsoEgE_GlobalError_Rank_;
  TH2I *nonIsoEgD_GlobalError_EtEtaPhi_;
  TH2I *nonIsoEgE_GlobalError_EtEtaPhi_;
  //cenJet
  TH1I *cenJetD_Rank_, *cenJetE_Rank_;
  TH2I *cenJetD_EtEtaPhi_, *cenJetE_EtEtaPhi_;
  TH2I *cenJetD_OccEtaPhi_, *cenJetE_OccEtaPhi_;
  TH1I *cenJet_errorFlag_;
  //Global Error Histograms
  TH1I *cenJetD_GlobalError_Rank_;
  TH1I *cenJetE_GlobalError_Rank_;
  TH2I *cenJetD_GlobalError_EtEtaPhi_;
  TH2I *cenJetE_GlobalError_EtEtaPhi_;
  //tauJet
  TH1I *tauJetD_Rank_, *tauJetE_Rank_;
  TH2I *tauJetD_EtEtaPhi_, *tauJetE_EtEtaPhi_;
  TH2I *tauJetD_OccEtaPhi_, *tauJetE_OccEtaPhi_;
  TH1I *tauJet_errorFlag_;
  //Global Error Histograms
  TH1I *tauJetD_GlobalError_Rank_;
  TH1I *tauJetE_GlobalError_Rank_;
  TH2I *tauJetD_GlobalError_EtEtaPhi_;
  TH2I *tauJetE_GlobalError_EtEtaPhi_;
  //forJet
  TH1I *forJetD_Rank_, *forJetE_Rank_;
  TH2I *forJetD_EtEtaPhi_, *forJetE_EtEtaPhi_;
  TH2I *forJetD_OccEtaPhi_, *forJetE_OccEtaPhi_;
  TH1I *forJet_errorFlag_;
  //Global Error Histograms
  TH1I *forJetD_GlobalError_Rank_;
  TH1I *forJetE_GlobalError_Rank_;
  TH2I *forJetD_GlobalError_EtEtaPhi_;
  TH2I *forJetE_GlobalError_EtEtaPhi_;
  //intJet
  TH2I *intJetEtEtaPhiE_;
  TH1I *intJetE_Et_;
  TH1I *intJetE_Of_;
  TH1I *intJetE_Jet1Et_;
  TH1I *intJetE_Jet2Et_;
  TH1I *intJetE_Jet3Et_;
  TH1I *intJetE_Jet4Et_;
  //ringSums
  TH1I *hfRingSumD_1pos_, *hfRingSumD_1neg_, *hfRingSumD_2pos_, *hfRingSumD_2neg_;
  TH1I *hfRingSumE_1pos_, *hfRingSumE_1neg_, *hfRingSumE_2pos_, *hfRingSumE_2neg_;
  TH1I *hfRingSum_errorFlag_;
  //bitcounts
  TH1I *hfBitCountD_1pos_, *hfBitCountD_1neg_, *hfBitCountD_2pos_, *hfBitCountD_2neg_;
  TH1I *hfBitCountE_1pos_, *hfBitCountE_1neg_, *hfBitCountE_2pos_, *hfBitCountE_2neg_;
  TH1I *hfBitCount_errorFlag_;
  //totalEt
  TH1I *totalEtD_, *totalEtE_;
  TH1I *totalEtD_Of_, *totalEtE_Of_;
  TH1I *totalEt_errorFlag_;
  //ET GlobalError Histograms
  //TH1I *totalEtD_GlobalError_, *totalEtE_GlobalError_;
  //TH1I *totalEtD_GlobalError_Of_, *totalEtE_GlobalError_Of_;
  //totalHt
  TH1I *totalHtD_, *totalHtE_;
  TH1I *totalHtD_Of_, *totalHtE_Of_;
  TH1I *totalHt_errorFlag_;
  //HT GlobalError Histograms
  //TH1I *totalHtD_GlobalError_, *totalHtE_GlobalError_;
  //TH1I *totalHtD_GlobalError_Of_, *totalHtE_GlobalError_Of_;
  //missingET
  TH1I *missingEtD_, *missingEtE_;
  TH1I *missingEtD_Of_, *missingEtE_Of_;
  TH1I *missingEtD_Phi_, *missingEtE_Phi_;
  TH1I *missingEt_errorFlag_;
  //missingHT
  TH1I *missingHtD_, *missingHtE_;
  TH1I *missingHtD_Of_, *missingHtE_Of_;
  TH1I *missingHtD_Phi_, *missingHtE_Phi_;
  TH1I *missingHt_errorFlag_;
  TH1I *missingHtD_HtXPosLeaf1, *missingHtD_HtXPosLeaf2, *missingHtD_HtXPosLeaf3, *missingHtD_HtXNegLeaf1,
      *missingHtD_HtXNegLeaf2, *missingHtD_HtXNegLeaf3;
  TH1I *missingHtD_HtYPosLeaf1, *missingHtD_HtYPosLeaf2, *missingHtD_HtYPosLeaf3, *missingHtD_HtYNegLeaf1,
      *missingHtD_HtYNegLeaf2, *missingHtD_HtYNegLeaf3;

  //error flags to decide whether or not to print debug info
  bool isIsoError;
  bool isNonIsoError;
  bool isCenJetError;
  bool isTauJetError;
  bool isForJetError;
  bool isRingSumError;
  bool isBitCountError;
  bool isTotalEError;
  bool isTotalHError;
  bool isMissingEError;
  bool isMissingHError;

  //Directories - put this here because we want to
  //add directories dynamically to this folder
  //depending on the errors we find
  std::vector<TFileDirectory> errorHistCat;

  //the event number
  unsigned int eventNumber;

  const unsigned int *RCT_REGION_QUANTA;
};

//
// constants, enums and typedefs
// use in conjunction with the templated bits
typedef compareCands<edm::Handle<L1GctEmCandCollection> > compareEG;
typedef compareCands<edm::Handle<L1GctJetCandCollection> > compareJets;
typedef compareTotalEnergySums<edm::Handle<L1GctEtTotalCollection> > compareTotalE;
typedef compareTotalEnergySums<edm::Handle<L1GctEtHadCollection> > compareTotalH;
typedef compareMissingEnergySums<edm::Handle<L1GctEtMissCollection> > compareMissingE;
typedef compareMissingEnergySums<edm::Handle<L1GctHtMissCollection> > compareMissingH;
//
// static data member definitions
//

//
// constructors and destructor
//
GctErrorAnalyzer::GctErrorAnalyzer(const edm::ParameterSet &iConfig)
    : doRCT_(iConfig.getUntrackedParameter<bool>("doRCT", true)),
      doEg_(iConfig.getUntrackedParameter<bool>("doEg", true)),
      doIsoDebug_(iConfig.getUntrackedParameter<bool>("doIsoDebug", true)),
      doNonIsoDebug_(iConfig.getUntrackedParameter<bool>("doNonIsoDebug", true)),
      doJets_(iConfig.getUntrackedParameter<bool>("doJets", true)),
      doCenJetsDebug_(iConfig.getUntrackedParameter<bool>("doCenJetsDebug", true)),
      doTauJetsDebug_(iConfig.getUntrackedParameter<bool>("doTauJetsDebug", true)),
      doForJetsDebug_(iConfig.getUntrackedParameter<bool>("doForJetsDebug", true)),
      doHF_(iConfig.getUntrackedParameter<bool>("doHF", true)),
      doRingSumDebug_(iConfig.getUntrackedParameter<bool>("doRingSumDebug", true)),
      doBitCountDebug_(iConfig.getUntrackedParameter<bool>("doBitCountDebug", true)),
      doTotalEnergySums_(iConfig.getUntrackedParameter<bool>("doTotalEnergySums", true)),
      doTotalEtDebug_(iConfig.getUntrackedParameter<bool>("doTotalEtDebug", true)),
      doTotalHtDebug_(iConfig.getUntrackedParameter<bool>("doTotalHtDebug", true)),
      doMissingEnergySums_(iConfig.getUntrackedParameter<bool>("doMissingEnergySums", true)),
      doMissingETDebug_(iConfig.getUntrackedParameter<bool>("doMissingETDebug", true)),
      doMissingHTDebug_(iConfig.getUntrackedParameter<bool>("doMissingHTDebug", true)),
      doExtraMissingHTDebug_(iConfig.getUntrackedParameter<bool>("doExtraMissingHTDebug", false)),
      doRCTMBx_(iConfig.getUntrackedParameter<bool>("doRCTMBx", false)),
      doEmuMBx_(iConfig.getUntrackedParameter<bool>("doEmuMBx", false)),
      doGCTMBx_(iConfig.getUntrackedParameter<bool>("doGCTMBx", false)),
      RCTTrigBx_(iConfig.getUntrackedParameter<int>("RCTTrigBx", 0)),
      EmuTrigBx_(iConfig.getUntrackedParameter<int>("EmuTrigBx", 0)),
      GCTTrigBx_(iConfig.getUntrackedParameter<int>("GCTTrigBx", 0)),
      dataTag_(iConfig.getUntrackedParameter<edm::InputTag>("dataTag", edm::InputTag("gctDigis"))),
      emuTag_(iConfig.getUntrackedParameter<edm::InputTag>("emuTag", edm::InputTag("gctEmuDigis"))),
      useSys_(iConfig.getUntrackedParameter<std::string>("useSys", "P5")) {
  //now do what ever initialization is needed
  //make the root file
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;

  //to try to make this look more elegant
  //make a string for each folder we'd like for the Data and Emulator Histograms
  std::vector<std::string> quantities;
  quantities.push_back("IsoEm");
  quantities.push_back("NonIsoEM");
  quantities.push_back("CenJets");
  quantities.push_back("TauJets");
  quantities.push_back("ForJets");
  quantities.push_back("HFRingSums");
  quantities.push_back("HFBitCounts");
  quantities.push_back("TotalESums");
  quantities.push_back("MissingESums");

  //make the Emulator Histogram directory
  TFileDirectory emuHist = fs->mkdir("EmulatorHistograms");
  std::vector<TFileDirectory> emuHistCat;

  //make the Data Histogram directory
  TFileDirectory dataHist = fs->mkdir("DataHistograms");
  std::vector<TFileDirectory> dataHistCat;

  //make the ErrorFlags directory
  TFileDirectory errorHistFlags = fs->mkdir("ErrorHistograms_Flags");

  //make the ErrorDebug directory
  TFileDirectory errorHistDetails = fs->mkdir("ErrorHistograms_Details");

  for (unsigned int i = 0; i < quantities.size(); i++) {
    //fill the data and emulator folders with the directories
    emuHistCat.push_back(emuHist.mkdir(quantities.at(i)));
    dataHistCat.push_back(dataHist.mkdir(quantities.at(i)));
  }

  //add a folder for RCT Regions - which only exist in data
  dataHistCat.push_back(dataHist.mkdir("RCTRegions"));
  //and add a folder for the Intermediate Jets - which only exist in emulator
  emuHistCat.push_back(emuHist.mkdir("IntJets"));

  //Fill the ErrorDebug folder with the directories
  errorHistCat.push_back(errorHistDetails.mkdir("EM"));
  errorHistCat.push_back(errorHistDetails.mkdir("Jets"));
  errorHistCat.push_back(errorHistDetails.mkdir("HF"));
  errorHistCat.push_back(errorHistDetails.mkdir("TotalE"));
  errorHistCat.push_back(errorHistDetails.mkdir("MissingE"));

  //BOOK HISTOGRAMS
  RCT_EtEtaPhi_ = dataHistCat.at(9).make<TH2I>(
      "RCT_EtEtaPhi", "RCT_EtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  RCT_TvEtaPhi_ = dataHistCat.at(9).make<TH2I>(
      "RCT_TvEtaPhi", "RCT_TvEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  RCT_FgEtaPhi_ = dataHistCat.at(9).make<TH2I>(
      "RCT_FgEtaPhi", "RCT_FgEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  RCT_OfEtaPhi_ = dataHistCat.at(9).make<TH2I>(
      "RCT_OfEtEtaPhi", "RCT_OfEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  //isoEg
  isoEgD_Rank_ = dataHistCat.at(0).make<TH1I>("isoEgD_Rank", "isoEgD_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  isoEgE_Rank_ = emuHistCat.at(0).make<TH1I>("isoEgE_Rank", "isoEgE_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  isoEgD_EtEtaPhi_ = dataHistCat.at(0).make<TH2I>(
      "isoEgD_EtEtaPhi", "isoEgD_EtEtaPhi;#eta (GCT Units);#phi(GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  isoEgE_EtEtaPhi_ = emuHistCat.at(0).make<TH2I>(
      "isoEgE_EtEtaPhi", "isoEgE_EtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  isoEgD_OccEtaPhi_ = dataHistCat.at(0).make<TH2I>(
      "isoEgD_OccEtaPhi", "isoEgD_OccEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  isoEgE_OccEtaPhi_ = emuHistCat.at(0).make<TH2I>(
      "isoEgE_OccEtaPhi", "isoEgE_OccEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  isoEg_errorFlag_ =
      errorHistFlags.make<TH1I>("isoEg_errorFlag", "isoEg_errorFlag;Status;Number of Candidates", 3, -0.5, 2.5);
  //Global isoEg Error
  isoEgD_GlobalError_Rank_ = errorHistCat.at(0).make<TH1I>(
      "isoEgD_GlobalError_Rank", "isoEgD_GlobalError_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  isoEgE_GlobalError_Rank_ = errorHistCat.at(0).make<TH1I>(
      "isoEgE_GlobalError_Rank", "isoEgE_GlobalError_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  isoEgD_GlobalError_EtEtaPhi_ = errorHistCat.at(0).make<TH2I>(
      "isoEgD_GlobalError_EtEtaPhi", "isoEgD_GlobalError_EtEtaPhi", 22, -0.5, 21.5, 18, -0.5, 17.5);
  isoEgE_GlobalError_EtEtaPhi_ = errorHistCat.at(0).make<TH2I>(
      "isoEgE_GlobalError_EtEtaPhi", "isoEgE_GlobalError_EtEtaPhi", 22, -0.5, 21.5, 18, -0.5, 17.5);
  //nonIsoEg
  nonIsoEgD_Rank_ =
      dataHistCat.at(1).make<TH1I>("nonIsoEgD_Rank", "nonIsoEgD_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  nonIsoEgE_Rank_ =
      emuHistCat.at(1).make<TH1I>("nonIsoEgE_Rank", "nonIsoEgE_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  nonIsoEgD_EtEtaPhi_ = dataHistCat.at(1).make<TH2I>(
      "nonIsoEgD_EtEtaPhi", "nonIsoEgD_EtEtaPhi;#eta (GCT Units);#phi(GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  nonIsoEgE_EtEtaPhi_ = emuHistCat.at(1).make<TH2I>(
      "nonIsoEgE_EtEtaPhi", "nonIsoEgE_EtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  nonIsoEgD_OccEtaPhi_ = dataHistCat.at(1).make<TH2I>(
      "nonIsoEgD_OccEtaPhi", "nonIsoEgD_OccEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  nonIsoEgE_OccEtaPhi_ = emuHistCat.at(1).make<TH2I>(
      "nonIsoEgE_OccEtaPhi", "nonIsoEgE_OccEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  nonIsoEg_errorFlag_ =
      errorHistFlags.make<TH1I>("nonIsoEg_errorFlag", "nonIsoEg_errorFlag;Status;Number of Candidates", 3, -0.5, 2.5);
  //Global nonIsoEg Error
  nonIsoEgD_GlobalError_Rank_ = errorHistCat.at(0).make<TH1I>(
      "nonIsoEgD_GlobalError_Rank", "nonIsoEgD_GlobalError_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  nonIsoEgE_GlobalError_Rank_ = errorHistCat.at(0).make<TH1I>(
      "nonIsoEgE_GlobalError_Rank", "nonIsoEgE_GlobalError_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  nonIsoEgD_GlobalError_EtEtaPhi_ = errorHistCat.at(0).make<TH2I>(
      "nonIsoEgD_GlobalError_EtEtaPhi", "nonIsoEgD_GlobalError_EtEtaPhi", 22, -0.5, 21.5, 18, -0.5, 17.5);
  nonIsoEgE_GlobalError_EtEtaPhi_ = errorHistCat.at(0).make<TH2I>(
      "nonIsoEgE_GlobalError_EtEtaPhi", "nonIsoEgE_GlobalError_EtEtaPhi", 22, -0.5, 21.5, 18, -0.5, 17.5);
  //CenJets
  cenJetD_Rank_ = dataHistCat.at(2).make<TH1I>("cenJetD_Rank", "cenJetD_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  cenJetE_Rank_ = emuHistCat.at(2).make<TH1I>("cenJetE_Rank", "cenJetE_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  cenJetD_EtEtaPhi_ = dataHistCat.at(2).make<TH2I>(
      "cenJetD_EtEtaPhi", "cenJetD_EtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  cenJetE_EtEtaPhi_ = emuHistCat.at(2).make<TH2I>(
      "cenJetE_EtEtaPhi", "cenJetE_EtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  cenJetD_OccEtaPhi_ = dataHistCat.at(2).make<TH2I>(
      "cenJetD_OccEtaPhi", "cenJetD_OccEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  cenJetE_OccEtaPhi_ = emuHistCat.at(2).make<TH2I>(
      "cenJetE_OccEtaPhi", "cenJetE_OccEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  cenJet_errorFlag_ =
      errorHistFlags.make<TH1I>("cenJet_errorFlag", "cenJet_errorFlag;Status;Number of Candidates", 3, -0.5, 2.5);
  //Global CenJet Error
  cenJetD_GlobalError_Rank_ =
      errorHistCat.at(1).make<TH1I>("cenJetD_GlobalError_Rank", "cenJetD_GlobalError_Rank", 64, -0.5, 63.5);
  cenJetE_GlobalError_Rank_ =
      errorHistCat.at(1).make<TH1I>("cenJetE_GlobalError_Rank", "cenJetE_GlobalError_Rank", 64, -0.5, 63.5);
  cenJetD_GlobalError_EtEtaPhi_ = errorHistCat.at(1).make<TH2I>(
      "cenJetD_GlobalError_EtEtaPhi", "cenJetD_GlobalError_EtEtaPhi", 22, -0.5, 21.5, 18, -0.5, 17.5);
  cenJetE_GlobalError_EtEtaPhi_ = errorHistCat.at(1).make<TH2I>(
      "cenJetE_GlobalError_EtEtaPhi", "cenJetE_GlobalError_EtEtaPhi", 22, -0.5, 21.5, 18, -0.5, 17.5);
  //TauJets
  tauJetD_Rank_ = dataHistCat.at(3).make<TH1I>("tauJetD_Rank", "tauJetD_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  tauJetE_Rank_ = emuHistCat.at(3).make<TH1I>("tauJetE_Rank", "tauJetE_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  tauJetD_EtEtaPhi_ = dataHistCat.at(3).make<TH2I>(
      "tauJetD_EtEtaPhi", "tauJetD_EtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  tauJetE_EtEtaPhi_ = emuHistCat.at(3).make<TH2I>(
      "tauJetE_EtEtaPhi", "tauJetE_EtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  tauJetD_OccEtaPhi_ = dataHistCat.at(3).make<TH2I>(
      "tauJetD_OccEtaPhi", "tauJetD_OccEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  tauJetE_OccEtaPhi_ = emuHistCat.at(3).make<TH2I>(
      "tauJetE_OccEtaPhi", "tauJetE_OccEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  tauJet_errorFlag_ =
      errorHistFlags.make<TH1I>("tauJet_errorFlag", "tauJet_errorFlag;Status;Number of Candidates", 3, -0.5, 2.5);
  //Global TauJet Error
  tauJetD_GlobalError_Rank_ =
      errorHistCat.at(1).make<TH1I>("tauJetD_GlobalError_Rank", "tauJetD_GlobalError_Rank", 64, -0.5, 63.5);
  tauJetE_GlobalError_Rank_ =
      errorHistCat.at(1).make<TH1I>("tauJetE_GlobalError_Rank", "tauJetE_GlobalError_Rank", 64, -0.5, 63.5);
  tauJetD_GlobalError_EtEtaPhi_ = errorHistCat.at(1).make<TH2I>(
      "tauJetD_GlobalError_EtEtaPhi", "tauJetD_GlobalError_EtEtaPhi", 22, -0.5, 21.5, 18, -0.5, 17.5);
  tauJetE_GlobalError_EtEtaPhi_ = errorHistCat.at(1).make<TH2I>(
      "tauJetE_GlobalError_EtEtaPhi", "tauJetE_GlobalError_EtEtaPhi", 22, -0.5, 21.5, 18, -0.5, 17.5);
  //ForJets
  forJetD_Rank_ = dataHistCat.at(4).make<TH1I>("forJetD_Rank", "forJetD_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  forJetE_Rank_ = emuHistCat.at(4).make<TH1I>("forJetE_Rank", "forJetE_Rank;Rank;Number of Events", 64, -0.5, 63.5);
  forJetD_EtEtaPhi_ = dataHistCat.at(4).make<TH2I>(
      "forJetD_EtEtaPhi", "forJetD_EtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  forJetE_EtEtaPhi_ = emuHistCat.at(4).make<TH2I>(
      "forJetE_EtEtaPhi", "forJetE_EtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  forJetD_OccEtaPhi_ = dataHistCat.at(4).make<TH2I>(
      "forJetD_OccEtaPhi", "forJetD_OccEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  forJetE_OccEtaPhi_ = emuHistCat.at(4).make<TH2I>(
      "forJetE_OccEtaPhi", "forJetE_OccEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  forJet_errorFlag_ =
      errorHistFlags.make<TH1I>("forJet_errorFlag", "forJet_errorFlag;Status;Number of Candidates", 3, -0.5, 2.5);
  //Global ForJet Error
  forJetD_GlobalError_Rank_ =
      errorHistCat.at(1).make<TH1I>("forJetD_GlobalError_Rank", "forJetD_GlobalError_Rank", 64, -0.5, 63.5);
  forJetE_GlobalError_Rank_ =
      errorHistCat.at(1).make<TH1I>("forJetE_GlobalError_Rank", "forJetE_GlobalError_Rank", 64, -0.5, 63.5);
  forJetD_GlobalError_EtEtaPhi_ = errorHistCat.at(1).make<TH2I>(
      "forJetD_GlobalError_EtEtaPhi", "forJetD_GlobalError_EtEtaPhi", 22, -0.5, 21.5, 18, -0.5, 17.5);
  forJetE_GlobalError_EtEtaPhi_ = errorHistCat.at(1).make<TH2I>(
      "forJetE_GlobalError_EtEtaPhi", "forJetE_GlobalError_EtEtaPhi", 22, -0.5, 21.5, 18, -0.5, 17.5);
  //IntJets
  intJetEtEtaPhiE_ = emuHistCat.at(9).make<TH2I>(
      "intJetEtEtaPhiE_", "intJetEtEtaPhiE_;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  intJetE_Et_ = emuHistCat.at(9).make<TH1I>("intJetE_Et", "intJetE_Et;E_{T};Number of Events", 1024, -0.5, 1023.5);
  intJetE_Of_ =
      emuHistCat.at(9).make<TH1I>("intJetE_Of", "intJetE_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  intJetE_Jet1Et_ =
      emuHistCat.at(9).make<TH1I>("intJetE_Jet1Et", "intJetE_Jet1Et;E_{T};Number of Events", 1024, -0.5, 1023.5);
  intJetE_Jet2Et_ =
      emuHistCat.at(9).make<TH1I>("intJetE_Jet2Et", "intJetE_Jet2Et;E_{T};Number of Events", 1024, -0.5, 1023.5);
  intJetE_Jet3Et_ =
      emuHistCat.at(9).make<TH1I>("intJetE_Jet3Et", "intJetE_Jet3Et;E_{T};Number of Events", 1024, -0.5, 1023.5);
  intJetE_Jet4Et_ =
      emuHistCat.at(9).make<TH1I>("intJetE_Jet4Et", "intJetE_Jet4Et;E_{T};Number of Events", 1024, -0.5, 1023.5);
  //HFRing Sums
  hfRingSumD_1pos_ = dataHistCat.at(5).make<TH1I>("hfRingSumD_1+", "hfRingSumD_1+;Rank;Number of Events", 8, -0.5, 7.5);
  hfRingSumD_1neg_ = dataHistCat.at(5).make<TH1I>("hfRingSumD_1-", "hfRingSumD_1-;Rank;Number of Events", 8, -0.5, 7.5);
  hfRingSumD_2pos_ = dataHistCat.at(5).make<TH1I>("hfRingSumD_2+", "hfRingSumD_2+;Rank;Number of Events", 8, -0.5, 7.5);
  hfRingSumD_2neg_ = dataHistCat.at(5).make<TH1I>("hfRingSumD_2-", "hfRingSumD_2-;Rank;Number of Events", 8, -0.5, 7.5);
  hfRingSumE_1pos_ = emuHistCat.at(5).make<TH1I>("hfRingSumE_1+", "hfRingSumE_1+;Rank;Number of Events", 8, -0.5, 7.5);
  hfRingSumE_1neg_ = emuHistCat.at(5).make<TH1I>("hfRingSumE_1-", "hfRingSumE_1-;Rank;Number of Events", 8, -0.5, 7.5);
  hfRingSumE_2pos_ = emuHistCat.at(5).make<TH1I>("hfRingSumE_2+", "hfRingSumE_2+;Rank;Number of Events", 8, -0.5, 7.5);
  hfRingSumE_2neg_ = emuHistCat.at(5).make<TH1I>("hfRingSumE_2-", "hfRingSumE_2-;Rank;Number of Events", 8, -0.5, 7.5);
  hfRingSum_errorFlag_ =
      errorHistFlags.make<TH1I>("hfRingSum_errorFlag", "hfRingSum_errorFlag;Status;Number of Candidates", 2, -0.5, 1.5);
  //HFRing BitCounts
  hfBitCountD_1pos_ =
      dataHistCat.at(6).make<TH1I>("hfBitCountD_1+", "hfBitCountD_1+;Rank;Number of Events", 8, -0.5, 7.5);
  hfBitCountD_1neg_ =
      dataHistCat.at(6).make<TH1I>("hfBitCountD_1-", "hfBitCountD_1-;Rank;Number of Events", 8, -0.5, 7.5);
  hfBitCountD_2pos_ =
      dataHistCat.at(6).make<TH1I>("hfBitCountD_2+", "hfBitCountD_2+;Rank;Number of Events", 8, -0.5, 7.5);
  hfBitCountD_2neg_ =
      dataHistCat.at(6).make<TH1I>("hfBitCountD_2-", "hfBitCountD_2-;Rank;Number of Events", 8, -0.5, 7.5);
  hfBitCountE_1pos_ =
      emuHistCat.at(6).make<TH1I>("hfBitCountE_1+", "hfBitCountE_1+;Rank;Number of Events", 8, -0.5, 7.5);
  hfBitCountE_1neg_ =
      emuHistCat.at(6).make<TH1I>("hfBitCountE_1-", "hfBitCountE_1-;Rank;Number of Events", 8, -0.5, 7.5);
  hfBitCountE_2pos_ =
      emuHistCat.at(6).make<TH1I>("hfBitCountE_2+", "hfBitCountE_2+;Rank;Number of Events", 8, -0.5, 7.5);
  hfBitCountE_2neg_ =
      emuHistCat.at(6).make<TH1I>("hfBitCountE_2-", "hfBitCountE_2-;Rank;Number of Events", 8, -0.5, 7.5);
  hfBitCount_errorFlag_ = errorHistFlags.make<TH1I>(
      "hfBitCount_errorFlag", "hfBitCount_errorFlag;Status;Number of Candidates", 2, -0.5, 1.5);
  //Total ET
  totalEtD_ = dataHistCat.at(7).make<TH1I>("totalEtD", "totalEtD;E_{T};Number of Events", 2048, -0.5, 2047.5);
  totalEtD_Of_ =
      dataHistCat.at(7).make<TH1I>("totalEtD_Of", "totalEtD_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  totalEtE_ = emuHistCat.at(7).make<TH1I>("totalEtE", "totalEtE;E_{T};Number of Events", 2048, -0.5, 2047.5);
  totalEtE_Of_ =
      emuHistCat.at(7).make<TH1I>("totalEtE_Of", "totalEtE_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  totalEt_errorFlag_ =
      errorHistFlags.make<TH1I>("totalEt_errorFlag", "totalEt_errorFlag;Status;Number of Candidates", 2, -0.5, 1.5);
  //Book the Global ET Error histograms in the errorHistCat
  //totalEtD_GlobalError_ = errorHistCat.at(3).make<TH1I>("totalEtD_GlobalError", "totalEtD_GlobalError;E_{T};Number of Events", 1024, -0.5, 1023.5);
  //totalEtE_GlobalError_ = errorHistCat.at(3).make<TH1I>("totalEtE_GlobalError", "totalEtE_GlobalError;E_{T};Number of Events", 1024, -0.5, 1023.5);
  //totalEtD_GlobalError_Of_ = errorHistCat.at(3).make<TH1I>("totalEtD_GlobalError_Of", "totalEtD_GlobalError_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  //totalEtE_GlobalError_Of_ = errorHistCat.at(3).make<TH1I>("totalEtE_GlobalError_Of", "totalEtE_GlobalError_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  //Total HT
  totalHtD_ = dataHistCat.at(7).make<TH1I>("totalHtD", "totalHtD;H_{T};Number of Events", 2048, -0.5, 2047.5);
  totalHtD_Of_ =
      dataHistCat.at(7).make<TH1I>("totalHtD_Of", "totalHtD_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  totalHtE_ = emuHistCat.at(7).make<TH1I>("totalHtE", "totalHtE;H_{T};Number of Events", 2048, -0.5, 2047.5);
  totalHtE_Of_ =
      emuHistCat.at(7).make<TH1I>("totalHtE_Of", "totalHtE_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  totalHt_errorFlag_ =
      errorHistFlags.make<TH1I>("totalHt_errorFlag", "totalHt_errorFlag;Status;Number of Candidates", 2, -0.5, 1.5);
  //Book the Global HT Error histograms in the errorHistCat
  //totalHtD_GlobalError_ = errorHistCat.at(3).make<TH1I>("totalHtD_GlobalError", "totalHtD_GlobalError;E_{T};Number of Events", 1024, -0.5, 1023.5);
  //totalHtE_GlobalError_ = errorHistCat.at(3).make<TH1I>("totalHtE_GlobalError", "totalHtE_GlobalError;E_{T};Number of Events", 1024, -0.5, 1023.5);
  //totalHtD_GlobalError_Of_ = errorHistCat.at(3).make<TH1I>("totalHtD_GlobalError_Of", "totalHtD_GlobalError_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  //totalHtE_GlobalError_Of_ = errorHistCat.at(3).make<TH1I>("totalHtE_GlobalError_Of", "totalHtE_GlobalError_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  //MissingEt
  missingEtD_ =
      dataHistCat.at(8).make<TH1I>("missingEtD", "missingEtD;Missing E_{T};Number of Events", 1024, -0.5, 1023.5);
  missingEtD_Of_ =
      dataHistCat.at(8).make<TH1I>("missingEtD_Of", "missingEtD_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  missingEtD_Phi_ = dataHistCat.at(8).make<TH1I>(
      "missingEtD_Phi", "missingEtD_Phi;Missing E_{T} #phi;Number of Events", 72, -0.5, 71.5);
  missingEtE_ =
      emuHistCat.at(8).make<TH1I>("missingEtE", "missingEtE;Missing E_{T};Number of Events", 1024, -0.5, 1023.5);
  missingEtE_Of_ =
      emuHistCat.at(8).make<TH1I>("missingEtE_Of", "missingEtE_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  missingEtE_Phi_ = emuHistCat.at(8).make<TH1I>(
      "missingEtE_Phi", "missingEtE_Phi;Missing E_{T} #phi;Number of Events", 72, -0.5, 71.5);
  missingEt_errorFlag_ =
      errorHistFlags.make<TH1I>("missingEt_errorFlag", "missingEt_errorFlag;Status;Number of Candidates", 4, -0.5, 3.5);
  //MissingHt
  missingHtD_ =
      dataHistCat.at(8).make<TH1I>("missingHtD", "missingHtD;Missing H_{T};Number of Events", 1024, -0.5, 1023.5);
  missingHtD_Of_ =
      dataHistCat.at(8).make<TH1I>("missingHtD_Of", "missingHtD_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  missingHtD_Phi_ = dataHistCat.at(8).make<TH1I>(
      "missingHtD_Phi", "missingHtD_Phi;Missing H_{T} #phi;Number of Events", 72, -0.5, 71.5);
  missingHtE_ =
      emuHistCat.at(8).make<TH1I>("missingHtE", "missingHtE;Missing H_{T};Number of Events", 1024, -0.5, 1023.5);
  missingHtE_Of_ =
      emuHistCat.at(8).make<TH1I>("missingHtE_Of", "missingHtE_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  missingHtE_Phi_ = emuHistCat.at(8).make<TH1I>(
      "missingHtE_Phi", "missingHtE_Phi;Missing H_{T} #phi;Number of Events", 72, -0.5, 71.5);
  missingHt_errorFlag_ =
      errorHistFlags.make<TH1I>("missingHt_errorFlag", "missingHt_errorFlag;Status;Number of Candidates", 4, -0.5, 3.5);
  //Additional MissingHt Debug histograms
  missingHtD_HtXPosLeaf1 = dataHistCat.at(8).make<TH1I>(
      "missingHtD_HtXPosLeaf1", "missingHtD;Missing H_{T} X PosLeaf1;Number of Events", 4096, -2048.5, 2047.5);
  missingHtD_HtXPosLeaf2 = dataHistCat.at(8).make<TH1I>(
      "missingHtD_HtXPosLeaf2", "missingHtD;Missing H_{T} X PosLeaf2;Number of Events", 4096, -2048.5, 2047.5);
  missingHtD_HtXPosLeaf3 = dataHistCat.at(8).make<TH1I>(
      "missingHtD_HtXPosLeaf3", "missingHtD;Missing H_{T} X PosLeaf3;Number of Events", 4096, -2048.5, 2047.5);
  missingHtD_HtXNegLeaf1 = dataHistCat.at(8).make<TH1I>(
      "missingHtD_HtXNegLeaf1", "missingHtD;Missing H_{T} X NegLeaf1;Number of Events", 4096, -2048.5, 2047.5);
  missingHtD_HtXNegLeaf2 = dataHistCat.at(8).make<TH1I>(
      "missingHtD_HtXNegLeaf2", "missingHtD;Missing H_{T} X NegLeaf2;Number of Events", 4096, -2048.5, 2047.5);
  missingHtD_HtXNegLeaf3 = dataHistCat.at(8).make<TH1I>(
      "missingHtD_HtXNegLeaf3", "missingHtD;Missing H_{T} X NegLeaf3;Number of Events", 4096, -2048.5, 2047.5);

  missingHtD_HtYPosLeaf1 = dataHistCat.at(8).make<TH1I>(
      "missingHtD_HtYPosLeaf1", "missingHtD;Missing H_{T} Y PosLeaf1;Number of Events", 4096, -2048.5, 2047.5);
  missingHtD_HtYPosLeaf2 = dataHistCat.at(8).make<TH1I>(
      "missingHtD_HtYPosLeaf2", "missingHtD;Missing H_{T} Y PosLeaf2;Number of Events", 4096, -2048.5, 2047.5);
  missingHtD_HtYPosLeaf3 = dataHistCat.at(8).make<TH1I>(
      "missingHtD_HtYPosLeaf3", "missingHtD;Missing H_{T} Y PosLeaf3;Number of Events", 4096, -2048.5, 2047.5);
  missingHtD_HtYNegLeaf1 = dataHistCat.at(8).make<TH1I>(
      "missingHtD_HtYNegLeaf1", "missingHtD;Missing H_{T} Y NegLeaf1;Number of Events", 4096, -2048.5, 2047.5);
  missingHtD_HtYNegLeaf2 = dataHistCat.at(8).make<TH1I>(
      "missingHtD_HtYNegLeaf2", "missingHtD;Missing H_{T} Y NegLeaf2;Number of Events", 4096, -2048.5, 2047.5);
  missingHtD_HtYNegLeaf3 = dataHistCat.at(8).make<TH1I>(
      "missingHtD_HtYNegLeaf3", "missingHtD;Missing H_{T} Y NegLeaf3;Number of Events", 4096, -2048.5, 2047.5);

  //Annotate the labels of the error flags
  //For the electrons and jets
  std::vector<std::string> errorFlagLabels;
  errorFlagLabels.push_back("Matched");
  errorFlagLabels.push_back("Unmatched Data Cand");
  errorFlagLabels.push_back("Unmatched Emul Cand");

  for (unsigned int i = 0; i < errorFlagLabels.size(); i++) {
    isoEg_errorFlag_->GetXaxis()->SetBinLabel(i + 1, errorFlagLabels.at(i).c_str());
    nonIsoEg_errorFlag_->GetXaxis()->SetBinLabel(i + 1, errorFlagLabels.at(i).c_str());
    cenJet_errorFlag_->GetXaxis()->SetBinLabel(i + 1, errorFlagLabels.at(i).c_str());
    tauJet_errorFlag_->GetXaxis()->SetBinLabel(i + 1, errorFlagLabels.at(i).c_str());
    forJet_errorFlag_->GetXaxis()->SetBinLabel(i + 1, errorFlagLabels.at(i).c_str());
  }
  errorFlagLabels.clear();

  //For the Total Energy Sums and HF
  errorFlagLabels.push_back("Matched");
  errorFlagLabels.push_back("Unmatched");

  for (unsigned int i = 0; i < errorFlagLabels.size(); i++) {
    hfRingSum_errorFlag_->GetXaxis()->SetBinLabel(i + 1, errorFlagLabels.at(i).c_str());
    hfBitCount_errorFlag_->GetXaxis()->SetBinLabel(i + 1, errorFlagLabels.at(i).c_str());
    totalEt_errorFlag_->GetXaxis()->SetBinLabel(i + 1, errorFlagLabels.at(i).c_str());
    totalHt_errorFlag_->GetXaxis()->SetBinLabel(i + 1, errorFlagLabels.at(i).c_str());
  }
  errorFlagLabels.clear();

  //For the Missing Energy Sums
  errorFlagLabels.push_back("Matched");
  errorFlagLabels.push_back("Matched Mag");
  errorFlagLabels.push_back("Matched Phi");
  errorFlagLabels.push_back("Unmatched");

  for (unsigned int i = 0; i < errorFlagLabels.size(); i++) {
    missingEt_errorFlag_->GetXaxis()->SetBinLabel(i + 1, errorFlagLabels.at(i).c_str());
    missingHt_errorFlag_->GetXaxis()->SetBinLabel(i + 1, errorFlagLabels.at(i).c_str());
  }

  //initialise - set all flags to false as they will be set on an event-by-event basis
  isIsoError = false;
  isNonIsoError = false;
  isCenJetError = false;
  isTauJetError = false;
  isForJetError = false;
  isRingSumError = false;
  isBitCountError = false;
  isTotalEError = false;
  isTotalHError = false;
  isMissingEError = false;
  isMissingHError = false;

  //fill the struct of MBXinformation. It is easier to pass this information to the respective functions as used below this way
  MBxInfo.RCTTrigBx = RCTTrigBx_;
  MBxInfo.EmuTrigBx = EmuTrigBx_;
  MBxInfo.GCTTrigBx = GCTTrigBx_;

  //set the parameters according to the system chosen
  if (useSys_ == "P5") {
    RCT_REGION_QUANTA = &RCT_REGION_QUANTA_P5;
  } else if (useSys_ == "Lab") {
    RCT_REGION_QUANTA = &RCT_REGION_QUANTA_LAB;
  } else {
    edm::LogWarning("ChosenSystem") << " "
                                    << "The system you chose to use (" << useSys_
                                    << ") was not recognised. Defaulting to the full system geometry";
    RCT_REGION_QUANTA = &RCT_REGION_QUANTA_P5;
  }
}

GctErrorAnalyzer::~GctErrorAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void GctErrorAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  using namespace std;

  Handle<L1CaloRegionCollection> caloRegions;
  Handle<L1CaloEmCollection> emRegions;

  Handle<L1GctEmCandCollection> nonIsoEgD;
  Handle<L1GctEmCandCollection> nonIsoEgE;
  Handle<L1GctEmCandCollection> isoEgD;
  Handle<L1GctEmCandCollection> isoEgE;

  Handle<L1GctJetCandCollection> cenJetsD;
  Handle<L1GctJetCandCollection> cenJetsE;
  Handle<L1GctJetCandCollection> forJetsD;
  Handle<L1GctJetCandCollection> forJetsE;
  Handle<L1GctJetCandCollection> tauJetsD;
  Handle<L1GctJetCandCollection> tauJetsE;

  Handle<L1GctInternJetDataCollection> intJetsE;

  Handle<L1GctHFRingEtSumsCollection> hfRingSumsD;
  Handle<L1GctHFRingEtSumsCollection> hfRingSumsE;

  Handle<L1GctHFBitCountsCollection> hfBitCountsD;
  Handle<L1GctHFBitCountsCollection> hfBitCountsE;

  Handle<L1GctEtTotalCollection> totalEtD;
  Handle<L1GctEtTotalCollection> totalEtE;

  Handle<L1GctEtHadCollection> totalHtD;
  Handle<L1GctEtHadCollection> totalHtE;

  Handle<L1GctEtMissCollection> missingEtD;
  Handle<L1GctEtMissCollection> missingEtE;

  Handle<L1GctHtMissCollection> missingHtD;
  Handle<L1GctHtMissCollection> missingHtE;

  Handle<L1GctInternHtMissCollection> intHtMissD;

  //we need this for all user cases...
  iEvent.getByLabel(dataTag_.label(), caloRegions);

  //in order to allow the debug folders to have a unique name (so that when jobs are split in crab, we can merge)
  //use the eventnum in the folder name
  eventNumber = iEvent.id().event();

  if (doRCT_) {
    if (checkCollections(caloRegions, *RCT_REGION_QUANTA, "RCT CaloRegions"))
      plotRCTRegions(caloRegions);
  }

  if (doEg_) {
    iEvent.getByLabel(dataTag_.label(), "nonIsoEm", nonIsoEgD);
    iEvent.getByLabel(emuTag_.label(), "nonIsoEm", nonIsoEgE);

    iEvent.getByLabel(dataTag_.label(), "isoEm", isoEgD);
    iEvent.getByLabel(emuTag_.label(), "isoEm", isoEgE);

    isIsoError = false;
    isNonIsoError = false;

    if (checkCollections(isoEgD, GCT_OBJECT_QUANTA, "Iso e/g Data") &&
        checkCollections(isoEgE, GCT_OBJECT_QUANTA, "Iso e/g Emulator")) {
      plotIsoEm(isoEgD, isoEgE);
      compareEG isoCompare(isoEgD, isoEgE, MBxInfo);
      isIsoError = isoCompare.doCompare(isoEg_errorFlag_,
                                        isoEgD_GlobalError_Rank_,
                                        isoEgD_GlobalError_EtEtaPhi_,
                                        isoEgE_GlobalError_Rank_,
                                        isoEgE_GlobalError_EtEtaPhi_);
    }

    if (checkCollections(nonIsoEgD, GCT_OBJECT_QUANTA, "NonIso e/g Data") &&
        checkCollections(nonIsoEgE, GCT_OBJECT_QUANTA, "NonIso e/g Emulator")) {
      plotNonIsoEm(nonIsoEgD, nonIsoEgE);
      compareEG nonIsoCompare(nonIsoEgD, nonIsoEgE, MBxInfo);
      isNonIsoError = nonIsoCompare.doCompare(nonIsoEg_errorFlag_,
                                              nonIsoEgD_GlobalError_Rank_,
                                              nonIsoEgD_GlobalError_EtEtaPhi_,
                                              nonIsoEgE_GlobalError_Rank_,
                                              nonIsoEgE_GlobalError_EtEtaPhi_);
    }

    if ((isIsoError && doIsoDebug_) || (isNonIsoError && doNonIsoDebug_)) {
      iEvent.getByLabel(dataTag_.label(), emRegions);
      if (checkCollections(emRegions, RCT_EM_OBJECT_QUANTA, "RCT EMRegions"))
        plotEGErrors(isoEgD, isoEgE, nonIsoEgD, nonIsoEgE, emRegions);
    }
  }

  if (doJets_) {
    iEvent.getByLabel(emuTag_.label(), "cenJets", cenJetsE);
    iEvent.getByLabel(dataTag_.label(), "cenJets", cenJetsD);

    iEvent.getByLabel(emuTag_.label(), "forJets", forJetsE);
    iEvent.getByLabel(dataTag_.label(), "forJets", forJetsD);

    iEvent.getByLabel(emuTag_.label(), "tauJets", tauJetsE);
    iEvent.getByLabel(dataTag_.label(), "tauJets", tauJetsD);

    iEvent.getByLabel(emuTag_.label(), intJetsE);

    isCenJetError = false;
    isTauJetError = false;
    isForJetError = false;

    //Central Jets
    if (checkCollections(cenJetsD, GCT_OBJECT_QUANTA, "Central Jets Data") &&
        checkCollections(cenJetsE, GCT_OBJECT_QUANTA, "Central Jets Emulator")) {
      plotCenJets(cenJetsD, cenJetsE);
      compareJets cenJetsCompare(cenJetsD, cenJetsE, MBxInfo);
      isCenJetError = cenJetsCompare.doCompare(cenJet_errorFlag_,
                                               cenJetD_GlobalError_Rank_,
                                               cenJetD_GlobalError_EtEtaPhi_,
                                               cenJetE_GlobalError_Rank_,
                                               cenJetE_GlobalError_EtEtaPhi_);
    }

    //Tau Jets
    if (checkCollections(tauJetsD, GCT_OBJECT_QUANTA, "Tau Jets Data") &&
        checkCollections(tauJetsE, GCT_OBJECT_QUANTA, "Tau Jets Emulator")) {
      plotTauJets(tauJetsD, tauJetsE);
      compareJets tauJetsCompare(tauJetsD, tauJetsE, MBxInfo);
      isTauJetError = tauJetsCompare.doCompare(tauJet_errorFlag_,
                                               tauJetD_GlobalError_Rank_,
                                               tauJetD_GlobalError_EtEtaPhi_,
                                               tauJetE_GlobalError_Rank_,
                                               tauJetE_GlobalError_EtEtaPhi_);
    }

    //For Jets
    if (checkCollections(forJetsD, GCT_OBJECT_QUANTA, "Forward Jets Data") &&
        checkCollections(forJetsE, GCT_OBJECT_QUANTA, "Forward Jets Emulator")) {
      plotForJets(forJetsD, forJetsE);
      compareJets forJetsCompare(forJetsD, forJetsE, MBxInfo);
      isForJetError = forJetsCompare.doCompare(forJet_errorFlag_,
                                               forJetD_GlobalError_Rank_,
                                               forJetD_GlobalError_EtEtaPhi_,
                                               forJetE_GlobalError_Rank_,
                                               forJetE_GlobalError_EtEtaPhi_);
    }

    //Emulator Intermediate Jets
    if (checkCollections(intJetsE, NUM_INT_JETS, "Intermediate Jets Emulator"))
      plotIntJets(intJetsE);

    if ((isCenJetError && doCenJetsDebug_) || (isTauJetError && doTauJetsDebug_) ||
        (isForJetError && doForJetsDebug_)) {
      plotJetErrors(cenJetsD, cenJetsE, tauJetsD, tauJetsE, forJetsD, forJetsE, caloRegions);
    }
  }

  if (doHF_) {
    iEvent.getByLabel(dataTag_.label(), hfRingSumsD);
    iEvent.getByLabel(emuTag_.label(), hfRingSumsE);

    iEvent.getByLabel(dataTag_.label(), hfBitCountsD);
    iEvent.getByLabel(emuTag_.label(), hfBitCountsE);

    isRingSumError = false;
    isBitCountError = false;

    if (checkCollections(hfRingSumsD, GCT_SUMS_QUANTA, "HF Ring Sums Data") &&
        checkCollections(hfRingSumsE, GCT_SUMS_QUANTA, "HF Ring Sums Emulator")) {
      plotHFRingSums(hfRingSumsD, hfRingSumsE);
      compareRingSums HFRingSums(hfRingSumsD, hfRingSumsE, MBxInfo);
      isRingSumError = HFRingSums.doCompare(hfRingSum_errorFlag_);
    }

    if (checkCollections(hfBitCountsD, GCT_SUMS_QUANTA, "HF Bit Counts Data") &&
        checkCollections(hfBitCountsE, GCT_SUMS_QUANTA, "HF Bit Counts Emulator")) {
      plotHFBitCounts(hfBitCountsD, hfBitCountsE);
      compareBitCounts HFBitCounts(hfBitCountsD, hfBitCountsE, MBxInfo);
      isBitCountError = HFBitCounts.doCompare(hfBitCount_errorFlag_);
    }

    if ((isRingSumError && doRingSumDebug_) || (isBitCountError && doBitCountDebug_)) {
      plotHFErrors(hfRingSumsD, hfRingSumsE, hfBitCountsD, hfBitCountsE, caloRegions);
    }
  }

  if (doTotalEnergySums_) {
    iEvent.getByLabel(dataTag_.label(), totalEtD);
    iEvent.getByLabel(emuTag_.label(), totalEtE);

    iEvent.getByLabel(dataTag_.label(), totalHtD);
    iEvent.getByLabel(emuTag_.label(), totalHtE);

    isTotalEError = false;
    isTotalHError = false;

    if (checkCollections(totalEtD, GCT_SUMS_QUANTA, "Total Et Data") &&
        checkCollections(totalEtE, GCT_SUMS_QUANTA, "Total Et Emulator")) {
      plotTotalE(totalEtD, totalEtE);
      compareTotalE compareET(totalEtD, totalEtE, MBxInfo);
      isTotalEError = compareET.doCompare(totalEt_errorFlag_);
    }

    if (checkCollections(totalHtD, GCT_SUMS_QUANTA, "Total Ht Data") &&
        checkCollections(totalHtE, GCT_SUMS_QUANTA, "Total Ht Emulator")) {
      plotTotalH(totalHtD, totalHtE);
      compareTotalH compareHT(totalHtD, totalHtE, MBxInfo);
      isTotalHError = compareHT.doCompare(totalHt_errorFlag_);
    }

    if ((isTotalEError && doTotalEtDebug_) || (isTotalHError && doTotalHtDebug_)) {
      plotTotalEErrors(totalEtD, totalEtE, totalHtD, totalHtE, caloRegions);
    }
  }

  if (doMissingEnergySums_) {
    iEvent.getByLabel(dataTag_.label(), missingEtD);
    iEvent.getByLabel(emuTag_.label(), missingEtE);

    iEvent.getByLabel(dataTag_.label(), missingHtD);
    iEvent.getByLabel(emuTag_.label(), missingHtE);

    isMissingEError = false;
    isMissingHError = false;

    if (checkCollections(missingEtD, GCT_SUMS_QUANTA, "Missing Et Data") &&
        checkCollections(missingEtE, GCT_SUMS_QUANTA, "Missing Et Emulator")) {
      plotMissingEt(missingEtD, missingEtE);
      compareMissingE compareMET(missingEtD, missingEtE, MBxInfo);
      isMissingEError = compareMET.doCompare(missingEt_errorFlag_);
    }

    if (checkCollections(missingHtD, GCT_SUMS_QUANTA, "Missing Ht Data") &&
        checkCollections(missingHtE, GCT_SUMS_QUANTA, "Missing Ht Emulator")) {
      plotMissingHt(missingHtD, missingHtE);
      compareMissingH compareMHT(missingHtD, missingHtE, MBxInfo);
      isMissingHError = compareMHT.doCompare(missingHt_errorFlag_);

      //added 19/03/2010 for intermediate information on MissingHt quantities in the data
      if (doExtraMissingHTDebug_) {
        iEvent.getByLabel(dataTag_.label(), "", intHtMissD);
        if (checkCollections(intHtMissD, GCT_INT_HTMISS_QUANTA, "Internal Missing Ht Data")) {
          for (unsigned int i = 0; i < intHtMissD->size(); i++) {
            if (doGCTMBx_ || intHtMissD->at(i).bx() == GCTTrigBx_) {
              if (!intHtMissD->at(i).overflow()) {
                //the capBlock 0x301 is the input pipeline at the wheel for positive eta, whereas 0x701 is for negative eta
                if (intHtMissD->at(i).capBlock() == 0x301 && intHtMissD->at(i).capIndex() == 0 &&
                    intHtMissD->at(i).isThereHtx())
                  missingHtD_HtXPosLeaf1->Fill(intHtMissD->at(i).htx());
                if (intHtMissD->at(i).capBlock() == 0x301 && intHtMissD->at(i).capIndex() == 1 &&
                    intHtMissD->at(i).isThereHtx())
                  missingHtD_HtXPosLeaf2->Fill(intHtMissD->at(i).htx());
                if (intHtMissD->at(i).capBlock() == 0x301 && intHtMissD->at(i).capIndex() == 2 &&
                    intHtMissD->at(i).isThereHtx())
                  missingHtD_HtXPosLeaf3->Fill(intHtMissD->at(i).htx());
                if (intHtMissD->at(i).capBlock() == 0x701 && intHtMissD->at(i).capIndex() == 0 &&
                    intHtMissD->at(i).isThereHtx())
                  missingHtD_HtXNegLeaf1->Fill(intHtMissD->at(i).htx());
                if (intHtMissD->at(i).capBlock() == 0x701 && intHtMissD->at(i).capIndex() == 1 &&
                    intHtMissD->at(i).isThereHtx())
                  missingHtD_HtXNegLeaf2->Fill(intHtMissD->at(i).htx());
                if (intHtMissD->at(i).capBlock() == 0x701 && intHtMissD->at(i).capIndex() == 2 &&
                    intHtMissD->at(i).isThereHtx())
                  missingHtD_HtXNegLeaf3->Fill(intHtMissD->at(i).htx());

                if (intHtMissD->at(i).capBlock() == 0x301 && intHtMissD->at(i).capIndex() == 0 &&
                    intHtMissD->at(i).isThereHty())
                  missingHtD_HtYPosLeaf1->Fill(intHtMissD->at(i).hty());
                if (intHtMissD->at(i).capBlock() == 0x301 && intHtMissD->at(i).capIndex() == 1 &&
                    intHtMissD->at(i).isThereHty())
                  missingHtD_HtYPosLeaf2->Fill(intHtMissD->at(i).hty());
                if (intHtMissD->at(i).capBlock() == 0x301 && intHtMissD->at(i).capIndex() == 2 &&
                    intHtMissD->at(i).isThereHty())
                  missingHtD_HtYPosLeaf3->Fill(intHtMissD->at(i).hty());
                if (intHtMissD->at(i).capBlock() == 0x701 && intHtMissD->at(i).capIndex() == 0 &&
                    intHtMissD->at(i).isThereHty())
                  missingHtD_HtYNegLeaf1->Fill(intHtMissD->at(i).hty());
                if (intHtMissD->at(i).capBlock() == 0x701 && intHtMissD->at(i).capIndex() == 1 &&
                    intHtMissD->at(i).isThereHty())
                  missingHtD_HtYNegLeaf2->Fill(intHtMissD->at(i).hty());
                if (intHtMissD->at(i).capBlock() == 0x701 && intHtMissD->at(i).capIndex() == 2 &&
                    intHtMissD->at(i).isThereHty())
                  missingHtD_HtYNegLeaf3->Fill(intHtMissD->at(i).hty());
              }
            }
          }
        }
      }
    }

    if ((isMissingEError && doMissingETDebug_) || (isMissingHError && doMissingHTDebug_)) {
      plotMissingEErrors(missingEtD, missingEtE, missingHtD, missingHtE, caloRegions, intJetsE, intHtMissD);
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void GctErrorAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void GctErrorAnalyzer::endJob() {}

void GctErrorAnalyzer::plotRCTRegions(const edm::Handle<L1CaloRegionCollection> &caloRegions) {
  //if more than one Bx is readout per event, then caloRegions->size() will be some multiple of 396
  for (unsigned int i = 0; i < caloRegions->size(); i++) {
    //if the RCTMBx flag is set to true, write out all the info into the same histogram
    //otherwise only the RCTTrigBx will be written out - could skip (RCT_REGION_QUANTA-1) events here to speed things up...
    if (doRCTMBx_ || caloRegions->at(i).bx() == RCTTrigBx_) {
      if (caloRegions->at(i).et() > 0)
        RCT_EtEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi(), caloRegions->at(i).et());
      if (caloRegions->at(i).tauVeto())
        RCT_TvEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi());
      if (caloRegions->at(i).fineGrain())
        RCT_FgEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi());
      if (caloRegions->at(i).overFlow())
        RCT_OfEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi());
    }
  }
}

void GctErrorAnalyzer::plotIsoEm(const edm::Handle<L1GctEmCandCollection> &isoEgD,
                                 const edm::Handle<L1GctEmCandCollection> &isoEgE) {
  //loop over all the data candidates - if multiple bx, then this should be a multiple of GCT_OBJECT_QUANTA
  for (unsigned int i = 0; i < isoEgD->size(); i++) {
    //if the GCTMBx flag is set, plot all Bx for this quantity, otherwise only plot Bx = GCTTrigBx_
    if (doGCTMBx_ || isoEgD->at(i).bx() == GCTTrigBx_) {
      isoEgD_Rank_->Fill(isoEgD->at(i).rank());
      if (isoEgD->at(i).rank() > 0) {
        isoEgD_EtEtaPhi_->Fill(isoEgD->at(i).regionId().ieta(), isoEgD->at(i).regionId().iphi(), isoEgD->at(i).rank());
        isoEgD_OccEtaPhi_->Fill(isoEgD->at(i).regionId().ieta(), isoEgD->at(i).regionId().iphi());
      }
    }
  }
  //now repeat for the emulator candidates
  for (unsigned int i = 0; i < isoEgE->size(); i++) {
    if (doEmuMBx_ || isoEgE->at(i).bx() == EmuTrigBx_) {
      isoEgE_Rank_->Fill(isoEgE->at(i).rank());
      if (isoEgE->at(i).rank() > 0) {
        isoEgE_EtEtaPhi_->Fill(isoEgE->at(i).regionId().ieta(), isoEgE->at(i).regionId().iphi(), isoEgE->at(i).rank());
        isoEgE_OccEtaPhi_->Fill(isoEgE->at(i).regionId().ieta(), isoEgE->at(i).regionId().iphi());
      }
    }
  }
}

void GctErrorAnalyzer::plotNonIsoEm(const edm::Handle<L1GctEmCandCollection> &nonIsoEgD,
                                    const edm::Handle<L1GctEmCandCollection> &nonIsoEgE) {
  //loop over all the data candidates - if multiple bx, then this should be a multiple of GCT_OBJECT_QUANTA
  for (unsigned int i = 0; i < nonIsoEgD->size(); i++) {
    //if the GCTMBx flag is set, plot all Bx for this quantity, otherwise only plot Bx = GCTTrigBx_
    if (doGCTMBx_ || nonIsoEgD->at(i).bx() == GCTTrigBx_) {
      nonIsoEgD_Rank_->Fill(nonIsoEgD->at(i).rank());
      if (nonIsoEgD->at(i).rank() > 0) {
        nonIsoEgD_EtEtaPhi_->Fill(
            nonIsoEgD->at(i).regionId().ieta(), nonIsoEgD->at(i).regionId().iphi(), nonIsoEgD->at(i).rank());
        nonIsoEgD_OccEtaPhi_->Fill(nonIsoEgD->at(i).regionId().ieta(), nonIsoEgD->at(i).regionId().iphi());
      }
    }
  }
  //now repeat for the emulator candidates
  for (unsigned int i = 0; i < nonIsoEgE->size(); i++) {
    if (doEmuMBx_ || nonIsoEgE->at(i).bx() == EmuTrigBx_) {
      nonIsoEgE_Rank_->Fill(nonIsoEgE->at(i).rank());
      if (nonIsoEgE->at(i).rank() > 0) {
        nonIsoEgE_EtEtaPhi_->Fill(
            nonIsoEgE->at(i).regionId().ieta(), nonIsoEgE->at(i).regionId().iphi(), nonIsoEgE->at(i).rank());
        nonIsoEgE_OccEtaPhi_->Fill(nonIsoEgE->at(i).regionId().ieta(), nonIsoEgE->at(i).regionId().iphi());
      }
    }
  }
}

void GctErrorAnalyzer::plotEGErrors(const edm::Handle<L1GctEmCandCollection> &isoEgD,
                                    const edm::Handle<L1GctEmCandCollection> &isoEgE,
                                    const edm::Handle<L1GctEmCandCollection> &nonIsoEgD,
                                    const edm::Handle<L1GctEmCandCollection> &nonIsoEgE,
                                    const edm::Handle<L1CaloEmCollection> &emRegions) {
  std::string errorDirName = "err_";
  if (isIsoError)
    errorDirName.append("I");
  if (isNonIsoError)
    errorDirName.append("N");
  std::stringstream caseNumber;
  caseNumber << eventNumber;
  errorDirName.append(caseNumber.str());
  TFileDirectory errorDir = errorHistCat.at(0).mkdir(errorDirName);

  TH2I *errorEmRegionIsoEtEtaPhi_ = errorDir.make<TH2I>("errorEmRegionIsoEtEtaPhi",
                                                        "errorEmRegionIsoEtEtaPhi;#eta (GCT Units);#phi (GCT Units)",
                                                        22,
                                                        -0.5,
                                                        21.5,
                                                        18,
                                                        -0.5,
                                                        17.5);
  TH2I *errorEmRegionNonIsoEtEtaPhi_ =
      errorDir.make<TH2I>("errorEmRegionNonIsoEtEtaPhi",
                          "errorEmRegionNonIsoEtEtaPhi;#eta (GCT Units);#phi (GCT Units)",
                          22,
                          -0.5,
                          21.5,
                          18,
                          -0.5,
                          17.5);
  TH2I *errorIsoEtEtaPhiD_ = errorDir.make<TH2I>(
      "errorIsoEtEtaPhiD", "errorIsoEtEtaPhiD;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  TH2I *errorNonIsoEtEtaPhiD_ = errorDir.make<TH2I>(
      "errorNonIsoEtEtaPhiD", "errorNonIsoEtEtaPhiD;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  TH2I *errorIsoEtEtaPhiE_ = errorDir.make<TH2I>(
      "errorIsoEtEtaPhiE", "errorIsoEtEtaPhiE;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  TH2I *errorNonIsoEtEtaPhiE_ = errorDir.make<TH2I>(
      "errorNonIsoEtEtaPhiE", "errorNonIsoEtEtaPhiE;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);

  //fill the EM input collection
  //should only fill the correct bx for emRegions - and since this is showing an error in the comparison, we should plot the input to this comparison i.e. Bx=RCTTrigBx
  //this assumes that comparison is done on the central Bx i.e. RctBx=0 corresponds to GctBx=0, and EmuBx=0 takes RctBx=0
  for (unsigned int i = 0; i < emRegions->size(); i++) {
    if (emRegions->at(i).bx() == RCTTrigBx_) {
      if (emRegions->at(i).isolated()) {
        if (emRegions->at(i).rank() > 0)
          errorEmRegionIsoEtEtaPhi_->Fill(
              emRegions->at(i).regionId().ieta(), emRegions->at(i).regionId().iphi(), emRegions->at(i).rank());
      } else {
        if (emRegions->at(i).rank() > 0)
          errorEmRegionNonIsoEtEtaPhi_->Fill(
              emRegions->at(i).regionId().ieta(), emRegions->at(i).regionId().iphi(), emRegions->at(i).rank());
      }
    }
  }

  //no need to have the rank plot, because you can't have two electrons in the same place (eta,phi), in the same event...
  //in this case, since we're actually comparing the GCTTrigBx_ with the EmuTrigBx_, we plot these individually
  for (unsigned int i = 0; i < isoEgD->size(); i++) {
    if (isoEgD->at(i).bx() == GCTTrigBx_) {
      if (isoEgD->at(i).rank() > 0)
        errorIsoEtEtaPhiD_->Fill(
            isoEgD->at(i).regionId().ieta(), isoEgD->at(i).regionId().iphi(), isoEgD->at(i).rank());
    }
  }
  for (unsigned int i = 0; i < nonIsoEgD->size(); i++) {
    if (nonIsoEgD->at(i).bx() == GCTTrigBx_) {
      if (nonIsoEgD->at(i).rank() > 0)
        errorNonIsoEtEtaPhiD_->Fill(
            nonIsoEgD->at(i).regionId().ieta(), nonIsoEgD->at(i).regionId().iphi(), nonIsoEgD->at(i).rank());
    }
  }

  //now for the emulator candidates
  for (unsigned int i = 0; i < isoEgE->size(); i++) {
    if (isoEgE->at(i).bx() == EmuTrigBx_) {
      if (isoEgE->at(i).rank() > 0)
        errorIsoEtEtaPhiE_->Fill(
            isoEgE->at(i).regionId().ieta(), isoEgE->at(i).regionId().iphi(), isoEgE->at(i).rank());
    }
  }
  for (unsigned int i = 0; i < nonIsoEgE->size(); i++) {
    if (nonIsoEgE->at(i).bx() == EmuTrigBx_) {
      if (nonIsoEgE->at(i).rank() > 0)
        errorNonIsoEtEtaPhiE_->Fill(
            nonIsoEgE->at(i).regionId().ieta(), nonIsoEgE->at(i).regionId().iphi(), nonIsoEgE->at(i).rank());
    }
  }
}

void GctErrorAnalyzer::plotCenJets(const edm::Handle<L1GctJetCandCollection> &cenJetsD,
                                   const edm::Handle<L1GctJetCandCollection> &cenJetsE) {
  for (unsigned int i = 0; i < cenJetsD->size(); i++) {
    if (doGCTMBx_ || cenJetsD->at(i).bx() == GCTTrigBx_) {
      cenJetD_Rank_->Fill(cenJetsD->at(i).rank());
      if (cenJetsD->at(i).rank() > 0) {
        cenJetD_EtEtaPhi_->Fill(
            cenJetsD->at(i).regionId().ieta(), cenJetsD->at(i).regionId().iphi(), cenJetsD->at(i).rank());
        cenJetD_OccEtaPhi_->Fill(cenJetsD->at(i).regionId().ieta(), cenJetsD->at(i).regionId().iphi());
      }
    }
  }

  for (unsigned int i = 0; i < cenJetsE->size(); i++) {
    if (doEmuMBx_ || cenJetsE->at(i).bx() == EmuTrigBx_) {
      cenJetE_Rank_->Fill(cenJetsE->at(i).rank());
      if (cenJetsE->at(i).rank() > 0) {
        cenJetE_EtEtaPhi_->Fill(
            cenJetsE->at(i).regionId().ieta(), cenJetsE->at(i).regionId().iphi(), cenJetsE->at(i).rank());
        cenJetE_OccEtaPhi_->Fill(cenJetsE->at(i).regionId().ieta(), cenJetsE->at(i).regionId().iphi());
      }
    }
  }
}

void GctErrorAnalyzer::plotTauJets(const edm::Handle<L1GctJetCandCollection> &tauJetsD,
                                   const edm::Handle<L1GctJetCandCollection> &tauJetsE) {
  for (unsigned int i = 0; i < tauJetsD->size(); i++) {
    if (doGCTMBx_ || tauJetsD->at(i).bx() == GCTTrigBx_) {
      tauJetD_Rank_->Fill(tauJetsD->at(i).rank());
      if (tauJetsD->at(i).rank() > 0) {
        tauJetD_EtEtaPhi_->Fill(
            tauJetsD->at(i).regionId().ieta(), tauJetsD->at(i).regionId().iphi(), tauJetsD->at(i).rank());
        tauJetD_OccEtaPhi_->Fill(tauJetsD->at(i).regionId().ieta(), tauJetsD->at(i).regionId().iphi());
      }
    }
  }

  for (unsigned int i = 0; i < tauJetsE->size(); i++) {
    if (doEmuMBx_ || tauJetsE->at(i).bx() == EmuTrigBx_) {
      tauJetE_Rank_->Fill(tauJetsE->at(i).rank());
      if (tauJetsE->at(i).rank() > 0) {
        tauJetE_EtEtaPhi_->Fill(
            tauJetsE->at(i).regionId().ieta(), tauJetsE->at(i).regionId().iphi(), tauJetsE->at(i).rank());
        tauJetE_OccEtaPhi_->Fill(tauJetsE->at(i).regionId().ieta(), tauJetsE->at(i).regionId().iphi());
      }
    }
  }
}

void GctErrorAnalyzer::plotForJets(const edm::Handle<L1GctJetCandCollection> &forJetsD,
                                   const edm::Handle<L1GctJetCandCollection> &forJetsE) {
  for (unsigned int i = 0; i < forJetsD->size(); i++) {
    if (doGCTMBx_ || forJetsD->at(i).bx() == GCTTrigBx_) {
      forJetD_Rank_->Fill(forJetsD->at(i).rank());
      if (forJetsD->at(i).rank() > 0) {
        forJetD_EtEtaPhi_->Fill(
            forJetsD->at(i).regionId().ieta(), forJetsD->at(i).regionId().iphi(), forJetsD->at(i).rank());
        forJetD_OccEtaPhi_->Fill(forJetsD->at(i).regionId().ieta(), forJetsD->at(i).regionId().iphi());
      }
    }
  }

  for (unsigned int i = 0; i < forJetsE->size(); i++) {
    if (doEmuMBx_ || forJetsE->at(i).bx() == EmuTrigBx_) {
      forJetE_Rank_->Fill(forJetsE->at(i).rank());
      if (forJetsE->at(i).rank() > 0) {
        forJetE_EtEtaPhi_->Fill(
            forJetsE->at(i).regionId().ieta(), forJetsE->at(i).regionId().iphi(), forJetsE->at(i).rank());
        forJetE_OccEtaPhi_->Fill(forJetsE->at(i).regionId().ieta(), forJetsE->at(i).regionId().iphi());
      }
    }
  }
}

void GctErrorAnalyzer::plotIntJets(const edm::Handle<L1GctInternJetDataCollection> &intJetsE) {
  jetData intJet;
  std::vector<jetData> intJetCollection(
      NUM_INT_JETS);  //define fixed size for the vector to avoid reallocation (i.e. max size possible)

  //since we don't read out the intermediate (i.e. leaf card) jets, we can only plot the emulator distributions
  //the 1st-4th jet Et will prove useful in understanding and motivating cuts on individual jets in HT and MHT.
  for (unsigned int i = 0; i < intJetsE->size(); i++) {
    if (doEmuMBx_ || intJetsE->at(i).bx() == EmuTrigBx_) {
      //the intermediate jets are not sorted in terms of Et so
      //in order to do this independently of the data format,
      //copy to a user defined struct and sort that way
      intJet.et = intJetsE->at(i).et();
      intJet.phi = intJetsE->at(i).phi();
      intJet.eta = intJetsE->at(i).eta();
      intJetCollection.at(i % NUM_INT_JETS) = intJet;

      //remember, if the event has 1 overflowed jet, then we fill the internal jet dist overflow histogram
      //and skip the event - there is no point looking at the leading jet distributions etc for an event
      //with an overflowed jet - this will imply HT, ET, MET and MHT all overflow too.
      if (intJetsE->at(i).oflow()) {
        intJetE_Of_->Fill(intJetsE->at(i).oflow());
        return;
      }

      //plot the (et,eta,phi) distribution of the intermediate jets (for non-zero et)
      if (intJetsE->at(i).et())
        intJetEtEtaPhiE_->Fill(
            intJetsE->at(i).regionId().ieta(), intJetsE->at(i).regionId().iphi(), intJetsE->at(i).et());
    }
  }

  //if we get this far, there are no jets with an overflow bit so fill the overflow histogram and
  //sort the intJetCollection according to the rule defined in sortJets (i.e. largest et first)
  intJetE_Of_->Fill(0);
  std::sort(intJetCollection.begin(), intJetCollection.end(), sortJets);

  std::vector<TH1I *> leadingJetDist(4);
  leadingJetDist.at(0) = intJetE_Jet1Et_;
  leadingJetDist.at(1) = intJetE_Jet2Et_;
  leadingJetDist.at(2) = intJetE_Jet3Et_;
  leadingJetDist.at(3) = intJetE_Jet4Et_;

  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int currentEt = 0;
  while (intJetCollection.at(i).et > 0) {
    if (j < leadingJetDist.size()) {
      if (i == 0) {
        leadingJetDist.at(j)->Fill(intJetCollection.at(i).et);
        currentEt = intJetCollection.at(i).et;
        j++;
      } else {
        if (intJetCollection.at(i).et < currentEt) {
          leadingJetDist.at(j)->Fill(intJetCollection.at(i).et);
          currentEt = intJetCollection.at(i).et;
          j++;
        }
      }
    }

    intJetE_Et_->Fill(intJetCollection.at(i).et);
    i++;
  }
  return;
}

bool GctErrorAnalyzer::sortJets(const jetData &jet1, const jetData &jet2) { return jet1.et > jet2.et; }

template <class T>
bool GctErrorAnalyzer::checkCollections(const T &collection, const unsigned int &constraint, const std::string &label) {
  //unfortunately, the dataformats are not consistent with the name() method (i.e. some have it, others don't)
  //and a typeof() function doesn't exist in ANSI C++, so to identify the templated type, we pass a std::string

  if (!collection.isValid()) {
    edm::LogWarning("DataNotFound") << " Could not find " << label << " label";
    return false;
  }
  if (collection->size() % constraint != 0 || collection->empty()) {
    edm::LogWarning("CollectionSizeError")
        << " " << label << " collection size is " << collection->size() << ", expected multiple of " << constraint;
    return false;
  }

  return true;
}

void GctErrorAnalyzer::plotJetErrors(const edm::Handle<L1GctJetCandCollection> &cenJetsD,
                                     const edm::Handle<L1GctJetCandCollection> &cenJetsE,
                                     const edm::Handle<L1GctJetCandCollection> &tauJetsD,
                                     const edm::Handle<L1GctJetCandCollection> &tauJetsE,
                                     const edm::Handle<L1GctJetCandCollection> &forJetsD,
                                     const edm::Handle<L1GctJetCandCollection> &forJetsE,
                                     const edm::Handle<L1CaloRegionCollection> &caloRegions) {
  std::string errorDirName = "err_";
  if (isCenJetError)
    errorDirName.append("C");
  if (isTauJetError)
    errorDirName.append("T");
  if (isForJetError)
    errorDirName.append("F");
  std::stringstream caseNumber;
  caseNumber << eventNumber;
  errorDirName.append(caseNumber.str());
  TFileDirectory errorDir = errorHistCat.at(1).mkdir(errorDirName);

  TH2I *errorRegionEtEtaPhi_ = errorDir.make<TH2I>(
      "errorRegionEtEtaPhi", "errorRegionEtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  TH2I *errorRegionTvEtaPhi_ = errorDir.make<TH2I>(
      "errorRegionTvEtaPhi", "errorRegionTvEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  TH2I *errorRegionOfEtaPhi_ = errorDir.make<TH2I>(
      "errorRegionOfEtaPhi", "errorRegionOfEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);

  //make sure to only plot the caloRegion bx which corresponds to the data vs emulator comparison
  for (unsigned int i = 0; i < caloRegions->size(); i++) {
    if (caloRegions->at(i).bx() == RCTTrigBx_) {
      if (caloRegions->at(i).et() > 0)
        errorRegionEtEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi(), caloRegions->at(i).et());
      if (caloRegions->at(i).tauVeto())
        errorRegionTvEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi());
      if (caloRegions->at(i).overFlow())
        errorRegionOfEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi());
    }
  }

  TH2I *cenJet_errorEtEtaPhiData_ = errorDir.make<TH2I>("cenJet_errorEtEtaPhiData",
                                                        "cenJet_errorEtEtaPhiData;#eta (GCT Units);#phi (GCT Units)",
                                                        22,
                                                        -0.5,
                                                        21.5,
                                                        18,
                                                        -0.5,
                                                        17.5);
  TH2I *cenJet_errorEtEtaPhiEmu_ = errorDir.make<TH2I>("cenJet_errorEtEtaPhiEmu",
                                                       "cenJet_errorEtEtaPhiEmu;#eta (GCT Units);#phi (GCT Units)",
                                                       22,
                                                       -0.5,
                                                       21.5,
                                                       18,
                                                       -0.5,
                                                       17.5);
  TH2I *tauJet_errorEtEtaPhiData_ = errorDir.make<TH2I>("tauJet_errorEtEtaPhiData",
                                                        "tauJet_errorEtEtaPhiData;#eta (GCT Units);#phi (GCT Units)",
                                                        22,
                                                        -0.5,
                                                        21.5,
                                                        18,
                                                        -0.5,
                                                        17.5);
  TH2I *tauJet_errorEtEtaPhiEmu_ = errorDir.make<TH2I>("tauJet_errorEtEtaPhiEmu",
                                                       "tauJet_errorEtEtaPhiEmu;#eta (GCT Units);#phi (GCT Units)",
                                                       22,
                                                       -0.5,
                                                       21.5,
                                                       18,
                                                       -0.5,
                                                       17.5);
  TH2I *forJet_errorEtEtaPhiData_ = errorDir.make<TH2I>("forJet_errorEtEtaPhiData",
                                                        "forJet_errorEtEtaPhiData;#eta (GCT Units);#phi (GCT Units)",
                                                        22,
                                                        -0.5,
                                                        21.5,
                                                        18,
                                                        -0.5,
                                                        17.5);
  TH2I *forJet_errorEtEtaPhiEmu_ = errorDir.make<TH2I>("forJet_errorEtEtaPhiEmu",
                                                       "forJet_errorEtEtaPhiEmu;#eta (GCT Units);#phi (GCT Units)",
                                                       22,
                                                       -0.5,
                                                       21.5,
                                                       18,
                                                       -0.5,
                                                       17.5);

  //first plot the data candiates for the Trigger Bx that this error corresponds to
  for (unsigned int i = 0; i < cenJetsD->size(); i++) {
    if (cenJetsD->at(i).bx() == GCTTrigBx_) {
      if (cenJetsD->at(i).rank() > 0)
        cenJet_errorEtEtaPhiData_->Fill(
            cenJetsD->at(i).regionId().ieta(), cenJetsD->at(i).regionId().iphi(), cenJetsD->at(i).rank());
    }
  }
  for (unsigned int i = 0; i < tauJetsD->size(); i++) {
    if (tauJetsD->at(i).bx() == GCTTrigBx_) {
      if (tauJetsD->at(i).rank() > 0)
        tauJet_errorEtEtaPhiData_->Fill(
            tauJetsD->at(i).regionId().ieta(), tauJetsD->at(i).regionId().iphi(), tauJetsD->at(i).rank());
    }
  }
  for (unsigned int i = 0; i < forJetsD->size(); i++) {
    if (forJetsD->at(i).bx() == GCTTrigBx_) {
      if (forJetsD->at(i).rank() > 0)
        forJet_errorEtEtaPhiData_->Fill(
            forJetsD->at(i).regionId().ieta(), forJetsD->at(i).regionId().iphi(), forJetsD->at(i).rank());
    }
  }

  //now the emulator candidates
  for (unsigned int i = 0; i < cenJetsE->size(); i++) {
    if (cenJetsE->at(i).bx() == EmuTrigBx_) {
      if (cenJetsE->at(i).rank() > 0)
        cenJet_errorEtEtaPhiEmu_->Fill(
            cenJetsE->at(i).regionId().ieta(), cenJetsE->at(i).regionId().iphi(), cenJetsE->at(i).rank());
    }
  }
  for (unsigned int i = 0; i < tauJetsE->size(); i++) {
    if (tauJetsE->at(i).bx() == EmuTrigBx_) {
      if (tauJetsE->at(i).rank() > 0)
        tauJet_errorEtEtaPhiEmu_->Fill(
            tauJetsE->at(i).regionId().ieta(), tauJetsE->at(i).regionId().iphi(), tauJetsE->at(i).rank());
    }
  }
  for (unsigned int i = 0; i < forJetsE->size(); i++) {
    if (forJetsE->at(i).bx() == EmuTrigBx_) {
      if (forJetsE->at(i).rank() > 0)
        forJet_errorEtEtaPhiEmu_->Fill(
            forJetsE->at(i).regionId().ieta(), forJetsE->at(i).regionId().iphi(), forJetsE->at(i).rank());
    }
  }
}

void GctErrorAnalyzer::plotHFRingSums(const edm::Handle<L1GctHFRingEtSumsCollection> &hfRingSumsD,
                                      const edm::Handle<L1GctHFRingEtSumsCollection> &hfRingSumsE) {
  for (unsigned int i = 0; i < hfRingSumsD->size(); i++) {
    if (doGCTMBx_ || hfRingSumsD->at(i).bx() == GCTTrigBx_) {
      //there are 4 rings - just fill the histograms
      hfRingSumD_1pos_->Fill(hfRingSumsD->at(i).etSum(0));
      hfRingSumD_1neg_->Fill(hfRingSumsD->at(i).etSum(1));
      hfRingSumD_2pos_->Fill(hfRingSumsD->at(i).etSum(2));
      hfRingSumD_2neg_->Fill(hfRingSumsD->at(i).etSum(3));
    }
  }

  for (unsigned int i = 0; i < hfRingSumsE->size(); i++) {
    if (doEmuMBx_ || hfRingSumsE->at(i).bx() == EmuTrigBx_) {
      hfRingSumE_1pos_->Fill(hfRingSumsE->at(i).etSum(0));
      hfRingSumE_1neg_->Fill(hfRingSumsE->at(i).etSum(1));
      hfRingSumE_2pos_->Fill(hfRingSumsE->at(i).etSum(2));
      hfRingSumE_2neg_->Fill(hfRingSumsE->at(i).etSum(3));
    }
  }
}

void GctErrorAnalyzer::plotHFBitCounts(const edm::Handle<L1GctHFBitCountsCollection> &hfBitCountsD,
                                       const edm::Handle<L1GctHFBitCountsCollection> &hfBitCountsE) {
  for (unsigned int i = 0; i < hfBitCountsD->size(); i++) {
    if (doGCTMBx_ || hfBitCountsD->at(i).bx() == GCTTrigBx_) {
      //there are 4 rings - just fill the histograms
      hfBitCountD_1pos_->Fill(hfBitCountsD->at(i).bitCount(0));
      hfBitCountD_1neg_->Fill(hfBitCountsD->at(i).bitCount(1));
      hfBitCountD_2pos_->Fill(hfBitCountsD->at(i).bitCount(2));
      hfBitCountD_2neg_->Fill(hfBitCountsD->at(i).bitCount(3));
    }
  }
  for (unsigned int i = 0; i < hfBitCountsE->size(); i++) {
    if (doEmuMBx_ || hfBitCountsE->at(i).bx() == EmuTrigBx_) {
      hfBitCountE_1pos_->Fill(hfBitCountsE->at(i).bitCount(0));
      hfBitCountE_1neg_->Fill(hfBitCountsE->at(i).bitCount(1));
      hfBitCountE_2pos_->Fill(hfBitCountsE->at(i).bitCount(2));
      hfBitCountE_2neg_->Fill(hfBitCountsE->at(i).bitCount(3));
    }
  }
}

void GctErrorAnalyzer::plotHFErrors(const edm::Handle<L1GctHFRingEtSumsCollection> &hfRingSumsD,
                                    const edm::Handle<L1GctHFRingEtSumsCollection> &hfRingSumsE,
                                    const edm::Handle<L1GctHFBitCountsCollection> &hfBitCountsD,
                                    const edm::Handle<L1GctHFBitCountsCollection> &hfBitCountsE,
                                    const edm::Handle<L1CaloRegionCollection> &caloRegions) {
  std::string errorDirName = "err_";
  if (isRingSumError)
    errorDirName.append("R");
  if (isBitCountError)
    errorDirName.append("B");
  std::stringstream caseNumber;
  caseNumber << eventNumber;
  errorDirName.append(caseNumber.str());
  TFileDirectory errorDir = errorHistCat.at(2).mkdir(errorDirName);

  TH2I *errorRegionEtEtaPhi_ = errorDir.make<TH2I>(
      "errorRegionEtEtaPhi", "errorRegionEtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  TH2I *errorRegionFgEtaPhi_ = errorDir.make<TH2I>(
      "errorRegionFgEtaPhi", "errorRegionFgEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  TH2I *errorRegionOfEtaPhi_ = errorDir.make<TH2I>(
      "errorRegionOfEtaPhi", "errorRegionOfEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);

  TH1I *errorHFRingSumD_1pos_ =
      errorDir.make<TH1I>("errorHFRingSumD_1+", "errorHFRingSumD_1+;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFRingSumD_2pos_ =
      errorDir.make<TH1I>("errorHFRingSumD_2+", "errorHFRingSumD_2+;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFRingSumD_1neg_ =
      errorDir.make<TH1I>("errorHFRingSumD_1-", "errorHFRingSumD_1-;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFRingSumD_2neg_ =
      errorDir.make<TH1I>("errorHFRingSumD_2-", "errorHFRingSumD_2-;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFRingSumE_1pos_ =
      errorDir.make<TH1I>("errorHFRingSumE_1+", "errorHFRingSumE_1+;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFRingSumE_2pos_ =
      errorDir.make<TH1I>("errorHFRingSumE_2+", "errorHFRingSumE_2+;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFRingSumE_1neg_ =
      errorDir.make<TH1I>("errorHFRingSumE_1-", "errorHFRingSumE_1-;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFRingSumE_2neg_ =
      errorDir.make<TH1I>("errorHFRingSumE_2-", "errorHFRingSumE_2-;Rank;Number of Events", 8, -0.5, 7.5);

  TH1I *errorHFBitCountD_1pos_ =
      errorDir.make<TH1I>("errorHFBitCountD_1+", "errorHFBitCountD_1+;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFBitCountD_2pos_ =
      errorDir.make<TH1I>("errorHFBitCountD_2+", "errorHFBitCountD_2+;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFBitCountD_1neg_ =
      errorDir.make<TH1I>("errorHFBitCountD_1-", "errorHFBitCountD_1-;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFBitCountD_2neg_ =
      errorDir.make<TH1I>("errorHFBitCountD_2-", "errorHFBitCountD_2-;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFBitCountE_1pos_ =
      errorDir.make<TH1I>("errorHFBitCountE_1+", "errorHFBitCountE_1+;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFBitCountE_2pos_ =
      errorDir.make<TH1I>("errorHFBitCountE_2+", "errorHFBitCountE_2+;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFBitCountE_1neg_ =
      errorDir.make<TH1I>("errorHFBitCountE_1-", "errorHFBitCountE_1-;Rank;Number of Events", 8, -0.5, 7.5);
  TH1I *errorHFBitCountE_2neg_ =
      errorDir.make<TH1I>("errorHFBitCountE_2-", "errorHFBitCountE_2-;Rank;Number of Events", 8, -0.5, 7.5);

  for (unsigned int i = 0; i < caloRegions->size(); i++) {
    if (caloRegions->at(i).bx() == RCTTrigBx_) {
      if (caloRegions->at(i).et() > 0)
        errorRegionEtEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi(), caloRegions->at(i).et());
      if (caloRegions->at(i).fineGrain())
        errorRegionFgEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi());
      if (caloRegions->at(i).overFlow())
        errorRegionOfEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi());
    }
  }

  for (unsigned int i = 0; i < hfRingSumsD->size(); i++) {
    if (hfRingSumsD->at(i).bx() == GCTTrigBx_) {
      errorHFRingSumD_1pos_->Fill(hfRingSumsD->at(i).etSum(0));
      errorHFRingSumD_1neg_->Fill(hfRingSumsD->at(i).etSum(1));
      errorHFRingSumD_2pos_->Fill(hfRingSumsD->at(i).etSum(2));
      errorHFRingSumD_2neg_->Fill(hfRingSumsD->at(i).etSum(3));
    }
  }
  for (unsigned int i = 0; i < hfRingSumsE->size(); i++) {
    if (hfRingSumsE->at(i).bx() == EmuTrigBx_) {
      errorHFRingSumE_1pos_->Fill(hfRingSumsE->at(i).etSum(0));
      errorHFRingSumE_1neg_->Fill(hfRingSumsE->at(i).etSum(1));
      errorHFRingSumE_2pos_->Fill(hfRingSumsE->at(i).etSum(2));
      errorHFRingSumE_2neg_->Fill(hfRingSumsE->at(i).etSum(3));
    }
  }

  for (unsigned int i = 0; i < hfBitCountsD->size(); i++) {
    if (hfBitCountsD->at(i).bx() == GCTTrigBx_) {
      errorHFBitCountD_1pos_->Fill(hfBitCountsD->at(i).bitCount(0));
      errorHFBitCountD_1neg_->Fill(hfBitCountsD->at(i).bitCount(1));
      errorHFBitCountD_2pos_->Fill(hfBitCountsD->at(i).bitCount(2));
      errorHFBitCountD_2neg_->Fill(hfBitCountsD->at(i).bitCount(3));
    }
  }
  for (unsigned int i = 0; i < hfBitCountsE->size(); i++) {
    if (hfBitCountsE->at(i).bx() == EmuTrigBx_) {
      errorHFBitCountE_1pos_->Fill(hfBitCountsE->at(i).bitCount(0));
      errorHFBitCountE_1neg_->Fill(hfBitCountsE->at(i).bitCount(1));
      errorHFBitCountE_2pos_->Fill(hfBitCountsE->at(i).bitCount(2));
      errorHFBitCountE_2neg_->Fill(hfBitCountsE->at(i).bitCount(3));
    }
  }
}

void GctErrorAnalyzer::plotTotalE(const edm::Handle<L1GctEtTotalCollection> &totalEtD,
                                  const edm::Handle<L1GctEtTotalCollection> &totalEtE) {
  for (unsigned int i = 0; i < totalEtD->size(); i++) {
    if (doGCTMBx_ || totalEtD->at(i).bx() == GCTTrigBx_) {
      totalEtD_Of_->Fill(totalEtD->at(i).overFlow());
      if (!totalEtD->at(i).overFlow())
        totalEtD_->Fill(totalEtD->at(i).et());
    }
  }
  for (unsigned int i = 0; i < totalEtE->size(); i++) {
    if (doEmuMBx_ || totalEtE->at(i).bx() == EmuTrigBx_) {
      totalEtE_Of_->Fill(totalEtE->at(i).overFlow());
      if (!totalEtE->at(i).overFlow())
        totalEtE_->Fill(totalEtE->at(i).et());
    }
  }
}

void GctErrorAnalyzer::plotTotalH(const edm::Handle<L1GctEtHadCollection> &totalHtD,
                                  const edm::Handle<L1GctEtHadCollection> &totalHtE) {
  for (unsigned int i = 0; i < totalHtD->size(); i++) {
    if (doGCTMBx_ || totalHtD->at(i).bx() == GCTTrigBx_) {
      totalHtD_Of_->Fill(totalHtD->at(i).overFlow());
      if (!totalHtD->at(i).overFlow())
        totalHtD_->Fill(totalHtD->at(i).et());
    }
  }
  for (unsigned int i = 0; i < totalHtE->size(); i++) {
    if (doEmuMBx_ || totalHtE->at(i).bx() == EmuTrigBx_) {
      totalHtE_Of_->Fill(totalHtE->at(i).overFlow());
      if (!totalHtE->at(i).overFlow())
        totalHtE_->Fill(totalHtE->at(i).et());
    }
  }
}

void GctErrorAnalyzer::plotTotalEErrors(const edm::Handle<L1GctEtTotalCollection> &totalEtD,
                                        const edm::Handle<L1GctEtTotalCollection> &totalEtE,
                                        const edm::Handle<L1GctEtHadCollection> &totalHtD,
                                        const edm::Handle<L1GctEtHadCollection> &totalHtE,
                                        const edm::Handle<L1CaloRegionCollection> &caloRegions) {
  std::string errorDirName = "err_";
  if (isTotalEError)
    errorDirName.append("E");
  if (isTotalHError)
    errorDirName.append("H");
  std::stringstream caseNumber;
  caseNumber << eventNumber;
  errorDirName.append(caseNumber.str());
  TFileDirectory errorDir = errorHistCat.at(3).mkdir(errorDirName);

  TH2I *errorRegionEtEtaPhi_ = errorDir.make<TH2I>(
      "errorRegionEtEtaPhi", "errorRegionEtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  TH2I *errorRegionOfEtaPhi_ = errorDir.make<TH2I>(
      "errorRegionOfEtaPhi", "errorRegionOfEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  TH1I *errorTotalEtD_ =
      errorDir.make<TH1I>("errorTotalEtD", "errorTotalEtD;E_{T};Number of Events", 1024, -0.5, 1023.5);
  TH1I *errorTotalEtD_Of_ =
      errorDir.make<TH1I>("errorTotalEtD_Of", "errorTotalEtD_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  TH1I *errorTotalEtE_ =
      errorDir.make<TH1I>("errorTotalEtE", "errorTotalEtE;E_{T};Number of Events", 1024, -0.5, 1023.5);
  TH1I *errorTotalEtE_Of_ =
      errorDir.make<TH1I>("errorTotalEtE_Of", "errorTotalEtE_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  TH1I *errorTotalHtD_ =
      errorDir.make<TH1I>("errorTotalHtD", "errorTotalHtD;E_{T};Number of Events", 1024, -0.5, 1023.5);
  TH1I *errorTotalHtD_Of_ =
      errorDir.make<TH1I>("errorTotalHtD_Of", "errorTotalHtD_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  TH1I *errorTotalHtE_ =
      errorDir.make<TH1I>("errorTotalHtE", "errorTotalHtE;E_{T};Number of Events", 1024, -0.5, 1023.5);
  TH1I *errorTotalHtE_Of_ =
      errorDir.make<TH1I>("errorTotalHtE_Of", "errorTotalHtE_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);

  //plot the region ET and OF bits
  for (unsigned int i = 0; i < caloRegions->size(); i++) {
    if (caloRegions->at(i).bx() == RCTTrigBx_) {
      if (caloRegions->at(i).et() > 0)
        errorRegionEtEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi(), caloRegions->at(i).et());
      if (caloRegions->at(i).overFlow())
        errorRegionOfEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi());
    }
  }
  //now plot the error ET
  for (unsigned int i = 0; i < totalEtD->size(); i++) {
    if (totalEtD->at(i).bx() == GCTTrigBx_) {
      errorTotalEtD_Of_->Fill(totalEtD->at(i).overFlow());
      if (!totalEtD->at(i).overFlow())
        errorTotalEtD_->Fill(totalEtD->at(i).et());
    }
  }
  for (unsigned int i = 0; i < totalEtE->size(); i++) {
    if (totalEtE->at(i).bx() == EmuTrigBx_) {
      errorTotalEtE_Of_->Fill(totalEtE->at(i).overFlow());
      if (!totalEtE->at(i).overFlow())
        errorTotalEtE_->Fill(totalEtE->at(i).et());
    }
  }
  //and now the error HT
  for (unsigned int i = 0; i < totalHtD->size(); i++) {
    if (totalHtD->at(i).bx() == GCTTrigBx_) {
      errorTotalHtD_Of_->Fill(totalHtD->at(i).overFlow());
      if (!totalHtD->at(i).overFlow())
        errorTotalHtD_->Fill(totalHtD->at(i).et());
    }
  }
  for (unsigned int i = 0; i < totalHtE->size(); i++) {
    if (totalHtE->at(i).bx() == EmuTrigBx_) {
      errorTotalHtE_Of_->Fill(totalHtE->at(i).overFlow());
      if (!totalHtE->at(i).overFlow())
        errorTotalHtE_->Fill(totalHtE->at(i).et());
    }
  }
}

void GctErrorAnalyzer::plotMissingEt(const edm::Handle<L1GctEtMissCollection> &missingEtD,
                                     const edm::Handle<L1GctEtMissCollection> &missingEtE) {
  for (unsigned int i = 0; i < missingEtD->size(); i++) {
    if (doGCTMBx_ || missingEtD->at(i).bx() == GCTTrigBx_) {
      missingEtD_Of_->Fill(missingEtD->at(i).overFlow());
      if (!missingEtD->at(i).overFlow() && missingEtD->at(i).et() > 0) {
        missingEtD_->Fill(missingEtD->at(i).et());
        missingEtD_Phi_->Fill(missingEtD->at(i).phi());
      }
    }
  }

  for (unsigned int i = 0; i < missingEtE->size(); i++) {
    if (doEmuMBx_ || missingEtE->at(i).bx() == EmuTrigBx_) {
      missingEtE_Of_->Fill(missingEtE->at(i).overFlow());
      if (!missingEtE->at(i).overFlow() && missingEtE->at(i).et()) {
        missingEtE_->Fill(missingEtE->at(i).et());
        missingEtE_Phi_->Fill(missingEtE->at(i).phi());
      }
    }
  }
}

void GctErrorAnalyzer::plotMissingHt(const edm::Handle<L1GctHtMissCollection> &missingHtD,
                                     const edm::Handle<L1GctHtMissCollection> &missingHtE) {
  for (unsigned int i = 0; i < missingHtD->size(); i++) {
    if (doGCTMBx_ || missingHtD->at(i).bx() == GCTTrigBx_) {
      missingHtD_Of_->Fill(missingHtD->at(i).overFlow());
      if (!missingHtD->at(i).overFlow() && missingHtD->at(i).et() > 0) {
        missingHtD_->Fill(missingHtD->at(i).et());
        missingHtD_Phi_->Fill(missingHtD->at(i).phi());
      }
    }
  }

  for (unsigned int i = 0; i < missingHtE->size(); i++) {
    if (doEmuMBx_ || missingHtE->at(i).bx() == EmuTrigBx_) {
      missingHtE_Of_->Fill(missingHtE->at(i).overFlow());
      if (!missingHtE->at(i).overFlow() && missingHtE->at(i).et() > 0) {
        missingHtE_->Fill(missingHtE->at(i).et());
        missingHtE_Phi_->Fill(missingHtE->at(i).phi());
      }
    }
  }
}

void GctErrorAnalyzer::plotMissingEErrors(const edm::Handle<L1GctEtMissCollection> &missingEtD,
                                          const edm::Handle<L1GctEtMissCollection> &missingEtE,
                                          const edm::Handle<L1GctHtMissCollection> &missingHtD,
                                          const edm::Handle<L1GctHtMissCollection> &missingHtE,
                                          edm::Handle<L1CaloRegionCollection> &caloRegions,
                                          const edm::Handle<L1GctInternJetDataCollection> &intJetsE,
                                          const edm::Handle<L1GctInternHtMissCollection> intMissingHtD) {
  std::string errorDirName = "err_";
  if (isMissingEError)
    errorDirName.append("E");
  if (isMissingHError)
    errorDirName.append("H");

  //added 05.03.2010 to highlight overflow errors in the missing energy sum calculation
  for (unsigned int i = 0; i < missingHtE->size(); i++) {
    if (missingHtE->at(i).bx() == EmuTrigBx_) {
      for (unsigned int j = 0; j < missingHtD->size(); j++) {
        if (missingHtD->at(j).bx() == GCTTrigBx_) {
          if (missingHtD->at(j).overFlow() != missingHtE->at(i).overFlow())
            errorDirName.append("O");
        }
      }
    }
  }

  std::stringstream caseNumber;
  caseNumber << eventNumber;
  errorDirName.append(caseNumber.str());
  TFileDirectory errorDir = errorHistCat.at(4).mkdir(errorDirName);

  TH2I *errorRegionEtEtaPhi_ = errorDir.make<TH2I>(
      "errorRegionEtEtaPhi", "errorRegionEtEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  TH2I *errorRegionOfEtaPhi_ = errorDir.make<TH2I>(
      "errorRegionOfEtaPhi", "errorRegionOfEtaPhi;#eta (GCT Units);#phi (GCT Units)", 22, -0.5, 21.5, 18, -0.5, 17.5);
  TH1I *errorMissingEtD_ =
      errorDir.make<TH1I>("errorMissingEtD", "errorMissingEtD;E_{T};Number of Events", 1024, -0.5, 1023.5);
  TH1I *errorMissingEtD_Of_ = errorDir.make<TH1I>(
      "errorMissingEtD_Of", "errorMissingEtD_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  TH1I *errorMissingEtD_Phi_ = errorDir.make<TH1I>(
      "errorMissingEtD_Phi", "errorMissingEtD_Phi;Missing E_{T} #phi;Number of Events", 72, -0.5, 71.5);
  TH1I *errorMissingEtE_ =
      errorDir.make<TH1I>("errorMissingEtE", "errorMissingEtE;E_{T};Number of Events", 1024, -0.5, 1023.5);
  TH1I *errorMissingEtE_Of_ = errorDir.make<TH1I>(
      "errorMissingEtE_Of", "errorMissingEtE_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  TH1I *errorMissingEtE_Phi_ = errorDir.make<TH1I>(
      "errorMissingEtE_Phi", "errorMissingEtE_Phi;Missing E_{T} #phi;Number of Events", 72, -0.5, 71.5);
  TH1I *errorMissingHtD_ =
      errorDir.make<TH1I>("errorMissingHtD", "errorMissingHtD;H_{T};Number of Events", 1024, -0.5, 1023.5);
  TH1I *errorMissingHtD_Of_ = errorDir.make<TH1I>(
      "errorMissingHtD_Of", "errorMissingHtD_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  TH1I *errorMissingHtD_Phi_ = errorDir.make<TH1I>(
      "errorMissingHtD_Phi", "errorMissingHtD_Phi;Missing H_{T} #phi;Number of Events", 72, -0.5, 71.5);
  TH1I *errorMissingHtE_ =
      errorDir.make<TH1I>("errorMissingHtE", "errorMissingHtE;H_{T};Number of Events", 1024, -0.5, 1023.5);
  TH1I *errorMissingHtE_Of_ = errorDir.make<TH1I>(
      "errorMissingHtE_Of", "errorMissingHtE_Of;Overflow Bit Status;Number of Events", 2, -0.5, 1.5);
  TH1I *errorMissingHtE_Phi_ = errorDir.make<TH1I>(
      "errorMissingHtE_Phi", "errorMissingHtE_Phi;Missing H_{T} #phi;Number of Events", 72, -0.5, 71.5);

  //Added 19.03.2010 to provide additional information in the case of missingHt failures
  //1. The MHT from both wheel inputs (i.e. the leaf cards)
  //2. The emulator jet et,eta,phi for all jets found in an event
  if (doExtraMissingHTDebug_) {
    if (checkCollections(intMissingHtD, GCT_INT_HTMISS_QUANTA, "Internal Missing Ht Data")) {
      TH1I *errorMissingHtD_HtXPosLeaf1 = errorDir.make<TH1I>(
          "errorMissingHtD_HtXPosLeaf1", "missingHtD;Missing H_{T} X PosLeaf1;Number of Events", 4096, -2048.5, 2047.5);
      TH1I *errorMissingHtD_HtXPosLeaf2 = errorDir.make<TH1I>(
          "errorMissingHtD_HtXPosLeaf2", "missingHtD;Missing H_{T} X PosLeaf2;Number of Events", 4096, -2048.5, 2047.5);
      TH1I *errorMissingHtD_HtXPosLeaf3 = errorDir.make<TH1I>(
          "errorMissingHtD_HtXPosLeaf3", "missingHtD;Missing H_{T} X PosLeaf3;Number of Events", 4096, -2048.5, 2047.5);
      TH1I *errorMissingHtD_HtXNegLeaf1 = errorDir.make<TH1I>(
          "errorMissingHtD_HtXNegLeaf1", "missingHtD;Missing H_{T} X NegLeaf1;Number of Events", 4096, -2048.5, 2047.5);
      TH1I *errorMissingHtD_HtXNegLeaf2 = errorDir.make<TH1I>(
          "errorMissingHtD_HtXNegLeaf2", "missingHtD;Missing H_{T} X NegLeaf2;Number of Events", 4096, -2048.5, 2047.5);
      TH1I *errorMissingHtD_HtXNegLeaf3 = errorDir.make<TH1I>(
          "errorMissingHtD_HtXNegLeaf3", "missingHtD;Missing H_{T} X NegLeaf3;Number of Events", 4096, -2048.5, 2047.5);

      TH1I *errorMissingHtD_HtYPosLeaf1 = errorDir.make<TH1I>(
          "errorMissingHtD_HtYPosLeaf1", "missingHtD;Missing H_{T} Y PosLeaf1;Number of Events", 4096, -2048.5, 2047.5);
      TH1I *errorMissingHtD_HtYPosLeaf2 = errorDir.make<TH1I>(
          "errorMissingHtD_HtYPosLeaf2", "missingHtD;Missing H_{T} Y PosLeaf2;Number of Events", 4096, -2048.5, 2047.5);
      TH1I *errorMissingHtD_HtYPosLeaf3 = errorDir.make<TH1I>(
          "errorMissingHtD_HtYPosLeaf3", "missingHtD;Missing H_{T} Y PosLeaf3;Number of Events", 4096, -2048.5, 2047.5);
      TH1I *errorMissingHtD_HtYNegLeaf1 = errorDir.make<TH1I>(
          "errorMissingHtD_HtYNegLeaf1", "missingHtD;Missing H_{T} Y NegLeaf1;Number of Events", 4096, -2048.5, 2047.5);
      TH1I *errorMissingHtD_HtYNegLeaf2 = errorDir.make<TH1I>(
          "errorMissingHtD_HtYNegLeaf2", "missingHtD;Missing H_{T} Y NegLeaf2;Number of Events", 4096, -2048.5, 2047.5);
      TH1I *errorMissingHtD_HtYNegLeaf3 = errorDir.make<TH1I>(
          "errorMissingHtD_HtYNegLeaf3", "missingHtD;Missing H_{T} Y NegLeaf3;Number of Events", 4096, -2048.5, 2047.5);

      for (unsigned int i = 0; i < intMissingHtD->size(); i++) {
        if (intMissingHtD->at(i).bx() == GCTTrigBx_) {
          if (!intMissingHtD->at(i).overflow()) {
            if (intMissingHtD->at(i).capBlock() == 0x301 && intMissingHtD->at(i).capIndex() == 0 &&
                intMissingHtD->at(i).isThereHtx())
              errorMissingHtD_HtXPosLeaf1->Fill(intMissingHtD->at(i).htx());
            if (intMissingHtD->at(i).capBlock() == 0x301 && intMissingHtD->at(i).capIndex() == 1 &&
                intMissingHtD->at(i).isThereHtx())
              errorMissingHtD_HtXPosLeaf2->Fill(intMissingHtD->at(i).htx());
            if (intMissingHtD->at(i).capBlock() == 0x301 && intMissingHtD->at(i).capIndex() == 2 &&
                intMissingHtD->at(i).isThereHtx())
              errorMissingHtD_HtXPosLeaf3->Fill(intMissingHtD->at(i).htx());
            if (intMissingHtD->at(i).capBlock() == 0x701 && intMissingHtD->at(i).capIndex() == 0 &&
                intMissingHtD->at(i).isThereHtx())
              errorMissingHtD_HtXNegLeaf1->Fill(intMissingHtD->at(i).htx());
            if (intMissingHtD->at(i).capBlock() == 0x701 && intMissingHtD->at(i).capIndex() == 1 &&
                intMissingHtD->at(i).isThereHtx())
              errorMissingHtD_HtXNegLeaf2->Fill(intMissingHtD->at(i).htx());
            if (intMissingHtD->at(i).capBlock() == 0x701 && intMissingHtD->at(i).capIndex() == 2 &&
                intMissingHtD->at(i).isThereHtx())
              errorMissingHtD_HtXNegLeaf3->Fill(intMissingHtD->at(i).htx());

            if (intMissingHtD->at(i).capBlock() == 0x301 && intMissingHtD->at(i).capIndex() == 0 &&
                intMissingHtD->at(i).isThereHty())
              errorMissingHtD_HtYPosLeaf1->Fill(intMissingHtD->at(i).hty());
            if (intMissingHtD->at(i).capBlock() == 0x301 && intMissingHtD->at(i).capIndex() == 1 &&
                intMissingHtD->at(i).isThereHty())
              errorMissingHtD_HtYPosLeaf2->Fill(intMissingHtD->at(i).hty());
            if (intMissingHtD->at(i).capBlock() == 0x301 && intMissingHtD->at(i).capIndex() == 2 &&
                intMissingHtD->at(i).isThereHty())
              errorMissingHtD_HtYPosLeaf3->Fill(intMissingHtD->at(i).hty());
            if (intMissingHtD->at(i).capBlock() == 0x701 && intMissingHtD->at(i).capIndex() == 0 &&
                intMissingHtD->at(i).isThereHty())
              errorMissingHtD_HtYNegLeaf1->Fill(intMissingHtD->at(i).hty());
            if (intMissingHtD->at(i).capBlock() == 0x701 && intMissingHtD->at(i).capIndex() == 1 &&
                intMissingHtD->at(i).isThereHty())
              errorMissingHtD_HtYNegLeaf2->Fill(intMissingHtD->at(i).hty());
            if (intMissingHtD->at(i).capBlock() == 0x701 && intMissingHtD->at(i).capIndex() == 2 &&
                intMissingHtD->at(i).isThereHty())
              errorMissingHtD_HtYNegLeaf3->Fill(intMissingHtD->at(i).hty());
          }
        }
      }
    }
  }

  if (checkCollections(intJetsE, NUM_INT_JETS, "Intermediate Jets Emulator")) {
    TH2I *errorIntJetsE_EtEtaPhi = errorDir.make<TH2I>("errorIntJetsE_EtEtaPhi",
                                                       "errorIntJetsE_EtEtaPhi;#eta (GCT Units);#phi (GCT Units)",
                                                       22,
                                                       -0.5,
                                                       21.5,
                                                       18,
                                                       -0.5,
                                                       17.5);

    for (unsigned int i = 0; i < intJetsE->size(); i++) {
      if (intJetsE->at(i).bx() == EmuTrigBx_) {
        if (!intJetsE->at(i).oflow() && intJetsE->at(i).et())
          errorIntJetsE_EtEtaPhi->Fill(
              intJetsE->at(i).regionId().ieta(), intJetsE->at(i).regionId().iphi(), intJetsE->at(i).et());
      }
    }
  }

  for (unsigned int i = 0; i < caloRegions->size(); i++) {
    if (caloRegions->at(i).bx() == RCTTrigBx_) {
      if (caloRegions->at(i).et() > 0)
        errorRegionEtEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi(), caloRegions->at(i).et());
      if (caloRegions->at(i).overFlow())
        errorRegionOfEtaPhi_->Fill(caloRegions->at(i).gctEta(), caloRegions->at(i).gctPhi());
    }
  }

  //plot the data candidates relating to this mismatch
  for (unsigned int i = 0; i < missingEtD->size(); i++) {
    if (missingEtD->at(i).bx() == GCTTrigBx_) {
      errorMissingEtD_Of_->Fill(missingEtD->at(i).overFlow());
      if (!missingEtD->at(i).overFlow()) {
        errorMissingEtD_->Fill(missingEtD->at(i).et());
        errorMissingEtD_Phi_->Fill(missingEtD->at(i).phi());
      }
    }
  }
  for (unsigned int i = 0; i < missingHtD->size(); i++) {
    if (missingHtD->at(i).bx() == GCTTrigBx_) {
      errorMissingHtD_Of_->Fill(missingHtD->at(i).overFlow());
      if (!missingHtD->at(i).overFlow()) {
        errorMissingHtD_->Fill(missingHtD->at(i).et());
        errorMissingHtD_Phi_->Fill(missingHtD->at(i).phi());
      }
    }
  }
  //and now for the emulator candidates
  for (unsigned int i = 0; i < missingEtE->size(); i++) {
    if (missingEtE->at(i).bx() == EmuTrigBx_) {
      errorMissingEtE_Of_->Fill(missingEtE->at(i).overFlow());
      if (!missingEtE->at(i).overFlow()) {
        errorMissingEtE_->Fill(missingEtE->at(i).et());
        errorMissingEtE_Phi_->Fill(missingEtE->at(i).phi());
      }
    }
  }
  for (unsigned int i = 0; i < missingHtE->size(); i++) {
    if (missingHtE->at(i).bx() == EmuTrigBx_) {
      errorMissingHtE_Of_->Fill(missingHtE->at(i).overFlow());
      if (!missingHtE->at(i)
               .overFlow()) {  //to see what values the emulator outputs despite the o/f bit being set comment out this statement
        errorMissingHtE_->Fill(missingHtE->at(i).et());
        errorMissingHtE_Phi_->Fill(missingHtE->at(i).phi());
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(GctErrorAnalyzer);
