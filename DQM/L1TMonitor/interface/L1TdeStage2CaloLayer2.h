#ifndef DQM_L1TMonitor_L1TdeStage2CaloLayer2
#define DQM_L1TMonitor_L1TdeStage2CaloLayer2

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

class L1TdeStage2CaloLayer2 : public DQMEDAnalyzer {

 public:
  L1TdeStage2CaloLayer2 (const edm::ParameterSet & ps);
  virtual ~L1TdeStage2CaloLayer2();

 protected:
  virtual void dqmBeginRun (const edm::Run&, const edm::EventSetup &) override;
  virtual void beginLuminosityBlock (const edm::LuminosityBlock&,
				     const edm::EventSetup&);
  virtual void endLuminosityBlock (const edm::LuminosityBlock&,
				   const edm::EventSetup&);
  virtual void bookHistograms (DQMStore::IBooker&,
			       const edm::Run&,
			       const edm::EventSetup&) override;
  virtual void analyze (const edm::Event&, const edm::EventSetup&) override;

 private:

  bool compareJets(const edm::Handle<l1t::JetBxCollection> & dataCol,
                   const edm::Handle<l1t::JetBxCollection> & emulCol,
                   TH1F * & summaryHist,
		   TH1F * & objSummaryHist,
		   TH1F * & objEtHist,
		   TH1F * & objEtaHist,
		   TH1F * & objPhiHist);
  bool compareEGs(const edm::Handle<l1t::EGammaBxCollection> & dataCol,
                  const edm::Handle<l1t::EGammaBxCollection> & emulCol,
                  TH1F * & summaryHist,
		  TH1F * & objSummaryHist,
		  TH1F * & objEtHist,
		  TH1F * & objEtaHist,
		  TH1F * & objPhiHist,
		  TH1F * & isoObjEtHist,
		  TH1F * & isoObjEtaHist,
		  TH1F * & isoObjPhiHist);
  bool compareTaus(const edm::Handle<l1t::TauBxCollection> & dataCol,
                   const edm::Handle<l1t::TauBxCollection> & emulCol,
		   TH1F * & summaryHist,
		   TH1F * & objSummaryHist,
		   TH1F * & objEtHist,
		   TH1F * & objEtaHist,
		   TH1F * & objPhiHist,
		   TH1F * & isoObjEtHist,
		   TH1F * & isoObjEtaHist,
		   TH1F * & isoObjPhiHist);
  bool compareSums(const edm::Handle<l1t::EtSumBxCollection> & dataCol,
                   const edm::Handle<l1t::EtSumBxCollection> & emulCol,
                   TH1F * & hist,
		   TH1F * & objSummaryHist);

  // Holds the name of directory in DQM where module hostograms will be shown.
  // Value is taken from python configuration file (passed in class constructor)
  std::string monitorDir;

  enum summaryVars {
    EVENTGOOD = 1,    // number of good events (100% agreement)
    JETGOOD_S,        // number of jets in agreement (energy and pos)
    EGGOOD_S,         // number of e/g in agremeent (energy and pos)
    TAUGOOD_S,        // number of taus in agremenet (energy and pos)
    SUMGOOD_S         // number of good sums across all events
  };

  enum jetVars {
    // NJETS = 1,
    JETGOOD = 1,
    JETPOSOFF,
    JETETOFF
  };

  enum egVars {
    // NEGS = 1,
    EGGOOD = 1,
    EGPOSOFF,
    EGETOFF,
    ISOEGGOOD,
    ISOEGPOSOFF,
    ISOEGETOFF
  };

  enum tauVars {
    // NTAUS = 1,
    TAUGOOD = 1,
    TAUPOSOFF,
    TAUETOFF,
    ISOTAUGOOD,
    ISOTAUPOSOFF,
    ISOTAUETOFF
  };

  enum sumVars {
    SUMGOOD = 1,
    ETTSUMGOOD,
    HTTSUMGOOD,
    METSUMGOOD,
    MHTSUMGOOD,
    MBHFSUMGOOD,
    TOWCOUNTGOOD
  };

  /*
  enum jetProperties {
    JET};
  enum egProperties {};
  enum isoegProperties {};
  enum tauProperties {};
  enum isotauProperties {};
  */

  // collections to hold entities reconstructed from data and emulation
  edm::EDGetTokenT<l1t::JetBxCollection> calol2JetCollectionData;
  edm::EDGetTokenT<l1t::JetBxCollection> calol2JetCollectionEmul;
  edm::EDGetTokenT<l1t::EGammaBxCollection> calol2EGammaCollectionData;
  edm::EDGetTokenT<l1t::EGammaBxCollection> calol2EGammaCollectionEmul;
  edm::EDGetTokenT<l1t::TauBxCollection> calol2TauCollectionData;
  edm::EDGetTokenT<l1t::TauBxCollection> calol2TauCollectionEmul;
  edm::EDGetTokenT<l1t::EtSumBxCollection> calol2EtSumCollectionData;
  edm::EDGetTokenT<l1t::EtSumBxCollection> calol2EtSumCollectionEmul;

  // objects to represent individual plots shown in DQM

  MonitorElement * agreementSummary;
  MonitorElement * jetSummary;
  MonitorElement * tauSummary;
  MonitorElement * egSummary;
  MonitorElement * sumSummary;
  MonitorElement * mpSummary;

  // add histograms to store the properties of mismatched objects
  MonitorElement * jetEt;
  MonitorElement * jetEta;
  MonitorElement * jetPhi;

  MonitorElement * egEt;
  MonitorElement * egEta;
  MonitorElement * egPhi;
  MonitorElement * isoEgEt;
  MonitorElement * isoEgEta;
  MonitorElement * isoEgPhi;

  MonitorElement * tauEt;
  MonitorElement * tauEta;
  MonitorElement * tauPhi;
  MonitorElement * isoTauEt;
  MonitorElement * isoTauEta;
  MonitorElement * isoTauPhi;

  // histograms for mismatched sums

  bool verbose;
  bool notProcessed = true;

  int totalEvents = 0;
  int goodEvents = 0;

  int totalJets = 0;
  int goodJets = 0;
  int posOffJets = 0;
  int etOffJets = 0;

  int totalEGs = 0;
  int goodEGs = 0;
  int posOffEGs = 0;
  int etOffEGs = 0;
  int totalIsoEGs = 0;
  int goodIsoEGs = 0;
  int posOffIsoEGs = 0;
  int etOffIsoEGs = 0;

  int totalTaus = 0;
  int goodTaus = 0;
  int posOffTaus = 0;
  int etOffTaus = 0;
  int totalIsoTaus = 0;
  int goodIsoTaus = 0;
  int posOffIsoTaus = 0;
  int etOffIsoTaus = 0;

  int totalSums = 0;
  int goodSums = 0;

  int totalETTSums = 0;
  int goodETTSums = 0;
  int totalHTTSums = 0;
  int goodHTTSums = 0;

  int totalMETSums = 0;
  int goodMETSums = 0;
  int totalMHTSums = 0;
  int goodMHTSums = 0;

  int totalMBHFSums = 0;
  int goodMBHFSums = 0;
  int totalTowCountSums = 0;
  int goodTowCountSums = 0;

  const unsigned int currBx = 0;
};

#endif
