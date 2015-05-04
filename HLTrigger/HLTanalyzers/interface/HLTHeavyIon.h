#ifndef HLTHEV_H
#define HLTHEV_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"

//ccla
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Common/interface/Provenance.h"

/* #include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h" */
/* #include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h" */
/* #include "CondFormats/L1TObjects/interface/L1CaloEtScale.h" */
/* #include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h" */
/* #include "CondFormats/L1TObjects/interface/L1RCTParameters.h" */
/* #include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h" */
/* #include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h" */
/* #include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"  */

#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

// #include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
//#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

typedef std::vector<std::string> MyStrings;

/** \class HLTHeavyIon
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */
class HLTHeavyIon {
public:
  HLTHeavyIon(edm::ConsumesCollector && iC); 

  void setup(const edm::ParameterSet& pSet, TTree* tree);
  void beginRun(const edm::Run& , const edm::EventSetup& );
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  /** Analyze the Data */
  void analyze(const edm::Handle<edm::TriggerResults>                 & hltresults,
	       const edm::Handle<reco::Centrality> & centrality,
               const edm::Handle<reco::EvtPlaneCollection> & evtPlanes,
	       const edm::Handle<edm::GenHIEvent> & hiMC,
	       edm::EventSetup const& eventSetup,
	       edm::Event const& iEvent,
	       TTree* tree);

private:

  // Tree variables

  float *hiEvtPlane;
  int nEvtPlanes;
  int HltEvtCnt;
  int hiBin;
  int hiNpix, hiNpixelTracks, hiNtracks, hiNtracksPtCut, hiNtracksEtaCut, hiNtracksEtaPtCut;
  float hiHF, hiHFplus, hiHFminus, hiHFhit, hiHFhitPlus, hiHFhitMinus, hiEB, hiET, hiEE, hiEEplus, hiEEminus, hiZDC, hiZDCplus, hiZDCminus;
  
  float fNpart;
  float fNcoll;
  float fNhard;
  float fPhi0;
  float fb;

  int fNcharged;
  int fNchargedMR;
  float fMeanPt;
  float fMeanPtMR;
  float fEtMR;
  int fNchargedPtCut;
  int fNchargedPtCutMR;


  TString * algoBitToName;
  TString * techBitToName;


  HLTConfigProvider hltConfig_; 
  std::string processName_;

  bool _OR_BXes;
  int UnpackBxInEvent; // save number of BXs unpacked in event

  edm::InputTag   centralityBin_Label;
  edm::EDGetTokenT<int> centralityBin_Token;

  // input variables
  bool _Debug;
  bool _Monte;
};

#endif
