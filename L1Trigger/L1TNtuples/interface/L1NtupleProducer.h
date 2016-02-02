
//-------------------------------------------------
//
//   \class L1NtupleProducer
/**
 *   Description:  This code is designed for l1 prompt analysis
//                 starting point is a GMTTreeMaker By Ivan Mikulec. 
*/                
//   $Date: 2010/09/15 10:06:12 $
//   $Revision: 1.15 $
//
//   I. Mikulec            HEPHY Vienna
//
//   06/01/2010 - A.C. Le Bihan : 
//   migration to L1Analysis classes...
//
//--------------------------------------------------
#ifndef L1_NTUPLEPRODUCER_H
#define L1_NTUPLEPRODUCER_H

//---------------
// C++ Headers --
//---------------

#include <memory>
#include <string>

//----------------------
// Framework Headers --
//----------------------

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"


#include "L1Trigger/L1TNtuples/interface/L1AnalysisEvent.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisGMT.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisGT.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisGCT.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRCT.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisDTTF.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisCSCTF.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisCaloTP.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisGenerator.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisSimulation.h"

#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h"
#include "CondFormats/DataRecord/interface/L1MuCSCPtLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"

class TFile;
class TTree;


class L1NtupleProducer : public edm::EDAnalyzer {

 
  public:

   // constructor
      explicit L1NtupleProducer(const edm::ParameterSet&);
      virtual ~L1NtupleProducer();

   // fill tree
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(void);
      virtual void endJob();

  private:
      
      void book();
      void initCSCTF(); 
      void analyzeEvent(const edm::Event& e);
      void analyzeGenerator(const edm::Event& e);
      void analyzeSimulation(const edm::Event& e);
      void analyzeGMT(const edm::Event& e);
      void analyzeGT(const edm::Event& e);
      void analyzeGCT(const edm::Event& e);
      void analyzeRCT(const edm::Event& e);
      void analyzeDTTF(const edm::Event& e);
      void analyzeCSCTF(const edm::Event& e, const edm::EventSetup&);
      void analyzeECAL(const edm::Event& e, const edm::EventSetup&);
      void analyzeHCAL(const edm::Event& e, const edm::EventSetup&);

   // Event info
    
      L1Analysis::L1AnalysisEvent* pL1evt; 
      L1Analysis::L1AnalysisEventDataFormat* pL1evt_data; 
   	
      edm::EDGetTokenT<edm::TriggerResults> hltSource_;

   // Generator info  
    
      edm::EDGetTokenT<reco::GenParticleCollection> generatorSource_;
      L1Analysis::L1AnalysisGenerator* pL1generator;
      L1Analysis::L1AnalysisGeneratorDataFormat* pL1generator_data;
      
   // Simulation info
    
      edm::InputTag simulationSource_;
      L1Analysis::L1AnalysisSimulation* pL1simulation;
      L1Analysis::L1AnalysisSimulationDataFormat* pL1simulation_data;
       
   // GMT data
    
      edm::EDGetTokenT<L1MuGMTReadoutCollection> gmtSource_;
      L1Analysis::L1AnalysisGMT* pL1gmt;
      L1Analysis::L1AnalysisGMTDataFormat* pL1gmt_data;
       
   // GT data
       
      edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> gtEvmSource_;
      edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtSource_;
      L1Analysis::L1AnalysisGT* pL1gt;
      L1Analysis::L1AnalysisGTDataFormat* pL1gt_data;
      
   // GCT data
     
      edm::EDGetTokenT<L1GctJetCandCollection> gctCenJetsSource_ ;
      edm::EDGetTokenT<L1GctJetCandCollection> gctForJetsSource_ ;
      edm::EDGetTokenT<L1GctJetCandCollection> gctTauJetsSource_ ;
      edm::EDGetTokenT<L1GctJetCandCollection> gctIsoTauJetsSource_ ;
      edm::EDGetTokenT<L1GctEtMissCollection> gctETMSource_;
      edm::EDGetTokenT<L1GctEtTotalCollection> gctETTSource_;
      edm::EDGetTokenT<L1GctEtHadCollection> gctHTTSource_;
      edm::EDGetTokenT<L1GctHtMissCollection> gctHTMSource_;
      edm::EDGetTokenT<L1GctHFRingEtSumsCollection> gctHFSumsSource_;
      edm::EDGetTokenT<L1GctHFBitCountsCollection> gctHFBitsSource_;
      edm::EDGetTokenT<L1GctEmCandCollection> gctIsoEmSource_ ;
      edm::EDGetTokenT<L1GctEmCandCollection> gctNonIsoEmSource_ ;   
      L1Analysis::L1AnalysisGCT* pL1gct;
      L1Analysis::L1AnalysisGCTDataFormat* pL1gct_data;
       
   // RCT data
        
      edm::EDGetTokenT<L1CaloRegionCollection> rctRgnSource_; 
      edm::EDGetTokenT<L1CaloEmCollection> rctEmSource_; 
      L1Analysis::L1AnalysisRCT* pL1rct;
      L1Analysis::L1AnalysisRCTDataFormat* pL1rct_data;
       
   // DTTF data
      
      edm::EDGetTokenT<L1MuDTChambPhContainer> dttfPhSource_; 
      edm::EDGetTokenT<L1MuDTChambThContainer> dttfThSource_; 
      edm::EDGetTokenT<L1MuDTTrackContainer> dttfTrkSource_; 
      L1Analysis::L1AnalysisDTTF* pL1dttf;
      L1Analysis::L1AnalysisDTTFDataFormat* pL1dttf_data;
  
   // CSCTF data
      
      edm::EDGetTokenT<L1CSCTrackCollection> csctfTrkSource_; 
      edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> csctfLCTSource_; 
      edm::EDGetTokenT<L1CSCStatusDigiCollection> csctfStatusSource_; 
      edm::EDGetTokenT<CSCTriggerContainer<csctf::TrackStub> > csctfDTStubsSource_; 
      L1Analysis::L1AnalysisCSCTF* pL1csctf;
      L1Analysis::L1AnalysisCSCTFDataFormat* pL1csctf_data;
      CSCSectorReceiverLUT* srLUTs_[5][2];
      CSCTFPtLUT* csctfPtLUTs_;
      bool initCSCTFPtLutsPSet;
      edm::ParameterSet csctfPtLutsPSet;     
      const L1MuTriggerScales  *ts;
      const L1MuTriggerPtScale *tpts;
      unsigned long long m_scalesCacheID ;
      unsigned long long m_ptScaleCacheID ;
      unsigned long long m_csctfptlutCacheID ;

      // Calo TP data
      edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalSource_;
      edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalSource_;
      L1Analysis::L1AnalysisCaloTP* pL1calotp;
      L1Analysis::L1AnalysisCaloTPDataFormat* pL1calotp_data;
      unsigned long long ecalScaleCacheID_;
      unsigned long long hcalScaleCacheID_;

   //   
      edm::Service<TFileService> tfs_;
      TTree* tree_;
    
      bool physVal_;
      bool verbose_;
      
      unsigned int maxGEN_;
      unsigned int maxGT_;
      unsigned int maxRCTREG_;
      unsigned int maxDTPH_;
      unsigned int maxDTTH_;
      unsigned int maxDTTR_;
      unsigned int maxRPC_;
      unsigned int maxDTBX_;
      unsigned int maxCSC_;
      unsigned int maxGMT_;
      unsigned int maxCSCTFTR_;
      unsigned int maxCSCTFLCTSTR_;
      unsigned int maxCSCTFLCTS_;
      unsigned int maxCSCTFSPS_;
};


#endif
