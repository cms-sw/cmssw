#ifndef HLTGetDigi_h
#define HLTGetDigi_h

/** \class HLTGetDigi
 *
 *  
 *  This class is an EDAnalyzer implementing a "get data into RAM"
 *  functionality for DIGIs, to simulate online FF running/timimg.
 *
 *
 *  \author various
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/DTDigi/interface/DTDigi.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

//
// class declaration
//

class HLTGetDigi : public edm::EDAnalyzer {

 public:
  explicit HLTGetDigi(const edm::ParameterSet&);
  ~HLTGetDigi();
  void analyze(const edm::Event&, const edm::EventSetup&);
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
 private:
  edm::InputTag EBdigiCollection_;
  edm::EDGetTokenT<EBDigiCollection> EBdigiToken_;
  edm::InputTag EEdigiCollection_;
  edm::EDGetTokenT<EEDigiCollection> EEdigiToken_;
  edm::InputTag ESdigiCollection_;
  edm::EDGetTokenT<ESDigiCollection> ESdigiToken_;
  edm::InputTag HBHEdigiCollection_;
  edm::EDGetTokenT<HBHEDigiCollection> HBHEdigiToken_;
  edm::InputTag HOdigiCollection_;
  edm::EDGetTokenT<HODigiCollection> HOdigiToken_;
  edm::InputTag HFdigiCollection_;
  edm::EDGetTokenT<HFDigiCollection> HFdigiToken_;
  edm::InputTag PXLdigiCollection_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi> > PXLdigiToken_;
  edm::InputTag SSTdigiCollection_;
  edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > SSTdigiToken_;
  edm::InputTag CSCStripdigiCollection_;
  edm::EDGetTokenT<CSCStripDigiCollection> CSCStripdigiToken_;
  edm::InputTag CSCWiredigiCollection_;
  edm::EDGetTokenT<CSCWireDigiCollection> CSCWiredigiToken_;
  edm::InputTag DTdigiCollection_;
  edm::EDGetTokenT<DTDigiCollection> DTdigiToken_;
  edm::InputTag RPCdigiCollection_;
  edm::EDGetTokenT<RPCDigiCollection> RPCdigiToken_;
  edm::InputTag GctCaloEmLabel_;
  edm::EDGetTokenT<L1CaloEmCollection> GctCaloEmToken_;
  edm::InputTag GctCaloRegionLabel_;
  edm::EDGetTokenT<L1CaloRegionCollection> GctCaloRegionToken_;
  edm::InputTag GctIsoEmLabel_;
  edm::EDGetTokenT<L1GctEmCandCollection> GctIsoEmToken_;
  edm::InputTag GctNonIsoEmLabel_;
  edm::EDGetTokenT<L1GctEmCandCollection> GctNonIsoEmToken_;
  edm::InputTag GctCenJetLabel_;
  edm::EDGetTokenT<L1GctJetCandCollection> GctCenJetToken_;
  edm::InputTag GctForJetLabel_;
  edm::EDGetTokenT<L1GctJetCandCollection> GctForJetToken_;
  edm::InputTag GctTauJetLabel_;
  edm::EDGetTokenT<L1GctJetCandCollection> GctTauJetToken_;
  edm::InputTag GctJetCountsLabel_;
  edm::EDGetTokenT<L1GctJetCounts> GctJetCountsToken_;
  edm::InputTag GctEtHadLabel_;
  edm::EDGetTokenT<L1GctEtHad> GctEtHadToken_;
  edm::InputTag GctEtMissLabel_;
  edm::EDGetTokenT<L1GctEtMiss> GctEtMissToken_;
  edm::InputTag GctEtTotalLabel_;
  edm::EDGetTokenT<L1GctEtTotal> GctEtTotalToken_;

  edm::InputTag GtEvmRRLabel_;
  edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> GtEvmRRToken_;
  edm::InputTag GtObjectMapLabel_;
  edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord> GtObjectMapToken_;
  edm::InputTag GtRRLabel_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> GtRRToken_;

  edm::InputTag GmtCandsLabel_;
  edm::EDGetTokenT<std::vector<L1MuGMTCand> > GmtCandsToken_;
  edm::InputTag GmtReadoutCollection_;
  edm::EDGetTokenT<L1MuGMTReadoutCollection> GmtReadoutToken_;
   
  bool getEcalDigis_ ; 
  bool getEcalESDigis_ ; 
  bool getHcalDigis_ ; 
  bool getPixelDigis_ ; 
  bool getStripDigis_ ; 
  bool getCSCDigis_ ; 
  bool getDTDigis_ ; 
  bool getRPCDigis_ ; 
  bool getL1Calo_ ; 
  bool getGctEmDigis_ ; 
  bool getGctJetDigis_ ; 
  bool getGctJetCounts_ ; 
  bool getGctEtDigis_ ; 
  bool getGtEvmRR_ ; 
  bool getGtObjectMap_ ; 
  bool getGtRR_ ; 
  bool getGmtCands_ ; 
  bool getGmtRC_ ; 

};

#endif //HLTGetDigi_h
