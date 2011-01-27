#ifndef HLTGetDigi_h
#define HLTGetDigi_h

/** \class HLTGetDigi
 *
 *  
 *  This class is an EDAnalyzer implementing a "get data into RAM"
 *  functionality for DIGIs, to simulate online FF running/timimg.
 *
 *  $Date: 2007/11/05 17:01:31 $
 *  $Revision: 1.4 $
 *
 *  \author various
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class HLTGetDigi : public edm::EDAnalyzer {

 public:
  explicit HLTGetDigi(const edm::ParameterSet&);
  ~HLTGetDigi();
  void analyze(const edm::Event&, const edm::EventSetup&);
  
 private:
  edm::InputTag EBdigiCollection_;
  edm::InputTag EEdigiCollection_;
  edm::InputTag ESdigiCollection_;      
  edm::InputTag HBHEdigiCollection_;
  edm::InputTag HOdigiCollection_;
  edm::InputTag HFdigiCollection_;      
  edm::InputTag PXLdigiCollection_;      
  edm::InputTag SSTdigiCollection_;      
  edm::InputTag CSCStripdigiCollection_;      
  edm::InputTag CSCWiredigiCollection_;      
  edm::InputTag DTdigiCollection_;      
  edm::InputTag RPCdigiCollection_;
  edm::InputTag GctCaloEmLabel_ ;
  edm::InputTag GctCaloRegionLabel_ ;
  edm::InputTag GctIsoEmLabel_ ;
  edm::InputTag GctNonIsoEmLabel_ ;
  edm::InputTag GctCenJetLabel_ ; 
  edm::InputTag GctForJetLabel_ ; 
  edm::InputTag GctTauJetLabel_ ; 
  edm::InputTag GctJetCountsLabel_ ; 
  edm::InputTag GctEtHadLabel_ ; 
  edm::InputTag GctEtMissLabel_ ; 
  edm::InputTag GctEtTotalLabel_ ; 

  edm::InputTag GtEvmRRLabel_ ; 
  edm::InputTag GtObjectMapLabel_ ; 
  edm::InputTag GtRRLabel_ ; 

  edm::InputTag GmtCandsLabel_ ; 
  edm::InputTag GmtReadoutCollection_ ; 
    
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
