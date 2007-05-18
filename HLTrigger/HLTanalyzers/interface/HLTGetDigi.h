#ifndef HLTGetDigi_h
#define HLTGetDigi_h

/** \class HLTGetDigi
 *
 *  
 *  This class is an EDAnalyzer implementing a "get data into RAM"
 *  functionality for DIGIs, to simulate online FF running/timimg.
 *
 *  $Date: 2007/04/20 06:58:26 $
 *  $Revision: 1.1 $
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
    
  bool getEcalDigis_ ; 
  bool getEcalESDigis_ ; 
  bool getHcalDigis_ ; 
  bool getPixelDigis_ ; 
  bool getStripDigis_ ; 
  bool getCSCDigis_ ; 
  bool getDTDigis_ ; 
  bool getRPCDigis_ ; 

};

#endif //HLTGetDigi_h
