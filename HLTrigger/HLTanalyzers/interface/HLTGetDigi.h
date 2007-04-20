#ifndef HLTGetDigi_h
#define HLTGetDigi_h

/** \class HLTGetDigi
 *
 *  
 *  This class is an EDAnalyzer implementing a "get data into RAM"
 *  functionality for DIGIs, to simulate online FF running/timimg.
 *
 *  $Date: 2007/04/12 09:57:12 $
 *  $Revision: 1.2 $
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

};

#endif //HLTGetDigi_h
