#ifndef HLTGetData_h
#define HLTGetData_h

/** \class HLTGetData
 *
 *  
 *  This class is an EDAnalyzer implementing a "get data into RAM"
 *  functionality, to simulate online FF running/timimg.
 *
 *  $Date: 2007/04/11 17:51:05 $
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

class HLTGetData : public edm::EDAnalyzer {

 public:
  explicit HLTGetData(const edm::ParameterSet&);
  ~HLTGetData();
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

#endif //HLTGetData_h
