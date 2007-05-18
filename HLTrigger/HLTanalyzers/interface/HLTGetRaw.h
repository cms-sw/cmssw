#ifndef HLTGetRaw_h
#define HLTGetRaw_h

/** \class HLTGetRaw
 *
 *  
 *  This class is an EDAnalyzer implementing a "get data into RAM"
 *  functionality for RAW, to simulate online FF running/timimg.
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

class HLTGetRaw : public edm::EDAnalyzer {

 public:
  explicit HLTGetRaw(const edm::ParameterSet&);
  ~HLTGetRaw();
  void analyze(const edm::Event&, const edm::EventSetup&);
  
 private:
  edm::InputTag RawDataCollection_;
};

#endif //HLTGetRaw_h
