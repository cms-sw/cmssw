#ifndef HLTFilter_h
#define HLTFilter_h

/** \class HLTFilter
 *
 *  
 *  This class derives from EDFilter and adds a few HLT specific
 *  items. Any and all HLT filters must derive from the HLTFilter
 *  class!
 *
 *  $Date: 2006/08/14 14:52:51 $
 *  $Revision: 1.5 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class HLTFilter : public edm::EDFilter {

 public:
  HLTFilter() : EDFilter() {}
  virtual ~HLTFilter();
  virtual bool filter(edm::Event&, const edm::EventSetup&)=0;

#include "HLTrigger/HLTcore/interface/HLTadd.h"

};

#endif //HLTFilter_h
