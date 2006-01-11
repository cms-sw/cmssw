#ifndef HCALTBTRIGGERFILTER_H
#define HCALTBTRIGGERFILTER_H 1

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

/** \class HcalTBTriggerFilter
    
   $Date: 2005/12/21 17:17:03 $
   $Revision: 1.1 $
   \author J. Mans - Minnesota
*/
class HcalTBTriggerFilter : public edm::EDFilter {
public:
  HcalTBTriggerFilter(const edm::ParameterSet& ps);
  virtual ~HcalTBTriggerFilter() {}
  virtual bool filter(edm::Event const& e, edm::EventSetup const& c);
private:
  bool allowPedestal_;
  bool allowPedestalInSpill_;
  bool allowPedestalOutSpill_;
  bool allowLaser_;
  bool allowLED_;
  bool allowBeam_;
};

#endif
