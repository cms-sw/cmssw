#ifndef HCALTBTRIGGERFILTER_H
#define HCALTBTRIGGERFILTER_H 1

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

/** \class HcalTBTriggerFilter
    
   $Date: 2006/08/17 17:28:23 $
   $Revision: 1.5 $
   \author J. Mans - Minnesota
*/
class HcalTBTriggerFilter : public edm::EDFilter {
public:
  HcalTBTriggerFilter(const edm::ParameterSet& ps);
  virtual ~HcalTBTriggerFilter() {}
  virtual bool filter(edm::Event& e, edm::EventSetup const& c);
private:
  bool allowPedestal_;
  bool allowPedestalInSpill_;
  bool allowPedestalOutSpill_;
  bool allowLaser_;
  bool allowLED_;
  bool allowBeam_;
  edm::InputTag label_;
};

#endif
