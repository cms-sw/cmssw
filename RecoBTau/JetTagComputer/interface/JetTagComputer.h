#ifndef RecoBTau_JetTagComputer_h
#define RecoBTau_JetTagComputer_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"

class JetTagComputer {
public:
  // default constructor
  JetTagComputer(void) { }

  // explicit constructor accepting a ParameterSet for configuration
  explicit JetTagComputer(const edm::ParameterSet & configuration) { }
  
  virtual float discriminator(const reco::BaseTagInfo &) const = 0;
  virtual void  setEventSetup(const edm::EventSetup &) const { }

  float operator()(const reco::BaseTagInfo & info) const {
    return discriminator(info);
  }
  
};

#endif // RecoBTau_JetTagComputer_h
