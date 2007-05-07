#ifndef RecoBTau_JetTagComputer_h
#define RecoBTau_JetTagComputer_h
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
class  JetTagComputer
{
 public:
  virtual float discriminator(const reco::BaseTagInfo &) const = 0;
  virtual void setEventSetup(const edm::EventSetup &) {}
};
#endif
