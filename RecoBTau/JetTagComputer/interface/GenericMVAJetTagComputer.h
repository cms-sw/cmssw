#ifndef RecoBTau_GenericMVAJetTagComputer_h
#define RecoBTau_GenericMVAJetTagComputer_h

#include <string>
#include <memory>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"

class GenericMVAJetTagComputer : public JetTagComputer
{
 public:
   GenericMVAJetTagComputer(const edm::ParameterSet & parameters);
   virtual ~GenericMVAJetTagComputer() {}

   virtual void setEventSetup(const edm::EventSetup &es) const;

   virtual float discriminator(const reco::BaseTagInfo &baseTag) const;

 private:
   std::string m_calibrationLabel;
   mutable std::auto_ptr<GenericMVAComputer> m_mvaComputer;

   mutable PhysicsTools::Calibration::MVAComputer::CacheId m_mvaComputerCacheId;
   mutable PhysicsTools::Calibration::MVAComputerContainer::CacheId m_mvaContainerCacheId;
};

#endif
