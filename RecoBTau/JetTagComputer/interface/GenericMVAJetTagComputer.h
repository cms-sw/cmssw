#ifndef RecoBTau_GenericMVAJetTagComputer_h
#define RecoBTau_GenericMVAJetTagComputer_h

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
//#include "PhysicsTools/MVAComputer/interface/MVAComputer.icc"

class  GenericMVAJetTagComputer : public JetTagComputer
{
 public:
 // GenericMVAJetTagComputer(const Calibration::DiscriminatorComputer *calib ) {}
  GenericMVAJetTagComputer(const edm::ParameterSet  & parameters) {}

  virtual void setEventSetup(const edm::EventSetup &es) 
    {
    //Check cacheId of the ES stuff or if m_mvaComputer is null
    //if needed create a new m_mvaComputer with update calib
    //
    
    //    m_eventSetup=&es;
    }
 
  virtual float discriminator(const reco::BaseTagInfo &) const 
    {
      //adapt  tagging variable (if we decide to add such a virtual method in the base
      // class BaseTagInfo) to DiscriminatorComputer generic interface 
  //     PhysicsTools::Variable::Value * values;

//      return m_mvaComputer->eval(values,values);
    }
 private:
   PhysicsTools::MVAComputer * m_mvaComputer;
  // edm::EventSetup * m_eventSetup ;
};
#endif
