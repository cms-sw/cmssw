#ifndef IsolationAlgos_CITKIsolationConeDefinitionBase_H
#define IsolationAlgos_CITKIsolationConeDefinitionBase_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <unordered_map>

namespace citk {
  class IsolationConeDefinitionBase {
  public:
  IsolationConeDefinitionBase(const edm::ParameterSet& c) :
    _coneSize2(std::pow(c.getParameter<double>("coneSize"),2.0)),
    _name(c.getParameter<std::string>("isolationAlgo")) {
    }
    IsolationConeDefinitionBase(const IsolationConeDefinitionBase&) = delete;
    IsolationConeDefinitionBase& operator=(const IsolationConeDefinitionBase&) =delete;
    


    virtual void getEventSetupInfo(const edm::EventSetup&) {}
    virtual void getEventInfo(const edm::Event&) {}
    virtual void setConsumes(edm::ConsumesCollector) = 0;

    virtual bool isInIsolationCone(const reco::CandidatePtr& physob,
				   const reco::CandidatePtr& other) const = 0;

    const std::string& name() const { return _name; }

    const std::string& additionalCode() const { return _additionalCode; }

    //! Destructor
    virtual ~IsolationConeDefinitionBase(){};

  protected:
    const float _coneSize2;
    std::string _additionalCode;
    
  private:    
    const std::string _name;
  };
}// ns citk

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< citk::IsolationConeDefinitionBase* (const edm::ParameterSet&) > CITKIsolationConeDefinitionFactory;

#endif
