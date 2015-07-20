#ifndef __CommonTools_CandAlgos_ModifyObjectValueBase_h__
#define __CommonTools_CandAlgos_ModifyObjectValueBase_h__

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <string>

class ModifyObjectValueBase {
 public:
 ModifyObjectValueBase(const edm::ParameterSet& conf) : 
  name_( conf.getParameter<std::string>("modifierName") ) {}

  virtual ~ModifyObjectValueBase() {}

  virtual void setEvent(const edm::Event&) {}
  virtual void setEventContent(const edm::EventSetup&) {}
  virtual void setConsumes(edm::ConsumesCollector&) {}
  
  virtual void modifyObject(pat::Electron&) const { 
    throw cms::Exception("InvalidConfiguration") 
      << name_ << " is not configured to handle electrons!"; 
  }
  virtual void modifyObject(pat::Photon&)   const { 
    throw cms::Exception("InvalidConfiguration") 
      << name_ << " is not configured to handle photons!"; 
  }
  virtual void modifyObject(pat::Muon&)     const { 
    throw cms::Exception("InvalidConfiguration") 
      << name_ << " is not configured to handle muons!"; 
  }
  virtual void modifyObject(pat::Tau&)      const { 
    throw cms::Exception("InvalidConfiguration") 
      << name_ << " is not configured to handle taus!"; 
  }
  virtual void modifyObject(pat::Jet&)      const { 
    throw cms::Exception("InvalidConfiguration") 
      << name_ << " is not configured to handle jets!"; 
  }

  const std::string& name() const { return name_; }

 private:
  const std::string name_;
};

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< ModifyObjectValueBase* (const edm::ParameterSet&) > ModifyObjectValueFactory;
#endif

#endif
