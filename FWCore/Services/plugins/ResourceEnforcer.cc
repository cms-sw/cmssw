// -*- C++ -*-
//
// Package:     Services
// Class  :     ResourceEnforcer
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Sun May  6 12:16:49 CDT 2012
// $Id: ResourceEnforcer.cc,v 1.1 2012/05/06 19:14:27 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Services/src/ProcInfoFetcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/CPUTimer.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"


namespace edm {
  class Event;
  class EventSetup;
  
  namespace service {
    class ResourceEnforcer {
    public:
      ResourceEnforcer(edm::ParameterSet const& iConfig, ActivityRegistry& iAR);
      
      void check();
      
      void postEventProcessing(Event const& e, EventSetup const&);
      
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:
      ProcInfoFetcher m_fetcher;
      CPUTimer m_timer;
      
      double m_maxVSize;
      double m_maxRSS;
      double m_maxTime;
    };
  }
}

using namespace edm::service;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ResourceEnforcer::ResourceEnforcer( edm::ParameterSet const& iConfig,ActivityRegistry& iReg):
m_maxVSize(iConfig.getUntrackedParameter<double>("maxVSize",0)*1000.), //convert to MB
m_maxRSS(iConfig.getUntrackedParameter<double>("maxRSS",0)*1000.),
m_maxTime(iConfig.getUntrackedParameter<double>("maxTime",0)*60.*60.) //convert from hours to seconds

{
  iReg.watchPostProcessEvent(this, &ResourceEnforcer::postEventProcessing);
  m_timer.start();

}

//
// member functions
//

//
// const member functions
//

void 
ResourceEnforcer::postEventProcessing(Event const& e, EventSetup const&) {
  this->check();
}


void
ResourceEnforcer::check()
{
  ProcInfo pi =m_fetcher.fetch();
  
  if(0!=m_maxVSize && m_maxVSize< pi.vsize) {
    throw edm::Exception(errors::ExceededResourceVSize)<<"Exceeded maximum allowed VSize of "<<m_maxVSize/1000.<<" GB (VSize is "<<pi.vsize/1000.<<")";
  }

  if(0!=m_maxRSS && m_maxRSS< pi.rss) {
    throw edm::Exception(errors::ExceededResourceRSS)<<"Exceeded maximum allowed RSS of "<<m_maxRSS/1000.<<" GB (VSize is "<<pi.rss/1000.<<")";
  }

  if(0!=m_maxTime && m_maxTime < m_timer.realTime()) {
    throw edm::Exception(errors::ExceededResourceTime)<<"Exceeded maximum allowed time of "<<m_maxTime/60./60.<<" hours";
  }
}


//
// static member functions
//

void ResourceEnforcer::fillDescriptions(ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;
  desc.addUntracked<double>("maxVSize", 0.)->setComment("Maximum allowed VSize for the job in GB. Ignored if set to 0.");
  desc.addUntracked<double>("maxRSS", 0.)->setComment("Maximum allowd RSS for the job in GB. Ignored if set to 0.");
  desc.addUntracked<double>("maxTime", 0.)->setComment("Maximum allowd wallclock time for the job in hours. Ignored if set to 0.");
  descriptions.add("ResourceEnforcer", desc);
}

DEFINE_FWK_SERVICE(ResourceEnforcer);
