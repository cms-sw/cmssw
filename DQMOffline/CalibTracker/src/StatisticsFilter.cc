// -*- C++ -*-
//
// Package:    StatisticsFilter
// Class:      StatisticsFilter
// 
/**\class StatisticsFilter StatisticsFilter.cc MyFilter/StatisticsFilter/src/StatisticsFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Gordon Kaussen,40 1-A15,+41227671647,
//         Created:  Mon Nov 15 10:48:54 CET 2010
// $Id: StatisticsFilter.cc,v 1.1 2010/11/15 17:16:31 kaussen Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declaration
//

class StatisticsFilter : public edm::EDFilter {
   public:
      explicit StatisticsFilter(const edm::ParameterSet&);
      ~StatisticsFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  DQMStore* dqmStore_;

  std::string filename, dirpath;
  int TotNumberOfEvents;
  int MinNumberOfEvents;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
StatisticsFilter::StatisticsFilter(const edm::ParameterSet& iConfig) : filename(iConfig.getUntrackedParameter<std::string>("rootFilename","")),
								       dirpath(iConfig.getUntrackedParameter<std::string>("histoDirPath","")),
								       MinNumberOfEvents(iConfig.getUntrackedParameter<int>("minNumberOfEvents"))
{
   //now do what ever initialization is needed

  dqmStore_ = edm::Service<DQMStore>().operator->();
  dqmStore_->open(filename.c_str(), false);
}


StatisticsFilter::~StatisticsFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
StatisticsFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  TotNumberOfEvents = 0;

  std::vector<MonitorElement*> MEs = dqmStore_->getAllContents(dirpath);

  std::vector<MonitorElement*>::const_iterator iter=MEs.begin();
  std::vector<MonitorElement*>::const_iterator iterEnd=MEs.end();

  for (; iter!=iterEnd;++iter)
    {
      std::string me_name = (*iter)->getName();

      if ( strstr(me_name.c_str(),"TotalNumberOfCluster__T")!=NULL && strstr(me_name.c_str(),"Profile")==NULL )
	{
	  TotNumberOfEvents = ((TH1F*)(*iter)->getTH1F())->GetEntries();

	  break;
	}
    }

  if ( TotNumberOfEvents<MinNumberOfEvents )
    {
      edm::LogInfo("StatisticsFilter") << "Only " << TotNumberOfEvents << " events in the run. Run will not be analyzed!";

      return false;
    }

  return true;
}

// ------------ method called once each job just before starting event loop  ------------
void 
StatisticsFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
StatisticsFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(StatisticsFilter);
