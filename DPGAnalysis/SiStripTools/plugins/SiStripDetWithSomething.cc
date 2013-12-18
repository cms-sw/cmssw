// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      SiStripDetWithSomething
//
/**\class SiStripDetWithSomething SiStripDetWithSomething.cc DPGAnalysis/SiStripTools/plugins/SiStripDetWithSomething.cc

 Description: template EDFilter to select events with selected modules with SiStripDigis or SiStripClusters

 Implementation:

*/
//
// Original Author:  Andrea Venturi
//         Created:  Wed Oct 22 17:54:30 CEST 2008
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

#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

//
// class declaration
//

template <class T>
class SiStripDetWithSomething : public edm::EDFilter {
   public:
      explicit SiStripDetWithSomething(const edm::ParameterSet&);
      ~SiStripDetWithSomething();

   private:
      virtual void beginJob() override ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      // ----------member data ---------------------------

  edm::EDGetTokenT<T> _digicollectionToken;
  std::vector<unsigned int> _wantedmod;

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
template <class T>
SiStripDetWithSomething<T>::SiStripDetWithSomething(const edm::ParameterSet& iConfig):
  _digicollectionToken(consumes<T>(iConfig.getParameter<edm::InputTag>("collectionName"))),
  _wantedmod(iConfig.getUntrackedParameter<std::vector<unsigned int> >("selectedModules"))

{
   //now do what ever initialization is needed

  sort(_wantedmod.begin(),_wantedmod.end());

  edm::LogInfo("SelectedModules") << "Selected module list";
  for(std::vector<unsigned int>::const_iterator mod = _wantedmod.begin();mod!=_wantedmod.end();mod++) {
    edm::LogVerbatim("SelectedModules") << *mod ;
  }

}


template <class T>
SiStripDetWithSomething<T>::~SiStripDetWithSomething()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
template <class T>
bool
SiStripDetWithSomething<T>::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<T> digis;
   iEvent.getByToken(_digicollectionToken,digis);

   for(typename T::const_iterator it = digis->begin();it!=digis->end();it++) {

     for(std::vector<unsigned int>::const_iterator mod=_wantedmod.begin();
	 mod!=_wantedmod.end()&&it->detId()>=*mod;
	 mod++) {
       if(*mod == it->detId()) {
	 edm::LogInfo("ModuleFound") << " module " << *mod << " found with "
				     << it->size() << " digis/clusters";
	 return true;
       }
     }
   }

   return false;
}

// ------------ method called once each job just before starting event loop  ------------
template <class T>
void
SiStripDetWithSomething<T>::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
template <class T>
void
SiStripDetWithSomething<T>::endJob() {
}

typedef SiStripDetWithSomething<edm::DetSetVector<SiStripDigi> > SiStripDetWithDigi;
typedef SiStripDetWithSomething<edmNew::DetSetVector<SiStripCluster> > SiStripDetWithCluster;

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripDetWithDigi);
DEFINE_FWK_MODULE(SiStripDetWithCluster);
