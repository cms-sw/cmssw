// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      LargeEvents
//
/**\class LargeEvents LargeEvents.cc DPGAnalysis/SiStripTools/LargeEvents.cc

 Description: templated EDFilter to select events with large number of SiStripDigi or SiStripCluster

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Tue Oct 21 20:55:22 CEST 2008
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

//
// class declaration
//

template <class T>
class LargeEvents : public edm::EDFilter {
   public:
      explicit LargeEvents(const edm::ParameterSet&);
      ~LargeEvents();

   private:
      virtual void beginJob() override ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      // ----------member data ---------------------------

  edm::EDGetTokenT<T> _collectionToken;
  int _absthr;
  int _modthr;
  bool _useQuality;
  std::string _qualityLabel;
  edm::ESHandle<SiStripQuality> _qualityHandle;
  edm::ESWatcher<SiStripQualityRcd> _qualityWatcher;

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
LargeEvents<T>::LargeEvents(const edm::ParameterSet& iConfig):
  _collectionToken(consumes<T>(iConfig.getParameter<edm::InputTag>("collectionName"))),
  _absthr(iConfig.getUntrackedParameter<int>("absoluteThreshold")),
  _modthr(iConfig.getUntrackedParameter<int>("moduleThreshold")),
  _useQuality(iConfig.getUntrackedParameter<bool>("useQuality",false)),
  _qualityLabel(iConfig.getUntrackedParameter<std::string>("qualityLabel",""))
{
   //now do what ever initialization is needed


}

template <class T>
LargeEvents<T>::~LargeEvents()
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
LargeEvents<T>::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   if( _useQuality) {
     if(_qualityWatcher.check(iSetup)) {
       iSetup.get<SiStripQualityRcd>().get(_qualityLabel,_qualityHandle);
       LogDebug("SiStripQualityUpdated") << "SiStripQuality has changed and it will be updated";
     }
   }

   Handle<T> digis;
   iEvent.getByToken(_collectionToken,digis);


   int ndigitot = 0;
   for(typename T::const_iterator it = digis->begin();it!=digis->end();it++) {

     if(!_useQuality || !_qualityHandle->IsModuleBad(it->detId()) ) {
       if(_modthr < 0 || int(it->size()) < _modthr ) {
	 ndigitot += it->size();
       }
     }
   }

   if(ndigitot > _absthr) {
     LogDebug("LargeEventSelected") << "event with " << ndigitot << " digi/cluster selected";
     return true;
   }

   return false;
}

// ------------ method called once each job just before starting event loop  ------------
template <class T>
void
LargeEvents<T>::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
template <class T>
void
LargeEvents<T>::endJob() {
}

//define this as a plug-in
typedef LargeEvents<edm::DetSetVector<SiStripDigi> > LargeSiStripDigiEvents;
typedef LargeEvents<edmNew::DetSetVector<SiStripCluster> > LargeSiStripClusterEvents;
typedef LargeEvents<edmNew::DetSetVector<SiPixelCluster> > LargeSiPixelClusterEvents;

DEFINE_FWK_MODULE(LargeSiStripDigiEvents);
DEFINE_FWK_MODULE(LargeSiStripClusterEvents);
DEFINE_FWK_MODULE(LargeSiPixelClusterEvents);
