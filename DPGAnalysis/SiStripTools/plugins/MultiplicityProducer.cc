// -*- C++ -*-
//
// Package:    MultiplicityProducer
// Class:      MultiplicityProducer
// 
/**\class MultiplicityProducer MultiplicityProducer.cc DPGAnalysis/SiStripTools/plugins/MultiplicityProducer.cc

 Description: EDProducer of multiplicity maps
 Implementation:
     
*/
//
// Original Author:  Andrea Venturi
//         Created:  Fri Dec 04 2009
//
//


// system include files
#include <memory>
#include <string>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DPGAnalysis/SiStripTools/interface/DetIdSelector.h"
//
// class decleration
//
template <class T>
class MultiplicityProducer : public edm::EDProducer {
   public:
      explicit MultiplicityProducer(const edm::ParameterSet&);
      ~MultiplicityProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  edm::InputTag m_collection;
  std::map<unsigned int, std::string> m_subdets;
  std::map<unsigned int, DetIdSelector> m_subdetsels;

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
MultiplicityProducer<T>::MultiplicityProducer(const edm::ParameterSet& iConfig):
  m_collection(iConfig.getParameter<edm::InputTag>("clusterdigiCollection")),
  m_subdets(),m_subdetsels()
{

  produces<std::map<unsigned int,int> >();

   //now do what ever other initialization is needed

  std::vector<edm::ParameterSet> wantedsubds(iConfig.getParameter<std::vector<edm::ParameterSet> >("wantedSubDets"));
					     
  for(std::vector<edm::ParameterSet>::iterator ps=wantedsubds.begin();ps!=wantedsubds.end();++ps) {
    m_subdets[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<std::string>("detLabel");
    m_subdetsels[ps->getParameter<unsigned int>("detSelection")] = 
      DetIdSelector(ps->getUntrackedParameter<std::vector<std::string> >("selection",std::vector<std::string>()));
  }
}

template <class T>
MultiplicityProducer<T>::~MultiplicityProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
template <class T>
void
MultiplicityProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  LogDebug("Multiplicity") << " Ready to loop";

  using namespace edm;

  std::auto_ptr<std::map<unsigned int,int> > mults(new std::map<unsigned int,int> );
  
  
  Handle<T> digis;
  iEvent.getByLabel(m_collection,digis);
  
  for(std::map<unsigned int,std::string>::const_iterator sdet=m_subdets.begin();sdet!=m_subdets.end();++sdet) { (*mults)[sdet->first]=0; }

  
  for(typename T::const_iterator det = digis->begin();det!=digis->end();++det) {
    
    if(m_subdets.find(0)!=m_subdets.end()) (*mults)[0]+= det->size();

    DetId detid(det->detId());
    unsigned int subdet = detid.subdetId();

    if(m_subdets.find(subdet)!=m_subdets.end() && !m_subdetsels[subdet].isValid() ) (*mults)[subdet] += det->size();

    for(std::map<unsigned int,DetIdSelector>::const_iterator detsel=m_subdetsels.begin();detsel!=m_subdetsels.end();++detsel) {

      if(detsel->second.isValid() && detsel->second.isSelected(detid)) (*mults)[detsel->first] += det->size();

    }

  }
  
  
  for(std::map<unsigned int,int>::const_iterator it=mults->begin();it!=mults->end();++it) {
    LogDebug("Multiplicity") << " Found " << it->second << " digis/clusters in " << it->first << " " << m_subdets[it->first];
  }
  
  iEvent.put(mults);
  
}

// ------------ method called once each job just before starting event loop  ------------
template <class T>
void 
MultiplicityProducer<T>::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
template <class T>
void 
MultiplicityProducer<T>::endJob() {
}

//define this as a plug-in
typedef MultiplicityProducer<edmNew::DetSetVector<SiStripCluster> > SiStripClusterMultiplicityProducer;
typedef MultiplicityProducer<edmNew::DetSetVector<SiPixelCluster> > SiPixelClusterMultiplicityProducer;
typedef MultiplicityProducer<edm::DetSetVector<SiStripDigi> > SiStripDigiMultiplicityProducer;


DEFINE_FWK_MODULE(SiStripClusterMultiplicityProducer);
DEFINE_FWK_MODULE(SiPixelClusterMultiplicityProducer);
DEFINE_FWK_MODULE(SiStripDigiMultiplicityProducer);
