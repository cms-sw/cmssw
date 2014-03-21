// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      ByMultiplicityEventFilter
//
/**\class ByMultiplicityEventFilter ByMultiplicityEventFilter.cc DPGAnalysis/SiStripTools/ByMultiplicityEventFilter.cc

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

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DPGAnalysis/SiStripTools/interface/Multiplicities.h"


//
// class declaration
//

template <class T, class M>
class ByMultiplicityEventFilter : public edm::EDFilter {
   public:
      explicit ByMultiplicityEventFilter(const edm::ParameterSet&);
      ~ByMultiplicityEventFilter();


   private:
      virtual void beginJob() override ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      // ----------member data ---------------------------

  edm::ParameterSet m_multiplicityConfig;
  T m_multiplicities;
  edm::EDGetTokenT<M> m_collectionToken;
  StringCutObjectSelector<T> m_selector;
  bool m_taggedMode, m_forcedValue;

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
template <class T, class M>
ByMultiplicityEventFilter<T,M>::ByMultiplicityEventFilter(const edm::ParameterSet& iConfig):
  m_multiplicityConfig(iConfig.getParameter<edm::ParameterSet>("multiplicityConfig")),
  m_multiplicities(m_multiplicityConfig),
  m_collectionToken(consumes<M>(m_multiplicityConfig.getParameter<edm::InputTag>("collectionName"))),
  m_selector(iConfig.getParameter<std::string>("cut")),
  m_taggedMode(iConfig.getUntrackedParameter<bool>("taggedMode", false)),
  m_forcedValue(iConfig.getUntrackedParameter<bool>("forcedValue", true))


{
   //now do what ever initialization is needed
  produces<bool>();

}

template <class T, class M>
ByMultiplicityEventFilter<T,M>::~ByMultiplicityEventFilter()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
template <class T, class M>
bool
ByMultiplicityEventFilter<T,M>::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<M> digis;
   iEvent.getByToken(m_collectionToken,digis);
   m_multiplicities.getEvent(iEvent,iSetup,digis);

   bool value = m_selector(m_multiplicities);
   iEvent.put( std::auto_ptr<bool>(new bool(value)) );

   if(m_taggedMode) return m_forcedValue;
   return value;

}

// ------------ method called once each job just before starting event loop  ------------
template <class T, class M>
void
ByMultiplicityEventFilter<T,M>::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
template <class T, class M>
void
ByMultiplicityEventFilter<T,M>::endJob() {
}



//
// class declaration
//

template <class T, class M1, class M2>
class ByMultiplicityPairEventFilter : public edm::EDFilter {
   public:
      explicit ByMultiplicityPairEventFilter(const edm::ParameterSet&);
      ~ByMultiplicityPairEventFilter();


   private:
      virtual void beginJob() override ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      // ----------member data ---------------------------

  edm::ParameterSet m_multiplicityConfig;
  T m_multiplicities;
  edm::ParameterSet m_firstMultiplicityConfig;
  edm::EDGetTokenT<M1> m_firstCollectionToken;
  edm::ParameterSet m_secondMultiplicityConfig;
  edm::EDGetTokenT<M2> m_secondCollectionToken;
  StringCutObjectSelector<T> m_selector;
  bool m_taggedMode, m_forcedValue;

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
template <class T, class M1, class M2>
ByMultiplicityPairEventFilter<T,M1,M2>::ByMultiplicityPairEventFilter(const edm::ParameterSet& iConfig):
  m_multiplicityConfig(iConfig.getParameter<edm::ParameterSet>("multiplicityConfig")),
  m_multiplicities(m_multiplicityConfig),
  m_firstMultiplicityConfig(m_multiplicityConfig.getParameter<edm::ParameterSet>("firstMultiplicityConfig")),
  m_firstCollectionToken(consumes<M1>(m_firstMultiplicityConfig.getParameter<edm::InputTag>("collectionName"))),
  m_secondMultiplicityConfig(m_multiplicityConfig.getParameter<edm::ParameterSet>("secondMultiplicityConfig")),
  m_secondCollectionToken(consumes<M2>(m_secondMultiplicityConfig.getParameter<edm::InputTag>("collectionName"))),
  m_selector(iConfig.getParameter<std::string>("cut")),
  m_taggedMode(iConfig.getUntrackedParameter<bool>("taggedMode", false)),
  m_forcedValue(iConfig.getUntrackedParameter<bool>("forcedValue", true))


{
   //now do what ever initialization is needed
  produces<bool>();

}

template <class T, class M1, class M2>
ByMultiplicityPairEventFilter<T,M1,M2>::~ByMultiplicityPairEventFilter()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
template <class T, class M1, class M2>
bool
ByMultiplicityPairEventFilter<T,M1,M2>::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<M1> digis1;
   iEvent.getByToken(m_firstCollectionToken,digis1);
   Handle<M2> digis2;
   iEvent.getByToken(m_secondCollectionToken,digis2);
   m_multiplicities.getEvent(iEvent,iSetup,digis1,digis2);

   bool value = m_selector(m_multiplicities);
   iEvent.put( std::auto_ptr<bool>(new bool(value)) );

   if(m_taggedMode) return m_forcedValue;
   return value;

}

// ------------ method called once each job just before starting event loop  ------------
template <class T, class M1, class M2>
void
ByMultiplicityPairEventFilter<T,M1,M2>::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
template <class T, class M1, class M2>
void
ByMultiplicityPairEventFilter<T,M1,M2>::endJob() {
}



//define this as a plug-in
/*
typedef ByMultiplicityEventFilter<SingleMultiplicity<edm::DetSetVector<SiStripDigi> >, edm::DetSetVector<SiStripDigi> > BySiStripDigiMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<SingleMultiplicity<edmNew::DetSetVector<SiStripCluster> >, edmNew::DetSetVector<SiStripCluster> > BySiStripClusterMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<SingleMultiplicity<edmNew::DetSetVector<SiPixelCluster> >, edmNew::DetSetVector<SiPixelCluster> > BySiPixelClusterMultiplicityEventFilter;
typedef ByMultiplicityPairEventFilter<MultiplicityPair<edmNew::DetSetVector<SiPixelCluster>, edmNew::DetSetVector<SiPixelCluster>,edmNew::DetSetVector<SiStripCluster> >, edmNew::DetSetVector<SiStripCluster> > BySiPixelClusterVsSiStripClusterMultiplicityEventFilter;
*/
typedef ByMultiplicityEventFilter<SingleSiStripDigiMultiplicity, edm::DetSetVector<SiStripDigi> > BySiStripDigiMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<SingleSiStripClusterMultiplicity, edmNew::DetSetVector<SiStripCluster> > BySiStripClusterMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<SingleSiPixelClusterMultiplicity, edmNew::DetSetVector<SiPixelCluster> > BySiPixelClusterMultiplicityEventFilter;
typedef ByMultiplicityPairEventFilter<SiPixelClusterSiStripClusterMultiplicityPair, edmNew::DetSetVector<SiPixelCluster>, edmNew::DetSetVector<SiStripCluster> > BySiPixelClusterVsSiStripClusterMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<ClusterSummarySingleMultiplicity, ClusterSummary> ByClusterSummarySingleMultiplicityEventFilter;
typedef ByMultiplicityPairEventFilter<ClusterSummaryMultiplicityPair, ClusterSummary, ClusterSummary> ByClusterSummaryMultiplicityPairEventFilter;

DEFINE_FWK_MODULE(BySiStripDigiMultiplicityEventFilter);
DEFINE_FWK_MODULE(BySiStripClusterMultiplicityEventFilter);
DEFINE_FWK_MODULE(BySiPixelClusterMultiplicityEventFilter);
DEFINE_FWK_MODULE(BySiPixelClusterVsSiStripClusterMultiplicityEventFilter);
DEFINE_FWK_MODULE(ByClusterSummarySingleMultiplicityEventFilter);
DEFINE_FWK_MODULE(ByClusterSummaryMultiplicityPairEventFilter);
