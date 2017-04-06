#ifndef DPGAnalysis_SiStripTools_Multiplicities_H
#define DPGAnalysis_SiStripTools_Multiplicities_H

#ifndef __GCCXML__
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#endif

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerCommon/interface/ClusterSummary.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

namespace edm {
  class EventSetup;
};

#include <string>

class ClusterSummarySingleMultiplicity {

 public:
  ClusterSummarySingleMultiplicity();
#ifndef __GCCXML__
  ClusterSummarySingleMultiplicity(const edm::ParameterSet& iConfig,edm::ConsumesCollector&& iC);
  ClusterSummarySingleMultiplicity(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC);
#endif

  void getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  int mult() const;

 private:
#ifndef __GCCXML__
  edm::EDGetTokenT<ClusterSummary> m_collection;
#endif
  ClusterSummary::CMSTracker m_subdetenum;
  ClusterSummary::VariablePlacement m_varenum;
  int m_mult;
    
};



template <class T>
class SingleMultiplicity {
  
 public:
  SingleMultiplicity();
#ifndef __GCCXML__
  SingleMultiplicity(const edm::ParameterSet& iConfig,edm::ConsumesCollector&& iC);
  SingleMultiplicity(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC);
#endif  
  
  void getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  int mult() const;
  //  int mult;
  
 private:
  
#ifndef __GCCXML__
  edm::EDGetTokenT<T> m_collection;
#endif
  int m_modthr;
  bool m_useQuality;
  std::string m_qualityLabel;
  int m_mult;
};

template <class T>
SingleMultiplicity<T>::SingleMultiplicity():
  //  mult(0), 
#ifndef __GCCXML__
m_collection(), 
#endif
    m_modthr(-1), m_useQuality(false),  m_qualityLabel(),
  m_mult(0)
{ }

#ifndef __GCCXML__
template <class T>
SingleMultiplicity<T>::SingleMultiplicity(const edm::ParameterSet& iConfig,edm::ConsumesCollector&& iC):
  //  mult(0),
m_collection(iC.consumes<T>(iConfig.getParameter<edm::InputTag>("collectionName"))),
  m_modthr(iConfig.getUntrackedParameter<int>("moduleThreshold")),
  m_useQuality(iConfig.getUntrackedParameter<bool>("useQuality",false)),
  m_qualityLabel(iConfig.getUntrackedParameter<std::string>("qualityLabel","")),
  m_mult(0)
{ }
template <class T>
SingleMultiplicity<T>::SingleMultiplicity(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
  //  mult(0),
m_collection(iC.consumes<T>(iConfig.getParameter<edm::InputTag>("collectionName"))),
  m_modthr(iConfig.getUntrackedParameter<int>("moduleThreshold")),
  m_useQuality(iConfig.getUntrackedParameter<bool>("useQuality",false)),
  m_qualityLabel(iConfig.getUntrackedParameter<std::string>("qualityLabel","")),
  m_mult(0)
{ }
#endif

#ifndef __GCCXML__
template <class T>
void
SingleMultiplicity<T>::getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  m_mult = 0;
  //  mult = 0;

  edm::ESHandle<SiStripQuality> qualityHandle;
   if( m_useQuality) {
     iSetup.get<SiStripQualityRcd>().get(m_qualityLabel,qualityHandle);
   }

   edm::Handle<T> digis;
   iEvent.getByToken(m_collection,digis);


   for(typename T::const_iterator it = digis->begin();it!=digis->end();it++) {

     if(!m_useQuality || !qualityHandle->IsModuleBad(it->detId()) ) {
       if(m_modthr < 0 || int(it->size()) < m_modthr ) {
	 m_mult += it->size();
	 //	 mult += it->size();
       }
     }
   }
}
#endif


template<class T>
int SingleMultiplicity<T>::mult() const { return m_mult; }

template <class T1, class T2>
  class MultiplicityPair {
    
 public:
    MultiplicityPair();
#ifndef __GCCXML__
    MultiplicityPair(const edm::ParameterSet& iConfig,edm::ConsumesCollector&& iC);
    MultiplicityPair(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC);
#endif
    
    void getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup);
    int mult1() const;
    int mult2() const;
    //    int mult1;
    //    int mult2;
    
 private:
    
    T1 m_multiplicity1;
    T2 m_multiplicity2;
    
  };

template <class T1, class T2>
  MultiplicityPair<T1,T2>::MultiplicityPair():
    //    mult1(0),mult2(0),  
    m_multiplicity1(),  m_multiplicity2()
{ }

#ifndef __GCCXML__
template <class T1, class T2>
      MultiplicityPair<T1,T2>::MultiplicityPair(const edm::ParameterSet& iConfig,edm::ConsumesCollector&& iC):
    //    mult1(0),mult2(0),
  m_multiplicity1(iConfig.getParameter<edm::ParameterSet>("firstMultiplicityConfig"),iC),
    m_multiplicity2(iConfig.getParameter<edm::ParameterSet>("secondMultiplicityConfig"),iC)
{ }
template <class T1, class T2>
      MultiplicityPair<T1,T2>::MultiplicityPair(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    //    mult1(0),mult2(0),
  m_multiplicity1(iConfig.getParameter<edm::ParameterSet>("firstMultiplicityConfig"),iC),
    m_multiplicity2(iConfig.getParameter<edm::ParameterSet>("secondMultiplicityConfig"),iC)
{ }
#endif

template <class T1, class T2>
  void
  MultiplicityPair<T1,T2>::getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  m_multiplicity1.getEvent(iEvent,iSetup);
  m_multiplicity2.getEvent(iEvent,iSetup);

  //  mult1=m_multiplicity1.mult;
  //  mult2=m_multiplicity2.mult;
  
}

template<class T1, class T2>
  int MultiplicityPair<T1,T2>::mult1() const { return m_multiplicity1.mult(); }

template<class T1, class T2>
  int MultiplicityPair<T1,T2>::mult2() const { return m_multiplicity2.mult(); }

typedef SingleMultiplicity<edm::DetSetVector<SiStripDigi> > SingleSiStripDigiMultiplicity;
typedef SingleMultiplicity<edmNew::DetSetVector<SiStripCluster> > SingleSiStripClusterMultiplicity;
typedef SingleMultiplicity<edmNew::DetSetVector<SiPixelCluster> > SingleSiPixelClusterMultiplicity;
typedef MultiplicityPair<SingleMultiplicity<edmNew::DetSetVector<SiPixelCluster> > ,SingleMultiplicity<edmNew::DetSetVector<SiStripCluster> > > 
SiPixelClusterSiStripClusterMultiplicityPair; 
typedef MultiplicityPair<ClusterSummarySingleMultiplicity,ClusterSummarySingleMultiplicity> ClusterSummaryMultiplicityPair; 


#endif // DPGAnalysis_SiStripTools_Multiplicities_H
