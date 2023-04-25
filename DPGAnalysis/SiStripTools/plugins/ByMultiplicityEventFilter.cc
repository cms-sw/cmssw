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
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "MultiplicityAlgorithms.h"

//
// class declaration
//

template <class T>
class ByMultiplicityEventFilter : public edm::stream::EDFilter<> {
public:
  explicit ByMultiplicityEventFilter(const edm::ParameterSet&);
  ~ByMultiplicityEventFilter() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  T m_multiplicities;
  StringCutObjectSelector<typename T::value_t> m_selector;
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
template <class T>
ByMultiplicityEventFilter<T>::ByMultiplicityEventFilter(const edm::ParameterSet& iConfig)
    : m_multiplicities(iConfig.getParameter<edm::ParameterSet>("multiplicityConfig"), consumesCollector()),
      m_selector(iConfig.getParameter<std::string>("cut")),
      m_taggedMode(iConfig.getUntrackedParameter<bool>("taggedMode", false)),
      m_forcedValue(iConfig.getUntrackedParameter<bool>("forcedValue", true))

{
  //now do what ever initialization is needed
  produces<bool>();
}

template <class T>
ByMultiplicityEventFilter<T>::~ByMultiplicityEventFilter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called on each new Event  ------------
template <class T>
bool ByMultiplicityEventFilter<T>::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  auto mult = m_multiplicities.getEvent(iEvent, iSetup);

  bool value = m_selector(mult);
  iEvent.put(std::make_unique<bool>(value));

  if (m_taggedMode)
    return m_forcedValue;
  return value;
}

//define this as a plug-in
/*
typedef ByMultiplicityEventFilter<SingleMultiplicity<edm::DetSetVector<SiStripDigi> > > BySiStripDigiMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<SingleMultiplicity<edmNew::DetSetVector<SiStripCluster> > > BySiStripClusterMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<SingleMultiplicity<edmNew::DetSetVector<SiPixelCluster> > > BySiPixelClusterMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<MultiplicityPair<edmNew::DetSetVector<SiPixelCluster>,edmNew::DetSetVector<SiStripCluster> > > BySiPixelClusterVsSiStripClusterMultiplicityEventFilter;
*/
using namespace sistriptools::algorithm;
typedef ByMultiplicityEventFilter<SingleSiStripDigiMultiplicity> BySiStripDigiMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<SingleSiStripClusterMultiplicity> BySiStripClusterMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<SingleSiPixelClusterMultiplicity> BySiPixelClusterMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<SiPixelClusterSiStripClusterMultiplicityPair>
    BySiPixelClusterVsSiStripClusterMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<ClusterSummarySingleMultiplicity> ByClusterSummarySingleMultiplicityEventFilter;
typedef ByMultiplicityEventFilter<ClusterSummaryMultiplicityPair> ByClusterSummaryMultiplicityPairEventFilter;

DEFINE_FWK_MODULE(BySiStripDigiMultiplicityEventFilter);
DEFINE_FWK_MODULE(BySiStripClusterMultiplicityEventFilter);
DEFINE_FWK_MODULE(BySiPixelClusterMultiplicityEventFilter);
DEFINE_FWK_MODULE(BySiPixelClusterVsSiStripClusterMultiplicityEventFilter);
DEFINE_FWK_MODULE(ByClusterSummarySingleMultiplicityEventFilter);
DEFINE_FWK_MODULE(ByClusterSummaryMultiplicityPairEventFilter);
