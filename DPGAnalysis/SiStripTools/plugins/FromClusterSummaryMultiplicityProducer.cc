// -*- C++ -*-
//
// Package:    FromClusterSummaryMultiplicityProducer
// Class:      FromClusterSummaryMultiplicityProducer
//
/**\class FromClusterSummaryMultiplicityProducer FromClusterSummaryMultiplicityProducer.cc DPGAnalysis/SiStripTools/plugins/FromClusterSummaryMultiplicityProducer.cc

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

#include "DataFormats/TrackerCommon/interface/ClusterSummary.h"

//
// class decleration
//
class FromClusterSummaryMultiplicityProducer : public edm::EDProducer {

public:
  explicit FromClusterSummaryMultiplicityProducer(const edm::ParameterSet&);
  ~FromClusterSummaryMultiplicityProducer() override;

private:
  void beginJob() override ;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override ;

      // ----------member data ---------------------------

  edm::EDGetTokenT<ClusterSummary>        m_collectionToken;
  std::vector<ClusterSummary::CMSTracker> m_subdetenums;
  std::vector<int>                        m_subdetsel;
  ClusterSummary::VariablePlacement       m_subdetvar;

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
FromClusterSummaryMultiplicityProducer::FromClusterSummaryMultiplicityProducer(const edm::ParameterSet& iConfig):
  m_collectionToken(consumes<ClusterSummary>(iConfig.getParameter<edm::InputTag>("clusterSummaryCollection"))),
  m_subdetenums(),m_subdetsel(),m_subdetvar(ClusterSummary::NCLUSTERS)
{
  produces<std::map<unsigned int,int> >();

   //now do what ever other initialization is needed

  std::vector<edm::ParameterSet> wantedsubds(iConfig.getParameter<std::vector<edm::ParameterSet> >("wantedSubDets"));
  m_subdetenums.reserve(wantedsubds.size());
  m_subdetsel.reserve(wantedsubds.size());

  for(std::vector<edm::ParameterSet>::iterator ps=wantedsubds.begin();ps!=wantedsubds.end();++ps) {
    m_subdetenums.push_back((ClusterSummary::CMSTracker)ps->getParameter<int >("subDetEnum"));
    m_subdetsel.push_back(ps->getParameter<int >("subDetEnum"));
  }
  m_subdetvar = (ClusterSummary::VariablePlacement)iConfig.getParameter<int>("varEnum");
}

FromClusterSummaryMultiplicityProducer::~FromClusterSummaryMultiplicityProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
FromClusterSummaryMultiplicityProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  LogDebug("Multiplicity") << " Ready to go";

  using namespace edm;

  std::unique_ptr<std::map<unsigned int,int> > mults(new std::map<unsigned int,int> );


  Handle<ClusterSummary> clustsumm;
  iEvent.getByToken(m_collectionToken,clustsumm);

  switch(m_subdetvar){
    case ClusterSummary::NCLUSTERS     :
      for(unsigned int iS = 0; iS < m_subdetenums.size(); ++iS)
        (*mults)[m_subdetsel[iS]] = int(clustsumm->getNClus     (m_subdetenums[iS]));
      break;
    case ClusterSummary::CLUSTERSIZE   :
      for(unsigned int iS = 0; iS < m_subdetenums.size(); ++iS)
        (*mults)[m_subdetsel[iS]] = int(clustsumm->getClusSize     (m_subdetenums[iS]));
      break;
    case ClusterSummary::CLUSTERCHARGE :
      for(unsigned int iS = 0; iS < m_subdetenums.size(); ++iS)
        (*mults)[m_subdetsel[iS]] = int(clustsumm->getClusCharge     (m_subdetenums[iS]));
      break;
    default :
      for(unsigned int iS = 0; iS < m_subdetenums.size(); ++iS)
        (*mults)[m_subdetsel[iS]] = -1;
  }

  for(unsigned int iS = 0; iS < m_subdetenums.size(); ++iS)
    LogDebug("Multiplicity") << "GetModuleLocation result: " << m_subdetenums[iS] << " " << clustsumm->getModuleLocation(m_subdetenums[iS]);

  for(std::map<unsigned int,int>::const_iterator it=mults->begin();it!=mults->end();++it) {
    LogDebug("Multiplicity") << " Found " << it->second << " digis/clusters in " << it->first;
  }

  iEvent.put(std::move(mults));

}

// ------------ method called once each job just before starting event loop  ------------
void
FromClusterSummaryMultiplicityProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
FromClusterSummaryMultiplicityProducer::endJob() {
}

DEFINE_FWK_MODULE(FromClusterSummaryMultiplicityProducer);
