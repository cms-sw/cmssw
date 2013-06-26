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
  ~FromClusterSummaryMultiplicityProducer();
  
private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;

      // ----------member data ---------------------------

  edm::InputTag m_collection;
  std::map<unsigned int, std::string> m_subdets;
  std::map<unsigned int, int> m_subdetenums;
  std::map<unsigned int, std::string> m_subdetvars;
  std::vector<std::string> m_clustsummvar;

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
  m_collection(iConfig.getParameter<edm::InputTag>("clusterSummaryCollection")),
  m_subdets(),m_subdetenums(),m_subdetvars(),m_clustsummvar()
{

  m_clustsummvar.push_back("cHits");
  m_clustsummvar.push_back("cSize");
  m_clustsummvar.push_back("cCharge");
  m_clustsummvar.push_back("pHits");
  m_clustsummvar.push_back("pSize");
  m_clustsummvar.push_back("pCharge");

  produces<std::map<unsigned int,int> >();

   //now do what ever other initialization is needed

  std::vector<edm::ParameterSet> wantedsubds(iConfig.getParameter<std::vector<edm::ParameterSet> >("wantedSubDets"));
					     
  for(std::vector<edm::ParameterSet>::iterator ps=wantedsubds.begin();ps!=wantedsubds.end();++ps) {
    m_subdets[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<std::string>("detLabel");
    m_subdetenums[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<int >("subDetEnum");
    m_subdetvars[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<std::string>("subDetVariable");
  }
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

  std::auto_ptr<std::map<unsigned int,int> > mults(new std::map<unsigned int,int> );
  
  
  Handle<ClusterSummary> clustsumm;
  iEvent.getByLabel(m_collection,clustsumm);

  clustsumm->SetUserContent(m_clustsummvar);
  
  for(std::map<unsigned int,std::string>::const_iterator sdet=m_subdets.begin();sdet!=m_subdets.end();++sdet) { (*mults)[sdet->first]=0; }

  for(std::map<unsigned int,int>::const_iterator detsel=m_subdetenums.begin();detsel!=m_subdetenums.end();++detsel) {

    //    (*mults)[detsel->first] = int(clustsumm->GetGenericVariable(m_subdetvars[detsel->first])[clustsumm->GetModuleLocation(detsel->second)]);
    (*mults)[detsel->first] = int(clustsumm->GetGenericVariable(m_subdetvars[detsel->first],detsel->second));
    LogDebug("Multiplicity") << "GetModuleLocation result: " << detsel->second << " " << clustsumm->GetModuleLocation(detsel->second);
  }

  
  
  for(std::map<unsigned int,int>::const_iterator it=mults->begin();it!=mults->end();++it) {
    LogDebug("Multiplicity") << " Found " << it->second << " digis/clusters in " << it->first << " " << m_subdets[it->first];
  }
  
  iEvent.put(mults);
  
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
