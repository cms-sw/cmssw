// -*- C++ -*-
//
// Package:    Utilities
// Class:      SimpleJetFilter
// 
/**\class SimpleJetFilter SimpleJetFilter.cc DPGAnalysis/Skims/src/SimpleJetFilter.cc

 Description: 

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

#include "DataFormats/JetReco/interface/JetID.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/strbitset.h"

//
// class declaration
//

class SimpleJetFilter : public edm::EDFilter {
   public:
      explicit SimpleJetFilter(const edm::ParameterSet&);
      ~SimpleJetFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  edm::InputTag m_jetCollection;
  edm::InputTag m_jetIDMap;
  const double m_ptcut;
  const double m_etamaxcut;
  const double m_njetmin;
  JetIDSelectionFunctor m_jetIDfunc;

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
SimpleJetFilter::SimpleJetFilter(const edm::ParameterSet& iConfig):
  m_jetCollection(iConfig.getParameter<edm::InputTag>("jetCollection")),
  m_jetIDMap(iConfig.getParameter<edm::InputTag>("jetIDMap")),
  m_ptcut(iConfig.getParameter<double>("ptCut")),
  m_etamaxcut(iConfig.getParameter<double>("maxRapidityCut")),
  m_njetmin(iConfig.getParameter<unsigned int>("nJetMin")),
  m_jetIDfunc(JetIDSelectionFunctor::PURE09,JetIDSelectionFunctor::LOOSE)
{
   //now do what ever initialization is needed


}

SimpleJetFilter::~SimpleJetFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
SimpleJetFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  bool selected = false;
  
  Handle<reco::CaloJetCollection> jetcoll;
  iEvent.getByLabel(m_jetCollection,jetcoll);
  
  Handle<reco::JetIDValueMap> jetIDmap;
  iEvent.getByLabel(m_jetIDMap,jetIDmap);
  
  unsigned int goodjets = 0;

  for(unsigned int ijet=0;ijet<jetcoll->size();++ijet) {
    
    const reco::CaloJetRef jet(jetcoll,ijet);

    LogDebug("JetUnderTest") << "Jet with eta = " << jet->eta() << " and pt = " << jet->pt() << " under test";

    if( !(std::abs(jet->eta()) < m_etamaxcut && jet->pt() > m_ptcut )) continue;

    LogDebug("JetUnderTest") << "kincut passed";

    if(jetIDmap->contains(jet.id())) {
      
      const reco::JetID & jetid = (*jetIDmap)[jet];
      pat::strbitset ret = m_jetIDfunc.getBitTemplate();
      ret.set(false);
      bool goodjet = m_jetIDfunc((*jetcoll)[ijet],jetid,ret);
      if(goodjet) { 
	++goodjets;
	LogDebug("JetUnderTest") << "JetID passed";
      }
      if(goodjets >= m_njetmin) return true;
      
    } else {
      edm::LogWarning("JetIDNotFound") << "JetID not found ";
      
    }
    
  }
  
  return selected;
}

// ------------ method called once each job just before starting event loop  ------------
void 
SimpleJetFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SimpleJetFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(SimpleJetFilter);
