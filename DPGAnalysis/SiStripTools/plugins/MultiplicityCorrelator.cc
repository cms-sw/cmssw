// -*- C++ -*-
//
// Package:    MultiplicityCorrelator
// Class:      MultiplicityCorrelator
//
/**\class MultiplicityCorrelator MultiplicityCorrelator.cc DPGAnalysis/SiStripTools/src/MultiplicityCorrelator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Mon Oct 27 17:37:53 CET 2008
// $Id: MultiplicityCorrelator.cc,v 1.3 2011/12/11 10:04:09 venturia Exp $
//
//


// system include files
#include <memory>

// user include files

#include <vector>
#include <map>
#include <limits>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DPGAnalysis/SiStripTools/interface/MultiplicityCorrelatorHistogramMaker.h"

//
// class decleration
//

class MultiplicityCorrelator : public edm::EDAnalyzer {
   public:
      explicit MultiplicityCorrelator(const edm::ParameterSet&);
      ~MultiplicityCorrelator();


private:
  virtual void beginJob() override ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void endJob() override ;

      // ----------member data ---------------------------

  std::vector<MultiplicityCorrelatorHistogramMaker*> m_mchms;

  std::vector<edm::EDGetTokenT<std::map<unsigned int, int> > > m_xMultiplicityMapTokens;
  std::vector<edm::EDGetTokenT<std::map<unsigned int, int> > > m_yMultiplicityMapTokens;
  std::vector<std::string> m_xLabels;
  std::vector<std::string> m_yLabels;
  std::vector<unsigned int> m_xSelections;
  std::vector<unsigned int> m_ySelections;


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
MultiplicityCorrelator::MultiplicityCorrelator(const edm::ParameterSet& iConfig):
  m_mchms(),
  m_xMultiplicityMapTokens(),m_yMultiplicityMapTokens(),
  m_xLabels(),m_yLabels(), m_xSelections(),m_ySelections()
{
   //now do what ever initialization is needed

  std::vector<edm::ParameterSet> correlationConfigs =
    iConfig.getParameter<std::vector<edm::ParameterSet> >("correlationConfigurations");

  for(std::vector<edm::ParameterSet>::const_iterator ps=correlationConfigs.begin();ps!=correlationConfigs.end();++ps) {

    m_xMultiplicityMapTokens.push_back(consumes<std::map<unsigned int, int> >(ps->getParameter<edm::InputTag>("xMultiplicityMap")));
    m_yMultiplicityMapTokens.push_back(consumes<std::map<unsigned int, int> >(ps->getParameter<edm::InputTag>("yMultiplicityMap")));
    m_xLabels.push_back(ps->getParameter<std::string>("xDetLabel"));
    m_yLabels.push_back(ps->getParameter<std::string>("yDetLabel"));
    m_xSelections.push_back(ps->getParameter<unsigned int>("xDetSelection"));
    m_ySelections.push_back(ps->getParameter<unsigned int>("yDetSelection"));

    m_mchms.push_back(new MultiplicityCorrelatorHistogramMaker(*ps, consumesCollector()));

  }

}


MultiplicityCorrelator::~MultiplicityCorrelator()
{

  for(unsigned int i=0;i<m_mchms.size();++i) {
    delete m_mchms[i];
  }

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
MultiplicityCorrelator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  for(unsigned int i=0;i<m_mchms.size();++i) {
    Handle<std::map<unsigned int, int> > xMults;
    iEvent.getByToken(m_xMultiplicityMapTokens[i],xMults);
    Handle<std::map<unsigned int, int> > yMults;
    iEvent.getByToken(m_yMultiplicityMapTokens[i],yMults);

    // check if the selection exists

    std::map<unsigned int, int>::const_iterator xmult = xMults->find(m_xSelections[i]);
    std::map<unsigned int, int>::const_iterator ymult = yMults->find(m_ySelections[i]);

    if(xmult!=xMults->end() && ymult!=yMults->end()) {


      m_mchms[i]->fill(iEvent,xmult->second,ymult->second);

    }
    else {
      edm::LogWarning("DetSelectionNotFound") << " DetSelection "
					      << m_xSelections[i] << " "
					      << m_ySelections[i] << " not found";
    }
  }

}


// ------------ method called once each job just before starting event loop  ------------
void
MultiplicityCorrelator::beginJob()
{

}

void
MultiplicityCorrelator::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {

  for(unsigned int i=0;i<m_mchms.size();++i) {
    m_mchms[i]->beginRun(iRun);
  }
}
// ------------ method called once each job just after ending the event loop  ------------
void
MultiplicityCorrelator::endJob() {
}
//define this as a plug-in
DEFINE_FWK_MODULE(MultiplicityCorrelator);
