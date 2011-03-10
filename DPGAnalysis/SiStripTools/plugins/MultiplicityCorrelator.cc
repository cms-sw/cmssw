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
// $Id: MultiplicityCorrelator.cc,v 1.1 2010/05/04 08:33:47 venturia Exp $
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
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  virtual void endJob() ;

      // ----------member data ---------------------------

  std::vector<MultiplicityCorrelatorHistogramMaker> _mchms;

  std::vector<edm::InputTag> _xMultiplicityMaps;
  std::vector<edm::InputTag> _yMultiplicityMaps;
  std::vector<std::string> _xLabels;
  std::vector<std::string> _yLabels;
  std::vector<unsigned int> _xSelections;
  std::vector<unsigned int> _ySelections;


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
  _mchms(),
  _xMultiplicityMaps(),_yMultiplicityMaps(),
  _xLabels(),_yLabels(), _xSelections(),_ySelections()
{
   //now do what ever initialization is needed

  std::vector<edm::ParameterSet> correlationConfigs = 
    iConfig.getParameter<std::vector<edm::ParameterSet> >("correlationConfigurations");

  for(std::vector<edm::ParameterSet>::const_iterator ps=correlationConfigs.begin();ps!=correlationConfigs.end();++ps) {

    _xMultiplicityMaps.push_back(ps->getParameter<edm::InputTag>("xMultiplicityMap"));
    _yMultiplicityMaps.push_back(ps->getParameter<edm::InputTag>("yMultiplicityMap"));
    _xLabels.push_back(ps->getParameter<std::string>("xDetLabel"));
    _yLabels.push_back(ps->getParameter<std::string>("yDetLabel"));
    _xSelections.push_back(ps->getParameter<unsigned int>("xDetSelection"));
    _ySelections.push_back(ps->getParameter<unsigned int>("yDetSelection"));

    _mchms.push_back(MultiplicityCorrelatorHistogramMaker(*ps));

  }

}


MultiplicityCorrelator::~MultiplicityCorrelator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
MultiplicityCorrelator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  for(unsigned int i=0;i<_mchms.size();++i) {
    Handle<std::map<unsigned int, int> > xMults;
    iEvent.getByLabel(_xMultiplicityMaps[i],xMults);
    Handle<std::map<unsigned int, int> > yMults;
    iEvent.getByLabel(_yMultiplicityMaps[i],yMults);

    // check if the selection exists

    std::map<unsigned int, int>::const_iterator xmult = xMults->find(_xSelections[i]);
    std::map<unsigned int, int>::const_iterator ymult = yMults->find(_ySelections[i]);

    if(xmult!=xMults->end() && ymult!=yMults->end()) {


      _mchms[i].fill(xmult->second,ymult->second);

    }
    else {
      edm::LogWarning("DetSelectionNotFound") << " DetSelection " 
					      << _xSelections[i] << " " 
					      << _ySelections[i] << " not found"; 
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


}

void
MultiplicityCorrelator::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
}
// ------------ method called once each job just after ending the event loop  ------------
void 
MultiplicityCorrelator::endJob() {
}
//define this as a plug-in
DEFINE_FWK_MODULE(MultiplicityCorrelator);
