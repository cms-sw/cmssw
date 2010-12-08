#include "GeneratorInterface/RivetInterface/interface/RivetAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/Handle.h"

#include "Rivet/AnalysisHandler.hh"
#include "Rivet/Analysis.hh"

#include <string>
#include <vector>
#include <iostream>

using namespace Rivet;
using namespace edm;

RivetAnalyzer::RivetAnalyzer(const edm::ParameterSet& pset) : 
//_analysis(0)
_analysisHandler("RivetAnalyzer"),
_isFirstEvent(true)
{
  //retrive the analysis name from paarmeter set
  std::vector<std::string> analysisNames = pset.getParameter<std::vector<std::string> >("AnalysisNames");
  
  _hepmcCollection = pset.getParameter<edm::InputTag>("HepMCCollection");

  //get the analyses
  _analysisHandler.addAnalyses(analysisNames);

  //go through the analyses and check those that need the cross section
  const std::set< AnaHandle, AnaHandleLess > & analyses = _analysisHandler.analyses();
  //std::cout << "BUILT " << analyses.size() << " ANALYSES" << std::endl;

  std::set< AnaHandle, AnaHandleLess >::const_iterator ibeg = analyses.begin();
  std::set< AnaHandle, AnaHandleLess >::const_iterator iend = analyses.end();
  std::set< AnaHandle, AnaHandleLess >::const_iterator iana; 
  bool got_xsec = false;
  double xsection = -1.;
  for (iana = ibeg; iana != iend; ++iana){
    if (!got_xsec){
      if ((*iana)->needsCrossSection ()){
        xsection = pset.getParameter<double>("CrossSection");   
      }
      got_xsec = true;
    }
    //std::cout << "Setting xsection for analysis " << (*iana)->name() << std::endl;
    (*iana)->setCrossSection(xsection);  
  }


  //if the analysis requires the cross section take it from the configuration
  //in such a case, the number has to be in the configuration
  //if (_analysis->needsCrossSection ()){
  //  double xsection = pset.getParameter<double>("CrossSection");
  //  _analysis->setCrossSection(xsection);
  //}
}

RivetAnalyzer::~RivetAnalyzer(){
}

void RivetAnalyzer::beginJob(){
}

void RivetAnalyzer::beginRun(const edm::Run& iRun,const edm::EventSetup& iSetup){
  return;
}

void RivetAnalyzer::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup){
  //get the hepmc product from the event
  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(_hepmcCollection, evt);

  // get HepMC GenEvent
  const HepMC::GenEvent *myGenEvent = evt->GetEvent();

  //aaply the beams initialization on the first event
  if (_isFirstEvent){
    _analysisHandler.init(*myGenEvent);
    _isFirstEvent = false;
  }

  //run the analysis
  _analysisHandler.analyze(*myGenEvent);
}


void RivetAnalyzer::endRun(const edm::Run& iRun,const edm::EventSetup& iSetup){
  return;
}

void RivetAnalyzer::endJob(){
  _analysisHandler.finalize();   
  _analysisHandler.writeData("Rivet.aida");
}

DEFINE_FWK_MODULE(RivetAnalyzer);
