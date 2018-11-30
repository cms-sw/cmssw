// system include files
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class MTDTopologyAnalyzer : public edm::one::EDAnalyzer<>
{
public:
  explicit MTDTopologyAnalyzer( const edm::ParameterSet& );
  ~MTDTopologyAnalyzer() override = default;
  
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

};

MTDTopologyAnalyzer::MTDTopologyAnalyzer( const edm::ParameterSet& iConfig )
{}

// ------------ method called to produce the data  ------------
void
MTDTopologyAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  edm::ESHandle<MTDTopology> mtdTopo;
  iSetup.get<MTDTopologyRcd>().get( mtdTopo );     
  edm::LogInfo("MTDTopologyAnalyzer") << "MTD topology mode = " << mtdTopo->getMTDTopologyMode();
    
}

//define this as a plug-in
DEFINE_FWK_MODULE(MTDTopologyAnalyzer);
