// -*- C++ -*-
//
// Package:    MTDTopologyAnalyzer
// Class:      MTDTopologyAnalyzer
// 
/**\class MTDTopologyAnalyzer MTDTopologyAnalyzer.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//

// system include files
#include <memory>
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
//
// class decleration
//


// #define PRINT(X) edm::LogInfo(X)
#define PRINT(X) std::cout << X << ": "

class MTDTopologyAnalyzer : public edm::one::EDAnalyzer<>
{
public:
      explicit MTDTopologyAnalyzer( const edm::ParameterSet& );
      ~MTDTopologyAnalyzer() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
};

MTDTopologyAnalyzer::MTDTopologyAnalyzer( const edm::ParameterSet& iConfig )
{}

MTDTopologyAnalyzer::~MTDTopologyAnalyzer()
{}

// ------------ method called to produce the data  ------------
void
MTDTopologyAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  PRINT("MTDTopologyAnalyzer")<< "Here I am" << std::endl;
  //
  // get the MTDTopology
  //
  edm::ESHandle<MTDTopology> mtdTopo;
  iSetup.get<MTDTopologyRcd>().get( mtdTopo );     
  PRINT("MTDTopologyAnalyzer") << "MTD topology mode = " << mtdTopo->getMTDTopologyMode() << std::endl;
    
}

//define this as a plug-in
DEFINE_FWK_MODULE(MTDTopologyAnalyzer);
