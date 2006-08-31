// File: JetAnalyzer.h
// Description:  Example of Jet Analysis driver originally from Jeremy Mans, 
//               developed by Lenny Apanesevich and Anwar Bhatti for various purposes.
// Date:  31-August-2006

#include <iostream>

#include "RecoJets/JetAnalyzers/interface/JetAnalysis.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class JetAnalyzer : public edm::EDAnalyzer {
public:
  explicit JetAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  virtual void endJob();
private:
  // variables persistent across events should be declared here.
  //
  JetAnalysis analysis_;
  std::string recjets_,genjets_,recmet_,genmet_,calotowers_;
  int errCnt;
  const int errMax(){return 100;}
};
