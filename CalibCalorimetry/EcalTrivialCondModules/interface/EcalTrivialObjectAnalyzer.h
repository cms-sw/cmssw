//
// Created: 2 Mar 2006
//          Shahram Rahatlou, University of Rome & INFN
//
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

  class EcalTrivialObjectAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  EcalTrivialObjectAnalyzer(edm::ParameterSet const& p) 
    { }
    explicit  EcalTrivialObjectAnalyzer(int i) 
    { }
    ~ EcalTrivialObjectAnalyzer() override { }
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  private:
  };
