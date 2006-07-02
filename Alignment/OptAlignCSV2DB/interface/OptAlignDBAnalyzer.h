#ifndef OptAlignDBAnalyzer_H
#define OptAlignDBAnalyzer_H

//#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace std;

class OptAlignDBAnalyzer : public edm::EDAnalyzer
{
  public:
    explicit  OptAlignDBAnalyzer(edm::ParameterSet const& p) 
    { }
    virtual ~OptAlignDBAnalyzer() {}
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
};

#endif

