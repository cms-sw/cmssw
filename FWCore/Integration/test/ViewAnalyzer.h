#ifndef Integration_ViewAnalyzer_h
#define Integration_ViewAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edmtest 
{
  
  class ViewAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit ViewAnalyzer(edm::ParameterSet const& /* no parameters*/);
    virtual ~ViewAnalyzer();
    virtual void analyze(edm::Event const& e,
			 edm::EventSetup const& /* unused */ );
  private:
    // nothing yet
  };
  
}

#endif
