#ifndef Integration_ViewAnalyzer_h
#define Integration_ViewAnalyzer_h

#include <string>

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
    template <class P> void testProduct(edm::Event const& e,
					std::string const& moduleLabel) const;
  };
  
}

#endif
