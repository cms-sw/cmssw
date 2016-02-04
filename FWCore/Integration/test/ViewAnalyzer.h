#ifndef Integration_ViewAnalyzer_h
#define Integration_ViewAnalyzer_h

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edmtest {
  
  class ViewAnalyzer : public edm::EDAnalyzer {
  public:
    explicit ViewAnalyzer(edm::ParameterSet const& /* no parameters*/);
    virtual ~ViewAnalyzer();
    virtual void analyze(edm::Event const& e,
			 edm::EventSetup const& /* unused */ );

    template <typename P, typename V>
    void testProduct(edm::Event const& e,
		     std::string const& moduleLabel) const;

    void testDSVProduct(edm::Event const& e,
			std::string const& moduleLabel) const;

    void testProductWithBaseClass(edm::Event const& e,
 			          std::string const& moduleLabel) const;

    void testRefVector(edm::Event const& e,
		       std::string const& moduleLabel) const;

    void testRefToBaseVector(edm::Event const& e,
			     std::string const& moduleLabel) const;

    void testPtrVector(edm::Event const& e,
			     std::string const& moduleLabel) const;
  };
  
}

#endif
