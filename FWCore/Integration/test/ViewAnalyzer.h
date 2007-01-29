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

    // Unfortunately DetSetVector requires a special version of testProduct
    // because there is no "==" function defined for DetSet.  Given that defining
    // defining this function in DetSet would require creating a "==" function
    // inconsistent with the existing definition of "<", I decided I would rather 
    // put some ugliness here in the test code than in DetSet that everybody uses.
    void testDSVProduct(edm::Event const& e,
			std::string const& moduleLabel) const;
  };
  
}

#endif
