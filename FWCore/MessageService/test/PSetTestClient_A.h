#ifndef FWCore_MessageService_test_PSetTestClient_A_h
#define FWCore_MessageService_test_PSetTestClient_A_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class PSetTestClient_A
  : public edm::EDAnalyzer
{
public:
  explicit
    PSetTestClient_A( edm::ParameterSet const & p);
  ~PSetTestClient_A() override {}
  
  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;
private:
  edm::ParameterSet a;
  edm::ParameterSet b;
  int xa;
  int xb;
};

}  // namespace edmtest


#endif  // FWCore_MessageService_test_PSetTestClient_A_h
