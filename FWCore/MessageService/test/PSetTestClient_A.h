#ifndef FWCore_MessageService_test_PSetTestClient_A_h
#define FWCore_MessageService_test_PSetTestClient_A_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"

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
  virtual ~PSetTestClient_A() {}
  
  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );
private:
  edm::ParameterSet a;
  edm::ParameterSet b;
  int xa;
  int xb;
};

}  // namespace edmtest


#endif  // FWCore_MessageService_test_PSetTestClient_A_h
