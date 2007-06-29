
#include "FWCore/Modules/src/Prescaler.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace std;

namespace edm
{
  Prescaler::Prescaler(edm::ParameterSet const& ps):
    count_(),
    n_(ps.getParameter<int>("prescaleFactor"))
  {
  }
    
  Prescaler::~Prescaler()
  {
  }

  bool Prescaler::filter(edm::Event & e,edm::EventSetup const&)
  {
    ++count_;
    return count_%n_ ==0 ? true : false;
  }

  void Prescaler::endJob()
  {
  }
}
