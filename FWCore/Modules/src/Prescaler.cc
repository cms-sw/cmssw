
#include "FWCore/Modules/src/Prescaler.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm
{
  Prescaler::Prescaler(edm::ParameterSet const& ps):
    count_(),
    n_(ps.getParameter<int>("prescaleFactor")),
    offset_(ps.getParameter<int>("prescaleOffset"))
  {
  }

  Prescaler::~Prescaler()
  {
  }

  bool Prescaler::filter(edm::Event & e,edm::EventSetup const&)
  {
    ++count_;
    return count_ % n_ == offset_ ? true : false;
  }

  void Prescaler::endJob()
  {
  }
}
