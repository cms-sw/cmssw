#ifndef __TAGGEDPROTONHEPMCFILTER__
#define __TAGGEDPROTONHEPMCFILTER__

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/Core/interface/BaseHepMCFilter.h"

class TaggedProtonHepMCFilter : public BaseHepMCFilter {
private:
  const int proton_PDGID_ = 2212;
  const int neutron_PDGID_ = 2112;
  const double xiMin_;
  const double xiMax_;
  const double oneOverbeamEnergy_;
  const int nProtons_;

public:
  explicit TaggedProtonHepMCFilter(const edm::ParameterSet &);
  ~TaggedProtonHepMCFilter() override = default;

  bool filter(const HepMC::GenEvent *evt) override;
};

#endif
