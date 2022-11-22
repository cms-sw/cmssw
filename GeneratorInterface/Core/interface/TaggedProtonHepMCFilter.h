#ifndef __TAGGEDPROTONHEPMCFILTER__
#define __TAGGEDPROTONHEPMCFILTER__

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/Core/interface/BaseHepMCFilter.h"

class TaggedProtonHepMCFilter : public BaseHepMCFilter {
private:
  const int proton_PDGID_ = 2212;
  const int neutron_PDGID_ = 2112;
  float OneOverbeamEnergy_;
  double xiMin_ = 0.02;
  double xiMax_ = 0.2;
  double comEnergy_ = 13600.0;
  int nProtons_ = 2;

public:
  explicit TaggedProtonHepMCFilter(const edm::ParameterSet &);
  ~TaggedProtonHepMCFilter() override;

  bool filter(const HepMC::GenEvent *evt) override;
};

#endif
