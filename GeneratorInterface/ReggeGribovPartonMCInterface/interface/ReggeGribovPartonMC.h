#ifndef REGGEGRIBOVPARTONMC_H
#define REGGEGRIBOVPARTONMC_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "CLHEP/Random/RandomEngine.h"

#include <map>
#include <string>
#include <vector>
#include <cmath>

namespace HepMC {
  class GenEvent;
  class GenParticle;
  class GenVertex;
}  // namespace HepMC

namespace gen {
  class ReggeGribovPartonMCHadronizer : public BaseHadronizer {
  public:
    ReggeGribovPartonMCHadronizer(const edm::ParameterSet &);
    ~ReggeGribovPartonMCHadronizer() override;
  };

}  // namespace gen

#endif  //ifndef REGGEGRIBOVPARTONMC_H
