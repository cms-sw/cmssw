#ifndef GenJetParticleSelector_h
#define GenJetParticleSelector_h
/* \class GenJetParticleSelector
 *
 * \author Luca Lista, INFN
 *
 * \version $Id$
 *
 */
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
#include <set>

namespace edm { class ParameterSet; class EventSetup; class Event; }
namespace reco { class Candidate; }

class GenJetParticleSelector {
public:
  GenJetParticleSelector(const edm::ParameterSet&);
  bool operator()(const reco::Candidate&);
  void init(const edm::EventSetup&);
private:
  typedef std::vector<PdtEntry> vpdt;
  bool stableOnly_;
  bool partons_; 
  vpdt pdtList_;
  bool bInclude_;
  std::set<int> pIds_;
};

#include "PhysicsTools/UtilAlgos/interface/EventSetupInitTrait.h"

namespace reco {
  namespace modules {
    struct GenJetParticleSelectorEventSetupInit {
      static void init(GenJetParticleSelector & selector, 
		       const edm::Event & evt,
		       const edm::EventSetup& es) { 
	selector.init(es);
      }
    };

    template<>
    struct EventSetupInit<GenJetParticleSelectorEventSetupInit> {
      typedef GenJetParticleSelectorEventSetupInit type;
    };
  }
}

#endif
