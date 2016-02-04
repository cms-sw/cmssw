#ifndef PARTICLEFILTRATIONDECISION_H_
#define PARTICLEFILTRATIONDECISION_H_
#include <string>
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace pftools {

/**
 * @class ParticleFiltrationDecision
 * @brief Articulates the decision of the ParticleFilter in RecoParticleFlow/PFAnalyses.
 *
 * Despite its generic name, it is currently only suitable for testbeam analysis and
 * particle gun use. To be reworked for collisions.
 *
 * @author 	Jamie Ballin
 * @since 	CMSSW 31X
 * @date 	Added July 2009
 *
 */
class ParticleFiltrationDecision {
public:
	ParticleFiltrationDecision() {};
	virtual ~ParticleFiltrationDecision() {};

	/* Bit field to contain user-defined vetos */
	char vetosPassed_;

	/*User-defined string representing who made this */
	std::string filtrationProvenance_;

	enum TestbeamParticle {
		PION, PROTON_KAON, PROTON, KAON, ELECTRON, MUON, NOISE, OTHER
	};

	/* This event contains a clean... */
	TestbeamParticle type_;

};

//Usual framework & EDM incantations
typedef std::vector<pftools::ParticleFiltrationDecision>
		ParticleFiltrationDecisionCollection;

typedef edm::Ref<ParticleFiltrationDecisionCollection>
		ParticleFiltrationDecisionRef;
typedef edm::RefProd<ParticleFiltrationDecisionCollection>
		ParticleFiltrationDecisionRefProd;
typedef edm::RefVector<ParticleFiltrationDecisionCollection>
		ParticleFiltrationDecisionRefVector;
typedef ParticleFiltrationDecisionRefVector::iterator
		particleFiltrationDecision_iterator;
typedef edm::RefToBase<pftools::ParticleFiltrationDecision>
		ParticleFiltrationDecisionBaseRef;

}

#endif /* PARTICLEFILTRATIONDECISION_H_ */
