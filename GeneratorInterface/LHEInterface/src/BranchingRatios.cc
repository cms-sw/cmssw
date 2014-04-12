#include <map>

#include <boost/shared_ptr.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"
#include "GeneratorInterface/LHEInterface/interface/BranchingRatios.h"

namespace lhef {

void BranchingRatios::reset()
{
	cache.clear();
}

void BranchingRatios::set(int pdgId, bool both, double value)
{
	cache[pdgId] = value;
	if (both)
		cache[-pdgId] = value;
}

double BranchingRatios::getFactor(
			const Hadronisation *hadronisation,
			const boost::shared_ptr<LHEEvent> &event) const
{
	double factor = 1.0;
	const HEPEUP *hepeup = event->getHEPEUP();
	for(int i = 0; i < hepeup->NUP; i++) {
		if (hepeup->ISTUP[i] == 1)
			factor *= get(hepeup->IDUP[i], hadronisation);
	}
	return factor;
}

double BranchingRatios::get(int pdgId,
                            const Hadronisation *hadronisation) const
{
	std::map<int, double>::const_iterator pos = cache.find(pdgId);
	if (pos == cache.end()) {
		double val = hadronisation->totalBranchingRatio(pdgId);
		if (val <= 0.0)
			val = 1.0;
		if (val < 0.99)
			edm::LogInfo("Generator|LHEInterface")
				<< "Particle with pdgId " << pdgId
				<< " only partly decayed in hadronizer, "
				   "reducing respective event cross-section "
				   "contribution with a factor of "
				<< val << "." << std::endl;

		cache.insert(std::make_pair(pdgId, val));
		return val;
	} else
		return pos->second;
}

} // namespace lhef
