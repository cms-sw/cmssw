#include <vector>

#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"

using namespace reco;
using namespace PhysicsTools;

// static cache
GenericMVAComputer::TaggingVariableMapping GenericMVAComputer::mapping;

GenericMVAComputer::TaggingVariableMapping::TaggingVariableMapping()
{
	for(unsigned int i = 0; i < btau::lastTaggingVariable; i++) {
		const char *name = TaggingVariableTokens[i];
		AtomicId id(name);

		taggingVarToAtomicId.push_back(id);
	}
}

// explicit instantiation the common case of reco::TaggingVariableList
template
double GenericMVAComputer::eval<reco::TaggingVariableList::const_iterator>(
				reco::TaggingVariableList::const_iterator,
				reco::TaggingVariableList::const_iterator) const;

template
double GenericMVAComputer::eval<reco::TaggingVariableList>(
				const reco::TaggingVariableList&) const;

template
class GenericMVAComputer::TaggingVariableIterator<
				reco::TaggingVariableList::const_iterator>;

