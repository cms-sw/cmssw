#ifndef TREEUTILITY_HH_
#define TREEUTILITY_HH_
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"
#include "RecoParticleFlow/PFClusterTools/interface/ParticleDeposit.h"

#include <boost/shared_ptr.hpp>
#include <TFile.h>
#include <vector>

namespace pftools {
/**
 * 
 * \class TreeUtility 
\brief Utility class to create particles and detector elements from a Root file

\todo Remove recreateFromRootFile(TFile& file) as this is only useful for testing purposes!

\author Jamie Ballin
\date   April 2008
*/
class TreeUtility {
public:
	TreeUtility();
	virtual ~TreeUtility();

	void recreateFromRootFile(TFile& file,
			std::vector<DetectorElementPtr >& elements,
			std::vector<ParticleDepositPtr >& toBeFilled);
	
	void recreateFromRootFile(TFile& file);
	
	std::vector<ParticleDepositPtr> extractParticles(TFile& f);
};
}
#endif /*TREEUTILITY_HH_*/
