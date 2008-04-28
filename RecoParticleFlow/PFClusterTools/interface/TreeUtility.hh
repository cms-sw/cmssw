#ifndef TREEUTILITY_HH_
#define TREEUTILITY_HH_
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.hh"
#include "RecoParticleFlow/PFClusterTools/interface/ParticleDeposit.hh"
#include "TFile.h"
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
			std::vector<DetectorElement* >& elements,
			std::vector<ParticleDeposit* >& toBeFilled);
	
	void recreateFromRootFile(TFile& file);
};
}
#endif /*TREEUTILITY_HH_*/
