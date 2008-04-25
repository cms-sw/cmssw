#ifndef TREEUTILITY_HH_
#define TREEUTILITY_HH_
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.hh"
#include "RecoParticleFlow/PFClusterTools/interface/ParticleDeposit.hh"
#include "TFile.h"
#include <vector>
namespace minimiser {
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
