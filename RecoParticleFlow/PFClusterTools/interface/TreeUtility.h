#ifndef TREEUTILITY_HH_
#define TREEUTILITY_HH_
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"
#include "RecoParticleFlow/PFClusterTools/interface/ParticleDeposit.h"
#include "DataFormats/ParticleFlowReco/interface/Calibratable.h"
#include "DataFormats/ParticleFlowReco/interface/CalibrationProvenance.h"

#include <boost/shared_ptr.hpp>
#include <TFile.h>
#include <TChain.h>
#include <vector>
#include <string>
#include <map>

namespace pftools {
/**
 *
 * \class TreeUtility
 \brief Utility class to create particles and detector elements from a Root file

 \todo Remove recreateFromRootFile(TFile& file) as this is only useful for testing purposes!

 \author Jamie Ballin
 \date   April 2008
 */
typedef boost::shared_ptr<Calibratable> CalibratablePtr;
class TreeUtility {
public:

	TreeUtility();
	virtual ~TreeUtility();

	unsigned getParticleDepositsDirectly(TChain& sourceChain,
			std::vector<ParticleDepositPtr>& toBeFilled,
			CalibrationTarget target, DetectorElementPtr offset,
			DetectorElementPtr ecal, DetectorElementPtr hcal, bool includeOffset = false);

	unsigned getCalibratablesFromRootFile(TChain& tree,
			std::vector<Calibratable>& toBeFilled);

	unsigned convertCalibratablesToParticleDeposits(
			const std::vector<Calibratable>& input,
			std::vector<ParticleDepositPtr>& toBeFilled,
			CalibrationTarget target, DetectorElementPtr offset,
			DetectorElementPtr ecal, DetectorElementPtr hcal, bool includeOffset = false);

	void dumpCaloDataToCSV(TChain& chain, std::string csvFilename, double range, bool gaus = false);


private:
	std::map<std::string, unsigned> vetos_;

};
}
#endif /*TREEUTILITY_HH_*/
