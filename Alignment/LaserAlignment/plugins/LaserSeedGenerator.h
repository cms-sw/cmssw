#ifndef LaserAlignment_LaserSeedGenerator_h
#define LaserAlignment_LaserSeedGenerator_h

/** \class LaserSeedGenerator
*  Seeds for Tracking of Laser Beams
	*
	*  $Date: 2007/03/18 19:00:20 $
	*  $Revision: 1.3 $
	*  \author Maarten Thomas
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDProducer.h"



#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/LaserAlignment/interface/SeedGeneratorForLaserBeams.h"

//
// class decleration
//

class LaserSeedGenerator : public edm::EDProducer {
public:
		/// constructor
	explicit LaserSeedGenerator(const edm::ParameterSet&);
			/// destructor
	~LaserSeedGenerator();

private:
		/// begin job
	virtual void beginJob(const edm::EventSetup&) ;
			/// produce seeds
	virtual void produce(edm::Event&, const edm::EventSetup&);
			/// end job
	virtual void endJob() ;

			// ----------member data ---------------------------
	edm::ParameterSet conf_;
	SeedGeneratorForLaserBeams laser_seed;
};

#endif
