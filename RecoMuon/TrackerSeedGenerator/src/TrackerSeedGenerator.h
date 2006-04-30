#ifndef RecoMuon_TrackerSeedGenerator_H
#define RecoMuon_TrackerSeedGenerator_H

/** \class MuonSeedGenerator
 *  Generate seed from muon trajectory.
 *
 *  $Date: $
 *  $Revision: $
 *  \author Chang Liu - Purdue University
 */

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

class Trajectory;
class Propagator;
class MagneticField;

namespace edm {class ParameterSet; class Event; class EventSetup;}

class TrackerSeedGenerator {

  public:
    /// constructor
    TrackerSeedGenerator(const MagneticField *field);
    /// destructor
    virtual ~TrackerSeedGenerator();
    /// create seeds from muon trajectory
    void findSeeds(const Trajectory& muon, const edm::Event& , const edm::EventSetup&, TrajectorySeedCollection& output) const;  

  private:

    int theMaxSeeds;
    const MagneticField * theField;

};

#endif

