#ifndef DataFormats_MuonSeed_L3MuonTrajectorySeedCollection_H
#define DataFormats_MuonSeed_L3MuonTrajectorySeedCollection_H

/** \class L3MuonTrajectorySeedCollection
 *  No description available.
 *
 *  $Revision: 1.2 $
 *  \author J.R. Vlimant
 */

#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include <vector>

typedef std::vector<L3MuonTrajectorySeed> L3MuonTrajectorySeedCollection;
typedef edm::Ref<L3MuonTrajectorySeedCollection> L3MuonTrajectorySeedRef;

#endif
