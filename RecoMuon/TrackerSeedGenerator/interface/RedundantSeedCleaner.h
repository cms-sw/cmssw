#ifndef RecoMuon_TrackerSeedGenerator_RedundantSeedCleaner_H
#define  RecoMuon_TrackerSeedGenerator_RedundantSeedCleaner_H

/** \class RedundantSeedCleaner
 * Description:
 * RedundantSeedCleaner (TrackerSeedGenerator) duplicate removal from triplets pairs pixel seeds .
 *
 * \author Alessandro Grelli, Jean-Roch Vlimant
*/

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------


class MuonSeedCombiner;

class RedundantSeedCleaner{
 public:
  ///constructor
  RedundantSeedCleaner(){

  }
  ///destructor
  ~RedundantSeedCleaner(){
   }
 /// clean
 void clean(const std::vector<TrajectorySeed > &,std::vector<TrajectorySeed > &);
 /// collection definition
 void define(std::vector<TrajectorySeed> &);

private:

  std::vector<TrajectorySeed> seedTriplets;

  std::string theCategory; 
};


#endif
