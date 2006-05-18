#ifndef GlobalTrackFinder_GlobalMuonSeedCleaner_H
#define GlobalTrackFinder_GlobalMuonSeedCleaner_H

/**  \class MuonSeedCleaner
 * 
 *   Muon seed cleaner
 *
 *
 *   $Date: $
 *   $Revision: $
 *
 *   \author   N. Neumeister     - Purdue University
 **/

//---------------
// C++ Headers --
//---------------

#include <vector>

class TrajectorySeed;
class Trajectory;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class GlobalMuonSeedCleaner {

 public:

   /// constructor
   GlobalMuonSeedCleaner() {}
   
   /// destructor
   ~GlobalMuonSeedCleaner() {}

   ///  clean
   static bool clean(const TrajectorySeed&,std::vector<Trajectory>&);

};

#endif


