#ifndef DQM_SiStripCommon_UpdateTProfile_H
#define DQM_SiStripCommon_UpdateTProfile_H

#include "boost/cstdint.hpp"
#include "TProfile.h"

/** */
class UpdateTProfile {
  
 public:

  UpdateTProfile();
  ~UpdateTProfile();
  
  static void setBinContent( TProfile* const profile,
			     const uint32_t& bin, 
			     const double& entries, 
			     const double& mean,
			     const double& spread );
  
  static void setBinContents( TProfile* const profile,
			      const uint32_t& bin, 
			      const double& num_of_entries, 
			      const double& sum_of_contents,
			      const double& sum_of_squares );
  
};

#endif // DQM_SiStripCommon_UpdateTProfile_H

