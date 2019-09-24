#include <cstdint>
#ifndef DQM_SiStripCommon_UpdateTProfile_H
#define DQM_SiStripCommon_UpdateTProfile_H

class TProfile;

/** */
class UpdateTProfile {
public:
  UpdateTProfile();
  ~UpdateTProfile();

  static void setBinContents(TProfile* const profile,
                             const uint32_t& bin,
                             const double& num_of_entries,
                             const double& sum_of_contents,
                             const double& sum_of_squares);

  static void setBinContent(
      TProfile* const profile, const uint32_t& bin, const double& entries, const double& mean, const double& spread);
};

#endif  // DQM_SiStripCommon_UpdateTProfile_H
