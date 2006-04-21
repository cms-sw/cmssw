#ifndef DataFormats_SiStripDigi_Profile_H
#define DataFormats_SiStripDigi_Profile_H

#include "TProfile.h"

class Profile {

 public:

  Profile() : prof_() {;}
  Profile(const TProfile& prof) : prof_(prof) {;}
  ~Profile() {;}

 inline const TProfile& get() const { return prof_;}
 inline void set(const TProfile& prof) {prof_ = prof;}
 inline bool operator<(const Profile& compare) const
    { return get().GetEntries() < compare.get().GetEntries(); }

private:

  TProfile prof_;

};

#endif //DataFormats_SiStripDigi_Profile_H
