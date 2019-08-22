#ifndef GUARD_csccableread_H
#define GUARD_csccableread_H

#include <iostream>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <string>

class csccableread {
private:
  oracle::occi::Environment *env;
  oracle::occi::Connection *con;

public:
  /**
   * Constructor for csccableread
   */
  csccableread() noexcept(false);
  /**
   * Destructor for cscmap
   */
  ~csccableread() noexcept(false);

  void cable_read(int chamber_index,
                  std::string *chamber_label,
                  float *cfeb_length,
                  std::string *cfeb_rev,
                  float *alct_length,
                  std::string *alct_rev,
                  float *cfeb_tmb_skew_delay,
                  float *cfeb_timing_corr);

};  // end of class csccableread
#endif
