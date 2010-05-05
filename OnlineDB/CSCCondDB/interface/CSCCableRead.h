#ifndef GUARD_csccableread_H
#define GUARD_csccableread_H

#include <iostream>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <string>

class csccableread
{
  private:

  oracle::occi::Environment *env;
  oracle::occi::Connection *con;

  public :
  /**
   * Constructor for csccableread
   */
  csccableread () throw (oracle::occi::SQLException);
  /**
   * Destructor for cscmap
   */
  ~csccableread () throw (oracle::occi::SQLException);

void cable_read (int chamber_index, std::string *chamber_label,
     int *cfeb_length, std::string *cfeb_rev, int *alct_length,
     std::string *alct_rev, int *cfeb_tmb_skew_delay, int *cfeb_timing_corr);

}; // end of class csccableread
#endif
