#ifndef GUARD_cscmap_H
#define GUARD_cscmap_H

#include <iostream>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <string>

class cscmap
{
  private:

  oracle::occi::Environment *env;
  oracle::occi::Connection *con;

  public :
  /**
   * Constructor for cscmap
   */
  cscmap () throw (oracle::occi::SQLException);
  /**
   * Destructor for cscmap
   */
  ~cscmap () throw (oracle::occi::SQLException);

/*  Method 'crate0_chamber' returns for a given logical crate
|   number 'crate0' (values 0-59) and DMB number 'dmb' 
|   (1-5,7-10; for station 4 (no ring 2) DMB No 1-3)
|   chamber identifiers: 'chamber_id' is a string like 'ME+2/2/27'
|   and 'chamber_num' - a corresponding numeric identifier:
|   'ME+2/2/27' => 220122270. Digits from left to right mean:
|   2 - Muon system, 2 - CSC (1=DT,3=RPC), 0 - to separate from further
|   digits, 1 - +Endcap (2 - -Endcap), 2 - station number, 2 - ring number,
|   27 - chamber number, 0 - digit reserved for layer number (=0 for
|   the whole chamber). 'sector' - returns trigger sector number.
|   If 'chamber_num' and 'sector' return -100, it means that 'crate0'
|   is outside the permitted range; -10 means that 'dmb' is outside
|   the permitted range.
*/
  void crate0_chamber (int crate0, int dmb, std::string *chamber_id,
		       int *chamber_num, int *sector,
                       int *first_strip_index, int *strips_per_layer,
                       int *chamber_index);
/* Method 'crate_chamber' returns similar to previous information,
|   but for physical (installed) 'crate' number. By now (2005/11/16)
|   only 2 crates are installed (0 and 1).
|   -1 returned means that infor mation for non-installed crate is requested.
*/
  void crate_chamber (int crate, int dmb, std::string *chamber_id,
		       int *chamber_num, int *sector,
                       int *first_strip_index, int *strips_per_layer,
                       int *chamber_index);
/* Method 'chamber_crate' returns information for a given chamber,
|   'chamber_id', (like 'ME+2/2/27'). 'crate' - physical (installed)
|   crate number (-1 means that crate for the given chamber is not
|   installed), 'dmb' - DMB number (1-5,7-10), 'sector' - trigger sector
|   number, 'chamber_num' - numeric chamber identifier (see above),
|   'crate0' - logical crate number.
|   -100 returned means that 'chamber_id' format is incorrect.
*/
  void chamber_crate (std::string chamber_id, int *crate, int *dmb,
		      int *sector, int *chamber_num, int *crate0,
                      int *first_strip_index, int *strips_per_layer,
                      int *chamber_index);

}; // end of class cscmap
#endif
