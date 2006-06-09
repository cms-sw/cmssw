#ifndef GUARD_cscmap_H
#define GUARD_cscmap_H

#include <iostream>
#include <occi.h>
#include <string>
using namespace oracle::occi;
using namespace std;

class cscmap
{
  private:

  Environment *env;
  Connection *con;

  public :
  /**
   * Constructor for cscmap
   */
  cscmap () throw (SQLException);
  /**
   * Destructor for cscmap
   */
  ~cscmap () throw (SQLException);

  void crate0_chamber (int crate0, int dmb, string *chamber_id,
		       int *chamber_num, int *sector);
  void crate_chamber (int crate, int dmb, string *chamber_id,
		       int *chamber_num, int *sector);
  void chamber_crate (string chamber_id, int *crate, int *dmb,
		      int *sector, int *chamber_num, int *crate0);

}; // end of class cscmap
#endif
