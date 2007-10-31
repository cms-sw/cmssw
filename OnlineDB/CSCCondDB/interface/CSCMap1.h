#ifndef GUARD_cscmap1_H
#define GUARD_cscmap1_H

#include "CondFormats/CSCObjects/interface/CSCMapItem.h"
#include <iostream>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <string>

class cscmap1
{
  private:

  oracle::occi::Environment *env;
  oracle::occi::Connection *con;

  public :
  /**
   * Constructor for cscmap1
   */
  cscmap1 () throw (oracle::occi::SQLException);
  /**
   * Destructor for cscmap1
   */

  ~cscmap1 () throw (oracle::occi::SQLException);
  /* 'chamberid' is a decimal chamber identifier like 122090 */
  void chamber (int chamberid, CSCMapItem::MapItem *item);

  /* 'crate' is either crateid (1-60) or crate logical number,
   corresponding to position of crate: VME+1/11 -> 111
                                       VME-3/04 -> 234       */
  void cratedmb (int crate, int dmb, CSCMapItem::MapItem *item);

}; // end of class cscmap1
#endif
