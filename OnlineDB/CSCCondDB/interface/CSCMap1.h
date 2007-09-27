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

}; // end of class cscmap1
#endif
