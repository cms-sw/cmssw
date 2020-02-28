#ifndef GUARD_cscmap1_H
#define GUARD_cscmap1_H

#include "CondFormats/CSCObjects/interface/CSCMapItem.h"
#include <iostream>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <string>

class cscmap1 {
private:
  oracle::occi::Environment *env;
  oracle::occi::Connection *con;

public:
  /**
   * Constructor for cscmap1
   */
  cscmap1() noexcept(false);
  /**
   * Destructor for cscmap1
   */
  ~cscmap1() noexcept(false);

  /* 'chamberid' is a decimal chamber identifier like 122090 */
  void chamber(int chamberid, CSCMapItem::MapItem *item);

  /* 'crate' is either crateid (1-60) or crate logical number,
   corresponding to position of crate: VME+1/11 -> 111
                                       VME-3/04 -> 234
     'dmb' : 1-5,7-10                                        */
  void cratedmb(int crate, int dmb, CSCMapItem::MapItem *item);

  /* 'rui' is a rui (ddu) number: 1-36
     'ddu_input' : 0-14                 */
  void ruiddu(int rui, int ddu_input, CSCMapItem::MapItem *item);

};  // end of class cscmap1
#endif
