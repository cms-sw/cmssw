#ifndef ECALCOMMON_HH
#define ECALCOMMON_HH

#include <stdexcept>

/**
 *  General-purpose detector related functions
 */
class EcalCommon {

  /******************\
  -  public methods  -
  \******************/

 public:

  /**
   *  Convert a supermodule crystal number to a trigger tower number
   */
  static int crystalToTriggerTower( int xtal )
    throw(std::runtime_error);

};

#endif
