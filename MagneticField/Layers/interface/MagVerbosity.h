#ifndef MagDebug_H
#define MagDebug_H

/*
 *  Verbosity flag to be shared between several classes
 *
 *  $Date: 2004/09/09 14:20:24 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - INFN Torino
 */

/* #include "Utilities/UI/interface/Verbosity.h" */

namespace {
/*   static const UserVerbosity bldVerb("MagGeomBuilder","silent","MagneticField"); */
/*   static const UserVerbosity verbose("MagGeometry","silent","MagneticField"); */

  class UserVerbosity {
  public:
    UserVerbosity() {debugOut = false;}
    bool debugOut;

  };
  static const UserVerbosity bldVerb;
  static const UserVerbosity verbose;
};
#endif

