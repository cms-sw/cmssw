#ifndef DTCompactMapPluginHandler_H
#define DTCompactMapPluginHandler_H
/** \class DTCompactMapPluginHandler
 *
 *  Description:
 *       Class to hold configuration identifier for chambers
 *
 *  $Date: 2011/02/08 15:46:42 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "CondFormats/DTObjects/interface/DTCompactMapAbstractHandler.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTReadOutMapping;

//---------------
// C++ Headers --
//---------------
#include <iostream>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTCompactMapPluginHandler:public DTCompactMapAbstractHandler {

 public:

  /** Destructor
   */
  virtual ~DTCompactMapPluginHandler();

  /** Operations
   */
  /// build static object
  static void build();

  /// expand compact map
  virtual DTReadOutMapping* expandMap( const DTReadOutMapping& compMap );

 private:

  /** Constructor
   */
  DTCompactMapPluginHandler();
  DTCompactMapPluginHandler( const DTCompactMapPluginHandler& x );
  const DTCompactMapPluginHandler& operator=( const DTCompactMapPluginHandler& x );

};

#endif // DTCompactMapPluginHandler_H


