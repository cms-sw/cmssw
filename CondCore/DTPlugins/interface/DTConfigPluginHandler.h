#ifndef DTConfigPluginHandler_H
#define DTConfigPluginHandler_H
/** \class DTConfigPluginHandler
 *
 *  Description:
 *       Class to hold configuration identifier for chambers
 *
 *  $Date: 2011/06/06 17:24:05 $
 *  $Revision: 1.3 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "CondFormats/DTObjects/interface/DTConfigAbstractHandler.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//---------------
// C++ Headers --
//---------------
#include <string>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTConfigPluginHandler: public DTConfigAbstractHandler {

 public:

  /** Destructor
   */
  virtual ~DTConfigPluginHandler();

  /** Operations
   */
  /// build static object
  static void build();

  /// get content
  virtual int get( const edm::EventSetup& context,
                   int cfgId, const DTKeyedConfig*& obj );
  virtual int get( const DTKeyedConfigListRcd& keyRecord,
                   int cfgId, const DTKeyedConfig*& obj );
  virtual void getData( const edm::EventSetup& context,
                        int cfgId, std::vector<std::string>& list );
  virtual void getData( const DTKeyedConfigListRcd& keyRecord,
                        int cfgId, std::vector<std::string>& list );

  void purge();

  static int maxBrickNumber;
  static int maxStringNumber;
  static int maxByteNumber;

 private:

  /** Constructor
   */
  DTConfigPluginHandler();
  DTConfigPluginHandler( const DTConfigPluginHandler& x );
  const DTConfigPluginHandler& operator=( const DTConfigPluginHandler& x );

  typedef std::pair<int,const DTKeyedConfig*> counted_brick;
  std::map<int,counted_brick> brickMap;
  int cachedBrickNumber;
  int cachedStringNumber;
  int cachedByteNumber;

};

#endif // DTConfigPluginHandler_H


