#ifndef DTConfigHandler_H
#define DTConfigHandler_H
/** \class DTConfigHandler
 *
 *  Description:
 *       Class to hold configuration identifier for chambers
 *
 *  $Date: 2007/12/07 15:12:15 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "CondFormats/DTObjects/interface/DTConfigList.h"
#include "CondFormats/DTObjects/interface/DTConfigData.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
class DTDBSession;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTConfigHandler {

 public:

  /** Operations
   */
  /// create static object
  static DTConfigHandler* create( DTDBSession* session,
                                  const std::string& token );
  static void remove( const DTDBSession* session );
  static void remove( const DTConfigHandler* handler );

  /// get content
  const DTConfigList* getContainer();
  int get( int cfgId, DTConfigData*& obj );

  void getData( int cfgId, std::vector<const std::string*>& list );

  int set( int cfgId, const std::string& token );
  /// Access methods to data
  typedef std::map< int, cond::TypedRef<DTConfigData>* >::const_iterator
                                                          const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

  /// purge db copy
  void purge();

  /// copy to other DB
  std::string clone( DTDBSession* newSession,
                     const std::string& newToken,
                     const std::string& objContainer,
                     const std::string& refContainer );

  static int maxBrickNumber;
  static int maxStringNumber;
  static int maxByteNumber;

 private:

  /** Constructor
   */
  DTConfigHandler( DTDBSession* session, const std::string& token );
  DTConfigHandler( const DTConfigHandler& x );
  const DTConfigHandler& operator=( const DTConfigHandler& x );

  /** Destructor
   */
  virtual ~DTConfigHandler();

  std::string clone( DTDBSession* newSession,
                     const std::string& objContainer,
                     const std::string& refContainer );

  static std::string compToken( const std::string& token, int id );
  static int         compToken( const std::string& token );

  DTDBSession* dbSession;
  std::string objToken;
  cond::TypedRef<DTConfigList>* refSet;
  std::map<int,cond::TypedRef<DTConfigData>*> refMap;
  int cachedBrickNumber;
  int cachedStringNumber;
  int cachedByteNumber;

//  typedef std::map<unsigned int,DTConfigHandler*> handler_map;
  typedef std::map<const DTDBSession*,DTConfigHandler*> handler_map;
  typedef handler_map::const_iterator c_map_iter;
  typedef handler_map::iterator         map_iter;
  static handler_map handlerMap;

};


#endif // DTConfigHandler_H






