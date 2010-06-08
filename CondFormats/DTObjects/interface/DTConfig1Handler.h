#ifndef DTConfig1Handler_H
#define DTConfig1Handler_H
/** \class DTConfig1Handler
 *
 *  Description:
 *       Class to hold configuration identifier for chambers
 *
 *  TEMPORARY TOOL TO HANDLE CONFIGURATIONS
 *  TO BE REMOVED IN FUTURE RELEASES
 *
 *
 *  $Date: 2009/05/28 17:18:55 $
 *  $Revision: 1.1.4.2 $
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
//#include "CondCore/DBCommon/interface/TypedRef.h"
namespace cond{
  template <typename T> class TypedRef;
}
class DTDB1Session;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTConfig1Handler {

 public:

  /** Operations
   */
  /// create static object
  static DTConfig1Handler* create( DTDB1Session* session,
                                  const std::string& token );
  static void remove( const DTDB1Session* session );
  static void remove( const DTConfig1Handler* handler );

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
  std::string clone( DTDB1Session* newSession,
                     const std::string& newToken,
                     const std::string& objContainer,
                     const std::string& refContainer );

  static int maxBrickNumber;
  static int maxStringNumber;
  static int maxByteNumber;

 private:

  /** Constructor
   */
  DTConfig1Handler( DTDB1Session* session, const std::string& token );
  DTConfig1Handler( const DTConfig1Handler& x );
  const DTConfig1Handler& operator=( const DTConfig1Handler& x );

  /** Destructor
   */
  virtual ~DTConfig1Handler();

  std::string clone( DTDB1Session* newSession,
                     const std::string& objContainer,
                     const std::string& refContainer );

  static std::string compToken( const std::string& token, int id );
  static int         compToken( const std::string& token );

  DTDB1Session* dbSession;
  std::string objToken;
  cond::TypedRef<DTConfigList>* refSet;
  std::map<int,cond::TypedRef<DTConfigData>*> refMap;
  int cachedBrickNumber;
  int cachedStringNumber;
  int cachedByteNumber;

//  typedef std::map<unsigned int,DTConfig1Handler*> handler_map;
  typedef std::map<const DTDB1Session*,DTConfig1Handler*> handler_map;
  typedef handler_map::const_iterator c_map_iter;
  typedef handler_map::iterator         map_iter;
  static handler_map handlerMap;

};


#endif // DTConfig1Handler_H






