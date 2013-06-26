#ifndef DTConfigAbstractHandler_H
#define DTConfigAbstractHandler_H
/** \class DTConfigAbstractHandler
 *
 *  Description:
 *       Abstract class to hold configuration identifier for chambers
 *
 *  $Date: 2010/05/14 11:42:55 $
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

class DTKeyedConfig;
class DTKeyedConfigListRcd;
namespace edm{
  class EventSetup;
}

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTConfigAbstractHandler {

 public:

  /** Constructor
   */
//  DTConfigAbstractHandler();

  /** Destructor
   */
  virtual ~DTConfigAbstractHandler();

  /** Operations
   */
  /// get static object
  static DTConfigAbstractHandler* getInstance();

  /// get content
  virtual int get( const edm::EventSetup& context,
                   int cfgId, const DTKeyedConfig*& obj );
  virtual int get( const DTKeyedConfigListRcd& keyRecord,
                   int cfgId, const DTKeyedConfig*& obj );
  virtual void getData( const edm::EventSetup& context,
                        int cfgId, std::vector<std::string>& list );
  virtual void getData( const DTKeyedConfigListRcd& keyRecord,
                        int cfgId, std::vector<std::string>& list );

  /// purge db copy
  virtual void purge();

 protected:

  /** Constructor
   */
  DTConfigAbstractHandler();
  static DTConfigAbstractHandler* instance;

 private:

  /** Constructor
   */
  DTConfigAbstractHandler( const DTConfigAbstractHandler& x );
  const DTConfigAbstractHandler& operator=( const DTConfigAbstractHandler& x );

};


#endif // DTConfigAbstractHandler_H

