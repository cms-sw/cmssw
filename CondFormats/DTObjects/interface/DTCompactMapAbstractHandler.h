#ifndef DTCompactMapAbstractHandler_H
#define DTCompactMapAbstractHandler_H
/** \class DTCompactMapAbstractHandler
 *
 *  Description:
 *       Abstract class to hold configuration identifier for chambers
 *
 *  $Date: 2011/02/08 15:48:27 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTReadOutMapping;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTCompactMapAbstractHandler {

 public:

  /** Constructor
   */
//  DTCompactMapAbstractHandler();

  /** Destructor
   */
  virtual ~DTCompactMapAbstractHandler();

  /** Operations
   */
  /// get static object
  static DTCompactMapAbstractHandler* getInstance();

  /// expand compact map
  virtual DTReadOutMapping* expandMap( const DTReadOutMapping& compMap );

 protected:

  /** Constructor
   */
  DTCompactMapAbstractHandler();
  static DTCompactMapAbstractHandler* instance;

 private:

  /** Constructor
   */
  DTCompactMapAbstractHandler( const DTCompactMapAbstractHandler& x );
  const DTCompactMapAbstractHandler& operator=( const DTCompactMapAbstractHandler& x );

};


#endif // DTCompactMapAbstractHandler_H

