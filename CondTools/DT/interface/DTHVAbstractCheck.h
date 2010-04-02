#ifndef DTHVAbstractCheck_H
#define DTHVAbstractCheck_H
/** \class DTHVAbstractCheck
 *
 *  Description: 
 *
 *
 *  $Date: 2010/01/18 18:59:47 $
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


//---------------
// C++ Headers --
//---------------
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTHVAbstractCheck {

 public:

  /** Constructor
   */
  DTHVAbstractCheck();

  /** Destructor
   */
  virtual ~DTHVAbstractCheck();

  /** Operations
   */
  /// check HV status
  static DTHVAbstractCheck* getInstance();

  /// check HV status
//  virtual int checkCurrentStatus( int part, int type, float value ) = 0;
  typedef std::pair<long long int,float> timedMeasurement;
  virtual int checkCurrentStatus( 
//              const std::pair<int,timedMeasurement>& entry,
              int dpId, int rawId, int type, float value,
              const std::map<int,timedMeasurement>& snapshotValues,
              const std::map<int,int>& aliasMap,
              const std::map<int,int>& layerMap ) = 0;

 protected:

  static DTHVAbstractCheck* instance;

 private:

};


#endif // DTHVAbstractCheck_H






