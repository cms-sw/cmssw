#ifndef DTHVAbstractCheck_H
#define DTHVAbstractCheck_H
/** \class DTHVAbstractCheck
 *
 *  Description: 
 *
 *
 *  $Date: 2010/09/14 13:54:04 $
 *  $Revision: 1.3 $
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
  typedef std::pair<long long int,float> timedMeasurement;
  struct flag { int a; int c; int s; };
  static bool chkFlag( const DTHVAbstractCheck::flag& f );
  static bool compare( const DTHVAbstractCheck::flag& fl,
                       const DTHVAbstractCheck::flag& fr );
  virtual DTHVAbstractCheck::flag checkCurrentStatus( 
               int rawId, int type,
               float valueA, float valueC, float valueS,
               const std::map<int,timedMeasurement>& snapshotValues,
               const std::map<int,int>& aliasMap,
               const std::map<int,int>& layerMap ) = 0;
  virtual void setValue(
               int rawId, int type,
               float valueA, float valueC, float valueS,
               const std::map<int,timedMeasurement>& snapshotValues,
               const std::map<int,int>& aliasMap,
               const std::map<int,int>& layerMap );
  virtual void setStatus(
               int rawId,
               int flagA, int flagC, int flagS,
               const std::map<int,timedMeasurement>& snapshotValues,
               const std::map<int,int>& aliasMap,
               const std::map<int,int>& layerMap );

 protected:

  static DTHVAbstractCheck* instance;

 private:

};


#endif // DTHVAbstractCheck_H






