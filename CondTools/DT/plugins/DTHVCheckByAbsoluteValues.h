#ifndef DTHVCheckByAbsoluteValues_H
#define DTHVCheckByAbsoluteValues_H
/** \class DTHVCheckByAbsoluteValues
 *
 *  Description: 
 *
 *
 *  $Date: 2010/01/18 18:36:32 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "CondTools/DT/interface/DTHVAbstractCheck.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace edm{
//  class Event;
//  class EventSetup;
  class ParameterSet;
  class ActivityRegistry;
}

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace cond { namespace service {
class DTHVCheckByAbsoluteValues: public DTHVAbstractCheck {

 public:

  /** Constructor
   */
  DTHVCheckByAbsoluteValues();
  DTHVCheckByAbsoluteValues( const edm::ParameterSet & iConfig, 
                 edm::ActivityRegistry & iAR );

  /** Destructor
   */
  virtual ~DTHVCheckByAbsoluteValues();

  /** Operations
   */
  /// check HV status
//  virtual int checkCurrentStatus( int part, int type, float value );
  virtual int checkCurrentStatus( 
//              const std::pair<int,timedMeasurement>& entry,
              int dpId, int rawId, int type, float value,
              const std::map<int,timedMeasurement>& snapshotValues,
              const std::map<int,int>& aliasMap,
              const std::map<int,int>& layerMap  );

 private:

  float* minHV;
  float* maxHV;
  float maxCurrent;

};

} }

#endif // DTHVCheckByAbsoluteValues_H






