#ifndef DTHVCheckWithHysteresis_H
#define DTHVCheckWithHysteresis_H
/** \class DTHVCheckWithHysteresis
 *
 *  Description: 
 *
 *
 *  $Date: 2010/04/02 14:10:27 $
 *  $Revision: 1.2 $
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
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace cond { namespace service {
class DTHVCheckWithHysteresis: public DTHVAbstractCheck {

 public:

  /** Constructor
   */
  DTHVCheckWithHysteresis();
  DTHVCheckWithHysteresis( const edm::ParameterSet & iConfig, 
                 edm::ActivityRegistry & iAR );

  /** Destructor
   */
  ~DTHVCheckWithHysteresis() override;

  /** Operations
   */
  /// check HV status
//  virtual int checkCurrentStatus( 
  DTHVAbstractCheck::flag checkCurrentStatus( 
          int rawId, int type,
          float valueA, float valueC, float valueS,
          const std::map<int,timedMeasurement>& snapshotValues,
          const std::map<int,int>& aliasMap,
          const std::map<int,int>& layerMap ) override;
  void setStatus(
               int rawId,
               int flagA, int flagC, int flagS,
               const std::map<int,timedMeasurement>& snapshotValues,
               const std::map<int,int>& aliasMap,
               const std::map<int,int>& layerMap ) override;

 private:

  float* minHVl;
  float* minHVh;
  float* maxHV;
  float maxCurrent;
  std::map<int,int>* oldStatusA;
  std::map<int,int>* oldStatusC;
  std::map<int,int>* oldStatusS;

};

} }

#endif // DTHVCheckWithHysteresis_H






