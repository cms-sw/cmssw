#ifndef DTHVCheckByAbsoluteValues_H
#define DTHVCheckByAbsoluteValues_H
/** \class DTHVCheckByAbsoluteValues
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
namespace edm {
  //  class Event;
  //  class EventSetup;
  class ParameterSet;
  class ActivityRegistry;
}  // namespace edm

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace cond {
  namespace service {
    class DTHVCheckByAbsoluteValues : public DTHVAbstractCheck {
    public:
      /** Constructor
   */
      DTHVCheckByAbsoluteValues();
      DTHVCheckByAbsoluteValues(const edm::ParameterSet& iConfig, edm::ActivityRegistry& iAR);

      /** Destructor
   */
      ~DTHVCheckByAbsoluteValues() override;

      /** Operations
   */
      /// check HV status
      //  virtual int checkCurrentStatus(
      DTHVAbstractCheck::flag checkCurrentStatus(int rawId,
                                                 int type,
                                                 float valueA,
                                                 float valueC,
                                                 float valueS,
                                                 const std::map<int, timedMeasurement>& snapshotValues,
                                                 const std::map<int, int>& aliasMap,
                                                 const std::map<int, int>& layerMap) override;

    private:
      float* minHV;
      float* maxHV;
      float maxCurrent;
    };

  }  // namespace service
}  // namespace cond

#endif  // DTHVCheckByAbsoluteValues_H
