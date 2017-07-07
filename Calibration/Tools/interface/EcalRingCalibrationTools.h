#ifndef EcalRingCalibrationTools_h
#define EcalRingCalibrationTools_h

/****************************************
 *
 *   25/09/2007 P. Meridiani (CERN)
 *   Description:
 *   Tools to ease the hanling of indices  
 *   for ECAL ring intercalibration
 *
 ***************************************/

#include <vector>
#include <mutex>
#include <atomic>
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

class DetId;
class CaloGeometry;

class EcalRingCalibrationTools 
{
 public:
  EcalRingCalibrationTools() {};
  ~EcalRingCalibrationTools() {};
  
  /// Retrieve the phi-ring index corresponding to a DetId 
  static short getRingIndex(DetId aDetId); 

  static short getModuleIndex(DetId aDetId); 
  
  /// Retrieve the DetIds in a phi-ring 
  static std::vector<DetId> getDetIdsInRing(short aRingIndex);
  static std::vector<DetId> getDetIdsInModule(short int);  
  static std::vector<DetId> getDetIdsInECAL();  

  static const short N_RING_TOTAL  = 248;
  static const short N_RING_BARREL = 170;
  static const short N_RING_ENDCAP =  78;

  static const short N_MODULES_BARREL = 144;

  static void setCaloGeometry(const CaloGeometry* geometry);

 private:
  static void initializeFromGeometry(CaloGeometry const* geometry); // needed only for the endcap
  
  static std::atomic<bool> isInitializedFromGeometry_;

  [[cms::thread_guard("isInitializedFromGeometry_")]]
  static short endcapRingIndex_[EEDetId::IX_MAX][EEDetId::IY_MAX];  // array needed only for the endcaps

  static std::once_flag once_;

};
#endif
