/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com)
*    
****************************************************************************/

#ifndef DataFormats_TotemRawData_TotemStructuralVFATId
#define DataFormats_TotemRawData_TotemStructuralVFATId

#include <cstdint>

/**
 * A read-out chip (VFAT) described by its detector and its position within the detector.
 *
 * detId can be interpreted by passing it to constructor of a class inheriting from DetId, e.g. TotemRPDetId.
 *
 * For RP data VFATs, chipPosition is 0 to 3.
 * For RP CC VFATs, detId corresponds to the first plane and chipPosition=100.
 *
 */
class TotemStructuralVFATId
{
  public:
    TotemStructuralVFATId(uint32_t di=0, uint32_t cp=0) : detId(di), chipPosition(cp)
    {
    }

    uint32_t getDetId() const
    {
      return detId;
    }

    uint32_t getChipPosition() const
    {
      return chipPosition;
    }

    bool operator < (const TotemStructuralVFATId &other) const
    {
      if (detId == other.detId)
		  return (chipPosition < other.chipPosition);

      return (detId < other.detId);
    }

  private:
    /// Raw representation of DetId class.
    uint32_t detId;

    /// Describes position of a chip within the detector
    uint32_t chipPosition;
};

#endif
