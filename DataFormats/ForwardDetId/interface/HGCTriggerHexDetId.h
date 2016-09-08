#ifndef DataFormats_ForwardDetId_HGCTriggerHexDetId_H
#define DataFormats_ForwardDetId_HGCTriggerHexDetId_H 1

#include <iosfwd>
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "TObject.h"


// Same bit structure as HGCalDetId, with additional static methods to work with raw ids
class HGCTriggerHexDetId : public HGCalDetId 
{
    // |   DetId           | HGCTriggerHexDetId 
    // | 1111     | 111    | 1     | 11111 |      1     | 1111111111 | 11111111
    // | detector | subdet | zside | layer | wafer type |    wafer   |   cell
    // | 15       | 7      | 2     | 31    |     2      |   1023     |   255


    public:
        // undefined cell, for module det id
        const static uint32_t UndefinedCell() { return kHGCalCellMask; }


        /** Create a null cellid*/
        HGCTriggerHexDetId();
        /** Create cellid from raw id (0=invalid tower id) */
        HGCTriggerHexDetId(uint32_t rawid);
        /** Constructor from subdetector, zplus, layer, wafer type, wafer, cell numbers */
        HGCTriggerHexDetId(ForwardSubdetector subdet, int zp, int lay, int wafertype, int wafer, int cell);
        /** Constructor from a generic cell id */
        HGCTriggerHexDetId(const DetId& id);
        /** Assignment from a generic cell id */
        HGCTriggerHexDetId& operator=(const DetId& id);

        // Static getters and setters to work on raw ids
        static int subdetIdOf(uint32_t id) {return getMaskedId(id, kSubdetOffset, 0x7);}
        static int cellOf(uint32_t id) {return getMaskedId(id, kHGCalCellOffset,kHGCalCellMask);}
        static int waferOf(uint32_t id) {return getMaskedId(id, kHGCalWaferOffset,kHGCalWaferMask);}
        static int waferTypeOf(uint32_t id) {return (getMaskedId(id, kHGCalWaferTypeOffset,kHGCalWaferTypeMask) ? 1 : -1);}
        static int layerOf(uint32_t id) {return getMaskedId(id, kHGCalLayerOffset,kHGCalLayerMask);}
        static int zsideOf(uint32_t id) {return (getMaskedId(id, kHGCalZsideOffset,kHGCalZsideMask) ? 1 : -1);}

        static void setCellOf(uint32_t& id, int cell) 
        {
            resetMaskedId(id, kHGCalCellOffset,kHGCalCellMask);
            setMaskedId(id, cell, kHGCalCellOffset,kHGCalCellMask);
        }
        static void setWaferOf(uint32_t& id, int mod) 
        {
            resetMaskedId(id, kHGCalWaferOffset,kHGCalWaferMask);
            setMaskedId(id, mod, kHGCalWaferOffset,kHGCalWaferMask);
        }
        static void setWaferTypeOf(uint32_t& id, int wafertype) 
        {
            resetMaskedId(id, kHGCalWaferTypeOffset, kHGCalWaferTypeMask);
            setMaskedId(id, wafertype, kHGCalWaferTypeOffset,kHGCalWaferTypeMask);
        }
        static void setLayerOf(uint32_t& id, int lay) 
        {
            resetMaskedId(id, kHGCalLayerOffset,kHGCalLayerMask);
            setMaskedId(id, lay, kHGCalLayerOffset,kHGCalLayerMask);
        }
        static void setZsideOf(uint32_t& id, int zside) 
        {
            resetMaskedId(id, kHGCalZsideOffset,kHGCalZsideMask);
            setMaskedId(id, zside, kHGCalZsideOffset,kHGCalZsideMask);
        }


    private:
        static const inline int getMaskedId(const uint32_t& id, const uint32_t &shift, const uint32_t &mask) { return (id >> shift) & mask ; }
        static inline void setMaskedId(uint32_t& id, const uint32_t value, const uint32_t &shift, const uint32_t &mask ){ id|= ((value & mask ) <<shift ); }
        static inline void resetMaskedId(uint32_t& id, const uint32_t &shift, const uint32_t &mask ){ id &= ~(mask<<shift); }

};


#endif
