#ifndef DataFormats_ForwardDetId_HGCTriggerHexDetId_H
#define DataFormats_ForwardDetId_HGCTriggerHexDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "TObject.h"


class HGCTriggerHexDetId : public DetId {


    public:
    // |   DetId           | HGCTriggerHexDetId 
    // | 1111     | 111    | 1     | 11111 |      1     | 1111111111 | 11111111
    // | detector | subdet | zside | layer | wafer type |    wafer   |   cell
    // | 15       | 7      | 2     | 31    |     2      |   1023     |   255
        static const int cell_shift = 0;
        static const int cell_mask = 0xFF;
        static const int wafer_shift = 8;
        static const int wafer_mask = 0x3FF;
        static const int wafer_type_shift = 18;
        static const int wafer_type_mask = 0x1;
        static const int layer_shift = 19;
        static const int layer_mask= 0x1F;
        static const int zside_shift = 24;
        static const int zside_mask = 0x1;

    public:
        // undefined cell, for module det id
        const static uint32_t UndefinedCell() { return cell_mask; }


        /** Create a null cellid*/
        HGCTriggerHexDetId();
        virtual ~HGCTriggerHexDetId(){}
        /** Create cellid from raw id (0=invalid tower id) */
        HGCTriggerHexDetId(uint32_t rawid);
        /** Constructor from subdetector, zplus, layer, wafer type, wafer, cell numbers */
        HGCTriggerHexDetId(ForwardSubdetector subdet, int zp, int lay, int wafertype, int wafer, int cell);
        /** Constructor from a generic cell id */
        HGCTriggerHexDetId(const DetId& id);
        /** Assignment from a generic cell id */
        HGCTriggerHexDetId& operator=(const DetId& id);

        /// get the cell #
        int cell() const { return getMaskedId(cell_shift,cell_mask); }

        /// get the wafer #
        int wafer() const { return getMaskedId(wafer_shift,wafer_mask); }

        /// get the wafer type
        int waferType() const { return ( getMaskedId(wafer_type_shift,wafer_type_mask) ? 1 : -1);}
            
        /// get the layer #
        int layer() const { return getMaskedId(layer_shift,layer_mask); }

        /// get the z-side of the cell (1/-1)
        int zside() const { return ( getMaskedId(zside_shift,zside_mask) ? 1 : -1); }


        // Static getters and setters to work on raw ids
        static int subdet(uint32_t id) {return getMaskedId(id, kSubdetOffset, 0x7);}
        static int cell(uint32_t id) {return getMaskedId(id, cell_shift,cell_mask);}
        static int wafer(uint32_t id) {return getMaskedId(id, wafer_shift,wafer_mask);}
        static int waferType(uint32_t id) {return (getMaskedId(id, wafer_type_shift,wafer_type_mask) ? 1 : -1);}
        static int layer(uint32_t id) {return getMaskedId(id, layer_shift,layer_mask);}
        static int zside(uint32_t id) {return (getMaskedId(id, zside_shift,zside_mask) ? 1 : -1);}

        static void setCell(uint32_t& id, int cell) 
        {
            resetMaskedId(id, cell_shift,cell_mask);
            setMaskedId(id, cell, cell_shift,cell_mask);
        }
        static void setWafer(uint32_t& id, int mod) 
        {
            resetMaskedId(id, wafer_shift,wafer_mask);
            setMaskedId(id, mod, wafer_shift,wafer_mask);
        }
        static void setWaferType(uint32_t& id, int wafertype) 
        {
            resetMaskedId(id, wafer_type_shift, wafer_type_mask);
            setMaskedId(id, wafertype, wafer_type_shift,wafer_type_mask);
        }
        static void setLayer(uint32_t& id, int lay) 
        {
            resetMaskedId(id, layer_shift,layer_mask);
            setMaskedId(id, lay, layer_shift,layer_mask);
        }
        static void setZside(uint32_t& id, int zside) 
        {
            resetMaskedId(id, zside_shift,zside_mask);
            setMaskedId(id, zside, zside_shift,zside_mask);
        }

        /// consistency check
        bool isHGCal()   const { return true; }
        bool isForward() const { return true; }

        static const HGCTriggerHexDetId Undefined;

    private:
        const inline int getMaskedId(const uint32_t &shift, const uint32_t &mask) const  { return (id_ >> shift) & mask ; }
        inline void setMaskedId( const uint32_t value, const uint32_t &shift, const uint32_t &mask ){ id_|= ((value & mask ) <<shift ); }

        static const inline int getMaskedId(const uint32_t& id, const uint32_t &shift, const uint32_t &mask) { return (id >> shift) & mask ; }
        static inline void setMaskedId(uint32_t& id, const uint32_t value, const uint32_t &shift, const uint32_t &mask ){ id|= ((value & mask ) <<shift ); }
        static inline void resetMaskedId(uint32_t& id, const uint32_t &shift, const uint32_t &mask ){ id &= ~(mask<<shift); }

};

std::ostream& operator<<(std::ostream&,const HGCTriggerHexDetId& id);


#endif
