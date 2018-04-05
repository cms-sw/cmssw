#ifndef DataFormats_L1TCalorimeter_HGCalTriggerCell_h
#define DataFormats_L1TCalorimeter_HGCalTriggerCell_h


#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

namespace l1t 
{

    class HGCalTriggerCell;
    typedef BXVector<HGCalTriggerCell> HGCalTriggerCellBxCollection;

    class HGCalTriggerCell : public L1Candidate 
    {

        public:

            HGCalTriggerCell() {}

            HGCalTriggerCell( const LorentzVector& p4,
                    int pt=0,
                    int eta=0,
                    int phi=0,
                    int qual=0, 
                    uint32_t detid=0);

            ~HGCalTriggerCell() override;

            void setDetId(uint32_t detid) {detid_ = HGCalDetId(detid);}
            void setPosition(const GlobalPoint& position) {position_ = position;}

            uint32_t detId() const {return detid_.rawId();}
            const GlobalPoint& position() const {return position_;}
            
            int subdetId() const {                
                return detid_.subdetId();               
            }
            int zside() const {                
                return detid_.zside();               
            }
            int layer() const {                
                return detid_.layer();               
            }
            
            void   setMipPt( double value ) { mipPt_ = value; }
            double mipPt() const            { return mipPt_;  }
            
        private:
            
            HGCalDetId detid_;
            GlobalPoint position_;
            
            double mipPt_;

    };

}

#endif
