#ifndef DataFormats_L1TCalorimeter_HGCalTriggerCell_h
#define DataFormats_L1TCalorimeter_HGCalTriggerCell_h


#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

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

            ~HGCalTriggerCell();

            void setDetId(uint32_t detid) {detid_ = detid;}
            void setPosition(const GlobalPoint& position) {position_ = position;}

            uint32_t detId() const {return detid_;}
            const GlobalPoint& position() const {return position_;}

        private:
            uint32_t detid_;
            GlobalPoint position_;

    };

}

#endif
