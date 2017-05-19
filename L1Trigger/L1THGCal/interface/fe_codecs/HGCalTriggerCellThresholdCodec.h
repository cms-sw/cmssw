#ifndef __L1Trigger_L1THGCal_HGCalTriggerCellThresholdCodec_h__
#define __L1Trigger_L1THGCal_HGCalTriggerCellThresholdCodec_h__

#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellThresholdCodecImpl.h"


inline std::ostream& operator<<(std::ostream& o, const HGCalTriggerCellThresholdDataPayload& data) 
{ 
    for(const auto& dat : data.payload)
    {
        o << "(" << std::hex << dat.detId() 
            << std::dec << " " << dat.hwPt() << ") ";
    }
    o << "\n";
    return o;
}


class HGCalTriggerCellThresholdCodec : public HGCalTriggerFE::Codec<HGCalTriggerCellThresholdCodec,HGCalTriggerCellThresholdDataPayload> 
{
    public:
        typedef HGCalTriggerCellThresholdDataPayload data_type;

        HGCalTriggerCellThresholdCodec(const edm::ParameterSet& conf);

        void setDataPayloadImpl(const HGCEEDigiCollection& ee,
                const HGCHEDigiCollection& fh,
                const HGCHEDigiCollection& bh );

        void setDataPayloadImpl(const l1t::HGCFETriggerDigi& digi);

        std::vector<bool> encodeImpl(const data_type&) const ;
        data_type         decodeImpl(const std::vector<bool>&, const uint32_t) const;  

    private:
        HGCalTriggerCellThresholdCodecImpl codecImpl_;
};

#endif
