#ifndef __L1Trigger_L1THGCal_HGCalTriggerCellBestChoiceCodec_h__
#define __L1Trigger_L1THGCal_HGCalTriggerCellBestChoiceCodec_h__

#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodecImpl.h"


inline std::ostream& operator<<(std::ostream& o, const HGCalTriggerCellBestChoiceDataPayload& data) 
{ 
    for(const auto& dat : data.payload)
    {
        o << "(" << std::hex << dat.detId() 
            << std::dec << " " << dat.hwPt() << ") ";
    }
    o << "\n";
    return o;
}


class HGCalTriggerCellBestChoiceCodec : public HGCalTriggerFE::Codec<HGCalTriggerCellBestChoiceCodec,HGCalTriggerCellBestChoiceDataPayload> 
{
    public:
        typedef HGCalTriggerCellBestChoiceDataPayload data_type;

        HGCalTriggerCellBestChoiceCodec(const edm::ParameterSet& conf);

        void setDataPayloadImpl(const HGCEEDigiCollection& ee,
                const HGCHEDigiCollection& fh,
                const HGCHEDigiCollection& bh );

        void setDataPayloadImpl(const l1t::HGCFETriggerDigi& digi);

        std::vector<bool> encodeImpl(const data_type&) const ;
        data_type         decodeImpl(const std::vector<bool>&, const uint32_t) const;  

    private:
        HGCalTriggerCellBestChoiceCodecImpl codecImpl_;
};

#endif
