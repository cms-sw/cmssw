#ifndef __L1Trigger_L1THGCal_HGCalBestChoiceCodec_h__
#define __L1Trigger_L1THGCal_HGCalBestChoiceCodec_h__

#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodecImpl.h"


inline std::ostream& operator<<(std::ostream& o, const HGCalBestChoiceDataPayload& data) 
{ 
    for(const auto& dat : data.payload)
    {
        o <<  dat << " ";
    }
    o << "\n";
    return o;
}


class HGCalBestChoiceCodec : public HGCalTriggerFE::Codec<HGCalBestChoiceCodec,HGCalBestChoiceDataPayload> 
{
    public:
        typedef HGCalBestChoiceDataPayload data_type;

        HGCalBestChoiceCodec(const edm::ParameterSet& conf, const HGCalTriggerGeometryBase* const geom);

        void setDataPayloadImpl(const HGCEEDigiCollection& ee,
                const HGCHEDigiCollection& fh,
                const HGCHEDigiCollection& bh );

        void setDataPayloadImpl(const l1t::HGCFETriggerDigi& digi);

        std::vector<bool> encodeImpl(const data_type&) const ;
        data_type         decodeImpl(const std::vector<bool>&) const;  

    private:
        HGCalBestChoiceCodecImpl codecImpl_;
};

#endif
