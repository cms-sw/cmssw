#ifndef __L1Trigger_L1THGCal_HGCalBestChoiceCodec_h__
#define __L1Trigger_L1THGCal_HGCalBestChoiceCodec_h__

#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include <limits>

#include "TRandom3.h"


//struct HGCalBestChoiceDataPayload 
//{ 
    //std::vector<HGCEEDataFrame> payload;
    //void reset() 
    //{ 
        //payload.clear();
    //}
//};

struct HGCalBestChoiceDataPayload
{
    typedef std::array< std::pair<uint32_t, HGCEEDetId>, 64 > trigger_cell_list; // list of (data, ID) pairs
    trigger_cell_list payload;

    void reset() 
    { 
        for(auto& value_id : payload)
        {
            value_id.first = 0;
            value_id.second = HGCEEDetId(0);
        }
    }
};


inline std::ostream& operator<<(std::ostream& o, const HGCalBestChoiceDataPayload& data) 
{ 
    for(const auto& dat_id : data.payload)
    {
        o <<  dat_id.second << " -> DATA=" << dat_id.first;
        o << "\n";
    }
    return o;
}

class HGCalBestChoiceCodec : public HGCalTriggerFE::Codec<HGCalBestChoiceCodec,HGCalBestChoiceDataPayload> 
{
    public:
        typedef HGCalBestChoiceDataPayload data_type;

        HGCalBestChoiceCodec(const edm::ParameterSet& conf) : Codec(conf)
        {
        }

        void setDataPayloadImpl(const Module& mod, 
                const HGCalTriggerGeometryBase& geom,
                const HGCEEDigiCollection& ee,
                const HGCHEDigiCollection& fh,
                const HGCHEDigiCollection& bh );

        std::vector<bool> encodeImpl(const data_type&) const ;
        data_type         decodeImpl(const std::vector<bool>&) const;  

    private:
        void triggerCellSums(const HGCalTriggerGeometryBase& , const std::vector<HGCEEDataFrame>&);
        void bestChoiceSelect();

};

#endif
