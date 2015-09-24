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
    typedef std::array<uint32_t, 64 > trigger_cell_list; // list of data in 64 trigger cells
    trigger_cell_list payload;

    void reset() 
    { 
        for(auto& value : payload)
        {
            value = 0;
        }
    }
};


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

        HGCalBestChoiceCodec(const edm::ParameterSet& conf);

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

        size_t nData_;
        size_t dataLength_;
        size_t nCellsInModule_;

};

#endif
