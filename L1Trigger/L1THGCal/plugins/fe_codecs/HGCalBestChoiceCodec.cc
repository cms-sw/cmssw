#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"
#include <limits>

using namespace HGCalTriggerFE;

DEFINE_EDM_PLUGIN(HGCalTriggerFECodecFactory, 
        HGCalBestChoiceCodec,
        "HGCalBestChoiceCodec");

/*****************************************************************/
HGCalBestChoiceCodec::HGCalBestChoiceCodec(const edm::ParameterSet& conf) : Codec(conf),
    codecImpl_(conf)
/*****************************************************************/
{
}


/*****************************************************************/
void HGCalBestChoiceCodec::setDataPayloadImpl(const Module& mod, 
        const HGCEEDigiCollection& ee,
        const HGCHEDigiCollection&,
        const HGCHEDigiCollection& ) 
/*****************************************************************/
{
    data_.reset();
    std::vector<HGCEEDataFrame> dataframes;
    // loop over EE digis and fill digis belonging to that module
    for(const auto& eedata : ee)
    {
        if(mod.containsCell(eedata.id()))
        {
            dataframes.push_back(eedata);
        }
    }
    // sum energy in trigger cells
    codecImpl_.triggerCellSums(mod, dataframes, data_);
    // choose best trigger cells in the module
    codecImpl_.bestChoiceSelect(data_);

}


/*****************************************************************/
std::vector<bool> HGCalBestChoiceCodec::encodeImpl(const HGCalBestChoiceCodec::data_type& data) const 
/*****************************************************************/
{
    return codecImpl_.encode(data);
}

/*****************************************************************/
HGCalBestChoiceCodec::data_type HGCalBestChoiceCodec::decodeImpl(const std::vector<bool>& data) const 
/*****************************************************************/
{
    return codecImpl_.decode(data);
}


