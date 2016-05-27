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
    std::vector<std::pair<HGCEEDetId, uint32_t > > linearized_dataframes;
    codecImpl_.linearize(mod, dataframes, linearized_dataframes);
    // sum energy in trigger cells
    codecImpl_.triggerCellSums(mod, linearized_dataframes, data_);
    // choose best trigger cells in the module
    codecImpl_.bestChoiceSelect(data_);

}

/*****************************************************************/
void HGCalBestChoiceCodec::setDataPayloadImpl(const Module& mod, 
        const l1t::HGCFETriggerDigi& digi)
/*****************************************************************/
{
    data_.reset();
    edm::ParameterSet conf;
    conf.addParameter<std::string>("CodecName",     name());
    conf.addParameter<uint32_t>   ("CodecIndex",    getCodecType());
    conf.addParameter<uint32_t>   ("NData",         HGCalBestChoiceCodec::data_type::size);
    conf.addParameter<uint32_t>   ("DataLength",    codecImpl_.dataLength());
    conf.addParameter<double>     ("linLSB",        codecImpl_.linLSB());
    conf.addParameter<double>     ("adcsaturation", codecImpl_.adcsaturation());
    conf.addParameter<uint32_t>   ("adcnBits",      codecImpl_.adcnBits());
    conf.addParameter<double>     ("tdcsaturation", codecImpl_.tdcsaturation());
    conf.addParameter<uint32_t>   ("tdcnBits",      codecImpl_.tdcnBits());
    conf.addParameter<double>     ("tdcOnsetfC",    codecImpl_.tdcOnsetfC());
    // decode input data with different parameters
    // (no selection, so NData=number of trigger cells in module)
    // FIXME:
    // Not very clean to define an alternative codec within this codec 
    // Also, the codec is built each time the method is called, which is not very efficient
    HGCalBestChoiceCodec codecInput(conf);
    digi.decode(codecInput,data_);
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


