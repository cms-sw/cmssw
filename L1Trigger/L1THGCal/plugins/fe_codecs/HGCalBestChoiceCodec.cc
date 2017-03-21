#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodec.h"
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
void HGCalBestChoiceCodec::setDataPayloadImpl(const HGCEEDigiCollection& ee,
        const HGCHEDigiCollection& fh,
        const HGCHEDigiCollection& ) 
/*****************************************************************/
{
    data_.reset();
    std::vector<HGCDataFrame<HGCalDetId,HGCSample>> dataframes;
    std::vector<std::pair<HGCalDetId, uint32_t > > linearized_dataframes;
    // convert ee and fh hit collections into the same object
    if(ee.size()>0)
    {
        for(const auto& eedata : ee)
        {
            dataframes.emplace_back(eedata.id());
            for(int i=0; i<eedata.size(); i++)
            {
                dataframes.back().setSample(i, eedata.sample(i));
            }
        }
    }
    else if(fh.size()>0)
    {
        for(const auto& fhdata : fh)
        {
            dataframes.emplace_back(fhdata.id());
            for(int i=0; i<fhdata.size(); i++)
            {
                dataframes.back().setSample(i, fhdata.sample(i));
            }
        }
    }
    // linearize input energy on 16 bits
    codecImpl_.linearize(dataframes, linearized_dataframes);
    // sum energy in trigger cells
    codecImpl_.triggerCellSums(*geometry_, linearized_dataframes, data_);
    // choose best trigger cells in the module
    codecImpl_.bestChoiceSelect(data_);
}

/*****************************************************************/
void HGCalBestChoiceCodec::setDataPayloadImpl(const l1t::HGCFETriggerDigi& digi)
/*****************************************************************/
{
    data_.reset();
    // decode input data with different parameters
    // (no selection, so NData=number of trigger cells in module)
    // FIXME:
    // Not very clean to define an alternative codec within this codec 
    // Also, the codec is built each time the method is called, which is not very efficient
    // This may need a restructuration of the FECodec
    edm::ParameterSet conf;
    conf.addParameter<std::string>("CodecName",     name());
    conf.addParameter<uint32_t>   ("CodecIndex",    getCodecType());
    conf.addParameter<uint32_t>   ("NData",         HGCalBestChoiceCodec::data_type::size);
    // The data length should be the same for input and output, which is limiting
    conf.addParameter<uint32_t>   ("DataLength",    codecImpl_.dataLength());
    conf.addParameter<double>     ("linLSB",        codecImpl_.linLSB());
    conf.addParameter<double>     ("adcsaturation", codecImpl_.adcsaturation());
    conf.addParameter<uint32_t>   ("adcnBits",      codecImpl_.adcnBits());
    conf.addParameter<double>     ("tdcsaturation", codecImpl_.tdcsaturation());
    conf.addParameter<uint32_t>   ("tdcnBits",      codecImpl_.tdcnBits());
    conf.addParameter<double>     ("tdcOnsetfC",    codecImpl_.tdcOnsetfC());
    conf.addParameter<uint32_t>   ("triggerCellTruncationBits", codecImpl_.triggerCellTruncationBits());
    HGCalBestChoiceCodec codecInput(conf);
    codecInput.setGeometry(geometry_);
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
HGCalBestChoiceCodec::data_type HGCalBestChoiceCodec::decodeImpl(const std::vector<bool>& data, const uint32_t) const 
/*****************************************************************/
{
    return codecImpl_.decode(data);
}


