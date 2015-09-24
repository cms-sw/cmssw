#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"
#include <limits>

using namespace HGCalTriggerFE;

DEFINE_EDM_PLUGIN(HGCalTriggerFECodecFactory, 
        HGCalBestChoiceCodec,
        "HGCalBestChoiceCodec");

/*****************************************************************/
HGCalBestChoiceCodec::HGCalBestChoiceCodec(const edm::ParameterSet& conf) : Codec(conf),
    nData_(conf.getParameter<uint32_t>("NData")),
    dataLength_(conf.getParameter<uint32_t>("DataLength")),
    nCellsInModule_(data_.payload.size())
/*****************************************************************/
{
}


/*****************************************************************/
void HGCalBestChoiceCodec::setDataPayloadImpl(const Module& mod, 
        const HGCalTriggerGeometryBase& geom,
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
        if(geom.getModuleFromCell(eedata.id())->moduleId()==mod.moduleId())
        {
            dataframes.push_back(eedata);
        }
    }
    // sum energy in trigger cells
    triggerCellSums(geom, dataframes);
    // choose best trigger cells in the module
    bestChoiceSelect();

}


/*****************************************************************/
std::vector<bool> HGCalBestChoiceCodec::encodeImpl(const HGCalBestChoiceCodec::data_type& data) const 
/*****************************************************************/
{
    // First nCellsInModule_ bits are encoding the map of selected trigger cells
    // Followed by nData_ words of dataLength_ bits, corresponding to energy/transverse energy of
    // the selected trigger cells
    std::vector<bool> result(nCellsInModule_ + dataLength_*nData_);
    size_t idata = 0;
    for(size_t itc=0; itc<nCellsInModule_; itc++)
    {
        uint32_t value = data.payload.at(itc);
        result[itc] = (value>0 ? 1 : 0);
        if(value>0)
        {
            // FIXME: a proper coding is needed here. Needs studies.
            // For the moment truncate to 8 bits by keeping bits 10----3. 
            // Values > 0x3FF are saturated to 0x3FF
            if(value>0x3FF) value=0x3FF; // 10 bit saturation
            for(size_t i=0; i<dataLength_; i++)
            {
                result[nCellsInModule_ + idata*dataLength_ + i] = static_cast<bool>(value & (0x1<<(i+2)));// remove the two lowest bits
            }
            idata++;
        }
    }
    return result;
}

/*****************************************************************/
HGCalBestChoiceCodec::data_type HGCalBestChoiceCodec::decodeImpl(const std::vector<bool>& data) const 
/*****************************************************************/
{
    data_type result;
    result.reset();
    if(data.size()!=nCellsInModule_+dataLength_*nData_)
    {
        edm::LogWarning("HGCalBestChoiceCodec") 
            << "decode: data length ("<<data.size()<<") inconsistent with codec parameters:\n"\
            << "      : Map size = "<<nCellsInModule_<<"\n"\
            << "      : Number of energy values = "<<nData_<<"\n"\
            << "      : Energy value length = "<<dataLength_<<"\n";
        return result;
    }
    size_t c = 0;
    for(size_t b=0; b<nCellsInModule_; b++)
    {
        if(data[b])
        {
            uint32_t value = 0;
            for(size_t i=0;i<dataLength_;i++)
            {
                size_t index = nCellsInModule_+c*dataLength_+i; 
                if(data[index]) value |= (0x1<<i);
            }
            c++;
            result.payload[b] = value;
        }
    }
    return result;
}


/*****************************************************************/
void HGCalBestChoiceCodec::triggerCellSums(const HGCalTriggerGeometryBase& geom, const std::vector<HGCEEDataFrame>& dataframes)
/*****************************************************************/
{
    std::map<HGCTriggerDetId, uint32_t> payload;
    // sum energies in trigger cells
    for(const auto& frame : dataframes)
    {
        // FIXME: only EE
        HGCEEDetId cellid(frame.id());
        HGCTriggerDetId triggercellid( geom.getTriggerCellFromCell(cellid)->triggerCellId() );
        payload.insert( std::make_pair(triggercellid, 0) ); // do nothing if key exists already
        // FIXME: need to transform ADC and TDC to the same linear scale on 12 bits
        uint32_t data = frame[2].data(); // 'data' has to be a 12 bit word
        payload[triggercellid] += data; // 32 bits integer should be largely enough (maximum 7 12-bits sums are done)

    }
    // fill data payload
    for(const auto& id_value : payload)
    {
        uint32_t id = id_value.first.cell();
        if(id>nCellsInModule_) // cell number starts at 1
        {
            edm::LogWarning("HGCalBestChoiceCodec") 
                << "Number of trigger cells in module too large for available data payload\n";
            continue;
        }
        data_.payload.at(id-1) = id_value.second;
    }
}


/*****************************************************************/
void HGCalBestChoiceCodec::bestChoiceSelect()
/*****************************************************************/
{
    // Store data payload in vector for energy sorting. Then refill the data payload after trigger
    // cell selection.
    // Probably not the most efficient way.
    // Should check in the firmware how cells with the same energy are sorted

    // copy for sorting
    std::vector< std::pair<uint32_t, uint32_t> > sortedtriggercells; // value, ID
    sortedtriggercells.reserve(nCellsInModule_);
    for(size_t i=0; i<nCellsInModule_; i++)
    {
        sortedtriggercells.push_back(std::make_pair(data_.payload[i], i));
    }
    // sort, reverse order
    sort(sortedtriggercells.begin(), sortedtriggercells.end(),
            [](const std::pair<uint32_t, uint32_t>& a, 
                const  std::pair<uint32_t, uint32_t>& b) -> bool
            { 
                return a > b; 
            } 
            );
    // keep only the 12 first trigger cells
    for(size_t i=nData_; i<nCellsInModule_; i++)
    {
        sortedtriggercells.at(i).first = 0;
    }
    for(const auto& value_id : sortedtriggercells)
    {
        if(value_id.second>nCellsInModule_) // cell number starts at 1
        {
            edm::LogWarning("HGCalBestChoiceCodec") 
                << "Number of trigger cells in module too large for available data payload\n";
        }
        data_.payload.at(value_id.second) = value_id.first;
    }
}


