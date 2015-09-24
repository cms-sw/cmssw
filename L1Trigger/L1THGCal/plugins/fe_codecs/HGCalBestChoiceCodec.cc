#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"
#include <limits>

using namespace HGCalTriggerFE;

DEFINE_EDM_PLUGIN(HGCalTriggerFECodecFactory, 
        HGCalBestChoiceCodec,
        "HGCalBestChoiceCodec");

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
        //std::cout<<" Cell "<<eedata.id()<<" Module "<<geom.getModuleFromCell(eedata.id())->moduleId()<<"\n";
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
    unsigned dataLength = 8;
    unsigned nData = 12;
    unsigned nCells = data.payload.size();
    std::vector<bool> result(nCells + dataLength*nData);
    unsigned idata = 0;
    for(unsigned itc=0; itc<nCells; itc++)
    {
        uint32_t value = data.payload.at(itc);
        result[itc] = (value>0 ? 1 : 0);
        if(value>0)
        {
            // FIXME: a proper coding is needed here. Needs studies.
            // truncate to 8 bits by keeping bits 10----3. 
            // Values > 0x3FF are saturated to 0x3FF
            if(value>0x3FF) value=0x3FF; // 10 bit saturation
            for(unsigned i=0; i<dataLength; i++)
            {
                result[nCells + idata*dataLength + i] = static_cast<bool>(value & (0x1<<(i+2)));// remove the two lowest bits
            }
            idata++;
        }
    }
    //unsigned nb = 0;
    //for(unsigned i=0;i<64;i++)
    //{
        //if(result[i]) nb++;
    //}
    //if(nb>6)
    //{
        //for(unsigned i=0;i<nCells;i++) std::cout<<result[i];
        //std::cout<<"|";
        //for(unsigned itc=0;itc<nData;itc++)
        //{
            //for(unsigned i=nCells+itc*dataLength;i<nCells+(itc+1)*dataLength;i++) std::cout<<result[i];
            //std::cout<<"|";
        //}
        //std::cout<<"\n";
    //}
    return result;
}

/*****************************************************************/
HGCalBestChoiceCodec::data_type HGCalBestChoiceCodec::decodeImpl(const std::vector<bool>& data) const 
/*****************************************************************/
{
    data_type result;
    result.reset();
    // FIXME: the number of best cells and value bits should be given in parameters
    if(data.size()!=64+8*12)
    {
        edm::LogWarning("HGCalBestChoiceCodec") 
            << "decode: data length ("<<data.size()<<") inconsistent with codec parameters:\n"\
            << "      : Map size = 64\n"\
            << "      : Number of energy values = 12\n"\
            << "      : Energy value length = 8\n";
        return result;
    }
    size_t c = 0;
    for(size_t b=0; b<64; b++)
    {
        if(data[b])
        {
            uint32_t value = 0;
            for(size_t i=0;i<8;i++)
            {
                size_t index = 64+c*8+i; 
                if(data[index]) value |= (0x1<<i);
            }
            c++;
            result.payload[b] = value;
        }
    }
    //unsigned nCells = 0;
    //for(const auto& value : data_.payload)
    //{
        //if(value>0) nCells++;
    //}
    //if(nCells>6)
    //{
        //std::cout<<"Data after decoding\n";
        //for(size_t i=0; i<data_.payload.size(); i++)
        //{
            //std::cout<<"  "<<i+1<<" -> "<<data_.payload.at(i)<<"\n";
        //}
    //}
    return result;
}


/*****************************************************************/
void HGCalBestChoiceCodec::triggerCellSums(const HGCalTriggerGeometryBase& geom, const std::vector<HGCEEDataFrame>& dataframes)
/*****************************************************************/
{
    //if(dataframes.size()>0)
    //{
        //std::cout<<">>>>\n";
        //std::cout<<dataframes.size()<<" Cells:\n";
    //}
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
        //std::cout<<" "<<cellid<<" ("<<triggercellid<<") = "<<data<<"\n";
        payload[triggercellid] += data; // 32 bits integer should be largely enough (maximum 7 12-bits sums are done)

    }
    // fill data payload
    for(const auto& id_value : payload)
    {
        uint32_t id = id_value.first.cell();
        if(static_cast<unsigned>(id>data_.payload.size())) // cell number starts at 1
        {
            edm::LogWarning("HGCalBestChoiceCodec") 
                << "Number of trigger cells in module too large for available data payload\n";
            continue;
        }
        data_.payload.at(id-1) = id_value.second;
    }
    //unsigned nCells = 0;
    //for(const auto& value : data_.payload)
    //{
        //if(value>0) nCells++;
    //}
    //if(nCells>6)
    //{
        //std::cout<<"Data before best choice\n";
        //for(size_t i=0; i<data_.payload.size(); i++)
        //{
            //std::cout<<"  "<<i+1<<" -> "<<data_.payload.at(i)<<"\n";
        //}
    //}
    //if(data_.payload.size()>0) std::cout<<data_.payload.size()<<" Trigger cells before selection:\n";
    //for(const auto& id_value : data_.payload)
    //{
        //std::cout<<"  "<<id_value.first<<" = "<<id_value.second<<"\n";
    //}
}


/*****************************************************************/
void HGCalBestChoiceCodec::bestChoiceSelect()
/*****************************************************************/
{
    // Store data payload in vector for energy sorting. Then refill the data payload after trigger
    // cell selection.
    // Probably not the most efficient way.
    // Should check in the firmware how cells with the same energy are sorted

    //HGCalBestChoiceDataPayload::trigger_cell_list sortedtriggercells(data_.payload); // copy for sorting
    std::vector< std::pair<uint32_t, uint32_t> > sortedtriggercells; // value, ID
    sortedtriggercells.reserve(data_.payload.size());
    for(size_t i=0; i<data_.payload.size(); i++)
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
    // FIXME: the number of best cells should be given in parameters
    for(size_t i=12; i<sortedtriggercells.size(); i++)
    {
        sortedtriggercells.at(i).first = 0;
    }
    for(const auto& value_id : sortedtriggercells)
    {
        if(static_cast<unsigned>(value_id.second>data_.payload.size())) // cell number starts at 1
        {
            edm::LogWarning("HGCalBestChoiceCodec") 
                << "Number of trigger cells in module too large for available data payload\n";
        }
        data_.payload.at(value_id.second) = value_id.first;
    }
    //unsigned nCells = 0;
    //for(const auto& value : data_.payload)
    //{
        //if(value>0) nCells++;
    //}
    //if(nCells>6)
    //{
        //std::cout<<"Data after best choice\n";
        //for(size_t i=0; i<data_.payload.size(); i++)
        //{
            //std::cout<<"  "<<i+1<<" -> "<<data_.payload.at(i)<<"\n";
        //}
    //}
}


