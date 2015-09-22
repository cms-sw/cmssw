#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodec.h"
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
    // initialize data payload with trigger cell DetIds contained in the module
    std::set<HGCEEDetId> triggerCellsInModule;
    for(const auto& triggercell : mod.components())
    {
        triggerCellsInModule.insert( HGCEEDetId(triggercell) );
    }
    //triggerCellsInModule.sort();
    unsigned index = 0;
    for(const auto& triggercell : triggerCellsInModule)
    {
        if(index>data_.payload.size()) 
        {
            edm::LogWarning("HGCalBestChoiceCodec") 
                << "Number of trigger cells in module too large for available data payload\n";
        }
        data_.payload.at(index).second = triggercell;
        index++; 
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
        uint32_t value = data.payload.at(itc).first;
        result[itc] = (value>0 ? 1 : 0);
        if(value>0)
        {
            // truncate 12 bits to 8 bits by keeping bits 10----3 + saturation
            if(value>=1024) value=1023; // 10 bit saturation
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
    std::map<HGCEEDetId, uint32_t> payload;
    // sum energies in trigger cells
    for(const auto& frame : dataframes)
    {
        HGCEEDetId cellid(frame.id());
        HGCEEDetId triggercellid( geom.getTriggerCellFromCell(cellid)->triggerCellId() );
        payload.insert( std::make_pair(triggercellid, 0) ); // do nothing if key exists already
        // FIXME: need to check how the cell energy can be retrieved
        uint32_t data = frame[2].data(); // This is 12 bit data
        //std::cout<<" "<<cellid<<" ("<<triggercellid<<") = "<<data<<"\n";
        payload[triggercellid] += data; // 32 bits integer should be largely enough (maximum 7 12-bits sums are done)

    }
    // fill data payload
    for(const auto& id_value : payload)
    {
        for(auto& value_id : data_.payload)
        {
            if(id_value.first==value_id.second)
            {
                value_id.first = id_value.second;
                break;
            }
        }
    }
    unsigned nCells = 0;
    for(const auto& value_id : data_.payload)
    {
        if(value_id.first>0) nCells++;
    }
    //if(nCells>6)
    //{
        //std::cout<<"Trigger cells in module before selection \n";//<<HGCEEDetId(mod.moduleId())<<" : "<<data_.payload.size()<<"\n";
        //for(const auto& value_id : data_.payload)
        //{
            //std::cout<<"  "<<value_id.second.cell()<<" -> "<<value_id.first<<"\n";//<<(value_id.second==HGCEEDetId(0) ? "not valid" : "valid")<<"\n";
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

    HGCalBestChoiceDataPayload::trigger_cell_list sortedtriggercells(data_.payload); // copy for sorting
    // sort, reverse order
    sort(sortedtriggercells.begin(), sortedtriggercells.end(),
            [](const HGCalBestChoiceDataPayload::trigger_cell_list::value_type& a, 
                const  HGCalBestChoiceDataPayload::trigger_cell_list::value_type& b) -> bool
            { 
                return a > b; 
            } 
            );
    // keep only the 12 first trigger cells
    for(size_t i=12; i<sortedtriggercells.size(); i++)
    {
        sortedtriggercells.at(i).first = 0;
    }
    for(auto& value_id : data_.payload)
    {
        for(const auto& value_id_sort : sortedtriggercells)
        {
            if(value_id.second==value_id_sort.second)
            {
                value_id.first = value_id_sort.first;
                break;
            }
        }
    }
    //data_.payload.resize(std::min(data_.payload.size(),size_t(12)));
    //unsigned nCells = 0;
    //for(const auto& value_id : data_.payload)
    //{
        //if(value_id.first>0) nCells++;
    //}
    //if(nCells>6)
    //{
        //std::cout<<"Trigger cells in module \n";//<<HGCEEDetId(mod.moduleId())<<" : "<<data_.payload.size()<<"\n";
        //for(const auto& value_id : data_.payload)
        //{
            //std::cout<<"  "<<value_id.second.cell()<<" -> "<<value_id.first<<"\n";//<<(value_id.second==HGCEEDetId(0) ? "not valid" : "valid")<<"\n";
        //}
    //}
    // refill the data payload
    //data_.reset();
    //for(const auto& value_id : sortedtriggercells)
    //{
        //data_.payload.insert( std::make_pair(value_id.second, value_id.first) );
    //}
    //if(data_.payload.size()>0) std::cout<<data_.payload.size()<<" Trigger cells after selection:\n";
    //for(const auto& id_value : data_.payload)
    //{
        //std::cout<<"  "<<id_value.first<<" = "<<id_value.second<<"\n";
    //}
}


