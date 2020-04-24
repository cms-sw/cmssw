
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodecImpl.h"


/*****************************************************************/
HGCalBestChoiceCodecImpl::HGCalBestChoiceCodecImpl(const edm::ParameterSet& conf) :
    nData_(conf.getParameter<uint32_t>("NData")),
    dataLength_(conf.getParameter<uint32_t>("DataLength")),
    nCellsInModule_(data_type::size),
    linLSB_(conf.getParameter<double>("linLSB")),
    adcsaturation_(conf.getParameter<double>("adcsaturation")),
    adcnBits_(conf.getParameter<uint32_t>("adcnBits")),
    tdcsaturation_(conf.getParameter<double>("tdcsaturation")), 
    tdcnBits_(conf.getParameter<uint32_t>("tdcnBits")), 
    tdcOnsetfC_(conf.getParameter<double>("tdcOnsetfC")),
    triggerCellTruncationBits_(conf.getParameter<uint32_t>("triggerCellTruncationBits"))
/*****************************************************************/
{
  // Cannot have more selected cells than the max number of cells
  if(nData_>nCellsInModule_) nData_ = nCellsInModule_;
  adcLSB_ =  adcsaturation_/pow(2.,adcnBits_);
  tdcLSB_ =  tdcsaturation_/pow(2.,tdcnBits_);
  triggerCellSaturationBits_ = triggerCellTruncationBits_ + dataLength_;
}



/*****************************************************************/
std::vector<bool> HGCalBestChoiceCodecImpl::encode(const HGCalBestChoiceCodecImpl::data_type& data) const 
/*****************************************************************/
{
    // First nCellsInModule_ bits are encoding the map of selected trigger cells
    // Followed by nData_ words of dataLength_ bits, corresponding to energy/transverse energy of
    // the selected trigger cells
    std::vector<bool> result(nCellsInModule_ + dataLength_*nData_, 0);
    size_t idata = 0;
    for(size_t itc=0; itc<nCellsInModule_; itc++)
    {
        uint32_t value = data.payload.at(itc);
        result[itc] = (value>0 ? 1 : 0);
        if(value>0)
        {
            if(idata>=nData_)
            {
                throw cms::Exception("BadData") 
                    << "encode: Number of non-zero trigger cells larger than codec parameter\n"\
                    << "      : Number of energy values = "<<nData_<<"\n";
            }
            // Saturate and truncate energy values
            if(value+1>(0x1u<<triggerCellSaturationBits_)) value = (0x1<<triggerCellSaturationBits_)-1;
            for(size_t i=0; i<dataLength_; i++)
            {
                result[nCellsInModule_ + idata*dataLength_ + i] = static_cast<bool>(value & (0x1<<(i+triggerCellTruncationBits_)));// remove the lowest bits (=triggerCellTruncationBits_)
            }
            idata++;
        }
    }
    return result;
}

/*****************************************************************/
HGCalBestChoiceCodecImpl::data_type HGCalBestChoiceCodecImpl::decode(const std::vector<bool>& data) const 
/*****************************************************************/
{
    data_type result;
    result.reset();
    if(data.size()!=nCellsInModule_+dataLength_*nData_)
    {
        throw cms::Exception("BadData") 
            << "decode: data length ("<<data.size()<<") inconsistent with codec parameters:\n"\
            << "      : Map size = "<<nCellsInModule_<<"\n"\
            << "      : Number of energy values = "<<nData_<<"\n"\
            << "      : Energy value length = "<<dataLength_<<"\n";
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
void HGCalBestChoiceCodecImpl::linearize(const std::vector<HGCDataFrame<DetId,HGCSample>>& dataframes,
        std::vector<std::pair<DetId, uint32_t > >& linearized_dataframes)
/*****************************************************************/
{
    double amplitude; uint32_t amplitude_int;
   

    for(const auto& frame : dataframes) {//loop on DIGI
        if (frame[2].mode()) {//TOT mode
            amplitude =( floor(tdcOnsetfC_/adcLSB_) + 1.0 )* adcLSB_ + double(frame[2].data()) * tdcLSB_;
        }
        else {//ADC mode
            amplitude = double(frame[2].data()) * adcLSB_;
        }

        amplitude_int = uint32_t (floor(amplitude/linLSB_+0.5));  
        if (amplitude_int>65535) amplitude_int = 65535;

        linearized_dataframes.push_back(std::make_pair (frame.id(), amplitude_int));
    }
}
  

/*****************************************************************/
void HGCalBestChoiceCodecImpl::triggerCellSums(const HGCalTriggerGeometryBase& geometry,  const std::vector<std::pair<DetId, uint32_t > >& linearized_dataframes, data_type& data)
/*****************************************************************/
{
    if(linearized_dataframes.size()==0) return;
    std::map<HGCalDetId, uint32_t> payload;
    // sum energies in trigger cells
    for(const auto& frame : linearized_dataframes)
    {
        DetId cellid(frame.first);
        // find trigger cell associated to cell
        uint32_t tcid = geometry.getTriggerCellFromCell(cellid);
        HGCalDetId triggercellid( tcid );
        payload.insert( std::make_pair(triggercellid, 0) ); // do nothing if key exists already
        // FIXME: need to transform ADC and TDC to the same linear scale on 12 bits
        uint32_t value = frame.second; // 'value' has to be a 12 bit word
        payload[triggercellid] += value; // 32 bits integer should be largely enough (maximum 7 12-bits sums are done)

    }
    uint32_t module = geometry.getModuleFromTriggerCell(payload.begin()->first);
    HGCalTriggerGeometryBase::geom_ordered_set trigger_cells_in_module = geometry.getOrderedTriggerCellsFromModule(module);
    // fill data payload
    for(const auto& id_value : payload)
    {
        // find the index of the trigger cell in the module (not necessarily equal to .cell())
        // FIXME: std::distance is linear with size for sets (no random access). In order to have constant
        // access would require to convert the set into a vector. 
        uint32_t id = std::distance(trigger_cells_in_module.begin(),trigger_cells_in_module.find(id_value.first));
        //uint32_t id = id_value.first.cell();
        //std::cerr<<"cell id in trigger cell sum: "<<id<<"("<<id_value.first.wafer()<<","<<id_value.first.cell()<<")\n";
        if(id>=nCellsInModule_) 
        {
            throw cms::Exception("BadGeometry")
                << "Number of trigger cells in module too large for available data payload\n";
        }
        data.payload.at(id) = id_value.second;
    }
}

/*****************************************************************/
void HGCalBestChoiceCodecImpl::bestChoiceSelect(data_type& data)
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
        sortedtriggercells.push_back(std::make_pair(data.payload[i], i));
    }
    // sort, reverse order
    sort(sortedtriggercells.begin(), sortedtriggercells.end(),
            [](const std::pair<uint32_t, uint32_t>& a, 
                const  std::pair<uint32_t, uint32_t>& b) -> bool
            { 
                return a > b; 
            } 
            );
    // keep only the first trigger cells
    for(size_t i=nData_; i<nCellsInModule_; i++)
    {
        sortedtriggercells.at(i).first = 0;
    }
    for(const auto& value_id : sortedtriggercells)
    {
        if(value_id.second>=nCellsInModule_)
        {
            throw cms::Exception("BadGeometry")
                << "Number of trigger cells in module too large for available data payload\n";
        }
        data.payload.at(value_id.second) = value_id.first;
    }
}


