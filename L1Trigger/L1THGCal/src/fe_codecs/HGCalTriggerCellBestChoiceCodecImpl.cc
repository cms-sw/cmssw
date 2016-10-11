
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodecImpl.h"


HGCalTriggerCellBestChoiceCodecImpl::
HGCalTriggerCellBestChoiceCodecImpl(const edm::ParameterSet& conf) :
    nData_(conf.getParameter<uint32_t>("NData")),
    dataLength_(conf.getParameter<uint32_t>("DataLength")),
    nCellsInModule_(116),
    linLSB_(conf.getParameter<double>("linLSB")),
    adcsaturation_(conf.getParameter<double>("adcsaturation")),
    adcnBits_(conf.getParameter<uint32_t>("adcnBits")),
    tdcsaturation_(conf.getParameter<double>("tdcsaturation")), 
    tdcnBits_(conf.getParameter<uint32_t>("tdcnBits")), 
    tdcOnsetfC_(conf.getParameter<double>("tdcOnsetfC")),
    triggerCellTruncationBits_(conf.getParameter<uint32_t>("triggerCellTruncationBits"))
{
  // Cannot have more selected cells than the max number of cells
  if(nData_>nCellsInModule_) nData_ = nCellsInModule_;
  adcLSB_ =  adcsaturation_/pow(2.,adcnBits_);
  tdcLSB_ =  tdcsaturation_/pow(2.,tdcnBits_);
  triggerCellSaturationBits_ = triggerCellTruncationBits_ + dataLength_;
}



std::vector<bool>
HGCalTriggerCellBestChoiceCodecImpl::
encode(const HGCalTriggerCellBestChoiceCodecImpl::data_type& data, const HGCalTriggerGeometryBase& geometry) const 
{
    // First nCellsInModule_ bits are encoding the map of selected trigger cells
    // Followed by nData_ words of dataLength_ bits, corresponding to energy/transverse energy of
    // the selected trigger cells
    std::vector<bool> result(nCellsInModule_ + dataLength_*nData_, 0);
    // No data: return vector of 0
    if(data.payload.size()==0) return result;
    size_t idata = 0; // counter for the number of non-zero energy values
    // All trigger cells are in the same module
    // Retrieve once the ordered list of trigger cells in this module
    uint32_t module = geometry.getModuleFromTriggerCell(data.payload.begin()->detId());
    HGCalTriggerGeometryBase::geom_ordered_set trigger_cells_in_module = geometry.getOrderedTriggerCellsFromModule(module);
    for(const auto& cell : data.payload)
    {
        // First need to find the index of this trigger cell in the module
        // FIXME: std::distance is linear with size for sets (no random access). In order to have constant
        // access would require to convert the set into a vector. 
        uint32_t index = std::distance(trigger_cells_in_module.begin(),trigger_cells_in_module.find(cell.detId()));
        if(index>=nCellsInModule_) 
        {
            throw cms::Exception("BadGeometry")
                << "Number of trigger cells in module too large for available data payload\n";
        }
        // Then set the corresponding adress bit and fill energy if >0
        uint32_t value = cell.hwPt(); // This is actually energy, not Et (and certainly not Pt)
        result[index] = (value>0 ? 1 : 0);
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
                // remove the lowest bits (=triggerCellTruncationBits_)
                result[nCellsInModule_ + idata*dataLength_ + i] = static_cast<bool>(value & (0x1<<(i+triggerCellTruncationBits_)));
            }
            idata++;
        }
    }
    return result;
}

HGCalTriggerCellBestChoiceCodecImpl::data_type 
HGCalTriggerCellBestChoiceCodecImpl::
decode(const std::vector<bool>& data, const uint32_t module, const HGCalTriggerGeometryBase& geometry) const 
{
    data_type result;
    result.reset();
    // TODO: could eventually reserve memory to the max size of trigger cells
    if(data.size()!=nCellsInModule_+dataLength_*nData_)
    {
        throw cms::Exception("BadData") 
            << "decode: data length ("<<data.size()<<") inconsistent with codec parameters:\n"\
            << "      : Map size = "<<nCellsInModule_<<"\n"\
            << "      : Number of energy values = "<<nData_<<"\n"\
            << "      : Energy value length = "<<dataLength_<<"\n";
    }
    HGCalTriggerGeometryBase::geom_ordered_set trigger_cells_in_module = geometry.getOrderedTriggerCellsFromModule(module);
    size_t iselected = 0;
    size_t index = 0;
    for(const auto& triggercell : trigger_cells_in_module)
    {
        if(index>=nCellsInModule_)
        {
            throw cms::Exception("BadGeometry")
                << "Number of trigger cells in module too large for available data payload\n";
        }
        if(data[index])
        {
            uint32_t value = 0;
            for(size_t i=0;i<dataLength_;i++)
            {
                size_t ibit = nCellsInModule_+iselected*dataLength_+i; 
                if(data[ibit]) value |= (0x1<<i);
            }
            iselected++;
            // Build trigger cell
            if(value>0)
            {
                // Currently no hardware eta, phi and quality values
                result.payload.emplace_back(reco::LeafCandidate::LorentzVector(),
                        value, 0, 0, 0, triggercell); 
                GlobalPoint point = geometry.getTriggerCellPosition(triggercell);
                // 'value' is hardware, so p4 is meaningless, except for eta and phi
                math::PtEtaPhiMLorentzVector p4((double)value/cosh(point.eta()), point.eta(), point.phi(), 0.);
                result.payload.back().setP4(p4);
            }
        }
        index++;
    }
    return result;
}


void
HGCalTriggerCellBestChoiceCodecImpl::
linearize(const std::vector<HGCDataFrame<HGCalDetId,HGCSample>>& dataframes,
        std::vector<std::pair<HGCalDetId, uint32_t > >& linearized_dataframes)
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
  

void 
HGCalTriggerCellBestChoiceCodecImpl::
triggerCellSums(const HGCalTriggerGeometryBase& geometry,  const std::vector<std::pair<HGCalDetId, uint32_t > >& linearized_dataframes, data_type& data)
{
    if(linearized_dataframes.size()==0) return;
    std::map<HGCalDetId, uint32_t> payload;
    // sum energies in trigger cells
    for(const auto& frame : linearized_dataframes)
    {
        HGCalDetId cellid(frame.first);
        // find trigger cell associated to cell
        uint32_t tcid = geometry.getTriggerCellFromCell(cellid);
        HGCalDetId triggercellid( tcid );
        payload.insert( std::make_pair(triggercellid, 0) ); // do nothing if key exists already
        uint32_t value = frame.second; 
        payload[triggercellid] += value; // 32 bits integer should be largely enough 

    }
    uint32_t module = geometry.getModuleFromTriggerCell(payload.begin()->first);
    HGCalTriggerGeometryBase::geom_ordered_set trigger_cells_in_module = geometry.getOrderedTriggerCellsFromModule(module);
    // fill data payload
    for(const auto& id_value : payload)
    {
        // Store only energy value and detid
        // No need position here
        data.payload.emplace_back(reco::LeafCandidate::LorentzVector(),
                        id_value.second, 0, 0, 0, id_value.first.rawId());
    }
}

void 
HGCalTriggerCellBestChoiceCodecImpl::
bestChoiceSelect(data_type& data)
{
    // sort, reverse order
    sort(data.payload.begin(), data.payload.end(),
            [](const l1t::HGCalTriggerCell& a, 
                const  l1t::HGCalTriggerCell& b) -> bool
            { 
                return a.hwPt() > b.hwPt(); 
            } 
            );
    // keep only the first trigger cells
    if(data.payload.size()>nData_) data.payload.resize(nData_);
}


