#ifndef __L1Trigger_L1THGCal_HGCalBestChoiceCodecImpl_h__
#define __L1Trigger_L1THGCal_HGCalBestChoiceCodecImpl_h__


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include <array>
#include <vector>


struct HGCalBestChoiceDataPayload
{
    static const size_t size = 64;
    typedef std::array<uint32_t, size> trigger_cell_list; // list of data in 64 trigger cells
    trigger_cell_list payload;

    void reset() 
    { 
        payload.fill(0);
    }
};



class HGCalBestChoiceCodecImpl
{
    public:
        typedef HGCalBestChoiceDataPayload data_type;

        HGCalBestChoiceCodecImpl(const edm::ParameterSet& conf);

        std::vector<bool> encode(const data_type&) const ;
        data_type         decode(const std::vector<bool>&) const;  

        void triggerCellSums(const HGCalTriggerGeometry::Module& , const std::vector<HGCEEDataFrame>&, data_type&);
        void bestChoiceSelect(data_type&);

    private:
        size_t nData_;
        size_t dataLength_;
        size_t nCellsInModule_;

};

#endif
