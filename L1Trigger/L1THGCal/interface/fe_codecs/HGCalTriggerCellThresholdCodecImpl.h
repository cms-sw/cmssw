#ifndef __L1Trigger_L1THGCal_HGCalTriggerCellThresholdCodecImpl_h__
#define __L1Trigger_L1THGCal_HGCalTriggerCellThresholdCodecImpl_h__


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

#include <array>
#include <vector>


struct HGCalTriggerCellThresholdDataPayload
{
    typedef std::vector<l1t::HGCalTriggerCell> trigger_cell_list; // list of trigger cell 
    trigger_cell_list payload;

    void reset() 
    { 
        payload.clear();
    }
};



class HGCalTriggerCellThresholdCodecImpl
{
    public:
        typedef HGCalTriggerCellThresholdDataPayload data_type;

        HGCalTriggerCellThresholdCodecImpl(const edm::ParameterSet& conf);

        std::vector<bool> encode(const data_type&, const HGCalTriggerGeometryBase&) const ;
        data_type         decode(const std::vector<bool>&, const uint32_t, const HGCalTriggerGeometryBase&) const;  

        void linearize(const std::vector<HGCDataFrame<DetId,HGCSample>>&,
                std::vector<std::pair<DetId, uint32_t > >&);

        void triggerCellSums(const HGCalTriggerGeometryBase& ,
                const std::vector<std::pair<DetId, uint32_t > >&,
                data_type&);
        void thresholdSelect(data_type&);

        // Retrieve parameters
        size_t   nCellsInModule() const {return nCellsInModule_;}
        size_t   dataLength() const {return dataLength_;}
        double   linLSB() const {return linLSB_;}
        double   adcsaturation() const {return adcsaturation_;}
        uint32_t adcnBits() const {return adcnBits_;}
        double   tdcsaturation() const {return tdcsaturation_;}
        uint32_t tdcnBits() const {return tdcnBits_;}
        double   tdcOnsetfC() const {return tdcOnsetfC_;}
        double   adcsaturationBH() const {return adcsaturationBH_;}
        uint32_t adcnBitsBH() const {return adcnBitsBH_;}
        uint32_t triggerCellTruncationBits() const {return triggerCellTruncationBits_;}
        uint32_t triggerCellSaturationBits() const {return triggerCellSaturationBits_;}
        int      TCThreshold_ADC() const {return TCThreshold_ADC_;} 
        double    TCThreshold_fC() const {return TCThreshold_fC_;} 
        int      TCThresholdBH_ADC() const {return TCThresholdBH_ADC_;} 
        double   TCThresholdBH_MIP() const {return TCThresholdBH_MIP_;} 
        const std::vector<double>& ThicknessCorrections() const {return thickness_corrections_;} 

    private:
        size_t   dataLength_;
        size_t   nCellsInModule_;
        double   linLSB_;
        double   adcsaturation_;
        uint32_t adcnBits_;
        double   tdcsaturation_ ;
        uint32_t tdcnBits_ ;
        double   tdcOnsetfC_ ;
        double   adcsaturationBH_;
        uint32_t adcnBitsBH_;
        double   adcLSB_;
        double   tdcLSB_;
        double   adcLSBBH_;
        uint32_t triggerCellTruncationBits_;
        uint32_t triggerCellSaturationBits_;
        int      TCThreshold_ADC_;
        double    TCThreshold_fC_;
        int      TCThresholdBH_ADC_;
        double   TCThresholdBH_MIP_;
        std::vector<double> thickness_corrections_;

};

#endif
