#ifndef __L1Trigger_L1THGCal_HGCalConcentratorSuperTriggerCellImpl_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorSuperTriggerCellImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

#include <array>
#include <vector>

class HGCalConcentratorSuperTriggerCellImpl
{
  public:
    HGCalConcentratorSuperTriggerCellImpl(const edm::ParameterSet& conf);

    void superTriggerCellSelectImpl(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput, std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput);

  private:

    int getSuperTriggerCellId(int detid) const ;

    class SuperTriggerCell {
  
    private:
        float sumPt, sumMipPt;
        int sumHwPt, maxHwPt; 
        unsigned maxId;

    public:
        SuperTriggerCell(){  sumPt=0, sumMipPt=0, sumHwPt=0, maxHwPt=0, maxId=0 ;}
        void add(const l1t::HGCalTriggerCell &c) {
            sumPt += c.pt();
            sumMipPt += c.mipPt();
            sumHwPt += c.hwPt();
            if (maxId == 0 || c.hwPt() > maxHwPt) {
                maxHwPt = c.hwPt();
                maxId = c.detId();
            }
        }
        void assignEnergy(l1t::HGCalTriggerCell &c) const {
            c.setHwPt(sumHwPt);
            c.setMipPt(sumMipPt);
            c.setP4(math::PtEtaPhiMLorentzVector(sumPt, c.eta(), c.phi(), c.mass())); // there's no setPt
        }
	unsigned GetMaxId()const{return maxId;}
    };
    
};

#endif
