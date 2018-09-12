#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFEProcessorSums.h"
#include <limits>

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

DEFINE_EDM_PLUGIN(HGCalVFEProcessorBaseFactory, 
        HGCalVFEProcessorSums,
        "HGCalVFEProcessorSums");


HGCalVFEProcessorSums::
HGCalVFEProcessorSums(const edm::ParameterSet& conf) : HGCalVFEProcessorBase(conf),
  vfeLinearizationImpl_(conf),
  vfeSummationImpl_(conf),
  calibration_( conf.getParameterSet("calib_parameters") )
{ 
}

void
HGCalVFEProcessorSums::run(const HGCEEDigiCollection& ee,
                           const HGCHEDigiCollection& fh, 
                           const HGCBHDigiCollection& bh, 
                           l1t::HGCalTriggerCellBxCollection& triggerCellColl, 
                           const edm::EventSetup& es) 
{ 
  calibration_.eventSetup(es);

  std::vector<HGCDataFrame<DetId,HGCSample>> dataframes;
  std::vector<std::pair<DetId, uint32_t >> linearized_dataframes;
  std::map<HGCalDetId, uint32_t> payload;

  // convert ee and fh hit collections into the same object  
  if(!ee.empty())
  {
    for(const auto& eedata : ee)
    { 
      uint32_t module = geometry_->getModuleFromCell(eedata.id());
      if(geometry_->disconnectedModule(module)) continue;
      dataframes.emplace_back(eedata.id());
      for(int i=0; i<eedata.size(); i++)
      {
        dataframes.back().setSample(i, eedata.sample(i));
      }
    }
  }
  else if(!fh.empty())
  {
    for(const auto& fhdata : fh)
    {
       uint32_t module = geometry_->getModuleFromCell(fhdata.id());
       if(geometry_->disconnectedModule(module)) continue;
       dataframes.emplace_back(fhdata.id());
       for(int i=0; i<fhdata.size(); i++)
       {
         dataframes.back().setSample(i, fhdata.sample(i));
       }
     }
   }
  else if(!bh.empty())
  {  
     for(const auto& bhdata : bh)
     { 
       if(HcalDetId(bhdata.id()).subdetId()!=HcalEndcap) continue;
       uint32_t module = geometry_->getModuleFromCell(bhdata.id());
       if(geometry_->disconnectedModule(module)) continue;
       dataframes.emplace_back(bhdata.id());
       for(int i=0; i<bhdata.size(); i++)
       {
         dataframes.back().setSample(i, bhdata.sample(i));
       }
     }
   }

  vfeLinearizationImpl_.linearize(dataframes, linearized_dataframes);
  vfeSummationImpl_.triggerCellSums(*geometry_, linearized_dataframes, payload);  
  
  // Transform map to trigger cell vector vector<HGCalTriggerCell>
  for(const auto& id_value : payload)
  { 
    if (id_value.second>0){
      l1t::HGCalTriggerCell triggerCell(reco::LeafCandidate::LorentzVector(), id_value.second, 0, 0, 0, id_value.first.rawId());
      GlobalPoint point = geometry_->getTriggerCellPosition(id_value.first.rawId());
      
      // 'value' is hardware, so p4 is meaningless, except for eta and phi
      math::PtEtaPhiMLorentzVector p4((double)id_value.second/cosh(point.eta()), point.eta(), point.phi(), 0.);
      triggerCell.setP4(p4);
      triggerCell.setPosition(point);    
    
      // calibration part ---------------------------
      if( triggerCell.hwPt() > 0 )
      { 
        l1t::HGCalTriggerCell calibratedtriggercell( triggerCell );
        calibration_.calibrateInGeV( calibratedtriggercell);     
        triggerCellColl.push_back(0, calibratedtriggercell);
      }
    }
  }    
}

