#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFEProcessor.h"
#include <limits>

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

DEFINE_EDM_PLUGIN(HGCalVFEProcessorBaseFactory, 
        HGCalVFEProcessor,
        "HGCalVFEProcessor");


HGCalVFEProcessor::
HGCalVFEProcessor(const edm::ParameterSet& conf) : HGCalVFEProcessorBase(conf),
    linLSB_(conf.getParameter<double>("linLSB")),
    vfeLinearizationImpl_(conf),
    vfeSummationImpl_(conf),
    calibration_( conf ),
    HGCalEESensitive_( conf.getParameter<std::string>("HGCalEESensitive_tag") ),
    HGCalHESiliconSensitive_( conf.getParameter<std::string>("HGCalHESiliconSensitive_tag") )
{ 
}

void
HGCalVFEProcessor::
vfeProcessing(const HGCEEDigiCollection& ee,
        		const HGCHEDigiCollection& fh, 
			const HGCBHDigiCollection& bh, const edm::EventSetup& es) 
{ 
  es.get<IdealGeometryRecord>().get( HGCalEESensitive_,        hgceeTopoHandle_ );
  es.get<IdealGeometryRecord>().get( HGCalHESiliconSensitive_, hgchefTopoHandle_ );
  //es.get<IdealGeometryRecord>().get("", triggerGeometry_);
  
  std::vector<HGCDataFrame<DetId,HGCSample>> dataframes;
  std::vector<std::pair<DetId, uint32_t > > linearized_dataframes;
  std::map<HGCalDetId, uint32_t> payload;
  
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
  else if(bh.size()>0)
  { 
    for(const auto& bhdata : bh)
    {
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
    
    l1t::HGCalTriggerCell triggerCell(reco::LeafCandidate::LorentzVector(), id_value.second, 0, 0, 0, id_value.first.rawId());
    GlobalPoint point = geometry_->getTriggerCellPosition(id_value.first.rawId());
    // 'value' is hardware, so p4 is meaningless, except for eta and phi
    math::PtEtaPhiMLorentzVector p4((double)id_value.second/cosh(point.eta()), point.eta(), point.phi(), 0.);
    triggerCell.setP4(p4);
    triggerCell.setPosition(point);    

    // calibration part ---------------------------
    if( triggerCell.hwPt() > 0 )
    {
      HGCalDetId detid(triggerCell.detId());
		
      int subdet = detid.subdetId();
      int cellThickness = 0;
                
      if( subdet == HGCEE ){ 
        cellThickness = hgceeTopoHandle_->dddConstants().waferTypeL( (unsigned int)detid.wafer() );
      }
      else if( subdet == HGCHEF ){
        cellThickness = hgchefTopoHandle_->dddConstants().waferTypeL( (unsigned int)detid.wafer() );
      }
        else if( subdet == HGCHEB ){
        edm::LogWarning("DataNotFound") << "ATTENTION: the BH trigger cells are not yet implemented";
      }

      l1t::HGCalTriggerCell calibratedtriggercell( triggerCell );
      calibration_.calibrateInGeV( calibratedtriggercell, cellThickness );     

      triggerCell_product_->push_back(0, calibratedtriggercell);
    }
    //--------------------------------------------- 
  }
    
}


void HGCalVFEProcessor::putInEvent(edm::Event& evt)
{ 
  evt.put(std::move(triggerCell_product_), "calibratedTriggerCells");
  evt.put(std::move(triggerSums_product_), "calibratedTriggerCells");
}
