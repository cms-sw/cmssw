#include <memory>

#include "CondTools/SiPixel/test/SiPixelCondObjForHLTReader.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"


namespace cms{
SiPixelCondObjForHLTReader::SiPixelCondObjForHLTReader(const edm::ParameterSet& conf): 
    conf_(conf),
    SiPixelGainCalibrationService_(conf)
{
}

void
SiPixelCondObjForHLTReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //Create Subdirectories
  edm::Service<TFileService> fs;
  TFileDirectory subDirPed = fs->mkdir("Pedestals");
  TFileDirectory subDirGain = fs->mkdir("Gains");
  char name[128];

  unsigned int nmodules = 0;
  uint32_t nchannels = 0;
  
  // Get the calibration data
  SiPixelGainCalibrationService_.setESObjects(iSetup);
  edm::LogInfo("SiPixelCondObjForHLTReader") << "[SiPixelCondObjForHLTReader::beginJob] End Reading CondObjForHLTects" << std::endl;

  // Get the Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get( tkgeom );     
  edm::LogInfo("SiPixelCondObjForHLTReader") <<" There are "<<tkgeom->dets().size() <<" detectors"<<std::endl;

  // Get the list of DetId's
  std::vector<uint32_t> vdetId_ = SiPixelGainCalibrationService_.getDetIds();

  //Create histograms
  _TH1F_Gains_sum =  fs->make<TH1F>("Summary_Gain","Gain Summary", vdetId_.size()+1,0,vdetId_.size()+1);
  _TH1F_Pedestals_sum =  fs->make<TH1F>("Summary_Pedestal","Pedestal Summary", vdetId_.size()+1,0,vdetId_.size()+1);
  _TH1F_Pedestals_all = fs->make<TH1F>("PedestalsAll","all Pedestals",350,-100,250);
  _TH1F_Gains_all = fs->make<TH1F>("GainsAll","all Gains",100,0,10);


  // Loop over DetId's
  int ibin=1;
  for (std::vector<uint32_t>::const_iterator detid_iter=vdetId_.begin();detid_iter!=vdetId_.end();detid_iter++){
    uint32_t detid = *detid_iter;

     sprintf(name,"Pedestals_%d",detid);
     _TH1F_Pedestals_m[detid] = subDirPed.make<TH1F>(name,name,250,0.,250.);    
     sprintf(name,"Gains_%d",detid);
     _TH1F_Gains_m[detid] = subDirGain.make<TH1F>(name,name,100,0.,10.); 

    DetId detIdObject(detid);
    nmodules++;

    std::map<uint32_t,TH1F*>::iterator p_iter =  _TH1F_Pedestals_m.find(detid);
    std::map<uint32_t,TH1F*>::iterator g_iter =  _TH1F_Gains_m.find(detid);
    
    const GeomDetUnit      * geoUnit = tkgeom->idToDetUnit( detIdObject );
    const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
    const PixelTopology & topol = pixDet->specificTopology();       

    // Get the module sizes.
    int nrows = topol.nrows();      // rows in x
    int ncols = topol.ncolumns();   // cols in y
    
    for(int col_iter=0; col_iter<ncols; col_iter++) {
       for(int row_iter=0; row_iter<nrows; row_iter++) {
          nchannels++;

          float gain  = SiPixelGainCalibrationService_.getGain(detid, col_iter, row_iter);
          _TH1F_Gains_m[detid]->Fill( gain );
	  _TH1F_Gains_all->Fill(gain); g_iter->second->Fill( gain );

          float ped  = SiPixelGainCalibrationService_.getPedestal(detid, col_iter, row_iter);
          _TH1F_Pedestals_m[detid]->Fill( ped );
       	  _TH1F_Pedestals_all->Fill(ped);

       }
    }

    _TH1F_Gains_sum->SetBinContent(ibin,_TH1F_Gains_m[detid]->GetMean());
    _TH1F_Gains_sum->SetBinError(ibin,_TH1F_Gains_m[detid]->GetRMS());
    _TH1F_Pedestals_sum->SetBinContent(ibin,_TH1F_Pedestals_m[detid]->GetMean());
    _TH1F_Pedestals_sum->SetBinError(ibin,_TH1F_Pedestals_m[detid]->GetRMS());
   
    ibin++;

  }
  
  edm::LogInfo("SiPixelCondObjForHLTReader") <<"[SiPixelCondObjForHLTReader::analyze] ---> PIXEL Modules  " << nmodules  << std::endl;
  edm::LogInfo("SiPixelCondObjForHLTReader") <<"[SiPixelCondObjForHLTReader::analyze] ---> PIXEL Channels (i.e. Number of Columns)" << nchannels << std::endl;
  
}
// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelCondObjForHLTReader::beginJob(const edm::EventSetup& iSetup)
{
   //functionality implemented in beginRun
}

// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelCondObjForHLTReader::beginRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelCondObjForHLTReader::endJob() {
  std::cout<<" ---> End job "<<std::endl;
}
}
