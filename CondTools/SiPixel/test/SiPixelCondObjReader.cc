#include <memory>

#include "CondTools/SiPixel/test/SiPixelCondObjReader.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"


namespace cms{
SiPixelCondObjReader::SiPixelCondObjReader(const edm::ParameterSet& conf): 
    conf_(conf),
    SiPixelGainCalibrationService_(conf)
{
}

void
SiPixelCondObjReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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
  edm::LogInfo("SiPixelCondObjReader") << "[SiPixelCondObjReader::beginJob] End Reading CondObjects" << std::endl;

  // Get the Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get( tkgeom );     
  edm::LogInfo("SiPixelCondObjReader") <<" There are "<<tkgeom->dets().size() <<" detectors"<<std::endl;
  
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
    const PixelGeomDetUnit* _PixelGeomDetUnit = dynamic_cast<const PixelGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
    if (_PixelGeomDetUnit==0){
      edm::LogError("SiPixelCondObjDisplay")<<"[SiPixelCondObjReader::beginJob] the detID "<<detid<<" doesn't seem to belong to Tracker"<<std::endl; 
      continue;
    }

    nmodules++;
    //if(nmodules>3) break;

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
	 _TH1F_Gains_all->Fill(gain);

	float ped  = SiPixelGainCalibrationService_.getPedestal(detid, col_iter, row_iter);
	 _TH1F_Pedestals_m[detid]->Fill( ped );
       	 _TH1F_Pedestals_all->Fill(ped);


	//std::cout << "       Col "<<col_iter<<" Row "<<row_iter<<" Ped "<<ped<<" Gain "<<gain<<std::endl;
	   
      }
    }

    _TH1F_Gains_sum->SetBinContent(ibin,_TH1F_Gains_m[detid]->GetMean());
    _TH1F_Gains_sum->SetBinError(ibin,_TH1F_Gains_m[detid]->GetRMS());
    _TH1F_Pedestals_sum->SetBinContent(ibin,_TH1F_Pedestals_m[detid]->GetMean());
    _TH1F_Pedestals_sum->SetBinError(ibin,_TH1F_Pedestals_m[detid]->GetRMS());

    ibin++;

  }

  
  edm::LogInfo("SiPixelCondObjReader") <<"[SiPixelCondObjReader::analyze] ---> PIXEL Modules  " << nmodules  << std::endl;
  edm::LogInfo("SiPixelCondObjReader") <<"[SiPixelCondObjReader::analyze] ---> PIXEL Channels " << nchannels << std::endl;

}

// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelCondObjReader::beginJob(const edm::EventSetup& iSetup)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelCondObjReader::endJob() {
  std::cout<<" ---> End job "<<std::endl;
}
}
