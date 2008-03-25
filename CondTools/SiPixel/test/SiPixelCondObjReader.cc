#include <memory>

#include "CondTools/SiPixel/test/SiPixelCondObjReader.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
namespace cms{
SiPixelCondObjReader::SiPixelCondObjReader(const edm::ParameterSet& conf): 
    conf_(conf),
    filename_(conf.getParameter<std::string>("fileName")),
    SiPixelGainCalibrationService_(conf)
{
}

void
SiPixelCondObjReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  unsigned int nmodules = 0;
  uint32_t nchannels = 0;
  fFile->cd();

  // Get the Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get( tkgeom );     
  edm::LogInfo("SiPixelCondObjReader") <<" There are "<<tkgeom->dets().size() <<" detectors"<<std::endl;
  
  // Get the calibration data
  //iSetup.get<SiPixelGainCalibrationRcd>().get(SiPixelGainCalibration_);
  //edm::LogInfo("SiPixelCondObjReader") << "[SiPixelCondObjReader::analyze] End Reading CondObjects" << std::endl;
  //SiPixelGainCalibrationService_.setESObjects(iSetup);

  //  for(TrackerGeometry::DetContainer::const_iterator it = tkgeom->dets().begin(); it != tkgeom->dets().end(); it++){
  //   if( dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
  //     uint32_t detid=((*it)->geographicalId()).rawId();
  // Get the list of DetId's
  
  std::vector<uint32_t> vdetId_ = SiPixelGainCalibrationService_.getDetIds();
  // Loop over DetId's
  for (std::vector<uint32_t>::const_iterator detid_iter=vdetId_.begin();detid_iter!=vdetId_.end();detid_iter++){
    uint32_t detid = *detid_iter;

    DetId detIdObject(detid);
    nmodules++;
    //if(nmodules>3) break;

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

	float ped  = SiPixelGainCalibrationService_.getPedestal(detid, col_iter, row_iter);
	p_iter->second->Fill( ped );

	float gain  = SiPixelGainCalibrationService_.getGain(detid, col_iter, row_iter);
	g_iter->second->Fill( gain );

	//std::cout << "       Col "<<col_iter<<" Row "<<row_iter<<" Ped "<<ped<<" Gain "<<gain<<std::endl;
	   
      }
    }
  }
  
  edm::LogInfo("SiPixelCondObjReader") <<"[SiPixelCondObjReader::analyze] ---> PIXEL Modules  " << nmodules  << std::endl;
  edm::LogInfo("SiPixelCondObjReader") <<"[SiPixelCondObjReader::analyze] ---> PIXEL Channels " << nchannels << std::endl;

  fFile->cd();
  _TH1F_Gains_sum = new TH1F("Summary_Gain","Gain Summary", vdetId_.size()+1,0,vdetId_.size()+1);
  _TH1F_Pedestals_sum = new TH1F("Summary_Pedestal","Pedestal Summary", vdetId_.size()+1,0,vdetId_.size()+1);
 
  // Loop over DetId's
  int ibin=1;
  for (std::vector<uint32_t>::const_iterator detid_iter=vdetId_.begin();detid_iter!=vdetId_.end();detid_iter++){
    uint32_t detid = *detid_iter;

    DetId detIdObject(detid);

    std::map<uint32_t,TH1F*>::iterator p_iter =  _TH1F_Pedestals_m.find(detid);
    std::map<uint32_t,TH1F*>::iterator g_iter =  _TH1F_Gains_m.find(detid);
    _TH1F_Gains_sum->SetBinContent(ibin,g_iter->second->GetMean());
    _TH1F_Gains_sum->SetBinError(ibin,g_iter->second->GetRMS());
    _TH1F_Pedestals_sum->SetBinContent(ibin,p_iter->second->GetMean());
    _TH1F_Pedestals_sum->SetBinError(ibin,p_iter->second->GetRMS());
    if(ibin==1){
      fFile->cd();
      _TH1F_Pedestals_all = (TH1F*) p_iter->second->Clone("PedestalsAll");
      _TH1F_Gains_all = (TH1F*) g_iter->second->Clone("GainAll");
    }
    else{
      _TH1F_Pedestals_all->Add(p_iter->second,1.);
      _TH1F_Gains_all->Add(g_iter->second,1.);
    }
    ibin++;
  }
  fFile->ls();
  fFile->Write();
  fFile->Close();    
}

// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelCondObjReader::beginJob(const edm::EventSetup& iSetup)
{

  edm::LogInfo("SiPixelCondObjReader") <<"[SiPixelCondObjReader::beginJob] Opening ROOT file  " <<std::endl;
  fFile = new TFile(filename_.c_str(),"RECREATE");
  fFile->mkdir("Pedestals");
  fFile->mkdir("Gains");
  fFile->cd();
  char name[128];

  // Get Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get( tkgeom );

  // Get the calibration data
  //edm::ESHandle<SiPixelGainCalibration> SiPixelGainCalibration_;
  //iSetup.get<SiPixelGainCalibrationRcd>().get(SiPixelGainCalibration_);
  SiPixelGainCalibrationService_.setESObjects(iSetup);
  edm::LogInfo("SiPixelCondObjReader") << "[SiPixelCondObjReader::beginJob] End Reading CondObjects" << std::endl;
  // Get the list of DetId's
  std::vector<uint32_t> vdetId_ = SiPixelGainCalibrationService_.getDetIds();
  //SiPixelGainCalibration_->getDetIds(vdetId_);
  // Loop over DetId's
  for (std::vector<uint32_t>::const_iterator detid_iter=vdetId_.begin();detid_iter!=vdetId_.end();detid_iter++){
    uint32_t detid = *detid_iter;
    
    const PixelGeomDetUnit* _PixelGeomDetUnit = dynamic_cast<const PixelGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
    if (_PixelGeomDetUnit==0){
      edm::LogError("SiPixelCondObjDisplay")<<"[SiPixelCondObjReader::beginJob] the detID "<<detid<<" doesn't seem to belong to Tracker"<<std::endl; 
      continue;
    }     
    // Book histograms
    sprintf(name,"Pedestals_%d",detid);
    fFile->cd();fFile->cd("Pedestals");
    _TH1F_Pedestals_m[detid] = new TH1F(name,name,50,0.,50.);    
    sprintf(name,"Gains_%d",detid);
    fFile->cd();fFile->cd("Gains");
    _TH1F_Gains_m[detid] = new TH1F(name,name,100,0.,10.);    
  }
  
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelCondObjReader::endJob() {
  std::cout<<" ---> End job "<<std::endl;
}
}
