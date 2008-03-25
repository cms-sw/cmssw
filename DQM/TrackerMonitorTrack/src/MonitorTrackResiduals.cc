#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/TrackerMonitorTrack/interface/MonitorTrackResiduals.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "DQMServices/Core/interface/DQMStore.h"

MonitorTrackResiduals::MonitorTrackResiduals(const edm::ParameterSet& iConfig) {
  dqmStore_ = edm::Service<DQMStore>().operator->();
  conf_ = iConfig;
}

MonitorTrackResiduals::~MonitorTrackResiduals() { }

void MonitorTrackResiduals::beginJob(edm::EventSetup const& iSetup) {
  using namespace edm;

  Parameters = conf_.getParameter<edm::ParameterSet>("TH1ResModules");
  int32_t i_residuals_Nbins =  Parameters.getParameter<int32_t>("Nbinx");
  double d_residual_xmin = Parameters.getParameter<double>("xmin");
  double d_residual_xmax = Parameters.getParameter<double>("xmax");
  Parameters = conf_.getParameter<edm::ParameterSet>("TH1NormResModules");
  int32_t i_normres_Nbins =  Parameters.getParameter<int32_t>("Nbinx");
  double d_normres_xmin = Parameters.getParameter<double>("xmin");
  double d_normres_xmax = Parameters.getParameter<double>("xmax");

  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  // create SiStripFolderOrganizer
  SiStripFolderOrganizer folder_organizer;
  folder_organizer.setSiStripFolder(); // top SiStrip folder

  // take from eventSetup the SiStripDetCabling object

  edm::ESHandle<SiStripDetCabling> tkmechstruct;
  iSetup.get<SiStripDetCablingRcd>().get(tkmechstruct);

  // get list of active detectors from SiStripDetCabling
  std::vector<uint32_t> activeDets;
  activeDets.clear(); // just in case
  tkmechstruct->addActiveDetectorsRawIds(activeDets);

  // use SiStripSubStructure for selecting certain regions
  SiStripSubStructure substructure;
  std::vector<uint32_t> DetIds = activeDets;
  
  // book histo per each detector module
  for (std::vector<uint32_t>::const_iterator DetItr=activeDets.begin(),  DetItrEnd = activeDets.end(); DetItr!=DetItrEnd; ++DetItr)
    {
      uint ModuleID = (*DetItr);
      
      // is this a StripModule?
      if( SiStripDetId(ModuleID).subDetector() != 0 ) {

	folder_organizer.setDetectorFolder(ModuleID); //  detid sets appropriate detector folder		
	// Book module histogramms? 
	if (conf_.getParameter<bool>("Mod_On")) { 
	  std::string hid = hidmanager.createHistoId("HitResiduals","det",ModuleID);
	  std::string normhid = hidmanager.createHistoId("NormalizedHitResiduals","det",ModuleID);	
	  HitResidual[ModuleID] = dqmStore_->book1D(hid, hid, i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
	  HitResidual[ModuleID]->setAxisTitle("x_{pred} - x_{rec} [cm]");
	  NormedHitResiduals[ModuleID] = dqmStore_->book1D(normhid, normhid, i_normres_Nbins,d_normres_xmin,d_normres_xmax);
	  NormedHitResiduals[ModuleID]->setAxisTitle("x_{pred} - x_{rec}/#sigma");
	}
	
	// book layer level histogramms
	std::pair<std::string,int32_t> subdetandlayer = GetSubDetAndLayer(ModuleID);
	folder_organizer.setLayerFolder(ModuleID,subdetandlayer.second);
	if(! m_SubdetLayerResiduals[subdetandlayer ] ) {
	  // book histogramms on layer level, check for barrel for correct labeling
	  std::string histoname = Form(subdetandlayer.first.find("B") != std::string::npos ? "HitResiduals_%s__Layer_%d" : "HitResiduals_%s__wheel_%d" ,subdetandlayer.first.c_str(),subdetandlayer.second);
	  std::string normhistoname = 
	    Form(subdetandlayer.first.find("B") != std::string::npos ? "NormalizedHitResidual_%s__Layer_%d" : "NormalizedHitResidual_%s__wheel_%d" ,subdetandlayer.first.c_str(),subdetandlayer.second);
	  m_SubdetLayerResiduals[subdetandlayer] = 
	    dqmStore_->book1D(histoname.c_str(),histoname.c_str(),i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
	  m_SubdetLayerResiduals[subdetandlayer]->setAxisTitle("x_{pred} - x_{rec} [cm]");
	  m_SubdetLayerNormedResiduals[subdetandlayer] = 
	    dqmStore_->book1D(normhistoname.c_str(),normhistoname.c_str(),i_normres_Nbins,d_normres_xmin,d_normres_xmax);
	  m_SubdetLayerNormedResiduals[subdetandlayer]->setAxisTitle("x_{pred} - x_{rec} [cm]/#sigma");
	} 
      } // end 'is strip module'
    } // end loop over activeDets
	 
}

void MonitorTrackResiduals::endJob(void){
  //dqmStore_->showDirStructure();
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dqmStore_->save(outputFileName);
  }
}


void MonitorTrackResiduals::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  TrackerValidationVariables avalidator_(iSetup,conf_);
  std::vector<TrackerValidationVariables::AVHitStruct> v_hitstruct;
  avalidator_.fillHitQuantities(iEvent,v_hitstruct);
  for (std::vector<TrackerValidationVariables::AVHitStruct>::const_iterator it = v_hitstruct.begin(),
       itEnd = v_hitstruct.end(); it != itEnd; ++it) {
    uint RawId = it->rawDetId;
    
    // fill if hit belongs to StripDetector and its error is not zero
    if( it->resErrX != 0 && SiStripDetId(RawId).subDetector()  != 0 )  {
      if (conf_.getParameter<bool>("Mod_On") && HitResidual[RawId]) { 
	HitResidual[RawId]->Fill(it->resX);
	NormedHitResiduals[RawId]->Fill(it->resX/it->resErrX);
      }
      if(m_SubdetLayerResiduals[GetSubDetAndLayer(RawId)]) {
	m_SubdetLayerResiduals[GetSubDetAndLayer(RawId)]->Fill(it->resX);
	m_SubdetLayerNormedResiduals[GetSubDetAndLayer(RawId)]->Fill(it->resX/it->resErrX);
      }
    }
  }

}

std::pair<std::string,int32_t> MonitorTrackResiduals::GetSubDetAndLayer(const uint32_t& detid)
{
  std::string cSubDet;
  int32_t layer=0;
  switch(StripSubdetector::SubDetector(StripSubdetector(detid).subdetId()))
    {
    case StripSubdetector::TIB :
      cSubDet="TIB";
      layer=TIBDetId(detid).layer();
      break;
    case StripSubdetector::TOB :
      cSubDet="TOB";
      layer=TOBDetId(detid).layer();
      break;
    case StripSubdetector::TID :
      cSubDet="TID";
      layer=TIDDetId(detid).wheel() * ( TIDDetId(detid).side()==1 ? -1 : +1);
      break;
    case StripSubdetector::TEC :
      cSubDet="TEC";
      layer=TECDetId(detid).wheel() * ( TECDetId(detid).side()==1 ? -1 : +1);
      break;
    default:
      edm::LogWarning("MonitorTrackResiduals") << "WARNING!!! this detid does not belong to tracker" << std::endl;
    }
  return std::make_pair(cSubDet,layer);
}



DEFINE_FWK_MODULE(MonitorTrackResiduals);

