#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/TrackerMonitorTrack/interface/MonitorTrackResiduals.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

MonitorTrackResiduals::MonitorTrackResiduals(const edm::ParameterSet& iConfig)
   : dqmStore_( edm::Service<DQMStore>().operator->() )
   , conf_(iConfig), m_cacheID_(0)
   , genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig, consumesCollector(), *this))
   , avalidator_(iConfig, consumesCollector()) {
  ModOn = conf_.getParameter<bool>("Mod_On");
}

MonitorTrackResiduals::~MonitorTrackResiduals() {
  if (genTriggerEventFlag_) delete genTriggerEventFlag_;
}


void MonitorTrackResiduals::beginJob(void) {
}

void MonitorTrackResiduals::bookHistograms(DQMStore::IBooker & ibooker , const edm::Run & run, const edm::EventSetup & iSetup)
{
  unsigned long long cacheID = iSetup.get<SiStripDetCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;
    this->createMEs( ibooker , iSetup);
  }
}

void MonitorTrackResiduals::dqmBeginRun(edm::Run const& run, edm::EventSetup const& iSetup) {

  // Initialize the GenericTriggerEventFlag
  if ( genTriggerEventFlag_->on() ) genTriggerEventFlag_->initRun( run, iSetup );
}

std::pair<std::string, int32_t> MonitorTrackResiduals::findSubdetAndLayer(uint32_t ModuleID, const TrackerTopology* tTopo) {
      std::string subdet = "";
      int32_t layer = 0;
      auto id = DetId(ModuleID);
      switch (id.subdetId()) {
        // Pixel Barrel, Endcap
	case 1:
	  subdet = "BPIX";
          layer = tTopo->pxbLayer(id);
	  break;
        case 2:
	  subdet = "FPIX";
          layer = tTopo->pxfDisk(id) * ( tTopo->pxfSide(ModuleID)==1 ? -1 : +1);
	  break;
	// Strip TIB, TID, TOB, TEC
	case 3:
	  subdet = "TIB";
          layer = tTopo->tibLayer(id);
	  break;
	case 4:
	  subdet = "TID";
          layer = tTopo->tidWheel(id) * ( tTopo->tecSide(ModuleID)==1 ? -1 : +1);
	  break;
	case 5:
	  subdet = "TOB";
          layer = tTopo->tobLayer(id);
	  break;
	case 6:
	  subdet = "TEC";
          layer = tTopo->tecWheel(id) * ( tTopo->tecSide(ModuleID)==1 ? -1 : +1);
	  break;
	default:
	  // TODO: Fail loudly.
	  subdet = "UNKNOWN";
	  layer = 0;
      }
      return std::make_pair(subdet, layer);
}


void MonitorTrackResiduals::createMEs( DQMStore::IBooker & ibooker , const edm::EventSetup& iSetup){

  //Retrieve tracker topology and geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  edm::ESHandle<TrackerGeometry> TG;
  iSetup.get<TrackerDigiGeometryRecord>().get(TG);
 
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

  SiStripFolderOrganizer strip_organizer;
  auto pixel_organizer = SiPixelFolderOrganizer(false);

  // Collect list of modules from Tracker Geometry
  std::vector<uint32_t> activeDets;
  auto ids = TG->detIds(); // or detUnitIds?
  for (DetId id : ids) {
    activeDets.push_back(id.rawId());
  }

  // book histo per each detector module
  for(auto ModuleID : activeDets)
    {
      //uint ModuleID = (*DetItr);

      // TODO: Not yet implemented for Pixel.
      // Book module histogramms?
      if (ModOn) {
	std::string hid = hidmanager.createHistoId("HitResiduals","det",ModuleID);
	std::string normhid = hidmanager.createHistoId("NormalizedHitResiduals","det",ModuleID);
	HitResidual[ModuleID] = ibooker.book1D(hid, hid,
					       i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
	HitResidual[ModuleID]->setAxisTitle("(x_{pred} - x_{rec})' [cm]");
	NormedHitResiduals[ModuleID] = ibooker.book1D(normhid, normhid,
						      i_normres_Nbins,d_normres_xmin,d_normres_xmax);
	NormedHitResiduals[ModuleID]->setAxisTitle("(x_{pred} - x_{rec})'/#sigma");
      }

      auto subdetandlayer = findSubdetAndLayer(ModuleID, tTopo);

      if(! m_SubdetLayerResiduals[subdetandlayer ] ) {
	
	auto id = DetId(ModuleID);
	switch (id.subdetId()) {
	  // Pixel Barrel, Endcap
	  case 1:
	    pixel_organizer.setModuleFolder(ibooker, ModuleID, 2);
	    break;
	  case 2:
	    pixel_organizer.setModuleFolder(ibooker, ModuleID, 6);
	    break;
	  // All strip
	  default:
	    strip_organizer.setLayerFolder(ModuleID,tTopo,subdetandlayer.second);
	}
	
	// book histogramms on layer level, check for barrel for correct labeling
	
	auto isBarrel = subdetandlayer.first.find("B") != std::string::npos;
	std::string histoname = Form("HitResiduals_%s%s__%s__%d",
		subdetandlayer.first.c_str(),
		isBarrel ? "" : (subdetandlayer.second > 0 ? "+" : "-"),
		isBarrel ? "Layer" : "wheel",
		std::abs(subdetandlayer.second));
	
	std::string normhistoname = Form("Normalized%s", histoname.c_str());

	//std::cout << "##### Booking: " << ibooker.pwd() << " title " << histoname << std::endl;
	
	m_SubdetLayerResiduals[subdetandlayer] =
	  ibooker.book1D(histoname.c_str(),histoname.c_str(),
			 i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
	m_SubdetLayerResiduals[subdetandlayer]->setAxisTitle("(x_{pred} - x_{rec})' [cm]");

	m_SubdetLayerNormedResiduals[subdetandlayer] =
	  ibooker.book1D(normhistoname.c_str(),normhistoname.c_str(),
			 i_normres_Nbins,d_normres_xmin,d_normres_xmax);
	m_SubdetLayerNormedResiduals[subdetandlayer]->setAxisTitle("(x_{pred} - x_{rec})'/#sigma");
      }      
    } // end loop over activeDets
}


void MonitorTrackResiduals::endRun(const edm::Run&, const edm::EventSetup&){
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

  // Filter out events if Trigger Filtering is requested
  if (genTriggerEventFlag_->on()&& ! genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  std::vector<TrackerValidationVariables::AVHitStruct> v_hitstruct;
  avalidator_.fillHitQuantities(iEvent,v_hitstruct);
  for (std::vector<TrackerValidationVariables::AVHitStruct>::const_iterator it = v_hitstruct.begin(),
       itEnd = v_hitstruct.end(); it != itEnd; ++it) {
    uint RawId = it->rawDetId;

    // fill if its error is not zero
    if( it->resXprimeErr != 0)  {
      if (ModOn && HitResidual[RawId]) {
	HitResidual[RawId]->Fill(it->resXprime);
	NormedHitResiduals[RawId]->Fill(it->resXprime/it->resXprimeErr);
      }
      auto subdetandlayer = findSubdetAndLayer(RawId, tTopo);
      if(m_SubdetLayerResiduals[subdetandlayer]) {
	m_SubdetLayerResiduals[subdetandlayer]->Fill(it->resXprime);
	m_SubdetLayerNormedResiduals[subdetandlayer]->Fill(it->resXprime/it->resXprimeErr);
      }
    }
  }
}



DEFINE_FWK_MODULE(MonitorTrackResiduals);

