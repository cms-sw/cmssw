#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/TrackerMonitorTrack/interface/MonitorTrackResiduals.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

template<TrackerType pixel_or_strip>
MonitorTrackResidualsBase<pixel_or_strip>::MonitorTrackResidualsBase(const edm::ParameterSet& iConfig)
   : conf_(iConfig), m_cacheID_(0)
   , genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig, consumesCollector(), *this))
   , avalidator_(iConfig, consumesCollector()) {
  ModOn = conf_.getParameter<bool>("Mod_On");
  offlinePrimaryVerticesToken_ = consumes<reco::VertexCollection>(std::string("offlinePrimaryVertices"));
}

template<TrackerType pixel_or_strip>
MonitorTrackResidualsBase<pixel_or_strip>::~MonitorTrackResidualsBase() {
  if (genTriggerEventFlag_) delete genTriggerEventFlag_;
}

template<TrackerType pixel_or_strip>
void MonitorTrackResidualsBase<pixel_or_strip>::bookHistograms(DQMStore::IBooker & ibooker , const edm::Run & run, const edm::EventSetup & iSetup)
{
  unsigned long long cacheID = iSetup.get<TrackerDigiGeometryRecord>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;
    this->createMEs( ibooker , iSetup);
  }
}

template<TrackerType pixel_or_strip>
void MonitorTrackResidualsBase<pixel_or_strip>::dqmBeginRun(edm::Run const& run, edm::EventSetup const& iSetup) {

  // Initialize the GenericTriggerEventFlag
  if ( genTriggerEventFlag_->on() ) genTriggerEventFlag_->initRun( run, iSetup );
}

template<TrackerType pixel_or_strip>
std::pair<std::string, int32_t> MonitorTrackResidualsBase<pixel_or_strip>::findSubdetAndLayer(uint32_t ModuleID, const TrackerTopology* tTopo) {
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
          layer = tTopo->tidWheel(id) * ( tTopo->tidSide(ModuleID)==1 ? -1 : +1);
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


template<TrackerType pixel_or_strip>
void MonitorTrackResidualsBase<pixel_or_strip>::createMEs( DQMStore::IBooker & ibooker , const edm::EventSetup& iSetup){

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
  // book histo per each detector module
  auto ids = TG->detIds(); // or detUnitIds?
  for (DetId id : ids) 
    {
      auto ModuleID = id.rawId();
      auto isPixel = id.subdetId() == 1 || id.subdetId() == 2;
      if (isPixel != (pixel_or_strip == TRACKERTYPE_PIXEL)) continue; 

      // Book module histogramms?
      if (ModOn) {
	switch (id.subdetId()) {
	  case 1:   pixel_organizer.setModuleFolder(ibooker, ModuleID, 0); break;
	  case 2:   pixel_organizer.setModuleFolder(ibooker, ModuleID, 0); break;
	  default:  strip_organizer.setDetectorFolder(ModuleID,tTopo);
	}
	{
	  // this sounds strip specific but also works for pixel
	  std::string hid = hidmanager.createHistoId("HitResidualsX","det",ModuleID);
	  std::string normhid = hidmanager.createHistoId("NormalizedHitResidualsX","det",ModuleID);
	  auto& histos = m_ModuleResiduals[std::make_pair("", ModuleID)];
	  histos.x.base = ibooker.book1D(hid, hid, i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
	  histos.x.base->setAxisTitle("(x_{pred} - x_{rec})' [cm]");
	  histos.x.normed = ibooker.book1D(normhid, normhid, i_normres_Nbins,d_normres_xmin,d_normres_xmax);
	  histos.x.normed->setAxisTitle("(x_{pred} - x_{rec})'/#sigma");
	}{ 
	  std::string hid = hidmanager.createHistoId("HitResidualsY","det",ModuleID);
	  std::string normhid = hidmanager.createHistoId("NormalizedHitResidualsY","det",ModuleID);
	  auto& histos = m_ModuleResiduals[std::make_pair("", ModuleID)];
	  histos.y.base = ibooker.book1D(hid, hid, i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
	  histos.y.base->setAxisTitle("(y_{pred} - y_{rec})' [cm]");
	  histos.y.normed = ibooker.book1D(normhid, normhid, i_normres_Nbins,d_normres_xmin,d_normres_xmax);
	  histos.y.normed->setAxisTitle("(y_{pred} - y_{rec})'/#sigma");
	}
      }

      auto subdetandlayer = findSubdetAndLayer(ModuleID, tTopo);
      if(m_SubdetLayerResiduals.find(subdetandlayer) == m_SubdetLayerResiduals.end()) {
	// add new histograms
	auto& histos = m_SubdetLayerResiduals[subdetandlayer];
	switch (id.subdetId()) {
	  // Pixel Barrel, Endcap
	  // We can't use the folder organizer here (SiPixelActionExecutor.cc#1638 does the same)
	  case 1:   ibooker.setCurrentFolder("Pixel/Barrel"); break;
	  case 2:   ibooker.setCurrentFolder("Pixel/Endcap"); break;
	  // All strip
	  default:  strip_organizer.setLayerFolder(ModuleID,tTopo,subdetandlayer.second);
	}
      
	auto isBarrel = subdetandlayer.first.find("B") != std::string::npos;
	
	auto xy = std::vector<std::pair<HistoPair&, const char*> >
      		{ std::make_pair(std::ref(histos.x), "X"),
	          std::make_pair(std::ref(histos.y), "Y") };
	for (auto& histopair : xy) {
	  // book histogramms on layer level, check for barrel/pixel only for correct labeling

	  // Skip the Y plots for strips.
	  if (!isPixel && histopair.second[0] == 'Y') continue;

	  std::string histoname = isPixel ? ( // Pixel name
	        Form("HitResiduals%s_%s%d",
	        histopair.second,
	        isBarrel ? "L" : (subdetandlayer.second > 0 ? "Dp" : "Dm"),
	        std::abs(subdetandlayer.second)))
	     : (Form("HitResiduals_%s__%s__%d", // Strip TODO: We use a legacy name. 
	        subdetandlayer.first.c_str(),
	        isBarrel ? "Layer" : "wheel",
	        std::abs(subdetandlayer.second)));

	  std::string histotitle = Form("HitResiduals %s on %s%s full %s %d",
		histopair.second,
		subdetandlayer.first.c_str(),
		isBarrel ? "" : (subdetandlayer.second > 0 ? "+" : "-"),
		isBarrel ? "Layer" : (isPixel ? "Disk" : "Wheel"),
		std::abs(subdetandlayer.second));
	

	  std::string normhistoname = Form("Normalized%s", histoname.c_str());
	  std::string normhistotitle = Form("Normalized%s", histotitle.c_str());

	  //std::cout << "##### Booking: " << ibooker.pwd() << " title " << histoname << std::endl;
	  
	  histopair.first.base =
	    ibooker.book1D(histoname.c_str(),histotitle.c_str(),
			 i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
	  histopair.first.base->setAxisTitle("(x_{pred} - x_{rec})' [cm]");

	  histopair.first.normed =
	    ibooker.book1D(normhistoname.c_str(),normhistotitle.c_str(),
			   i_normres_Nbins,d_normres_xmin,d_normres_xmax);
	  histopair.first.normed->setAxisTitle("(x_{pred} - x_{rec})'/#sigma");
	}
      }      
    } // end loop over activeDets
}

template<TrackerType pixel_or_strip>
void MonitorTrackResidualsBase<pixel_or_strip>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  auto vtracks = std::vector<TrackerValidationVariables::AVTrackStruct>();
  // Filter out events if Trigger Filtering is requested
  if (genTriggerEventFlag_->on()&& ! genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(offlinePrimaryVerticesToken_, vertices);
  if (!vertices.isValid() || vertices->size() == 0) return;
  const auto primaryVertex = vertices->at(0); 

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  avalidator_.fillTrackQuantities(iEvent, iSetup, 
                  // tell the validator to only look at good tracks
                  [&](const reco::Track& track) -> bool { 
                    return track.pt() > 0.75
                        && abs( track.dxy(primaryVertex.position()) ) < 5*track.dxyError();
                  }, vtracks);

  for (auto& track : vtracks) {
    for (auto& it : track.hits) {
      uint RawId = it.rawDetId;

      auto id = DetId(RawId);
      auto isPixel = id.subdetId() == 1 || id.subdetId() == 2;
      if (isPixel != (pixel_or_strip == TRACKERTYPE_PIXEL)) continue; 
      

      if (ModOn) {
	auto& mod_histos = m_ModuleResiduals[std::make_pair("",RawId)];
	mod_histos.x.base->Fill(it.resXprime);
	mod_histos.x.normed->Fill(it.resXprime/it.resXprimeErr);
	mod_histos.y.base->Fill(it.resYprime);
	mod_histos.y.normed->Fill(it.resYprime/it.resYprimeErr);
      }

      auto subdetandlayer = findSubdetAndLayer(RawId, tTopo);
      auto histos = m_SubdetLayerResiduals[subdetandlayer];
      // fill if its error is not zero
      if(it.resXprimeErr != 0 && histos.x.base) {
	histos.x.base->Fill(it.resXprime);
	histos.x.normed->Fill(it.resXprime/it.resXprimeErr);
      }
      if(it.resYprimeErr != 0 && histos.y.base) {
	histos.y.base->Fill(it.resYprime);
	histos.y.normed->Fill(it.resYprime/it.resYprimeErr);
      }
    }
  }
}



DEFINE_FWK_MODULE(MonitorTrackResiduals);
DEFINE_FWK_MODULE(SiPixelMonitorTrackResiduals);
 
