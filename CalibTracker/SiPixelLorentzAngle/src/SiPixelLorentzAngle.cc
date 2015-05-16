
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <TMath.h>

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "SiPixelLorentzAngle.h"
int lower_bin_;

using namespace std;
using namespace edm;
using namespace reco;
using namespace analyzer;

SiPixelLorentzAngle::SiPixelLorentzAngle(edm::ParameterSet const& conf) : 
  filename_(conf.getParameter<std::string>("fileName")), filenameFit_(conf.getParameter<std::string>("fileNameFit")), ptmin_(conf.getParameter<double>("ptMin")), simData_(conf.getParameter<bool>("simData")),	normChi2Max_(conf.getParameter<double>("normChi2Max")), clustSizeYMin_(conf.getParameter<int>("clustSizeYMin")), residualMax_(conf.getParameter<double>("residualMax")), clustChargeMax_(conf.getParameter<double>("clustChargeMax")),hist_depth_(conf.getParameter<int>("binsDepth")), hist_drift_(conf.getParameter<int>("binsDrift")), trackerHitAssociatorConfig_(consumesCollector())
{
  //   	anglefinder_=new  TrackLocalAngle(conf);
  hist_x_ = 50;
  hist_y_ = 100;
  min_x_ = -500.;
  max_x_ = 500.;
  min_y_ = -1500.;
  max_y_ = 500.;
  width_ = 0.0285;
  min_depth_ = -100.;
  max_depth_ = 400.;
  min_drift_ = -1000.; //-200.;(conf.getParameter<double>("residualMax"))
  max_drift_ = 1000.; //400.;

  t_trajTrack = consumes<TrajTrackAssociationCollection> (conf.getParameter<edm::InputTag>("src"));

}

// Virtual destructor needed.
SiPixelLorentzAngle::~SiPixelLorentzAngle() {  }  

void SiPixelLorentzAngle::beginJob()
{

  // 	cout << "started SiPixelLorentzAngle" << endl;
  hFile_ = new TFile (filename_.c_str(), "RECREATE" );
  int bufsize = 64000;
  // create tree structure
  SiPixelLorentzAngleTree_ = new TTree("SiPixelLorentzAngleTree_","SiPixel LorentzAngle tree", bufsize);
  SiPixelLorentzAngleTree_->Branch("run", &run_, "run/I", bufsize);
  SiPixelLorentzAngleTree_->Branch("event", &event_, "event/I", bufsize);
  SiPixelLorentzAngleTree_->Branch("module", &module_, "module/I", bufsize);
  SiPixelLorentzAngleTree_->Branch("ladder", &ladder_, "ladder/I", bufsize);
  SiPixelLorentzAngleTree_->Branch("layer", &layer_, "layer/I", bufsize);
  SiPixelLorentzAngleTree_->Branch("isflipped", &isflipped_, "isflipped/I", bufsize);
  SiPixelLorentzAngleTree_->Branch("pt", &pt_, "pt/F", bufsize);
  SiPixelLorentzAngleTree_->Branch("eta", &eta_, "eta/F", bufsize);
  SiPixelLorentzAngleTree_->Branch("phi", &phi_, "phi/F", bufsize);
  SiPixelLorentzAngleTree_->Branch("chi2", &chi2_, "chi2/D", bufsize);
  SiPixelLorentzAngleTree_->Branch("ndof", &ndof_, "ndof/D", bufsize);
  SiPixelLorentzAngleTree_->Branch("trackhit", &trackhit_, "x/F:y/F:alpha/D:beta/D:gamma_/D", bufsize);
  SiPixelLorentzAngleTree_->Branch("simhit", &simhit_, "x/F:y/F:alpha/D:beta/D:gamma_/D", bufsize);
  SiPixelLorentzAngleTree_->Branch("npix", &pixinfo_.npix, "npix/I", bufsize);
  SiPixelLorentzAngleTree_->Branch("rowpix", pixinfo_.row, "row[npix]/F", bufsize);
  SiPixelLorentzAngleTree_->Branch("colpix", pixinfo_.col, "col[npix]/F", bufsize);
  SiPixelLorentzAngleTree_->Branch("adc", pixinfo_.adc, "adc[npix]/F", bufsize);
  SiPixelLorentzAngleTree_->Branch("xpix", pixinfo_.x, "x[npix]/F", bufsize);
  SiPixelLorentzAngleTree_->Branch("ypix", pixinfo_.y, "y[npix]/F", bufsize);
  SiPixelLorentzAngleTree_->Branch("clust", &clust_, "x/F:y/F:charge/F:size_x/I:size_y/I:maxPixelCol/I:maxPixelRow:minPixelCol/I:minPixelRow/I", bufsize);
  SiPixelLorentzAngleTree_->Branch("rechit", &rechit_, "x/F:y/F", bufsize);
	
  SiPixelLorentzAngleTreeForward_ = new TTree("SiPixelLorentzAngleTreeForward_","SiPixel LorentzAngle tree forward", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("run", &run_, "run/I", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("event", &event_, "event/I", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("side", &sideF_, "side/I", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("disk", &diskF_, "disk/I", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("blade", &bladeF_, "blade/I", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("panel", &panelF_, "panel/I", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("module", &moduleF_, "module/I", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("pt", &pt_, "pt/F", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("eta", &eta_, "eta/F", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("phi", &phi_, "phi/F", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("chi2", &chi2_, "chi2/D", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("ndof", &ndof_, "ndof/D", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("trackhit", &trackhitF_, "x/F:y/F:alpha/D:beta/D:gamma_/D", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("simhit", &simhitF_, "x/F:y/F:alpha/D:beta/D:gamma_/D", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("npix", &pixinfoF_.npix, "npix/I", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("rowpix", pixinfoF_.row, "row[npix]/F", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("colpix", pixinfoF_.col, "col[npix]/F", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("adc", pixinfoF_.adc, "adc[npix]/F", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("xpix", pixinfoF_.x, "x[npix]/F", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("ypix", pixinfoF_.y, "y[npix]/F", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("clust", &clustF_, "x/F:y/F:charge/F:size_x/I:size_y/I:maxPixelCol/I:maxPixelRow:minPixelCol/I:minPixelRow/I", bufsize);
  SiPixelLorentzAngleTreeForward_->Branch("rechit", &rechitF_, "x/F:y/F", bufsize);
	
	
  //book histograms
  char name[128];
  for(int i_module = 1; i_module<=8; i_module++){
    for(int i_layer = 1; i_layer<=3; i_layer++){
      sprintf(name, "h_drift_depth_adc_layer%i_module%i", i_layer, i_module); 
      _h_drift_depth_adc_[i_module + (i_layer -1) * 8] = new TH2F(name,name,hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
      sprintf(name, "h_drift_depth_adc2_layer%i_module%i", i_layer, i_module); 
      _h_drift_depth_adc2_[i_module + (i_layer -1) * 8] = new TH2F(name,name,hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
      sprintf(name, "h_drift_depth_noadc_layer%i_module%i", i_layer, i_module); 
      _h_drift_depth_noadc_[i_module + (i_layer -1) * 8] = new TH2F(name,name,hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
      sprintf(name, "h_drift_depth_layer%i_module%i", i_layer, i_module); 
      _h_drift_depth_[i_module + (i_layer -1) * 8] = new TH2F(name,name,hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
      sprintf(name, "h_mean_layer%i_module%i", i_layer, i_module); 
      _h_mean_[i_module + (i_layer -1) * 8] = new TH1F(name,name,hist_depth_, min_depth_, max_depth_);
    }
  }
	
  // just for some expaining plots
  h_cluster_shape_adc_  = new TH2F("h_cluster_shape_adc","cluster shape with adc weight", hist_x_, min_x_, max_x_, hist_y_, min_y_, max_y_);
  h_cluster_shape_noadc_  = new TH2F("h_cluster_shape_noadc","cluster shape without adc weight", hist_x_, min_x_, max_x_, hist_y_, min_y_, max_y_);
  h_cluster_shape_  = new TH2F("h_cluster_shape","cluster shape", hist_x_, min_x_, max_x_, hist_y_, min_y_, max_y_);
  h_cluster_shape_adc_rot_  = new TH2F("h_cluster_shape_adc_rot","cluster shape with adc weight", hist_x_, min_x_, max_x_, hist_y_, -max_y_, -min_y_);
  h_cluster_shape_noadc_rot_  = new TH2F("h_cluster_shape_noadc_rot","cluster shape without adc weight", hist_x_, min_x_, max_x_, hist_y_, -max_y_, -min_y_);
  h_cluster_shape_rot_  = new TH2F("h_cluster_shape_rot","cluster shape", hist_x_, min_x_, max_x_, hist_y_, -max_y_, -min_y_);
  h_tracks_ = new TH1F("h_tracks","h_tracks",2,0.,2.);
  event_counter_ = 0;
  trackEventsCounter_ = 0;
  // 	trackcounter_ = 0;
  hitCounter_ = 0;
  usedHitCounter_ = 0;
  pixelTracksCounter_ = 0;
//   edm::ESHandle<TrackerGeometry> estracker; //this block should not be in beginJob()
//   c.get<TrackerDigiGeometryRecord>().get(estracker);
//   tracker=&(* estracker);
}


// Functions that gets called by framework every event
void SiPixelLorentzAngle::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  
  event_counter_++;
  // 	if(event_counter_ % 500 == 0) cout << "event number " << event_counter_ << endl;
  cout << "event number " << event_counter_ << endl;

  edm::ESHandle<TrackerGeometry> estracker;
  es.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(* estracker);

  std::unique_ptr<TrackerHitAssociator> associate;
  if (simData_) associate.reset(new TrackerHitAssociator(e, trackerHitAssociatorConfig_));
  // restet values
  module_=-1;
  layer_=-1;
  ladder_ = -1;
  isflipped_ = -1;
  pt_ = -999;
  eta_ = 999;
  phi_ = 999;
  pixinfo_.npix = 0;

  run_       = e.id().run();
  event_     = e.id().event();
	
  // get the association map between tracks and trajectories
  edm::Handle<TrajTrackAssociationCollection> trajTrackCollectionHandle;
  e.getByToken(t_trajTrack,trajTrackCollectionHandle);
  if(trajTrackCollectionHandle->size() >0){
    trackEventsCounter_++;
    for(TrajTrackAssociationCollection::const_iterator it = trajTrackCollectionHandle->begin(); it!=trajTrackCollectionHandle->end();++it){
      const Track&      track = *it->val;
      const Trajectory& traj  = *it->key;
		
      // get the trajectory measurements
      std::vector<TrajectoryMeasurement> tmColl = traj.measurements(); 
      // 			TrajectoryStateOnSurface tsos = tsoscomb( itTraj->forwardPredictedState(), itTraj->backwardPredictedState() );
      pt_ = track.pt();
      eta_ = track.eta();
      phi_ = track.phi();
      chi2_ = traj.chiSquared();
      ndof_ = traj.ndof();
      if(pt_ < ptmin_) continue;
      // iterate over trajectory measurements
      std::vector<PSimHit> matched;
      h_tracks_->Fill(0);
      bool pixeltrack = false;
      for(std::vector<TrajectoryMeasurement>::const_iterator itTraj = tmColl.begin(); itTraj != tmColl.end(); itTraj++) {
	if(! itTraj->updatedState().isValid()) continue;
	TransientTrackingRecHit::ConstRecHitPointer recHit = itTraj->recHit();
	if(! recHit->isValid() || recHit->geographicalId().det() != DetId::Tracker ) continue;
	unsigned int subDetID = (recHit->geographicalId().subdetId());
	if( subDetID == PixelSubdetector::PixelBarrel || subDetID == PixelSubdetector::PixelEndcap){
	  if(!pixeltrack){
	    h_tracks_->Fill(1);
	    pixelTracksCounter_++;
	  }
	  pixeltrack = true;
	}
	if( subDetID == PixelSubdetector::PixelBarrel){
				
	  hitCounter_++;
				
	  DetId detIdObj = recHit->geographicalId();
	  const PixelGeomDetUnit * theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(detIdObj) );
	  if(!theGeomDet) continue;

	  const PixelTopology * topol = &(theGeomDet->specificTopology());

	  if(!topol) continue;
	  
	  layer_ = tTopo->pxbLayer(detIdObj);
	  ladder_ = tTopo->pxbLadder(detIdObj);
	  module_ = tTopo->pxbModule(detIdObj);
	  float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
	  float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
	  if ( tmp2<tmp1 ) isflipped_ = 1;
	  else isflipped_ = 0;
	  const SiPixelRecHit * recHitPix = dynamic_cast<const SiPixelRecHit *>((*recHit).hit());
	  if(!recHitPix) continue;
	  rechit_.x  = recHitPix->localPosition().x();
	  rechit_.y  = recHitPix->localPosition().y();
	  SiPixelRecHit::ClusterRef const& cluster = recHitPix->cluster();	
	
	  // fill entries in clust_
	  clust_.x = (cluster)->x();
	  clust_.y = (cluster)->y();
	  clust_.charge = (cluster->charge())/1000.;
	  clust_.size_x = cluster->sizeX();
	  clust_.size_y = cluster->sizeY();
	  clust_.maxPixelCol = cluster->maxPixelCol();
	  clust_.maxPixelRow = cluster->maxPixelRow();
	  clust_.minPixelCol = cluster->minPixelCol();
	  clust_.minPixelRow = cluster->minPixelRow();
	  // fill entries in pixinfo_:
	  fillPix(*cluster ,topol, pixinfo_);
	  // fill the trackhit info
	  TrajectoryStateOnSurface tsos=itTraj->updatedState();
	  if(!tsos.isValid()){
	    cout << "tsos not valid" << endl;
	    continue;	
	  }	
	  LocalVector trackdirection=tsos.localDirection();
	  LocalPoint trackposition=tsos.localPosition();
	
	  if(trackdirection.z()==0) continue;				
	  // the local position and direction
	  trackhit_.alpha = atan2(trackdirection.z(),trackdirection.x());
	  trackhit_.beta = atan2(trackdirection.z(),trackdirection.y());
	  trackhit_.gamma = atan2(trackdirection.x(),trackdirection.y());
	  trackhit_.x = trackposition.x();
	  trackhit_.y = trackposition.y();
					
	
	  // fill entries in simhit_:	
	  if(simData_){
	    matched.clear();        
	    matched = associate->associateHit((*recHitPix));	
	    float dr_start=9999.;
	    for (std::vector<PSimHit>::iterator isim = matched.begin(); isim != matched.end(); ++isim){
	      DetId simdetIdObj((*isim).detUnitId());
	      if (simdetIdObj == detIdObj) {
		float sim_x1 = (*isim).entryPoint().x(); // width (row index, in col direction)
		float sim_y1 = (*isim).entryPoint().y(); // length (col index, in row direction)
		float sim_x2 = (*isim).exitPoint().x();
		float sim_y2 = (*isim).exitPoint().y();
		float sim_xpos = 0.5*(sim_x1+sim_x2);
		float sim_ypos = 0.5*(sim_y1+sim_y2);
		float sim_px = (*isim).momentumAtEntry().x();
		float sim_py = (*isim).momentumAtEntry().y();
		float sim_pz = (*isim).momentumAtEntry().z();
						
		float dr = (sim_xpos-(recHitPix->localPosition().x()))*(sim_xpos-recHitPix->localPosition().x()) +
		  (sim_ypos-recHitPix->localPosition().y())*(sim_ypos-recHitPix->localPosition().y());
		if(dr<dr_start) {
		  simhit_.x     = sim_xpos;
		  simhit_.y     = sim_ypos;
		  simhit_.alpha = atan2(sim_pz, sim_px);
		  simhit_.beta  = atan2(sim_pz, sim_py);
		  simhit_.gamma = atan2(sim_px, sim_py);
		  dr_start = dr;
		}
	      }
	    } // end of filling simhit_
	  }
	  // is one pixel in cluster a large pixel ? (hit will be excluded)
	  bool large_pix = false;
	  for (int j = 0; j <  pixinfo_.npix; j++){
	    int colpos = static_cast<int>(pixinfo_.col[j]);
	    if (pixinfo_.row[j] == 0 || pixinfo_.row[j] == 79 || pixinfo_.row[j] == 80 || pixinfo_.row[j] == 159 || colpos % 52 == 0 || colpos % 52 == 51 ){
	      large_pix = true;	
	    }
	  }
					
	  double residual = TMath::Sqrt( (trackhit_.x - rechit_.x) * (trackhit_.x - rechit_.x) + (trackhit_.y - rechit_.y) * (trackhit_.y - rechit_.y) );
	
	  SiPixelLorentzAngleTree_->Fill();
	  if( !large_pix && (chi2_/ndof_) < normChi2Max_ && cluster->sizeY() >= clustSizeYMin_ && residual < residualMax_ && (cluster->charge() < clustChargeMax_)){
	    usedHitCounter_++;
	    // iterate over pixels in hit
	    for (int j = 0; j <  pixinfo_.npix; j++){
	      // use trackhits
	      float dx = (pixinfo_.x[j]  - (trackhit_.x - width_/2. / TMath::Tan(trackhit_.alpha))) * 10000.;
	      float dy = (pixinfo_.y[j]  - (trackhit_.y - width_/2. / TMath::Tan(trackhit_.beta))) * 10000.;
	      float depth = dy * tan(trackhit_.beta);
	      float drift = dx - dy * tan(trackhit_.gamma);
	      _h_drift_depth_adc_[module_ + (layer_ -1) * 8]->Fill(drift, depth, pixinfo_.adc[j]);
	      _h_drift_depth_adc2_[module_ + (layer_ -1) * 8]->Fill(drift, depth, pixinfo_.adc[j]*pixinfo_.adc[j]);
	      _h_drift_depth_noadc_[module_ + (layer_ -1) * 8]->Fill(drift, depth);		
	      if( layer_ == 3 && module_==1 && isflipped_){
		float dx_rot = dx * TMath::Cos(trackhit_.gamma) + dy * TMath::Sin(trackhit_.gamma);
		float dy_rot = dy * TMath::Cos(trackhit_.gamma) - dx * TMath::Sin(trackhit_.gamma) ;
		h_cluster_shape_adc_->Fill(dx, dy, pixinfo_.adc[j]);
		h_cluster_shape_noadc_->Fill(dx, dy);
		h_cluster_shape_adc_rot_->Fill(dx_rot, dy_rot, pixinfo_.adc[j]);
		h_cluster_shape_noadc_rot_->Fill(dx_rot, dy_rot);
	      }
	    }
	  }
	} else if (subDetID == PixelSubdetector::PixelEndcap) {
	  DetId detIdObj = recHit->geographicalId();
	  const PixelGeomDetUnit * theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(detIdObj) );
	  if(!theGeomDet) continue;
	  
	  const PixelTopology * topol = &(theGeomDet->specificTopology());

	  if(!topol) continue;
	  
	  sideF_ = tTopo->pxfSide(detIdObj);
	  diskF_ = tTopo->pxfDisk(detIdObj);
	  bladeF_ = tTopo->pxfBlade(detIdObj);
	  panelF_ = tTopo->pxfPanel(detIdObj);
	  moduleF_ = tTopo->pxfModule(detIdObj);
	  // 					float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
	  // 					float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
	  // 					if ( tmp2<tmp1 ) isflipped_ = 1;
	  // 					else isflipped_ = 0;
	  const SiPixelRecHit * recHitPix = dynamic_cast<const SiPixelRecHit *>((*recHit).hit());
	  if(!recHitPix) continue;
	  rechitF_.x  = recHitPix->localPosition().x();
	  rechitF_.y  = recHitPix->localPosition().y();
	  SiPixelRecHit::ClusterRef const& cluster = recHitPix->cluster();	
	
	  // fill entries in clust_
	  clustF_.x = (cluster)->x();
	  clustF_.y = (cluster)->y();
	  clustF_.charge = (cluster->charge())/1000.;
	  clustF_.size_x = cluster->sizeX();
	  clustF_.size_y = cluster->sizeY();
	  clustF_.maxPixelCol = cluster->maxPixelCol();
	  clustF_.maxPixelRow = cluster->maxPixelRow();
	  clustF_.minPixelCol = cluster->minPixelCol();
	  clustF_.minPixelRow = cluster->minPixelRow();
	  // fill entries in pixinfo_:
	  fillPix(*cluster ,topol, pixinfoF_);
	  // fill the trackhit info
	  TrajectoryStateOnSurface tsos=itTraj->updatedState();
	  if(!tsos.isValid()){
	    cout << "tsos not valid" << endl;
	    continue;	
	  }	
	  LocalVector trackdirection=tsos.localDirection();
	  LocalPoint trackposition=tsos.localPosition();
	
	  if(trackdirection.z()==0) continue;				
	  // the local position and direction
	  trackhitF_.alpha = atan2(trackdirection.z(),trackdirection.x());
	  trackhitF_.beta = atan2(trackdirection.z(),trackdirection.y());
	  trackhitF_.gamma = atan2(trackdirection.x(),trackdirection.y());
	  trackhitF_.x = trackposition.x();
	  trackhitF_.y = trackposition.y();
					
	
	  // fill entries in simhit_:	
	  if(simData_){
	    matched.clear();        
	    matched = associate->associateHit((*recHitPix));	
	    float dr_start=9999.;
	    for (std::vector<PSimHit>::iterator isim = matched.begin(); isim != matched.end(); ++isim){
	      DetId simdetIdObj((*isim).detUnitId());
	      if (simdetIdObj == detIdObj) {
		float sim_x1 = (*isim).entryPoint().x(); // width (row index, in col direction)
		float sim_y1 = (*isim).entryPoint().y(); // length (col index, in row direction)
		float sim_x2 = (*isim).exitPoint().x();
		float sim_y2 = (*isim).exitPoint().y();
		float sim_xpos = 0.5*(sim_x1+sim_x2);
		float sim_ypos = 0.5*(sim_y1+sim_y2);
		float sim_px = (*isim).momentumAtEntry().x();
		float sim_py = (*isim).momentumAtEntry().y();
		float sim_pz = (*isim).momentumAtEntry().z();
						
		float dr = (sim_xpos-(recHitPix->localPosition().x()))*(sim_xpos-recHitPix->localPosition().x()) +
		  (sim_ypos-recHitPix->localPosition().y())*(sim_ypos-recHitPix->localPosition().y());
		if(dr<dr_start) {
		  simhitF_.x     = sim_xpos;
		  simhitF_.y     = sim_ypos;
		  simhitF_.alpha = atan2(sim_pz, sim_px);
		  simhitF_.beta  = atan2(sim_pz, sim_py);
		  simhitF_.gamma = atan2(sim_px, sim_py);
		  dr_start = dr;
		}
	      }
	    } // end of filling simhit_
	  }
	  SiPixelLorentzAngleTreeForward_->Fill();
	}
      }	//end iteration over trajectory measurements
    } //end iteration over trajectories
  }

	
	
	
	
}

void SiPixelLorentzAngle::endJob()
{
  // produce histograms with the average adc counts
  for(int i_ring = 1; i_ring<=24; i_ring++){
    _h_drift_depth_[i_ring]->Divide(_h_drift_depth_adc_[i_ring], _h_drift_depth_noadc_[i_ring]);
  }
	
  h_drift_depth_adc_slice_ = new TH1F("h_drift_depth_adc_slice", "slice of adc histogram", hist_drift_ , min_drift_, max_drift_);

  TF1 *f1 = new TF1("f1","[0] + [1]*x",50., 235.); 
  f1->SetParName(0,"p0");
  f1->SetParName(1,"p1");
  f1->SetParameter(0,0);
  f1->SetParameter(1,0.4);
  ofstream fLorentzFit( filenameFit_.c_str(), ios::trunc );
  cout.precision( 4 );
  fLorentzFit << "module" << "\t" << "layer" << "\t" << "offset" << "\t" << "error" << "\t" << "slope" << "\t" << "error" << "\t" "rel.err" << "\t" "pull" << "\t" << "chi2" << "\t" << "prob" << endl;
  //loop over modlues and layers to fit the lorentz angle
  for( int i_layer = 1; i_layer<=3; i_layer++){
    for(int i_module = 1; i_module<=8; i_module++){
      //loop over bins in depth (z-local-coordinate) (in order to fit slices)
      for( int i = 1; i <= hist_depth_; i++){
	findMean(i, (i_module + (i_layer - 1) * 8));	
      }// end loop over bins in depth 
      _h_mean_[i_module + (i_layer - 1) * 8]->Fit(f1,"ERQ");
      double p0 = f1->GetParameter(0);
      double e0 = f1->GetParError(0);
      double p1 = f1->GetParameter(1);
      double e1 = f1->GetParError(1);
      double chi2 = f1->GetChisquare();
      double prob = f1->GetProb();
      fLorentzFit << setprecision( 4 ) << i_module << "\t" << i_layer << "\t" << p0 << "\t" << e0 << "\t" << p1 << setprecision( 3 ) << "\t" << e1 << "\t" << e1 / p1 *100. << "\t"<< (p1 - 0.424) / e1 << "\t"<< chi2 << "\t" << prob << endl;
    }
  } // end loop over modules and layers
  fLorentzFit.close(); 
  hFile_->cd();
  for(int i_module = 1; i_module<=8; i_module++){
    for(int i_layer = 1; i_layer<=3; i_layer++){
      _h_drift_depth_adc_[i_module + (i_layer -1) * 8]->Write();
      _h_drift_depth_adc2_[i_module + (i_layer -1) * 8]->Write();
      _h_drift_depth_noadc_[i_module + (i_layer -1) * 8]->Write();
      _h_drift_depth_[i_module + (i_layer -1) * 8]->Write();
      _h_mean_[i_module + (i_layer -1) * 8]->Write();
    }
  }
  h_cluster_shape_adc_->Write();
  h_cluster_shape_noadc_->Write();
  h_cluster_shape_adc_rot_->Write();
  h_cluster_shape_noadc_rot_->Write();
  h_tracks_->Write();
	
  hFile_->Write();
  hFile_->Close();
  cout << "events: " << event_counter_ << endl;
  cout << "events with tracks: " << trackEventsCounter_ << endl;	
  cout << "events with pixeltracks: " << pixelTracksCounter_ << endl;	
  cout << "hits in the pixel: " << hitCounter_ << endl;
  cout << "number of used Hits: " << usedHitCounter_ << endl;
}

inline void SiPixelLorentzAngle::fillPix(const SiPixelCluster & LocPix, const PixelTopology * topol, Pixinfo& pixinfo)

{
  const std::vector<SiPixelCluster::Pixel>& pixvector = LocPix.pixels();
  pixinfo.npix = 0;
  for(std::vector<SiPixelCluster::Pixel>::const_iterator itPix = pixvector.begin(); itPix != pixvector.end(); itPix++){
    // 	for(pixinfo.npix = 0; pixinfo.npix < static_cast<int>(pixvector.size()); ++pixinfo.npix) {
    pixinfo.row[pixinfo.npix] = itPix->x;
    pixinfo.col[pixinfo.npix] = itPix->y;
    pixinfo.adc[pixinfo.npix] = itPix->adc;
    LocalPoint lp = topol->localPosition(MeasurementPoint(itPix->x + 0.5, itPix->y+0.5));
    pixinfo.x[pixinfo.npix] = lp.x();
    pixinfo.y[pixinfo.npix]= lp.y();
    pixinfo.npix++;
  }
}

void SiPixelLorentzAngle::findMean(int i, int i_ring)
{
  double nentries = 0;
	
  h_drift_depth_adc_slice_->Reset("ICE");
	
  // determine sigma and sigma^2 of the adc counts and average adc counts
  //loop over bins in drift width
  for( int j = 1; j<= hist_drift_; j++){
    if(_h_drift_depth_noadc_[i_ring]->GetBinContent(j, i) >= 1){
      double adc_error2 = (_h_drift_depth_adc2_[i_ring]->GetBinContent(j,i) - _h_drift_depth_adc_[i_ring]->GetBinContent(j,i)*_h_drift_depth_adc_[i_ring]->GetBinContent(j, i) / _h_drift_depth_noadc_[i_ring]->GetBinContent(j, i)) /  _h_drift_depth_noadc_[i_ring]->GetBinContent(j, i);
      _h_drift_depth_adc_[i_ring]->SetBinError(j, i, sqrt(adc_error2));
      double error2 = adc_error2 / (_h_drift_depth_noadc_[i_ring]->GetBinContent(j,i) - 1.);
      _h_drift_depth_[i_ring]->SetBinError(j,i,sqrt(error2));
    } 
    else{
      _h_drift_depth_[i_ring]->SetBinError(j,i,0);
      _h_drift_depth_adc_[i_ring]->SetBinError(j, i, 0);
    }
    h_drift_depth_adc_slice_->SetBinContent(j, _h_drift_depth_adc_[i_ring]->GetBinContent(j,i));
    h_drift_depth_adc_slice_->SetBinError(j, _h_drift_depth_adc_[i_ring]->GetBinError(j,i));
    nentries += _h_drift_depth_noadc_[i_ring]->GetBinContent(j,i);	
  } // end loop over bins in drift width
		
  double mean = h_drift_depth_adc_slice_->GetMean(1); 
  double error = 0;
  if(nentries != 0){
    error = h_drift_depth_adc_slice_->GetRMS(1) / sqrt(nentries);
  }
		
  _h_mean_[i_ring]->SetBinContent(i, mean);
  _h_mean_[i_ring]->SetBinError(i, error);

}

DEFINE_FWK_MODULE(SiPixelLorentzAngle);
