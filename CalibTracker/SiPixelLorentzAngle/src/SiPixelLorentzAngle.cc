
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <TMath.h>
#include "CalibTracker/SiPixelLorentzAngle/interface/SiPixelLorentzAngle.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "CalibTracker/SiPixelLorentzAngle/interface/TrackLocalAngle.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include <Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h>
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"


#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"


void fit_chi2_fcn(int &npar, double *gin, double &chi2, double* par, int iflag);
TH1F h_drift_depth_int_slice2_;	
TMatrixFSym m_covariance_inv_;
int lower_bin_;

using namespace std;
//TH1F SiPixelLorentzAngle::h_drift_depth_int_slice2_ = TH1F();
// TMatrixFSym SiPixelLorentzAngle::m_covariance_inv_ = TMatrixFSym(); 
// void fit_chi2_fcn(int &npar, double *gin, double &chi2, double* par, int iflag);

SiPixelLorentzAngle::SiPixelLorentzAngle(edm::ParameterSet const& conf) : 
		conf_(conf), filename_(conf.getParameter<std::string>("fileName")), hist_depth_(conf.getParameter<int>("binsDepth")), hist_drift_(conf.getParameter<int>("binsDrift"))
{
  anglefinder_=new  TrackLocalAngle(conf);
	hist_x_ = 50;
	hist_y_ = 100;
	min_x_ = -500.;
	max_x_ = 500.;
	min_y_ = -1500.;
	max_y_ = 500.;
	width_ = 0.0285;
	min_depth_ = -100.;
	max_depth_ = 400.;
	min_drift_ = -200.;
	max_drift_ = 400.;
// 	hist_depth_ = 25;
// 	hist_drift_ = 60;
	event_counter_ = 0;
}

void SiPixelLorentzAngle::beginJob(const edm::EventSetup& c)
{
  hFile_ = new TFile (filename_.c_str(), "RECREATE" );
	int bufsize = 64000;
	SiPixelLorentzAngleTree_ = new TTree("SiPixelLorentzAngleTree_","SiPixel LorentzAngle tree", bufsize);
	SiPixelLorentzAngleTree_->Branch("run", &run_, "run/I", bufsize);
	SiPixelLorentzAngleTree_->Branch("event", &event_, "event/I", bufsize);
	SiPixelLorentzAngleTree_->Branch("module", &module_, "module/I", bufsize);
	SiPixelLorentzAngleTree_->Branch("ladder", &ladder_, "ladder/I", bufsize);
//   SiPixelLorentzAngleTree_->Branch("type", &type, "type/I");
	SiPixelLorentzAngleTree_->Branch("layer", &layer_, "layer/I", bufsize);
	SiPixelLorentzAngleTree_->Branch("isflipped", &isflipped_, "isflipped/I", bufsize);
//   SiPixelLorentzAngleTree_->Branch("string", &string, "string/I");
//   SiPixelLorentzAngleTree_->Branch("extint", &extint, "extint/I");
//   SiPixelLorentzAngleTree_->Branch("size", &size, "size/I");	
	SiPixelLorentzAngleTree_->Branch("pt", &pt_, "pt/F", bufsize);
	SiPixelLorentzAngleTree_->Branch("eta", &eta_, "eta/F", bufsize);
	SiPixelLorentzAngleTree_->Branch("phi", &eta_, "phi/F", bufsize);
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
	
	//histograms for trackhits
	h_cluster_shape_adc_  = new TH2F("h_cluster_shape_adc","cluster shape with adc weight", hist_x_, min_x_, max_x_, hist_y_, min_y_, max_y_);
	h_cluster_shape_noadc_  = new TH2F("h_cluster_shape_noadc","cluster shape without adc weight", hist_x_, min_x_, max_x_, hist_y_, min_y_, max_y_);
	h_cluster_shape_  = new TH2F("h_cluster_shape","cluster shape", hist_x_, min_x_, max_x_, hist_y_, min_y_, max_y_);
	h_cluster_shape_adc_rot_  = new TH2F("h_cluster_shape_adc_rot","cluster shape with adc weight", hist_x_, min_x_, max_x_, hist_y_, -max_y_, -min_y_);
	h_cluster_shape_noadc_rot_  = new TH2F("h_cluster_shape_noadc_rot","cluster shape without adc weight", hist_x_, min_x_, max_x_, hist_y_, -max_y_, -min_y_);
	h_cluster_shape_rot_  = new TH2F("h_cluster_shape_rot","cluster shape", hist_x_, min_x_, max_x_, hist_y_, -max_y_, -min_y_);
	h_drift_depth_adc_  = new TH2F("h_drift_depth_adc","drift distance in x vs. depth vs. adc", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_adc2_  = new TH2F("h_drift_depth_adc2","drift distance in x vs. depth vs. adc^2", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_adc_error2_  = new TH2F("h_drift_depth_adc_error2","drift distance in x vs. depth vs. adc error^2", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_noadc_  = new TH2F("h_drift_depth_noadc","drift distance in x vs. depth vs. events", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_  = new TH2F("h_drift_depth","drift distance in x vs. depth vs. average charge", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_error2_ = new TH2F("h_drift_depth_error2","drift distance in x vs. depth vs. average charge error^2", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_int_  = new TH2F("h_drift_depth_int","drift distance in x vs. depth vs. average charge integrated", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_int_error_matrix_  = new TH3F("h_drift_depth_int_error_matrix","error matrix of drift distance in x vs. depth vs. average charge integrated", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_, hist_drift_ , min_drift_, max_drift_);
	h_fit_middle_ = new TH1F("h_fit_middle","offset parameter of fit", hist_depth_, min_depth_, max_depth_);
	h_fit_width_ = new TH1F("h_fit_width","width parameter of fit", hist_depth_, min_depth_, max_depth_);
	h_mean_ = new TH1F("h_mean","mean of the adc distribution", hist_depth_, min_depth_, max_depth_);
	
	// histograms for simhits
	h_cluster_shape_adc_sim_  = new TH2F("h_cluster_shape_adc_sim","cluster shape with adc weight", hist_x_, min_x_, max_x_, hist_y_, min_y_, max_y_);
	h_cluster_shape_noadc_sim_  = new TH2F("h_cluster_shape_noadc_sim","cluster shape without adc weight", hist_x_, min_x_, max_x_, hist_y_, min_y_, max_y_);
	h_cluster_shape_sim_  = new TH2F("h_cluster_shape","cluster shape_sim", hist_x_, min_x_, max_x_, hist_y_, min_y_, max_y_);
	h_cluster_shape_adc_rot_sim_  = new TH2F("h_cluster_shape_adc_rot_sim","cluster shape with adc weight", hist_x_, min_x_, max_x_, hist_y_, -max_y_, -min_y_);
	h_cluster_shape_noadc_rot_sim_  = new TH2F("h_cluster_shape_noadc_rot_sim","cluster shape without adc weight", hist_x_, min_x_, max_x_, hist_y_, -max_y_, -min_y_);
	h_cluster_shape_rot_sim_  = new TH2F("h_cluster_shape_rot_sim","cluster shape", hist_x_, min_x_, max_x_, hist_y_, -max_y_, -min_y_);
	h_drift_depth_adc_sim_  = new TH2F("h_drift_depth_adc_sim","drift distance in x vs. depth vs. adc", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_adc2_sim_  = new TH2F("h_drift_depth_adc2_sim","drift distance in x vs. depth vs. adc^2", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_adc_error2_sim_  = new TH2F("h_drift_depth_adc_error2_sim","drift distance in x vs. depth vs. adc error^2", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_noadc_sim_  = new TH2F("h_drift_depth_noadc_sim","drift distance in x vs. depth vs. events", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_sim_  = new TH2F("h_drift_depth_sim","drift distance in x vs. depth vs. average charge", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_error2_sim_ = new TH2F("h_drift_depth_error2_sim","drift distance in x vs. depth vs. average charge error^2", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_int_sim_  = new TH2F("h_drift_depth_int_sim","drift distance in x vs. depth vs. average charge integrated", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_int_error_matrix_sim_  = new TH3F("h_drift_depth_int_error_matrix_sim","error matrix of drift distance in x vs. depth vs. average charge integrated", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_, hist_drift_ , min_drift_, max_drift_);
	h_fit_middle_sim_ = new TH1F("h_fit_middle_sim","offset parameter of fit", hist_depth_, min_depth_, max_depth_);
	h_fit_width_sim_ = new TH1F("h_fit_width_sim","width parameter of fit", hist_depth_, min_depth_, max_depth_);
	h_mean_sim_ = new TH1F("h_mean_sim","mean of the adc distribution", hist_depth_, min_depth_, max_depth_);
	
  eventcounter_ = 0;
  eventnumber_ = -1;
  trackcounter_ = 0;
  
  edm::ESHandle<TrackerGeometry> estracker;
  c.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(* estracker);
  //edm::ESHandle<MagneticField> esmagfield;
  //c.get<IdealMagneticFieldRecord>().get(esmagfield);
  //magfield=&(*esmagfield);
}

// Virtual destructor needed.
SiPixelLorentzAngle::~SiPixelLorentzAngle() {  }  

// Functions that gets called by framework every event
void SiPixelLorentzAngle::analyze(const edm::Event& e, const edm::EventSetup& es)
{  
	event_counter_++;
	if(event_counter_ % 500 == 0) cout << "event number " << event_counter_ << endl;
	TrackerHitAssociator associate(e); 
  module_=-1;
//   type=-1;
  layer_=-1;
//   string=-1;
//   size=-1;
//   extint=-1;
	ladder_ = -1;
	isflipped_ = -1;
	pt_ = -999;
	eta_ = 999;
	phi_ = 999;
	pixinfo_.npix = 0;

  run_       = e.id().run();
  event_     = e.id().event();
  
  if(event_ != eventnumber_){
  	eventcounter_+=1;
  	eventnumber_ = event_;
  }
  
  using namespace edm;
  // Step A: Get Inputs 
  anglefinder_->init(e,es);
	std::vector<std::pair<const TrackingRecHit * , TrackLocalAngle::Trackhit> > trackhitvector;
  

	LogDebug("SiPixelLorentzAngle::analyze")<<"Getting tracks";
	std::string src=conf_.getParameter<std::string>( "src" );
	edm::Handle<reco::TrackCollection> trackCollection;
	e.getByLabel(src, trackCollection);
	const reco::TrackCollection *tracks=trackCollection.product();
	reco::TrackCollection::const_iterator tciter;
	if(tracks->size()>0){
		// find the positon and direction of the hits oft each track coming from the track fitter
		for(tciter=tracks->begin();tciter!=tracks->end();tciter++){
			pt_ = tciter->pt();
			eta_ = tciter->momentum().eta();
			phi_ = tciter->momentum().phi();
			chi2_ = tciter->chi2();
			ndof_ = tciter->ndof();
			std::vector<std::pair<const TrackingRecHit *,TrackLocalAngle::Trackhit> > tmptrackhit=anglefinder_->findPixelParameters(*tciter);
			std::vector<std::pair<const TrackingRecHit *,TrackLocalAngle::Trackhit> >::iterator tmpiter;
			std::vector<PSimHit> matched;
			
			// iterate over hits
			for(tmpiter=tmptrackhit.begin();tmpiter!=tmptrackhit.end();tmpiter++){

				DetId detIdObj((tmpiter->first)->geographicalId());
				unsigned int subid = detIdObj.subdetId();
					
				// Pixel Barrel only
				if (subid == 1) {
					
					const PixelGeomDetUnit * theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(detIdObj) );
					const RectangularPixelTopology * topol = dynamic_cast<const RectangularPixelTopology*>(&(theGeomDet->specificTopology()));
					
					PXBDetId pxbdetIdObj(detIdObj);
					layer_ = pxbdetIdObj.layer();
					ladder_ = pxbdetIdObj.ladder();
					module_ = pxbdetIdObj.module();
          // Flipped modules
					float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
					float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
          if ( tmp2<tmp1 ) isflipped_ = 1;
					else isflipped_ = 0;

					const SiPixelRecHit * rechit = dynamic_cast<const SiPixelRecHit *>(tmpiter->first);
					edm::Ref<edm::DetSetVector<SiPixelCluster>, SiPixelCluster> const& cluster = (rechit)->cluster();				
					rechit_.x  = rechit->localPosition().x();
					rechit_.y  = rechit->localPosition().y();
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
					fillPix(*cluster ,topol);
					
					// fill entries in trackhit_:
					trackhit_=tmpiter->second;	
					
					// fill entries in simhit_:
					matched.clear();        
					matched = associate.associateHit(*(tmpiter->first));	
					float dr_start=9999.;
					for (std::vector<PSimHit>::iterator isim = matched.begin(); isim != matched.end(); ++isim){
						DetId simdetIdObj((*isim).detUnitId());
						if (simdetIdObj == detIdObj) {
                //cout << " Association found " << endl;
							float sim_x1 = (*isim).entryPoint().x(); // width (row index, in col direction)
							float sim_y1 = (*isim).entryPoint().y(); // length (col index, in row direction)
							float sim_x2 = (*isim).exitPoint().x();
							float sim_y2 = (*isim).exitPoint().y();
							float sim_xpos = 0.5*(sim_x1+sim_x2);
							float sim_ypos = 0.5*(sim_y1+sim_y2);
							float sim_px = (*isim).momentumAtEntry().x();
							float sim_py = (*isim).momentumAtEntry().y();
							float sim_pz = (*isim).momentumAtEntry().z();
                
							float dr = (sim_xpos-rechit->localPosition().x())*(sim_xpos-rechit->localPosition().x()) +
									(sim_ypos-rechit->localPosition().y())*(sim_ypos-rechit->localPosition().y());
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
					bool large_pix = false;
// 					bool large_drift = false;
					
					for (int j = 0; j <  pixinfo_.npix; j++){
						int colpos = static_cast<int>(pixinfo_.col[j]-0.5);
						if (pixinfo_.row[j] == 0.5 || pixinfo_.row[j] == 79.5 || pixinfo_.row[j] == 80.5 || pixinfo_.row[j] == 159.5 || colpos % 52 == 0 || colpos % 52 == 51 ){
							large_pix = true;	
// 							n_large++;
						}
					}
// 					if ( !large_pix ) 
// 					if( isflipped_ == 0 && layer_ == 1 && (chi2_/ndof_) < 2. && (cluster->charge())/1000. < 120.) SiPixelLorentzAngleTree_->Fill();	
// 					if( !large_pix && isflipped_ == 0 && layer_ == 1 && (cluster->charge())/1000. >= 120.){
					if( !large_pix && layer_ == 1 && (chi2_/ndof_) < 2. && (cluster->charge())/1000. < 120. && module_ == 7){
// 					if( !large_pix && isflipped_ == 0 && layer_ == 1){
						SiPixelLorentzAngleTree_->Fill();
						for (int j = 0; j <  pixinfo_.npix; j++){
							
							// use trackhits
							float dx = (pixinfo_.x[j]  - (trackhit_.x - width_/2. / TMath::Tan(trackhit_.alpha))) * 10000.;
							float dy = (pixinfo_.y[j]  - (trackhit_.y - width_/2. / TMath::Tan(trackhit_.beta))) * 10000.;
							float dx_rot = dx * TMath::Cos(trackhit_.gamma) + dy * TMath::Sin(trackhit_.gamma);
							float dy_rot = dy * TMath::Cos(trackhit_.gamma) - dx * TMath::Sin(trackhit_.gamma) ;
							float depth = dy * tan(trackhit_.beta);
							float drift = dx - dy * tan(trackhit_.gamma);
														
							h_cluster_shape_adc_->Fill(dx, dy, pixinfo_.adc[j]);
							h_cluster_shape_noadc_->Fill(dx, dy);
							h_cluster_shape_adc_rot_->Fill(dx_rot, dy_rot, pixinfo_.adc[j]);
							h_cluster_shape_noadc_rot_->Fill(dx_rot, dy_rot);
							h_drift_depth_adc_->Fill(drift, depth, pixinfo_.adc[j]);
							h_drift_depth_adc2_->Fill(drift, depth, pixinfo_.adc[j]*pixinfo_.adc[j]);
							h_drift_depth_noadc_->Fill(drift, depth);						
							
							// use simhits
							float dx_sim = (pixinfo_.x[j]  - (simhit_.x - width_/2. / TMath::Tan(simhit_.alpha))) * 10000.;
							float dy_sim = (pixinfo_.y[j]  - (simhit_.y - width_/2. / TMath::Tan(simhit_.beta))) * 10000.;
							float dx_rot_sim = dx_sim * TMath::Cos(simhit_.gamma) + dy_sim * TMath::Sin(simhit_.gamma);
							float dy_rot_sim = dy_sim * TMath::Cos(simhit_.gamma) - dx_sim * TMath::Sin(simhit_.gamma) ;
							float depth_sim = dy_sim * tan(simhit_.beta);
							float drift_sim = dx_sim - dy_sim * tan(simhit_.gamma);
							
// 							if(dx_rot > -50. && dx_rot < 100. && dy_rot > 100.){
// 								cout << "dx: " << dx << ", dy: " << dy << ", " << trackhit_.gamma << endl;
// 							}

							h_cluster_shape_adc_sim_->Fill(dx_sim, dy_sim, pixinfo_.adc[j]);
							h_cluster_shape_noadc_sim_->Fill(dx_sim, dy_sim);
							h_cluster_shape_adc_rot_sim_->Fill(dx_rot_sim, dy_rot_sim, pixinfo_.adc[j]);
							h_cluster_shape_noadc_rot_sim_->Fill(dx_rot_sim, dy_rot_sim);
							h_drift_depth_adc_sim_->Fill(drift_sim, depth_sim, pixinfo_.adc[j]);
							h_drift_depth_adc2_sim_->Fill(drift_sim, depth_sim, pixinfo_.adc[j]*pixinfo_.adc[j]);
							h_drift_depth_noadc_sim_->Fill(drift_sim, depth_sim);					
						}
					}
				}
			}
		}
	}
}

void SiPixelLorentzAngle::endJob()
{
	fillHistograms();
	minuit = new TMinuit(15);		
	h_drift_depth_int_slice_ = new TH1F("h_drift_depth_int_slice", "slice of integrated histogram", hist_drift_ , min_drift_, max_drift_);
	h_drift_depth_int_slice_error_matrix_ = new TH2F("h_drift_depth_int_slice_error_matrix", "error matrix of slice of integrated histogram", hist_drift_ , min_drift_, max_drift_, hist_drift_ , min_drift_, max_drift_);
	h_drift_depth_adc_slice_ = new TH1F("h_drift_depth_adc_slice", "slice of adc histogram", hist_drift_ , min_drift_, max_drift_);
	h_drift_depth_int_slice_sim_ = new TH1F("h_drift_depth_int_slice", "slice of integrated histogram", hist_drift_ , min_drift_, max_drift_);
	h_drift_depth_int_slice_error_matrix_sim_ = new TH2F("h_drift_depth_int_slice_error_matrix", "error matrix of slice of integrated histogram", hist_drift_ , min_drift_, max_drift_, hist_drift_ , min_drift_, max_drift_);
	h_drift_depth_adc_slice_sim_ = new TH1F("h_drift_depth_adc_slice", "slice of adc histogram", hist_drift_ , min_drift_, max_drift_);
	m_covariance_ = new TMatrixFSym();
	TF1 *f1 = new TF1("f1","(x - [0] + [1]*0.5) / [1]",min_depth_, max_depth_);
	//loop over bins in depth (z-local-coordinate) (in order to fit slices)
	for( int i = 1; i <= hist_depth_; i++){
		// find bins which contain values 0.1 and 0.9 (fit-range)		
// 		cout << "loop begin" << endl;
		lower_bin_ = 0;
		upper_bin_ = 0;
		bool is_lower_limit = false;
		bool is_upper_limit = false;
		h_drift_depth_int_slice_->Reset("ICE");
		h_drift_depth_int_slice_error_matrix_->Reset("ICE");
		h_drift_depth_adc_slice_->Reset("ICE");
		h_drift_depth_int_slice_sim_->Reset("ICE");
		h_drift_depth_int_slice_error_matrix_sim_->Reset("ICE");
		h_drift_depth_adc_slice_sim_->Reset("ICE");
		double nentries = 0;
		double nentries_sim = 0;
		
		//loop over bins in drift (x-local-coordinate) 
		for(int j = 1; j <= hist_drift_; j++){
			h_drift_depth_int_slice_->SetBinContent(j, h_drift_depth_int_->GetBinContent(j,i));
			h_drift_depth_adc_slice_->SetBinContent(j, h_drift_depth_adc_->GetBinContent(j,i));
			h_drift_depth_adc_slice_->SetBinError(j, h_drift_depth_adc_->GetBinError(j,i));
			h_drift_depth_int_slice_->SetBinError(j, h_drift_depth_int_->GetBinError(j,i));	
			nentries += h_drift_depth_noadc_->GetBinContent(j,i);
			
			h_drift_depth_int_slice_sim_->SetBinContent(j, h_drift_depth_int_sim_->GetBinContent(j,i));
			h_drift_depth_adc_slice_sim_->SetBinContent(j, h_drift_depth_adc_sim_->GetBinContent(j,i));
			h_drift_depth_adc_slice_sim_->SetBinError(j, h_drift_depth_adc_sim_->GetBinError(j,i));
			h_drift_depth_int_slice_sim_->SetBinError(j, h_drift_depth_int_sim_->GetBinError(j,i));	
			nentries_sim += h_drift_depth_noadc_sim_->GetBinContent(j,i);
			
			for(int k = 1; k <= hist_drift_; k++){
				h_drift_depth_int_slice_error_matrix_->SetBinContent(j,k, h_drift_depth_int_error_matrix_->GetBinContent(j,i,k));
			}
			
			// find the starting and end-point for the fit
			if(!is_lower_limit && h_drift_depth_int_->GetBinContent(j,i) > 0.2){
				lower_bin_ = j;
				is_lower_limit = true;
			}
			if(!is_upper_limit && h_drift_depth_int_->GetBinContent(j,i) > 0.8){
				upper_bin_ = j;
				is_upper_limit = true;
			}
		}// end loop over bins in drift
		
		
		// mean for each slice in depth for trackhits
		char name[128];
		sprintf(name, "h_drift_depth_adc_slice_%d", i); 
		h_drift_depth_adc_slice_->SetNameTitle(name,name);
		_h_drift_depth_adc_slice_[i] = new TH1F(*h_drift_depth_adc_slice_);
		double mean = h_drift_depth_adc_slice_->GetMean(1); 
		double error = 0;
		if(nentries != 0){
			error = h_drift_depth_adc_slice_->GetRMS(1) / sqrt(nentries);
		}
		
		h_mean_->SetBinContent(i, mean);
		h_mean_->SetBinError(i, error);
		
		// mean for each slice in depth for trackhits
		
		sprintf(name, "h_drift_depth_adc_slice_%d_sim", i); 
		h_drift_depth_adc_slice_sim_->SetNameTitle(name,name);
		_h_drift_depth_adc_slice_sim_[i] = new TH1F(*h_drift_depth_adc_slice_sim_);
		double mean_sim = h_drift_depth_adc_slice_sim_->GetMean(1); 
		double error_sim = 0;
		if(nentries_sim != 0){
			error_sim = h_drift_depth_adc_slice_sim_->GetRMS(1) / sqrt(nentries_sim);
		}
		
		h_mean_sim_->SetBinContent(i, mean_sim);
		h_mean_sim_->SetBinError(i, error_sim);
		
		// fit linear function to integrated histogram
		if(is_lower_limit){
			sprintf(name, "h_drift_depth_int_slice_%d", i); 
			h_drift_depth_int_slice_->SetNameTitle(name,name);
			_h_drift_depth_int_slice_[i] = new TH1F(*h_drift_depth_int_slice_);
// 			fitLinear(i);
// 			TF1 *f1 = new TF1("f1","[0] + [1]*x",h_drift_depth_int_slice_->GetBinLowEdge(lower_bin_), h_drift_depth_int_slice_->GetBinLowEdge(upper_bin+1));
// 			sprintf(name, "f_drift_depth_int_slice_%d", i); 
// 			f1->SetName(name);
// 			f1->SetRange(h_drift_depth_int_slice_->GetBinLowEdge(lower_bin_), h_drift_depth_int_slice_->GetBinLowEdge(upper_bin_));
// 			f1->SetParName(0,"middle");
// 			f1->SetParName(1,"width");
// 			f1->SetParameter(0,h_fit_middle_->GetBinContent(i));
// 			f1->SetParameter(1,h_fit_width_->GetBinContent(i));
// 			_f_drift_depth_int_slice_[i] = new TF1(*f1);
// 			_f_drift_depth_int_slice_[i]->Write();
		}
		
		// find mean of adc distribution
		
		
				
	}// end loop over bins in depth 
// 	delete h_drift_depth_int_slice_;
// 	delete h_drift_depth_int_slice_error_matrix_;
		
  hFile_->Write();
  hFile_->Close();
}

void SiPixelLorentzAngle::fillPix(const SiPixelCluster & LocPix, const RectangularPixelTopology * topol)
{
	const std::vector<SiPixelCluster::Pixel>& pixvector = LocPix.pixels();
	
	for(pixinfo_.npix = 0; pixinfo_.npix < static_cast<int>(pixvector.size()); ++pixinfo_.npix) {
		SiPixelCluster::Pixel holdpix = pixvector[pixinfo_.npix];
		pixinfo_.row[pixinfo_.npix] = holdpix.x;
		pixinfo_.col[pixinfo_.npix] = holdpix.y;
		pixinfo_.adc[pixinfo_.npix] = holdpix.adc;
		LocalPoint lp = topol->localPosition(MeasurementPoint(holdpix.x, holdpix.y));
		pixinfo_.x[pixinfo_.npix] = lp.x();
		pixinfo_.y[pixinfo_.npix]= lp.y();
	}
}

void SiPixelLorentzAngle::fitLinear(int i)
{
// 	data_covariance_ = new float[(upper_bin_-lower_bin_+1)*(upper_bin_-lower_bin_+1)];
	for(int k = 0; k < 10000; k++) data_covariance_[k] = 0.;
	for(int m = lower_bin_; m <= upper_bin_; m++){
		for(int n = m; n <= upper_bin_; n++){
// 				m_covariance_
// 			if(m == n){
				data_covariance_[(m-lower_bin_) + (n-lower_bin_)*(upper_bin_-lower_bin_ +1)] = h_drift_depth_int_slice_error_matrix_->GetBinContent(m,n);
// 			}
// 			else{
// 				data_covariance_[(m-lower_bin_) + (n-lower_bin_)*(upper_bin_-lower_bin_+1)] = 0.;
// 				data_covariance_[(n-lower_bin_) + (m-lower_bin_)*(upper_bin_-lower_bin_+1)] = 0.;
// 			}
		}
	}
	h_drift_depth_int_slice2_ = 	*h_drift_depth_int_slice_;
// 	TMatrixFSym* m_covariance_ = new TMatrixFSym(upper_bin_ - lower_bin_ +1, data_covariance_, "");
	m_covariance_->ResizeTo(upper_bin_ - lower_bin_ +1, upper_bin_ - lower_bin_ +1);
	m_covariance_->SetMatrixArray( data_covariance_ ,"" );
// 	m_covariance_inv_ = new TMatrixFSym(upper_bin_ - lower_bin_ +1);
// 			double * det = new double; 
	m_covariance_inv_.ResizeTo(m_covariance_->GetNcols(), m_covariance_->GetNcols());
	m_covariance_inv_ = m_covariance_->Invert();
	
	minuit->SetFCN( (void (*)(int&, double*, double&, double*, int)) fit_chi2_fcn);
// 	minuit->SetFCN(&fit_chi2_fcn);

	double arglist[2];
	int ierflg = 0;
	arglist[0] = 1; // error for chi^2
	minuit->mnexcm("SET ERR", arglist ,1,ierflg);
	// 	setting initial values
	double vstart[2] = {0., 50.};
	// setting parameter steps
	double step[2] = {0.01, 0.01};
	minuit->mnparm(0, "middle", vstart[0], step[0], -100., 400.,ierflg);
	minuit->mnparm(1, "width", vstart[1], step[1], 0., 500.,ierflg);

// 	arglist[0] = 3;
// 	minuit->mnexcm("Fix", arglist ,1,ierflg);
	// Now ready for minimization step
	arglist[0] = 5000;
	arglist[1] = 1;
	minuit->mnexcm("MIGRAD", arglist ,2,ierflg);
	arglist[0] = 500;
	minuit->mnexcm("MINOS", arglist ,1,ierflg);
	cout << "pass 4" << endl;
	
	// Print results
	double amin,edm,errdef;
	double middle[2];
	double width[2];
	int nvpar,nparx,icstat;
	minuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);
	minuit->mnprin(3,amin);
	minuit->mnmatu(1);
	minuit->GetParameter(0, middle[0], middle[1]);
	minuit->GetParameter(1, width[0], width[1]);
	cout << "pass 5" << endl;
	h_fit_middle_->SetBinContent(i,middle[0]);
	h_fit_middle_->SetBinError(i,middle[1]);
	h_fit_width_->SetBinContent(i,width[0]);
	h_fit_width_->SetBinError(i,width[1]);
	cout << "fit values: " << middle[0] <<", " << middle[1] <<", " << width[0] <<", " << width[1] <<endl;
// 	delete f1;
// 	delete m_covariance_;
// 	delete m_covariance_inv_;
// 	delete[] data_covariance_;
// 	data_covariance_ = 0;
// 	delete minuit;	
	
// 	cout << "pass 6" << endl;
}

void fit_chi2_fcn(int &npar, double *gin, double &chi2, double* par, int iflag)
{
	// 	npar	=	number of Parameters
	// 	gin 	=
	// 	chi2	=	value of chi2
	// 	*par	=	pointer to array of parameters
	// 	iflag=

	double x= 0.;
// 	cout << "pass fcn1" << endl;
	float* delta_x = new float[m_covariance_inv_.GetNcols()];
// 	float delta_x[30];
// 	cout << "pointer to matrix in fcn: " << m_covariance_ << endl;
// 	cout << "pointer to inv matrix in fcn: " << m_covariance_inv_ << endl;
// 	cout << "pointer to histo in fcn: " << h_drift_depth_int_slice_ << endl;
	for(int i = 0;  i < m_covariance_inv_.GetNcols(); i++){
// 		cout << "loop in fcn" << endl;
		delta_x[i] = h_drift_depth_int_slice2_.GetBinContent(i + lower_bin_) - (h_drift_depth_int_slice2_.GetBinCenter(i + lower_bin_) - par[0] + par[1]*0.5) / par[1];
	}
// 	cout << "pass fcn2" << endl;
// 	cout << m_covariance_inv_.GetNcols() << endl;
	TVectorF v_chi2_(m_covariance_inv_.GetNcols(), delta_x);
// 	cout << "pass fcn3" << endl;
	TVectorF v_chi2_m(v_chi2_);
	
	(v_chi2_m) *= (m_covariance_inv_);
// 	cout << "pass fcn4" << endl;
	for(int i = 0;  i < m_covariance_inv_.GetNcols(); i++){
		x += v_chi2_m[i] * v_chi2_[i];
	}
// 	cout << "pass fcn5" << endl;
// 	delete v_chi2_m;
// 	delete v_chi2_;
	delete[] delta_x;
	delta_x = 0;
// 	cout << "pass fcn6" << endl;
// 	cout << "chi^2: " << x << endl;
	chi2 = x;
}

void SiPixelLorentzAngle::fillHistograms()
{
	// produce histograms with the average adc counts
	h_cluster_shape_->Divide(h_cluster_shape_adc_, h_cluster_shape_noadc_);
	h_cluster_shape_rot_->Divide(h_cluster_shape_adc_rot_, h_cluster_shape_noadc_rot_);
	h_drift_depth_->Divide(h_drift_depth_adc_, h_drift_depth_noadc_);
	
	//loop over bins in depth (z-local-coordinate)
	for( int i = 1; i <= hist_depth_; i++){	
		
		// determine sigma and sigma^2 of the adc counts and average adc counts
		//loop over bins in drift width
		for( int j = 1; j<= hist_drift_; j++){
			double adc_error2 = (h_drift_depth_adc2_->GetBinContent(j,i) - h_drift_depth_adc_->GetBinContent(j,i)*h_drift_depth_adc_->GetBinContent(j, i) / h_drift_depth_noadc_->GetBinContent(j, i)) /  h_drift_depth_noadc_->GetBinContent(j, i);
			h_drift_depth_adc_->SetBinError(j, i, sqrt(adc_error2));
			if(h_drift_depth_noadc_->GetBinContent(j, i) > 1){
				h_drift_depth_adc_error2_->SetBinContent(j, i,adc_error2);
				double error2 = adc_error2 / (h_drift_depth_noadc_->GetBinContent(j,i) - 1.);
				h_drift_depth_error2_->SetBinContent(j,i,error2);
				h_drift_depth_->SetBinError(j,i,sqrt(error2));
			} 
			else{
				h_drift_depth_->SetBinError(j,i,0);
			}
		} // end loop over bins in drift width
		
		
		// integrate histograms
		double integral = 0.;
		double integral_error2 = 0.;
		double total_integral = 0.;
		double total_integral_error2 = 0.;
		int start_int = hist_drift_ - 1; 
		int stop_int = 0;
		
		// check where to start the integration (at least 4 hit bins in a row)
		for( int j = 1; j< hist_drift_-3; j++){
			if(h_drift_depth_->GetBinContent(j,i) != 0 && h_drift_depth_->GetBinContent(j+1,i) != 0 && h_drift_depth_->GetBinContent(j+2,i) != 0 && h_drift_depth_->GetBinContent(j+3,i) != 0){
				start_int = j -1;
				break;
			}
		}
		for( int j = hist_drift_; j >= 4; j--){
			if(h_drift_depth_->GetBinContent(j,i) != 0 && h_drift_depth_->GetBinContent(j-1,i) != 0 && h_drift_depth_->GetBinContent(j-2,i) != 0 && h_drift_depth_->GetBinContent(j-3,i) != 0){
				stop_int = j;
				break;
			}
		}
		
		// normalization for the integrals
		for(int j = start_int; j <= stop_int; j++){
			total_integral += h_drift_depth_->GetBinContent(j , i);
			total_integral_error2 += h_drift_depth_error2_->GetBinContent(j, i);
		}
		// determine integrated histograms
		
		for(int j = 1; j <= hist_drift_; j++){
			if( j <= start_int ){
				h_drift_depth_int_->SetBinContent(j,i,0);
				h_drift_depth_int_->SetBinError(j,i,0);
			}
			else if( j <= stop_int ){
				integral += h_drift_depth_->GetBinContent(j, i);
				integral_error2 += h_drift_depth_error2_->GetBinContent(j, i);
				if(total_integral != 0){
					h_drift_depth_int_->SetBinContent(j,i,integral/total_integral);
// 					double error = sqrt( ( (1.-integral/total_integral)*(1.-integral/total_integral) * integral_error2 * integral_error2 + integral/total_integral*integral/total_integral * ( total_integral_error2 - integral_error2 ) * ( total_integral_error2 - integral_error2 ) ) / (total_integral*total_integral) );
					double error2 = (1. - integral/total_integral) * (1. - integral/total_integral) * integral_error2 / total_integral /total_integral + integral/total_integral * integral/total_integral * (total_integral_error2 - integral_error2) / total_integral / total_integral;
					h_drift_depth_int_->SetBinError(j,i,sqrt(error2));
					h_drift_depth_int_error_matrix_->SetBinContent(j,i,j,error2);
					double integral_error2b = integral_error2;
					double integralb = integral;
					for(int k = j+1; k<stop_int; k++){
						integralb += h_drift_depth_->GetBinContent(k, i);
						integral_error2b += h_drift_depth_error2_->GetBinContent(k, i);
						double correlation = (  ((total_integral - integral) * (total_integral - integralb) * integral_error2) + (integral * (total_integral - integralb) * (integral_error2b - integral_error2)) + ( integral * integralb *  (total_integral_error2 - integral_error2b) ) ) /total_integral/total_integral/total_integral/total_integral;
						h_drift_depth_int_error_matrix_->SetBinContent(k,i,j,correlation);
						h_drift_depth_int_error_matrix_->SetBinContent(j,i,k,correlation);
					}
				}
				else{
					h_drift_depth_int_->SetBinContent(j,i,0);
					h_drift_depth_int_->SetBinError(j,i,0);
				}
			}
			else{
				if(total_integral != 0){
					h_drift_depth_int_->SetBinContent(j,i,integral/total_integral);
					h_drift_depth_int_->SetBinError(j,i,0);
				}
				else{
					h_drift_depth_int_->SetBinContent(j,i,0);
					h_drift_depth_int_->SetBinError(j,i,0);
				}
			}
		}
	} // end loop over bins in depth 
}
