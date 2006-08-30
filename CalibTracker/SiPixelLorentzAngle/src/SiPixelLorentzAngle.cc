
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

using namespace std;
SiPixelLorentzAngle::SiPixelLorentzAngle(edm::ParameterSet const& conf) : 
  conf_(conf), filename_(conf.getParameter<std::string>("fileName"))
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
	hist_depth_ = 100;
	hist_drift_ = 120;
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
	h_cluster_shape_adc_  = new TH2F("h_cluster_shape_adc","cluster shape with adc weight", hist_x_, min_x_, max_x_, hist_y_, min_y_, max_y_);
	h_cluster_shape_noadc_  = new TH2F("h_cluster_shape_noadc","cluster shape without adc weight", hist_x_, min_x_, max_x_, hist_y_, min_y_, max_y_);
	h_cluster_shape_  = new TH2F("h_cluster_shape","cluster shape", hist_x_, min_x_, max_x_, hist_y_, min_y_, max_y_);
	h_cluster_shape_adc_rot_  = new TH2F("h_cluster_shape_adc_rot","cluster shape with adc weight", hist_x_, min_x_, max_x_, hist_y_, -max_y_, -min_y_);
	h_cluster_shape_noadc_rot_  = new TH2F("h_cluster_shape_noadc_rot","cluster shape without adc weight", hist_x_, min_x_, max_x_, hist_y_, -max_y_, -min_y_);
	h_cluster_shape_rot_  = new TH2F("h_cluster_shape_rot","cluster shape", hist_x_, min_x_, max_x_, hist_y_, -max_y_, -min_y_);
	h_drift_depth_adc_  = new TH2F("h_drift_depth_adc","drift distance in x vs. depth vs. adc", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_noadc_  = new TH2F("h_drift_depth_noadc","drift distance in x vs. depth vs. events", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_  = new TH2F("h_drift_depth","drift distance in x vs. depth vs. average charge", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	h_drift_depth_int_  = new TH2F("h_drift_depth_int","drift distance in x vs. depth vs. average charge integrated", hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
	
	
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
					bool large_drift = false;
					for (int j = 0; j <  pixinfo_.npix; j++){
						if (pixinfo_.row[j] == 0.5 || pixinfo_.row[j] == 79.5 || pixinfo_.row[j] == 80.5 || pixinfo_.row[j] == 159.5 ){
							large_pix = true;	
// 							n_large++;
						}
					}
					if ( !large_pix ) SiPixelLorentzAngleTree_->Fill();	
// 					if( !large_pix && isflipped_ == 0 && layer_ == 1 && (cluster->charge())/1000. >= 120.){
					if( !large_pix && isflipped_ == 0 && layer_ == 1 && (chi2_/ndof_) < 2. && (cluster->charge())/1000. < 120.){
// 					if( !large_pix && isflipped_ == 0 && layer_ == 1){
						for (int j = 0; j <  pixinfo_.npix; j++){
							float dx = (pixinfo_.x[j]  - (trackhit_.x - width_/2. / TMath::Tan(trackhit_.alpha))) * 10000.;
							float dy = (pixinfo_.y[j]  - (trackhit_.y - width_/2. / TMath::Tan(trackhit_.beta))) * 10000.;
							float dx_rot = dx * TMath::Cos(trackhit_.gamma) + dy * TMath::Sin(trackhit_.gamma);
							float dy_rot = dy * TMath::Cos(trackhit_.gamma) - dx * TMath::Sin(trackhit_.gamma) ;
							float depth = dy * tan(trackhit_.beta);
							float drift = dx - dy * tan(trackhit_.gamma);
// 							if(dx_rot > -50. && dx_rot < 100. && dy_rot > 100.){
// 								cout << "dx: " << dx << ", dy: " << dy << ", " << trackhit_.gamma << endl;
// 							}
							h_cluster_shape_adc_->Fill(dx, dy, pixinfo_.adc[j]);
							h_cluster_shape_noadc_->Fill(dx, dy);
							h_cluster_shape_adc_rot_->Fill(dx_rot, dy_rot, pixinfo_.adc[j]);
							h_cluster_shape_noadc_rot_->Fill(dx_rot, dy_rot);
							h_drift_depth_adc_->Fill(drift, depth, pixinfo_.adc[j]);
							h_drift_depth_noadc_->Fill(drift, depth);
						
						}
					}
				}	
			}
		}
	}
}

void SiPixelLorentzAngle::endJob()
{
	h_cluster_shape_->Divide(h_cluster_shape_adc_, h_cluster_shape_noadc_);
	h_cluster_shape_rot_->Divide(h_cluster_shape_adc_rot_, h_cluster_shape_noadc_rot_);
	h_drift_depth_->Divide(h_drift_depth_adc_, h_drift_depth_noadc_);
	for( int i = 0; i < hist_depth_; i++){
		double integral = 0.;
		double total_integral = 0.;
		int start_int = hist_drift_ - 1; 
		int stop_int = 0;
		for( int j = 1; j< hist_drift_-1; j++){
			if(h_drift_depth_->GetBinContent(j,i) != 0 && h_drift_depth_->GetBinContent(j-1,i) != 0 && h_drift_depth_->GetBinContent(j+1,i) != 0 && h_drift_depth_->GetBinContent(j+2,i) != 0 && h_drift_depth_->GetBinContent(j+3,i) != 0){
				start_int = j -1;
				break;
			}
		}
		for( int j = hist_drift_ - 1; j > 1; j--){
			if(h_drift_depth_->GetBinContent(j,i) != 0 && h_drift_depth_->GetBinContent(j+1,i) != 0 && h_drift_depth_->GetBinContent(j-1,i) != 0 && h_drift_depth_->GetBinContent(j-2,i) != 0 && h_drift_depth_->GetBinContent(j-3,i) != 0){
				stop_int = j + 1;
				break;
			}
		}
		total_integral = h_drift_depth_->Integral(start_int, stop_int, i, i);
		for(int j = 0; j < hist_drift_; j++){
			if( j < start_int ){
				h_drift_depth_int_->SetBinContent(j,i,0);
			}
			else if( j <= stop_int ){
				integral = h_drift_depth_->Integral(start_int, j, i, i);
				if(total_integral != 0){
					h_drift_depth_int_->SetBinContent(j,i,integral/total_integral);
				}
				else{
					h_drift_depth_int_->SetBinContent(j,i,0);
				}
			}
			else{
				if(total_integral != 0){
					h_drift_depth_int_->SetBinContent(j,i,integral/total_integral);
				}
				else{
					h_drift_depth_int_->SetBinContent(j,i,0);
				}
			}
		}
	}
// 	h_cluster_shape_adc_->Write();
// 	h_cluster_shape_noadc_->Write();
// 	h_cluster_shape_->Write();
// 	h_cluster_shape_adc_rot_->Write();
// 	h_cluster_shape_noadc_rot_->Write();
// 	h_cluster_shape_rot_->Write();
// 	h_drift_depth_adc_->Write();
// 	h_drift_depth_noadc_->Write();
// 	h_drift_depth_->Write();
// 	h_drift_depth_int_->Write();
	
//   ofstream fit;
//   fit.open("fit.txt");
  
//   fit.close();
  hFile_->Write();
  hFile_->Close();
// 	delete h_cluster_shape_adc_;
// 	delete h_cluster_shape_noadc_;
// 	delete h_cluster_shape_;
// 	delete h_cluster_shape_adc_rot_;
// 	delete h_cluster_shape_noadc_rot_;
// 	delete h_cluster_shape_rot_;
// 	delete h_drift_depth_adc_;
// 	delete h_drift_depth_noadc_;
// 	delete h_drift_depth_;
// 	delete h_drift_depth_int_;
// 	delete hFile_;
// 	delete SiPixelLorentzAngleTree_;
}

void SiPixelLorentzAngle::fillPix(const SiPixelCluster & LocPix, const RectangularPixelTopology * topol)
{
	const std::vector<SiPixelCluster::Pixel>& pixvector = LocPix.pixels();
	pixinfo_.npix = 0;
	for( ; pixinfo_.npix < pixvector.size(); ++pixinfo_.npix) {
		SiPixelCluster::Pixel holdpix = pixvector[pixinfo_.npix];
		pixinfo_.row[pixinfo_.npix] = holdpix.x;
		pixinfo_.col[pixinfo_.npix] = holdpix.y;
		pixinfo_.adc[pixinfo_.npix] = holdpix.adc;
		LocalPoint lp = topol->localPosition(MeasurementPoint(holdpix.x, holdpix.y));
		pixinfo_.x[pixinfo_.npix] = lp.x();
		pixinfo_.y[pixinfo_.npix]= lp.y();
	}
}
