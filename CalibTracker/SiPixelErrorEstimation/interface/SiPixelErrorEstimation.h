
#ifndef CalibTracker_SiPixelerrorEstimation_SiPixelErrorEstimation_h
#define CalibTracker_SiPixelerrorEstimation_SiPixelErrorEstimation_h

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h" 

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//--- for SimHit association
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"  
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h" 

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h" 
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h" 
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

#include <string>

#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TProfile.h>

class TTree;
class TFile; 

class SiPixelErrorEstimation : public edm::EDAnalyzer 
{
 public:
  
  explicit SiPixelErrorEstimation(const edm::ParameterSet&);
  virtual ~SiPixelErrorEstimation();
    
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  void computeAnglesFromDetPosition( const SiPixelCluster & cl, 
				     const GeomDetUnit    & det, 
				     float& alpha, float& beta );
    
 private: 
  
  edm::ParameterSet conf_;
  std::string outputFile_;
  std::string src_;
  bool checkType_; // do we check that the simHit associated with recHit is of the expected particle type ?
  int genType_; // the type of particle that the simHit associated with recHits should be
  bool include_trk_hits_; // if set to false, take only hits directly from detector modules (don't ntuplize hits from tracks)

  // variables that go in the output ttree_track_hits_
  float rechitx; // x position of hit 
  float rechity; // y position of hit
  float rechitz; // z position of hit
  float rechiterrx; // x position error of hit (error not squared)
  float rechiterry; // y position error of hit (error not squared)
  float rechitresx; // difference between reconstructed hit x position and 'true' x position
  float rechitresy; // difference between reconstructed hit y position and 'true' y position
  float rechitpullx; // x residual divideded by error
  float rechitpully; // y residual divideded by error





  float strip_rechitx; // x position of hit 
  float strip_rechity; // y position of hit
  float strip_rechitz; // z position of hit
  float strip_rechiterrx; // x position error of hit (error not squared)
  float strip_rechiterry; // y position error of hit (error not squared)
  float strip_rechitresx; // difference between reconstructed hit x position and 'true' x position
  
  
  float strip_rechitresx2;

  float strip_rechitresy; // difference between reconstructed hit y position and 'true' y position
  float strip_rechitpullx; // x residual divideded by error
  float strip_rechitpully; // y residual divideded by error
  int strip_is_stereo;
  int strip_hit_type; // matched=0, 1D=1 or 2D=2
  int detector_type; //IB1=1, IB2=2, OB1=3, OB2=4

  float strip_trk_pt;
  float strip_cotalpha;
  float strip_cotbeta;
  float strip_locbx;
  float strip_locby;
  float strip_locbz;
  float strip_charge;
  int strip_size;
  int strip_edge;
  int strip_nsimhit; // number of simhits associated with a rechit
  int strip_pidhit; // PID of the particle that produced the simHit associated with the recHit
  int strip_simproc; // procces type

  int strip_subdet_id; // enum SubDetector { UNKNOWN=0, TIB=3, TID=4, TOB=5, TEC=6 };
 
  int strip_tib_layer             ;
  int strip_tib_module            ;
  int strip_tib_order             ;
  int strip_tib_side              ;
  int strip_tib_is_double_side    ;
  int strip_tib_is_z_plus_side    ;
  int strip_tib_is_z_minus_side   ;
  int strip_tib_layer_number      ;
  int strip_tib_string_number     ;
  int strip_tib_module_number     ;
  int strip_tib_is_internal_string;
  int strip_tib_is_external_string;
  int strip_tib_is_rphi           ;
  int strip_tib_is_stereo         ;          
  
  int strip_tob_layer             ;
  int strip_tob_module            ;
  //int strip_tob_order             ;
  int strip_tob_side              ;
  int strip_tob_is_double_side    ;
  int strip_tob_is_z_plus_side    ;
  int strip_tob_is_z_minus_side   ;
  int strip_tob_layer_number      ;
  int strip_tob_rod_number     ;
  int strip_tob_module_number     ;

  int strip_tob_is_rphi           ;
  int strip_tob_is_stereo         ;     

  float strip_prob;
  int   strip_qbin;
  
  int   strip_nprm;

  int strip_pidhit1;
  int strip_simproc1;
  
  int strip_pidhit2;
  int strip_simproc2;
  
  int strip_pidhit3;
  int strip_simproc3;

  int strip_pidhit4;
  int strip_simproc4;

  int strip_pidhit5;
  int strip_simproc5;

  int strip_split;
  float strip_clst_err_x;
  float strip_clst_err_y;

  int npix; // number of pixel in the cluster
  int nxpix; // size of cluster (number of pixels) along x direction
  int nypix; // size of cluster (number of pixels) along y direction
  float charge; // total charge in cluster

  int edgex; // edgex = 1 if the cluster is at the x edge of the module  
  int edgey; // edgey = 1 if the cluster is at the y edge of the module 

  int bigx; // bigx = 1 if the cluster contains at least one big pixel in x
  int bigy; // bigy = 1 if the cluster contains at least one big pixel in y

  float alpha; // track angle in the xz plane of the module local coordinate system  
  float beta;  // track angle in the yz plane of the module local coordinate system  

  float trk_alpha; // reconstructed track angle in the xz plane of the module local coordinate system  
  float trk_beta;  // reconstructed track angle in the yz plane of the module local coordinate system  

  float phi;   // polar track angle
  float eta;   // pseudo-rapidity (function of theta, the azimuthal angle)

  int subdetId;
  int layer;  
  int ladder; 
  int mod;    
  int side;    
  int disk;   
  int blade;  
  int panel;  
  int plaq;   
 
  int half; // half = 1 if the barrel module is half size and 0 if it is full size (only defined for barrel) 
  int flipped; // flipped = 1 if the module is flipped and 0 if non-flipped (only defined for barrel) 

  int nsimhit; // number of simhits associated with a rechit
  int pidhit; // PID of the particle that produced the simHit associated with the recHit
  int simproc; // procces tye

  float simhitx; // true x position of hit 
  float simhity; // true y position of hit

  int evt;
  int run;

  float hit_probx;
  float hit_proby;
  float hit_cprob0;
  float hit_cprob1;
  float hit_cprob2;

  int pixel_split;

  float pixel_clst_err_x;
  float pixel_clst_err_y;


   // variables that go in the output ttree_sll_hits_
  
  int all_subdetid;

  int all_layer;
  int all_ladder;
  int all_mod;

  int all_side;
  int all_disk;
  int all_blade;
  int all_panel;
  int all_plaq;
  
  int all_half;
  int all_flipped;

  int all_cols;
  int all_rows;

  float all_rechitx;
  float all_rechity;
  float all_rechitz;

  float all_simhitx;
  float all_simhity;

  float all_rechiterrx;
  float all_rechiterry;

  float all_rechitresx;
  float all_rechitresy;

  float all_rechitpullx;
  float all_rechitpully;

  int all_npix;
  int all_nxpix;
  int all_nypix;

  int all_edgex;
  int all_edgey;

  int all_bigx;
  int all_bigy;

  float all_alpha;
  float all_beta;

  float all_simphi;
  float all_simtheta;

  int all_nsimhit;
  int all_pidhit;
  int all_simproc;

  float all_vtxr;
  float all_vtxz;

  float all_simpx;
  float all_simpy;
  float all_simpz;

  float all_eloss;

  int all_trkid;

  float all_x1;
  float all_x2;
  float all_y1;
  float all_y2;
  float all_z1;
  float all_z2;

  float all_row1;
  float all_row2;
  float all_col1;
  float all_col2;

  float all_gx1;
  float all_gx2;
  float all_gy1;
  float all_gy2;
  float all_gz1;
  float all_gz2;

  float all_simtrketa;
  float all_simtrkphi;

  float all_clust_row;
  float all_clust_col;

  float all_clust_x;
  float all_clust_y;

  float all_clust_q;

  int all_clust_maxpixcol;
  int all_clust_maxpixrow;
  int all_clust_minpixcol;
  int all_clust_minpixrow;

  int all_clust_geoid;

  float all_clust_alpha;
  float all_clust_beta;

  static const int maxpix = 10000;
  float all_pixrow[maxpix];
  float all_pixcol[maxpix];
  float all_pixadc[maxpix];
  // Just added
  float all_pixx[maxpix];
  float all_pixy[maxpix];
  float all_pixgx[maxpix];
  float all_pixgy[maxpix];
  float all_pixgz[maxpix];
    
  float all_hit_probx;
  float all_hit_proby;
  float all_hit_cprob0;
  float all_hit_cprob1;
  float all_hit_cprob2;

  int all_pixel_split;
  float all_pixel_clst_err_x;
  float all_pixel_clst_err_y;


  // ----------------------------------

  TFile * tfile_;
  TTree * ttree_all_hits_;
  TTree * ttree_track_hits_;

  TTree * ttree_track_hits_strip_;
  
};

#endif
