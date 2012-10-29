
// Package:          SiPixelErrorEstimation
// Class:            SiPixelErrorEstimation
// Original Author:  Gavril Giurgiu (JHU)
// Created:          Fri May  4 17:48:24 CDT 2007

#include <iostream>
#include "CalibTracker/SiPixelErrorEstimation/interface/SiPixelErrorEstimation.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/DetId/interface/DetId.h" 
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

using namespace std;
using namespace edm;

SiPixelErrorEstimation::SiPixelErrorEstimation(const edm::ParameterSet& ps):tfile_(0), ttree_all_hits_(0), 
									    ttree_track_hits_(0), ttree_track_hits_strip_(0) 
{
  //Read config file
  outputFile_ = ps.getUntrackedParameter<string>( "outputFile", "SiPixelErrorEstimation_Ntuple.root" );
  
  // Replace  "ctfWithMaterialTracks" with "generalTracks"
  //src_ = ps.getUntrackedParameter<std::string>( "src", "ctfWithMaterialTracks" );
  src_ = ps.getUntrackedParameter<std::string>( "src", "generalTracks" );

  checkType_ = ps.getParameter<bool>( "checkType" );
  genType_ = ps.getParameter<int>( "genType" );
  include_trk_hits_ = ps.getParameter<bool>( "include_trk_hits" );
}

SiPixelErrorEstimation::~SiPixelErrorEstimation()
{}

void SiPixelErrorEstimation::beginJob()
{
   int bufsize = 64000;


  tfile_ = new TFile ( outputFile_.c_str() , "RECREATE");

  
  ttree_track_hits_strip_ = new TTree("TrackHitNtupleStrip", "TrackHitNtupleStrip");
  
  ttree_track_hits_strip_->Branch("strip_rechitx", &strip_rechitx, "strip_rechitx/F"    , bufsize);
  ttree_track_hits_strip_->Branch("strip_rechity", &strip_rechity, "strip_rechity/F"    , bufsize);
  ttree_track_hits_strip_->Branch("strip_rechitz", &strip_rechitz, "strip_rechitz/F"    , bufsize);
  
  ttree_track_hits_strip_->Branch("strip_rechiterrx", &strip_rechiterrx, "strip_rechiterrx/F" , bufsize);
  ttree_track_hits_strip_->Branch("strip_rechiterry", &strip_rechiterry, "strip_rechiterry/F" , bufsize);

  ttree_track_hits_strip_->Branch("strip_rechitresx", &strip_rechitresx, "strip_rechitresx/F" , bufsize);
  
  ttree_track_hits_strip_->Branch("strip_rechitresx2", &strip_rechitresx2, "strip_rechitresx2/F" , bufsize);


  ttree_track_hits_strip_->Branch("strip_rechitresy", &strip_rechitresy, "strip_rechitresy/F" , bufsize);
  
  ttree_track_hits_strip_->Branch("strip_rechitpullx", &strip_rechitpullx, "strip_rechitpullx/F", bufsize);
  ttree_track_hits_strip_->Branch("strip_rechitpully", &strip_rechitpully, "strip_rechitpully/F", bufsize);

  ttree_track_hits_strip_->Branch("strip_is_stereo", &strip_is_stereo, "strip_is_stereo/I", bufsize);
  ttree_track_hits_strip_->Branch("strip_hit_type" , &strip_hit_type , "strip_hit_type/I" , bufsize);
  ttree_track_hits_strip_->Branch("detector_type"  , &detector_type  , "detector_type/I"  , bufsize);

  ttree_track_hits_strip_->Branch("strip_trk_pt"   , &strip_trk_pt   , "strip_trk_pt/F"   , bufsize);
  ttree_track_hits_strip_->Branch("strip_cotalpha" , &strip_cotalpha , "strip_cotalpha/F" , bufsize);
  ttree_track_hits_strip_->Branch("strip_cotbeta"  , &strip_cotbeta  , "strip_cotbeta/F"  , bufsize);
  ttree_track_hits_strip_->Branch("strip_locbx"    , &strip_locbx    , "strip_locbx/F"    , bufsize);
  ttree_track_hits_strip_->Branch("strip_locby"    , &strip_locby    , "strip_locby/F"    , bufsize);
  ttree_track_hits_strip_->Branch("strip_locbz"    , &strip_locbz    , "strip_locbz/F"    , bufsize);
  ttree_track_hits_strip_->Branch("strip_charge"   , &strip_charge   , "strip_charge/F"   , bufsize);
  ttree_track_hits_strip_->Branch("strip_size"     , &strip_size     , "strip_size/I"     , bufsize);

  
  ttree_track_hits_strip_->Branch("strip_edge"   , &strip_edge   , "strip_edge/I"    , bufsize);
  ttree_track_hits_strip_->Branch("strip_nsimhit", &strip_nsimhit, "strip_nsimhit/I" , bufsize);
  ttree_track_hits_strip_->Branch("strip_pidhit" , &strip_pidhit , "strip_pidhit/I"  , bufsize);
  ttree_track_hits_strip_->Branch("strip_simproc", &strip_simproc, "strip_simproc/I" , bufsize);


ttree_track_hits_strip_->Branch("strip_subdet_id"             , &strip_subdet_id             , "strip_subdet_id/I"            , bufsize);
 
ttree_track_hits_strip_->Branch("strip_tib_layer"             , &strip_tib_layer             , "strip_tib_layer/I"            , bufsize);
ttree_track_hits_strip_->Branch("strip_tib_module"            , &strip_tib_module            , "strip_tib_module/I"           , bufsize);
ttree_track_hits_strip_->Branch("strip_tib_order"             , &strip_tib_order             , "strip_tib_order/I"            , bufsize);
ttree_track_hits_strip_->Branch("strip_tib_side"              , &strip_tib_side              , "strip_tib_side/I"             , bufsize);
ttree_track_hits_strip_->Branch("strip_tib_is_double_side"    , &strip_tib_is_double_side    , "strip_tib_is_double_side/I"   , bufsize);
ttree_track_hits_strip_->Branch("strip_tib_is_z_plus_side"    , &strip_tib_is_z_plus_side    , "strip_tib_is_z_plus_side/I"   , bufsize);
ttree_track_hits_strip_->Branch("strip_tib_is_z_minus_side"   , &strip_tib_is_z_minus_side   , "strip_tib_is_z_minus_side/I"  , bufsize);
ttree_track_hits_strip_->Branch("strip_tib_layer_number"      , &strip_tib_layer_number      , "strip_tib_layer_number/I"     , bufsize);
ttree_track_hits_strip_->Branch("strip_tib_string_number"     , &strip_tib_string_number     , "strip_tib_string_number/I"    , bufsize);
ttree_track_hits_strip_->Branch("strip_tib_module_number"     , &strip_tib_module_number     ,"strip_tib_module_number/I"     , bufsize);
ttree_track_hits_strip_->Branch("strip_tib_is_internal_string", &strip_tib_is_internal_string,"strip_tib_is_internal_string/I", bufsize);
ttree_track_hits_strip_->Branch("strip_tib_is_external_string", &strip_tib_is_external_string,"strip_tib_is_external_string/I", bufsize);
ttree_track_hits_strip_->Branch("strip_tib_is_rphi"           , &strip_tib_is_rphi           , "strip_tib_is_rphi/I"          , bufsize);
ttree_track_hits_strip_->Branch("strip_tib_is_stereo"         , &strip_tib_is_stereo         , "strip_tib_is_stereo/I"        , bufsize);
ttree_track_hits_strip_->Branch("strip_tob_layer"             , &strip_tob_layer             , "strip_tob_layer/I"            , bufsize);
ttree_track_hits_strip_->Branch("strip_tob_module"            , &strip_tob_module            , "strip_tob_module/I"           , bufsize);
ttree_track_hits_strip_->Branch("strip_tob_side"              , &strip_tob_side              , "strip_tob_side/I"             , bufsize);
ttree_track_hits_strip_->Branch("strip_tob_is_double_side"    , &strip_tob_is_double_side    , "strip_tob_is_double_side/I"   , bufsize);
ttree_track_hits_strip_->Branch("strip_tob_is_z_plus_side"    , &strip_tob_is_z_plus_side    , "strip_tob_is_z_plus_side/I"   , bufsize);
ttree_track_hits_strip_->Branch("strip_tob_is_z_minus_side"   , &strip_tob_is_z_minus_side   , "strip_tob_is_z_minus_side/I"  , bufsize);
ttree_track_hits_strip_->Branch("strip_tob_layer_number"      , &strip_tob_layer_number      , "strip_tob_layer_number/I"     , bufsize);
ttree_track_hits_strip_->Branch("strip_tob_rod_number"        , &strip_tob_rod_number        , "strip_tob_rod_number/I"       , bufsize);
ttree_track_hits_strip_->Branch("strip_tob_module_number"     , &strip_tob_module_number     , "strip_tob_module_number/I"    , bufsize);


ttree_track_hits_strip_->Branch("strip_prob", &strip_prob, "strip_prob/F"   , bufsize);
ttree_track_hits_strip_->Branch("strip_qbin", &strip_qbin, "strip_qbin/I", bufsize);

ttree_track_hits_strip_->Branch("strip_nprm", &strip_nprm, "strip_nprm/I", bufsize);

ttree_track_hits_strip_->Branch("strip_pidhit1" , &strip_pidhit1 , "strip_pidhit1/I"  , bufsize);
ttree_track_hits_strip_->Branch("strip_simproc1", &strip_simproc1, "strip_simproc1/I" , bufsize);

ttree_track_hits_strip_->Branch("strip_pidhit2" , &strip_pidhit2 , "strip_pidhit2/I"  , bufsize);
ttree_track_hits_strip_->Branch("strip_simproc2", &strip_simproc2, "strip_simproc2/I" , bufsize);

ttree_track_hits_strip_->Branch("strip_pidhit3" , &strip_pidhit3 , "strip_pidhit3/I"  , bufsize);
ttree_track_hits_strip_->Branch("strip_simproc3", &strip_simproc3, "strip_simproc3/I" , bufsize);

ttree_track_hits_strip_->Branch("strip_pidhit4" , &strip_pidhit4 , "strip_pidhit4/I"  , bufsize);
ttree_track_hits_strip_->Branch("strip_simproc4", &strip_simproc4, "strip_simproc4/I" , bufsize);

ttree_track_hits_strip_->Branch("strip_pidhit5" , &strip_pidhit5 , "strip_pidhit5/I"  , bufsize);
ttree_track_hits_strip_->Branch("strip_simproc5", &strip_simproc5, "strip_simproc5/I" , bufsize);

ttree_track_hits_strip_->Branch("strip_split", &strip_split, "strip_split/I" , bufsize);

ttree_track_hits_strip_->Branch("strip_clst_err_x", &strip_clst_err_x, "strip_clst_err_x/F"   , bufsize);
ttree_track_hits_strip_->Branch("strip_clst_err_y", &strip_clst_err_y, "strip_clst_err_y/F"   , bufsize);

 if ( include_trk_hits_ )
   {
     //tfile_ = new TFile ("SiPixelErrorEstimation_Ntuple.root" , "RECREATE");
     //const char* tmp_name = outputFile_.c_str();
     
     
     ttree_track_hits_ = new TTree("TrackHitNtuple", "TrackHitNtuple");
     
      ttree_track_hits_->Branch("evt", &evt, "evt/I", bufsize);
      ttree_track_hits_->Branch("run", &run, "run/I", bufsize);
      
      ttree_track_hits_->Branch("subdetId", &subdetId, "subdetId/I", bufsize);
      
      ttree_track_hits_->Branch("layer" , &layer , "layer/I" , bufsize);
      ttree_track_hits_->Branch("ladder", &ladder, "ladder/I", bufsize);
      ttree_track_hits_->Branch("mod"   , &mod   , "mod/I"   , bufsize);
      
      ttree_track_hits_->Branch("side"  , &side  , "side/I"  , bufsize);
      ttree_track_hits_->Branch("disk"  , &disk  , "disk/I"  , bufsize);
      ttree_track_hits_->Branch("blade" , &blade , "blade/I" , bufsize);
      ttree_track_hits_->Branch("panel" , &panel , "panel/I" , bufsize);
      ttree_track_hits_->Branch("plaq"  , &plaq  , "plaq/I"  , bufsize);
      
      ttree_track_hits_->Branch("half"   , &half   , "half/I"   , bufsize);
      ttree_track_hits_->Branch("flipped", &flipped, "flipped/I", bufsize);
      
      ttree_track_hits_->Branch("rechitx", &rechitx, "rechitx/F"    , bufsize);
      ttree_track_hits_->Branch("rechity", &rechity, "rechity/F"    , bufsize);
      ttree_track_hits_->Branch("rechitz", &rechitz, "rechitz/F"    , bufsize);
      
      ttree_track_hits_->Branch("rechiterrx", &rechiterrx, "rechiterrx/F" , bufsize);
      ttree_track_hits_->Branch("rechiterry", &rechiterry, "rechiterry/F" , bufsize);
      
      ttree_track_hits_->Branch("rechitresx", &rechitresx, "rechitresx/F" , bufsize);
      ttree_track_hits_->Branch("rechitresy", &rechitresy, "rechitresy/F" , bufsize);
      
      ttree_track_hits_->Branch("rechitpullx", &rechitpullx, "rechitpullx/F", bufsize);
      ttree_track_hits_->Branch("rechitpully", &rechitpully, "rechitpully/F", bufsize);

      
      ttree_track_hits_->Branch("npix"  , &npix  , "npix/I"  , bufsize);
      ttree_track_hits_->Branch("nxpix" , &nxpix , "nxpix/I" , bufsize);
      ttree_track_hits_->Branch("nypix" , &nypix , "nypix/I" , bufsize);
      ttree_track_hits_->Branch("charge", &charge, "charge/F", bufsize);
      
      ttree_track_hits_->Branch("edgex", &edgex, "edgex/I", bufsize);
      ttree_track_hits_->Branch("edgey", &edgey, "edgey/I", bufsize);
      
      ttree_track_hits_->Branch("bigx", &bigx, "bigx/I", bufsize);
      ttree_track_hits_->Branch("bigy", &bigy, "bigy/I", bufsize);
      
      ttree_track_hits_->Branch("alpha", &alpha, "alpha/F", bufsize);
      ttree_track_hits_->Branch("beta" , &beta , "beta/F" , bufsize);
      
      ttree_track_hits_->Branch("trk_alpha", &trk_alpha, "trk_alpha/F", bufsize);
      ttree_track_hits_->Branch("trk_beta" , &trk_beta , "trk_beta/F" , bufsize);

      ttree_track_hits_->Branch("phi", &phi, "phi/F", bufsize);
      ttree_track_hits_->Branch("eta", &eta, "eta/F", bufsize);
      
      ttree_track_hits_->Branch("simhitx", &simhitx, "simhitx/F", bufsize);
      ttree_track_hits_->Branch("simhity", &simhity, "simhity/F", bufsize);
      
      ttree_track_hits_->Branch("nsimhit", &nsimhit, "nsimhit/I", bufsize);
      ttree_track_hits_->Branch("pidhit" , &pidhit , "pidhit/I" , bufsize);
      ttree_track_hits_->Branch("simproc", &simproc, "simproc/I", bufsize);
      
      ttree_track_hits_->Branch("pixel_split", &pixel_split, "pixel_split/I", bufsize);

      ttree_track_hits_->Branch("pixel_clst_err_x", &pixel_clst_err_x, "pixel_clst_err_x/F"   , bufsize);
      ttree_track_hits_->Branch("pixel_clst_err_y", &pixel_clst_err_y, "pixel_clst_err_y/F"   , bufsize);

    } // if ( include_trk_hits_ )

  // ----------------------------------------------------------------------
  
  ttree_all_hits_ = new TTree("AllHitNtuple", "AllHitNtuple");

  ttree_all_hits_->Branch("evt", &evt, "evt/I", bufsize);
  ttree_all_hits_->Branch("run", &run, "run/I", bufsize);

  ttree_all_hits_->Branch("subdetid", &all_subdetid, "subdetid/I", bufsize);
  
  ttree_all_hits_->Branch("layer" , &all_layer , "layer/I" , bufsize);
  ttree_all_hits_->Branch("ladder", &all_ladder, "ladder/I", bufsize);
  ttree_all_hits_->Branch("mod"   , &all_mod   , "mod/I"   , bufsize);
  
  ttree_all_hits_->Branch("side"  , &all_side  , "side/I"  , bufsize);
  ttree_all_hits_->Branch("disk"  , &all_disk  , "disk/I"  , bufsize);
  ttree_all_hits_->Branch("blade" , &all_blade , "blade/I" , bufsize);
  ttree_all_hits_->Branch("panel" , &all_panel , "panel/I" , bufsize);
  ttree_all_hits_->Branch("plaq"  , &all_plaq  , "plaq/I"  , bufsize);

  ttree_all_hits_->Branch("half"   , &all_half   , "half/I"   , bufsize);
  ttree_all_hits_->Branch("flipped", &all_flipped, "flipped/I", bufsize);

  ttree_all_hits_->Branch("cols", &all_cols, "cols/I", bufsize);
  ttree_all_hits_->Branch("rows", &all_rows, "rows/I", bufsize);

  ttree_all_hits_->Branch("rechitx"    , &all_rechitx    , "rechitx/F"    , bufsize);
  ttree_all_hits_->Branch("rechity"    , &all_rechity    , "rechity/F"    , bufsize);
  ttree_all_hits_->Branch("rechitz"    , &all_rechitz    , "rechitz/F"    , bufsize);
 
  ttree_all_hits_->Branch("rechiterrx" , &all_rechiterrx , "rechiterrx/F" , bufsize);
  ttree_all_hits_->Branch("rechiterry" , &all_rechiterry , "rechiterry/F" , bufsize);
  
  ttree_all_hits_->Branch("rechitresx" , &all_rechitresx , "rechitresx/F" , bufsize);
  ttree_all_hits_->Branch("rechitresy" , &all_rechitresy , "rechitresy/F" , bufsize);
  
  ttree_all_hits_->Branch("rechitpullx", &all_rechitpullx, "rechitpullx/F", bufsize);
  ttree_all_hits_->Branch("rechitpully", &all_rechitpully, "rechitpully/F", bufsize);

  ttree_all_hits_->Branch("npix"  , &all_npix  , "npix/I"  , bufsize);
  ttree_all_hits_->Branch("nxpix" , &all_nxpix , "nxpix/I" , bufsize);
  ttree_all_hits_->Branch("nypix" , &all_nypix , "nypix/I" , bufsize);

  ttree_all_hits_->Branch("edgex", &all_edgex, "edgex/I", bufsize);
  ttree_all_hits_->Branch("edgey", &all_edgey, "edgey/I", bufsize);
 
  ttree_all_hits_->Branch("bigx", &all_bigx, "bigx/I", bufsize);
  ttree_all_hits_->Branch("bigy", &all_bigy, "bigy/I", bufsize);
  
  ttree_all_hits_->Branch("alpha", &all_alpha, "alpha/F", bufsize);
  ttree_all_hits_->Branch("beta" , &all_beta , "beta/F" , bufsize);

  ttree_all_hits_->Branch("simhitx", &all_simhitx, "simhitx/F", bufsize);
  ttree_all_hits_->Branch("simhity", &all_simhity, "simhity/F", bufsize);

  ttree_all_hits_->Branch("nsimhit", &all_nsimhit, "nsimhit/I", bufsize);
  ttree_all_hits_->Branch("pidhit" , &all_pidhit , "pidhit/I" , bufsize);
  ttree_all_hits_->Branch("simproc", &all_simproc, "simproc/I", bufsize);

  ttree_all_hits_->Branch("vtxr", &all_vtxr, "vtxr/F", bufsize);
  ttree_all_hits_->Branch("vtxz", &all_vtxz, "vtxz/F", bufsize);

  ttree_all_hits_->Branch("simpx", &all_simpx, "simpx/F", bufsize);
  ttree_all_hits_->Branch("simpy", &all_simpy, "simpy/F", bufsize);
  ttree_all_hits_->Branch("simpz", &all_simpz, "simpz/F", bufsize);

  ttree_all_hits_->Branch("eloss", &all_eloss, "eloss/F", bufsize);
  
  ttree_all_hits_->Branch("simphi", &all_simphi, "simphi/F", bufsize);
  ttree_all_hits_->Branch("simtheta", &all_simtheta, "simtheta/F", bufsize);
  
  ttree_all_hits_->Branch("trkid", &all_trkid, "trkid/I", bufsize);
  
  ttree_all_hits_->Branch("x1", &all_x1, "x1/F", bufsize);
  ttree_all_hits_->Branch("x2", &all_x2, "x2/F", bufsize);
  ttree_all_hits_->Branch("y1", &all_x1, "y1/F", bufsize);
  ttree_all_hits_->Branch("y2", &all_x2, "y2/F", bufsize);
  ttree_all_hits_->Branch("z1", &all_x1, "z1/F", bufsize);
  ttree_all_hits_->Branch("z2", &all_x2, "z2/F", bufsize);

  ttree_all_hits_->Branch("row1", &all_row1, "row1/F", bufsize);
  ttree_all_hits_->Branch("row2", &all_row2, "row2/F", bufsize);
  ttree_all_hits_->Branch("col1", &all_col1, "col1/F", bufsize);
  ttree_all_hits_->Branch("col2", &all_col2, "col2/F", bufsize);

  ttree_all_hits_->Branch("gx1", &all_gx1, "gx1/F", bufsize);
  ttree_all_hits_->Branch("gx2", &all_gx2, "gx2/F", bufsize);
  ttree_all_hits_->Branch("gy1", &all_gx1, "gy1/F", bufsize);
  ttree_all_hits_->Branch("gy2", &all_gx2, "gy2/F", bufsize);
  ttree_all_hits_->Branch("gz1", &all_gx1, "gz1/F", bufsize);
  ttree_all_hits_->Branch("gz2", &all_gx2, "gz2/F", bufsize);
  
  ttree_all_hits_->Branch("simtrketa", &all_simtrketa, "simtrketa/F", bufsize);
  ttree_all_hits_->Branch("simtrkphi", &all_simtrkphi, "simtrkphi/F", bufsize);

  ttree_all_hits_->Branch("clust_row", &all_clust_row, "clust_row/F", bufsize);
  ttree_all_hits_->Branch("clust_col", &all_clust_col, "clust_col/F", bufsize);
  
  ttree_all_hits_->Branch("clust_x", &all_clust_x, "clust_x/F", bufsize);
  ttree_all_hits_->Branch("clust_y", &all_clust_y, "clust_y/F", bufsize);

  ttree_all_hits_->Branch("clust_q", &all_clust_q, "clust_q/F", bufsize);

  ttree_all_hits_->Branch("clust_maxpixcol", &all_clust_maxpixcol, "clust_maxpixcol/I", bufsize);
  ttree_all_hits_->Branch("clust_maxpixrow", &all_clust_maxpixrow, "clust_maxpixrow/I", bufsize);
  ttree_all_hits_->Branch("clust_minpixcol", &all_clust_minpixcol, "clust_minpixcol/I", bufsize);
  ttree_all_hits_->Branch("clust_minpixrow", &all_clust_minpixrow, "clust_minpixrow/I", bufsize);
  
  ttree_all_hits_->Branch("clust_geoid", &all_clust_geoid, "clust_geoid/I", bufsize);
  
  ttree_all_hits_->Branch("clust_alpha", &all_clust_alpha, "clust_alpha/F", bufsize);
  ttree_all_hits_->Branch("clust_beta" , &all_clust_beta , "clust_beta/F" , bufsize);

  ttree_all_hits_->Branch("rowpix", all_pixrow, "row[npix]/F", bufsize);
  ttree_all_hits_->Branch("colpix", all_pixcol, "col[npix]/F", bufsize);
  ttree_all_hits_->Branch("adc", all_pixadc, "adc[npix]/F", bufsize);
  ttree_all_hits_->Branch("xpix", all_pixx, "x[npix]/F", bufsize);
  ttree_all_hits_->Branch("ypix", all_pixy, "y[npix]/F", bufsize);
  ttree_all_hits_->Branch("gxpix", all_pixgx, "gx[npix]/F", bufsize);
  ttree_all_hits_->Branch("gypix", all_pixgy, "gy[npix]/F", bufsize);
  ttree_all_hits_->Branch("gzpix", all_pixgz, "gz[npix]/F", bufsize);
  
  ttree_all_hits_->Branch("hit_probx", &all_hit_probx, "hit_probx/F" , bufsize);
  ttree_all_hits_->Branch("hit_proby", &all_hit_proby, "hit_proby/F" , bufsize);

  ttree_all_hits_->Branch("all_pixel_split", &all_pixel_split, "all_pixel_split/I" , bufsize);

  ttree_all_hits_->Branch("all_pixel_clst_err_x", &all_pixel_clst_err_x, "all_pixel_clst_err_x/F"   , bufsize);
  ttree_all_hits_->Branch("all_pixel_clst_err_y", &all_pixel_clst_err_y, "all_pixel_clst_err_y/F"   , bufsize);

}

void SiPixelErrorEstimation::endJob() 
{
  tfile_->Write();
  tfile_->Close();
}

void
SiPixelErrorEstimation::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  using namespace edm;
  
  run = e.id().run();
  evt = e.id().event();
  
  if ( evt%1000 == 0 ) 
    cout << "evt = " << evt << endl;

  float math_pi = 3.14159265;
  float radtodeg = 180.0 / math_pi;
    
  LocalPoint position;
  LocalError error;
  float mindist = 999999.9;

  std::vector<PSimHit> matched;
  TrackerHitAssociator associate(e);

  edm::ESHandle<TrackerGeometry> pDD;
  es.get<TrackerDigiGeometryRecord> ().get (pDD);
  const TrackerGeometry* tracker = &(* pDD);
  

  //cout << "...1..." << endl;


  edm::ESHandle<MagneticField> magneticField;
  es.get<IdealMagneticFieldRecord>().get( magneticField );
  //const MagneticField* magField_ = magFieldHandle.product();

  edm::FileInPath FileInPath_;
    


  // Strip hits ==============================================================================================================


  edm::Handle<vector<Trajectory> > trajCollectionHandle;
  
  e.getByLabel( src_, trajCollectionHandle);
  //e.getByLabel( "generalTracks", trajCollectionHandle);

  for ( vector<Trajectory>::const_iterator it = trajCollectionHandle->begin(); it!=trajCollectionHandle->end(); ++it )
    {
      
      vector<TrajectoryMeasurement> tmColl = it->measurements();
      for ( vector<TrajectoryMeasurement>::const_iterator itTraj = tmColl.begin(); itTraj!=tmColl.end(); ++itTraj )
	{
	  
	  if ( !itTraj->updatedState().isValid() ) 
	    continue;
       
	  strip_rechitx     = -9999999.9;
	  strip_rechity     = -9999999.9;
	  strip_rechitz     = -9999999.9;
	  strip_rechiterrx  = -9999999.9;
	  strip_rechiterry  = -9999999.9;
	  strip_rechitresx  = -9999999.9;
	  strip_rechitresx2  = -9999999.9;
	  

	  strip_rechitresy  = -9999999.9;
	  strip_rechitpullx = -9999999.9;
	  strip_rechitpully = -9999999.9;
	  strip_is_stereo   = -9999999  ;
	  strip_hit_type    = -9999999  ;
	  detector_type     = -9999999  ;
	  
	  strip_trk_pt    = -9999999.9;
	  strip_cotalpha  = -9999999.9;
	  strip_cotbeta   = -9999999.9;
	  strip_locbx     = -9999999.9;
	  strip_locby     = -9999999.9;
	  strip_locbz     = -9999999.9;
	  strip_charge    = -9999999.9;
	  strip_size        = -9999999  ;

	  strip_edge        = -9999999  ;
	  strip_nsimhit     = -9999999  ;
	  strip_pidhit      = -9999999  ;
	  strip_simproc     = -9999999  ;


	  strip_subdet_id = -9999999;
 
	  strip_tib_layer             = -9999999;
	  strip_tib_module            = -9999999;
	  strip_tib_order             = -9999999;
	  strip_tib_side              = -9999999;
	  strip_tib_is_double_side    = -9999999;
	  strip_tib_is_z_plus_side    = -9999999;
	  strip_tib_is_z_minus_side   = -9999999;
	  strip_tib_layer_number      = -9999999;
	  strip_tib_string_number     = -9999999;
	  strip_tib_module_number     = -9999999;
	  strip_tib_is_internal_string= -9999999;
	  strip_tib_is_external_string= -9999999;
	  strip_tib_is_rphi           = -9999999;
	  strip_tib_is_stereo         = -9999999;          
	  
	  strip_tob_layer             = -9999999;
	  strip_tob_module            = -9999999;
	  strip_tob_side              = -9999999;
	  strip_tob_is_double_side    = -9999999;
	  strip_tob_is_z_plus_side    = -9999999;
	  strip_tob_is_z_minus_side   = -9999999;
	  strip_tob_layer_number      = -9999999;
	  strip_tob_rod_number        = -9999999;
	  strip_tob_module_number     = -9999999;
	  	  
	  strip_tob_is_rphi           = -9999999;
	  strip_tob_is_stereo         = -9999999;     

	  strip_prob                  = -9999999.9;
	  strip_qbin                  = -9999999;

	  strip_nprm                  = -9999999;
	  
	  strip_pidhit1      = -9999999  ;
	  strip_simproc1     = -9999999  ;
	  
	  strip_pidhit2      = -9999999  ;
	  strip_simproc2     = -9999999  ;
	  
	  strip_pidhit3      = -9999999  ;
	  strip_simproc3     = -9999999  ;
	  
	  strip_pidhit4      = -9999999  ;
	  strip_simproc4     = -9999999  ;

	  strip_pidhit5      = -9999999  ;
	  strip_simproc5     = -9999999  ;

	  strip_split        = -9999999;   
	  strip_clst_err_x   = -9999999.9;
	  strip_clst_err_y   = -9999999.9;

	  const TransientTrackingRecHit::ConstRecHitPointer trans_trk_rec_hit_point = itTraj->recHit();
	 
	  if ( trans_trk_rec_hit_point == NULL )
	    continue;

	  const TrackingRecHit *trk_rec_hit = (*trans_trk_rec_hit_point).hit();

	  if ( trk_rec_hit == NULL )
	    continue;

	  DetId detid = (trk_rec_hit)->geographicalId();

	  strip_subdet_id = (int)detid.subdetId();
	  
	  if ( (int)detid.subdetId() != (int)(StripSubdetector::TIB) && (int)detid.subdetId() != (int)(StripSubdetector::TOB) )
	    continue;

	  const SiStripMatchedRecHit2D* matchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>( (*trans_trk_rec_hit_point).hit() );
	  const SiStripRecHit2D       * hit2d      = dynamic_cast<const SiStripRecHit2D       *>( (*trans_trk_rec_hit_point).hit() );
	  const SiStripRecHit1D       * hit1d      = dynamic_cast<const SiStripRecHit1D       *>( (*trans_trk_rec_hit_point).hit() );
	  
	  if ( !matchedhit && !hit2d && !hit1d )
	    continue;
	  
	  position = (trk_rec_hit)->localPosition(); 
	  error = (trk_rec_hit)->localPositionError();
	  
	  strip_rechitx = position.x();
	  strip_rechity = position.y();
	  strip_rechitz = position.z();
	  strip_rechiterrx = sqrt( error.xx() );
	  strip_rechiterry = sqrt( error.yy() );
	  
	  
	  //cout << "strip_rechitx = " << strip_rechitx << endl;	      
	  //cout << "strip_rechity = " << strip_rechity << endl;
	  //cout << "strip_rechitz = " << strip_rechitz << endl;
	  
	  //cout << "strip_rechiterrx = " << strip_rechiterrx << endl;
	  //cout << "strip_rechiterry = " << strip_rechiterry << endl;
	  
	  TrajectoryStateOnSurface tsos = itTraj->updatedState(); 
	  
	  strip_trk_pt = tsos.globalMomentum().perp();
	  
	  LocalTrajectoryParameters ltp = tsos.localParameters();
	  
	  LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
	  
	  float locx = localDir.x();
	  float locy = localDir.y();
	  float locz = localDir.z();
	  
	  //alpha_ = atan2( locz, locx );
	  //beta_  = atan2( locz, locy );
	  
	  strip_cotalpha = locx/locz;
	  strip_cotbeta  = locy/locz;
	      

	  StripSubdetector StripSubdet = (StripSubdetector)detid;

	  if ( StripSubdet.stereo() == 0 )
	    strip_is_stereo = 0;
	  else
	    strip_is_stereo = 1;


	  SiStripDetId si_strip_det_id = SiStripDetId( detid );
	  
	  //const StripGeomDetUnit* strip_geom_det_unit = dynamic_cast<const StripGeomDetUnit*> ( tracker->idToDet( detid ) );
	  const StripGeomDetUnit* strip_geom_det_unit = (const StripGeomDetUnit*)tracker->idToDetUnit( detid );
	  
	  if ( strip_geom_det_unit != NULL )
	    {
	      LocalVector lbfield 
		= (strip_geom_det_unit->surface()).toLocal( (*magneticField).inTesla( strip_geom_det_unit->surface().position() ) ); 
	      
	      strip_locbx = lbfield.x();
	      strip_locby = lbfield.y();
	      strip_locbz = lbfield.z();
	    }
	  

	  //enum ModuleGeometry {UNKNOWNGEOMETRY, IB1, IB2, OB1, OB2, W1A, W2A, W3A, W1B, W2B, W3B, W4, W5, W6, W7};
	  
	  if ( si_strip_det_id.moduleGeometry() == 1 )
	    {
	      detector_type = 1;
	      //cout << "si_strip_det_id.moduleGeometry() = IB1" << endl;
	      //cout << "si_strip_det_id.moduleGeometry() = " << si_strip_det_id.moduleGeometry() << endl;
	    }
	  else if ( si_strip_det_id.moduleGeometry() == 2 )
	    {
	      detector_type = 2;
	      //cout << "si_strip_det_id.moduleGeometry() = IB2" << endl;
	      //cout << "si_strip_det_id.moduleGeometry() = " << si_strip_det_id.moduleGeometry() << endl;
	    }
	  else if ( si_strip_det_id.moduleGeometry() == 3 )
	    {
	      detector_type = 3;
	      //cout << "si_strip_det_id.moduleGeometry() = OB1" << endl;
	      //cout << "si_strip_det_id.moduleGeometry() = " << si_strip_det_id.moduleGeometry() << endl;
	    }
	  else if ( si_strip_det_id.moduleGeometry() == 4 )
	    {
	      detector_type = 4;
	      //cout << "si_strip_det_id.moduleGeometry() = OB2" << endl;
	      //cout << "si_strip_det_id.moduleGeometry() = " << si_strip_det_id.moduleGeometry() << endl;
	    }
	  

	  // Store ntuple variables 
	  
	  if ( (int)detid.subdetId() == int(StripSubdetector::TIB) )
	    {
	      
	      TIBDetId tib_detid( detid );
	      
	      strip_tib_layer              = (int)tib_detid.layer();
	      strip_tib_module             = (int)tib_detid.module(); 
	      strip_tib_order              = (int)tib_detid.order();
	      strip_tib_side               = (int)tib_detid.side();
	      strip_tib_is_double_side     = (int)tib_detid.isDoubleSide();
	      strip_tib_is_z_plus_side     = (int)tib_detid.isZPlusSide();
	      strip_tib_is_z_minus_side    = (int)tib_detid.isZMinusSide();
	      strip_tib_layer_number       = (int)tib_detid.layerNumber();
	      strip_tib_string_number      = (int)tib_detid.stringNumber() ;
	      strip_tib_module_number      = (int)tib_detid.moduleNumber();
	      strip_tib_is_internal_string = (int)tib_detid.isInternalString();
	      strip_tib_is_external_string = (int)tib_detid.isExternalString();
	      strip_tib_is_rphi            = (int)tib_detid.isRPhi();
	      strip_tib_is_stereo          = (int)tib_detid.isStereo();
	    }
	  
	  
	  if ( (int)detid.subdetId() == int(StripSubdetector::TOB) )
	    {
	      
	      TOBDetId tob_detid( detid );
	      
	      strip_tob_layer              = (int)tob_detid.layer();
	      strip_tob_module             = (int)tob_detid.module();
	      
	      strip_tob_side               = (int)tob_detid.side();
	      strip_tob_is_double_side     = (int)tob_detid.isDoubleSide();
	      strip_tob_is_z_plus_side     = (int)tob_detid.isZPlusSide();
	      strip_tob_is_z_minus_side    = (int)tob_detid.isZMinusSide();
	      strip_tob_layer_number       = (int)tob_detid.layerNumber();
	      strip_tob_rod_number         = (int)tob_detid.rodNumber();
	      strip_tob_module_number      = (int)tob_detid.moduleNumber();
	      
	      
	      strip_tob_is_rphi            = (int)tob_detid.isRPhi();
	      strip_tob_is_stereo          = (int)tob_detid.isStereo();
	          
	    }
	 
	 
	  if ( matchedhit ) 
	    {
	      //cout << endl << endl << endl;
	      //cout << "Found a SiStripMatchedRecHit2D !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << endl;
	      //cout << endl << endl << endl;
	      strip_hit_type = 0;

	    } //  if ( matchedhit )

   
	  if ( hit1d )
	    {
	      strip_hit_type = 1;

	      const SiStripRecHit1D::ClusterRef & cluster = hit1d->cluster();

	      if ( cluster->getSplitClusterError()  > 0.0 )
		strip_split = 1;
	      else 
		strip_split = 0;
	      
	      strip_clst_err_x = cluster->getSplitClusterError();
	      //strip_clst_err_y = ... 

	      // Get cluster total charge
	      const std::vector<uint8_t>& stripCharges = cluster->amplitudes();
	      uint16_t charge = 0;
	      for (unsigned int i = 0; i < stripCharges.size(); ++i) 
		{
		  charge += stripCharges.at(i);
		}
	      
	      strip_charge = (float)charge;
	      strip_size = (int)( (cluster->amplitudes()).size() );


	      // Association of the rechit to the simhit
	      float mindist = 999999.9;
	      matched.clear();  
	      
	      matched = associate.associateHit(*hit1d);

	      strip_nsimhit = (int)matched.size();
	      
	      if ( !matched.empty()) 
		{
		  PSimHit closest;
		  
		  // Get the closest simhit
		  
		  int strip_nprimaries = 0; 
		  int current_index = 0;
		 
		  for ( vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); ++m)
		    {
		      ++current_index;
		      
		      if ( (*m).processType() == 2 )
			++strip_nprimaries;

		      if ( current_index == 1 )
			{
			  strip_pidhit1 = (*m).particleType();
			  strip_simproc1 = (*m).processType();
			}
		      else if ( current_index == 2 )
			{
			  strip_pidhit2 = (*m).particleType();
			  strip_simproc2 = (*m).processType();
			}
		      else if ( current_index == 3 )
			{
			  strip_pidhit3 = (*m).particleType();
			  strip_simproc3 = (*m).processType();
			}
		      else if ( current_index == 4 )
			{
			  strip_pidhit4 = (*m).particleType();
			  strip_simproc4 = (*m).processType();
			}
		      else if ( current_index == 5 )
			{
			  strip_pidhit5 = (*m).particleType();
			  strip_simproc5 = (*m).processType();
			}


		      float dist = abs( (hit1d)->localPosition().x() - (*m).localPosition().x() );
		      
		      if ( dist<mindist )
			{
			  mindist = dist;
			  closest = (*m);
			}
		    
		    } // for ( vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); ++m)
		  
		  strip_nprm = strip_nprimaries;

		  strip_rechitresx = strip_rechitx - closest.localPosition().x();
		  strip_rechitresy = strip_rechity - closest.localPosition().y();
 
		  strip_rechitresx2 = strip_rechitx - matched[0].localPosition().x();

		  strip_rechitpullx = strip_rechitresx / strip_rechiterrx;
		  strip_rechitpully = strip_rechitresy / strip_rechiterry;

		  strip_pidhit = closest.particleType();
		  strip_simproc = closest.processType();

		  
		} //   if( !matched.empty()) 
		  

	      //strip_prob = (hit1d)->getTemplProb();
	      //strip_qbin = (hit1d)->getTemplQbin();

	      //cout << endl;
	      //cout << "SiPixelErrorEstimation 1d hit: " << endl;
	      //cout << "prob 1d = " << strip_prob << endl;
	      //cout << "qbin 1d = " << strip_qbin << endl;
	      //cout << endl;

	      
	      // Is the cluster on edge ?
	      /*
		SiStripDetInfoFileReader* reader;
		
		FileInPath_("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
		
		reader = new SiStripDetInfoFileReader( FileInPath_.fullPath() );
		
		uint16_t firstStrip = cluster->firstStrip();
		uint16_t lastStrip = firstStrip + (cluster->amplitudes()).size() -1;
		unsigned short Nstrips;
		Nstrips = reader->getNumberOfApvsAndStripLength(id1).first*128;
		
		if ( firstStrip == 0 || lastStrip == (Nstrips-1) ) 
		strip_edge = 1;
		else
		strip_edge = 0;
	      */

	    } // if ( hit1d )


	  if ( hit2d )
	    {
	      strip_hit_type = 2;
	      
	      const SiStripRecHit1D::ClusterRef & cluster = hit2d->cluster();
	      
	      //if ( cluster->getSplitClusterError()  > 0.0 )
	      //strip_split = 1;
	      //else 
	      //strip_split = 0;

	      //strip_clst_err_x = cluster->getSplitClusterError();
	      //strip_clst_err_y = ... 

	      // Get cluster total charge
	      const std::vector<uint8_t>& stripCharges = cluster->amplitudes();
	      uint16_t charge = 0;
	      for (unsigned int i = 0; i < stripCharges.size(); ++i) 
		{
		  charge += stripCharges.at(i);
		}
	      
	      strip_charge = (float)charge;
	      strip_size = (int)( (cluster->amplitudes()).size() );
	     
	      // Association of the rechit to the simhit
	      float mindist = 999999.9;
	      matched.clear();  
	      
	      matched = associate.associateHit(*hit2d);

	      strip_nsimhit = (int)matched.size();
	      
	      if ( !matched.empty()) 
		{
		  PSimHit closest;
		  
		  // Get the closest simhit
		  
		  for ( vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); ++m)
		    {
		      float dist = abs( (hit2d)->localPosition().x() - (*m).localPosition().x() );
		      
		      if ( dist<mindist )
			{
			  mindist = dist;
			  closest = (*m);
			}
		    }
		  
		  strip_rechitresx = strip_rechitx - closest.localPosition().x();
		  strip_rechitresy = strip_rechity - closest.localPosition().y();
 
		  strip_rechitpullx = strip_rechitresx / strip_rechiterrx;
		  strip_rechitpully = strip_rechitresy / strip_rechiterry;

		  strip_pidhit = closest.particleType();
		  strip_simproc = closest.processType();

		  
		} //   if( !matched.empty()) 
		  

	      //strip_prob = (hit2d)->getTemplProb();
	      //strip_qbin = (hit2d)->getTemplQbin();

	      //cout << endl;
	      //cout << "SiPixelErrorEstimation 2d hit: " << endl;
	      //cout << "prob 2d = " << strip_prob << endl;
	      //cout << "qbin 2d = " << strip_qbin << endl;
	      //cout << endl;
		  
	      
	    } // if ( hit2d )


	  ttree_track_hits_strip_->Fill();


	} // for ( vector<TrajectoryMeasurement>::const_iterator itTraj = tmColl.begin(); itTraj!=tmColl.end(); ++itTraj )

    } // for( vector<Trajectory>::const_iterator it = trajCollectionHandle->begin(); it!=trajCollectionHandle->end(); ++it)








  //cout << "...2..." << endl;






  // --------------------------------------- all hits -----------------------------------------------------------
  edm::Handle<SiPixelRecHitCollection> recHitColl;
  e.getByLabel( "siPixelRecHits", recHitColl);

  Handle<edm::SimTrackContainer> simtracks;
  e.getByLabel("g4SimHits", simtracks);

  //-----Iterate over detunits
  for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) 
    {
      DetId detId = ((*it)->geographicalId());
      
      SiPixelRecHitCollection::const_iterator dsmatch = recHitColl->find(detId);
      if (dsmatch == recHitColl->end()) continue;

      SiPixelRecHitCollection::DetSet pixelrechitRange = *dsmatch;
      SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.begin();
      SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorEnd = pixelrechitRange.end();
      SiPixelRecHitCollection::DetSet::const_iterator pixeliter = pixelrechitRangeIteratorBegin;
      std::vector<PSimHit> matched;
      
      //----Loop over rechits for this detId
      for ( ; pixeliter != pixelrechitRangeIteratorEnd; ++pixeliter) 
	{
	  matched.clear();
	  matched = associate.associateHit(*pixeliter);
	  // only consider rechits that have associated simhit
	  // otherwise cannot determine residiual
	  if ( matched.empty() )
	    {
	      cout << "SiPixelErrorEstimation::analyze: rechits without associated simhit !!!!!!!" 
		   << endl;
	      continue;
	    }
		
	  all_subdetid = -9999;
	  
	  all_layer = -9999;
	  all_ladder = -9999;
	  all_mod = -9999;
	  
	  all_side = -9999;
	  all_disk = -9999;
	  all_blade = -9999;
	  all_panel = -9999;
	  all_plaq = -9999;
	  
	  all_half = -9999;
	  all_flipped = -9999;
	  
	  all_cols = -9999;
	  all_rows = -9999;
	  
	  all_rechitx = -9999;
	  all_rechity = -9999;
	  all_rechitz = -9999;
	  
	  all_simhitx = -9999;
	  all_simhity = -9999;

	  all_rechiterrx = -9999;
	  all_rechiterry = -9999;
	  
	  all_rechitresx = -9999;
	  all_rechitresy = -9999;
	  
	  all_rechitpullx = -9999;
	  all_rechitpully = -9999;
	  
	  all_npix = -9999;
	  all_nxpix = -9999;
	  all_nypix = -9999;
	 	  
	  all_edgex = -9999;
	  all_edgey = -9999;
	  
	  all_bigx = -9999;
	  all_bigy = -9999;
	  
	  all_alpha = -9999;
	  all_beta = -9999;
	  
	  all_simphi = -9999;
	  all_simtheta = -9999;
	  
	  all_simhitx = -9999;
	  all_simhity = -9999;
	  
	  all_nsimhit = -9999;
	  all_pidhit = -9999;
	  all_simproc = -9999;
	  
	  all_vtxr = -9999;
	  all_vtxz = -9999;
	  
	  all_simpx = -9999;
	  all_simpy = -9999;
	  all_simpz = -9999;
	  
	  all_eloss = -9999;
	  	  
	  all_trkid = -9999;
	  
	  all_x1 = -9999;
	  all_x2 = -9999;
	  all_y1 = -9999;
	  all_y2 = -9999;
	  all_z1 = -9999;
	  all_z2 = -9999;
	  
	  all_row1 = -9999;
	  all_row2 = -9999;
	  all_col1 = -9999;
	  all_col2 = -9999;
	  
	  all_gx1 = -9999;
	  all_gx2 = -9999;
	  all_gy1 = -9999;
	  all_gy2 = -9999;
	  all_gz1 = -9999;
	  all_gz2 = -9999;
	  
	  all_simtrketa = -9999;
	  all_simtrkphi = -9999;
	  
	  all_clust_row = -9999;
	  all_clust_col = -9999;
	  
	  all_clust_x = -9999;
	  all_clust_y = -9999;
	  
	  all_clust_q = -9999;
	  
	  all_clust_maxpixcol = -9999;
	  all_clust_maxpixrow = -9999;
	  all_clust_minpixcol = -9999;
	  all_clust_minpixrow = -9999;
	  
	  all_clust_geoid = -9999;
	  
	  all_clust_alpha = -9999;
	  all_clust_beta = -9999;
	  
	  /*
	    for (int i=0; i<all_npix; ++i)
	    {
	    all_pixrow[i] = -9999;
	    all_pixcol[i] = -9999;
	    all_pixadc[i] = -9999;
	    all_pixx[i] = -9999;
	    all_pixy[i] = -9999;
	    all_pixgx[i] = -9999;
	    all_pixgy[i] = -9999;
	    all_pixgz[i] = -9999;
	    }
	  */

	  all_hit_probx = -9999;
	  all_hit_proby = -9999;
	  all_hit_cprob0 = -9999;
	  all_hit_cprob1 = -9999;
	  all_hit_cprob2 = -9999;

	  all_pixel_split = -9999;
	  all_pixel_clst_err_x = -9999.9;
	  all_pixel_clst_err_y = -9999.9;

	  all_nsimhit = (int)matched.size();
	  
	  all_subdetid = (int)detId.subdetId();
	  // only consider rechits in pixel barrel and pixel forward 
	  if ( !(all_subdetid==1 || all_subdetid==2) ) 
	    {
	      cout << "SiPixelErrorEstimation::analyze: Not in a pixel detector !!!!!" << endl; 
	      continue;
	    }

	  const PixelGeomDetUnit* theGeomDet 
	    = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(detId) );
	  
	  const PixelTopology* topol = &(theGeomDet->specificTopology());

	  if ( pixeliter->cluster()->getSplitClusterErrorX()  > 0.0 && 
	       pixeliter->cluster()->getSplitClusterErrorY()  > 0.0 )
	    {
	      all_pixel_split = 1;
	    }
	  else
	    {
	      all_pixel_split = 0;
	    }
	
	  all_pixel_clst_err_x = pixeliter->cluster()->getSplitClusterErrorX();
	  all_pixel_clst_err_y = pixeliter->cluster()->getSplitClusterErrorY();


	  const int maxPixelCol = pixeliter->cluster()->maxPixelCol();
	  const int maxPixelRow = pixeliter->cluster()->maxPixelRow();
	  const int minPixelCol = pixeliter->cluster()->minPixelCol();
	  const int minPixelRow = pixeliter->cluster()->minPixelRow();
	  
	  //all_hit_probx  = (float)pixeliter->probabilityX();
	  //all_hit_proby  = (float)pixeliter->probabilityY();
	  all_hit_cprob0 = (float)pixeliter->clusterProbability(0);
	  all_hit_cprob1 = (float)pixeliter->clusterProbability(1);
	  all_hit_cprob2 = (float)pixeliter->clusterProbability(2);
	  
	  // check whether the cluster is at the module edge 
	  if ( topol->isItEdgePixelInX( minPixelRow ) || 
	       topol->isItEdgePixelInX( maxPixelRow ) )
	    all_edgex = 1;
	  else 
	    all_edgex = 0;
	  
	  if ( topol->isItEdgePixelInY( minPixelCol ) || 
	       topol->isItEdgePixelInY( maxPixelCol ) )
	    all_edgey = 1;
	  else 
	    all_edgey = 0;
	  
	  // check whether this rechit contains big pixels
	  if ( topol->containsBigPixelInX(minPixelRow, maxPixelRow) )
	    all_bigx = 1;
	  else 
	    all_bigx = 0;
	  
	  if ( topol->containsBigPixelInY(minPixelCol, maxPixelCol) )
	    all_bigy = 1;
	  else 
	    all_bigy = 0;
	  
	  if ( (int)detId.subdetId() == (int)PixelSubdetector::PixelBarrel ) 
	    {
	      PXBDetId bdetid(detId);
	      all_layer = bdetid.layer();
	      all_ladder = bdetid.ladder();
	      all_mod = bdetid.module();
	      
	      int tmp_nrows = theGeomDet->specificTopology().nrows();
	      if ( tmp_nrows == 80 ) 
		all_half = 1;
	      else if ( tmp_nrows == 160 ) 
		all_half = 0;
	      else 
		cout << "-------------------------------------------------- Wrong module size !!!" << endl;
	      
	      float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
	      float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
	      
	      if ( tmp2<tmp1 ) 
		all_flipped = 1;
	      else 
		all_flipped = 0;
	    }
	  else if ( (int)detId.subdetId() == (int)PixelSubdetector::PixelEndcap )
	    {
	      PXFDetId fdetid(detId);
	      all_side  = fdetid.side();
	      all_disk  = fdetid.disk();
	      all_blade = fdetid.blade();
	      all_panel = fdetid.panel();
	      all_plaq  = fdetid.module(); // also known as plaquette
	      
	    } // else if ( detId.subdetId()==PixelSubdetector::PixelEndcap )
	  else std::cout << "We are not in the pixel detector" << (int)detId.subdetId() << endl;
	  
	  all_cols = theGeomDet->specificTopology().ncolumns();
	  all_rows = theGeomDet->specificTopology().nrows();
	  	  
	  LocalPoint lp = pixeliter->localPosition();
	  // gavril: change this name
	  all_rechitx = lp.x();
	  all_rechity = lp.y();
	  all_rechitz = lp.z();
	  
	  LocalError le = pixeliter->localPositionError();
	  all_rechiterrx = sqrt( le.xx() );
	  all_rechiterry = sqrt( le.yy() );

	  bool found_hit_from_generated_particle = false;
	  
	  //---Loop over sim hits, fill closest
	  float closest_dist = 99999.9;
	  std::vector<PSimHit>::const_iterator closest_simhit = matched.begin();
	  
	  for (std::vector<PSimHit>::const_iterator m = matched.begin(); m < matched.end(); m++) 
	    {
	      if ( checkType_ )
		{
		  int pid = (*m).particleType();
		  if ( abs(pid) != genType_ )
		    continue;
		} 
	      
	      float simhitx = 0.5 * ( (*m).entryPoint().x() + (*m).exitPoint().x() );
	      float simhity = 0.5 * ( (*m).entryPoint().y() + (*m).exitPoint().y() );
	      
	      float x_res = simhitx - rechitx;
	      float y_res = simhity - rechity;
		  
	      float dist = sqrt( x_res*x_res + y_res*y_res );		  
	      
	      if ( dist < closest_dist ) 
		{
		  closest_dist = dist;
		  closest_simhit = m;
		  found_hit_from_generated_particle = true;
		} 
	    } // end sim hit loop
	  
	  // If this recHit does not have any simHit with the same particleType as the particles generated
	  // ignore it as most probably comes from delta rays.
	  if ( checkType_ && !found_hit_from_generated_particle )
	    continue; 

	  all_x1 = (*closest_simhit).entryPoint().x(); // width (row index, in col direction)
	  all_y1 = (*closest_simhit).entryPoint().y(); // length (col index, in row direction) 
	  all_z1 = (*closest_simhit).entryPoint().z(); 
	  all_x2 = (*closest_simhit).exitPoint().x();
	  all_y2 = (*closest_simhit).exitPoint().y();
	  all_z2 = (*closest_simhit).exitPoint().z();
	  GlobalPoint GP1 = 
	    theGeomDet->surface().toGlobal( Local3DPoint( (*closest_simhit).entryPoint().x(),
							  (*closest_simhit).entryPoint().y(),
							  (*closest_simhit).entryPoint().z() ) );
	  GlobalPoint GP2 = 
	    theGeomDet->surface().toGlobal (Local3DPoint( (*closest_simhit).exitPoint().x(),
							  (*closest_simhit).exitPoint().y(),
							  (*closest_simhit).exitPoint().z() ) );
	  all_gx1 = GP1.x();
	  all_gx2 = GP2.x();
	  all_gy1 = GP1.y();
	  all_gy2 = GP2.y();
	  all_gz1 = GP1.z();
	  all_gz2 = GP2.z();
	  
	  MeasurementPoint mp1 =
	    topol->measurementPosition( LocalPoint( (*closest_simhit).entryPoint().x(),
						    (*closest_simhit).entryPoint().y(),
						    (*closest_simhit).entryPoint().z() ) );
	  MeasurementPoint mp2 =
	    topol->measurementPosition( LocalPoint( (*closest_simhit).exitPoint().x(),
						    (*closest_simhit).exitPoint().y(), 
						    (*closest_simhit).exitPoint().z() ) );
	  all_row1 = mp1.x();
	  all_col1 = mp1.y();
	  all_row2 = mp2.x();
	  all_col2 = mp2.y();
	  
	  all_simhitx = 0.5*(all_x1+all_x2);  
	  all_simhity = 0.5*(all_y1+all_y2);  
	  
	  all_rechitresx = all_rechitx - all_simhitx;
	  all_rechitresy = all_rechity - all_simhity;

	  all_rechitpullx = all_rechitresx / all_rechiterrx;
	  all_rechitpully = all_rechitresy / all_rechiterry;
	  
	  SiPixelRecHit::ClusterRef const& clust = pixeliter->cluster();
	  
	  all_npix = clust->size();
	  all_nxpix = clust->sizeX();
	  all_nypix = clust->sizeY();

	  all_clust_row = clust->x();
	  all_clust_col = clust->y();
	  
	  LocalPoint lp2 = topol->localPosition( MeasurementPoint( all_clust_row, all_clust_col ) );
	  all_clust_x = lp2.x();
	  all_clust_y = lp2.y();

	  all_clust_q = clust->charge();

	  all_clust_maxpixcol = clust->maxPixelCol();
	  all_clust_maxpixrow = clust->maxPixelRow();
	  all_clust_minpixcol = clust->minPixelCol();
	  all_clust_minpixrow = clust->minPixelRow();
	  
	  all_clust_geoid = 0;  // never set!
  
	  all_simpx  = (*closest_simhit).momentumAtEntry().x();
	  all_simpy  = (*closest_simhit).momentumAtEntry().y();
	  all_simpz  = (*closest_simhit).momentumAtEntry().z();
	  all_eloss = (*closest_simhit).energyLoss();
	  all_simphi   = (*closest_simhit).phiAtEntry();
	  all_simtheta = (*closest_simhit).thetaAtEntry();
	  all_pidhit = (*closest_simhit).particleType();
	  all_trkid = (*closest_simhit).trackId();
	  
	  //--- Fill alpha and beta -- more useful for exploring the residuals...
	  all_beta  = atan2(all_simpz, all_simpy);
	  all_alpha = atan2(all_simpz, all_simpx);
	  
	  all_simproc = (int)closest_simhit->processType();
	  
	  const edm::SimTrackContainer& trks = *(simtracks.product());
	  SimTrackContainer::const_iterator trksiter;
	  for (trksiter = trks.begin(); trksiter != trks.end(); trksiter++) 
	    if ( (int)trksiter->trackId() == all_trkid ) 
	      {
		all_simtrketa = trksiter->momentum().eta();
		all_simtrkphi = trksiter->momentum().phi();
	      }
	  
	  all_vtxz = theGeomDet->surface().position().z();
	  all_vtxr = theGeomDet->surface().position().perp();
	  
	  //computeAnglesFromDetPosition(clust, 
	  //		       theGeomDet, 
	  //		       all_clust_alpha, all_clust_beta )

	  const std::vector<SiPixelCluster::Pixel>& pixvector = clust->pixels();
	  for ( int i=0;  i<(int)pixvector.size(); ++i)
	    {
	      SiPixelCluster::Pixel holdpix = pixvector[i];
	      all_pixrow[i] = holdpix.x;
	      all_pixcol[i] = holdpix.y;
	      all_pixadc[i] = holdpix.adc;
	      LocalPoint lp = topol->localPosition(MeasurementPoint(holdpix.x, holdpix.y));
	      all_pixx[i] = lp.x();
	      all_pixy[i]= lp.y();
	      GlobalPoint GP =  theGeomDet->surface().toGlobal(Local3DPoint(lp.x(),lp.y(),lp.z()));
	      all_pixgx[i] = GP.x();	
	      all_pixgy[i]= GP.y();
	      all_pixgz[i]= GP.z();
	    }

	  ttree_all_hits_->Fill();
	  
	} // for ( ; pixeliter != pixelrechitRangeIteratorEnd; ++pixeliter)

    } // for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) 
 
  // ------------------------------------------------ all hits ---------------------------------------------------------------


  //cout << "...3..." << endl;


  // ------------------------------------------------ track hits only -------------------------------------------------------- 

  
  
  if ( include_trk_hits_ )
    {
      // Get tracks
      edm::Handle<reco::TrackCollection> trackCollection;
      e.getByLabel(src_, trackCollection);
      const reco::TrackCollection *tracks = trackCollection.product();
      reco::TrackCollection::const_iterator tciter;
      
      if ( tracks->size() > 0 )
	{
	  // Loop on tracks
	  for ( tciter=tracks->begin(); tciter!=tracks->end(); ++tciter)
	    {
	      // First loop on hits: find matched hits
	      for ( trackingRecHit_iterator it = tciter->recHitsBegin(); it != tciter->recHitsEnd(); ++it) 
		{
		  const TrackingRecHit &trk_rec_hit = **it;
		  // Is it a matched hit?
		  const SiPixelRecHit* matchedhit = dynamic_cast<const SiPixelRecHit*>(&trk_rec_hit);
		  
		  if ( matchedhit ) 
		    {
		      rechitx = -9999.9;
		      rechity = -9999.9;
		      rechitz = -9999.9;
		      rechiterrx = -9999.9;
		      rechiterry = -9999.9;		      
		      rechitresx = -9999.9;
		      rechitresy = -9999.9;
		      rechitpullx = -9999.9;
		      rechitpully = -9999.9;
		      
		      npix = -9999;
		      nxpix = -9999;
		      nypix = -9999;
		      charge = -9999.9;
		      
		      edgex = -9999;
		      edgey = -9999;
		      
		      bigx = -9999;
		      bigy = -9999;
		      
		      alpha = -9999.9;
		      beta  = -9999.9;
		      
		      phi = -9999.9;
		      eta = -9999.9;
		      
		      subdetId = -9999;
		      
		      layer  = -9999; 
		      ladder = -9999; 
		      mod    = -9999; 
		      side   = -9999;  
		      disk   = -9999;  
		      blade  = -9999; 
		      panel  = -9999; 
		      plaq   = -9999; 
		      
		      half = -9999;
		      flipped = -9999;
		      
		      nsimhit = -9999;
		      pidhit  = -9999;
		      simproc = -9999;
		      
		      simhitx = -9999.9;
		      simhity = -9999.9;

		      hit_probx = -9999.9;
		      hit_proby = -9999.9;
		      hit_cprob0 = -9999.9;
		      hit_cprob1 = -9999.9;
		      hit_cprob2 = -9999.9;
  
		      pixel_split = -9999;

		      pixel_clst_err_x = -9999.9;
		      pixel_clst_err_y = -9999.9;
		      
		      position = (*it)->localPosition();
		      error = (*it)->localPositionError();
		      
		      rechitx = position.x();
		      rechity = position.y();
		      rechitz = position.z();
		      rechiterrx = sqrt(error.xx());
		      rechiterry = sqrt(error.yy());
		      
		      npix = matchedhit->cluster()->size();
		      nxpix = matchedhit->cluster()->sizeX();
		      nypix = matchedhit->cluster()->sizeY();
		      charge = matchedhit->cluster()->charge();

		      if ( matchedhit->cluster()->getSplitClusterErrorX() > 0.0 && 
			   matchedhit->cluster()->getSplitClusterErrorY() > 0.0 )
			pixel_split = 1;
		      else
			pixel_split = 0;
		      
		      pixel_clst_err_x = matchedhit->cluster()->getSplitClusterErrorX();
		      pixel_clst_err_y = matchedhit->cluster()->getSplitClusterErrorY();

		      //hit_probx  = (float)matchedhit->probabilityX();
		      //hit_proby  = (float)matchedhit->probabilityY();
		      hit_cprob0 = (float)matchedhit->clusterProbability(0);
		      hit_cprob1 = (float)matchedhit->clusterProbability(1);
		      hit_cprob2 = (float)matchedhit->clusterProbability(2);
		      
		      
		      //Association of the rechit to the simhit
		      matched.clear();
		      matched = associate.associateHit(*matchedhit);
		      
		      nsimhit = (int)matched.size();
		      
		      if ( !matched.empty() ) 
			{
			  mindist = 999999.9;
			  float distx, disty, dist;
			  bool found_hit_from_generated_particle = false;
			  
			  int n_assoc_muon = 0;
			  
			  vector<PSimHit>::const_iterator closestit = matched.begin();
			  for (vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); ++m)
			    {
			      if ( checkType_ )
				{ // only consider associated simhits with the generated pid (muons)
				  int pid = (*m).particleType();
				  if ( abs(pid) != genType_ )
				    continue;
				}
			      
			      float simhitx = 0.5 * ( (*m).entryPoint().x() + (*m).exitPoint().x() );
			      float simhity = 0.5 * ( (*m).entryPoint().y() + (*m).exitPoint().y() );
			      
			      distx = fabs(rechitx - simhitx);
			      disty = fabs(rechity - simhity);
			      dist = sqrt( distx*distx + disty*disty );
			      
			      if ( dist < mindist )
				{
				  n_assoc_muon++;
				  
				  mindist = dist;
				  closestit = m;
				  found_hit_from_generated_particle = true;
				}
			    } // for (vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++)
			  
			  // This recHit does not have any simHit with the same particleType as the particles generated
			  // Ignore it as most probably come from delta rays.
			  if ( checkType_ && !found_hit_from_generated_particle )
			    continue; 
			  
			  //if ( n_assoc_muon > 1 )
			  //{
			  //  // cout << " ----- This is not good: n_assoc_muon = " << n_assoc_muon << endl;
			  //  // cout << "evt = " << evt << endl;
			  //}
			  
			  DetId detId = (*it)->geographicalId();

			  const PixelGeomDetUnit* theGeomDet =
			    dynamic_cast<const PixelGeomDetUnit*> ((*tracker).idToDet(detId) );
			  
			  const PixelTopology* theTopol = &(theGeomDet->specificTopology());

			  pidhit = (*closestit).particleType();
			  simproc = (int)(*closestit).processType();
			  
			  simhitx = 0.5*( (*closestit).entryPoint().x() + (*closestit).exitPoint().x() );
			  simhity = 0.5*( (*closestit).entryPoint().y() + (*closestit).exitPoint().y() );
			  
			  rechitresx = rechitx - simhitx;
			  rechitresy = rechity - simhity;
			  rechitpullx = ( rechitx - simhitx ) / sqrt(error.xx());
			  rechitpully = ( rechity - simhity ) / sqrt(error.yy());
			  
			  float simhitpx = (*closestit).momentumAtEntry().x();
			  float simhitpy = (*closestit).momentumAtEntry().y();
			  float simhitpz = (*closestit).momentumAtEntry().z();
			  
			  beta = atan2(simhitpz, simhitpy) * radtodeg;
			  alpha = atan2(simhitpz, simhitpx) * radtodeg;
			  
			  //beta  = fabs(atan2(simhitpz, simhitpy)) * radtodeg;
			  //alpha = fabs(atan2(simhitpz, simhitpx)) * radtodeg;

			  // calculate alpha and beta exactly as in PixelCPEBase.cc
			  float locx = simhitpx;
			  float locy = simhitpy;
			  float locz = simhitpz;
			  
			  bool isFlipped = false;
			  float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
			  float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
			  if ( tmp2<tmp1 ) 
			    isFlipped = true;
			  else 
			    isFlipped = false;    

			  trk_alpha = acos(locx/sqrt(locx*locx+locz*locz)) * radtodeg;
			  if ( isFlipped )                    // &&& check for FPIX !!!
			    trk_alpha = 180.0 - trk_alpha ;
			  
			  trk_beta = acos(locy/sqrt(locy*locy+locz*locz)) * radtodeg;
			  

			  phi = tciter->momentum().phi() / math_pi*180.0;
			  eta = tciter->momentum().eta();
			  
			  const int maxPixelCol = (*matchedhit).cluster()->maxPixelCol();
			  const int maxPixelRow = (*matchedhit).cluster()->maxPixelRow();
			  const int minPixelCol = (*matchedhit).cluster()->minPixelCol();
			  const int minPixelRow = (*matchedhit).cluster()->minPixelRow();
					  
			  // check whether the cluster is at the module edge 
			  if ( theTopol->isItEdgePixelInX( minPixelRow ) || 
			       theTopol->isItEdgePixelInX( maxPixelRow ) )
			    edgex = 1;
			  else 
			    edgex = 0;
			  
			  if ( theTopol->isItEdgePixelInY( minPixelCol ) || 
			       theTopol->isItEdgePixelInY( maxPixelCol ) )
			    edgey = 1;
			  else 
			    edgey = 0;
			  
			  // check whether this rechit contains big pixels
			  if ( theTopol->containsBigPixelInX(minPixelRow, maxPixelRow) )
			    bigx = 1;
			  else 
			    bigx = 0;
			  
			  if ( theTopol->containsBigPixelInY(minPixelCol, maxPixelCol) )
			    bigy = 1;
			  else 
			    bigy = 0;
			  
			  subdetId = (int)detId.subdetId();
			  
			  if ( (int)detId.subdetId() == (int)PixelSubdetector::PixelBarrel ) 
			    { 
	  
			      int tmp_nrows = theGeomDet->specificTopology().nrows();
			      if ( tmp_nrows == 80 ) 
				half = 1;
			      else if ( tmp_nrows == 160 ) 
				half = 0;
			      else 
				cout << "-------------------------------------------------- Wrong module size !!!" << endl;
			      
			      float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
			      float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
			      
			      if ( tmp2<tmp1 ) 
				flipped = 1;
			      else 
				flipped = 0;
			      
			      PXBDetId  bdetid(detId);
			      layer  = bdetid.layer();   // Layer: 1,2,3.
			      ladder = bdetid.ladder();  // Ladder: 1-20, 32, 44. 
			      mod   = bdetid.module();  // Mod: 1-8.
			    }			  
			  else if ( (int)detId.subdetId() == (int)PixelSubdetector::PixelEndcap )
			    {
			      PXFDetId fdetid(detId);
			      side  = fdetid.side();
			      disk  = fdetid.disk();
			      blade = fdetid.blade();
			      panel = fdetid.panel();
			      plaq  = fdetid.module(); // also known as plaquette
			      
			      float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
			      float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
			      
			      if ( tmp2<tmp1 ) 
				flipped = 1;
			      else 
				flipped = 0;
			      
			    } // else if ( detId.subdetId()==PixelSubdetector::PixelEndcap )
			  //else std::// cout << "We are not in the pixel detector. detId.subdetId() = " << (int)detId.subdetId() << endl;
			  
			  ttree_track_hits_->Fill();
			  
			} // if ( !matched.empty() )
		      else
			cout << "---------------- RecHit with no associated SimHit !!! -------------------------- " << endl;
		      
		    } // if ( matchedhit )
		  
		} // end of loop on hits
	      
	    } //end of loop on track 
      
	} // tracks > 0.
     
    } // if ( include_trk_hits_ )

 

  // ----------------------------------------------- track hits only -----------------------------------------------------------
  
}

void SiPixelErrorEstimation::
computeAnglesFromDetPosition(const SiPixelCluster & cl, 
			     const GeomDetUnit    & det, 
			     float& alpha, float& beta )
{
  //--- This is a new det unit, so cache it
  const PixelGeomDetUnit* theDet = dynamic_cast<const PixelGeomDetUnit*>( &det );
  if (! theDet) 
    {
      cout << "---------------------------------------------- Not a pixel detector !!!!!!!!!!!!!!" << endl;
      assert(0);
    }

  const PixelTopology* theTopol = &(theDet->specificTopology());

  // get cluster center of gravity (of charge)
  float xcenter = cl.x();
  float ycenter = cl.y();
  
  // get the cluster position in local coordinates (cm) 
  LocalPoint lp = theTopol->localPosition( MeasurementPoint(xcenter, ycenter) );
  //float lp_mod = sqrt( lp.x()*lp.x() + lp.y()*lp.y() + lp.z()*lp.z() );

  // get the cluster position in global coordinates (cm)
  GlobalPoint gp = theDet->surface().toGlobal( lp );
  float gp_mod = sqrt( gp.x()*gp.x() + gp.y()*gp.y() + gp.z()*gp.z() );

  // normalize
  float gpx = gp.x()/gp_mod;
  float gpy = gp.y()/gp_mod;
  float gpz = gp.z()/gp_mod;

  // make a global vector out of the global point; this vector will point from the 
  // origin of the detector to the cluster
  GlobalVector gv(gpx, gpy, gpz);

  // make local unit vector along local X axis
  const Local3DVector lvx(1.0, 0.0, 0.0);

  // get the unit X vector in global coordinates/
  GlobalVector gvx = theDet->surface().toGlobal( lvx );

  // make local unit vector along local Y axis
  const Local3DVector lvy(0.0, 1.0, 0.0);

  // get the unit Y vector in global coordinates
  GlobalVector gvy = theDet->surface().toGlobal( lvy );
   
  // make local unit vector along local Z axis
  const Local3DVector lvz(0.0, 0.0, 1.0);

  // get the unit Z vector in global coordinates
  GlobalVector gvz = theDet->surface().toGlobal( lvz );
    
  // calculate the components of gv (the unit vector pointing to the cluster) 
  // in the local coordinate system given by the basis {gvx, gvy, gvz}
  // note that both gv and the basis {gvx, gvy, gvz} are given in global coordinates
  float gv_dot_gvx = gv.x()*gvx.x() + gv.y()*gvx.y() + gv.z()*gvx.z();
  float gv_dot_gvy = gv.x()*gvy.x() + gv.y()*gvy.y() + gv.z()*gvy.z();
  float gv_dot_gvz = gv.x()*gvz.x() + gv.y()*gvz.y() + gv.z()*gvz.z();

  // calculate angles
  alpha = atan2( gv_dot_gvz, gv_dot_gvx );
  beta  = atan2( gv_dot_gvz, gv_dot_gvy );

  // calculate cotalpha and cotbeta
  //   cotalpha_ = 1.0/tan(alpha_);
  //   cotbeta_  = 1.0/tan(beta_ );
  // or like this
  //cotalpha_ = gv_dot_gvx / gv_dot_gvz;
  //cotbeta_  = gv_dot_gvy / gv_dot_gvz;
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelErrorEstimation);
