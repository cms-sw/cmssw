#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGeneric.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

// this is needed to get errors from templates
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"

// Services
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "boost/multi_array.hpp"

#include <iostream>
using namespace std;

const double HALF_PI = 1.57079632679489656;

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEGeneric::PixelCPEGeneric(edm::ParameterSet const & conf, 
	const MagneticField * mag, const SiPixelLorentzAngle * lorentzAngle, const SiPixelCPEGenericErrorParm * genErrorParm, const SiPixelTemplateDBObject * templateDBobject) 
  : PixelCPEBase(conf, mag, lorentzAngle, genErrorParm, templateDBobject)
{
  
  if (theVerboseLevel > 0) 
    LogDebug("PixelCPEGeneric") 
      << " constructing a generic algorithm for ideal pixel detector.\n"
      << " CPEGeneric:: VerboseLevel = " << theVerboseLevel;

  // Externally settable cuts  
  the_eff_charge_cut_lowX = conf.getParameter<double>("eff_charge_cut_lowX");
  the_eff_charge_cut_lowY = conf.getParameter<double>("eff_charge_cut_lowY");
  the_eff_charge_cut_highX = conf.getParameter<double>("eff_charge_cut_highX");
  the_eff_charge_cut_highY = conf.getParameter<double>("eff_charge_cut_highY");
  the_size_cutX = conf.getParameter<double>("size_cutX");
  the_size_cutY = conf.getParameter<double>("size_cutY");

  EdgeClusterErrorX_ = conf.getParameter<double>("EdgeClusterErrorX");
  EdgeClusterErrorY_ = conf.getParameter<double>("EdgeClusterErrorY");

  // Externally settable flags to inflate errors
  inflate_errors = conf.getParameter<bool>("inflate_errors");
  inflate_all_errors_no_trk_angle = conf.getParameter<bool>("inflate_all_errors_no_trk_angle");

  UseErrorsFromTemplates_    = conf.getParameter<bool>("UseErrorsFromTemplates");
  TruncatePixelCharge_       = conf.getParameter<bool>("TruncatePixelCharge");
  IrradiationBiasCorrection_ = conf.getParameter<bool>("IrradiationBiasCorrection");
  DoCosmics_                 = conf.getParameter<bool>("DoCosmics");
  LoadTemplatesFromDB_       = conf.getParameter<bool>("LoadTemplatesFromDB");

  Upgrade_                   = conf.getParameter<bool>("Upgrade");
  SmallPitch_                = conf.getParameter<bool>("SmallPitch");

  if ( !UseErrorsFromTemplates_ && ( TruncatePixelCharge_       || 
				     IrradiationBiasCorrection_ || 
				     DoCosmics_                 ||
				     LoadTemplatesFromDB_ ) )
    {
      throw cms::Exception("PixelCPEGeneric::PixelCPEGeneric: ") 
	  << "\nERROR: UseErrorsFromTemplates_ is set to False in PixelCPEGeneric_cfi.py. "
	  << " In this case it does not make sense to set any of the following to True: " 
	  << " TruncatePixelCharge_, IrradiationBiasCorrection_, DoCosmics_, LoadTemplatesFromDB_ !!!" 
	  << "\n\n";
    }

  if ( UseErrorsFromTemplates_ )
	{
		templID_ = -999;
		if ( LoadTemplatesFromDB_ )
		{
			// Initialize template store to the selected ID [Morris, 6/25/08]  
			if ( !templ_.pushfile( *templateDBobject_) )
				throw cms::Exception("InvalidCalibrationLoaded") 
					<< "ERROR: Templates not filled correctly. Check the sqlite file. Using SiPixelTemplateDBObject version " 
					<< ( *templateDBobject_ ).version() << ". Template ID is " << templID_;
		}
		else 
		{
			if ( !templ_.pushfile( templID_ ) )
				throw cms::Exception("InvalidCalibrationLoaded") 
					<< "ERROR: Templates not loaded correctly from text file. Reconstruction will fail." << " Template ID is " << templID_;
		}
		
	} // if ( UseErrorsFromTemplates_ )
  
  //cout << endl;
  //cout << "From PixelCPEGeneric::PixelCPEGeneric(...)" << endl;
  //cout << "(int)UseErrorsFromTemplates_ = " << (int)UseErrorsFromTemplates_    << endl;
  //cout << "TruncatePixelCharge_         = " << (int)TruncatePixelCharge_       << endl;      
  //cout << "IrradiationBiasCorrection_   = " << (int)IrradiationBiasCorrection_ << endl;
  //cout << "(int)DoCosmics_              = " << (int)DoCosmics_                 << endl;
  //cout << "(int)LoadTemplatesFromDB_    = " << (int)LoadTemplatesFromDB_       << endl;
  //cout << endl;

}


MeasurementPoint 
PixelCPEGeneric::measurementPosition(const SiPixelCluster& cluster, 
				     const GeomDetUnit & det) const
{
  LocalPoint lp = localPosition(cluster,det);

  // ggiurgiu@jhu.edu 12/09/2010 : trk angles needed for bow/kink correction
  if ( with_track_angle ) 
    return theTopol->measurementPosition(lp, Topology::LocalTrackAngles( loc_traj_param_.dxdz(), loc_traj_param_.dydz() ) );
  else
    return theTopol->measurementPosition(lp);

}



//-----------------------------------------------------------------------------
//! Hit position in the local frame (in cm).  Unlike other CPE's, this
//! one converts everything from the measurement frame (in channel numbers) 
//! into the local frame (in centimeters).  
//-----------------------------------------------------------------------------
LocalPoint
PixelCPEGeneric::localPosition(const SiPixelCluster& cluster, 
			       const GeomDetUnit & det) const 
{
  setTheDet( det, cluster );  //!< Initialize this det unit
  computeLorentzShifts();  //!< correctly compute lorentz shifts in X and Y
  if ( UseErrorsFromTemplates_ )
    {
      templID_ = templateDBobject_->getTemplateID(theDet->geographicalId().rawId());
      bool fpix;  //!< barrel(false) or forward(true)
      if ( thePart == GeomDetEnumerators::PixelBarrel )
	fpix = false;    // no, it's not forward -- it's barrel
      else
	fpix = true;     // yes, it's forward
      
      float qclus = cluster.charge();
      
      GlobalVector bfield = magfield_->inTesla( theDet->surface().position() ); 
      
      Frame detFrame( theDet->surface().position(), theDet->surface().rotation() );
      LocalVector Bfield = detFrame.toLocal( bfield );
      float locBz = Bfield.z();
      //cout << "PixelCPEGeneric::localPosition(...) : locBz = " << locBz << endl;
      
      pixmx  = -999.9; // max pixel charge for truncation of 2-D cluster
      sigmay = -999.9; // CPE Generic y-error for multi-pixel cluster
      deltay = -999.9; // CPE Generic y-bias for multi-pixel cluster
      sigmax = -999.9; // CPE Generic x-error for multi-pixel cluster
      deltax = -999.9; // CPE Generic x-bias for multi-pixel cluster
      sy1    = -999.9; // CPE Generic y-error for single single-pixel
      dy1    = -999.9; // CPE Generic y-bias for single single-pixel cluster
      sy2    = -999.9; // CPE Generic y-error for single double-pixel cluster
      dy2    = -999.9; // CPE Generic y-bias for single double-pixel cluster
      sx1    = -999.9; // CPE Generic x-error for single single-pixel cluster
      dx1    = -999.9; // CPE Generic x-bias for single single-pixel cluster
      sx2    = -999.9; // CPE Generic x-error for single double-pixel cluster
      dx2    = -999.9; // CPE Generic x-bias for single double-pixel cluster
      
      qBin_ = templ_.qbin( templID_, cotalpha_, cotbeta_, locBz, qclus,  // inputs
			   pixmx,                                       // returned by reference
			   sigmay, deltay, sigmax, deltax,              // returned by reference
			   sy1, dy1, sy2, dy2, sx1, dx1, sx2, dx2 );    // returned by reference
      
      // These numbers come in microns from the qbin(...) call. Transform them to cm.
      const float micronsToCm = 1.0e-4;
      
      deltax = deltax * micronsToCm;
      dx1 = dx1 * micronsToCm;
      dx2 = dx2 * micronsToCm;
      
      deltay = deltay * micronsToCm;
      dy1 = dy1 * micronsToCm;
      dy2 = dy2 * micronsToCm;
      
      sigmax = sigmax * micronsToCm;
      sx1 = sx1 * micronsToCm;
      sx2 = sx2 * micronsToCm;
      
      sigmay = sigmay * micronsToCm;
      sy1 = sy1 * micronsToCm;
      sy2 = sy2 * micronsToCm;
      
    } // if ( UseErrorsFromTemplates_ )
  
  
  float Q_f_X = 0.0;        //!< Q of the first  pixel  in X 
  float Q_l_X = 0.0;        //!< Q of the last   pixel  in X
  float Q_m_X = 0.0;        //!< Q of the middle pixels in X
  float Q_f_Y = 0.0;        //!< Q of the first  pixel  in Y 
  float Q_l_Y = 0.0;        //!< Q of the last   pixel  in Y
  float Q_m_Y = 0.0;        //!< Q of the middle pixels in Y
  collect_edge_charges( cluster, 
			Q_f_X, Q_l_X, Q_m_X, 
			Q_f_Y, Q_l_Y, Q_m_Y );

  //--- Find the inner widths along X and Y in one shot.  We
  //--- compute the upper right corner of the inner pixels
  //--- (== lower left corner of upper right pixel) and
  //--- the lower left corner of the inner pixels
  //--- (== upper right corner of lower left pixel), and then
  //--- subtract these two points in the formula.

  //--- Upper Right corner of Lower Left pixel -- in measurement frame
  MeasurementPoint meas_URcorn_LLpix( cluster.minPixelRow()+1.0,
				      cluster.minPixelCol()+1.0 );

  //--- Lower Left corner of Upper Right pixel -- in measurement frame
  MeasurementPoint meas_LLcorn_URpix( cluster.maxPixelRow(),
				      cluster.maxPixelCol() );

  //--- These two now converted into the local
  
  LocalPoint local_URcorn_LLpix;
  LocalPoint local_LLcorn_URpix;


  // PixelCPEGeneric can be used with or without track angles
  // If PixelCPEGeneric is called with track angles, use them to correct for bows/kinks:
  if ( with_track_angle )
    {
      local_URcorn_LLpix = theTopol->localPosition(meas_URcorn_LLpix, loc_trk_pred_);
      local_LLcorn_URpix = theTopol->localPosition(meas_LLcorn_URpix, loc_trk_pred_);
    }
  else
    {
      local_URcorn_LLpix = theTopol->localPosition(meas_URcorn_LLpix);
      local_LLcorn_URpix = theTopol->localPosition(meas_LLcorn_URpix);
    }

  if (theVerboseLevel > 20) {
    cout  
      << "\n\t >>> cluster.x = " << cluster.x()
      << "\n\t >>> cluster.y = " << cluster.y()
      << "\n\t >>> cluster: minRow = " << cluster.minPixelRow()
      << "  minCol = " << cluster.minPixelCol()
      << "\n\t >>> cluster: maxRow = " << cluster.maxPixelRow()
      << "  maxCol = " << cluster.maxPixelCol()
      << "\n\t >>> meas: inner lower left  = " << meas_URcorn_LLpix.x() 
      << "," << meas_URcorn_LLpix.y()
      << "\n\t >>> meas: inner upper right = " << meas_LLcorn_URpix.x() 
      << "," << meas_LLcorn_URpix.y() 
      << endl;
  }

  //--- &&& Note that the cuts below should not be hardcoded (like in Orca and
  //--- &&& CPEFromDetPosition/PixelCPEInitial), but rather be
  //--- &&& externally settable (but tracked) parameters.  


  //--- Position, including the half lorentz shift
  if (theVerboseLevel > 20) 
    cout << "\t >>> Generic:: processing X" << endl;
  float xPos = 
    generic_position_formula( cluster.sizeX(),
			      Q_f_X, Q_l_X, 
			      local_URcorn_LLpix.x(), local_LLcorn_URpix.x(),
			      0.5*lorentzShiftInCmX_,   // 0.5 * lorentz shift in 
			      cotalpha_,
			      thePitchX,
			      theTopol->isItBigPixelInX( cluster.minPixelRow() ),
			      theTopol->isItBigPixelInX( cluster.maxPixelRow() ),
			      the_eff_charge_cut_lowX,
                              the_eff_charge_cut_highX,
                              the_size_cutX);           // cut for eff charge width &&&


  if (theVerboseLevel > 20) 
    cout << "\t >>> Generic:: processing Y" << endl;
  float yPos = 
    generic_position_formula( cluster.sizeY(),
			      Q_f_Y, Q_l_Y, 
			      local_URcorn_LLpix.y(), local_LLcorn_URpix.y(),
			      0.5*lorentzShiftInCmY_,   // 0.5 * lorentz shift in cm
			      cotbeta_,
			      thePitchY,   // 0.5 * lorentz shift (may be 0)
			      theTopol->isItBigPixelInY( cluster.minPixelCol() ),
			      theTopol->isItBigPixelInY( cluster.maxPixelCol() ),
			      the_eff_charge_cut_lowY,
                              the_eff_charge_cut_highY,
                              the_size_cutY);           // cut for eff charge width &&&
			     
  // Apply irradiation corrections.
  if ( IrradiationBiasCorrection_ )
    {
      if ( cluster.sizeX() == 1 )
	{
	  // sanity chack
	  if ( cluster.maxPixelRow() != cluster.minPixelRow() )
	    throw cms::Exception("PixelCPEGeneric::localPosition") 
	      << "\nERROR: cluster.maxPixelRow() != cluster.minPixelRow() although x-size = 1 !!!!! " << "\n\n";
	  
	  // Find if pixel is double (big). 
	  bool bigInX = theTopol->isItBigPixelInX( cluster.maxPixelRow() );
	  
	  if ( !bigInX ) 
	    {
	      //cout << "Apply correction dx1 = " << dx1 << " to xPos = " << xPos << endl;
	      xPos -= dx1;
	    }
	  else           
	    {
	      //cout << "Apply correction dx2 = " << dx2 << " to xPos = " << xPos << endl;
	      xPos -= dx2;
	    }
	}
      else if ( cluster.sizeX() > 1 )
	{
	  //cout << "Apply correction correction_deltax = " << deltax << " to xPos = " << xPos << endl;
	  xPos -= deltax;
	}
      else
	throw cms::Exception("PixelCPEGeneric::localPosition") 
	  << "\nERROR: Unphysical cluster x-size = " <<  (int)cluster.sizeX() << "\n\n";
      
  if ( cluster.sizeY() == 1 )
    {
      // sanity chack
      if ( cluster.maxPixelCol() != cluster.minPixelCol() )
	throw cms::Exception("PixelCPEGeneric::localPosition") 
	  << "\nERROR: cluster.maxPixelCol() != cluster.minPixelCol() although y-size = 1 !!!!! " << "\n\n";
      
      // Find if pixel is double (big). 
      bool bigInY = theTopol->isItBigPixelInY( cluster.maxPixelCol() );
      
      if ( !bigInY ) 
	{
	  //cout << "Apply correction dy1 = " << dy1 << " to yPos = " << yPos  << endl;
	  yPos -= dy1;
	}
      else           
	{
	  //cout << "Apply correction dy2 = " << dy2  << " to yPos = " << yPos << endl;
	  yPos -= dy2;
	}
    }
  else if ( cluster.sizeY() > 1 )
    {
      //cout << "Apply correction deltay = " << deltay << " to yPos = " << yPos << endl;
      yPos -= deltay;
    }
  else
    throw cms::Exception("PixelCPEGeneric::localPosition") 
      << "\nERROR: Unphysical cluster y-size = " << (int)cluster.sizeY() << "\n\n";

	} // if ( IrradiationBiasCorrection_ )
	
  //--- Now put the two together
  LocalPoint pos_in_local( xPos, yPos );
  return pos_in_local;
}



//-----------------------------------------------------------------------------
//!  A generic version of the position formula.  Since it works for both
//!  X and Y, in the interest of the simplicity of the code, all parameters
//!  are passed by the caller.  The only class variable used by this method
//!  is the theThickness, since that's common for both X and Y.
//-----------------------------------------------------------------------------
double
PixelCPEGeneric::    
generic_position_formula( int size,                //!< Size of this projection.
			  double Q_f,              //!< Charge in the first pixel.
			  double Q_l,              //!< Charge in the last pixel.
			  double upper_edge_first_pix, //!< As the name says.
			  double lower_edge_last_pix,  //!< As the name says.
			  double half_lorentz_shift,   //!< L-shift at half thickness
			  double cot_angle,        //!< cot of alpha_ or beta_
			  double pitch,            //!< thePitchX or thePitchY
			  bool first_is_big,       //!< true if the first is big
			  bool last_is_big,        //!< true if the last is big
			  double eff_charge_cut_low, //!< Use edge if > W_eff  &&&
			  double eff_charge_cut_high,//!< Use edge if < W_eff  &&&
			  double size_cut         //!< Use edge when size == cuts
			 ) const
{
  double geom_center = 0.5 * ( upper_edge_first_pix + lower_edge_last_pix );

  //--- The case of only one pixel in this projection is separate.  Note that
  //--- here first_pix == last_pix, so the average of the two is still the
  //--- center of the pixel.
  if ( size == 1 ) 
    {
      // ggiurgiu@jhu.edu, 02/03/09 : for size = 1, the Lorentz shift is already accounted by the irradiation correction
      if ( IrradiationBiasCorrection_ ) 
	return geom_center;
      else
	return geom_center + half_lorentz_shift;
    }


  //--- Width of the clusters minus the edge (first and last) pixels.
  //--- In the note, they are denoted x_F and x_L (and y_F and y_L)
  double W_inner      = lower_edge_last_pix - upper_edge_first_pix;  // in cm


  //--- Predicted charge width from geometry
  double W_pred = 
    theThickness * cot_angle                     // geometric correction (in cm)
    - 2 * half_lorentz_shift;                    // (in cm) &&& check fpix!  
  

  //--- Total length of the two edge pixels (first+last)
  double sum_of_edge = 0.0;
  if (first_is_big) sum_of_edge += 2.0;
  else              sum_of_edge += 1.0;
  
  if (last_is_big)  sum_of_edge += 2.0;
  else              sum_of_edge += 1.0;
  

  //--- The `effective' charge width -- particle's path in first and last pixels only
  double W_eff = fabs( W_pred ) - W_inner;


  //--- If the observed charge width is inconsistent with the expectations
  //--- based on the track, do *not* use W_pred-W_innner.  Instead, replace
  //--- it with an *average* effective charge width, which is the average
  //--- length of the edge pixels.
  //
  bool usedEdgeAlgo = false;
  if (( W_eff/pitch < eff_charge_cut_low ) ||
      ( W_eff/pitch > eff_charge_cut_high ) || (size >= size_cut)) 
    {
      W_eff = pitch * 0.5 * sum_of_edge;  // ave. length of edge pixels (first+last) (cm)
      usedEdgeAlgo = true;
      nRecHitsUsedEdge_++;
    }

  
  //--- Finally, compute the position in this projection
  double Qdiff = Q_l - Q_f;
  double Qsum  = Q_l + Q_f;

	//--- Temporary fix for clusters with both first and last pixel with charge = 0
	if(Qsum==0) Qsum=1.0;
  double hit_pos = geom_center + 0.5*(Qdiff/Qsum) * W_eff + half_lorentz_shift;

  //--- Debugging output
  if (theVerboseLevel > 20) {
    if ( thePart == GeomDetEnumerators::PixelBarrel ) {
      cout << "\t >>> We are in the Barrel." ;
    } else {
      cout << "\t >>> We are in the Forward." ;
    }
    cout 
      << "\n\t >>> cot(angle) = " << cot_angle << "  pitch = " << pitch << "  size = " << size
      << "\n\t >>> upper_edge_first_pix = " << upper_edge_first_pix
      << "\n\t >>> lower_edge_last_pix  = " << lower_edge_last_pix
      << "\n\t >>> geom_center          = " << geom_center
      << "\n\t >>> half_lorentz_shift   = " << half_lorentz_shift
      << "\n\t >>> W_inner              = " << W_inner
      << "\n\t >>> W_pred               = " << W_pred
      << "\n\t >>> W_eff(orig)          = " << fabs( W_pred ) - W_inner
      << "\n\t >>> W_eff(used)          = " << W_eff
      << "\n\t >>> sum_of_edge          = " << sum_of_edge
      << "\n\t >>> Qdiff = " << Qdiff << "  Qsum = " << Qsum 
      << "\n\t >>> hit_pos              = " << hit_pos 
      << "\n\t >>> RecHits: total = " << nRecHitsTotal_ 
      << "  used edge = " << nRecHitsUsedEdge_
      << endl;
    if (usedEdgeAlgo) 
      cout << "\n\t >>> Used Edge algorithm." ;
    else
      cout << "\n\t >>> Used angle information." ;
    cout << endl;
  }


  return hit_pos;
}





//-----------------------------------------------------------------------------
//!  Collect the edge charges in x and y, in a single pass over the pixel vector.
//!  Calculate charge in the first and last pixel projected in x and y
//!  and the inner cluster charge, projected in x and y.
//-----------------------------------------------------------------------------
void
PixelCPEGeneric::
collect_edge_charges(const SiPixelCluster& cluster,  //!< input, the cluster
		     float & Q_f_X,              //!< output, Q first  in X 
		     float & Q_l_X,              //!< output, Q last   in X
		     float & Q_m_X,              //!< output, Q middle in X
		     float & Q_f_Y,              //!< output, Q first  in Y 
		     float & Q_l_Y,              //!< output, Q last   in Y
		     float & Q_m_Y               //!< output, Q middle in Y
		     ) const
{
  // Initialize return variables.
  Q_f_X = Q_l_X = Q_m_X = 0.0;
  Q_f_Y = Q_l_Y = Q_m_Y = 0.0;

  // Fetch the pixels vector from the cluster.
  const vector<SiPixelCluster::Pixel>& pixelsVec = cluster.pixels();

  // Obtain boundaries in index units
  int xmin = cluster.minPixelRow();
  int xmax = cluster.maxPixelRow();
  int ymin = cluster.minPixelCol();
  int ymax = cluster.maxPixelCol();

//   // Obtain the cluster boundaries (note: in measurement units!)
//   float xmin = cluster.minPixelRow()+0.5;
//   float xmax = cluster.maxPixelRow()+0.5;  
//   float ymin = cluster.minPixelCol()+0.5;
//   float ymax = cluster.maxPixelCol()+0.5;
  
  // Iterate over the pixels.
  int isize = pixelsVec.size();
  
  for (int i = 0;  i < isize; ++i) 
    {
      float pix_adc = -999.9;

      // ggiurgiu@fnal.gov: add pixel charge truncation
      if ( UseErrorsFromTemplates_ && TruncatePixelCharge_ ) 
	pix_adc = min( (float)(pixelsVec[i].adc), pixmx );
      else 
	pix_adc = pixelsVec[i].adc;

      //
      // X projection
      if      ( pixelsVec[i].x == xmin )       // need to match with tolerance!!! &&&
	Q_f_X += pix_adc;
      else if ( pixelsVec[i].x == xmax ) 
	Q_l_X += pix_adc;
      else 
	Q_m_X += pix_adc;
      //
      // Y projection
      if      ( pixelsVec[i].y == ymin ) 
	Q_f_Y += pix_adc;
      else if ( pixelsVec[i].y == ymax ) 
	Q_l_Y += pix_adc;
      else 
	Q_m_Y += pix_adc;
    }
  
  return;
} 


//==============  INFLATED ERROR AND ERRORS FROM DB BELOW  ================

//-------------------------------------------------------------------------
//  Hit error in the local frame
//-------------------------------------------------------------------------
LocalError 
PixelCPEGeneric::localError( const SiPixelCluster& cluster, 
			     const GeomDetUnit & det) const 
{
  setTheDet( det, cluster );

  // The squared errors
  float xerr_sq = -99999.9;
  float yerr_sq = -99999.9;

  int sizex = cluster.sizeX();
  int sizey = cluster.sizeY();
   
  // Default errors are the maximum error used for edge clusters.
  /*
    int row_offset = cluster.minPixelRow();
    int col_offset = cluster.minPixelCol();
    int n_bigInX = 0;
    for (int irow = 0; irow < sizex; ++irow)
    if ( theTopol->isItBigPixelInX( irow+row_offset ) )
    ++n_bigInX;
    int n_bigInY = 0;
    for (int icol = 0; icol < sizey; ++icol) 
    if ( theTopol->isItBigPixelInY( icol+col_offset ) )
    ++n_bigInX;
    float xerr = (float)(sizex + n_bigInX) * thePitchX / sqrt(12.0);      
    float yerr = (float)(sizey + n_bigInY) * thePitchY / sqrt(12.0); 
  */

  // These are determined by looking at residuals for edge clusters
  const float micronsToCm = 1.0e-4;
  float xerr = EdgeClusterErrorX_ * micronsToCm;
  float yerr = EdgeClusterErrorY_ * micronsToCm;
  
  // Find if cluster is at the module edge. 
  int maxPixelCol = cluster.maxPixelCol();
  int maxPixelRow = cluster.maxPixelRow();
  int minPixelCol = cluster.minPixelCol();
  int minPixelRow = cluster.minPixelRow();       
  bool edgex = ( theTopol->isItEdgePixelInX( minPixelRow ) ) || ( theTopol->isItEdgePixelInX( maxPixelRow ) );
  bool edgey = ( theTopol->isItEdgePixelInY( minPixelCol ) ) || ( theTopol->isItEdgePixelInY( maxPixelCol ) );

  // Find if cluster contains double (big) pixels. 
  bool bigInX = theTopol->containsBigPixelInX( minPixelRow, maxPixelRow ); 	 
  bool bigInY = theTopol->containsBigPixelInY( minPixelCol, maxPixelCol );

  if( Upgrade_ )
   {
     //cout << " I'm in Upgrade!! " << endl;

     DetId id = (det.geographicalId());
     int layer = PXBDetId::PXBDetId(id).layer();
     // cout << "layer " << layer << endl;
     // ------------------------------------------------
       if ( thePart == GeomDetEnumerators::PixelBarrel ) {
         if( SmallPitch_ && layer == 1 )
           {   // PXB Small Pitch
             if  ( !edgex )
                 {
                   if      ( sizex == 1 ) xerr = 0.00104;      // Old 0.0015
                   else if ( sizex == 2 ) xerr = 0.000691;     // Old 0.00089
                   else if ( sizex == 3 ) xerr = 0.00122;      // Old 0.0014
                   else                   xerr = 0.00321;      // Old 0.0046
                 } // End of PXB Small Pitch X
             if ( !edgey )
                 {
                   if      ( sizey == 1 ) yerr = 0.00199;      // Old 0.0022
                   else if ( sizey == 2 ) yerr = 0.00136;      // Old 0.0015
                   else if ( sizey == 3 ) yerr = 0.0015;       // Old 0.0017
                   else if ( sizey == 4 ) yerr = 0.00153;      // Old 0.0018
                   else if ( sizey == 5 ) yerr = 0.00152;      // Old 0.0019
                   else if ( sizey == 6 ) yerr = 0.00171;      // Old 0.0020
                   else if ( sizey == 7 ) yerr = 0.00154;      // Old 0.0021
                   else if ( sizey == 8 ) yerr = 0.00157;      // Old 0.0021
                   else if ( sizey == 9 ) yerr = 0.00154;      // Old 0.0021
                   else                   yerr = 0.00164;      // Old 0.0029
                 } // End of PXB Small Pitch Y
           } // End of PXB Small Pitch
         else
           {   // PXB Reg Pitch
             if  ( !edgex )
                 {
                   if      ( sizex == 1 ) xerr = 0.00114;      // Old 0.0016
                   else if ( sizex == 2 ) xerr = 0.00104;      // Old 0.0012
                   else if ( sizex == 3 ) xerr = 0.00214;      // Old 0.0024
                   else                   xerr = 0.00425;      // Old 0.0048
                 } // End of PXB Reg Pitch X
             if ( !edgey )
                 {
                   if      ( sizey == 1 ) yerr = 0.00299;      // Old 0.0034
                   else if ( sizey == 2 ) yerr = 0.00203;      // Old 0.0021
                   else if ( sizey == 3 ) yerr = 0.0023;       // Old 0.0025
                   else if ( sizey == 4 ) yerr = 0.00237;      // Old 0.0027
                   else if ( sizey == 5 ) yerr = 0.00233;      // Old 0.0029
                   else if ( sizey == 6 ) yerr = 0.00243;      // Old 0.0031
                   else if ( sizey == 7 ) yerr = 0.00232;      // Old 0.0033
                   else if ( sizey == 8 ) yerr = 0.00259;      // Old 0.0033
                   else if ( sizey == 9 ) yerr = 0.00176;      // Old 0.0041
                   else                   yerr = 0.00245;      // Old 0.0055
                 } // End of PXB Reg Pitch Y
           } // End of PXB Reg Pitch
       } // End of PXB
       else
         { // FPix
           if ( !edgex )
              {
                   if      ( sizex == 1 ) xerr = 0.00151;      // Old 0.0014
                   else if ( sizex == 2 ) xerr = 0.000813;     // Old 0.00094
                   else if ( sizex == 3 ) xerr = 0.00221;      // Old 0.0023
                   else                   xerr = 0.00218;      // Old 0.0032 for >=4
              } // End of FPix X
           if ( !edgey )
              {
                   if      ( sizey == 1 ) yerr = 0.00261;      // Old 0.0029
                   else if ( sizey == 2 ) yerr = 0.00107;      // Old 0.0013
                   else if ( sizey == 3 ) yerr = 0.00264;      // Old 0.0032
                   else                   yerr = 0.00357;      // Old 0.0040 for >=4
              } // End of FPix Y
          } // End FPix
     // ------------------------------------------------
     //cout << "Small " << SmallPitch_ << " " << thePart << " layer " << layer 
     //     << " errors finally " << sizex << " " << xerr << " " << sizey << " " << yerr << endl;
   } // end if Upgrade_
  else if ( !with_track_angle && DoCosmics_ )
    {
      //cout << "Track angles are not known and we are processing cosmics." << endl; 
      //cout << "Default angle estimation which assumes track from PV (0,0,0) does not work." << endl;
      //cout << "Use an error parameterization which only depends on cluster size (by Vincenzo Chiochia)." << endl; 
      
      if ( thePart == GeomDetEnumerators::PixelBarrel ) 
	{
	  if ( !edgex )
	    {
	      if      ( sizex == 1 ) xerr = 0.00115; // Size = 1 -> Sigma = 11.5 um 
	      else if ( sizex == 2 ) xerr = 0.00120; // Size = 2 -> Sigma = 12.0 um      
	      else if ( sizex == 3 ) xerr = 0.00088; // Size = 3 -> Sigma =  8.8 um
	      else                   xerr = 0.01030;
	    }
	  
	  if ( !edgey )
	    {
	      if      ( sizey ==  1 ) yerr = 0.00375; // 37.5 um 
	      else if ( sizey ==  2 ) yerr = 0.00230; // 23 um      
	      else if ( sizey ==  3 ) yerr = 0.00250; // 25 um
	      else if ( sizey ==  4 ) yerr = 0.00250; // 25 um
	      else if ( sizey ==  5 ) yerr = 0.00230; // 23 um
	      else if ( sizey ==  6 ) yerr = 0.00230; // 23 um
	      else if ( sizey ==  7 ) yerr = 0.00210; // 21 um
	      else if ( sizey ==  8 ) yerr = 0.00210; // 21 um
	      else if ( sizey ==  9 ) yerr = 0.00240; // 24 um
	      else                    yerr = 0.00210; // 21um
	    }
	} 
      else // EndCap
	{
	  if ( !edgex )
	    {
	      if      ( sizex == 1 ) xerr = 0.0020;
	      else if ( sizex == 2 ) xerr = 0.0020;
	      else                   xerr = 0.0020;
	    }
	
	  if ( !edgey )
	    {
	      if  ( sizey == 1 ) yerr = 0.00210; // 21 um
	      else               yerr = 0.00075; // 7.5 um
	    }
	}

    } // if ( !with_track_angle )
  else
    {
      //cout << "Track angles are known. We can use either errors from templates or the error parameterization from DB." << endl;
      
      if ( UseErrorsFromTemplates_ )
	{
	  if (qBin_ == 0 && inflate_errors )
	    {	       
	      int n_bigx = 0;
	      int n_bigy = 0;
	      
	      int row_offset = cluster.minPixelRow();
	      int col_offset = cluster.minPixelCol();
	      
	      for (int irow = 0; irow < 7; ++irow)
		{
		  if ( theTopol->isItBigPixelInX( irow+row_offset ) )
		    ++n_bigx;
		}
	      
	      for (int icol = 0; icol < 21; ++icol) 
		{
		  if ( theTopol->isItBigPixelInY( icol+col_offset ) )
		    ++n_bigy;
		}
	      
	      xerr = (float)(sizex + n_bigx) * thePitchX / sqrt( 12.0 );
	      yerr = (float)(sizey + n_bigy) * thePitchY / sqrt( 12.0 );
	      
	    } // if ( qbin == 0 && inflate_errors )
	  else
	    {
	      // Default errors

	      if ( !edgex )
		{
		  if ( sizex == 1 )
		    {
		      if ( !bigInX ) 
						xerr = sx1; 
		      else           
						xerr = sx2;
		    }
		  else if ( sizex > 1 )
		    xerr = sigmax;
		  else
		    throw cms::Exception("PixelCPEGeneric::localError") 
		      << "\nERROR: Unphysical cluster x-size = " << sizex << "\n\n";
		}
	      
	      if ( !edgey )
		{
		  if ( sizey == 1 )
		    {
		      if ( !bigInY )
			yerr = sy1;
		      else
			yerr = sy2;
		    }
		  else if ( sizey > 1 )
		    yerr = sigmay;
		  else
		    throw cms::Exception("PixelCPEGeneric::localError") 
		      << "\nERROR: Unphysical cluster y-size = " << sizex << "\n\n";
		}
	    } // if ( qbin == 0 && inflate_errors ) else

	} //if ( UseErrorsFromTemplates_ )
      else 
	{
	  //cout << endl << "Use errors from DB:" << endl;
	  
	  if ( edgex && edgey ) 
	    { 	 
	      //--- Both axes on the edge, no point in calling PixelErrorParameterization, 	 
	      //--- just return the max errors on both. 	 
	    } 
	  else 
	    { 	 
	      pair<float,float> errPair = 	 
		genErrorsFromDB_->getError( genErrorParm_, thePart, cluster.sizeX(), cluster.sizeY(), 	 
					    alpha_, beta_, bigInX, bigInY ); 
	      if ( !edgex ) 
		xerr = errPair.first; 	 
	      if ( !edgey ) 
		yerr = errPair.second; 	 
	    } 	 
	  
	  if (theVerboseLevel > 9) 
	    { 	 
	      LogDebug("PixelCPEGeneric") << 	 
		" Sizex = " << cluster.sizeX() << " Sizey = " << cluster.sizeY() << 	 
		" Edgex = " << edgex           << " Edgey = " << edgey           << 	 
		" ErrX  = " << xerr            << " ErrY  = " << yerr; 	 
	    }
	  
	} //if ( UseErrorsFromTemplates_ ) else 
      
    } // if ( !with_track_angle ) else
  
  if ( !(xerr > 0.0) )
    throw cms::Exception("PixelCPEGeneric::localError") 
      << "\nERROR: Negative pixel error xerr = " << xerr << "\n\n";
  
  if ( !(yerr > 0.0) )
    throw cms::Exception("PixelCPEGeneric::localError") 
      << "\nERROR: Negative pixel error yerr = " << yerr << "\n\n";
 
  xerr_sq = xerr*xerr; 
  yerr_sq = yerr*yerr;
 
  return LocalError( xerr_sq, 0, yerr_sq );

}






