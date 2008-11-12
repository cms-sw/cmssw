#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGeneric.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"


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

  // Externally settable flags to inflate errors
  inflate_errors = conf.getParameter<bool>("inflate_errors");
  inflate_all_errors_no_trk_angle = conf.getParameter<bool>("inflate_all_errors_no_trk_angle");
}


MeasurementPoint 
PixelCPEGeneric::measurementPosition(const SiPixelCluster& cluster, 
				     const GeomDetUnit & det) const
{
  LocalPoint lp = localPosition(cluster,det);
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
  setTheDet( det );  //!< Initialize this det unit
  computeLorentzShifts();  //!< correctly compute lorentz shifts in X and Y

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
  LocalPoint local_URcorn_LLpix = theTopol->localPosition(meas_URcorn_LLpix);
  LocalPoint local_LLcorn_URpix = theTopol->localPosition(meas_LLcorn_URpix);
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
                              the_size_cutX,           // cut for eff charge width &&&
			      cotAlphaFromCluster_ );  // returned to us


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
                              the_size_cutY,           // cut for eff charge width &&&
			      cotBetaFromCluster_ );   // returned to us


  //--- Now put the two together
  LocalPoint pos_in_local(xPos,yPos);
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
			  double size_cut,         //!< Use edge when size == cuts
			  float & cot_angle_from_length  //!< Aux output: angle from len
			  ) const
{
  double geom_center = 0.5 * ( upper_edge_first_pix + lower_edge_last_pix );

  //--- The case of only one pixel in this projection is separate.  Note that
  //--- here first_pix == last_pix, so the average of the two is still the
  //--- center of the pixel.
  if (size == 1) {
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
  double hit_pos = geom_center + 0.5*(Qdiff/Qsum) * W_eff + half_lorentz_shift;
  


  //--- At the end, also compute the *average* angle consistent
  //--- with the cluster length in this dimension.  This variable will
  //--- be copied to cotAlphaFromCluster_ and cotBetaFromCluster_.  It's
  //--- basically inverting W_pred to get cot_angle, except this one is
  //--- driven by the cluster length.
  //--- (See the comment in PixelCPEBase header file for what these are for.)
  double ave_path_length_projected =
    pitch*0.5*sum_of_edge + W_inner - 2*half_lorentz_shift;
  cot_angle_from_length = ave_path_length_projected / theThickness;

  
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
  for (int i = 0;  i < isize; ++i) {
    //
    // X projection
    if (pixelsVec[i].x == xmin)       // need to match with tolerance!!! &&&
      Q_f_X += float(pixelsVec[i].adc);
    else if (pixelsVec[i].x == xmax) 
      Q_l_X += float(pixelsVec[i].adc);
    else 
      Q_m_X += float(pixelsVec[i].adc);
    //
    // Y projection
    if (pixelsVec[i].y == ymin) 
      Q_f_Y += float(pixelsVec[i].adc);
    else if (pixelsVec[i].y == ymax) 
      Q_l_Y += float(pixelsVec[i].adc);
    else 
      Q_m_Y += float(pixelsVec[i].adc);
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
  setTheDet( det );

  float errx_sq = -99999.9;
  float erry_sq = -99999.9;

  bool errors_inflated = false;
  bool errors_standard = false;

  //cout << endl;
  //cout << "inflate_errors                  = " << (int)inflate_errors                  << endl;
  //cout << "inflate_all_errors_no_trk_angle = " << (int)inflate_all_errors_no_trk_angle << endl; 
  //cout << "with_track_angle                = " << (int)with_track_angle                << endl;

  int sizex = cluster.sizeX();
  int sizey = cluster.sizeY();

  if ( inflate_errors )
    {
      if ( !with_track_angle && inflate_all_errors_no_trk_angle )
	{
	  errors_inflated = true;
	} // if ( !with_track_angle && inflate_all_errors_no_trk_angle )
      else
	{
	  // here call templates to determine if cluster quality (qbin==0 -> bad cluster, qbin!=0 -> good cluster )
	  
	  int ID = 4;
	  
	  bool fpix;  //!< barrel(false) or forward(true)
	  if ( thePart == GeomDetEnumerators::PixelBarrel )   
	    fpix = false;    // no, it's not forward -- it's barrel
	  else                                              
	    fpix = true;     // yes, it's forward
	  
	  // Make from cluster (a SiPixelCluster) a boost multi_array_2d called clust_array_2d.
	  boost::multi_array<float, 2> clust_array_2d(boost::extents[7][21]);
	  
	  // Preparing to retrieve ADC counts from the SiPixelCluster.  In the cluster,
	  // we have the following:
	  //   int minPixelRow(); // Minimum pixel index in the x direction (low edge).
	  //   int maxPixelRow(); // Maximum pixel index in the x direction (top edge).
	  //   int minPixelCol(); // Minimum pixel index in the y direction (left edge).
	  //   int maxPixelCol(); // Maximum pixel index in the y direction (right edge).
	  // So the pixels from minPixelRow() will go into clust_array_2d[0][*],
	  // and the pixels from minPixelCol() will go into clust_array_2d[*][0].
	  int row_offset = cluster.minPixelRow();
	  int col_offset = cluster.minPixelCol();
      
	  // Store the coordinates of the center of the (0,0) pixel of the array that 
	  // gets passed to PixelTempReco2D
	  // Will add these values to the output of  PixelTempReco2D
	  float tmp_x = float(cluster.minPixelRow()) + 0.5;
	  float tmp_y = float(cluster.minPixelCol()) + 0.5;
	  
	  // Store these offsets (to be added later) in a LocalPoint after tranforming 
	  // them from measurement units (pixel units) to local coordinates (cm)
	  LocalPoint lp = theTopol->localPosition( MeasurementPoint(tmp_x, tmp_y) );
	  
	  const std::vector<SiPixelCluster::Pixel> & pixVec = cluster.pixels();
	  std::vector<SiPixelCluster::Pixel>::const_iterator 
	    pixIter = pixVec.begin(), pixEnd = pixVec.end();
	  
	  // Copy clust's pixels (calibrated in electrons) into clust_array_2d;
	  for ( ; pixIter != pixEnd; ++pixIter ) 
	    {
	      // *pixIter dereferences to Pixel struct, with public vars x, y, adc (all float)
	      // 02/13/2008 ggiurgiu@fnal.gov: type of x, y and adc has been changed to unsigned char, unsigned short, unsigned short
	      // in DataFormats/SiPixelCluster/interface/SiPixelCluster.h so the type cast to int is redundant. Leave it there, it 
	      // won't hurt. 
	      int irow = int(pixIter->x) - row_offset;   // &&& do we need +0.5 ???
	      int icol = int(pixIter->y) - col_offset;   // &&& do we need +0.5 ???
	      
	      // Gavril : what do we do here if the row/column is larger than 7/21 ?
	      // Ignore them for the moment...
	      if ( irow<7 && icol<21 )
		// 02/13/2008 ggiurgiu@fnal.gov typecast pixIter->adc to float
		clust_array_2d[irow][icol] = (float)pixIter->adc;
	      //else
	      //cout << " ----- Cluster is too large" << endl;
	    }
	  
	  // Make and fill the bool arrays flagging double pixels
	  // &&& Need to define constants for 7 and 21 somewhere!
	  std::vector<bool> ydouble(21), xdouble(7);
	  // x directions (shorter), rows
	  
	  bool n_bigx = 0;
	  bool n_bigy = 0;
	  
	  for (int irow = 0; irow < 7; ++irow)
	    {
	      xdouble[irow] = RectangularPixelTopology::isItBigPixelInX( irow+row_offset );
	      
	      if ( xdouble[irow] )
		++n_bigx;
	    }
	  
	  // y directions (longer), columns
	  for (int icol = 0; icol < 21; ++icol) 
	    {
	      ydouble[icol] = RectangularPixelTopology::isItBigPixelInY( icol+col_offset );
	      
	      if ( ydouble[icol] )
		++n_bigy;
	    }
	  
	  //cout << "n_bigx = " << n_bigx << endl;
	  //cout << "n_bigy = " << n_bigy << endl;
	  
	  // Output:
	  float nonsense = -99999.9; // nonsense init value
	  float templXrec_   = nonsense; 
	  float templYrec_   = nonsense;
	  float templSigmaX_ = nonsense;
	  float templSigmaY_ = nonsense;
	  float templProbY_  = nonsense;
	  float templProbX_  = nonsense;
	  
	  // ******************************************************************
	  // Do it! Use cotalpha_ and cotbeta_ calculated in PixelCPEBase
	  
	  SiPixelTemplate templ_;
	  templ_.pushfile(*templateDBobject_);
		
	  
	  int templQbin_ = -99999;
	  int speed_ = 0;
	  
	  bool ierr =
	    SiPixelTemplateReco::PixelTempReco2D( ID, fpix, cotalpha_, cotbeta_,
						  clust_array_2d, ydouble, xdouble,
						  templ_,
						  templYrec_, templSigmaY_, templProbY_,
						  templXrec_, templSigmaX_, templProbX_, 
						  templQbin_, 
						  speed_ );
	        
	  // ******************************************************************
	  	  
	  if ( templQbin_ == 0 || ierr !=0 )
	    {
	      errors_inflated = true;
	  
	    } // if ( templQbin_ == 0 )
	  else
	    {
	      errors_standard = true;
	      
	    } // if ( templQbin_ == 0 )... else
	
	} // if ( !with_track_angle && inflate_all_errors_no_trk_angle )... else
    
    } // if ( inflate_errors ) 
  else 
    {
      errors_standard = true;
            
    } // if ( inflate_errors )... else 
  

  //cout << "errors_standard = " << errors_standard << endl; 
  //cout << "errors_inflated = " << errors_inflated << endl; 

  
  if ( errors_standard && !errors_inflated )
	{
		//cout << "standard errors:" << endl;

		//--- Default is the maximum error used for edge clusters. 	 
		float xerr = thePitchX / sqrt(12.); 	 
		float yerr = thePitchY / sqrt(12.); 

		// Use edge methods from the Toplogy class
		int maxPixelCol = cluster.maxPixelCol();
		int maxPixelRow = cluster.maxPixelRow();
		int minPixelCol = cluster.minPixelCol();
		int minPixelRow = cluster.minPixelRow();       
		// edge method moved to topologu class
		bool edgex = (theTopol->isItEdgePixelInX(minPixelRow)) ||
			(theTopol->isItEdgePixelInX(maxPixelRow));
		bool edgey = (theTopol->isItEdgePixelInY(minPixelCol)) ||
			(theTopol->isItEdgePixelInY(maxPixelCol));

		bool bigInX = theTopol->containsBigPixelInX(minPixelRow, maxPixelRow); 	 
		bool bigInY = theTopol->containsBigPixelInY(minPixelCol, maxPixelCol);

		if (edgex && edgey) { 	 
			//--- Both axes on the edge, no point in calling PixelErrorParameterization, 	 
			//--- just return the max errors on both. 	 
		} else { 	 
			pair<float,float> errPair = 	 
				genErrorsFromDB_->getError(genErrorParm_, thePart, cluster.sizeX(), cluster.sizeY(), 	 
					alpha_, beta_, bigInX, bigInY); 
			if (!edgex) xerr = errPair.first; 	 
			if (!edgey) yerr = errPair.second; 	 
		} 	 
		
		if (theVerboseLevel > 9) { 	 
			LogDebug("PixelCPEGeneric") << 	 
				"Sizex = " << cluster.sizeX() << " Sizey = " << cluster.sizeY() << 	 
				" Edgex = " << edgex          << " Edgey = " << edgey << 	 
				" ErrX = " << xerr            << " ErrY  = " << yerr; 	 
		}

		errx_sq = xerr*xerr; 
		erry_sq = yerr*yerr;
		
	}
  else if ( errors_inflated && !errors_standard )
	{
      //cout << "inflated errors:" << endl;
            
      int n_bigx = 0;
      int n_bigy = 0;
      
      int row_offset = cluster.minPixelRow();
      int col_offset = cluster.minPixelCol();
      
      for (int irow = 0; irow < 7; ++irow)
	{
	  if ( RectangularPixelTopology::isItBigPixelInX( irow+row_offset ) )
	    ++n_bigx;
	}
      
      for (int icol = 0; icol < 21; ++icol) 
	{
	  if ( RectangularPixelTopology::isItBigPixelInY( icol+col_offset ) )
	    ++n_bigy;
	}
      
      float errx = (float)(sizex + n_bigx) * thePitchX / sqrt( 12.0 );
      float erry = (float)(sizey + n_bigy) * thePitchY / sqrt( 12.0 );
            
      errx_sq = errx*errx;
      erry_sq = erry*erry;
    }
  else
    {
      //cout << "Impossible !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
      //assert(0);
    }

  //  cout << "errors_inflated = " << errors_inflated << endl;
  //cout << "errors_standard = " << errors_standard << endl;

  // errors in microns
  //cout << "sqrt( errx_sq ) = " << sqrt( errx_sq )*10000.0 << endl;
  //cout << "sqrt( erry_sq ) = " << sqrt( erry_sq )*10000.0 << endl;
  //cout << endl;
  

  return LocalError( errx_sq, 0, erry_sq );

}






