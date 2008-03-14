
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGeneric.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"


// Services
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"


#include <iostream>
using namespace std;

const double HALF_PI = 1.57079632679489656;

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEGeneric::PixelCPEGeneric(edm::ParameterSet const & conf, 
				       const MagneticField *mag) 
  : PixelCPEBase(conf, mag)
{
  if (theVerboseLevel > 0) 
    LogDebug("PixelCPEGeneric") 
      << " constructing a generic algorithm for ideal pixel detector.\n"
      << " CPEGeneric:: VerboseLevel = " << theVerboseLevel;
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

  double angle_from_clust = 0;

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
			      1.0,
			      2.0,
			      4.0,   // cut for eff charge width &&&
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
			      1.0,
			      2.0,
			      3.0,    // cut for eff charge width  &&&
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
    + 2 * half_lorentz_shift;                    // (in cm) &&& check fpix!  
  

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

  // Obtain the cluster boundaries (note: in measurement units!)
  float xmin = cluster.minPixelRow()+0.5;
  float xmax = cluster.maxPixelRow()+0.5;
  
  float ymin = cluster.minPixelCol()+0.5;
  float ymax = cluster.maxPixelCol()+0.5;
  
  // Iterate over the pixels.
  int isize = pixelsVec.size();
  for (int i = 0;  i < isize; ++i) {
    //
    // X projection
    if (pixelsVec[i].x == xmin)       // need to match with tolerance!!! &&&
      Q_f_X += pixelsVec[i].adc;
    else if (pixelsVec[i].x == xmax) 
      Q_l_X += pixelsVec[i].adc;
    else 
      Q_m_X += pixelsVec[i].adc;
    //
    // Y projection
    if (pixelsVec[i].y == ymin) 
      Q_f_Y += pixelsVec[i].adc;
    else if (pixelsVec[i].y == ymax) 
      Q_l_Y += pixelsVec[i].adc;
    else 
      Q_m_Y += pixelsVec[i].adc;
  }

  return;
} 




//=================  ONLY OLD ERROR STUFF BELOW  ==========================



//-------------------------------------------------------------------------
// Hit error in measurement coordinates
//-------------------------------------------------------------------------
// MeasurementError 
// PixelCPEGeneric::measurementError( const SiPixelCluster& cluster, 
// 				      const GeomDetUnit & det) const {
//   LocalPoint lp( localPosition(cluster, det) );
//   LocalError le( localError(   cluster, det) );

//   return theTopol->measurementError( lp, le );
// }

//-------------------------------------------------------------------------
//  Hit error in the local frame
//-------------------------------------------------------------------------
LocalError  
PixelCPEGeneric::localError( const SiPixelCluster& cluster, 
				const GeomDetUnit & det) const {
  setTheDet( det );
  int sizex = cluster.sizeX();
  int sizey = cluster.sizeY();

  // Find edge clusters
  //bool edgex = (cluster.edgeHitX()) || (cluster.maxPixelRow()> theNumOfRow);//wrong 
  //bool edgey = (cluster.edgeHitY()) || (cluster.maxPixelCol() > theNumOfCol);   
  /* bool edgex = (cluster.minPixelRow()==0) ||  // use min and max pixels
   (cluster.maxPixelRow()==(theNumOfRow-1));
   bool edgey = (cluster.minPixelCol()==0) ||
   (cluster.maxPixelCol()==(theNumOfCol-1));*/

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
  
  //&&& testing...
  if (theVerboseLevel > 9) {
    LogDebug("PixelCPEGeneric") <<
      "Sizex = " << sizex << 
      " Sizey = " << sizey << 
      " Edgex = " << edgex << 
      " Edgey = " << edgey ;
  }

  return LocalError( err2X(edgex, sizex), 0, err2Y(edgey, sizey) );
}


//-----------------------------------------------------------------------------
// Position error estimate in X (square returned).
//-----------------------------------------------------------------------------
float 
PixelCPEGeneric::err2X(bool& edgex, int& sizex) const
{
// Assign maximum error
  // if edge cluster the maximum error is assigned: Pitch/sqrt(12)=43mu
  //  float xerr = 0.0043; 
  float xerr = thePitchX/3.464;
  //
  // Pixels not at the edge: errors parameterized as function of the cluster size
  // V.Chiochia - 12/4/06
  //
  if (!edgex){
    //    if (fabs(thePitchX-0.010)<0.001){   // 100um pixel size
      if (thePart == GeomDetEnumerators::PixelBarrel) {
	if ( sizex == 1) xerr = 0.00115;      // Size = 1 -> Sigma = 11.5 um 
	else if ( sizex == 2) xerr = 0.0012;  // Size = 2 -> Sigma = 12 um      
	else if ( sizex == 3) xerr = 0.00088; // Size = 3 -> Sigma = 8.8 um
	else xerr = 0.0103;
      } else { //forward
	if ( sizex == 1) {
	  xerr = 0.0020;
	}  else if ( sizex == 2) {
	  xerr = 0.0020;
	  // xerr = (0.005351 - atan(fabs(theDetZ/theDetR)) * 0.003291);  
	} else {
	  xerr = 0.0020;
	  //xerr = (0.003094 - atan(fabs(theDetZ/theDetR)) * 0.001854);  
	}
      }
      //    }
//     }else if (fabs(thePitchX-0.015)<0.001){  // 150 um pixel size
//       if (thePart == GeomDetEnumerators::PixelBarrel) {
// 	if ( sizex == 1) xerr = 0.0014;     // 14um 
// 	else xerr = 0.0008;   // 8um      
//       } else { //forward
// 	if ( sizex == 1) 
// 	  xerr = (-0.00385 + atan(fabs(theDetZ/theDetR)) * 0.00407);
// 	else xerr = (0.00366 - atan(fabs(theDetZ/theDetR)) * 0.00209);  
//       }
//     }

  }
  return xerr*xerr;
}


//-----------------------------------------------------------------------------
// Position error estimate in Y (square returned).
//-----------------------------------------------------------------------------
float 
PixelCPEGeneric::err2Y(bool& edgey, int& sizey) const
{
// Assign maximum error
// if edge cluster the maximum error is assigned: Pitch/sqrt(12)=43mu
//  float yerr = 0.0043;
  float yerr = thePitchY/3.464; 
  if (!edgey){
    if (thePart == GeomDetEnumerators::PixelBarrel) { // Barrel
      if ( sizey == 1) {
	yerr = 0.00375;     // 37.5um 
      } else if ( sizey == 2) {
	yerr = 0.0023;   // 23 um      
      } else if ( sizey == 3) {
	yerr = 0.0025; // 25 um
      } else if ( sizey == 4) {
	yerr = 0.0025; // 25um
      } else if ( sizey == 5) {
	yerr = 0.0023; // 23um
      } else if ( sizey == 6) {
	yerr = 0.0023; // 23um
      } else if ( sizey == 7) {
	yerr = 0.0021; // 21um
      } else if ( sizey == 8) {
	yerr = 0.0021; // 21um
      } else if ( sizey == 9) {
	yerr = 0.0024; // 24um
      } else if ( sizey >= 10) {
	yerr = 0.0021; // 21um
      }
    } else { // Endcaps
      if ( sizey == 1)      yerr = 0.0021; // 21 um
      else if ( sizey >= 2) yerr = 0.00075;// 7.5 um
    }
  }
  return yerr*yerr;
}




