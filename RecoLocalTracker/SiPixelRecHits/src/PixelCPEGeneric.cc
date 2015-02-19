#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGeneric.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

// this is needed to get errors from templates
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"
#include "DataFormats/DetId/interface/DetId.h"


// Services
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "boost/multi_array.hpp"

#include <iostream>
using namespace std;

#define NEW_CPEERROR // must be constistent with base.cc, generic cc/h and genericProducer.cc 

namespace {
  constexpr float micronsToCm = 1.0e-4;
  const bool MYDEBUG = false;
}

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
#ifdef NEW_CPEERROR
PixelCPEGeneric::PixelCPEGeneric(edm::ParameterSet const & conf, 
				 const MagneticField * mag,
                                 const TrackerGeometry& geom,
				 const TrackerTopology& ttopo,
				 const SiPixelLorentzAngle * lorentzAngle, 
				 const SiPixelGenErrorDBObject * genErrorDBObject, 
				 const SiPixelLorentzAngle * lorentzAngleWidth=0) 
  : PixelCPEBase(conf, mag, geom, ttopo, lorentzAngle, genErrorDBObject, 0,lorentzAngleWidth,0) {
#else
PixelCPEGeneric::PixelCPEGeneric(edm::ParameterSet const & conf, 
				 const MagneticField * mag,
                                 const TrackerGeometry& geom,
				 const TrackerTopology& ttopo,
				 const SiPixelLorentzAngle * lorentzAngle, 
				 const SiPixelGenErrorDBObject * genErrorDBObject, 
				 const SiPixelTemplateDBObject * templateDBobject,
				 const SiPixelLorentzAngle * lorentzAngleWidth=0) 
  : PixelCPEBase(conf, mag, geom, ttopo, lorentzAngle, genErrorDBObject, templateDBobject,lorentzAngleWidth,0) {
#endif

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
  //LoadTemplatesFromDB_       = conf.getParameter<bool>("LoadTemplatesFromDB");

  bool isUpgrade=false;
  if ( conf.exists("Upgrade") && conf.getParameter<bool>("Upgrade")) {
    isUpgrade=true;
    xerr_barrel_ln_= {0.00114,0.00104,0.00214};
    xerr_barrel_ln_def_=0.00425;
    yerr_barrel_ln_= {0.00299,0.00203,0.0023,0.00237,0.00233,0.00243,0.00232,0.00259,0.00176};
    yerr_barrel_ln_def_=0.00245;
    xerr_endcap_= {0.00151,0.000813,0.00221};
    xerr_endcap_def_=0.00218;
    yerr_endcap_= {0.00261,0.00107,0.00264};
    yerr_endcap_def_=0.00357;
    
    if ( conf.exists("SmallPitch") && conf.getParameter<bool>("SmallPitch")) {
      xerr_barrel_l1_= {0.00104, 0.000691, 0.00122};
      xerr_barrel_l1_def_=0.00321;
      yerr_barrel_l1_= {0.00199,0.00136,0.0015,0.00153,0.00152,0.00171,0.00154,0.00157,0.00154};
      yerr_barrel_l1_def_=0.00164;
    }
    else{
      xerr_barrel_l1_= {0.00114,0.00104,0.00214};
      xerr_barrel_l1_def_=0.00425;
      yerr_barrel_l1_= {0.00299,0.00203,0.0023,0.00237,0.00233,0.00243,0.00232,0.00259,0.00176};
      yerr_barrel_l1_def_=0.00245;
    }
  }
  isUpgrade_=isUpgrade;

  // Select the position error source 
  // For upgrde and cosmics force the use simple errors 
  if( isUpgrade_ || (DoCosmics_) ) UseErrorsFromTemplates_ = false;

  if ( !UseErrorsFromTemplates_ && ( TruncatePixelCharge_       || 
				     IrradiationBiasCorrection_ || 
				     DoCosmics_                 ||
				     LoadTemplatesFromDB_ ) )  {
    throw cms::Exception("PixelCPEGeneric::PixelCPEGeneric: ") 
      << "\nERROR: UseErrorsFromTemplates_ is set to False in PixelCPEGeneric_cfi.py. "
      << " In this case it does not make sense to set any of the following to True: " 
      << " TruncatePixelCharge_, IrradiationBiasCorrection_, DoCosmics_, LoadTemplatesFromDB_ !!!" 
      << "\n\n";
  }

  // Use errors from templates or from GenError
  if ( UseErrorsFromTemplates_ ) {

#ifdef NEW_CPEERROR

      if ( LoadTemplatesFromDB_ )  {  // From DB
	if ( !SiPixelGenError::pushfile( *genErrorDBObject_, thePixelGenError_) )
          throw cms::Exception("InvalidCalibrationLoaded") 
            << "ERROR: GenErrors not filled correctly. Check the sqlite file. Using SiPixelTemplateDBObject version " 
            << ( *genErrorDBObject_ ).version();
	if(MYDEBUG) cout<<"Loaded genErrorDBObject v"<<( *genErrorDBObject_ ).version()<<endl;
      } else  { // From file      
        if ( !SiPixelGenError::pushfile( -999, thePixelGenError_ ) )
          throw cms::Exception("InvalidCalibrationLoaded") 
            << "ERROR: GenErrors not loaded correctly from text file. Reconstruction will fail.";
      } // if load from DB

#else 

      if ( LoadTemplatesFromDB_ ) {
	// Initialize template store to the selected ID [Morris, 6/25/08]  
	if ( !SiPixelTemplate::pushfile( *templateDBobject_, thePixelTemp_) )
	  throw cms::Exception("InvalidCalibrationLoaded") 
	    << "ERROR: Templates not filled correctly. Check the sqlite file. Using SiPixelTemplateDBObject version " 
	    << ( *templateDBobject_ ).version();
	if(MYDEBUG) cout<<"Loaded templateDBobject "<<( *templateDBobject_ ).version()<<endl;

      } else {
	if ( !SiPixelTemplate::pushfile( -999, thePixelTemp_ ) )
	  throw cms::Exception("InvalidCalibrationLoaded") 
	    << "ERROR: Templates not loaded correctly from text file. Reconstruction will fail.";
      } // if load from DB

#endif // NEW_CPEERROR

  } // if ( UseErrorsFromTemplates_ )
  
  if(MYDEBUG) {
    cout << "From PixelCPEGeneric::PixelCPEGeneric(...)" << endl;
    cout << "(int)UseErrorsFromTemplates_ = " << (int)UseErrorsFromTemplates_    << endl;
    cout << "TruncatePixelCharge_         = " << (int)TruncatePixelCharge_       << endl;      
    cout << "IrradiationBiasCorrection_   = " << (int)IrradiationBiasCorrection_ << endl;
    cout << "(int)DoCosmics_              = " << (int)DoCosmics_                 << endl;
    cout << "(int)LoadTemplatesFromDB_    = " << (int)LoadTemplatesFromDB_       << endl;
  }


  // Default case for rechit errors in case other, more correct, errors are not vailable
  // This are constants. Maybe there is a more efficienct way to store them.
  xerr_barrel_l1_= {0.00115, 0.00120, 0.00088};
  xerr_barrel_l1_def_=0.01030;
  yerr_barrel_l1_= {0.00375,0.00230,0.00250,0.00250,0.00230,0.00230,0.00210,0.00210,0.00240};
  yerr_barrel_l1_def_=0.00210;
  xerr_barrel_ln_= {0.00115, 0.00120, 0.00088};
  xerr_barrel_ln_def_=0.01030;
  yerr_barrel_ln_= {0.00375,0.00230,0.00250,0.00250,0.00230,0.00230,0.00210,0.00210,0.00240};
  yerr_barrel_ln_def_=0.00210;
  xerr_endcap_= {0.0020, 0.0020};
  xerr_endcap_def_=0.0020;
  yerr_endcap_= {0.00210};
  yerr_endcap_def_=0.00075;

}

PixelCPEBase::ClusterParam* PixelCPEGeneric::createClusterParam(const SiPixelCluster & cl) const
{
   return new ClusterParamGeneric(cl);
}



//-----------------------------------------------------------------------------
//! Hit position in the local frame (in cm).  Unlike other CPE's, this
//! one converts everything from the measurement frame (in channel numbers) 
//! into the local frame (in centimeters).  
//-----------------------------------------------------------------------------
LocalPoint
PixelCPEGeneric::localPosition(DetParam const & theDetParam, ClusterParam & theClusterParamBase) const 
{

  ClusterParamGeneric & theClusterParam = static_cast<ClusterParamGeneric &>(theClusterParamBase);

  //cout<<" in PixelCPEGeneric:localPosition - "<<endl; //dk

  float chargeWidthX = (theDetParam.lorentzShiftInCmX * theDetParam.widthLAFractionX);
  float chargeWidthY = (theDetParam.lorentzShiftInCmY * theDetParam.widthLAFractionY);
  float shiftX = 0.5f*theDetParam.lorentzShiftInCmX;
  float shiftY = 0.5f*theDetParam.lorentzShiftInCmY;

  //cout<<" main la width "<<chargeWidthX<<" "<<chargeWidthY<<endl;

  if ( UseErrorsFromTemplates_ ) {

      float qclus = theClusterParam.theCluster->charge();     
      float locBz = theDetParam.bz;
      //cout << "PixelCPEGeneric::localPosition(...) : locBz = " << locBz << endl;
      
      theClusterParam.pixmx  = -999.9; // max pixel charge for truncation of 2-D cluster
      theClusterParam.sigmay = -999.9; // CPE Generic y-error for multi-pixel cluster
      theClusterParam.deltay = -999.9; // CPE Generic y-bias for multi-pixel cluster
      theClusterParam.sigmax = -999.9; // CPE Generic x-error for multi-pixel cluster
      theClusterParam.deltax = -999.9; // CPE Generic x-bias for multi-pixel cluster
      theClusterParam.sy1    = -999.9; // CPE Generic y-error for single single-pixel
      theClusterParam.dy1    = -999.9; // CPE Generic y-bias for single single-pixel cluster
      theClusterParam.sy2    = -999.9; // CPE Generic y-error for single double-pixel cluster
      theClusterParam.dy2    = -999.9; // CPE Generic y-bias for single double-pixel cluster
      theClusterParam.sx1    = -999.9; // CPE Generic x-error for single single-pixel cluster
      theClusterParam.dx1    = -999.9; // CPE Generic x-bias for single single-pixel cluster
      theClusterParam.sx2    = -999.9; // CPE Generic x-error for single double-pixel cluster
      theClusterParam.dx2    = -999.9; // CPE Generic x-bias for single double-pixel cluster
  

#ifdef NEW_CPEERROR

      SiPixelGenError gtempl(thePixelGenError_);
      int gtemplID_ = theDetParam.detTemplateId;

      //int gtemplID0 = genErrorDBObject_->getGenErrorID(theDetParam.theDet->geographicalId().rawId());
      //if(gtemplID0!=gtemplID_) cout<<" different id "<< gtemplID_<<" "<<gtemplID0<<endl;

      theClusterParam.qBin_ = gtempl.qbin( gtemplID_, theClusterParam.cotalpha, theClusterParam.cotbeta, locBz, qclus,  
					   theClusterParam.pixmx, theClusterParam.sigmay, theClusterParam.deltay, 
					   theClusterParam.sigmax, theClusterParam.deltax, theClusterParam.sy1, 
					   theClusterParam.dy1, theClusterParam.sy2, theClusterParam.dy2, theClusterParam.sx1, 
					   theClusterParam.dx1, theClusterParam.sx2, theClusterParam.dx2 );  
      
      // now use the charge widths stored in the new generic template headers (change to the
      // incorrect sign convention of the base class)       
      bool useLAWidthFromGenError = false;
      if(useLAWidthFromGenError) {
	chargeWidthX = (-micronsToCm*gtempl.lorxwidth());
	chargeWidthY = (-micronsToCm*gtempl.lorywidth());
	if(MYDEBUG) cout<< " redefine la width (gen-error) "<< chargeWidthX<<" "<< chargeWidthY <<endl;
      }
      if(MYDEBUG) cout<<" GenError: "<<gtemplID_<<endl;
      
#else // select templates

      SiPixelTemplate templ(thePixelTemp_);
      int templID_ = theDetParam.detTemplateId;
      
      theClusterParam.qBin_ = templ.qbin( templID_, theClusterParam.cotalpha, theClusterParam.cotbeta, locBz, qclus,  // inputs
					  theClusterParam.pixmx,                                       // returned by reference
					  theClusterParam.sigmay, theClusterParam.deltay, theClusterParam.sigmax, theClusterParam.deltax,              // returned by reference
					  theClusterParam.sy1, theClusterParam.dy1, theClusterParam.sy2, theClusterParam.dy2, theClusterParam.sx1, theClusterParam.dx1, theClusterParam.sx2, theClusterParam.dx2 );    // returned by reference
      
      //if(MYDEBUG) {
      //cout<<" errors "<<templID_<<" "<<theClusterParam.qBin_<<" "<<theClusterParam.sigmax<<" "<<theClusterParam.sigmay<<endl;
      //int templID0 = templateDBobject_->getTemplateID(theDetParam.theDet->geographicalId().rawId());
      //if(templID0!=templID_) cout<<" different id"<< templID_<<" "<<templID0<<endl;
      //}

#endif // NEW_CPEERROR

      // These numbers come in microns from the qbin(...) call. Transform them to cm.
      theClusterParam.deltax = theClusterParam.deltax * micronsToCm;
      theClusterParam.dx1 = theClusterParam.dx1 * micronsToCm;
      theClusterParam.dx2 = theClusterParam.dx2 * micronsToCm;
      
      theClusterParam.deltay = theClusterParam.deltay * micronsToCm;
      theClusterParam.dy1 = theClusterParam.dy1 * micronsToCm;
      theClusterParam.dy2 = theClusterParam.dy2 * micronsToCm;
      
      theClusterParam.sigmax = theClusterParam.sigmax * micronsToCm;
      theClusterParam.sx1 = theClusterParam.sx1 * micronsToCm;
      theClusterParam.sx2 = theClusterParam.sx2 * micronsToCm;
      
      theClusterParam.sigmay = theClusterParam.sigmay * micronsToCm;
      theClusterParam.sy1 = theClusterParam.sy1 * micronsToCm;
      theClusterParam.sy2 = theClusterParam.sy2 * micronsToCm;
      
  } // if ( UseErrorsFromTemplates_ )
  
  float Q_f_X = 0.0;        //!< Q of the first  pixel  in X 
  float Q_l_X = 0.0;        //!< Q of the last   pixel  in X
  float Q_f_Y = 0.0;        //!< Q of the first  pixel  in Y 
  float Q_l_Y = 0.0;        //!< Q of the last   pixel  in Y
  collect_edge_charges( theClusterParam, 
			Q_f_X, Q_l_X, 
			Q_f_Y, Q_l_Y );

  //--- Find the inner widths along X and Y in one shot.  We
  //--- compute the upper right corner of the inner pixels
  //--- (== lower left corner of upper right pixel) and
  //--- the lower left corner of the inner pixels
  //--- (== upper right corner of lower left pixel), and then
  //--- subtract these two points in the formula.

  //--- Upper Right corner of Lower Left pixel -- in measurement frame
  MeasurementPoint meas_URcorn_LLpix( theClusterParam.theCluster->minPixelRow()+1.0,
				      theClusterParam.theCluster->minPixelCol()+1.0 );

  //--- Lower Left corner of Upper Right pixel -- in measurement frame
  MeasurementPoint meas_LLcorn_URpix( theClusterParam.theCluster->maxPixelRow(),
				      theClusterParam.theCluster->maxPixelCol() );

  //--- These two now converted into the local  
  LocalPoint local_URcorn_LLpix;
  LocalPoint local_LLcorn_URpix;

  // PixelCPEGeneric can be used with or without track angles
  // If PixelCPEGeneric is called with track angles, use them to correct for bows/kinks:
  if ( theClusterParam.with_track_angle ) {
    local_URcorn_LLpix = theDetParam.theTopol->localPosition(meas_URcorn_LLpix, theClusterParam.loc_trk_pred);
    local_LLcorn_URpix = theDetParam.theTopol->localPosition(meas_LLcorn_URpix, theClusterParam.loc_trk_pred);
  } else {
    local_URcorn_LLpix = theDetParam.theTopol->localPosition(meas_URcorn_LLpix);
    local_LLcorn_URpix = theDetParam.theTopol->localPosition(meas_LLcorn_URpix);
  }

 #ifdef EDM_ML_DEBUG
  if (theVerboseLevel > 20) {
    cout  
      << "\n\t >>> theClusterParam.theCluster->x = " << theClusterParam.theCluster->x()
      << "\n\t >>> theClusterParam.theCluster->y = " << theClusterParam.theCluster->y()
      << "\n\t >>> cluster: minRow = " << theClusterParam.theCluster->minPixelRow()
      << "  minCol = " << theClusterParam.theCluster->minPixelCol()
      << "\n\t >>> cluster: maxRow = " << theClusterParam.theCluster->maxPixelRow()
      << "  maxCol = " << theClusterParam.theCluster->maxPixelCol()
      << "\n\t >>> meas: inner lower left  = " << meas_URcorn_LLpix.x() 
      << "," << meas_URcorn_LLpix.y()
      << "\n\t >>> meas: inner upper right = " << meas_LLcorn_URpix.x() 
      << "," << meas_LLcorn_URpix.y() 
      << endl;
  }
#endif

  //--- &&& Note that the cuts below should not be hardcoded (like in Orca and
  //--- &&& CPEFromDetPosition/PixelCPEInitial), but rather be
  //--- &&& externally settable (but tracked) parameters.  

  //--- Position, including the half lorentz shift

 #ifdef EDM_ML_DEBUG
  if (theVerboseLevel > 20) 
    cout << "\t >>> Generic:: processing X" << endl;
#endif

  float xPos = 
    generic_position_formula( theClusterParam.theCluster->sizeX(),
			      Q_f_X, Q_l_X, 
			      local_URcorn_LLpix.x(), local_LLcorn_URpix.x(),
			      chargeWidthX,   // lorentz shift in cm
			      theDetParam.theThickness,
			      theClusterParam.cotalpha,
			      theDetParam.thePitchX,
			      theDetParam.theRecTopol->isItBigPixelInX( theClusterParam.theCluster->minPixelRow() ),
			      theDetParam.theRecTopol->isItBigPixelInX( theClusterParam.theCluster->maxPixelRow() ),
			      the_eff_charge_cut_lowX,
                              the_eff_charge_cut_highX,
                              the_size_cutX);           // cut for eff charge width &&&


  // apply the lorentz offset correction 			     
  xPos = xPos + shiftX;

#ifdef EDM_ML_DEBUG
  if (theVerboseLevel > 20) 
    cout << "\t >>> Generic:: processing Y" << endl;
#endif

  float yPos = 
    generic_position_formula( theClusterParam.theCluster->sizeY(),
			      Q_f_Y, Q_l_Y, 
			      local_URcorn_LLpix.y(), local_LLcorn_URpix.y(),
			      chargeWidthY,   // lorentz shift in cm
			      theDetParam.theThickness,
			      theClusterParam.cotbeta,
			      theDetParam.thePitchY,  
			      theDetParam.theRecTopol->isItBigPixelInY( theClusterParam.theCluster->minPixelCol() ),
			      theDetParam.theRecTopol->isItBigPixelInY( theClusterParam.theCluster->maxPixelCol() ),
			      the_eff_charge_cut_lowY,
                              the_eff_charge_cut_highY,
                              the_size_cutY);           // cut for eff charge width &&&

  // apply the lorentz offset correction 			     
  yPos = yPos + shiftY;

  // Apply irradiation corrections. NOT USED FOR NOW
  if ( IrradiationBiasCorrection_ ) {
    if ( theClusterParam.theCluster->sizeX() == 1 ) {  // size=1	  
      // ggiurgiu@jhu.edu, 02/03/09 : for size = 1, the Lorentz shift is already accounted by the irradiation correction
      //float tmp1 =  (0.5 * theDetParam.lorentzShiftInCmX);
      //cout << "Apply correction correction_dx1 = " << theClusterParam.dx1 << " to xPos = " << xPos;
      xPos = xPos - (0.5 * theDetParam.lorentzShiftInCmX);
      // Find if pixel is double (big). 
      bool bigInX = theDetParam.theRecTopol->isItBigPixelInX( theClusterParam.theCluster->maxPixelRow() );	  
      if ( !bigInX ) xPos -= theClusterParam.dx1;	   
      else           xPos -= theClusterParam.dx2;
      //cout<<" to "<<xPos<<" "<<(tmp1+theClusterParam.dx1)<<endl;
    } else { // size>1
      //cout << "Apply correction correction_deltax = " << theClusterParam.deltax << " to xPos = " << xPos;
      xPos -= theClusterParam.deltax;
      //cout<<" to "<<xPos<<endl;
    }
    
    if ( theClusterParam.theCluster->sizeY() == 1 ) {
      // ggiurgiu@jhu.edu, 02/03/09 : for size = 1, the Lorentz shift is already accounted by the irradiation correction
      yPos = yPos - (0.5 * theDetParam.lorentzShiftInCmY);
      
      // Find if pixel is double (big). 
      bool bigInY = theDetParam.theRecTopol->isItBigPixelInY( theClusterParam.theCluster->maxPixelCol() );      
      if ( !bigInY ) yPos -= theClusterParam.dy1;
      else           yPos -= theClusterParam.dy2;
      
    } else { 
      //cout << "Apply correction correction_deltay = " << theClusterParam.deltay << " to yPos = " << yPos << endl;
      yPos -= theClusterParam.deltay; 
    }
 
  } // if ( IrradiationBiasCorrection_ )
	
  //cout<<" in PixelCPEGeneric:localPosition - pos = "<<xPos<<" "<<yPos<<endl; //dk

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
float
PixelCPEGeneric::    
generic_position_formula( int size,                //!< Size of this projection.
			  float Q_f,              //!< Charge in the first pixel.
			  float Q_l,              //!< Charge in the last pixel.
			  float upper_edge_first_pix, //!< As the name says.
			  float lower_edge_last_pix,  //!< As the name says.
			  float lorentz_shift,   //!< L-shift at half thickness
                          float theThickness,   //detector thickness
			  float cot_angle,        //!< cot of alpha_ or beta_
			  float pitch,            //!< thePitchX or thePitchY
			  bool first_is_big,       //!< true if the first is big
			  bool last_is_big,        //!< true if the last is big
			  float eff_charge_cut_low, //!< Use edge if > W_eff  &&&
			  float eff_charge_cut_high,//!< Use edge if < W_eff  &&&
			  float size_cut         //!< Use edge when size == cuts
			 ) const
{

  //cout<<" in PixelCPEGeneric:generic_position_formula - "<<endl; //dk

  float geom_center = 0.5f * ( upper_edge_first_pix + lower_edge_last_pix );

  //--- The case of only one pixel in this projection is separate.  Note that
  //--- here first_pix == last_pix, so the average of the two is still the
  //--- center of the pixel.
  if ( size == 1 ) {return geom_center;}

  //--- Width of the clusters minus the edge (first and last) pixels.
  //--- In the note, they are denoted x_F and x_L (and y_F and y_L)
  float W_inner      = lower_edge_last_pix - upper_edge_first_pix;  // in cm

  //--- Predicted charge width from geometry
  float W_pred = theThickness * cot_angle                     // geometric correction (in cm)
    - lorentz_shift;                    // (in cm) &&& check fpix!  

  //cout<<" in PixelCPEGeneric:generic_position_formula - "<<W_inner<<" "<<W_pred<<endl; //dk

  //--- Total length of the two edge pixels (first+last)
  float sum_of_edge = 2.0f;
  if (first_is_big) sum_of_edge += 1.0f;
  if (last_is_big)  sum_of_edge += 1.0f;
  

  //--- The `effective' charge width -- particle's path in first and last pixels only
  float W_eff = std::abs( W_pred ) - W_inner;


  //--- If the observed charge width is inconsistent with the expectations
  //--- based on the track, do *not* use W_pred-W_innner.  Instead, replace
  //--- it with an *average* effective charge width, which is the average
  //--- length of the edge pixels.
  //
  //  bool usedEdgeAlgo = false;
  if ( (size >= size_cut) || (
       ( W_eff/pitch < eff_charge_cut_low ) |
       ( W_eff/pitch > eff_charge_cut_high ) ) ) 
    {
      W_eff = pitch * 0.5f * sum_of_edge;  // ave. length of edge pixels (first+last) (cm)
      //  usedEdgeAlgo = true;
#ifdef EDM_ML_DEBUG
      nRecHitsUsedEdge_++;
#endif
    }

  
  //--- Finally, compute the position in this projection
  float Qdiff = Q_l - Q_f;
  float Qsum  = Q_l + Q_f;

  //--- Temporary fix for clusters with both first and last pixel with charge = 0
  if(Qsum==0) Qsum=1.0f;
  //float hit_pos = geom_center + 0.5f*(Qdiff/Qsum) * W_eff + half_lorentz_shift;
  float hit_pos = geom_center + 0.5f*(Qdiff/Qsum) * W_eff;

  //cout<<" in PixelCPEGeneric:generic_position_formula - "<<hit_pos<<" "<<lorentz_shift*0.5<<endl; //dk

 #ifdef EDM_ML_DEBUG
  //--- Debugging output
#warning "Debug printouts in PixelCPEGeneric.cc has been commented because they cannot be compiled"
  /*   This part is commented because some variables used here are not defined !!
  if (theVerboseLevel > 20) {
    if ( theDetParam.thePart == GeomDetEnumerators::PixelBarrel || theDetParam.thePart == GeomDetEnumerators::P1PXB ) {
      cout << "\t >>> We are in the Barrel." ;
    } else if ( theDetParam.thePart == GeomDetEnumerators::PixelEndcap || 
		theDetParam.thePart == GeomDetEnumerators::P1PXEC ||
		theDetParam.thePart == GeomDetEnumerators::P2PXEC ) {
      cout << "\t >>> We are in the Forward." ;
    } else {
      cout << "\t >>> We are in an unexpected subdet " << theDetParam.thePart;
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
  */
#endif

  return hit_pos;
}


//-----------------------------------------------------------------------------
//!  Collect the edge charges in x and y, in a single pass over the pixel vector.
//!  Calculate charge in the first and last pixel projected in x and y
//!  and the inner cluster charge, projected in x and y.
//-----------------------------------------------------------------------------
void
PixelCPEGeneric::
collect_edge_charges(ClusterParam & theClusterParamBase,  //!< input, the cluster
		     float & Q_f_X,              //!< output, Q first  in X 
		     float & Q_l_X,              //!< output, Q last   in X
		     float & Q_f_Y,              //!< output, Q first  in Y 
		     float & Q_l_Y               //!< output, Q last   in Y
		     ) const
{
  ClusterParamGeneric & theClusterParam = static_cast<ClusterParamGeneric &>(theClusterParamBase);

  // Initialize return variables.
  Q_f_X = Q_l_X = 0.0;
  Q_f_Y = Q_l_Y = 0.0;


  // Obtain boundaries in index units
  int xmin = theClusterParam.theCluster->minPixelRow();
  int xmax = theClusterParam.theCluster->maxPixelRow();
  int ymin = theClusterParam.theCluster->minPixelCol();
  int ymax = theClusterParam.theCluster->maxPixelCol();


  // Iterate over the pixels.
  int isize = theClusterParam.theCluster->size();
  for (int i = 0;  i != isize; ++i) 
    {
      auto const & pixel = theClusterParam.theCluster->pixel(i); 
      // ggiurgiu@fnal.gov: add pixel charge truncation
      float pix_adc = pixel.adc;
      if ( UseErrorsFromTemplates_ && TruncatePixelCharge_ ) 
	pix_adc = std::min(pix_adc, theClusterParam.pixmx );

      //
      // X projection
      if ( pixel.x == xmin ) Q_f_X += pix_adc;
      if ( pixel.x == xmax ) Q_l_X += pix_adc;
      //
      // Y projection
      if ( pixel.y == ymin ) Q_f_Y += pix_adc;
      if ( pixel.y == ymax ) Q_l_Y += pix_adc;
    }
  
  return;
} 


//==============  INFLATED ERROR AND ERRORS FROM DB BELOW  ================

//-------------------------------------------------------------------------
//  Hit error in the local frame
//-------------------------------------------------------------------------
LocalError 
PixelCPEGeneric::localError(DetParam const & theDetParam,  ClusterParam & theClusterParamBase) const 
{

  ClusterParamGeneric & theClusterParam = static_cast<ClusterParamGeneric &>(theClusterParamBase);

  const bool localPrint = false;
  // Default errors are the maximum error used for edge clusters.
  // These are determined by looking at residuals for edge clusters
  float xerr = EdgeClusterErrorX_ * micronsToCm;
  float yerr = EdgeClusterErrorY_ * micronsToCm;
  

  // Find if cluster is at the module edge. 
  int maxPixelCol = theClusterParam.theCluster->maxPixelCol();
  int maxPixelRow = theClusterParam.theCluster->maxPixelRow();
  int minPixelCol = theClusterParam.theCluster->minPixelCol();
  int minPixelRow = theClusterParam.theCluster->minPixelRow();       

  bool edgex = ( theDetParam.theRecTopol->isItEdgePixelInX( minPixelRow ) ) || ( theDetParam.theRecTopol->isItEdgePixelInX( maxPixelRow ) );
  bool edgey = ( theDetParam.theRecTopol->isItEdgePixelInY( minPixelCol ) ) || ( theDetParam.theRecTopol->isItEdgePixelInY( maxPixelCol ) );

  unsigned int sizex = theClusterParam.theCluster->sizeX();
  unsigned int sizey = theClusterParam.theCluster->sizeY();
  if(MYDEBUG) {
    if( int(sizex) != (maxPixelRow - minPixelRow+1) ) cout<<" wrong x"<<endl;
    if( int(sizey) != (maxPixelCol - minPixelCol+1) ) cout<<" wrong y"<<endl;
  }

  // Find if cluster contains double (big) pixels. 
  bool bigInX = theDetParam.theRecTopol->containsBigPixelInX( minPixelRow, maxPixelRow ); 	 
  bool bigInY = theDetParam.theRecTopol->containsBigPixelInY( minPixelCol, maxPixelCol );

  if(localPrint) {
   cout<<" endge clus "<<xerr<<" "<<yerr<<endl;  //dk
   if(bigInX || bigInY) cout<<" big "<<bigInX<<" "<<bigInY<<endl;
   if(edgex || edgey) cout<<" edge "<<edgex<<" "<<edgey<<endl;
   cout<<" before if "<<UseErrorsFromTemplates_<<" "<<theClusterParam.qBin_<<endl;
   if(theClusterParam.qBin_ == 0) 
     cout<<" qbin 0! "<<edgex<<" "<<edgey<<" "<<bigInX<<" "<<bigInY<<" "
	<<sizex<<" "<<sizey<<endl;
  }

  //if likely(UseErrorsFromTemplates_ && (qBin_!= 0) ) {
  if likely(UseErrorsFromTemplates_ ) {
      //
      // Use template errors 
      //cout << "Track angles are known. We can use either errors from templates or the error parameterization from DB." << endl;
      

      if ( !edgex ) { // Only use this for non-edge clusters
	if ( sizex == 1 ) {
	  if ( !bigInX ) {xerr = theClusterParam.sx1;} 
	  else           {xerr = theClusterParam.sx2;}
	} else {xerr = theClusterParam.sigmax;}		  
      }
	      
      if ( !edgey ) { // Only use for non-edge clusters
	if ( sizey == 1 ) {
	  if ( !bigInY ) {yerr = theClusterParam.sy1;}
	  else           {yerr = theClusterParam.sy2;}
	} else {yerr = theClusterParam.sigmay;}		  
      }

      if(localPrint) {
	cout<<" in if "<<edgex<<" "<<edgey<<" "<<sizex<<" "<<sizey<<endl;
	cout<<" errors  "<<xerr<<" "<<yerr<<" "<<theClusterParam.sx1<<" "<<theClusterParam.sx2<<" "<<theClusterParam.sigmax<<endl;  //dk
      }

  } else  { // simple errors

    // This are the simple errors, hardcoded in the code 
    cout << "Track angles are not known and we are processing cosmics." << endl; 
    //cout << "Default angle estimation which assumes track from PV (0,0,0) does not work." << endl;
      
    if ( theDetParam.thePart == GeomDetEnumerators::PixelBarrel || theDetParam.thePart == GeomDetEnumerators::P1PXB )  {

      DetId id = (theDetParam.theDet->geographicalId());
      int layer=ttopo_.layer(id);
      if ( layer==1 ) {
	if ( !edgex ) {
	  if ( sizex<=xerr_barrel_l1_.size() ) xerr=xerr_barrel_l1_[sizex-1];
	  else xerr=xerr_barrel_l1_def_;
	}
	
	if ( !edgey ) {
	  if ( sizey<=yerr_barrel_l1_.size() ) yerr=yerr_barrel_l1_[sizey-1];
	  else yerr=yerr_barrel_l1_def_;
	}
      } else{  // layer 2,3
	if ( !edgex ) {
	  if ( sizex<=xerr_barrel_ln_.size() ) xerr=xerr_barrel_ln_[sizex-1];
	  else xerr=xerr_barrel_ln_def_;
	}
	    
	if ( !edgey ) {
	  if ( sizey<=yerr_barrel_ln_.size() ) yerr=yerr_barrel_ln_[sizey-1];
	  else yerr=yerr_barrel_ln_def_;
	}
      }

    } else if ( theDetParam.thePart == GeomDetEnumerators::PixelEndcap || 
		theDetParam.thePart == GeomDetEnumerators::P1PXEC  ||
		theDetParam.thePart == GeomDetEnumerators::P2PXEC )  { // EndCap

      if ( !edgex ) {
	if ( sizex<=xerr_endcap_.size() ) xerr=xerr_endcap_[sizex-1];
	else xerr=xerr_endcap_def_;
      }
      
      if ( !edgey ) {
	if ( sizey<=yerr_endcap_.size() ) yerr=yerr_endcap_[sizey-1];
	else yerr=yerr_endcap_def_;
      }
    } // end endcap

    if(inflate_errors) {
      int n_bigx = 0;
      int n_bigy = 0;
      
      for (int irow = 0; irow < 7; ++irow) {
	if ( theDetParam.theRecTopol->isItBigPixelInX( irow+minPixelRow ) ) ++n_bigx;
      }
      
      for (int icol = 0; icol < 21; ++icol) {
	  if ( theDetParam.theRecTopol->isItBigPixelInY( icol+minPixelCol ) ) ++n_bigy;
      }
      
      xerr = (float)(sizex + n_bigx) * theDetParam.thePitchX / std::sqrt( 12.0f );
      yerr = (float)(sizey + n_bigy) * theDetParam.thePitchY / std::sqrt( 12.0f );
      
    } // if(inflate_errors)

  } // end 

#ifdef EDM_ML_DEBUG
  if ( !(xerr > 0.0) )
    throw cms::Exception("PixelCPEGeneric::localError") 
      << "\nERROR: Negative pixel error xerr = " << xerr << "\n\n";
  
  if ( !(yerr > 0.0) )
    throw cms::Exception("PixelCPEGeneric::localError") 
      << "\nERROR: Negative pixel error yerr = " << yerr << "\n\n";
#endif
 
  //if(localPrint) {
  //cout<<" errors  "<<xerr<<" "<<yerr<<endl;  //dk
  //if(theClusterParam.qBin_ == 0) cout<<" qbin 0 "<<xerr<<" "<<yerr<<endl;
  //}

  auto xerr_sq = xerr*xerr; 
  auto yerr_sq = yerr*yerr;
  
  return LocalError( xerr_sq, 0, yerr_sq );

}






