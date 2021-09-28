#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericForBricked.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "DataFormats/DetId/interface/DetId.h"

// Pixel templates contain the rec hit error parameterizaiton
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"

// The generic formula
#include "CondFormats/SiPixelTransient/interface/SiPixelUtils.h"

// Services
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "boost/multi_array.hpp"

#include <iostream>
using namespace std;

namespace {
  constexpr float micronsToCm = 1.0e-4;
}  // namespace

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEGenericForBricked::PixelCPEGenericForBricked(edm::ParameterSet const& conf,
                                                     const MagneticField* mag,
                                                     const TrackerGeometry& geom,
                                                     const TrackerTopology& ttopo,
                                                     const SiPixelLorentzAngle* lorentzAngle,
                                                     const SiPixelGenErrorDBObject* genErrorDBObject,
                                                     const SiPixelLorentzAngle* lorentzAngleWidth = nullptr)
    : PixelCPEGeneric(conf, mag, geom, ttopo, lorentzAngle, genErrorDBObject, lorentzAngleWidth) {
  if (theVerboseLevel > 0)
    LogDebug("PixelCPEGenericBricked") << "constructing a generic algorithm for ideal pixel detector.\n"
                                       << "CPEGenericForBricked::VerboseLevel =" << theVerboseLevel;
#ifdef EDM_ML_DEBUG
  cout << "From PixelCPEGenericForBricked::PixelCPEGenericForBricked(...)" << endl;
  cout << "(int)useErrorsFromTemplates_ = " << (int)useErrorsFromTemplates_ << endl;
  cout << "truncatePixelCharge_         = " << (int)truncatePixelCharge_ << endl;
  cout << "IrradiationBiasCorrection_   = " << (int)IrradiationBiasCorrection_ << endl;
  cout << "(int)DoCosmics_              = " << (int)DoCosmics_ << endl;
  cout << "(int)LoadTemplatesFromDB_    = " << (int)LoadTemplatesFromDB_ << endl;
#endif
}

//-----------------------------------------------------------------------------
//! Hit position in the local frame (in cm).  Unlike other CPE's, this
//! one converts everything from the measurement frame (in channel numbers)
//! into the local frame (in centimeters).
//-----------------------------------------------------------------------------
LocalPoint PixelCPEGenericForBricked::localPosition(DetParam const& theDetParam,
                                                    ClusterParam& theClusterParamBase) const {
  ClusterParamGeneric& theClusterParam = static_cast<ClusterParamGeneric&>(theClusterParamBase);

  float chargeWidthX = (theDetParam.lorentzShiftInCmX * theDetParam.widthLAFractionX);
  float chargeWidthY = (theDetParam.lorentzShiftInCmY * theDetParam.widthLAFractionY);
  float shiftX = 0.5f * theDetParam.lorentzShiftInCmX;
  float shiftY = 0.5f * theDetParam.lorentzShiftInCmY;

  //cout<<" main la width "<<chargeWidthX<<" "<<chargeWidthY<<endl;

  if (useErrorsFromTemplates_) {
    float qclus = theClusterParam.theCluster->charge();
    float locBz = theDetParam.bz;
    float locBx = theDetParam.bx;
    //cout << "PixelCPEGenericForBricked::localPosition(...) : locBz = " << locBz << endl;

    theClusterParam.pixmx = -999;     // max pixel charge for truncation of 2-D cluster
    theClusterParam.sigmay = -999.9;  // CPE Generic y-error for multi-pixel cluster
    theClusterParam.deltay = -999.9;  // CPE Generic y-bias for multi-pixel cluster
    theClusterParam.sigmax = -999.9;  // CPE Generic x-error for multi-pixel cluster
    theClusterParam.deltax = -999.9;  // CPE Generic x-bias for multi-pixel cluster
    theClusterParam.sy1 = -999.9;     // CPE Generic y-error for single single-pixel
    theClusterParam.dy1 = -999.9;     // CPE Generic y-bias for single single-pixel cluster
    theClusterParam.sy2 = -999.9;     // CPE Generic y-error for single double-pixel cluster
    theClusterParam.dy2 = -999.9;     // CPE Generic y-bias for single double-pixel cluster
    theClusterParam.sx1 = -999.9;     // CPE Generic x-error for single single-pixel cluster
    theClusterParam.dx1 = -999.9;     // CPE Generic x-bias for single single-pixel cluster
    theClusterParam.sx2 = -999.9;     // CPE Generic x-error for single double-pixel cluster
    theClusterParam.dx2 = -999.9;     // CPE Generic x-bias for single double-pixel cluster

    SiPixelGenError gtempl(thePixelGenError_);
    int gtemplID_ = theDetParam.detTemplateId;

    //int gtemplID0 = genErrorDBObject_->getGenErrorID(theDetParam.theDet->geographicalId().rawId());
    //if(gtemplID0!=gtemplID_) cout<<" different id "<< gtemplID_<<" "<<gtemplID0<<endl;

    theClusterParam.qBin_ = gtempl.qbin(gtemplID_,
                                        theClusterParam.cotalpha,
                                        theClusterParam.cotbeta,
                                        locBz,
                                        locBx,
                                        qclus,
                                        IrradiationBiasCorrection_,
                                        theClusterParam.pixmx,
                                        theClusterParam.sigmay,
                                        theClusterParam.deltay,
                                        theClusterParam.sigmax,
                                        theClusterParam.deltax,
                                        theClusterParam.sy1,
                                        theClusterParam.dy1,
                                        theClusterParam.sy2,
                                        theClusterParam.dy2,
                                        theClusterParam.sx1,
                                        theClusterParam.dx1,
                                        theClusterParam.sx2,
                                        theClusterParam.dx2);

    // now use the charge widths stored in the new generic template headers (change to the
    // incorrect sign convention of the base class)
    bool useLAWidthFromGenError = false;
    if (useLAWidthFromGenError) {
      chargeWidthX = (-micronsToCm * gtempl.lorxwidth());
      chargeWidthY = (-micronsToCm * gtempl.lorywidth());
      edm::LogInfo("PixelCPE bricked localPosition():")
          << "redefine la width (gen-error)" << chargeWidthX << " " << chargeWidthY;
    }
    edm::LogInfo("PixelCPEGeneric bricked localPosition():") << "GenError:" << gtemplID_;

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

  }  // if ( useErrorsFromTemplates_ )
  else {
    theClusterParam.qBin_ = 0;
  }

  int q_f_X;  //!< Q of the first  pixel  in X
  int q_l_X;  //!< Q of the last   pixel  in X
  int q_f_Y;  //!< Q of the first  pixel  in Y
  int q_l_Y;  //!< Q of the last   pixel  in Y
  int q_f_b;  // Q first bricked: charge of the dented row at the bootom of a cluster
  int q_l_b;  // Same but at the top of the cluster
  int lowest_is_bricked = 1;
  int highest_is_bricked = 0;

  if (theDetParam.theTopol->isBricked()) {
    collect_edge_charges_bricked(theClusterParam,
                                 q_f_X,
                                 q_l_X,
                                 q_f_Y,
                                 q_l_Y,
                                 q_f_b,
                                 q_l_b,
                                 lowest_is_bricked,
                                 highest_is_bricked,
                                 useErrorsFromTemplates_ && truncatePixelCharge_);
  } else {
    collect_edge_charges(theClusterParam, q_f_X, q_l_X, q_f_Y, q_l_Y, useErrorsFromTemplates_ && truncatePixelCharge_);
  }

  //--- Find the inner widths along X and Y in one shot.  We
  //--- compute the upper right corner of the inner pixels
  //--- (== lower left corner of upper right pixel) and
  //--- the lower left corner of the inner pixels
  //--- (== upper right corner of lower left pixel), and then
  //--- subtract these two points in the formula.

  //--- Upper Right corner of Lower Left pixel -- in measurement frame
  MeasurementPoint meas_URcorn_LLpix(theClusterParam.theCluster->minPixelRow() + 1.0,
                                     theClusterParam.theCluster->minPixelCol() + 1.0);
  //--- Lower Left corner of Upper Right pixel -- in measurement frame
  MeasurementPoint meas_LLcorn_URpix(theClusterParam.theCluster->maxPixelRow(),
                                     theClusterParam.theCluster->maxPixelCol());

  if (theDetParam.theTopol->isBricked()) {
    if (lowest_is_bricked)
      meas_URcorn_LLpix = MeasurementPoint(theClusterParam.theCluster->minPixelRow() + 1.0,
                                           theClusterParam.theCluster->minPixelCol() + 1.5);
    if (highest_is_bricked)
      meas_LLcorn_URpix =
          MeasurementPoint(theClusterParam.theCluster->maxPixelRow(), theClusterParam.theCluster->maxPixelCol() + 0.5);
  }

  //--- These two now converted into the local
  LocalPoint local_URcorn_LLpix;
  LocalPoint local_LLcorn_URpix;

  // PixelCPEGenericForBricked can be used with or without track angles
  // If PixelCPEGenericForBricked is called with track angles, use them to correct for bows/kinks:
  if (theClusterParam.with_track_angle) {
    local_URcorn_LLpix = theDetParam.theTopol->localPosition(meas_URcorn_LLpix, theClusterParam.loc_trk_pred);
    local_LLcorn_URpix = theDetParam.theTopol->localPosition(meas_LLcorn_URpix, theClusterParam.loc_trk_pred);
  } else {
    local_URcorn_LLpix = theDetParam.theTopol->localPosition(meas_URcorn_LLpix);
    local_LLcorn_URpix = theDetParam.theTopol->localPosition(meas_LLcorn_URpix);
  }

#ifdef EDM_ML_DEBUG
  if (theVerboseLevel > 0) {
    cout << "\n\t >>> theClusterParam.theCluster->x = " << theClusterParam.theCluster->x()
         << "\n\t >>> theClusterParam.theCluster->y = " << theClusterParam.theCluster->y()
         << "\n\t >>> cluster: minRow = " << theClusterParam.theCluster->minPixelRow()
         << "  minCol = " << theClusterParam.theCluster->minPixelCol()
         << "\n\t >>> cluster: maxRow = " << theClusterParam.theCluster->maxPixelRow()
         << "  maxCol = " << theClusterParam.theCluster->maxPixelCol()
         << "\n\t >>> meas: inner lower left  = " << meas_URcorn_LLpix.x() << "," << meas_URcorn_LLpix.y()
         << "\n\t >>> meas: inner upper right = " << meas_LLcorn_URpix.x() << "," << meas_LLcorn_URpix.y() << endl;
  }
#endif

  //--- &&& Note that the cuts below should not be hardcoded (like in Orca and
  //--- &&& CPEFromDetPosition/PixelCPEInitial), but rather be
  //--- &&& externally settable (but tracked) parameters.

  //--- Position, including the half lorentz shift

#ifdef EDM_ML_DEBUG
  if (theVerboseLevel > 0)
    cout << "\t >>> Generic:: processing X" << endl;
#endif

  float xPos = SiPixelUtils::generic_position_formula(
      theClusterParam.theCluster->sizeX(),
      q_f_X,
      q_l_X,
      local_URcorn_LLpix.x(),
      local_LLcorn_URpix.x(),
      chargeWidthX,  // lorentz shift in cm
      theDetParam.theThickness,
      theClusterParam.cotalpha,
      theDetParam.thePitchX,
      theDetParam.theRecTopol->isItBigPixelInX(theClusterParam.theCluster->minPixelRow()),
      theDetParam.theRecTopol->isItBigPixelInX(theClusterParam.theCluster->maxPixelRow()),
      the_eff_charge_cut_lowX,
      the_eff_charge_cut_highX,
      the_size_cutX);  // cut for eff charge width &&&

  // apply the lorentz offset correction
  xPos = xPos + shiftX;

#ifdef EDM_ML_DEBUG
  if (theVerboseLevel > 0)
    cout << "\t >>> Generic:: processing Y" << endl;
#endif

  // staggering pf the pixel cells allowed along local-Y direction only
  float yPos;
  if (theDetParam.theTopol->isBricked()) {
    yPos = SiPixelUtils::generic_position_formula_y_bricked(
        theClusterParam.theCluster->sizeY(),
        q_f_Y,
        q_l_Y,
        q_f_b,
        q_l_b,
        local_URcorn_LLpix.y(),
        local_LLcorn_URpix.y(),
        chargeWidthY,  // lorentz shift in cm
        theDetParam.theThickness,
        theClusterParam.cotbeta,
        theDetParam.thePitchY,
        theDetParam.theRecTopol->isItBigPixelInY(theClusterParam.theCluster->minPixelCol()),
        theDetParam.theRecTopol->isItBigPixelInY(theClusterParam.theCluster->maxPixelCol()),
        the_eff_charge_cut_lowY,
        the_eff_charge_cut_highY,
        the_size_cutY);  // cut for eff charge width &&&
  } else {
    yPos = SiPixelUtils::generic_position_formula(
        theClusterParam.theCluster->sizeY(),
        q_f_Y,
        q_l_Y,
        local_URcorn_LLpix.y(),
        local_LLcorn_URpix.y(),
        chargeWidthY,  // lorentz shift in cm
        theDetParam.theThickness,
        theClusterParam.cotbeta,
        theDetParam.thePitchY,
        theDetParam.theRecTopol->isItBigPixelInY(theClusterParam.theCluster->minPixelCol()),
        theDetParam.theRecTopol->isItBigPixelInY(theClusterParam.theCluster->maxPixelCol()),
        the_eff_charge_cut_lowY,
        the_eff_charge_cut_highY,
        the_size_cutY);  // cut for eff charge width &&&
  }

  // apply the lorentz offset correction
  yPos = yPos + shiftY;

  // Apply irradiation corrections
  if (IrradiationBiasCorrection_) {
    if (theClusterParam.theCluster->sizeX() == 1) {  // size=1
      // ggiurgiu@jhu.edu, 02/03/09 : for size = 1, the Lorentz shift is already accounted by the irradiation correction
      //float tmp1 =  (0.5 * theDetParam.lorentzShiftInCmX);
      //cout << "Apply correction correction_dx1 = " << theClusterParam.dx1 << " to xPos = " << xPos;
      xPos = xPos - (0.5f * theDetParam.lorentzShiftInCmX);
      // Find if pixel is double (big).
      bool bigInX = theDetParam.theRecTopol->isItBigPixelInX(theClusterParam.theCluster->maxPixelRow());
      if (!bigInX)
        xPos -= theClusterParam.dx1;
      else
        xPos -= theClusterParam.dx2;
      //cout<<" to "<<xPos<<" "<<(tmp1+theClusterParam.dx1)<<endl;
    } else {  // size>1
      //cout << "Apply correction correction_deltax = " << theClusterParam.deltax << " to xPos = " << xPos;
      xPos -= theClusterParam.deltax;
      //cout<<" to "<<xPos<<endl;
    }

    if (theClusterParam.theCluster->sizeY() == 1) {
      // ggiurgiu@jhu.edu, 02/03/09 : for size = 1, the Lorentz shift is already accounted by the irradiation correction
      yPos = yPos - (0.5f * theDetParam.lorentzShiftInCmY);

      // Find if pixel is double (big).
      bool bigInY = theDetParam.theRecTopol->isItBigPixelInY(theClusterParam.theCluster->maxPixelCol());
      if (!bigInY)
        yPos -= theClusterParam.dy1;
      else
        yPos -= theClusterParam.dy2;

    } else {
      //cout << "Apply correction correction_deltay = " << theClusterParam.deltay << " to yPos = " << yPos << endl;
      yPos -= theClusterParam.deltay;
    }

  }  // if ( IrradiationBiasCorrection_ )

  //cout<<" in PixelCPEGenericForBricked:localPosition - pos = "<<xPos<<" "<<yPos<<endl; //dk

  //--- Now put the two together
  LocalPoint pos_in_local(xPos, yPos);
  return pos_in_local;
}

void PixelCPEGenericForBricked::collect_edge_charges_bricked(ClusterParam& theClusterParamBase,  //!< input, the cluster
                                                             int& q_f_X,  //!< output, Q first  in X
                                                             int& q_l_X,  //!< output, Q last   in X
                                                             int& q_f_Y,  //!< output, Q first  in Y
                                                             int& q_l_Y,  //!< output, Q last   in Y
                                                             int& q_f_b,
                                                             int& q_l_b,               //Bricked correction
                                                             int& lowest_is_bricked,   //Bricked correction
                                                             int& highest_is_bricked,  //Bricked correction
                                                             bool truncate) {
  ClusterParamGeneric& theClusterParam = static_cast<ClusterParamGeneric&>(theClusterParamBase);

  // Initialize return variables.
  q_f_X = q_l_X = 0.0;
  q_f_Y = q_l_Y = 0.0;
  q_f_b = q_l_b = 0.0;

  // Obtain boundaries in index units
  int xmin = theClusterParam.theCluster->minPixelRow();
  int xmax = theClusterParam.theCluster->maxPixelRow();
  int ymin = theClusterParam.theCluster->minPixelCol();
  int ymax = theClusterParam.theCluster->maxPixelCol();

  //bool lowest_is_bricked = 1; //Tells you if the lowest pixel of the cluster is on a bricked row or not.
  //bool highest_is_bricked = 0;

  //Sums up the charge of the non-bricked pixels at the top of the clusters in the event that the highest pixel of the cluster is on a bricked row.
  int q_t_b = 0;
  int q_t_nb = 0;
  int q_b_b = 0;
  int q_b_nb = 0;

  //This is included in the main loop.
  // Iterate over the pixels to find out if a bricked row is lowest/highest.
  /*  int isize = theClusterParam.theCluster->size();
  for (int i = 0; i != isize; ++i) {
    auto const& pixel = theClusterParam.theCluster->pixel(i);

    // Y projection
    if (pixel.y == ymin && !(pixel.x%2) ) lowest_is_bricked = 0;
    if (pixel.y == ymax && (pixel.x%2) ) highest_is_bricked = 1;
  } */

  // Iterate over the pixels.
  int isize = theClusterParam.theCluster->size();
  for (int i = 0; i != isize; ++i) {
    auto const& pixel = theClusterParam.theCluster->pixel(i);
    // ggiurgiu@fnal.gov: add pixel charge truncation
    int pix_adc = pixel.adc;
    if (truncate)
      pix_adc = std::min(pix_adc, theClusterParam.pixmx);
    //
    // X projection
    if (pixel.x == xmin)
      q_f_X += pix_adc;
    if (pixel.x == xmax)
      q_l_X += pix_adc;
    //
    // Y projection
    if (pixel.y == ymin) {
      q_f_Y += pix_adc;
      if (pixel.x % 2)
        q_b_nb += pix_adc;
      else
        lowest_is_bricked = 0;
    }
    if (pixel.y == ymin + 1 && !(pixel.x % 2))
      q_b_b += pix_adc;
    if (pixel.y == ymax) {
      q_l_Y += pix_adc;
      if (!(pixel.x % 2))
        q_t_b += pix_adc;
      else
        highest_is_bricked = 1;
    }
    if (pixel.y == ymax - 1 && (pixel.x % 2))
      q_t_nb += pix_adc;
  }

  edm::LogInfo("PixelCPE: collect_edge_charges_bricked: l/h") << lowest_is_bricked << "it" << highest_is_bricked;

  if (lowest_is_bricked)
    q_f_b = q_b_b;
  else
    q_f_b = q_b_nb;

  if (highest_is_bricked)
    q_l_b = -q_t_b;
  else
    q_l_b = -q_t_nb;

  //Need to add the edge pixels that were missed:
  for (int i = 0; i != isize; ++i) {
    auto const& pixel = theClusterParam.theCluster->pixel(i);
    int pix_adc = pixel.adc;
    if (truncate)
      pix_adc = std::min(pix_adc, theClusterParam.pixmx);

    if (lowest_is_bricked && pixel.y == ymin + 1 && !(pixel.x % 2))
      q_f_Y += pix_adc;

    if (!highest_is_bricked && pixel.y == ymax - 1 && (pixel.x % 2))
      q_l_Y += pix_adc;

    edm::LogInfo("PixelCPE: collect_edge_charges_bricked: Q") << q_l_b << q_f_b << q_f_X << q_l_X << q_f_Y << q_l_Y;

    return;
  }
}
