#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGeneric.h"

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
PixelCPEGeneric::PixelCPEGeneric(edm::ParameterSet const& conf,
                                 const MagneticField* mag,
                                 const TrackerGeometry& geom,
                                 const TrackerTopology& ttopo,
                                 const SiPixelLorentzAngle* lorentzAngle,
                                 const SiPixelGenErrorDBObject* genErrorDBObject,
                                 const SiPixelLorentzAngle* lorentzAngleWidth = nullptr)
    : PixelCPEGenericBase(conf, mag, geom, ttopo, lorentzAngle, genErrorDBObject, lorentzAngleWidth) {
  if (theVerboseLevel > 0)
    LogDebug("PixelCPEGeneric") << " constructing a generic algorithm for ideal pixel detector.\n"
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

  NoTemplateErrorsWhenNoTrkAngles_ = conf.getParameter<bool>("NoTemplateErrorsWhenNoTrkAngles");
  IrradiationBiasCorrection_ = conf.getParameter<bool>("IrradiationBiasCorrection");
  DoCosmics_ = conf.getParameter<bool>("DoCosmics");

  isPhase2_ = conf.getParameter<bool>("isPhase2");

  // For cosmics force the use of simple errors
  if ((DoCosmics_))
    useErrorsFromTemplates_ = false;

  if (!useErrorsFromTemplates_ && (truncatePixelCharge_ || IrradiationBiasCorrection_ || LoadTemplatesFromDB_)) {
    throw cms::Exception("PixelCPEGeneric::PixelCPEGeneric: ")
        << "\nERROR: useErrorsFromTemplates_ is set to False in PixelCPEGeneric_cfi.py. "
        << " In this case it does not make sense to set any of the following to True: "
        << " truncatePixelCharge_, IrradiationBiasCorrection_, DoCosmics_, LoadTemplatesFromDB_ !!!"
        << "\n\n";
  }

  // Use errors from templates or from GenError
  if (useErrorsFromTemplates_) {
    if (LoadTemplatesFromDB_) {  // From DB
      if (!SiPixelGenError::pushfile(*genErrorDBObject_, thePixelGenError_))
        throw cms::Exception("InvalidCalibrationLoaded")
            << "ERROR: GenErrors not filled correctly. Check the sqlite file. Using SiPixelTemplateDBObject version "
            << (*genErrorDBObject_).version();
      LogDebug("PixelCPEGeneric") << "Loaded genErrorDBObject v" << (*genErrorDBObject_).version();
    } else {  // From file
      if (!SiPixelGenError::pushfile(-999, thePixelGenError_))
        throw cms::Exception("InvalidCalibrationLoaded")
            << "ERROR: GenErrors not loaded correctly from text file. Reconstruction will fail.";
    }  // if load from DB

  } else {
#ifdef EDM_ML_DEBUG
    cout << " Use simple parametrised errors " << endl;
#endif
  }  // if ( useErrorsFromTemplates_ )

#ifdef EDM_ML_DEBUG
  cout << "From PixelCPEGeneric::PixelCPEGeneric(...)" << endl;
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
LocalPoint PixelCPEGeneric::localPosition(DetParam const& theDetParam, ClusterParam& theClusterParamBase) const {
  ClusterParamGeneric& theClusterParam = static_cast<ClusterParamGeneric&>(theClusterParamBase);

  //cout<<" in PixelCPEGeneric:localPosition - "<<endl; //dk

  float chargeWidthX = (theDetParam.lorentzShiftInCmX * theDetParam.widthLAFractionX);
  float chargeWidthY = (theDetParam.lorentzShiftInCmY * theDetParam.widthLAFractionY);
  float shiftX = 0.5f * theDetParam.lorentzShiftInCmX;
  float shiftY = 0.5f * theDetParam.lorentzShiftInCmY;

  //cout<<" main la width "<<chargeWidthX<<" "<<chargeWidthY<<endl;

  if (useErrorsFromTemplates_) {
    float qclus = theClusterParam.theCluster->charge();
    float locBz = theDetParam.bz;
    float locBx = theDetParam.bx;
    //cout << "PixelCPEGeneric::localPosition(...) : locBz = " << locBz << endl;

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
      LogDebug("PixelCPE localPosition():") << "redefine la width (gen-error)" << chargeWidthX << chargeWidthY;
    }
    LogDebug("PixelCPE localPosition():") << "GenError:" << gtemplID_;

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
  collect_edge_charges(theClusterParam, q_f_X, q_l_X, q_f_Y, q_l_Y, useErrorsFromTemplates_ && truncatePixelCharge_);

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

  //--- These two now converted into the local
  LocalPoint local_URcorn_LLpix;
  LocalPoint local_LLcorn_URpix;

  // PixelCPEGeneric can be used with or without track angles
  // If PixelCPEGeneric is called with track angles, use them to correct for bows/kinks:
  if (theClusterParam.with_track_angle) {
    local_URcorn_LLpix = theDetParam.theTopol->localPosition(meas_URcorn_LLpix, theClusterParam.loc_trk_pred);
    local_LLcorn_URpix = theDetParam.theTopol->localPosition(meas_LLcorn_URpix, theClusterParam.loc_trk_pred);
  } else {
    local_URcorn_LLpix = theDetParam.theTopol->localPosition(meas_URcorn_LLpix);
    local_LLcorn_URpix = theDetParam.theTopol->localPosition(meas_LLcorn_URpix);
  }

#ifdef EDM_ML_DEBUG
  if (theVerboseLevel > 20) {
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
  if (theVerboseLevel > 20)
    cout << "\t >>> Generic:: processing X" << endl;
#endif

  float xPos = siPixelUtils::generic_position_formula(
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
  if (theVerboseLevel > 20)
    cout << "\t >>> Generic:: processing Y" << endl;
#endif

  float yPos = siPixelUtils::generic_position_formula(
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

  //cout<<" in PixelCPEGeneric:localPosition - pos = "<<xPos<<" "<<yPos<<endl; //dk

  //--- Now put the two together
  LocalPoint pos_in_local(xPos, yPos);
  return pos_in_local;
}

//==============  INFLATED ERROR AND ERRORS FROM DB BELOW  ================

//-------------------------------------------------------------------------
//  Hit error in the local frame
//-------------------------------------------------------------------------
LocalError PixelCPEGeneric::localError(DetParam const& theDetParam, ClusterParam& theClusterParamBase) const {
  ClusterParamGeneric& theClusterParam = static_cast<ClusterParamGeneric&>(theClusterParamBase);

  // local variables
  float xerr, yerr;
  bool edgex, edgey, bigInX, bigInY;
  int maxPixelCol, maxPixelRow, minPixelCol, minPixelRow;
  uint sizex, sizey;

  initializeLocalErrorVariables(xerr,
                                yerr,
                                edgex,
                                edgey,
                                bigInX,
                                bigInY,
                                maxPixelCol,
                                maxPixelRow,
                                minPixelCol,
                                minPixelRow,
                                sizex,
                                sizey,
                                theDetParam,
                                theClusterParam);

  bool useTempErrors =
      useErrorsFromTemplates_ && (!NoTemplateErrorsWhenNoTrkAngles_ || theClusterParam.with_track_angle);

  if (int(sizex) != (maxPixelRow - minPixelRow + 1))
    LogDebug("PixelCPEGeneric") << " wrong x";
  if (int(sizey) != (maxPixelCol - minPixelCol + 1))
    LogDebug("PixelCPEGeneric") << " wrong y";

  LogDebug("PixelCPEGeneric") << " edge clus " << xerr << " " << yerr;  //dk
  if (bigInX || bigInY)
    LogDebug("PixelCPEGeneric") << " big " << bigInX << " " << bigInY;
  if (edgex || edgey)
    LogDebug("PixelCPEGeneric") << " edge " << edgex << " " << edgey;
  LogDebug("PixelCPEGeneric") << " before if " << useErrorsFromTemplates_ << " " << theClusterParam.qBin_;
  if (theClusterParam.qBin_ == 0)
    LogDebug("PixelCPEGeneric") << " qbin 0! " << edgex << " " << edgey << " " << bigInX << " " << bigInY << " "
                                << sizex << " " << sizey;

  // from PixelCPEGenericBase
  setXYErrors(xerr, yerr, edgex, edgey, sizex, sizey, bigInX, bigInY, useTempErrors, theDetParam, theClusterParam);

  if (!useTempErrors) {
    LogDebug("PixelCPEGeneric") << "Track angles are not known.\n"
                                << "Default angle estimation which assumes track from PV (0,0,0) does not work.";
  }

  if (!useTempErrors && inflate_errors) {
    int n_bigx = 0;
    int n_bigy = 0;

    for (int irow = 0; irow < 7; ++irow) {
      if (theDetParam.theRecTopol->isItBigPixelInX(irow + minPixelRow))
        ++n_bigx;
    }

    for (int icol = 0; icol < 21; ++icol) {
      if (theDetParam.theRecTopol->isItBigPixelInY(icol + minPixelCol))
        ++n_bigy;
    }

    xerr = (float)(sizex + n_bigx) * theDetParam.thePitchX / std::sqrt(12.0f);
    yerr = (float)(sizey + n_bigy) * theDetParam.thePitchY / std::sqrt(12.0f);
  }

#ifdef EDM_ML_DEBUG
  if (!(xerr > 0.0))
    throw cms::Exception("PixelCPEGeneric::localError") << "\nERROR: Negative pixel error xerr = " << xerr << "\n\n";

  if (!(yerr > 0.0))
    throw cms::Exception("PixelCPEGeneric::localError") << "\nERROR: Negative pixel error yerr = " << yerr << "\n\n";
#endif

  LogDebug("PixelCPEGeneric") << " errors  " << xerr << " " << yerr;  //dk
  if (theClusterParam.qBin_ == 0)
    LogDebug("PixelCPEGeneric") << " qbin 0 " << xerr << " " << yerr;

  auto xerr_sq = xerr * xerr;
  auto yerr_sq = yerr * yerr;

  return LocalError(xerr_sq, 0, yerr_sq);
}

void PixelCPEGeneric::fillPSetDescription(edm::ParameterSetDescription& desc) {
  PixelCPEGenericBase::fillPSetDescription(desc);
  desc.add<double>("eff_charge_cut_highX", 1.0);
  desc.add<double>("eff_charge_cut_highY", 1.0);
  desc.add<double>("eff_charge_cut_lowX", 0.0);
  desc.add<double>("eff_charge_cut_lowY", 0.0);
  desc.add<double>("size_cutX", 3.0);
  desc.add<double>("size_cutY", 3.0);
  desc.add<double>("EdgeClusterErrorX", 50.0);
  desc.add<double>("EdgeClusterErrorY", 85.0);
  desc.add<bool>("inflate_errors", false);
  desc.add<bool>("inflate_all_errors_no_trk_angle", false);
  desc.add<bool>("NoTemplateErrorsWhenNoTrkAngles", false);
  desc.add<bool>("UseErrorsFromTemplates", true);
  desc.add<bool>("TruncatePixelCharge", true);
  desc.add<bool>("IrradiationBiasCorrection", false);
  desc.add<bool>("DoCosmics", false);
  desc.add<bool>("isPhase2", false);
  desc.add<bool>("SmallPitch", false);
}
