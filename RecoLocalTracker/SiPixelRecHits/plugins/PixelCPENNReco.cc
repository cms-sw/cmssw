// Include our own header first
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPETemplateReco.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/PixelCPENNReco.h"

// Geometry services
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

//#define DEBUG

// MessageLogger

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"

// The template header files
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"

// Commented for now (3/10/17) until we figure out how to resuscitate 2D template splitter
/// #include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateSplit.h"

#include <vector>
#include "boost/multi_array.hpp"

#include <iostream>

using namespace std;

namespace {
  constexpr float micronsToCm = 1.0e-4;
  constexpr float output_scale =
      50.;  //Defined by NN value = CMSSW value * output_scale. To read the NN outputs back to CMSSW, do CMSSW value = NN value / output_scale
  constexpr float CHARGENORM = 25000.;
}  // namespace

//-----------------------------------------------------------------------------
//  Constructor.
//
//-----------------------------------------------------------------------------
PixelCPENNReco::PixelCPENNReco(edm::ParameterSet const& conf,
                               const MagneticField* mag,
                               const TrackerGeometry& geom,
                               const TrackerTopology& ttopo,
                               const SiPixelLorentzAngle* lorentzAngle,
                               const SiPixelGenErrorDBObject* genErrorDBObject,
                               std::vector<const tensorflow::Session*> session_x_vec_,
                               std::vector<const tensorflow::Session*> session_y_vec_)
    : PixelCPEGenericBase(conf, mag, geom, ttopo, lorentzAngle, genErrorDBObject, nullptr) {
  if (session_x_vec_.size() != 7 || session_y_vec_.size() != 7) {
    throw cms::Exception("Configuration") << "PixelCPENNReco requires seven X and seven Y sessions";
  }
  session_x_vec = session_x_vec_;
  session_y_vec = session_y_vec_;
  inputTensorName_x = conf.getParameter<std::string>("inputTensorName_x");
  anglesTensorName_x = conf.getParameter<std::string>("anglesTensorName_x");
  cchargeTensorName_x = conf.getParameter<std::string>("cchargeTensorName_x");
  outputTensorName_x = conf.getParameter<std::string>("outputTensorName_x");

  inputTensorName_y = conf.getParameter<std::string>("inputTensorName_y");
  anglesTensorName_y = conf.getParameter<std::string>("anglesTensorName_y");
  cchargeTensorName_y = conf.getParameter<std::string>("cchargeTensorName_y");
  outputTensorName_y = conf.getParameter<std::string>("outputTensorName_y");


  if (!SiPixelGenError::pushfile(*genErrorDBObject_, thePixelGenError_))
    throw cms::Exception("InvalidCalibrationLoaded")
        << "ERROR: GenErrors not filled correctly. Check the sqlite file. Using SiPixelTemplateDBObject version "
        << (*genErrorDBObject_).version();
}

//-----------------------------------------------------------------------------
//  Clean up.
//-----------------------------------------------------------------------------
PixelCPENNReco::~PixelCPENNReco() {}
std::unique_ptr<PixelCPEBase::ClusterParam> PixelCPENNReco::createClusterParam(const SiPixelCluster& cl) const {
  return std::make_unique<ClusterParamNN>(cl);
}

//------------------------------------------------------------------
//  Public methods mandated by the base class.
//------------------------------------------------------------------

//------------------------------------------------------------------
//  The main call to the template code.
//------------------------------------------------------------------

int PixelCPENNReco::PixelPreprocess(const SiPixelCluster& cluster,
                                    const PixelTopology& topol,
                                    float (&Cluster_raw)[TXSIZE][TYSIZE],
                                    float (&Cluster_xRaw)[TXSIZE],
                                    float (&Cluster_yRaw)[TYSIZE],
                                    float (&Cluster)[TXSIZE][TYSIZE],
                                    float (&Cluster_x)[TXSIZE],
                                    float (&Cluster_y)[TYSIZE],
                                    float& Cluster_charge,
                                    int& Cluster_size,
                                    int& Cluster_sizeX,
                                    int& Cluster_sizeY,
                                    float& ClusterCenter_x,
                                    float& ClusterCenter_y,
                                    int& Row_offset,
                                    int& Col_offset) const {
  //-------------------------------------------------------
  //Cluster preprocessing. This step should align with training cluster preprocessing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //-------------------------------------------------------
  float clusbuf_temp[TXSIZE][TYSIZE];
  for (int i = 0; i < TXSIZE; ++i) {
    for (int j = 0; j < TYSIZE; ++j) {
      clusbuf_temp[i][j] = 0.;
    }
  }
  Row_offset = cluster.minPixelRow();
  Col_offset = cluster.minPixelCol();
  int row_offset = Row_offset;
  int col_offset = Col_offset;
  int mrow = 0, mcol = 0;  //maximum row and column of the cluster
  for (int i = 0; i != cluster.size(); ++i) {
    auto pix = cluster.pixel(i);
    int irow = int(pix.x);
    int icol = int(pix.y);
    mrow = std::max(mrow, irow);
    mcol = std::max(mcol, icol);
  }
  mrow -= row_offset;  // convert to local cluster coordinates, offset is the min row/col of the cluster
  mrow += 1;
  mrow = std::min(mrow, TXSIZE);  // limit cluster size to the dimensions of the input matrix for the NN / template reco
  mcol -= col_offset;
  mcol += 1;
  mcol = std::min(mcol, TYSIZE);
  assert(mrow > 0);
  assert(mcol > 0);

  int n_double_x = 0, n_double_y = 0;
  int clustersize = 0;  // number of pixels in the cluster, wide pixels still count as 1 here!
  int double_pixel_buffer_size = 5;
  int double_row[double_pixel_buffer_size], double_col[double_pixel_buffer_size];
  for (int i = 0; i < double_pixel_buffer_size; i++) {
    double_row[i] = -1;
    double_col[i] = -1;
  }

  int irow_sum = 0, icol_sum = 0;
  for (int i = 0; i < cluster.size(); ++i) {
    auto pix = cluster.pixel(i);
    int irow = int(pix.x) - row_offset;
    int icol = int(pix.y) - col_offset;
    int absRow = int(pix.x);
    int absCol = int(pix.y);
    if ((irow >= mrow) || (icol >= mcol))
      continue;
    const bool bigInX = topol.isItBigPixelInX(absRow);
    const bool bigInY = topol.isItBigPixelInY(absCol);
    if (bigInX) {    // Phase-1 specific wide-pixel rows
      int flag = 0;  // check if this row is already counter as a double pixel row
      for (int j = 0; j < n_double_x; j++) {
        if (irow == double_row[j]) {
          flag = 1;
          break;
        }
      }
      if (flag != 1) {
        double_row[n_double_x] = irow;
        n_double_x++;
      }
    }
    if (bigInY) {    // Phase-1 specific wide-pixel columns
      int flag = 0;  // check if this column is already counter as a double pixel column
      for (int j = 0; j < n_double_y; j++) {
        if (icol == double_col[j]) {
          flag = 1;
          break;
        }
      }
      if (flag != 1) {
        double_col[n_double_y] = icol;
        n_double_y++;
      }
    }
    irow_sum += irow;
    icol_sum += icol;
    clustersize++;
  }

  if (clustersize == 0) {
    LogDebug("PixelCPENNReco") << "EMPTY CLUSTER\n";
    return 1;
  }
  if (n_double_x > 2 or n_double_y > 2) {
    LogDebug("PixelCPENNReco") << "MORE THAN 2 DOUBLE ROWS OR COLS\n";
    return 1;
  }

  Cluster_size = cluster.size();  // this is the total number of pixels in the cluster, wide pixels still count as 1
  Cluster_sizeX =
      cluster.sizeX() +
      n_double_x;  // if there is a double pixel in x/y, the effective cluster size in x/y is increased by 1, we will unpack wide pixels later and fill the input matrix accordingly
  Cluster_sizeY = cluster.sizeY() + n_double_y;
  int mid_x =
      round(float(irow_sum) / float(clustersize));  //pixel that will be flagged as the center in the input matrix
  int mid_y = round(float(icol_sum) / float(clustersize));

  float pitch_center_x = 0.5f;
  float pitch_center_y = 0.5f;
  // if the center pixel is wide, it gets split into two, so the center of the selected mid-pix will actually be at the quarter of the origina, wide pixel's pitch
  if (topol.isItBigPixelInX(mid_x + row_offset))
    pitch_center_x = 0.25f;
  if (topol.isItBigPixelInY(mid_y + col_offset))
    pitch_center_y = 0.25f;
  MeasurementPoint meas_center_pix(row_offset + mid_x + pitch_center_x,
                                   col_offset + mid_y + pitch_center_y);  // lower-left corner
  LocalPoint local_center_pix = topol.localPosition(meas_center_pix);
  ClusterCenter_x = local_center_pix.x();
  ClusterCenter_y = local_center_pix.y();

  int n_wide_before_mid_x =
      0;  //number of wide pixels in x/y before the cluster center, compensates the center shift due to expansion of wide rows/columns before the selected center
  int n_wide_before_mid_y = 0;
  for (int i = 0; i < n_double_x; ++i) {
    if (double_row[i] < mid_x) {
      ++n_wide_before_mid_x;
    }
  }
  for (int i = 0; i < n_double_y; ++i) {
    if (double_col[i] < mid_y) {
      ++n_wide_before_mid_y;
    }
  }
  int offset_x =
      TXSIZE / 2 - mid_x -
      n_wide_before_mid_x;  // compensate expansion shift from wide rows before the selected center, ensures that center pixel will end up at (TXSIZE/2, TYSIZE/2) in the input matrix after wide pixel expansion
  int offset_y = TYSIZE / 2 - mid_y - n_wide_before_mid_y;

  if (Cluster_sizeX > TXSIZE or Cluster_sizeY > TYSIZE) {
    LogDebug("PixelCPENNReco") << "SIZE LAERGE THAN EXPECTED\n";
    return 1;
  };
  if (offset_x + Cluster_sizeX > TXSIZE or offset_y + Cluster_sizeY > TYSIZE) {
    LogDebug("PixelCPENNReco") << "SIZE LAERGE THAN EXPECTED\n";
    return 1;
  }
  if (mrow + offset_x > TXSIZE or mcol + offset_y > TYSIZE or offset_x < 0 or offset_y < 0) {
    LogDebug("PixelCPENNReco") << "SIZE LAERGE THAN EXPECTED\n";
    return 1;
  }

  for (int i = 0; i < cluster.size(); ++i) {
    auto pix = cluster.pixel(i);
    int irow = int(pix.x) - row_offset +
               offset_x;  // place the cluster center in the middle of the input matrix (TXSIZE/2, TYSIZE/2)
    int icol = int(pix.y) - col_offset + offset_y;

    if ((irow >= mrow + offset_x) || (icol >= mcol + offset_y)) {
      LogDebug("PixelCPENNReco") << "irow or icol exceeded, SKIPPING.\n";
      continue;
    }
    clusbuf_temp[irow][icol] = float(pix.adc) / CHARGENORM;  //pix.adc is actually in units of electrons
    Cluster_charge += float(pix.adc) / CHARGENORM;
  }

  int double_row_centered[double_pixel_buffer_size];
  int double_col_centered[double_pixel_buffer_size];
  for (int i = 0; i < double_pixel_buffer_size; ++i) {
    double_row_centered[i] = double_row[i] + offset_x;
    double_col_centered[i] = double_col[i] + offset_y;
  }

  //Expand double width rows
  int k = 0, m = 0;
  for (int i = 0; i < TXSIZE; i++) {
    if (m < n_double_x && i == double_row_centered[m]) {
      for (int j = 0; j < TYSIZE; j++) {
        Cluster_raw[i][j] = clusbuf_temp[k][j] / 2.;
        Cluster_raw[i + 1][j] = clusbuf_temp[k][j] / 2.;
      }
      i++;
      if (m == 0 && n_double_x == 2) {
        double_row_centered
            [1]++;  // If two rows are wide, they will be next to each other, so the second wide row will be right after the first one and needs to be shifted by 1 after the first one is expanded
        m++;
      } else if (m > 0) {
        m++;
      }
    } else {
      for (int j = 0; j < TYSIZE; j++) {
        Cluster_raw[i][j] = clusbuf_temp[k][j];
      }
    }
    k++;
  }
  k = 0;
  m = 0;

  // Set clusbuf to the original cluster with expanded rows
  for (int i = 0; i < TXSIZE; i++) {
    for (int j = 0; j < TYSIZE; j++) {
      clusbuf_temp[i][j] = Cluster_raw[i][j];
      Cluster_raw[i][j] = 0.;
    }
  }

  // Expand double width columns
  for (int j = 0; j < TYSIZE; j++) {
    if (m < n_double_y && j == double_col_centered[m]) {
      for (int i = 0; i < TXSIZE; i++) {
        Cluster_raw[i][j] = clusbuf_temp[i][k] / 2.;
        Cluster_raw[i][j + 1] = clusbuf_temp[i][k] / 2.;
      }
      j++;
      if (m == 0 && n_double_y == 2) {
        double_col_centered[1]++;
        m++;
      } else if (m > 0) {
        m++;
      }
    } else {
      for (int i = 0; i < TXSIZE; i++) {
        Cluster_raw[i][j] = clusbuf_temp[i][k];
      }
    }
    k++;
  }
  //compute the 1d projection & compute cluster max
  float cluster_max_x = 0., cluster_max_y = 0., cluster_max_2d = 0.;
  for (int i = 0; i < TXSIZE; i++) {
    for (int j = 0; j < TYSIZE; j++) {
      Cluster_xRaw[i] += Cluster_raw[i][j];
      Cluster_yRaw[j] += Cluster_raw[i][j];
      if (Cluster_raw[i][j] > cluster_max_2d)
        cluster_max_2d = Cluster_raw[i][j];
    }
    if (Cluster_xRaw[i] > cluster_max_x)
      cluster_max_x = Cluster_xRaw[i];
  }
  for (int j = 0; j < TYSIZE; j++) {
    if (Cluster_yRaw[j] > cluster_max_y)
      cluster_max_y = Cluster_yRaw[j];
  }

  assert(cluster_max_x > 0);
  assert(cluster_max_y > 0);
  assert(cluster_max_2d > 0);
  //normalize 2d inputs
  for (int i = 0; i < TXSIZE; i++) {
    for (int j = 0; j < TYSIZE; j++) {
      Cluster[i][j] = Cluster_raw[i][j] / cluster_max_2d;
    }
    Cluster_x[i] = Cluster_xRaw[i] / cluster_max_x;
  }
  for (int j = 0; j < TYSIZE; j++) {
    Cluster_y[j] = Cluster_yRaw[j] / cluster_max_y;
  }
  //-------------------------------------------------------
  //End if cluster preprocessing
  //-------------------------------------------------------

  return 0;
}

LocalPoint PixelCPENNReco::localPosition(DetParam const& theDetParam, ClusterParam& theClusterParamBase) const {
  ClusterParamNN& theClusterParam = static_cast<ClusterParamNN&>(theClusterParamBase);
  //Placeholder
  theClusterParam.qBin_ = 0;
  theClusterParam.hasFilledProb_ = false;

  theClusterParam.ierr = 0;

  if (!GeomDetEnumerators::isTrackerPixel(theDetParam.thePart))
    throw cms::Exception("PixelCPENNReco::localPosition :") << "A non-pixel detector type in here?";
  int layer, ladder, module;
  const bool fpix = GeomDetEnumerators::isEndcap(theDetParam.thePart);

  if (fpix) {
    theClusterParam.ierr = 12345;
  }

  layer = ttopo_.pxbLayer(theDetParam.theDet->geographicalId().rawId());
  ladder = ttopo_.pxbLadder(theDetParam.theDet->geographicalId().rawId());
  module = ttopo_.pxbModule(theDetParam.theDet->geographicalId().rawId());
  //outer ladders = unflipped = odd nos

  const tensorflow::Session* session_x;
  const tensorflow::Session* session_y;
  if (layer == 1 and ladder % 2 != 0) {
    session_x = session_x_vec.at(0);
    session_y = session_y_vec.at(0);
  } else if (layer == 1 and ladder % 2 == 0) {
    session_x = session_x_vec.at(1);
    session_y = session_y_vec.at(1);
  } else if (layer == 2) {
    session_x = session_x_vec.at(2);
    session_y = session_y_vec.at(2);

  }  // using L2new model for all of L2
  else if (layer == 3 and module <= 4) {
    session_x = session_x_vec.at(3);
    session_y = session_y_vec.at(3);
  } else if (layer == 3 and module > 4) {
    session_x = session_x_vec.at(4);
    session_y = session_y_vec.at(4);
  } else if (layer == 4 and module <= 4) {
    session_x = session_x_vec.at(5);
    session_y = session_y_vec.at(5);
  } else  //if (layer == 4 and module > 4)
  {
    session_x = session_x_vec.at(6);
    session_y = session_y_vec.at(6);
  }

  // Preparing to retrieve ADC counts from the SiPixeltheClusterParam.theCluster->  In the cluster,
  // we have the following:
  //   int minPixelRow(); // Minimum pixel index in the x direction (low edge).
  //   int maxPixelRow(); // Maximum pixel index in the x direction (top edge).
  //   int minPixelCol(); // Minimum pixel index in the y direction (left edge).
  //   int maxPixelCol(); // Maximum pixel index in the y direction (right edge).
  // So the pixels from minPixelRow() will go into clust_array_2d[0][*],
  // and the pixels from minPixelCol() will go into clust_array_2d[*][0].

  // Store the coordinates of the center of the (0,0) pixel of the array that
  // gets passed to PixelTempReco1D
  // Will add these values to the output of  PixelTempReco1D
  // Store these offsets (to be added later) in a LocalPoint after tranforming
  // them from measurement units (pixel units) to local coordinates (cm)
  //
  //

  // In case of template reco failure, these are the lorentz drift corrections
  // to be applied
  float lorentzshiftX = 0.5f * theDetParam.lorentzShiftInCmX;
  float lorentzshiftY = 0.5f * theDetParam.lorentzShiftInCmY;

  // Not all information is needed during inferance, but defined here anyway to align with training cluster preposcessing function
  float Cluster_raw[TXSIZE][TYSIZE];
  float Cluster_xRaw[TXSIZE];
  float Cluster_yRaw[TYSIZE];
  float Cluster[TXSIZE][TYSIZE];  // Normalized so that the max pixel charge in the cluster is 1
  float Cluster_x[TXSIZE];
  float Cluster_y[TYSIZE];
  for (int j = 0; j < TYSIZE; j++) {
    Cluster_yRaw[j] = 0.f;
    Cluster_y[j] = 0.f;
  }
  for (int i = 0; i < TXSIZE; i++) {
    Cluster_xRaw[i] = 0.f;
    Cluster_x[i] = 0.f;
    for (int j = 0; j < TYSIZE; j++) {
      Cluster_raw[i][j] = 0.f;
      Cluster[i][j] = 0.f;
    }
  }

  float ClusterCenter_x = std::numeric_limits<float>::max();
  float ClusterCenter_y = std::numeric_limits<float>::max();
  int Row_offset = std::numeric_limits<int>::max();
  int Col_offset = std::numeric_limits<int>::max();
  int Cluster_size = std::numeric_limits<int>::max();
  int Cluster_sizeX = std::numeric_limits<int>::max();
  int Cluster_sizeY = std::numeric_limits<int>::max();
  float Cluster_charge = 0.f;
  int status = PixelPreprocess(*theClusterParam.theCluster,
                               *theDetParam.theTopol,
                               Cluster_raw,
                               Cluster_xRaw,
                               Cluster_yRaw,
                               Cluster,
                               Cluster_x,
                               Cluster_y,
                               Cluster_charge,
                               Cluster_size,
                               Cluster_sizeX,
                               Cluster_sizeY,
                               ClusterCenter_x,
                               ClusterCenter_y,
                               Row_offset,
                               Col_offset);
  if (status != 0) {
    theClusterParam.ierr = 12345;
  }
  //if (status != 0) continue;

  //========================================================================================
  if (theClusterParam.ierr != 12345) {
    // define a tensor and fill it with cluster projection
    tensorflow::Tensor cluster_flat_x(tensorflow::DT_FLOAT, {1, TXSIZE, 1});
    tensorflow::Tensor cluster_flat_y(tensorflow::DT_FLOAT, {1, TYSIZE, 1});
    // angles
    tensorflow::Tensor angles(tensorflow::DT_FLOAT, {1, 2});
    tensorflow::Tensor ccharge(tensorflow::DT_FLOAT, {1, 1});

    angles.tensor<float, 2>()(0, 0) = theClusterParam.cotalpha;
    angles.tensor<float, 2>()(0, 1) = theClusterParam.cotbeta;
    //ccharge.tensor<float,2>()(0, 0) = pixmax;
    ccharge.tensor<float, 2>()(0, 0) = Cluster_charge;

    for (int i = 0; i < TXSIZE; i++)
      cluster_flat_x.tensor<float, 3>()(0, i, 0) = Cluster_x[i];
    for (int j = 0; j < TYSIZE; j++)
      cluster_flat_y.tensor<float, 3>()(0, j, 0) = Cluster_y[j];

    //  Determine current time

    std::vector<tensorflow::Tensor> output_x, output_y;

    tensorflow::run(const_cast<tensorflow::Session*>(session_x),
                    {{inputTensorName_x, cluster_flat_x}, {cchargeTensorName_x, ccharge}, {anglesTensorName_x, angles}},
                    {outputTensorName_x},
                    &output_x);
    tensorflow::run(const_cast<tensorflow::Session*>(session_y),
                    {{inputTensorName_y, cluster_flat_y}, {cchargeTensorName_y, ccharge}, {anglesTensorName_y, angles}},
                    {outputTensorName_y},
                    &output_y);

    const float outputX = output_x[0].matrix<float>()(0, 0);
    const float outputSigmaX = output_x[0].matrix<float>()(0, 1);
    const float outputY = output_y[0].matrix<float>()(0, 0);
    const float outputSigmaY = output_y[0].matrix<float>()(0, 1);
    const float nnOffsetX = outputX / output_scale;
    const float nnOffsetY = outputY / output_scale;
    const float nnSigmaX = outputSigmaX / output_scale;
    const float nnSigmaY = outputSigmaY / output_scale;

    constexpr float maxPositionX = 1300.f * micronsToCm;
    constexpr float maxPositionY = 3150.f * micronsToCm;
    constexpr float maxSigmaX = 650.f * micronsToCm;
    constexpr float maxSigmaY = 1575.f * micronsToCm;

    if (!std::isfinite(nnOffsetX) || !std::isfinite(nnOffsetY) || std::abs(nnOffsetX) >= maxPositionX ||
        std::abs(nnOffsetY) >= maxPositionY) {
      theClusterParam.ierr = 12345;
    } else if (nnSigmaX <= 0.f || nnSigmaY <= 0.f || !std::isfinite(nnSigmaX) || !std::isfinite(nnSigmaY) ||
               std::abs(nnSigmaX) >= maxSigmaX || std::abs(nnSigmaY) >= maxSigmaY) {
      theClusterParam.ierr = 12345;
    } else
      theClusterParam.ierr = 0;

    theClusterParam.NNXrec_ = nnOffsetX + ClusterCenter_x;
    theClusterParam.NNSigmaX_ = nnSigmaX;
    theClusterParam.NNYrec_ = nnOffsetY + ClusterCenter_y;
    theClusterParam.NNSigmaY_ = nnSigmaY;
  }

  // Check exit status
  if (theClusterParam.ierr != 0) {
    LogDebug("PixelCPENNReco::localPosition") << "reconstruction failed with error " << theClusterParam.ierr << "\n";
    // Template reco has failed, compute position estimates based on cluster center of gravity + Lorentz drift
    // Future improvement would be to call generic reco instead

    // ggiurgiu@jhu.edu, 21/09/2010 : trk angles needed to correct for bows/kinks
    if (theClusterParam.with_track_angle) {
      theClusterParam.NNXrec_ =
          theDetParam.theTopol->localX(theClusterParam.theCluster->x(), theClusterParam.loc_trk_pred) + lorentzshiftX;
      theClusterParam.NNYrec_ =
          theDetParam.theTopol->localY(theClusterParam.theCluster->y(), theClusterParam.loc_trk_pred) + lorentzshiftY;
    } else {
      LogDebug("PixelCPENNReco") << "@SUB = PixelCPENNReco::localPosition"
                                 << "Should never be here. PixelCPENNReco should always be called "
                                    "with track angles. This is a bad error !!! ";

      theClusterParam.NNXrec_ = theDetParam.theTopol->localX(theClusterParam.theCluster->x()) + lorentzshiftX;
      theClusterParam.NNYrec_ = theDetParam.theTopol->localY(theClusterParam.theCluster->y()) + lorentzshiftY;
    }
  }

  theClusterParam.probabilityX_ = 0.05;
  theClusterParam.probabilityY_ = 0.05;
  theClusterParam.probabilityQ_ = 0.05;

  return LocalPoint(theClusterParam.NNXrec_, theClusterParam.NNYrec_);
}

//------------------------------------------------------------------
//  localError() relies on localPosition() being called FIRST!!!
//------------------------------------------------------------------
LocalError PixelCPENNReco::localError(DetParam const& theDetParam, ClusterParam& theClusterParamBase) const {
  ClusterParamNN& theClusterParam = static_cast<ClusterParamNN&>(theClusterParamBase);

  float xerr, yerr;

  // Check if the errors were already set at the clusters splitting level
  if (theClusterParam.theCluster->getSplitClusterErrorX() > 0.0f &&
      theClusterParam.theCluster->getSplitClusterErrorX() < clusterSplitMaxError_ &&
      theClusterParam.theCluster->getSplitClusterErrorY() > 0.0f &&
      theClusterParam.theCluster->getSplitClusterErrorY() < clusterSplitMaxError_) {
    xerr = theClusterParam.theCluster->getSplitClusterErrorX() * micronsToCm;
    yerr = theClusterParam.theCluster->getSplitClusterErrorY() * micronsToCm;

    LogDebug("PixelCPENNReco") << "Errors set at cluster splitting level : " << "xerr = " << xerr << ", yerr = " << yerr
                               << endl;
  } else {
    // If errors are not split at the cluster splitting level, set the errors here

    int maxPixelCol = theClusterParam.theCluster->maxPixelCol();
    int maxPixelRow = theClusterParam.theCluster->maxPixelRow();
    int minPixelCol = theClusterParam.theCluster->minPixelCol();
    int minPixelRow = theClusterParam.theCluster->minPixelRow();

    bool edgex =
        (theDetParam.theTopol->isItEdgePixelInX(minPixelRow) || theDetParam.theTopol->isItEdgePixelInX(maxPixelRow));
    bool edgey =
        (theDetParam.theTopol->isItEdgePixelInY(minPixelCol) || theDetParam.theTopol->isItEdgePixelInY(maxPixelCol));
    if (theClusterParam.ierr != 0) {
      // If reconstruction fails the hit position is calculated from cluster center of gravity
      // corrected in x by average Lorentz drift. Assign huge errors.

      if (!GeomDetEnumerators::isTrackerPixel(theDetParam.thePart))
        throw cms::Exception("PixelCPENNReco::localPosition :") << "A non-pixel detector type in here?";

      // Assign better errors based on the residuals for failed template cases
      if (GeomDetEnumerators::isBarrel(theDetParam.thePart)) {
        xerr = 55.0f * micronsToCm;
        yerr = 36.0f * micronsToCm;
      } else {
        xerr = 42.0f * micronsToCm;
        yerr = 39.0f * micronsToCm;
      }

    } else if (edgex || edgey) {
      // for edge pixels assign errors according to observed residual RMS
      if (edgex && !edgey) {
        xerr = xEdgeXError_ * micronsToCm;
        yerr = xEdgeYError_ * micronsToCm;
      } else if (!edgex && edgey) {
        xerr = yEdgeXError_ * micronsToCm;
        yerr = yEdgeYError_ * micronsToCm;
      } else if (edgex && edgey) {
        xerr = bothEdgeXError_ * micronsToCm;
        yerr = bothEdgeYError_ * micronsToCm;
      } else {
        throw cms::Exception(" PixelCPENNReco::localError: Something wrong with pixel edge flag !!!");
      }

    } else {
      xerr = theClusterParam.NNSigmaX_;
      yerr = theClusterParam.NNSigmaY_;
    }

    if (theVerboseLevel > 9) {
      LogDebug("PixelCPENNReco") << " Sizex = " << theClusterParam.theCluster->sizeX()
                                 << " Sizey = " << theClusterParam.theCluster->sizeY() << " Edgex = " << edgex
                                 << " Edgey = " << edgey << " ErrX  = " << xerr << " ErrY  = " << yerr;
    }

  }  // else

  if (!(xerr > 0.0f)) {
    throw cms::Exception("PixelCPENNReco::localError") << "\nERROR: Negative pixel error xerr = " << xerr << "\n";
  }
  if (!(yerr > 0.0f)) {
    throw cms::Exception("PixelCPENNReco::localError") << "\nERROR: Negative pixel error yerr = " << yerr << "\n";
  }

  return LocalError(xerr * xerr, 0, yerr * yerr);
}

void PixelCPENNReco::fillPSetDescription(edm::ParameterSetDescription& desc) {
  PixelCPEGenericBase::fillPSetDescription(desc);
  desc.add<std::string>("inputTensorName_x", "pixel_projection_x");
  desc.add<std::string>("anglesTensorName_x", "angles");
  desc.add<std::string>("cchargeTensorName_x", "cluster_charge");
  desc.add<std::string>("outputTensorName_x", "Identity");
  desc.add<std::string>("inputTensorName_y", "pixel_projection_y");
  desc.add<std::string>("anglesTensorName_y", "angles");
  desc.add<std::string>("cchargeTensorName_y", "cluster_charge");
  desc.add<std::string>("outputTensorName_y", "Identity");
  // used by PixelCPEGenericBase
  desc.add<double>("EdgeClusterErrorX", 50.0);
  desc.add<double>("EdgeClusterErrorY", 85.0);
  desc.add<bool>("UseErrorsFromTemplates", false);
  desc.add<bool>("TruncatePixelCharge", false);
}
