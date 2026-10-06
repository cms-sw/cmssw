// Include our own header first
#include "RecoLocalTracker/SiPixelRecHits/plugins/PixelCPENNReco.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

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
                               const SiPixelLorentzAngle* lorentzAngleWidth,
                               const cms::Ort::ONNXRuntime* model_)
    : PixelCPEGeneric(conf, mag, geom, ttopo, lorentzAngle, genErrorDBObject, lorentzAngleWidth), model(model_) {
  inputTensorName_x = conf.getParameter<std::string>("inputTensorName_x");
  outputTensorName_x = conf.getParameter<std::string>("outputTensorName_x");

  inputTensorName_y = conf.getParameter<std::string>("inputTensorName_y");
  outputTensorName_y = conf.getParameter<std::string>("outputTensorName_y");

  anglesTensorName = conf.getParameter<std::string>("anglesTensorName");
  cchargeTensorName = conf.getParameter<std::string>("cchargeTensorName");
  modelCategoryName = conf.getParameter<std::string>("modelCategoryName");
}

std::unique_ptr<PixelCPEBase::ClusterParam> PixelCPENNReco::createClusterParam(const SiPixelCluster& cl) const {
  return std::make_unique<ClusterParamNN>(cl);
}

int PixelCPENNReco::PixelPreprocess(const SiPixelCluster& cluster,
                                    const PixelTopology& topol,
                                    const Topology::LocalTrackPred& loc_trk_pred,
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
  constexpr int double_pixel_buffer_size = 5;
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
  // Pixels are stored in clusterizer order, not sorted, while the wide pixel expansion below
  // assumes ascending order of the wide rows/columns
  if (n_double_x == 2 && double_row[0] > double_row[1])
    std::swap(double_row[0], double_row[1]);
  if (n_double_y == 2 && double_col[0] > double_col[1])
    std::swap(double_col[0], double_col[1]);

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
                                   col_offset + mid_y + pitch_center_y);             // lower-left corner
  LocalPoint local_center_pix = topol.localPosition(meas_center_pix, loc_trk_pred);  // takes module bows into account
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

  if (Cluster_sizeX > TXSIZE or Cluster_sizeY > TYSIZE or offset_x + Cluster_sizeX > TXSIZE or
      offset_y + Cluster_sizeY > TYSIZE or mrow + offset_x > TXSIZE or mcol + offset_y > TYSIZE or offset_x < 0 or
      offset_y < 0) {
    LogDebug("PixelCPENNReco") << "cluster does not fit in the NN input matrix";
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
  theClusterParam.useGeneric_ = true;

  if (!GeomDetEnumerators::isTrackerPixel(theDetParam.thePart))
    throw cms::Exception("PixelCPENNReco::localPosition :") << "A non-pixel detector type in here?";

  // NN models exist only for BPIX and are trained with track angles
  if (GeomDetEnumerators::isEndcap(theDetParam.thePart) || !theClusterParam.with_track_angle)
    return PixelCPEGeneric::localPosition(theDetParam, theClusterParam);

  const auto rawId = theDetParam.theDet->geographicalId().rawId();
  const int layer = ttopo_.pxbLayer(rawId);
  const int ladder = ttopo_.pxbLadder(rawId);
  const int module = ttopo_.pxbModule(rawId);

  // category order matches the combined graph's category mapping
  // outer ladders = unflipped = odd nos
  unsigned int iModel;
  if (layer == 1)
    iModel = (ladder % 2 != 0) ? 0 : 1;
  else if (layer == 2)
    iModel = 2;  // using L2new model for all of L2
  else if (layer == 3)
    iModel = (module <= 4) ? 3 : 4;
  else
    iModel = (module <= 4) ? 5 : 6;

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
                               theClusterParam.loc_trk_pred,
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
  if (status != 0)
    return PixelCPEGeneric::localPosition(theDetParam, theClusterParam);

  // define a tensor and fill it with cluster projection
  cms::Ort::FloatArrays inputs{
      std::vector<float>(Cluster_x, Cluster_x + TXSIZE),
      std::vector<float>(Cluster_y, Cluster_y + TYSIZE),
      {Cluster_charge},
      {theClusterParam.cotalpha, theClusterParam.cotbeta},
      {static_cast<float>(iModel)},
  };

  const std::vector<std::string> inputNames{
      inputTensorName_x,
      inputTensorName_y,
      cchargeTensorName,
      anglesTensorName,
      modelCategoryName,
  };

  const std::vector<std::vector<int64_t>> inputShapes{
      {1, TXSIZE, 1},
      {1, TYSIZE, 1},
      {1, 1},
      {1, 2},
      {1, 1},
  };

  auto output = model->run(inputNames, inputs, inputShapes, {outputTensorName_x, outputTensorName_y}, 1);

  if (output.size() != 2 || output[0].size() != 2 || output[1].size() != 2) {
    throw cms::Exception("Configuration") << "Unexpected PixelCPENN ONNX output dimensions";
  }

  const float nnOffsetX = output[0][0] / output_scale;
  const float nnSigmaX = output[0][1] / output_scale;
  const float nnOffsetY = output[1][0] / output_scale;
  const float nnSigmaY = output[1][1] / output_scale;

  constexpr float maxPositionX = 1300.f * micronsToCm;
  constexpr float maxPositionY = 3150.f * micronsToCm;
  constexpr float maxSigmaX = 650.f * micronsToCm;
  constexpr float maxSigmaY = 1575.f * micronsToCm;

  // negated comparisons so that NaN outputs also fail
  if (!(std::abs(nnOffsetX) < maxPositionX && std::abs(nnOffsetY) < maxPositionY && nnSigmaX > 0.f &&
        nnSigmaX < maxSigmaX && nnSigmaY > 0.f && nnSigmaY < maxSigmaY)) {
    LogDebug("PixelCPENNReco") << "NN output out of range, falling back to generic: x = " << nnOffsetX
                               << " sigmaX = " << nnSigmaX << " y = " << nnOffsetY << " sigmaY = " << nnSigmaY;
    return PixelCPEGeneric::localPosition(theDetParam, theClusterParam);
  }

  theClusterParam.useGeneric_ = false;
  theClusterParam.NNXrec_ = nnOffsetX + ClusterCenter_x;
  theClusterParam.NNSigmaX_ = nnSigmaX;
  theClusterParam.NNYrec_ = nnOffsetY + ClusterCenter_y;
  theClusterParam.NNSigmaY_ = nnSigmaY;

  // The NN does not provide a charge bin nor hit probabilities
  theClusterParam.qBin_ = 0;
  theClusterParam.hasFilledProb_ = false;

  return LocalPoint(theClusterParam.NNXrec_, theClusterParam.NNYrec_);
}

//------------------------------------------------------------------
//  localError() relies on localPosition() being called FIRST!!!
//------------------------------------------------------------------
LocalError PixelCPENNReco::localError(DetParam const& theDetParam, ClusterParam& theClusterParamBase) const {
  ClusterParamNN& theClusterParam = static_cast<ClusterParamNN&>(theClusterParamBase);

  if (theClusterParam.useGeneric_)
    return PixelCPEGeneric::localError(theDetParam, theClusterParam);

  float xerr, yerr;

  // Check if the errors were already set at the clusters splitting level
  if (theClusterParam.theCluster->getSplitClusterErrorX() > 0.0f &&
      theClusterParam.theCluster->getSplitClusterErrorX() < clusterSplitMaxError_ &&
      theClusterParam.theCluster->getSplitClusterErrorY() > 0.0f &&
      theClusterParam.theCluster->getSplitClusterErrorY() < clusterSplitMaxError_) {
    xerr = theClusterParam.theCluster->getSplitClusterErrorX() * micronsToCm;
    yerr = theClusterParam.theCluster->getSplitClusterErrorY() * micronsToCm;

    LogDebug("PixelCPENNReco") << "Errors set at cluster splitting level : " << "xerr = " << xerr
                               << ", yerr = " << yerr;
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
    if (edgex || edgey) {
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
  PixelCPEGeneric::fillPSetDescription(desc);
  desc.add<std::string>("inputTensorName_x", "pixel_projection_x");
  desc.add<std::string>("outputTensorName_x", "output_x");
  desc.add<std::string>("inputTensorName_y", "pixel_projection_y");
  desc.add<std::string>("outputTensorName_y", "output_y");
  desc.add<std::string>("anglesTensorName", "angles");
  desc.add<std::string>("cchargeTensorName", "cluster_charge");
  desc.add<std::string>("modelCategoryName", "model_category");
}
