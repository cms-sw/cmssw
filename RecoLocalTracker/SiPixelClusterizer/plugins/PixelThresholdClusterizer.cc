//----------------------------------------------------------------------------
//! \class PixelThresholdClusterizer
//! \brief A specific threshold-based pixel clustering algorithm
//!
//! General logic of PixelThresholdClusterizer:
//!
//! The clusterization is performed on a matrix with size
//! equal to the size of the pixel detector, each cell containing
//! the ADC count of the corresponding pixel.
//! The matrix is reset after each clusterization.
//!
//! The search starts from seed pixels, i.e. pixels with sufficiently
//! large amplitudes, found at the time of filling of the matrix
//! and stored in a SiPixelArrayBuffer.
//!
//! Translate the pixel charge to electrons, we are suppose to
//! do the calibrations ADC->electrons here.
//! Modify the thresholds to be in electrons, convert adc to electrons. d.k. 20/3/06
//! Get rid of the noiseVector. d.k. 28/3/06
//----------------------------------------------------------------------------

// Our own includes
#include "PixelThresholdClusterizer.h"
#include "SiPixelArrayBuffer.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
// Geometry
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
//#include "Geometry/CommonTopologies/RectangularPixelTopology.h"

// STL
#include <stack>
#include <vector>
#include <iostream>
#include <atomic>
#include <algorithm>
#include <limits>

//----------------------------------------------------------------------------
//! Constructor:
//!  Initilize the buffer to hold pixels from a detector module.
//!  This is a vector of 44k ints, stays valid all the time.
//----------------------------------------------------------------------------
PixelThresholdClusterizer::PixelThresholdClusterizer(edm::ParameterSet const& conf)
    :  // Get thresholds in electrons
      thePixelThreshold(conf.getParameter<int>("ChannelThreshold")),
      theSeedThreshold(conf.getParameter<int>("SeedThreshold")),
      theClusterThreshold(conf.getParameter<int>("ClusterThreshold")),
      theClusterThreshold_L1(conf.getParameter<int>("ClusterThreshold_L1")),
      theConversionFactor(conf.getParameter<int>("VCaltoElectronGain")),
      theConversionFactor_L1(conf.getParameter<int>("VCaltoElectronGain_L1")),
      theOffset(conf.getParameter<int>("VCaltoElectronOffset")),
      theOffset_L1(conf.getParameter<int>("VCaltoElectronOffset_L1")),
      theElectronPerADCGain(conf.getParameter<double>("ElectronPerADCGain")),
      doPhase2Calibration(conf.getParameter<bool>("Phase2Calibration")),
      dropDuplicates(conf.getParameter<bool>("DropDuplicates")),
      thePhase2ReadoutMode(conf.getParameter<int>("Phase2ReadoutMode")),
      thePhase2DigiBaseline(conf.getParameter<double>("Phase2DigiBaseline")),
      thePhase2KinkADC(conf.getParameter<int>("Phase2KinkADC")),
      theNumOfRows(0),
      theNumOfCols(0),
      theDetid(0),
      // Get the constants for the miss-calibration studies
      doMissCalibrate(conf.getParameter<bool>("MissCalibrate")),
      doSplitClusters(conf.getParameter<bool>("SplitClusters")) {
  theBuffer.setSize(theNumOfRows, theNumOfCols);
  theFakePixels.clear();
  thePixelOccurrence.clear();
}
/////////////////////////////////////////////////////////////////////////////
PixelThresholdClusterizer::~PixelThresholdClusterizer() {}

// Configuration descriptions
void PixelThresholdClusterizer::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<int>("ChannelThreshold", 1000);
  desc.add<bool>("MissCalibrate", true);
  desc.add<bool>("SplitClusters", false);
  desc.add<int>("VCaltoElectronGain", 65);
  desc.add<int>("VCaltoElectronGain_L1", 65);
  desc.add<int>("VCaltoElectronOffset", -414);
  desc.add<int>("VCaltoElectronOffset_L1", -414);
  desc.add<int>("SeedThreshold", 1000);
  desc.add<int>("ClusterThreshold_L1", 4000);
  desc.add<int>("ClusterThreshold", 4000);
  desc.add<double>("ElectronPerADCGain", 135.);
  desc.add<bool>("DropDuplicates", true);
  desc.add<bool>("Phase2Calibration", false);
  desc.add<int>("Phase2ReadoutMode", -1);
  desc.add<double>("Phase2DigiBaseline", 1200.);
  desc.add<int>("Phase2KinkADC", 8);
}

//----------------------------------------------------------------------------
//!  Prepare the Clusterizer to work on a particular DetUnit.  Re-init the
//!  size of the panel/plaquette (so update nrows and ncols),
//----------------------------------------------------------------------------
bool PixelThresholdClusterizer::setup(const PixelGeomDetUnit* pixDet) {
  // Cache the topology.
  const PixelTopology& topol = pixDet->specificTopology();

  // Get the new sizes.
  int nrows = topol.nrows();     // rows in x
  int ncols = topol.ncolumns();  // cols in y

  theNumOfRows = nrows;  // Set new sizes
  theNumOfCols = ncols;

  if (nrows > theBuffer.rows() || ncols > theBuffer.columns()) {  // change only when a larger is needed
    if (nrows != theNumOfRows || ncols != theNumOfCols)
      edm::LogWarning("setup()") << "pixel buffer redefined to" << nrows << " * " << ncols;
    //theNumOfRows = nrows;  // Set new sizes
    //theNumOfCols = ncols;
    // Resize the buffer
    theBuffer.setSize(nrows, ncols);  // Modify
  }

  theFakePixels.resize(nrows * ncols, false);

  thePixelOccurrence.resize(nrows * ncols, 0);

  return true;
}

#include "PixelThresholdClusterizer.icc"

//----------------------------------------------------------------------------
//!  \brief Clear the internal buffer array.
//!
//!  Pixels which are not part of recognized clusters are NOT ERASED
//!  during the cluster finding.  Erase them now.
//!
//!  TO DO: ask Danek... wouldn't it be faster to simply memcopy() zeros into
//!  the whole buffer array?
//----------------------------------------------------------------------------
void PixelThresholdClusterizer::clear_buffer(DigiIterator begin, DigiIterator end) {
  for (DigiIterator di = begin; di != end; ++di) {
    theBuffer.set_adc(di->row(), di->column(), 0);  // reset pixel adc to 0
  }
}

void PixelThresholdClusterizer::clear_buffer(ClusterIterator begin, ClusterIterator end) {
  for (ClusterIterator ci = begin; ci != end; ++ci) {
    for (int i = 0; i < ci->size(); ++i) {
      const SiPixelCluster::Pixel pixel = ci->pixel(i);

      theBuffer.set_adc(pixel.x, pixel.y, 0);  // reset pixel adc to 0
    }
  }
}

//----------------------------------------------------------------------------
//! \brief Copy adc counts from PixelDigis into the buffer, identify seeds.
//----------------------------------------------------------------------------
void PixelThresholdClusterizer::copy_to_buffer(DigiIterator begin, DigiIterator end) {
#ifdef PIXELREGRESSION
  static std::atomic<int> s_ic = 0;
  in ic = ++s_ic;
  if (ic == 1) {
    // std::cout << (doMissCalibrate ? "VI from db" : "VI linear") << std::endl;
  }
#endif

  //If called with empty/invalid DetSet, warn the user
  if (end <= begin) {
    edm::LogWarning("PixelThresholdClusterizer") << " copy_to_buffer called with empty or invalid range" << std::endl;
    return;
  }

  int electron[end - begin];  // pixel charge in electrons
  memset(electron, 0, (end - begin) * sizeof(int));

  if (doPhase2Calibration) {
    int i = 0;
    for (DigiIterator di = begin; di != end; ++di) {
      electron[i] = calibrate(di->adc(), di->column(), di->row());
      i++;
    }
    assert(i == (end - begin));
  }

  else {
    if (doMissCalibrate) {
      if (theLayer == 1) {
        (*theSiPixelGainCalibrationService_)
            .calibrate(theDetid, begin, end, theConversionFactor_L1, theOffset_L1, electron);
      } else {
        (*theSiPixelGainCalibrationService_).calibrate(theDetid, begin, end, theConversionFactor, theOffset, electron);
      }
    } else {
      int i = 0;
      const float gain = theElectronPerADCGain;  // default: 1 ADC = 135 electrons
      for (DigiIterator di = begin; di != end; ++di) {
        auto adc = di->adc();
        const float pedestal = 0.;  //
        electron[i] = int(adc * gain + pedestal);
        ++i;
      }
      assert(i == (end - begin));
    }
  }

  int i = 0;
#ifdef PIXELREGRESSION
  static std::atomic<int> eqD = 0;
#endif
  for (DigiIterator di = begin; di != end; ++di) {
    int row = di->row();
    int col = di->column();
    // VV: do not calibrate a fake pixel, it already has a unit of 10e-:
    int adc = (di->flag() != 0) ? di->adc() * 10 : electron[i];  // this is in electrons
    i++;

#ifdef PIXELREGRESSION
    int adcOld = calibrate(di->adc(), col, row);
    //assert(adc==adcOld);
    if (adc != adcOld)
      std::cout << "VI " << eqD << ' ' << ic << ' ' << end - begin << ' ' << i << ' ' << di->adc() << ' ' << adc << ' '
                << adcOld << std::endl;
    else
      ++eqD;
#endif

    if (adc < 100)
      adc = 100;  // put all negative pixel charges into the 100 elec bin
    /* This is semi-random good number. The exact number (in place of 100) is irrelevant from the point
       of view of the final cluster charge since these are typically >= 20000.
    */

    thePixelOccurrence[theBuffer.index(row, col)]++;  // increment the occurrence counter
    uint8_t occurrence =
        (!dropDuplicates) ? 1 : thePixelOccurrence[theBuffer.index(row, col)];  // get the occurrence counter

    switch (occurrence) {
      // the 1st occurrence (standard treatment)
      case 1:
        if (adc >= thePixelThreshold) {
          theBuffer.set_adc(row, col, adc);
          // VV: add pixel to the fake list. Only when running on digi collection
          if (di->flag() != 0)
            theFakePixels[row * theNumOfCols + col] = true;
          if (adc >= theSeedThreshold)
            theSeeds.push_back(SiPixelCluster::PixelPos(row, col));
        }
        break;

      // the 2nd occurrence (duplicate pixel: reset the buffer to 0 and remove from the list of seed pixels)
      case 2:
        theBuffer.set_adc(row, col, 0);
        std::remove(theSeeds.begin(), theSeeds.end(), SiPixelCluster::PixelPos(row, col));
        break;

        // in case a pixel appears more than twice, nothing needs to be done because it was already removed at the 2nd occurrence
    }
  }
  assert(i == (end - begin));
}

void PixelThresholdClusterizer::copy_to_buffer(ClusterIterator begin, ClusterIterator end) {
  // loop over clusters
  for (ClusterIterator ci = begin; ci != end; ++ci) {
    // loop over pixels
    for (int i = 0; i < ci->size(); ++i) {
      const SiPixelCluster::Pixel pixel = ci->pixel(i);

      int row = pixel.x;
      int col = pixel.y;
      int adc = pixel.adc;
      if (adc >= thePixelThreshold) {
        theBuffer.add_adc(row, col, adc);
        if (adc >= theSeedThreshold)
          theSeeds.push_back(SiPixelCluster::PixelPos(row, col));
      }
    }
  }
}

//----------------------------------------------------------------------------
// Calibrate adc counts to electrons
//-----------------------------------------------------------------
int PixelThresholdClusterizer::calibrate(int adc, int col, int row) {
  int electrons = 0;

  if (doPhase2Calibration) {
    const float gain = theElectronPerADCGain;
    int p2rm = (thePhase2ReadoutMode < -1 ? -1 : thePhase2ReadoutMode);

    if (p2rm == -1) {
      electrons = int(adc * gain);
    } else {
      if (adc < thePhase2KinkADC) {
        electrons = int((adc + 0.5) * gain);
      } else {
        const int dualslopeparam = (thePhase2ReadoutMode < 10 ? thePhase2ReadoutMode : 10);
        const int dualslope = int(dualslopeparam <= 1 ? 1. : pow(2, dualslopeparam - 1));
        adc -= thePhase2KinkADC;
        adc *= dualslope;
        adc += thePhase2KinkADC;
        electrons = int((adc + 0.5 * dualslope) * gain);
      }
      electrons += int(thePhase2DigiBaseline);
    }

    return electrons;
  }

  if (doMissCalibrate) {
    // do not perform calibration if pixel is dead!

    if (!theSiPixelGainCalibrationService_->isDead(theDetid, col, row) &&
        !theSiPixelGainCalibrationService_->isNoisy(theDetid, col, row)) {
      // Linear approximation of the TANH response
      // Pixel(0,0,0)
      //const float gain = 2.95; // 1 ADC = 2.95 VCALs (1/0.339)
      //const float pedestal = -83.; // -28/0.339
      // Roc-0 average
      //const float gain = 1./0.357; // 1 ADC = 2.80 VCALs
      //const float pedestal = -28.2 * gain; // -79.

      float DBgain = theSiPixelGainCalibrationService_->getGain(theDetid, col, row);
      float pedestal = theSiPixelGainCalibrationService_->getPedestal(theDetid, col, row);
      float DBpedestal = pedestal * DBgain;

      // Roc-6 average
      //const float gain = 1./0.313; // 1 ADC = 3.19 VCALs
      //const float pedestal = -6.2 * gain; // -19.8
      //
      float vcal = adc * DBgain - DBpedestal;

      // atanh calibration
      // Roc-6 average
      //const float p0 = 0.00492;
      //const float p1 = 1.998;
      //const float p2 = 90.6;
      //const float p3 = 134.1;
      // Roc-6 average
      //const float p0 = 0.00382;
      //const float p1 = 0.886;
      //const float p2 = 112.7;
      //const float p3 = 113.0;
      //float vcal = ( atanh( (adc-p3)/p2) + p1)/p0;

      if (theLayer == 1) {
        electrons = int(vcal * theConversionFactor_L1 + theOffset_L1);
      } else {
        electrons = int(vcal * theConversionFactor + theOffset);
      }
    }
  } else {  // No misscalibration in the digitizer
    // Simple (default) linear gain
    const float gain = theElectronPerADCGain;  // default: 1 ADC = 135 electrons
    const float pedestal = 0.;                 //
    electrons = int(adc * gain + pedestal);
  }

  return electrons;
}

//----------------------------------------------------------------------------
//!  \brief The actual clustering algorithm: group the neighboring pixels around the seed.
//----------------------------------------------------------------------------
SiPixelCluster PixelThresholdClusterizer::make_cluster(const SiPixelCluster::PixelPos& pix,
                                                       edmNew::DetSetVector<SiPixelCluster>::FastFiller& output) {
  //First we acquire the seeds for the clusters
  uint16_t seed_adc;
  std::stack<SiPixelCluster::PixelPos, std::vector<SiPixelCluster::PixelPos> > dead_pixel_stack;

  //The individual modules have been loaded into a buffer.
  //After each pixel has been considered by the clusterizer, we set the adc count to 1
  //to mark that we have already considered it.
  //The only difference between dead/noisy pixels and standard ones is that for dead/noisy pixels,
  //We consider the charge of the pixel to always be zero.

  /*  this is not possible as dead and noisy pixel cannot make it into a seed...
  if ( doMissCalibrate &&
       (theSiPixelGainCalibrationService_->isDead(theDetid,pix.col(),pix.row()) ||
	theSiPixelGainCalibrationService_->isNoisy(theDetid,pix.col(),pix.row())) )
    {
      std::cout << "IMPOSSIBLE" << std::endl;
      seed_adc = 0;
      theBuffer.set_adc(pix, 1);
    }
    else {
  */
  // Note: each ADC value is limited here to 65535 (std::numeric_limits<uint16_t>::max),
  //       as it is later stored as uint16_t in SiPixelCluster and PixelClusterizerBase/AccretionCluster
  //       (reminder: ADC values here may be expressed in number of electrons)
  seed_adc = std::min(theBuffer(pix.row(), pix.col()), int(std::numeric_limits<uint16_t>::max()));
  theBuffer.set_adc(pix, 1);
  //  }

  AccretionCluster acluster, cldata;
  acluster.add(pix, seed_adc);
  cldata.add(pix, seed_adc);

  //Here we search all pixels adjacent to all pixels in the cluster.
  bool dead_flag = false;
  while (!acluster.empty()) {
    //This is the standard algorithm to find and add a pixel
    auto curInd = acluster.top();
    acluster.pop();
    for (auto c = std::max(0, int(acluster.y[curInd]) - 1);
         c < std::min(int(acluster.y[curInd]) + 2, theBuffer.columns());
         ++c) {
      for (auto r = std::max(0, int(acluster.x[curInd]) - 1);
           r < std::min(int(acluster.x[curInd]) + 2, theBuffer.rows());
           ++r) {
        if (theBuffer(r, c) >= thePixelThreshold) {
          SiPixelCluster::PixelPos newpix(r, c);
          auto const newpix_adc = std::min(theBuffer(r, c), int(std::numeric_limits<uint16_t>::max()));
          if (!acluster.add(newpix, newpix_adc))
            goto endClus;
          // VV: no fake pixels in cluster, leads to non-contiguous clusters
          if (!theFakePixels[r * theNumOfCols + c]) {
            cldata.add(newpix, newpix_adc);
          }
          theBuffer.set_adc(newpix, 1);
        }

        /* //Commenting out the addition of dead pixels to the cluster until further testing -- dfehling 06/09
	      //Check on the bounds of the module; this is to keep the isDead and isNoisy modules from returning errors
	      else if(r>= 0 && c >= 0 && (r <= (theNumOfRows-1.)) && (c <= (theNumOfCols-1.))){
	      //Check for dead/noisy pixels check that the buffer is not -1 (already considered).  Check whether we want to split clusters separated by dead pixels or not.
	      if((theSiPixelGainCalibrationService_->isDead(theDetid,c,r) || theSiPixelGainCalibrationService_->isNoisy(theDetid,c,r)) && theBuffer(r,c) != 1){

	      //If a pixel is dead or noisy, check to see if we want to split the clusters or not.
	      //Push it into a dead pixel stack in case we want to split the clusters.  Otherwise add it to the cluster.
   	      //If we are splitting the clusters, we will iterate over the dead pixel stack later.

	      SiPixelCluster::PixelPos newpix(r,c);
	      if(!doSplitClusters){

	      cluster.add(newpix, std::min(theBuffer(r, c), int(std::numeric_limits<uint16_t>::max())));}
	      else if(doSplitClusters){
	      dead_pixel_stack.push(newpix);
	      dead_flag = true;}

	      theBuffer.set_adc(newpix, 1);
	      }

	      }
	      */
      }
    }

  }  // while accretion
endClus:
  SiPixelCluster cluster(cldata.isize, cldata.adc, cldata.x, cldata.y, cldata.xmin, cldata.ymin);
  //Here we split the cluster, if the flag to do so is set and we have found a dead or noisy pixel.

  if (dead_flag && doSplitClusters) {
    // Set separate cluster threshold for L1 (needed for phase1)
    auto clusterThreshold = theClusterThreshold;
    if (theLayer == 1)
      clusterThreshold = theClusterThreshold_L1;

    //Set the first cluster equal to the existing cluster.
    SiPixelCluster first_cluster = cluster;
    bool have_second_cluster = false;
    while (!dead_pixel_stack.empty()) {
      //consider each found dead pixel
      SiPixelCluster::PixelPos deadpix = dead_pixel_stack.top();
      dead_pixel_stack.pop();
      theBuffer.set_adc(deadpix, 1);

      //Clusterize the split cluster using the dead pixel as a seed
      SiPixelCluster second_cluster = make_cluster(deadpix, output);

      //If both clusters would normally have been found by the clusterizer, put them into output
      if (second_cluster.charge() >= clusterThreshold && first_cluster.charge() >= clusterThreshold) {
        output.push_back(second_cluster);
        have_second_cluster = true;
      }

      //We also want to keep the merged cluster in data and let the RecHit algorithm decide which set to keep
      //This loop adds the second cluster to the first.
      const std::vector<SiPixelCluster::Pixel>& branch_pixels = second_cluster.pixels();
      for (unsigned int i = 0; i < branch_pixels.size(); i++) {
        auto const temp_x = branch_pixels[i].x;
        auto const temp_y = branch_pixels[i].y;
        auto const temp_adc = branch_pixels[i].adc;
        SiPixelCluster::PixelPos newpix(temp_x, temp_y);
        cluster.add(newpix, temp_adc);
      }
    }

    //Remember to also add the first cluster if we added the second one.
    if (first_cluster.charge() >= clusterThreshold && have_second_cluster) {
      output.push_back(first_cluster);
      std::push_heap(output.begin(), output.end(), [](SiPixelCluster const& cl1, SiPixelCluster const& cl2) {
        return cl1.minPixelRow() < cl2.minPixelRow();
      });
    }
  }

  return cluster;
}
