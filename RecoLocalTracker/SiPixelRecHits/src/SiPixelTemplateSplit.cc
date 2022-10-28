//
//  SiPixelTemplateSplit.cc (Version 2.30)
//
//  Procedure to fit two templates (same angle hypotheses) to a single cluster
//  Return two x- and two y-coordinates for the cluster
//
//  Created by Morris Swartz on 04/10/08.
//
//  Incorporate "cluster repair" to handle dead pixels
//  Take truncation size from new pixmax information
//  Change to allow template sizes to be changed at compile time
//  Move interpolation range error to LogDebug
//  Add q2bin = 5 and change 1-pixel probability to use new template info
//  Add floor for probabilities (no exact zeros)
//  Add ambiguity resolution with crude 2-D templates (v2.00)
//  Pass all containers by alias to prevent excessive cpu-usage (v2.01)
//  Add ambiguity resolution with fancy 2-D templates (v2.10)
//  Make small change to indices for ambiguity resolution (v2.11)
//  Tune x and y errors separately (v2.12)
//  Use template cytemp/cxtemp methods to center the data cluster in the right place when the templates become asymmetric after irradiation (v2.20)
//  Add charge probability to the splitter [tests consistency with a two-hit merged cluster hypothesis]  (v2.20)
//  Improve likelihood normalization slightly (v2.21)
//  Replace hardwired pixel size derived errors with ones from templated pixel sizes (v2.22)
//  Add shape and charge probabilities for the merged cluster hypothesis (v2.23)
//  Incorporate VI-like speed improvements (v2.25)
//  Improve speed by eliminating the triple index boost::multiarray objects and add speed switch to optimize the algorithm (v2.30)
//  Change VVIObjF so it only reads kappa
//

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
//#include <cmath.h>
#else
#include <math.h>
#endif
#include <algorithm>
#include <vector>
#include <list>
#include <iostream>
// ROOT::Math has a c++ function that does the probability calc, but only in v5.12 and later
#include "Math/DistFunc.h"
#include "TMath.h"
// Use current version of gsl instead of ROOT::Math
//#include <gsl/gsl_cdf.h>

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateSplit.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/VVIObj.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) edm::LogError(x)
#define LOGDEBUG(x) LogDebug(x)
static constexpr int theVerboseLevel = 2;
#define ENDL " "
#else
#include "SiPixelTemplateSplit.h"
#include "VVIObj.h"
//#include "SiPixelTemplate2D.h"
//#include "SimpleTemplate2D.h"
//static int theVerboseLevel = {2};
#define LOGERROR(x) std::cout << x << ": "
#define LOGDEBUG(x) std::cout << x << ": "
#define ENDL std::endl
#endif

using namespace SiPixelTemplateSplit;

// *************************************************************************************************************************************
//! Reconstruct the best estimate of the hit positions for pixel clusters.
//! \param         id - (input) identifier of the template to use
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param    cluster - (input) boost multi_array container of 7x21 array of pixel signals,
//!           origin of local coords (0,0) at center of pixel cluster[0][0].
//! \param    ydouble - (input) STL vector of 21 element array to flag a double-pixel
//! \param    xdouble - (input) STL vector of 7 element array to flag a double-pixel
//! \param      templ - (input) the template used in the reconstruction
//! \param      yrec1 - (output) best estimate of first y-coordinate of hit in microns
//! \param      yrec2 - (output) best estimate of second y-coordinate of hit in microns
//! \param     sigmay - (output) best estimate of uncertainty on yrec1 and yrec2 in microns
//! \param     prob2y - (output) probability describing goodness-of-fit to a merged cluster hypothesis for y-reco
//! \param      xrec1 - (output) best estimate of first x-coordinate of hit in microns
//! \param      xrec2 - (output) best estimate of second x-coordinate of hit in microns
//! \param     sigmax - (output) best estimate of uncertainty on xrec1 and xrec2 in microns
//! \param     prob2x - (output) probability describing goodness-of-fit to a merged cluster hypothesis for x-reco
//! \param       q2bin - (output) index (0-4) describing the charge of the cluster assuming a merged 2-hit cluster hypothesis
//!                     [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin]
//! \param     prob2Q - (output) probability that the cluster charge is compatible with a 2-hit merging
//! \param    resolve - (input) if true, use 2-D chisquare to resolve the 2-fold x-y association ambiguity (costs a factor of two in speed)
//! \param      speed - (input) switch (-1->2) trading speed vs robustness
//!                     -1       totally bombproof, searches the entire ranges of template bins,
//!                              calculates Q probability w/ VVIObj
//!                      0       totally bombproof, searches the entire template bin range at full density (no Qprob)
//!                      1       faster, searches same range as 0 but at 1/2 density
//!                      2       fastest, searches same range as 1 but at 1/4 density (no big pix) and 1/2 density (big pix in cluster)
//! \param    dchisq - (output) the delta chisquare estimator used to break the degeneracy (0 means no discrimination, larger than 0.1 is good)
//! \param    deadpix - (input) bool to indicate that there are dead pixels to be included in the analysis
//! \param    zeropix - (input) vector of index pairs pointing to the dead pixels
// *************************************************************************************************************************************
int SiPixelTemplateSplit::PixelTempSplit(int id,
                                         float cotalpha,
                                         float cotbeta,
                                         array_2d& clust,
                                         std::vector<bool>& ydouble,
                                         std::vector<bool>& xdouble,
                                         SiPixelTemplate& templ,
                                         float& yrec1,
                                         float& yrec2,
                                         float& sigmay,
                                         float& prob2y,
                                         float& xrec1,
                                         float& xrec2,
                                         float& sigmax,
                                         float& prob2x,
                                         int& q2bin,
                                         float& prob2Q,
                                         bool resolve,
                                         int speed,
                                         float& dchisq,
                                         bool deadpix,
                                         std::vector<std::pair<int, int> >& zeropix,
                                         SiPixelTemplate2D& templ2D)

{
  // Local variables
  int i, j, k, binq, midpix, fypix, nypix, lypix, logypx, lparm;
  int fxpix, nxpix, lxpix, logxpx, shifty, shiftx;
  int nclusx, nclusy;
  int nybin, ycbin, nxbin, xcbin, minbinj, minbink;
  int deltaj, jmin, jmax, kmin, kmax, km, fxbin, lxbin, fybin, lybin, djy, djx;
  float sythr, sxthr, delta, sigma, sigavg, pseudopix, xsize, ysize, qscale, lanpar[2][5];
  float ss2, ssa, sa2, rat, fq, qtotal, qpixel, qavg;
  float x1p, x2p, y1p, y2p, deltachi2;
  float originx, originy, bias, maxpix, minmax;
  double chi2x, meanx, chi2y, meany, chi2ymin, chi2xmin, chi21max;
  double hchi2, hndof, sigmal1, sigmal2, mpv1, mpv2, arg1, arg2, q05, like, loglike1, loglike2;
  double prvav, mpv, sigmaQ, kappa, xvav;
  float ysum[BYSIZE], xsum[BXSIZE], ysort[BYSIZE], xsort[BXSIZE];
  float ysig2[BYSIZE], xsig2[BXSIZE];
  float yw2[BYSIZE], xw2[BXSIZE], ysw[BYSIZE], xsw[BXSIZE];
  float cluster2[BXM2][BYM2], temp2d1[BXM2][BYM2], temp2d2[BXM2][BYM2];
  bool yd[BYSIZE], xd[BXSIZE], anyyd, anyxd, any2dfail;
  const float sqrt2x = {3.0000}, sqrt2y = {1.7000};
  const float sqrt12 = {3.4641};
  const float probmin = {1.110223e-16};
  const float prob2Qmin = {1.e-5};
  std::pair<int, int> pixel;

  //	bool SiPixelTemplateSplit::SimpleTemplate2D(float cotalpha, float cotbeta, float xhit, float yhit, float thick, float lorxwidth, float lorywidth,
  //						  float qavg, std::vector<bool> ydouble, std::vector<bool> xdouble, float template2d[BXM2][BYM2]);

  // The minimum chi2 for a valid one pixel cluster = pseudopixel contribution only

  const double mean1pix = {0.100}, chi21min = {0.160};

  // First, interpolate the template needed to analyze this cluster
  // check to see of the track direction is in the physical range of the loaded template

  if (!templ.interpolate(id, cotalpha, cotbeta)) {
    LOGDEBUG("SiPixelTemplateReco") << "input cluster direction cot(alpha) = " << cotalpha
                                    << ", cot(beta) = " << cotbeta
                                    << " is not within the acceptance of template ID = " << id
                                    << ", no reconstruction performed" << ENDL;
    return 20;
  }

  // Get pixel dimensions from the template (to allow multiple detectors in the future)

  xsize = templ.xsize();
  ysize = templ.ysize();

  // Define size of pseudopixel

  pseudopix = templ.s50();
  //	q05 = 0.28*pseudopix;
  q05 = 0.05f * pseudopix;

  // Get charge scaling factor

  qscale = templ.qscale();

  // Make a local copy of the cluster container so that we can muck with it

  array_2d cluster = clust;

  // Check that the cluster container is (up to) a 7x21 matrix and matches the dimensions of the double pixel flags

  if (cluster.num_dimensions() != 2) {
    LOGERROR("SiPixelTemplateReco") << "input cluster container (BOOST Multiarray) has wrong number of dimensions"
                                    << ENDL;
    return 3;
  }
  nclusx = (int)cluster.shape()[0];
  nclusy = (int)cluster.shape()[1];
  if (nclusx != (int)xdouble.size()) {
    LOGERROR("SiPixelTemplateReco") << "input cluster container x-size is not equal to double pixel flag container size"
                                    << ENDL;
    return 4;
  }
  if (nclusy != (int)ydouble.size()) {
    LOGERROR("SiPixelTemplateReco") << "input cluster container y-size is not equal to double pixel flag container size"
                                    << ENDL;
    return 5;
  }

  // enforce maximum size

  if (nclusx > TXSIZE) {
    nclusx = TXSIZE;
  }
  if (nclusy > TYSIZE) {
    nclusy = TYSIZE;
  }

  // First, rescale all pixel charges

  for (i = 0; i < nclusy; ++i) {
    for (j = 0; j < nclusx; ++j) {
      if (cluster[j][i] > 0) {
        cluster[j][i] *= qscale;
      }
    }
  }

  // Next, sum the total charge and "decapitate" big pixels

  qtotal = 0.f;
  minmax = 2.0f * templ.pixmax();
  for (i = 0; i < nclusy; ++i) {
    maxpix = minmax;
    if (ydouble[i]) {
      maxpix *= 2.f;
    }
    for (j = 0; j < nclusx; ++j) {
      qtotal += cluster[j][i];
      if (cluster[j][i] > maxpix) {
        cluster[j][i] = maxpix;
      }
    }
  }

  // Do the cluster repair here

  if (deadpix) {
    fypix = BYM3;
    lypix = -1;
    for (i = 0; i < nclusy; ++i) {
      ysum[i] = 0.f;
      // Do preliminary cluster projection in y
      for (j = 0; j < nclusx; ++j) {
        ysum[i] += cluster[j][i];
      }
      if (ysum[i] > 0.f) {
        // identify ends of cluster to determine what the missing charge should be
        if (i < fypix) {
          fypix = i;
        }
        if (i > lypix) {
          lypix = i;
        }
      }
    }

    // Now loop over dead pixel list and "fix" everything

    //First see if the cluster ends are redefined and that we have only one dead pixel per column
    int nyzero[TYSIZE]{};
    std::vector<std::pair<int, int> >::const_iterator zeroIter = zeropix.begin(), zeroEnd = zeropix.end();
    for (; zeroIter != zeroEnd; ++zeroIter) {
      i = zeroIter->second;
      if (i < 0 || i > TYSIZE - 1) {
        LOGERROR("SiPixelTemplateReco") << "dead pixel column y-index " << i << ", no reconstruction performed" << ENDL;
        return 11;
      }

      // count the number of dead pixels in each column
      ++nyzero[i];
      // allow them to redefine the cluster ends
      if (i < fypix) {
        fypix = i;
      }
      if (i > lypix) {
        lypix = i;
      }
    }

    nypix = lypix - fypix + 1;

    // Now adjust the charge in the dead pixels to sum to 0.5*truncation value in the end columns and the truncation value in the interior columns

    for (zeroIter = zeropix.begin(); zeroIter != zeroEnd; ++zeroIter) {
      i = zeroIter->second;
      j = zeroIter->first;
      if (j < 0 || j > TXSIZE - 1) {
        LOGERROR("SiPixelTemplateReco") << "dead pixel column x-index " << j << ", no reconstruction performed" << ENDL;
        return 12;
      }
      if ((i == fypix || i == lypix) && nypix > 1) {
        maxpix = templ.symax() / 2.;
      } else {
        maxpix = templ.symax();
      }
      if (ydouble[i]) {
        maxpix *= 2.;
      }
      if (nyzero[i] > 0 && nyzero[i] < 3) {
        qpixel = (maxpix - ysum[i]) / (float)nyzero[i];
      } else {
        qpixel = 1.;
      }
      if (qpixel < 1.f) {
        qpixel = 1.f;
      }
      cluster[j][i] = qpixel;
      // Adjust the total cluster charge to reflect the charge of the "repaired" cluster
      qtotal += qpixel;
    }
    // End of cluster repair section
  }

  // Next, make y-projection of the cluster and copy the double pixel flags into a 25 element container

  for (i = 0; i < BYSIZE; ++i) {
    ysum[i] = 0.f;
    yd[i] = false;
  }
  k = 0;
  anyyd = false;
  for (i = 0; i < nclusy; ++i) {
    for (j = 0; j < nclusx; ++j) {
      ysum[k] += cluster[j][i];
    }

    // If this is a double pixel, put 1/2 of the charge in 2 consective single pixels

    if (ydouble[i]) {
      ysum[k] /= 2.f;
      ysum[k + 1] = ysum[k];
      yd[k] = true;
      yd[k + 1] = false;
      k = k + 2;
      anyyd = true;
    } else {
      yd[k] = false;
      ++k;
    }
    if (k > BYM1) {
      break;
    }
  }

  // Next, make x-projection of the cluster and copy the double pixel flags into an 11 element container

  for (i = 0; i < BXSIZE; ++i) {
    xsum[i] = 0.f;
    xd[i] = false;
  }
  k = 0;
  anyxd = false;
  for (j = 0; j < nclusx; ++j) {
    for (i = 0; i < nclusy; ++i) {
      xsum[k] += cluster[j][i];
    }

    // If this is a double pixel, put 1/2 of the charge in 2 consective single pixels

    if (xdouble[j]) {
      xsum[k] /= 2.f;
      xsum[k + 1] = xsum[k];
      xd[k] = true;
      xd[k + 1] = false;
      k = k + 2;
      anyxd = true;
    } else {
      xd[k] = false;
      ++k;
    }
    if (k > BXM1) {
      break;
    }
  }

  // next, identify the y-cluster ends, count total pixels, nypix, and logical pixels, logypx

  fypix = -1;
  nypix = 0;
  lypix = 0;
  logypx = 0;
  for (i = 0; i < BYSIZE; ++i) {
    if (ysum[i] > 0.) {
      if (fypix == -1) {
        fypix = i;
      }
      if (!yd[i]) {
        ysort[logypx] = ysum[i];
        ++logypx;
      }
      ++nypix;
      lypix = i;
    }
  }

  // Make sure cluster is continuous

  if ((lypix - fypix + 1) != nypix || nypix == 0) {
    LOGDEBUG("SiPixelTemplateReco") << "y-length of pixel cluster doesn't agree with number of pixels above threshold"
                                    << ENDL;
    if (theVerboseLevel > 2) {
      LOGDEBUG("SiPixelTemplateReco") << "ysum[] = ";
      for (i = 0; i < BYSIZE - 1; ++i) {
        LOGDEBUG("SiPixelTemplateReco") << ysum[i] << ", ";
      }
      LOGDEBUG("SiPixelTemplateReco") << ysum[BYSIZE - 1] << ENDL;
    }

    return 1;
  }

  // If cluster is longer than max template size, technique fails

  if (nypix > TYSIZE) {
    LOGDEBUG("SiPixelTemplateReco") << "y-length of pixel cluster is larger than maximum template size" << ENDL;
    if (theVerboseLevel > 2) {
      LOGDEBUG("SiPixelTemplateReco") << "ysum[] = ";
      for (i = 0; i < BYSIZE - 1; ++i) {
        LOGDEBUG("SiPixelTemplateReco") << ysum[i] << ", ";
      }
      LOGDEBUG("SiPixelTemplateReco") << ysum[BYSIZE - 1] << ENDL;
    }

    return 6;
  }

  // next, center the cluster on template center if necessary

  midpix = (fypix + lypix) / 2;
  //	shifty = BHY - midpix;
  shifty = templ.cytemp() - midpix;
  if (shifty > 0) {
    for (i = lypix; i >= fypix; --i) {
      ysum[i + shifty] = ysum[i];
      ysum[i] = 0.;
      yd[i + shifty] = yd[i];
      yd[i] = false;
    }
  } else if (shifty < 0) {
    for (i = fypix; i <= lypix; ++i) {
      ysum[i + shifty] = ysum[i];
      ysum[i] = 0.;
      yd[i + shifty] = yd[i];
      yd[i] = false;
    }
  }
  lypix += shifty;
  fypix += shifty;

  // If the cluster boundaries are OK, add pesudopixels, otherwise quit

  if (fypix > 1 && fypix < BYM2) {
    ysum[fypix - 1] = pseudopix;
    ysum[fypix - 2] = 0.2f * pseudopix;
  } else {
    return 8;
  }
  if (lypix > 1 && lypix < BYM2) {
    ysum[lypix + 1] = pseudopix;
    ysum[lypix + 2] = 0.2f * pseudopix;
  } else {
    return 8;
  }

  // finally, determine if pixel[0] is a double pixel and make an origin correction if it is

  if (ydouble[0]) {
    originy = -0.5f;
  } else {
    originy = 0.f;
  }

  // next, identify the x-cluster ends, count total pixels, nxpix, and logical pixels, logxpx

  fxpix = -1;
  nxpix = 0;
  lxpix = 0;
  logxpx = 0;
  for (i = 0; i < BXSIZE; ++i) {
    if (xsum[i] > 0.) {
      if (fxpix == -1) {
        fxpix = i;
      }
      if (!xd[i]) {
        xsort[logxpx] = xsum[i];
        ++logxpx;
      }
      ++nxpix;
      lxpix = i;
    }
  }

  // Make sure cluster is continuous

  if ((lxpix - fxpix + 1) != nxpix) {
    LOGDEBUG("SiPixelTemplateReco") << "x-length of pixel cluster doesn't agree with number of pixels above threshold"
                                    << ENDL;
    if (theVerboseLevel > 2) {
      LOGDEBUG("SiPixelTemplateReco") << "xsum[] = ";
      for (i = 0; i < BXSIZE - 1; ++i) {
        LOGDEBUG("SiPixelTemplateReco") << xsum[i] << ", ";
      }
      LOGDEBUG("SiPixelTemplateReco") << ysum[BXSIZE - 1] << ENDL;
    }

    return 2;
  }

  // If cluster is longer than max template size, technique fails

  if (nxpix > TXSIZE) {
    LOGDEBUG("SiPixelTemplateReco") << "x-length of pixel cluster is larger than maximum template size" << ENDL;
    if (theVerboseLevel > 2) {
      LOGDEBUG("SiPixelTemplateReco") << "xsum[] = ";
      for (i = 0; i < BXSIZE - 1; ++i) {
        LOGDEBUG("SiPixelTemplateReco") << xsum[i] << ", ";
      }
      LOGDEBUG("SiPixelTemplateReco") << ysum[BXSIZE - 1] << ENDL;
    }

    return 7;
  }

  // next, center the cluster on template center if necessary

  midpix = (fxpix + lxpix) / 2;
  //	shiftx = BHX - midpix;
  shiftx = templ.cxtemp() - midpix;
  if (shiftx > 0) {
    for (i = lxpix; i >= fxpix; --i) {
      xsum[i + shiftx] = xsum[i];
      xsum[i] = 0.f;
      xd[i + shiftx] = xd[i];
      xd[i] = false;
    }
  } else if (shiftx < 0) {
    for (i = fxpix; i <= lxpix; ++i) {
      xsum[i + shiftx] = xsum[i];
      xsum[i] = 0.f;
      xd[i + shiftx] = xd[i];
      xd[i] = false;
    }
  }
  lxpix += shiftx;
  fxpix += shiftx;

  // If the cluster boundaries are OK, add pesudopixels, otherwise quit

  if (fxpix > 1 && fxpix < BXM2) {
    xsum[fxpix - 1] = pseudopix;
    xsum[fxpix - 2] = 0.2f * pseudopix;
  } else {
    return 9;
  }
  if (lxpix > 1 && lxpix < BXM2) {
    xsum[lxpix + 1] = pseudopix;
    xsum[lxpix + 2] = 0.2f * pseudopix;
  } else {
    return 9;
  }

  // finally, determine if pixel[0] is a double pixel and make an origin correction if it is

  if (xdouble[0]) {
    originx = -0.5f;
  } else {
    originx = 0.f;
  }

  // uncertainty and final corrections depend upon total charge bin

  qavg = templ.qavg();
  fq = qtotal / qavg;
  if (fq > 3.0f) {
    binq = 0;
  } else {
    if (fq > 2.0f) {
      binq = 1;
    } else {
      if (fq > 1.70f) {
        binq = 2;
      } else {
        binq = 3;
      }
    }
  }

  // Calculate the Vavilov probability that the cluster charge is consistent with a merged cluster

  if (speed < 0) {
    templ.vavilov2_pars(mpv, sigmaQ, kappa);
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if ((sigmaQ <= 0.) || (mpv <= 0.) || (kappa < 0.01) || (kappa > 9.9)) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplateSplit::Vavilov parameters mpv/sigmaQ/kappa = " << mpv
                                          << "/" << sigmaQ << "/" << kappa << std::endl;
    }
#else
    assert((sigmaQ > 0.) && (mpv > 0.) && (kappa > 0.01) && (kappa < 10.));
#endif
    xvav = ((double)qtotal - mpv) / sigmaQ;
    //  VVIObj is a private port of CERNLIB VVIDIS
    VVIObj vvidist(kappa);
    prvav = vvidist.fcn(xvav);
    prob2Q = 1. - prvav;
    if (prob2Q < prob2Qmin) {
      prob2Q = prob2Qmin;
    }
  } else {
    prob2Q = -1.f;
  }

  // Return the charge bin via the parameter list unless the charge is too small (then flag it)

  q2bin = binq;
  if (!deadpix && qtotal < 1.9f * templ.qmin()) {
    q2bin = 5;
  } else {
    if (!deadpix && qtotal < 1.9f * templ.qmin(1)) {
      q2bin = 4;
    }
  }

  if (theVerboseLevel > 9) {
    LOGDEBUG("SiPixelTemplateSplit") << "ID = " << id << " cot(alpha) = " << cotalpha << " cot(beta) = " << cotbeta
                                     << " nclusx = " << nclusx << " nclusy = " << nclusy << ENDL;
  }

  // Next, generate the 3d y- and x-templates

  templ.ytemp3d_int(nypix, nybin);

  ycbin = nybin / 2;

  templ.xtemp3d_int(nxpix, nxbin);

  // retrieve the number of x-bins

  xcbin = nxbin / 2;

  // First, decide on chi^2 min search parameters

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (speed < -1 || speed > 2) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplateReco::PixelTempReco2D called with illegal speed = " << speed
                                        << std::endl;
  }
#else
  assert(speed >= -1 && speed < 3);
#endif
  fybin = 0;
  lybin = nybin - 1;
  fxbin = 0;
  lxbin = nxbin - 1;
  djy = 1;
  djx = 1;
  if (speed > 0) {
    djy = 2;
    djx = 2;
    if (speed > 1) {
      if (!anyyd) {
        djy = 4;
      }
      if (!anyxd) {
        djx = 4;
      }
    }
  }

  if (theVerboseLevel > 9) {
    LOGDEBUG("SiPixelTemplateReco") << "fypix " << fypix << " lypix = " << lypix << " fybin = " << fybin
                                    << " lybin = " << lybin << " djy = " << djy << " logypx = " << logypx << ENDL;
    LOGDEBUG("SiPixelTemplateReco") << "fxpix " << fxpix << " lxpix = " << lxpix << " fxbin = " << fxbin
                                    << " lxbin = " << lxbin << " djx = " << djx << " logxpx = " << logxpx << ENDL;
  }

  // Do the y-reconstruction first

  // Define the maximum signal to allow before de-weighting a pixel

  sythr = 2.1f * (templ.symax());

  // Make sure that there will be at least two pixels that are not de-weighted

  std::sort(&ysort[0], &ysort[logypx]);
  if (logypx == 1) {
    sythr = 1.01f * ysort[0];
  } else {
    if (ysort[1] > sythr) {
      sythr = 1.01f * ysort[1];
    }
  }

  // Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis

  //	for(i=0; i<BYSIZE; ++i) { ysig2[i] = 0.;}
  templ.ysigma2(fypix, lypix, sythr, ysum, ysig2);

  // Find the template bin that minimizes the Chi^2

  chi2ymin = 1.e15;
  ss2 = 0.f;
  for (i = fypix - 2; i <= lypix + 2; ++i) {
    yw2[i] = 1.f / ysig2[i];
    ysw[i] = ysum[i] * yw2[i];
    ss2 += ysum[i] * ysw[i];
  }
  minbinj = -1;
  minbink = -1;
  deltaj = djy;
  jmin = fybin;
  jmax = lybin;
  kmin = fybin;
  kmax = lybin;
  std::vector<float> ytemp(BYSIZE);
  while (deltaj > 0) {
    for (j = jmin; j < jmax; j += deltaj) {
      km = std::min(kmax, j);
      for (k = kmin; k <= km; k += deltaj) {
        // Get the template for this set of indices

        templ.ytemp3d(j, k, ytemp);

        // Modify the template if double pixels are present

        if (nypix > logypx) {
          i = fypix;
          while (i < lypix) {
            if (yd[i] && !yd[i + 1]) {
              // Sum the adjacent cells and put the average signal in both

              sigavg = (ytemp[i] + ytemp[i + 1]) / 2.f;
              ytemp[i] = sigavg;
              ytemp[i + 1] = sigavg;
              i += 2;
            } else {
              ++i;
            }
          }
        }
        ssa = 0.f;
        sa2 = 0.f;
        for (i = fypix - 2; i <= lypix + 2; ++i) {
          ssa += ysw[i] * ytemp[i];
          sa2 += ytemp[i] * ytemp[i] * yw2[i];
        }
        rat = ssa / ss2;
        if (rat <= 0.) {
          LOGERROR("SiPixelTemplateSplit") << "illegal chi2ymin normalization = " << rat << ENDL;
          rat = 1.;
        }
        chi2y = ss2 - 2.f * ssa / rat + sa2 / (rat * rat);
        if (chi2y < chi2ymin) {
          chi2ymin = chi2y;
          minbinj = j;
          minbink = k;
        }
      }
    }
    deltaj /= 2;
    if (minbinj > fybin) {
      jmin = minbinj - deltaj;
    } else {
      jmin = fybin;
    }
    if (minbinj < lybin) {
      jmax = minbinj + deltaj;
    } else {
      jmax = lybin;
    }
    if (minbink > fybin) {
      kmin = minbink - deltaj;
    } else {
      kmin = fybin;
    }
    if (minbink < lybin) {
      kmax = minbink + deltaj;
    } else {
      kmax = lybin;
    }
  }

  if (theVerboseLevel > 9) {
    LOGDEBUG("SiPixelTemplateReco") << "minbins " << minbinj << "," << minbink << " chi2ymin = " << chi2ymin << ENDL;
  }

  // Do not apply final template pass to 1-pixel clusters (use calibrated offset)

  if (logypx == 1) {
    if (nypix == 1) {
      delta = templ.dyone();
      sigma = templ.syone();
    } else {
      delta = templ.dytwo();
      sigma = templ.sytwo();
    }

    yrec1 = 0.5f * (fypix + lypix - 2 * shifty + 2.f * originy) * ysize - delta;
    yrec2 = yrec1;

    if (sigma <= 0.) {
      sigmay = ysize / sqrt12;
    } else {
      sigmay = sigma;
    }

    // Do probability calculation for one-pixel clusters

    chi21max = fmax(chi21min, (double)templ.chi2yminone());
    chi2ymin -= chi21max;
    if (chi2ymin < 0.) {
      chi2ymin = 0.;
    }
    //	   prob2y = gsl_cdf_chisq_Q(chi2ymin, mean1pix);
    meany = fmax(mean1pix, (double)templ.chi2yavgone());
    hchi2 = chi2ymin / 2.;
    hndof = meany / 2.;
    prob2y = 1. - TMath::Gamma(hndof, hchi2);

  } else {
    // For cluster > 1 pix, use chi^2 minimm to recontruct the two y-positions

    // at small eta, the templates won't actually work on two pixel y-clusters so just return the pixel centers

    if (logypx == 2 && fabsf(cotbeta) < 0.25f) {
      switch (nypix) {
        case 2:
          //  Both pixels are small
          yrec1 = (fypix - shifty + originy) * ysize;
          yrec2 = (lypix - shifty + originy) * ysize;
          sigmay = ysize / sqrt12;
          break;
        case 3:
          //  One big pixel and one small pixel
          if (yd[fypix]) {
            yrec1 = (fypix + 0.5f - shifty + originy) * ysize;
            yrec2 = (lypix - shifty + originy) * ysize;
            sigmay = ysize / sqrt12;
          } else {
            yrec1 = (fypix - shifty + originy) * ysize;
            yrec2 = (lypix - 0.5f - shifty + originy) * ysize;
            sigmay = 1.5f * ysize / sqrt12;
          }
          break;
        case 4:
          //  Two big pixels
          yrec1 = (fypix + 0.5f - shifty + originy) * ysize;
          yrec2 = (lypix - 0.5f - shifty + originy) * ysize;
          sigmay = 2.f * ysize / sqrt12;
          break;
        default:
          //  Something is screwy ...
          LOGERROR("SiPixelTemplateReco")
              << "weird problem: logical y-pixels = " << logypx << ", total ysize in normal pixels = " << nypix << ENDL;
          return 10;
      }
    } else {
      // uncertainty and final correction depend upon charge bin

      bias = templ.yavgc2m(binq);
      yrec1 = (0.125f * (minbink - ycbin) + BHY - (float)shifty + originy) * ysize - bias;
      yrec2 = (0.125f * (minbinj - ycbin) + BHY - (float)shifty + originy) * ysize - bias;
      sigmay = sqrt2y * templ.yrmsc2m(binq);
    }

    // Do goodness of fit test in y

    if (chi2ymin < 0.0) {
      chi2ymin = 0.0;
    }
    meany = templ.chi2yavgc2m(binq);
    if (meany < 0.01) {
      meany = 0.01;
    }
    // gsl function that calculates the chi^2 tail prob for non-integral dof
    //	   prob2y = gsl_cdf_chisq_Q(chi2y, meany);
    //	   prob2y = ROOT::Math::chisquared_cdf_c(chi2y, meany);
    hchi2 = chi2ymin / 2.;
    hndof = meany / 2.;
    prob2y = 1. - TMath::Gamma(hndof, hchi2);
  }

  // Do the x-reconstruction next

  // Define the maximum signal to allow before de-weighting a pixel

  sxthr = 2.1f * templ.sxmax();

  // Make sure that there will be at least two pixels that are not de-weighted
  std::sort(&xsort[0], &xsort[logxpx]);
  if (logxpx == 1) {
    sxthr = 1.01f * xsort[0];
  } else {
    if (xsort[1] > sxthr) {
      sxthr = 1.01f * xsort[1];
    }
  }

  // Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis

  //	for(i=0; i<BYSIZE; ++i) { xsig2[i] = 0.; }
  templ.xsigma2(fxpix, lxpix, sxthr, xsum, xsig2);

  // Find the template bin that minimizes the Chi^2

  chi2xmin = 1.e15;
  ss2 = 0.f;
  for (i = fxpix - 2; i <= lxpix + 2; ++i) {
    xw2[i] = 1.f / xsig2[i];
    xsw[i] = xsum[i] * xw2[i];
    ss2 += xsum[i] * xsw[i];
  }
  minbinj = -1;
  minbink = -1;
  deltaj = djx;
  jmin = fxbin;
  jmax = lxbin;
  kmin = fxbin;
  kmax = lxbin;
  std::vector<float> xtemp(BXSIZE);
  while (deltaj > 0) {
    for (j = jmin; j < jmax; j += deltaj) {
      km = std::min(kmax, j);
      for (k = kmin; k <= km; k += deltaj) {
        // Get the template for this set of indices

        templ.xtemp3d(j, k, xtemp);

        // Modify the template if double pixels are present

        if (nxpix > logxpx) {
          i = fxpix;
          while (i < lxpix) {
            if (xd[i] && !xd[i + 1]) {
              // Sum the adjacent cells and put the average signal in both

              sigavg = (xtemp[i] + xtemp[i + 1]) / 2.f;
              xtemp[i] = sigavg;
              xtemp[i + 1] = sigavg;
              i += 2;
            } else {
              ++i;
            }
          }
        }
        ssa = 0.f;
        sa2 = 0.f;
        for (i = fxpix - 2; i <= lxpix + 2; ++i) {
          ssa += xsw[i] * xtemp[i];
          sa2 += xtemp[i] * xtemp[i] * xw2[i];
        }
        rat = ssa / ss2;
        if (rat <= 0.f) {
          LOGERROR("SiPixelTemplateSplit") << "illegal chi2xmin normalization = " << rat << ENDL;
          rat = 1.;
        }
        chi2x = ss2 - 2.f * ssa / rat + sa2 / (rat * rat);
        if (chi2x < chi2xmin) {
          chi2xmin = chi2x;
          minbinj = j;
          minbink = k;
        }
      }
    }
    deltaj /= 2;
    if (minbinj > fxbin) {
      jmin = minbinj - deltaj;
    } else {
      jmin = fxbin;
    }
    if (minbinj < lxbin) {
      jmax = minbinj + deltaj;
    } else {
      jmax = lxbin;
    }
    if (minbink > fxbin) {
      kmin = minbink - deltaj;
    } else {
      kmin = fxbin;
    }
    if (minbink < lxbin) {
      kmax = minbink + deltaj;
    } else {
      kmax = lxbin;
    }
  }

  if (theVerboseLevel > 9) {
    LOGDEBUG("SiPixelTemplateSplit") << "minbinj/minbink " << minbinj << "/" << minbink << " chi2xmin = " << chi2xmin
                                     << ENDL;
  }

  // Do not apply final template pass to 1-pixel clusters (use calibrated offset)

  if (logxpx == 1) {
    if (nxpix == 1) {
      delta = templ.dxone();
      sigma = templ.sxone();
    } else {
      delta = templ.dxtwo();
      sigma = templ.sxtwo();
    }
    xrec1 = 0.5f * (fxpix + lxpix - 2 * shiftx + 2.f * originx) * xsize - delta;
    xrec2 = xrec1;
    if (sigma <= 0.f) {
      sigmax = xsize / sqrt12;
    } else {
      sigmax = sigma;
    }

    // Do probability calculation for one-pixel clusters

    chi21max = fmax(chi21min, (double)templ.chi2xminone());
    chi2xmin -= chi21max;
    if (chi2xmin < 0.) {
      chi2xmin = 0.;
    }
    meanx = fmax(mean1pix, (double)templ.chi2xavgone());
    hchi2 = chi2xmin / 2.;
    hndof = meanx / 2.;
    prob2x = 1. - TMath::Gamma(hndof, hchi2);

  } else {
    // For cluster > 1 pix, use chi^2 minimm to recontruct the two x-positions

    // uncertainty and final correction depend upon charge bin

    bias = templ.xavgc2m(binq);
    xrec1 = (0.125f * (minbink - xcbin) + BHX - (float)shiftx + originx) * xsize - bias;
    xrec2 = (0.125f * (minbinj - xcbin) + BHX - (float)shiftx + originx) * xsize - bias;
    sigmax = sqrt2x * templ.xrmsc2m(binq);

    // Do goodness of fit test in y

    if (chi2xmin < 0.0) {
      chi2xmin = 0.0;
    }
    meanx = templ.chi2xavgc2m(binq);
    if (meanx < 0.01) {
      meanx = 0.01;
    }
    hchi2 = chi2xmin / 2.;
    hndof = meanx / 2.;
    prob2x = 1. - TMath::Gamma(hndof, hchi2);
  }

  //  Don't return exact zeros for the probability

  if (prob2y < probmin) {
    prob2y = probmin;
  }
  if (prob2x < probmin) {
    prob2x = probmin;
  }

  //  New code: resolve the ambiguity if resolve is set to true

  dchisq = 0.;
  if (resolve) {
    //  First copy the unexpanded cluster into a new working buffer

    for (i = 0; i < BYM2; ++i) {
      for (j = 0; j < BXM2; ++j) {
        if ((i > 0 && i < BYM3) && (j > 0 && j < BXM3)) {
          cluster2[j][i] = qscale * clust[j - 1][i - 1];
        } else {
          cluster2[j][i] = 0.f;
        }
      }
    }

    //  Next, redefine the local coordinates to start at the lower LH corner of pixel[1][1];

    if (xdouble[0]) {
      x1p = xrec1 + xsize;
      x2p = xrec2 + xsize;
    } else {
      x1p = xrec1 + xsize / 2.f;
      x2p = xrec2 + xsize / 2.f;
    }

    if (ydouble[0]) {
      y1p = yrec1 + ysize;
      y2p = yrec2 + ysize;
    } else {
      y1p = yrec1 + ysize / 2.f;
      y2p = yrec2 + ysize / 2.f;
    }

    // Next, calculate 2-d templates for the (x1,y1)+(x2,y2) and the (x1,y2)+(x2,y1) hypotheses

    //  First zero the two 2-d hypotheses

    for (i = 0; i < BYM2; ++i) {
      for (j = 0; j < BXM2; ++j) {
        temp2d1[j][i] = 0.f;
        temp2d2[j][i] = 0.f;
      }
    }

    // Add the two hits in the first hypothesis

    any2dfail = templ2D.xytemp(id, cotalpha, cotbeta, x1p, y1p, xdouble, ydouble, temp2d1) &&
                templ2D.xytemp(id, cotalpha, cotbeta, x2p, y2p, xdouble, ydouble, temp2d1);

    // And then the second hypothesis

    any2dfail = any2dfail && templ2D.xytemp(id, cotalpha, cotbeta, x1p, y2p, xdouble, ydouble, temp2d2);

    any2dfail = any2dfail && templ2D.xytemp(id, cotalpha, cotbeta, x2p, y1p, xdouble, ydouble, temp2d2);

    // If any of these have failed, use the simple templates instead

    if (!any2dfail) {
      //  Rezero the two 2-d hypotheses

      for (i = 0; i < BYM2; ++i) {
        for (j = 0; j < BXM2; ++j) {
          temp2d1[j][i] = 0.f;
          temp2d2[j][i] = 0.f;
        }
      }

      // Add the two hits in the first hypothesis

      if (!templ.simpletemplate2D(x1p, y1p, xdouble, ydouble, temp2d1)) {
        return 1;
      }

      if (!templ.simpletemplate2D(x2p, y2p, xdouble, ydouble, temp2d1)) {
        return 1;
      }

      // And then the second hypothesis

      if (!templ.simpletemplate2D(x1p, y2p, xdouble, ydouble, temp2d2)) {
        return 1;
      }

      if (!templ.simpletemplate2D(x2p, y1p, xdouble, ydouble, temp2d2)) {
        return 1;
      }
    }

    // Keep lists of pixels and nearest neighbors

    std::list<std::pair<int, int> > pixellst;

    // Loop through the array and find non-zero elements

    for (i = 0; i < BYM2; ++i) {
      for (j = 0; j < BXM2; ++j) {
        if (cluster2[j][i] > 0.f || temp2d1[j][i] > 0.f || temp2d2[j][i] > 0.f) {
          pixel.first = j;
          pixel.second = i;
          pixellst.push_back(pixel);
        }
      }
    }

    // Now calculate the product of Landau probabilities (alpha probability)

    templ2D.landau_par(lanpar);
    loglike1 = 0.;
    loglike2 = 0.;

    // Now, for each neighbor, match it again the pixel list.
    // If found, delete it from the neighbor list

    std::list<std::pair<int, int> >::const_iterator pixIter, pixEnd;
    pixIter = pixellst.begin();
    pixEnd = pixellst.end();
    for (; pixIter != pixEnd; ++pixIter) {
      j = pixIter->first;
      i = pixIter->second;
      if ((i < BHY && cotbeta > 0.) || (i >= BHY && cotbeta < 0.)) {
        lparm = 0;
      } else {
        lparm = 1;
      }
      mpv1 = lanpar[lparm][0] + lanpar[lparm][1] * temp2d1[j][i];
      sigmal1 = lanpar[lparm][2] * mpv1;
      sigmal1 = sqrt(sigmal1 * sigmal1 + lanpar[lparm][3] * lanpar[lparm][3]);
      if (sigmal1 < q05)
        sigmal1 = q05;
      arg1 = (cluster2[j][i] - mpv1) / sigmal1 - 0.22278;
      mpv2 = lanpar[lparm][0] + lanpar[lparm][1] * temp2d2[j][i];
      sigmal2 = lanpar[lparm][2] * mpv2;
      sigmal2 = sqrt(sigmal2 * sigmal2 + lanpar[lparm][3] * lanpar[lparm][3]);
      if (sigmal2 < q05)
        sigmal2 = q05;
      arg2 = (cluster2[j][i] - mpv2) / sigmal2 - 0.22278;
      //			like = ROOT::Math::landau_pdf(arg1)/sigmal1;
      like = ROOT::Math::landau_pdf(arg1);
      if (like < 1.e-30)
        like = 1.e-30;
      loglike1 += log(like);
      //			like = ROOT::Math::landau_pdf(arg2)/sigmal2;
      like = ROOT::Math::landau_pdf(arg2);
      if (like < 1.e-30)
        like = 1.e-30;
      loglike2 += log(like);
    }

    // Calculate chisquare difference for the two hypotheses 9don't multiply by 2 for less inconvenient scaling

    deltachi2 = loglike1 - loglike2;

    if (deltachi2 < 0.) {
      // Flip the x1 and x2

      x1p = xrec1;
      xrec1 = xrec2;
      xrec2 = x1p;
    }

    // Return a positive definite value

    dchisq = fabs(deltachi2);
  }

  return 0;
}  // PixelTempSplit

// *************************************************************************************************************************************
//! Reconstruct the best estimate of the hit positions for pixel clusters.
//! \param         id - (input) identifier of the template to use
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param    cluster - (input) boost multi_array container of 7x21 array of pixel signals,
//!           origin of local coords (0,0) at center of pixel cluster[0][0].
//! \param    ydouble - (input) STL vector of 21 element array to flag a double-pixel
//! \param    xdouble - (input) STL vector of 7 element array to flag a double-pixel
//! \param      templ - (input) the template used in the reconstruction
//! \param      yrec1 - (output) best estimate of first y-coordinate of hit in microns
//! \param      yrec2 - (output) best estimate of second y-coordinate of hit in microns
//! \param     sigmay - (output) best estimate of uncertainty on yrec in microns
//! \param      prob2y - (output) probability describing goodness-of-fit for y-reco
//! \param      xrec1 - (output) best estimate of first x-coordinate of hit in microns
//! \param      xrec2 - (output) best estimate of second x-coordinate of hit in microns
//! \param     sigmax - (output) best estimate of uncertainty on xrec in microns
//! \param      prob2x - (output) probability describing goodness-of-fit for x-reco
//! \param       q2bin - (output) index (0-4) describing the charge of the cluster
//!                     [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin]
//! \param      prob2Q - (output) probability that the cluster charge is compatible with a 2-hit merging
//! \param    resolve - (input) use 2-D chisquare to resolve the 2-fold x-y association ambiguity (costs a factor of two in speed)
//! \param     dchisq - (output) the delta chisquare estimator used to break the degeneracy (0 means no discrimination, larger than 0.1 is good)
// *************************************************************************************************************************************

int SiPixelTemplateSplit::PixelTempSplit(int id,
                                         float cotalpha,
                                         float cotbeta,
                                         array_2d& cluster,
                                         std::vector<bool>& ydouble,
                                         std::vector<bool>& xdouble,
                                         SiPixelTemplate& templ,
                                         float& yrec1,
                                         float& yrec2,
                                         float& sigmay,
                                         float& prob2y,
                                         float& xrec1,
                                         float& xrec2,
                                         float& sigmax,
                                         float& prob2x,
                                         int& q2bin,
                                         float& prob2Q,
                                         bool resolve,
                                         int speed,
                                         float& dchisq,
                                         SiPixelTemplate2D& templ2D) {
  // Local variables
  const bool deadpix = false;
  std::vector<std::pair<int, int> > zeropix;

  return SiPixelTemplateSplit::PixelTempSplit(id,
                                              cotalpha,
                                              cotbeta,
                                              cluster,
                                              ydouble,
                                              xdouble,
                                              templ,
                                              yrec1,
                                              yrec2,
                                              sigmay,
                                              prob2y,
                                              xrec1,
                                              xrec2,
                                              sigmax,
                                              prob2x,
                                              q2bin,
                                              prob2Q,
                                              resolve,
                                              speed,
                                              dchisq,
                                              deadpix,
                                              zeropix,
                                              templ2D);

}  // PixelTempSplit

// *************************************************************************************************************************************
//! Reconstruct the best estimate of the hit positions for pixel clusters.
//! \param         id - (input) identifier of the template to use
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param    cluster - (input) boost multi_array container of 7x21 array of pixel signals,
//!           origin of local coords (0,0) at center of pixel cluster[0][0].
//! \param    ydouble - (input) STL vector of 21 element array to flag a double-pixel
//! \param    xdouble - (input) STL vector of 7 element array to flag a double-pixel
//! \param      templ - (input) the template used in the reconstruction
//! \param      yrec1 - (output) best estimate of first y-coordinate of hit in microns
//! \param      yrec2 - (output) best estimate of second y-coordinate of hit in microns
//! \param     sigmay - (output) best estimate of uncertainty on yrec in microns
//! \param      prob2y - (output) probability describing goodness-of-fit for y-reco
//! \param      xrec1 - (output) best estimate of first x-coordinate of hit in microns
//! \param      xrec2 - (output) best estimate of second x-coordinate of hit in microns
//! \param     sigmax - (output) best estimate of uncertainty on xrec in microns
//! \param      prob2x - (output) probability describing goodness-of-fit for x-reco
//! \param       q2bin - (output) index (0-4) describing the charge of the cluster
//!                     [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin]
//! \param      prob2Q - (output) probability that the cluster charge is compatible with a 2-hit merging
//! \param    resolve - (input) use 2-D chisquare to resolve the 2-fold x-y association ambiguity (costs a factor of two in speed)
//! \param     dchisq - (output) the delta chisquare estimator used to break the degeneracy (0 means no discrimination, larger than 0.1 is good)
// *************************************************************************************************************************************

int SiPixelTemplateSplit::PixelTempSplit(int id,
                                         float cotalpha,
                                         float cotbeta,
                                         array_2d& cluster,
                                         std::vector<bool>& ydouble,
                                         std::vector<bool>& xdouble,
                                         SiPixelTemplate& templ,
                                         float& yrec1,
                                         float& yrec2,
                                         float& sigmay,
                                         float& prob2y,
                                         float& xrec1,
                                         float& xrec2,
                                         float& sigmax,
                                         float& prob2x,
                                         int& q2bin,
                                         float& prob2Q,
                                         bool resolve,
                                         float& dchisq,
                                         SiPixelTemplate2D& templ2D) {
  // Local variables
  const bool deadpix = false;
  std::vector<std::pair<int, int> > zeropix;
  const int speed = 1;

  return SiPixelTemplateSplit::PixelTempSplit(id,
                                              cotalpha,
                                              cotbeta,
                                              cluster,
                                              ydouble,
                                              xdouble,
                                              templ,
                                              yrec1,
                                              yrec2,
                                              sigmay,
                                              prob2y,
                                              xrec1,
                                              xrec2,
                                              sigmax,
                                              prob2x,
                                              q2bin,
                                              prob2Q,
                                              resolve,
                                              speed,
                                              dchisq,
                                              deadpix,
                                              zeropix,
                                              templ2D);

}  // PixelTempSplit

// *************************************************************************************************************************************
//! Reconstruct the best estimate of the hit positions for pixel clusters. Legacy interface to older code.
//! \param         id - (input) identifier of the template to use
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param    cluster - (input) boost multi_array container of 7x21 array of pixel signals,
//!           origin of local coords (0,0) at center of pixel cluster[0][0].
//! \param    ydouble - (input) STL vector of 21 element array to flag a double-pixel
//! \param    xdouble - (input) STL vector of 7 element array to flag a double-pixel
//! \param      templ - (input) the template used in the reconstruction
//! \param      yrec1 - (output) best estimate of first y-coordinate of hit in microns
//! \param      yrec2 - (output) best estimate of second y-coordinate of hit in microns
//! \param     sigmay - (output) best estimate of uncertainty on yrec in microns
//! \param      prob2y - (output) probability describing goodness-of-fit for y-reco
//! \param      xrec1 - (output) best estimate of first x-coordinate of hit in microns
//! \param      xrec2 - (output) best estimate of second x-coordinate of hit in microns
//! \param     sigmax - (output) best estimate of uncertainty on xrec in microns
//! \param      prob2x - (output) probability describing goodness-of-fit for x-reco
//! \param       q2bin - (output) index (0-4) describing the charge of the cluster
//!                     [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin]
//! \param      prob2Q - (output) probability that the cluster charge is compatible with a 2-hit merging
// *************************************************************************************************************************************

int SiPixelTemplateSplit::PixelTempSplit(int id,
                                         float cotalpha,
                                         float cotbeta,
                                         array_2d& cluster,
                                         std::vector<bool>& ydouble,
                                         std::vector<bool>& xdouble,
                                         SiPixelTemplate& templ,
                                         float& yrec1,
                                         float& yrec2,
                                         float& sigmay,
                                         float& prob2y,
                                         float& xrec1,
                                         float& xrec2,
                                         float& sigmax,
                                         float& prob2x,
                                         int& q2bin,
                                         float& prob2Q,
                                         SiPixelTemplate2D& templ2D) {
  // Local variables
  const bool deadpix = false;
  const bool resolve = true;
  float dchisq;
  std::vector<std::pair<int, int> > zeropix;
  const int speed = 1;

  return SiPixelTemplateSplit::PixelTempSplit(id,
                                              cotalpha,
                                              cotbeta,
                                              cluster,
                                              ydouble,
                                              xdouble,
                                              templ,
                                              yrec1,
                                              yrec2,
                                              sigmay,
                                              prob2y,
                                              xrec1,
                                              xrec2,
                                              sigmax,
                                              prob2x,
                                              q2bin,
                                              prob2Q,
                                              resolve,
                                              speed,
                                              dchisq,
                                              deadpix,
                                              zeropix,
                                              templ2D);

}  // PixelTempSplit

// *************************************************************************************************************************************
//! Reconstruct the best estimate of the hit positions for pixel clusters. Legacy interface to older code.
//! \param         id - (input) identifier of the template to use
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param    cluster - (input) boost multi_array container of 7x21 array of pixel signals,
//!           origin of local coords (0,0) at center of pixel cluster[0][0].
//! \param    ydouble - (input) STL vector of 21 element array to flag a double-pixel
//! \param    xdouble - (input) STL vector of 7 element array to flag a double-pixel
//! \param      templ - (input) the template used in the reconstruction
//! \param      yrec1 - (output) best estimate of first y-coordinate of hit in microns
//! \param      yrec2 - (output) best estimate of second y-coordinate of hit in microns
//! \param     sigmay - (output) best estimate of uncertainty on yrec in microns
//! \param      prob2y - (output) probability describing goodness-of-fit for y-reco
//! \param      xrec1 - (output) best estimate of first x-coordinate of hit in microns
//! \param      xrec2 - (output) best estimate of second x-coordinate of hit in microns
//! \param     sigmax - (output) best estimate of uncertainty on xrec in microns
//! \param      prob2x - (output) probability describing goodness-of-fit for x-reco
//! \param       q2bin - (output) index (0-4) describing the charge of the cluster
//!                     [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin]
// *************************************************************************************************************************************

int SiPixelTemplateSplit::PixelTempSplit(int id,
                                         float cotalpha,
                                         float cotbeta,
                                         array_2d& cluster,
                                         std::vector<bool>& ydouble,
                                         std::vector<bool>& xdouble,
                                         SiPixelTemplate& templ,
                                         float& yrec1,
                                         float& yrec2,
                                         float& sigmay,
                                         float& prob2y,
                                         float& xrec1,
                                         float& xrec2,
                                         float& sigmax,
                                         float& prob2x,
                                         int& q2bin,
                                         SiPixelTemplate2D& templ2D) {
  // Local variables
  const bool deadpix = false;
  const bool resolve = true;
  float dchisq, prob2Q;
  std::vector<std::pair<int, int> > zeropix;
  const int speed = 1;

  return SiPixelTemplateSplit::PixelTempSplit(id,
                                              cotalpha,
                                              cotbeta,
                                              cluster,
                                              ydouble,
                                              xdouble,
                                              templ,
                                              yrec1,
                                              yrec2,
                                              sigmay,
                                              prob2y,
                                              xrec1,
                                              xrec2,
                                              sigmax,
                                              prob2x,
                                              q2bin,
                                              prob2Q,
                                              resolve,
                                              speed,
                                              dchisq,
                                              deadpix,
                                              zeropix,
                                              templ2D);

}  // PixelTempSplit
