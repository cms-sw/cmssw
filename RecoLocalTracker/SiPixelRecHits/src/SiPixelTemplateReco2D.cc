//
//  SiPixelTemplateReco2D.cc (Version 2.90)
//  Updated to work with the 2D template generation code
//  Include all bells and whistles for edge clusters
//  2.10 - Add y-lorentz drift to estimate starting point [for FPix]
//  2.10 - Remove >1 pixel requirement
//  2.20 - Fix major bug, change chi2 scan to 9x5 [YxX]
//  2.30 - Allow one pixel clusters, improve cosmetics for increased style points from judges
//  2.50 - Add variable cluster shifting to make the position parameter space more symmetric,
//         also fix potential problems with variable size input clusters and double pixel flags
//  2.55 - Fix another double pixel flag problem and a small pseudopixel problem in the edgegflagy = 3 case.
//  2.60 - Modify the algorithm to return the point with the best chi2 from the starting point scan when
//         the iterative procedure does not converge [eg 1 pixel clusters]
//  2.70 - Change convergence criterion to require it in both planes [it was either]
//  2.80 - Change 3D to 2D
//  2.90 - Fix divide by zero for separate 1D convergence branch
//  3.00 - Change VVIObjF so it only reads kappa
//
//
//
//  Created by Morris Swartz on 7/13/17.
//  Simplification of VVIObjF by Tamas Vami
//

#ifdef SI_PIXEL_TEMPLATE_STANDALONE
#include <math.h>
#endif
#include <algorithm>
#include <vector>
#include <utility>
#include <iostream>
// ROOT::Math has a c++ function that does the probability calc, but only in v5.12 and later
#include "TMath.h"
#include "Math/DistFunc.h"

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco2D.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/VVIObjF.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) edm::LogError(x)
#define LOGDEBUG(x) LogDebug(x)
#define ENDL " "
#include "FWCore/Utilities/interface/Exception.h"
#else
#include "SiPixelTemplateReco2D.h"
#include "VVIObjF.h"
#define LOGERROR(x) std::cout << x << ": "
#define LOGDEBUG(x) std::cout << x << ": "
#define ENDL std::endl
#endif

using namespace SiPixelTemplateReco2D;

// *************************************************************************************************************************************
//! Reconstruct the best estimate of the hit position for pixel clusters.
//! \param         id - (input) identifier of the template to use
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param locBz - (input) the sign of this quantity is used to determine whether to flip cot(beta)<0 quantities from cot(beta)>0 (FPix only)
//!                    for Phase 0 FPix IP-related tracks, locBz < 0 for cot(beta) > 0 and locBz > 0 for cot(beta) < 0
//!                    for Phase 1 FPix IP-related tracks, see next comment
//! \param locBx - (input) the sign of this quantity is used to determine whether to flip cot(alpha/beta)<0 quantities from cot(alpha/beta)>0 (FPix only)
//!                    for Phase 1 FPix IP-related tracks, locBx/locBz > 0 for cot(alpha) > 0 and locBx/locBz < 0 for cot(alpha) < 0
//!                    for Phase 1 FPix IP-related tracks, locBx > 0 for cot(beta) > 0 and locBx < 0 for cot(beta) < 0
//! \param   edgeflagy - (input) flag to indicate the present of edges in y: 0-none (or interior gap), 1-edge at small y, 2-edge at large y,
//!                                                                          3-edge at either end
//! \param   edgeflagx - (input) flag to indicate the present of edges in x: 0-none, 1-edge at small x, 2-edge at large x
//! \param    cluster - (input) pixel array struct with double pixel flags and bounds
//!           origin of local coords (0,0) at center of pixel cluster[0][0].
//! \param    templ2D - (input) the 2D template used in the reconstruction
//! \param       yrec - (output) best estimate of y-coordinate of hit in microns
//! \param     sigmay - (output) best estimate of uncertainty on yrec in microns
//! \param       xrec - (output) best estimate of x-coordinate of hit in microns
//! \param     sigmax - (output) best estimate of uncertainty on xrec in microns
//! \param     probxy - (output) probability describing goodness-of-fit
//! \param     probQ  - (output) probability describing upper cluster charge tail
//! \param       qbin - (output) index (0-4) describing the charge of the cluster
//!                       qbin = 0        Q/Q_avg > 1.5   [few % of all hits]
//!                              1  1.5 > Q/Q_avg > 1.0   [~30% of all hits]
//!                              2  1.0 > Q/Q_avg > 0.85  [~30% of all hits]
//!                              3 0.85 > Q/Q_avg > min1  [~30% of all hits]
//! \param     deltay - (output) template y-length - cluster length [when > 0, possibly missing end]
// *************************************************************************************************************************************
int SiPixelTemplateReco2D::PixelTempReco2D(int id,
                                           float cotalpha,
                                           float cotbeta,
                                           float locBz,
                                           float locBx,
                                           int edgeflagy,
                                           int edgeflagx,
                                           ClusMatrix& cluster,
                                           SiPixelTemplate2D& templ2D,
                                           float& yrec,
                                           float& sigmay,
                                           float& xrec,
                                           float& sigmax,
                                           float& probxy,
                                           float& probQ,
                                           int& qbin,
                                           float& deltay,
                                           int& npixels)

{
  // Local variables

  float template2d[BXM2][BYM2], dpdx2d[2][BXM2][BYM2], fbin[3];

  // fraction of truncation signal to measure the cluster ends
  const float fracpix = 0.45f;

  const int nilist = 9, njlist = 5;
  const float ilist[nilist] = {0.f, -1.f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f, 1.f};
  const float jlist[njlist] = {0.f, -0.5f, -0.25f, 0.25f, 0.50f};

  // Extract some relevant info from the 2D template

  if (id >= 0) {  // if id < 0, bypass interpolation (used in calibration)
    if (!templ2D.interpolate(id, cotalpha, cotbeta, locBz, locBx))
      return 4;
  }
  float xsize = templ2D.xsize();
  float ysize = templ2D.ysize();

  // Allow Qbin Q/Q_avg fractions to vary to optimize error estimation

  for (int i = 0; i < 3; ++i) {
    fbin[i] = templ2D.fbin(i);
  }

  float q50 = templ2D.s50();
  float pseudopix = 0.2f * q50;
  float pseudopix2 = q50 * q50;

  // Get charge scaling factor

  float qscale = templ2D.qscale();

  // Check that the cluster container is (up to) a 7x21 matrix and matches the dimensions of the double pixel flags

  int nclusx = cluster.mrow;
  int nclusy = (int)cluster.mcol;
  bool* xdouble = cluster.xdouble;
  bool* ydouble = cluster.ydouble;

  // First, rescale all pixel charges and compute total charge
  float qtotal = 0.f;
  int imin = BYM2, imax = 0, jmin = BXM2, jmax = 0;
  for (int k = 0; k < nclusx * nclusy; ++k) {
    cluster.matrix[k] *= qscale;
    qtotal += cluster.matrix[k];
    int j = k / nclusy;
    int i = k - j * nclusy;
    if (cluster.matrix[k] > 0.f) {
      if (j < jmin) {
        jmin = j;
      }
      if (j > jmax) {
        jmax = j;
      }
      if (i < imin) {
        imin = i;
      }
      if (i > imax) {
        imax = i;
      }
    }
  }

  //  Calculate the shifts needed to center the cluster in the buffer

  int shiftx = THXP1 - (jmin + jmax) / 2;
  int shifty = THYP1 - (imin + imax) / 2;
  //  Always shift by at least one pixel
  if (shiftx < 1)
    shiftx = 1;
  if (shifty < 1)
    shifty = 1;
  //  Shift the min maxes too
  jmin += shiftx;
  jmax += shiftx;
  imin += shifty;
  imax += shifty;

  // uncertainty and final corrections depend upon total charge bin

  float fq0 = qtotal / templ2D.qavg();

  // Next, copy the cluster and double pixel flags into a shifted workspace to
  // allow for significant zeros [pseudopixels] to be added

  float clusxy[BXM2][BYM2];
  for (int j = 0; j < BXM2; ++j) {
    for (int i = 0; i < BYM2; ++i) {
      clusxy[j][i] = 0.f;
    }
  }

  const unsigned int NPIXMAX = 200;

  int indexxy[2][NPIXMAX];
  float pixel[NPIXMAX];
  float sigma2[NPIXMAX];
  float minmax = templ2D.pixmax();
  float ylow0 = 0.f, yhigh0 = 0.f, xlow0 = 0.f, xhigh0 = 0.f;
  int npixel = 0;
  float ysum[BYM2], ye[BYM2 + 1], ypos[BYM2], xpos[BXM2], xe[BXM2 + 1];
  bool yd[BYM2], xd[BXM2];

  //  Copy double pixel flags to shifted arrays

  for (int i = 0; i < BYM2; ++i) {
    ysum[i] = 0.f;
    yd[i] = false;
  }
  for (int j = 0; j < BXM2; ++j) {
    xd[j] = false;
  }
  for (int i = 0; i < nclusy; ++i) {
    if (ydouble[i]) {
      int iy = i + shifty;
      if (iy > -1 && iy < BYM2)
        yd[iy] = true;
    }
  }
  for (int j = 0; j < nclusx; ++j) {
    if (xdouble[j]) {
      int jx = j + shiftx;
      if (jx > -1 && jx < BXM2)
        xd[jx] = true;
    }
  }
  // Work out the positions of the rows+columns relative to the lower left edge of pixel[1,1]
  xe[0] = -xsize;
  ye[0] = -ysize;
  for (int i = 0; i < BYM2; ++i) {
    float ypitch = ysize;
    if (yd[i]) {
      ypitch += ysize;
    }
    ye[i + 1] = ye[i] + ypitch;
    ypos[i] = ye[i] + ypitch / 2.;
  }
  for (int j = 0; j < BXM2; ++j) {
    float xpitch = xsize;
    if (xd[j]) {
      xpitch += xsize;
    }
    xe[j + 1] = xe[j] + xpitch;
    xpos[j] = xe[j] + xpitch / 2.;
  }
  // Shift the cluster center to the central pixel of the array, truncate big signals
  for (int i = 0; i < nclusy; ++i) {
    int iy = i + shifty;
    float maxpix = minmax;
    if (ydouble[i]) {
      maxpix *= 2.f;
    }
    if (iy > -1 && iy < BYM2) {
      for (int j = 0; j < nclusx; ++j) {
        int jx = j + shiftx;
        if (jx > -1 && jx < BXM2) {
          if (cluster(j, i) > maxpix) {
            clusxy[jx][iy] = maxpix;
          } else {
            clusxy[jx][iy] = cluster(j, i);
          }
          if (clusxy[jx][iy] > 0.f) {
            ysum[iy] += clusxy[jx][iy];
            indexxy[0][npixel] = jx;
            indexxy[1][npixel] = iy;
            pixel[npixel] = clusxy[jx][iy];
            ++npixel;
          }
        }
      }
    }
  }

  // Make sure that we find at least one pixel
  if (npixel < 1) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    throw cms::Exception("DataCorrupt") << "PixelTemplateReco2D::number of pixels above threshold = " << npixel
                                        << std::endl;
#else
    std::cout << "PixelTemplateReco2D::number of pixels above threshold = " << npixel << std::endl;
#endif
    return 1;
  }

  // Get the shifted coordinates of the cluster ends
  xlow0 = xe[jmin];
  xhigh0 = xe[jmax + 1];
  ylow0 = ye[imin];
  yhigh0 = ye[imax + 1];

  // Next, calculate the error^2 [need to know which is the middle y pixel of the cluster]

  int ypixoff = T2HYP1 - (imin + imax) / 2;
  for (int k = 0; k < npixel; ++k) {
    int ypixeff = ypixoff + indexxy[1][k];
    templ2D.xysigma2(pixel[k], ypixeff, sigma2[k]);
  }

  // Next, find the half-height edges of the y-projection and identify any
  // missing columns to remove from the fit

  int imisslow = -1, imisshigh = -1, jmisslow = -1, jmisshigh = -1;
  float ylow = -1.f, yhigh = -1.f;
  float hmaxpix = fracpix * templ2D.sxymax();
  for (int i = imin; i <= imax; ++i) {
    if (ysum[i] > hmaxpix && ysum[i - 1] < hmaxpix && ylow < 0.f) {
      ylow = ypos[i - 1] + (ypos[i] - ypos[i - 1]) * (hmaxpix - ysum[i - 1]) / (ysum[i] - ysum[i - 1]);
    }
    if (ysum[i] < q50) {
      if (imisslow < 0)
        imisslow = i;
      imisshigh = i;
    }
    if (ysum[i] > hmaxpix && ysum[i + 1] < hmaxpix) {
      yhigh = ypos[i] + (ypos[i + 1] - ypos[i]) * (ysum[i] - hmaxpix) / (ysum[i] - ysum[i + 1]);
    }
  }
  if (ylow < 0.f || yhigh < 0.f) {
    ylow = ylow0;
    yhigh = yhigh0;
  }

  float templeny = templ2D.clsleny();
  deltay = templeny - (yhigh - ylow) / ysize;

  //  x0 and y0 are best guess seeds for the fit

  float x0 = 0.5f * (xlow0 + xhigh0) - templ2D.lorxdrift();
  float y0 = 0.5f * (ylow + yhigh) - templ2D.lorydrift();
  //   float y1 = yhigh - halfy - templ2D.lorydrift();
  //   printf("y0 = %f, y1 = %f \n", y0, y1);
  //   float y0 = 0.5f*(ylow + yhigh);

  // If there are missing edge columns, set up missing column flags and number
  // of minimization passes

  int npass = 1;

  switch (edgeflagy) {
    case 0:
      break;
    case 1:
      imisshigh = imin - 1;
      imisslow = -1;
      break;
    case 2:
      imisshigh = -1;
      imisslow = imax + 1;
      break;
    case 3:
      imisshigh = imin - 1;
      imisslow = -1;
      npass = 2;
      break;
    default:
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
      throw cms::Exception("DataCorrupt") << "PixelTemplateReco2D::illegal edgeflagy = " << edgeflagy << std::endl;
#else
      std::cout << "PixelTemplate:2D:illegal edgeflagy = " << edgeflagy << std::endl;
#endif
  }

  switch (edgeflagx) {
    case 0:
      break;
    case 1:
      jmisshigh = jmin - 1;
      jmisslow = -1;
      break;
    case 2:
      jmisshigh = -1;
      jmisslow = jmax + 1;
      break;
    default:
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
      throw cms::Exception("DataCorrupt") << "PixelTemplateReco2D::illegal edgeflagx = " << edgeflagx << std::endl;
#else
      std::cout << "PixelTemplate:2D:illegal edgeflagx = " << edgeflagx << std::endl;
#endif
  }

  // Define quantities to be saved for each pass

  float chi2min[2], xerr2[2], yerr2[2];
  float x2D0[2], y2D0[2], qtfrac0[2];
  int ipass, tpixel;
  //int niter0[2];

  for (ipass = 0; ipass < npass; ++ipass) {
    if (ipass == 1) {
      // Now try again if both edges are possible

      imisshigh = -1;
      imisslow = imax + 1;
      // erase pseudo pixels from previous pass
      for (int k = npixel; k < tpixel; ++k) {
        int j = indexxy[0][k];
        int i = indexxy[1][k];
        clusxy[j][i] = 0.f;
      }
    }

    // Next, add pseudo pixels around the periphery of the cluster

    tpixel = npixel;
    for (int k = 0; k < npixel; ++k) {
      int j = indexxy[0][k];
      int i = indexxy[1][k];
      if ((j - 1) != jmisshigh) {
        if (clusxy[j - 1][i] < pseudopix) {
          indexxy[0][tpixel] = j - 1;
          indexxy[1][tpixel] = i;
          clusxy[j - 1][i] = pseudopix;
          pixel[tpixel] = pseudopix;
          sigma2[tpixel] = pseudopix2;
          ++tpixel;
        }
      }
      if ((j + 1) != jmisslow) {
        if (clusxy[j + 1][i] < pseudopix) {
          indexxy[0][tpixel] = j + 1;
          indexxy[1][tpixel] = i;
          clusxy[j + 1][i] = pseudopix;
          pixel[tpixel] = pseudopix;
          sigma2[tpixel] = pseudopix2;
          ++tpixel;
        }
      }
      // Don't add them if this is a dead column
      if ((i + 1) != imisslow) {
        if ((j - 1) != jmisshigh) {
          if (clusxy[j - 1][i + 1] < pseudopix) {
            indexxy[0][tpixel] = j - 1;
            indexxy[1][tpixel] = i + 1;
            clusxy[j - 1][i + 1] = pseudopix;
            pixel[tpixel] = pseudopix;
            sigma2[tpixel] = pseudopix2;
            ++tpixel;
          }
        }
        if (clusxy[j][i + 1] < pseudopix) {
          indexxy[0][tpixel] = j;
          indexxy[1][tpixel] = i + 1;
          clusxy[j][i + 1] = pseudopix;
          pixel[tpixel] = pseudopix;
          sigma2[tpixel] = pseudopix2;
          ++tpixel;
        }
        if ((j + 1) != jmisslow) {
          if (clusxy[j + 1][i + 1] < pseudopix) {
            indexxy[0][tpixel] = j + 1;
            indexxy[1][tpixel] = i + 1;
            clusxy[j + 1][i + 1] = pseudopix;
            pixel[tpixel] = pseudopix;
            sigma2[tpixel] = pseudopix2;
            ++tpixel;
          }
        }
      }
      // Don't add them if this is a dead column
      if ((i - 1) != imisshigh) {
        if ((j - 1) != jmisshigh) {
          if (clusxy[j - 1][i - 1] < pseudopix) {
            indexxy[0][tpixel] = j - 1;
            indexxy[1][tpixel] = i - 1;
            clusxy[j - 1][i - 1] = pseudopix;
            pixel[tpixel] = pseudopix;
            sigma2[tpixel] = pseudopix2;
            ++tpixel;
          }
        }
        if (clusxy[j][i - 1] < pseudopix) {
          indexxy[0][tpixel] = j;
          indexxy[1][tpixel] = i - 1;
          clusxy[j][i - 1] = pseudopix;
          pixel[tpixel] = pseudopix;
          sigma2[tpixel] = pseudopix2;
          ++tpixel;
        }
        if ((j + 1) != jmisslow) {
          if (clusxy[j + 1][i - 1] < pseudopix) {
            indexxy[0][tpixel] = j + 1;
            indexxy[1][tpixel] = i - 1;
            clusxy[j + 1][i - 1] = pseudopix;
            pixel[tpixel] = pseudopix;
            sigma2[tpixel] = pseudopix2;
            ++tpixel;
          }
        }
      }
    }

    // Calculate chi2 over a grid of several seeds and choose the smallest

    chi2min[ipass] = 1000000.f;
    float chi2, qtemplate, qactive, qtfrac = 0.f, x2D = 0.f, y2D = 0.f;
    //  Scale the y search grid for long clusters [longer than 7 pixels]
    float ygridscale = 0.271 * cotbeta;
    if (ygridscale < 1.f)
      ygridscale = 1.f;
    for (int is = 0; is < nilist; ++is) {
      for (int js = 0; js < njlist; ++js) {
        float xtry = x0 + jlist[js] * xsize;
        float ytry = y0 + ilist[is] * ygridscale * ysize;
        chi2 = 0.f;
        qactive = 0.f;
        for (int j = 0; j < BXM2; ++j) {
          for (int i = 0; i < BYM2; ++i) {
            template2d[j][i] = 0.f;
          }
        }
        templ2D.xytemp(xtry, ytry, yd, xd, template2d, false, dpdx2d, qtemplate);
        for (int k = 0; k < tpixel; ++k) {
          int jpix = indexxy[0][k];
          int ipix = indexxy[1][k];
          float qtpixel = template2d[jpix][ipix];
          float pt = pixel[k] - qtpixel;
          chi2 += pt * pt / sigma2[k];
          if (k < npixel) {
            qactive += qtpixel;
          }
        }
        if (chi2 < chi2min[ipass]) {
          chi2min[ipass] = chi2;
          x2D = xtry;
          y2D = ytry;
          qtfrac = qactive / qtemplate;
        }
      }
    }

    int niter = 0;
    float xstep = 1.0f, ystep = 1.0f;
    float minv11 = 1000.f, minv12 = 1000.f, minv22 = 1000.f;
    chi2 = chi2min[ipass];
    while (chi2 <= chi2min[ipass] && niter < 15 && (niter < 2 || (std::abs(xstep) > 0.2 || std::abs(ystep) > 0.2))) {
      // Remember the present parameters
      x2D0[ipass] = x2D;
      y2D0[ipass] = y2D;
      qtfrac0[ipass] = qtfrac;
      xerr2[ipass] = minv11;
      yerr2[ipass] = minv22;
      chi2min[ipass] = chi2;
      //niter0[ipass] = niter;

      // Calculate the initial template which also allows the error calculation for the struck pixels

      for (int j = 0; j < BXM2; ++j) {
        for (int i = 0; i < BYM2; ++i) {
          template2d[j][i] = 0.f;
        }
      }
      templ2D.xytemp(x2D, y2D, yd, xd, template2d, true, dpdx2d, qtemplate);

      float sumptdt1 = 0., sumptdt2 = 0.;
      float sumdtdt11 = 0., sumdtdt12 = 0., sumdtdt22 = 0.;
      chi2 = 0.f;
      qactive = 0.f;
      // Loop over all pixels and calculate matrices

      for (int k = 0; k < tpixel; ++k) {
        int jpix = indexxy[0][k];
        int ipix = indexxy[1][k];
        float qtpixel = template2d[jpix][ipix];
        float pt = pixel[k] - qtpixel;
        float dtdx = dpdx2d[0][jpix][ipix];
        float dtdy = dpdx2d[1][jpix][ipix];
        chi2 += pt * pt / sigma2[k];
        sumptdt1 += pt * dtdx / sigma2[k];
        sumptdt2 += pt * dtdy / sigma2[k];
        sumdtdt11 += dtdx * dtdx / sigma2[k];
        sumdtdt12 += dtdx * dtdy / sigma2[k];
        sumdtdt22 += dtdy * dtdy / sigma2[k];
        if (k < npixel) {
          qactive += qtpixel;
        }
      }

      // invert the parameter covariance matrix

      float D = sumdtdt11 * sumdtdt22 - sumdtdt12 * sumdtdt12;

      // If the matrix is non-singular invert and solve

      if (std::abs(D) > 1.e-3) {
        minv11 = sumdtdt22 / D;
        minv12 = -sumdtdt12 / D;
        minv22 = sumdtdt11 / D;

        qtfrac = qactive / qtemplate;

        //Calculate the step size

        xstep = minv11 * sumptdt1 + minv12 * sumptdt2;
        ystep = minv12 * sumptdt1 + minv22 * sumptdt2;

      } else {
        //  Assume alternately that ystep = 0 and then xstep = 0
        if (sumdtdt11 > 0.0001f) {
          xstep = sumptdt1 / sumdtdt11;
        } else {
          xstep = 0.f;
        }
        if (sumdtdt22 > 0.0001f) {
          ystep = sumptdt2 / sumdtdt22;
        } else {
          ystep = 0.f;
        }
      }
      xstep *= 0.9f;
      ystep *= 0.9f;
      if (std::abs(xstep) > 2. * xsize || std::abs(ystep) > 2. * ysize)
        break;
      x2D += xstep;
      y2D += ystep;
      ++niter;
    }
  }

  ipass = 0;

  if (npass > 1) {
    // two passes, take smaller chisqared
    if (chi2min[1] < chi2min[0]) {
      ipass = 1;
    }
  }

  // Correct the charge ratio for missing pixels

  float fq;
  if (qtfrac0[ipass] < 0.10f || qtfrac0[ipass] > 1.f) {
    qtfrac0[ipass] = 1.f;
  }
  fq = fq0 / qtfrac0[ipass];

  //   printf("qtfrac0 = %f \n", qtfrac0);

  if (fq > fbin[0]) {
    qbin = 0;
  } else {
    if (fq > fbin[1]) {
      qbin = 1;
    } else {
      if (fq > fbin[2]) {
        qbin = 2;
      } else {
        qbin = 3;
      }
    }
  }

  // Get charge related quantities

  float scalex = templ2D.scalex(qbin);
  float scaley = templ2D.scaley(qbin);
  float offsetx = templ2D.offsetx(qbin);
  float offsety = templ2D.offsety(qbin);

  // This 2D code has the origin (0,0) at the lower left edge of the input cluster
  // That is now pixel [shiftx,shifty] and the template reco convention is the middle
  // of that pixel, so we need to correct

  xrec = x2D0[ipass] - xpos[shiftx] - offsetx;
  yrec = y2D0[ipass] - ypos[shifty] - offsety;
  if (xerr2[ipass] > 0.f) {
    sigmax = scalex * sqrt(xerr2[ipass]);
    if (sigmax < 3.f)
      sigmax = 3.f;
  } else {
    sigmax = 10000.f;
  }
  if (yerr2[ipass] > 0.f) {
    sigmay = scaley * sqrt(yerr2[ipass]);
    if (sigmay < 3.f)
      sigmay = 3.f;
  } else {
    sigmay = 10000.f;
  }
  if (id >= 0) {  //if id < 0 bypass interpolation (used in calibration)
    // Do chi2 probability calculation
    double meanxy = (double)(npixel * templ2D.chi2ppix());
    double chi2scale = (double)templ2D.chi2scale();
    if (meanxy < 0.01) {
      meanxy = 0.01;
    }
    double hndof = meanxy / 2.f;
    double hchi2 = chi2scale * chi2min[ipass] / 2.f;
    probxy = (float)(1. - TMath::Gamma(hndof, hchi2));
    // Do the charge probability
    float mpv = templ2D.mpvvav();
    float sigmaQ = templ2D.sigmavav();
    float kappa = templ2D.kappavav();
    float xvav = (qtotal / qtfrac0[ipass] - mpv) / sigmaQ;
    //  VVIObj is a private port of CERNLIB VVIDIS
    VVIObjF vvidist(kappa);
    float prvav = vvidist.fcn(xvav);
    probQ = 1.f - prvav;
  } else {
    probxy = chi2min[ipass];
    npixels = npixel;
    probQ = 0.f;
  }

  return 0;
}  // PixelTempReco2D

int SiPixelTemplateReco2D::PixelTempReco2D(int id,
                                           float cotalpha,
                                           float cotbeta,
                                           float locBz,
                                           float locBx,
                                           int edgeflagy,
                                           int edgeflagx,
                                           ClusMatrix& cluster,
                                           SiPixelTemplate2D& templ2D,
                                           float& yrec,
                                           float& sigmay,
                                           float& xrec,
                                           float& sigmax,
                                           float& probxy,
                                           float& probQ,
                                           int& qbin,
                                           float& deltay)

{
  // Local variables
  int npixels;
  return SiPixelTemplateReco2D::PixelTempReco2D(id,
                                                cotalpha,
                                                cotbeta,
                                                locBz,
                                                locBx,
                                                edgeflagy,
                                                edgeflagx,
                                                cluster,
                                                templ2D,
                                                yrec,
                                                sigmay,
                                                xrec,
                                                sigmax,
                                                probxy,
                                                probQ,
                                                qbin,
                                                deltay,
                                                npixels);
}  // PixelTempReco2D
