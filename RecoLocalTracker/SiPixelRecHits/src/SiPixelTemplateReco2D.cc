//
//  SiPixelTemplateReco2D.cc (Version 2.20)
//  Updated to work with the 2D template generation code
//  Include all bells and whistles for edge clusters
//  2.10 - Add y-lorentz drift to estimate starting point [for FPix]
//  2.10 - Remove >1 pixel requirement
//  2.20 - Fix major bug, change chi2 scan to 9x5 [YxX]


//
//
//  Created by Morris Swartz on 7/13/17.
//
//

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
//#include <cmath.h>
#else
#include <math.h>
#endif
#include <algorithm>
#include <vector>
#include <utility>
#include <iostream>
// ROOT::Math has a c++ function that does the probability calc, but only in v5.12 and later
#include "TMath.h"
#include "Math/DistFunc.h"
// Use current version of gsl instead of ROOT::Math
//#include <gsl/gsl_cdf.h>

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco2D.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/VVIObjF.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) edm::LogError(x)
#define LOGDEBUG(x) LogDebug(x)
static const int theVerboseLevel = 2;
#define ENDL " "
#include "FWCore/Utilities/interface/Exception.h"
#else
#include "SiPixelTemplateReco2D.h"
#include "VVIObjF.h"
//static int theVerboseLevel = {2};
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
int SiPixelTemplateReco2D::PixelTempReco3D(int id, float cotalpha, float cotbeta, float locBz, float locBx, int edgeflagy, int edgeflagx,
                                           ClusMatrix & cluster, SiPixelTemplate2D& templ2D,float& yrec, float& sigmay,
                                           float& xrec, float& sigmax, float& probxy, float& probQ, int& qbin, float& deltay, int& npixels)

{
   // Local variables
   int i, j, k;
   float template2d[BXM2][BYM2], dpdx2d[2][BXM2][BYM2], fbin[3];
   
   // fraction of truncation signal to measure the cluster ends
   const float fracpix = 0.45f;
   
   
   // Extract some relevant info from the 2D template
   
   if(id > 0) {templ2D.interpolate(id, cotalpha, cotbeta, locBz, locBx);}
   float xsize = templ2D.xsize();
   float ysize = templ2D.ysize();
   
   // Allow Qbin Q/Q_avg fractions to vary to optimize error estimation
   
   for(i=0; i<3; ++i) {fbin[i] = templ2D.fbin(i);}
   
   float q50 = templ2D.s50();
   float pseudopix = 0.1f*q50;
   float pseudopix2 = q50*q50;
   
   
   // Get charge scaling factor
   
   float qscale = templ2D.qscale();
   
   // Check that the cluster container is (up to) a 7x21 matrix and matches the dimensions of the double pixel flags
   
   int nclusx = cluster.mrow;
   int nclusy = (int)cluster.mcol;
   bool * xdouble = cluster.xdouble;
   bool * ydouble = cluster.ydouble;
   
   // First, rescale all pixel charges and compute total charge
   float qtotal = 0.f;
   for(i=0; i<nclusx*nclusy; ++i) {
      cluster.matrix[i] *= qscale;
      qtotal +=cluster.matrix[i];
   }
   
   // uncertainty and final corrections depend upon total charge bin
   
   float fq0 = qtotal/templ2D.qavg();
   
   // Next, copy the cluster and double pixel flags into a shifted workspace to
   // allow for significant zeros [pseudopixels] to be added
   
   float clusxy[BXM2][BYM2];
   for(j=0; j<BXM2; ++j) {for(i=0; i<BYM2; ++i) {clusxy[j][i] = 0.f;}}
   
   int indexxy[2][NPIXMAX];
   float pixel[NPIXMAX];
   float sigma2[NPIXMAX];
   float minmax = templ2D.pixmax();
   int imin=BYM2, imax=0, jmin=BXM2, jmax=0;
   float ylow0 = 0.f, yhigh0 = 0.f, xlow0 = 0.f, xhigh0 = 0.f;
   float ye = 0.f, xe;
   int npixel = 0;
   float ysum[BYM2], ypos[BYM2];
   //   float xq = 0.f, qtot = 0.f;
   
   //Truncate pixels and make a y-projection
   
   ysum[0] = 0.f;
   ypos[0] = -0.5f*ysize;
   for(i=1; i<nclusy+1; ++i) {
      float ypitch = ysize;
      float maxpix = minmax;
      if(ydouble[i-1]) {ypitch += ysize; maxpix *=2.f;}
      ypos[i] = ye + ypitch/2.;
      xe = 0.f;
      ysum[i] = 0.f;
      for(j=1; j<nclusx+1; ++j) {
         float xpitch = xsize;
         if(xdouble[j-1]) {xpitch += xsize;}
         // Truncate pixel charges
         if(cluster(j-1,i-1) > maxpix) {
            clusxy[j][i] = maxpix;
         } else {
            clusxy[j][i] = cluster(j-1,i-1);
         }
         if(clusxy[j][i] > 0.) {
            ysum[i] += clusxy[j][i];
            //          xq += (xe + xpitch/2.)*clusxy[j][i];
            if(j < jmin) {jmin = j; xlow0 = xe;}
            if(j > jmax) {jmax = j; xhigh0 = xe+xpitch;}
            if(i < imin) {imin = i; ylow0 = ye;}
            if(i > imax) {imax = i; yhigh0 = ye+ypitch;}
            indexxy[0][npixel] = j;
            indexxy[1][npixel] = i;
            pixel[npixel] = clusxy[j][i];
            ++npixel;
         }
         xe += xpitch;
      }
      //      qtot += ysum[i];
      ye += ypitch;
   }
   ysum[nclusy+1] = 0.f;
   ypos[nclusy+1] = yhigh0+0.5f*ysize;
   
   //   float xbaryc = xq/qtot;
   
   // Quit if only one pixel in cluster
   
      if(npixel < 2 ) {
//         LOGDEBUG("SiPixelTemplateReco2D") << "2D fit not possible with single pixel" << ENDL;
         return 1;
      }
   
   
   //   if(jmax-jmin == 0 ) {
   //      LOGDEBUG("SiPixelTemplateReco2D") << "2D fit not possible with single x-pixel" << ENDL;
   //      return 1;
   //   }
   
   // Quit if only one pixel in cluster
   
   //   if(imax-imin == 0) {
   //      LOGDEBUG("SiPixelTemplateReco2D") << "2D fit not possible with single y-pixel" << ENDL;
   //      return 1;
   //   }
   
   // Next, calculate the error^2 [need to know which is the middle y pixel of the cluster]
   
   int ypixoff = T2HYP1 - (imin+imax)/2;
   for(k=0; k<npixel; ++k) {
      i = indexxy[1][k];
      int ypixeff = ypixoff + i;
      templ2D.xysigma2(pixel[k], ypixeff, sigma2[k]);
   }
   
   // Next, find the half-height edges of the y-projection and identify any
   // missing columns to remove from the fit
   
   int imisslow = -1, imisshigh = -1, jmisslow = -1, jmisshigh = -1;
   float ylow = -1.f, yhigh = -1.f;
   float hmaxpix = fracpix*templ2D.sxymax();
   for(i=imin; i<=imax; ++i) {
      if(ysum[i] > hmaxpix && ysum[i-1] < hmaxpix && ylow < 0.f) {
         ylow = ypos[i-1] + (ypos[i]-ypos[i-1])*(hmaxpix-ysum[i-1])/(ysum[i]-ysum[i-1]);
      }
      if(ysum[i] < q50) {
         if(imisslow < 0) imisslow = i;
         imisshigh = i;
      }
      if(ysum[i] > hmaxpix && ysum[i+1] < hmaxpix) {
         yhigh = ypos[i] + (ypos[i+1]-ypos[i])*(ysum[i]-hmaxpix)/(ysum[i]-ysum[i+1]);
      }
   }
   if(ylow < 0.f || yhigh < 0.f) {
      ylow = ylow0;
      yhigh = yhigh0;
   }
   
   float templeny = templ2D.clsleny();
   deltay = templeny - (yhigh - ylow)/ysize;
   
   // Calculate the corrected distance from the ylow and yhigh to the cluster center
   
//   float delycor = templ2D.delyavg();
//   float halfy = 0.5f*(templeny-delycor)*ysize;
   
//   printf("templeny = %f, deltay = %f, delycor = %f, halfy = %f \n", templeny, deltay, delycor, halfy);
   
   //  x0 and y0 are best guess seeds for the fit
   
   //   float x0 = 0.5f*(xlow0 + xhigh0) + 0.5f*xsize;
   float x0 = 0.5f*(xlow0 + xhigh0) - templ2D.lorxdrift();
   float y0 = 0.5f*(ylow + yhigh) - templ2D.lorydrift();
//   float y1 = yhigh - halfy - templ2D.lorydrift();
//   printf("y0 = %f, y1 = %f \n", y0, y1);
   //   float y0 = 0.5f*(ylow + yhigh);
   
   // If there are missing edge columns, set up missing column flags and number
   // of minimization passes
   
   int npass = 1;
   
   switch(edgeflagy) {
      case 0:
         break;
      case 1:
//         y0 = yhigh - halfy - templ2D.lorydrift();
         imisshigh = imin-1;
         imisslow = -1;
         break;
      case 2:
//         y0 = ylow + halfy - templ2D.lorydrift();
         imisshigh = -1;
         imisslow = imax+1;
         break;
      case 3:
//         y0 = yhigh - halfy - templ2D.lorydrift();
         imisshigh = imin-1;
         imisslow = -1;
         npass = 2;
         break;
      default:
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
         throw cms::Exception("DataCorrupt") << "PixelTemplateReco3D::illegal edgeflagy = " << edgeflagy << std::endl;
#else
         std::cout << "PixelTemplate:3D:illegal edgeflagy = " << edgeflagy << std::endl;
#endif
   }
   
   switch(edgeflagx) {
      case 0:
         break;
      case 1:
         jmisshigh = jmin-1;
         jmisslow = -1;
         break;
      case 2:
         jmisshigh = -1;
         jmisslow = jmax+1;
         break;
      default:
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
         throw cms::Exception("DataCorrupt") << "PixelTemplateReco3D::illegal edgeflagx = " << edgeflagx << std::endl;
#else
         std::cout << "PixelTemplate:3D:illegal edgeflagx = " << edgeflagx << std::endl;
#endif
   }
   
   // Define quantities to be saved for each pass
   
   
   float chi2min[2], xerr2[2], yerr2[2];
   float x2D0[2], y2D0[2], qtfrac0[2];
   int ipass, niter0[2];
   
   for(ipass = 0; ipass < npass; ++ipass) {
      
      if(ipass == 1) {
         
         // Now try again if both edges are possible
         
         //         y0 = ylow + halfy - templ2D.lorydrift();
         imisshigh = -1;
         imisslow = imax+1;
      }
      
      // Next, add pseudo pixels around the periphery of the cluster
      
      int tpixel = npixel;
      for(k=0; k<npixel; ++k) {
         j = indexxy[0][k];
         i = indexxy[1][k];
         if((j-1) != jmisshigh) {
            if(clusxy[j-1][i] < pseudopix) {
               indexxy[0][tpixel] = j-1;
               indexxy[1][tpixel] = i;
               clusxy[j-1][i] = pseudopix;
               pixel[tpixel] = pseudopix;
               sigma2[tpixel] = pseudopix2;
               ++tpixel;
            }
         }
         if((j+1) != jmisslow) {
            if(clusxy[j+1][i] < pseudopix) {
               indexxy[0][tpixel] = j+1;
               indexxy[1][tpixel] = i;
               clusxy[j+1][i] = pseudopix;
               pixel[tpixel] = pseudopix;
               sigma2[tpixel] = pseudopix2;
               ++tpixel;
            }
         }
         // Don't add them if this is a dead column
         if((i+1) != imisslow) {
            if((j-1) != jmisshigh) {
               if(clusxy[j-1][i+1] < pseudopix) {
                  indexxy[0][tpixel] = j-1;
                  indexxy[1][tpixel] = i+1;
                  clusxy[j-1][i+1] = pseudopix;
                  pixel[tpixel] = pseudopix;
                  sigma2[tpixel] = pseudopix2;
                  ++tpixel;
               }
            }
            if(clusxy[j][i+1] < pseudopix) {
               indexxy[0][tpixel] = j;
               indexxy[1][tpixel] = i+1;
               clusxy[j][i+1] = pseudopix;
               pixel[tpixel] = pseudopix;
               sigma2[tpixel] = pseudopix2;
               ++tpixel;
            }
            if((j+1) != jmisslow) {
               if(clusxy[j+1][i+1] < pseudopix) {
                  indexxy[0][tpixel] = j+1;
                  indexxy[1][tpixel] = i+1;
                  clusxy[j+1][i+1] = pseudopix;
                  pixel[tpixel] = pseudopix;
                  sigma2[tpixel] = pseudopix2;
                  ++tpixel;
               }
            }
         }
         // Don't add them if this is a dead column
         if((i-1) != imisshigh) {
            if((j-1) != jmisshigh) {
               if(clusxy[j-1][i-1] < pseudopix) {
                  indexxy[0][tpixel] = j-1;
                  indexxy[1][tpixel] = i-1;
                  clusxy[j-1][i-1] = pseudopix;
                  pixel[tpixel] = pseudopix;
                  sigma2[tpixel] = pseudopix2;
                  ++tpixel;
               }
            }
            if(clusxy[j][i-1] < pseudopix) {
               indexxy[0][tpixel] = j;
               indexxy[1][tpixel] = i-1;
               clusxy[j][i-1] = pseudopix;
               pixel[tpixel] = pseudopix;
               sigma2[tpixel] = pseudopix2;
               ++tpixel;
            }
            if((j+1) != jmisslow) {
               if(clusxy[j+1][i-1] < pseudopix) {
                  indexxy[0][tpixel] = j+1;
                  indexxy[1][tpixel] = i-1;
                  clusxy[j+1][i-1] = pseudopix;
                  pixel[tpixel] = pseudopix;
                  sigma2[tpixel] = pseudopix2;
                  ++tpixel;
               }
            }
         }
      }
      
      
      //   		 printf("** 2D-cluster, cotalpha = %f, cotbeta = %f **\n", cotalpha, cotbeta);
      
      //		 for (k=1; k < BXM3; ++k) {
      //			 printf("%5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f \n",
      //clusxy[k][1],clusxy[k][2],clusxy[k][3],clusxy[k][4],clusxy[k][5],clusxy[k][6],clusxy[k][7],clusxy[k][8],clusxy[k][9],
      //clusxy[k][10],clusxy[k][11],clusxy[k][12],clusxy[k][13],clusxy[k][14],clusxy[k][15],clusxy[k][16],clusxy[k][17],clusxy[k][18],
      //					  clusxy[k][19],clusxy[k][20],clusxy[k][21]);
      //		 }
      
      
      
      // Calculate chi2 over a grid of several seeds and choose the smallest
      
      chi2min[ipass] = 1000000.f;
      float chi2, qtemplate, qactive, qtfrac = 0.f, x2D = 0.f, y2D = 0.f;
      for(int is = -4; is<5; ++is) {
         for(int js = -2; js<3; ++js) {
            float xtry = x0 + js*xsize/4.;
            float ytry = y0 + is*ysize/4.;
//      for(int is = -8; is<9; ++is) {
//         for(int js = -4; js<5; ++js) {
//            float xtry = x0 + js*xsize/8.;
//            float ytry = y0 + is*ysize/8.;
            chi2 = 0.f;
            qactive = 0.f;
            for(j=0; j<BXM2; ++j) {for(i=0; i<BYM2; ++i) {template2d[j][i] = 0.f;}}
            templ2D.xytemp(xtry, ytry, ydouble, xdouble, template2d, false, dpdx2d, qtemplate);
            for(k=0; k<tpixel; ++k) {
               int jpix = indexxy[0][k];
               int ipix = indexxy[1][k];
               float qtpixel = template2d[jpix][ipix];
               float pt = pixel[k]-qtpixel;
               chi2 += pt*pt/sigma2[k];
               if(k < npixel) {qactive += qtpixel;}
            }
            if(chi2 < chi2min[ipass]) {
               chi2min[ipass] = chi2;
               x2D = xtry;
               y2D = ytry;
               qtfrac = qactive/qtemplate;
            }
         }
      }
      //      printf("jsel/isel = %d/%d\n", jsel, isel);
      
      
      //   printf("\n begin iterative minimization \n");

      
      int niter = 0;
      float xstep = 1.0f, ystep = 1.0f;
      float minv11 = 1000.f, minv12 = 1000.f, minv22 = 1000.f;
      //      chi2min[ipass] = 100000.f;
      //      chi2 = 10000.f;
      chi2 = chi2min[ipass];
      while(chi2 <= chi2min[ipass] && niter < 15 && (niter < 2 || (fabs(xstep) > 0.2 && fabs(ystep) > 0.2))) {
         
         // Remember the present parameters
         x2D0[ipass] = x2D;
         y2D0[ipass] = y2D;
         qtfrac0[ipass] = qtfrac;
         xerr2[ipass] = minv11;
         yerr2[ipass] = minv22;
         chi2min[ipass] = chi2;
         niter0[ipass] = niter;
         
         // Calculate the initial template which also allows the error calculation for the struck pixels
         
         for(j=0; j<BXM2; ++j) {for(i=0; i<BYM2; ++i) {template2d[j][i] = 0.f;}}
         templ2D.xytemp(x2D, y2D, ydouble, xdouble, template2d, true, dpdx2d, qtemplate);
         
         float sumptdt1 = 0.f, sumptdt2 = 0.f;
         float sumdtdt11 = 0.f, sumdtdt12 = 0.f, sumdtdt22 = 0.f;
         chi2 = 0.f;
         qactive = 0.f;
         // Loop over all pixels and calculate matrices
         
         for(k=0; k<tpixel; ++k) {
            int jpix = indexxy[0][k];
            int ipix = indexxy[1][k];
            float qtpixel = template2d[jpix][ipix];
            float pt = pixel[k]-qtpixel;
            chi2 += pt*pt/sigma2[k];
            sumptdt1 += pt*dpdx2d[0][jpix][ipix]/sigma2[k];
            sumptdt2 += pt*dpdx2d[1][jpix][ipix]/sigma2[k];
            sumdtdt11 += dpdx2d[0][jpix][ipix]*dpdx2d[0][jpix][ipix]/sigma2[k];
            sumdtdt12 += dpdx2d[0][jpix][ipix]*dpdx2d[1][jpix][ipix]/sigma2[k];
            sumdtdt22 += dpdx2d[1][jpix][ipix]*dpdx2d[1][jpix][ipix]/sigma2[k];
            if(k < npixel) {qactive += qtpixel;}
         }
         
         // invert the parameter covariance matrix
         
         float D = sumdtdt11*sumdtdt22 - sumdtdt12*sumdtdt12;
         minv11 = sumdtdt22/D;
         minv12 = -sumdtdt12/D;
         minv22 = sumdtdt11/D;
         
         qtfrac = qactive/qtemplate;
         
         //Calculate the step size
         
         xstep = minv11*sumptdt1 + minv12*sumptdt2;
         xstep *= 0.9f;
         ystep = minv12*sumptdt1 + minv22*sumptdt2;
         ystep *= 0.9f;
//         printf("niter = %d, chisq = %f, xstep = %f, ystep = %f \n", niter, chi2, xstep, ystep);
         if(fabs(xstep) > 2.*xsize || fabs(ystep) > 2.*ysize) break;
         x2D += xstep;
         y2D += ystep;
         ++niter;
      }
   }
   
   ipass = 0;
   if(npass == 1) {
      // one pass, require that it have iterated
      if(niter0[0] == 0) {return 2;}
   } else {
      // two passes
      if(niter0[0] == 0 && niter0[1] == 0) {return 2;}
      if(niter0[0] > 0 && niter0[1] > 0) {
         // if both have iterated, take the smaller chi2
         if(chi2min[1] < chi2min[0]) {ipass = 1;}
      } else {
         // if one has iterated, take it
         if(niter0[1] > 0) {ipass = 1;}
      }
   }
   
   //   printf("niter = %d \n", niter);
   
   // Correct the charge ratio for missing pixels
   
   float fq;
   if(qtfrac0[ipass] < 0.10f || qtfrac0[ipass] > 1.f) {qtfrac0[ipass] = 1.f;}
   fq = fq0/qtfrac0[ipass];
   
   //   printf("qtfrac0 = %f \n", qtfrac0);
   
   if(fq > fbin[0]) {
      qbin=0;
   } else {
      if(fq > fbin[1]) {
         qbin=1;
      } else {
         if(fq > fbin[2]) {
            qbin=2;
         } else {
            qbin=3;
         }
      }
   }
   
   // Get charge related quantities
   
   float scalex = templ2D.scalex(qbin);
   float scaley = templ2D.scaley(qbin);
   float offsetx = templ2D.offsetx(qbin);
   float offsety = templ2D.offsety(qbin);
//   printf("scalex = %f, scaley = %f, offsetx = %f, offsety = %f \n", scalex, scaley, offsetx, offsety);
   
   // This 2D code has the origin (0,0) at the lower left edge of the input cluster
   // That is now pixel [1,1] and the template reco convention is the middle
   // of that pixel, so we need to correct
   
   xrec = x2D0[ipass] - xsize/2. - offsetx;
   if(xdouble[0]) xrec -= xsize/2.f;
   yrec = y2D0[ipass] - ysize/2. - offsety;
   if(ydouble[0]) yrec -= ysize/2.f;
   if(xerr2[ipass] > 0.f) {
      sigmax = scalex*sqrt(xerr2[ipass]);
      if(sigmax < 3.f) sigmax = 3.f;
   } else {
      sigmax = 10000.f;
   }
   if(yerr2[ipass] > 0.f) {
      sigmay = scaley*sqrt(yerr2[ipass]);
      if(sigmay < 3.f) sigmay = 3.f;
   } else {
      sigmay = 10000.f;
   }
   if(id > 0) {
      // Do chi2 probability calculation
      double meanxy = (double)(npixel*templ2D.chi2ppix());
      double chi2scale = (double)templ2D.chi2scale();
      if(meanxy < 0.01) {meanxy = 0.01;}
      double hndof = meanxy/2.f;
      double hchi2 = chi2scale*chi2min[ipass]/2.f;
      probxy = (float)(1. - TMath::Gamma(hndof, hchi2));
      // Do the charge probability
      float mpv = templ2D.mpvvav();
      float sigmaQ = templ2D.sigmavav();
      float kappa = templ2D.kappavav();
      float xvav = (qtotal/qtfrac0[ipass]-mpv)/sigmaQ;
      float beta2 = 1.f;
      //  VVIObj is a private port of CERNLIB VVIDIS
      VVIObjF vvidist(kappa, beta2, 1);
      float prvav = vvidist.fcn(xvav);
      probQ = 1.f - prvav;
   } else {
      probxy = chi2min[ipass];
      npixels = npixel;
      probQ = 0.f;
   }
   
   // Now
   
   //   printf("end minimization, errors = %f, %f \n\n", sqrt(xerr2), sqrt(yerr2));
   
   //   		 printf("** 2D-template, cotalpha = %f, cotbeta = %f, x2D = %f, y2D = %f **\n", cotalpha, cotbeta, x2D, y2D);
		 
   //		 for (k=1; k < BXM3; ++k) {
   //			 printf("%5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f // %5.0f %5.0f %5.0f \n",
   //template2d[k][1],template2d[k][2],template2d[k][3],template2d[k][4],template2d[k][5],template2d[k][6],template2d[k][7],template2d[k][8],template2d[k][9],
   //template2d[k][10],template2d[k][11],template2d[k][12],template2d[k][13],template2d[k][14],template2d[k][15],template2d[k][16],template2d[k][17],template2d[k][18],
   //					  template2d[k][19],template2d[k][20],template2d[k][21]);
   //		 }
		 
   
   
   return 0;
} // PixelTempReco2D

int SiPixelTemplateReco2D::PixelTempReco3D(int id, float cotalpha, float cotbeta, float locBz, float locBx, int edgeflagy,
                                           int edgeflagx, ClusMatrix & cluster, SiPixelTemplate2D& templ2D,float& yrec, float& sigmay, float& xrec, float& sigmax,
                                           float& probxy, float& probQ, int& qbin, float& deltay)

{
   // Local variables
   int npixels;
   return SiPixelTemplateReco2D::PixelTempReco3D(id, cotalpha, cotbeta, locBz, locBx, edgeflagy, edgeflagx, cluster,
                                                 templ2D, yrec, sigmay, xrec, sigmax, probxy, probQ, qbin, deltay, npixels);
} // PixelTempReco2D
