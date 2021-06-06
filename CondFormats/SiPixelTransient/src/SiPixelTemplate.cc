//
//  SiPixelTemplate.cc  Version 10.24
//
//  Add goodness-of-fit info and spare entries to templates, version number in template header, more error checking
//  Add correction for (Q_F-Q_L)/(Q_F+Q_L) bias
//  Add cot(beta) reflection to reduce y-entries and more sophisticated x-interpolation
//  Fix small index searching bug in interpolate method
//  Change interpolation indexing to avoid complier complaining about possible un-initialized variables
//  Replace containers with static arrays in calls to ysigma2 and xsigma2
//  Add external threshold to calls to ysigma2 and xsigma2, fix parameter signal max for xsigma2
//  Return to 5 pixel spanning but adjust boundaries to use only when needed
//  Implement improved (faster) chi2min search that depends on pixel types
//  Fill template arrays in single calls to this object
//  Add qmin to the template
//  Add qscale to match charge scales
//  Small improvement to x-chisquare interpolation
//  Enlarge SiPixelTemplateStore to accommodate larger templates and increased alpha acceptance (reduce PT threshold to ~200 MeV)
//  Change output streams to conform to CMSSW info and error logging.
//  Store x and y cluster sizes in fractional pixels to facilitate cluster splitting
//  Add methods to return 3-d templates needed for cluster splitting
//  Keep interpolated central 9 template bins in private storage and expand/shift in the getter functions (faster for speed=2/3) and easier to build 3d templates
//  Store error and bias information for the simple chi^2 min position analysis (no interpolation or Q_{FB} corrections) to use in cluster splitting
//  To save time, the gaussian centers and sigma are not interpolated right now (they aren't currently used).  They can be restored by un-commenting lines in the interpolate method.
//  Add a new method to calculate qbin for input cotbeta and cluster charge.  To be used for error estimation of merged clusters in PixelCPEGeneric.
//  Improve the charge estimation for larger cot(alpha) tracks
//  Change interpolate method to return false boolean if track angles are outside of range
//  Add template info and method for truncation information
//  Change to allow template sizes to be changed at compile time
//  Fix bug in track angle checking
//  Accommodate Dave's new DB pushfile which overloads the old method (file input)
//  Add CPEGeneric error information and expand qbin method to access useful info for PixelCPEGeneric
//  Fix large cot(alpha) bug in qmin interpolation
//  Add second qmin to allow a qbin=5 state
//  Use interpolated chi^2 info for one-pixel clusters
//  Fix DB pushfile version number checking bug.
//  Remove assert from qbin method
//  Replace asserts with exceptions in CMSSW
//  Change calling sequence to interpolate method to handle cot(beta)<0 for FPix cosmics
//  Add getter for pixelav Lorentz width estimates to qbin method
//  Add check on template size to interpolate and qbin methods
//  Add qbin population information, charge distribution information
//
//
//  V7.00 - Decouple BPix and FPix information into separate templates
//  Add methods to facilitate improved cluster splitting
//  Fix small charge scaling bug (affects FPix only)
//  Change y-slice used for the x-template to be closer to the actual cotalpha-cotbeta point
//  (there is some weak breakdown of x-y factorization in the FPix after irradiation)
//
//
//  V8.00 - Add method to calculate a simple 2D template
//  Reorganize the interpolate method to extract header info only once per ID
//  V8.01 - Improve simple template normalization
//  V8.05 - Change qbin normalization to work better after irradiation
//  V8.10 - Add Vavilov distribution interpolation
//  V8.11 - Renormalize the x-templates for Guofan's cluster size calculation
//  V8.12 - Technical fix to qavg issue.
//  V8.13 - Fix qbin and fastsim interpolaters to avoid changing class variables
//  V8.20 - Add methods to identify the central pixels in the x- and y-templates (to help align templates with clusters in radiation damaged detectors)
//          Rename class variables from pxxxx (private xxxx) to xxxx_ to follow standard convention.
//          Add compiler option to store the template entries in BOOST multiarrays of structs instead of simple c arrays
//          (allows dynamic resizing template storage and has bounds checking but costs ~10% in execution time).
//  V8.21 - Add new qbin method to use in cluster splitting
//  V8.23 - Replace chi-min position errors with merged cluster chi2 probability info
//  V8.25 - Incorporate VI's speed changes into the current version
//  V8.26 - Modify the Vavilov lookups to work better with the FPix (offset y-templates)
//  V8.30 - Change the splitting template generation and access to improve speed and eliminate triple index boost::multiarray
//  V8.31 - Add correction factor: measured/true charge
//  V8.31 - Fix version number bug in db object I/O (pushfile)
//  V8.32 - Check for illegal qmin during loading
//  V8.33 - Fix small type conversion warnings
//  V8.40 - Incorporate V.I. optimizations
//  V9.00 - Expand header to include multi and single dcol thresholds, LA biases, and (variable) Qbin definitions
//  V9.01 - Protect against negative error squared
//  V10.00 - Update to work with Phase 1 FPix.  Fix some code problems introduced by other maintainers.
//  V10.01 - Fix initialization style as suggested by S. Krutelyov
//  V10.10 - Add class variables and methods to be used to correctly calculate the probabilities of single pixel clusters
//  V10.11 - Allow subdetector ID=5 for FPix R2P2 [allows better internal labeling of templates]
//  V10.12 - Enforce minimum signal size in pixel charge uncertainty calculation
//  V10.13 - Update the variable size [SI_PIXEL_TEMPLATE_USE_BOOST] option so that it works with VI's enhancements
//  V10.20 - Add directory path selection to the ascii pushfile method
//  V10.21 - Address runtime issues in pushfile() for gcc 7.X due to using tempfile as char string + misc. cleanup [Petar]
//  V10.22 - Move templateStore to the heap, fix variable name in pushfile() [Petar]
//  V10.24 - Add sideload() + associated gymnastics [Petar and Oz]

//  Created by Morris Swartz on 10/27/06.
//
//

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include <cmath>
#else
#include <math.h>
#endif
#include <algorithm>
#include <vector>
#include "boost/multi_array.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <list>

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#include "CondFormats/SiPixelTransient/interface/SimplePixel.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) LogError(x)
#define LOGINFO(x) LogInfo(x)
#define LOGWARNING(x) LogWarning(x)
#define ENDL " "
#include "FWCore/Utilities/interface/Exception.h"
using namespace edm;
#else
#include "SiPixelTemplate.h"
#include "SimplePixel.h"
#define LOGERROR(x) std::cout << x << ": "
#define LOGINFO(x) std::cout << x << ": "
#define ENDL std::endl
#endif

//****************************************************************
//! This routine initializes the global template structures from
//! an external file template_summary_zpNNNN where NNNN are four
//! digits of filenum.
//! \param filenum - an integer NNNN used in the filename template_summary_zpNNNN
//****************************************************************
bool SiPixelTemplate::pushfile(int filenum, std::vector<SiPixelTemplateStore>& pixelTemp, std::string dir) {
  // Add template stored in external file numbered filenum to theTemplateStore

  // Local variables
  int i, j, k, l;
  float qavg_avg;
  char c;
  const int code_version = {17};

  //  Create a filename for this run
  std::string tempfile = std::to_string(filenum);

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  // If integer filenum has less than 4 digits, prepend 0's until we have four numerical characters, e.g. "0292"
  int nzeros = 4 - tempfile.length();
  if (nzeros > 0)
    tempfile = std::string(nzeros, '0') + tempfile;
  /// Alt implementation: for (unsigned cnt=4-tempfile.length(); cnt > 0; cnt-- ){ tempfile = "0" + tempfile; }

  tempfile = dir + "template_summary_zp" + tempfile + ".out";
  edm::FileInPath file(tempfile);  // Find the file in CMSSW
  tempfile = file.fullPath();      // Put it back with the whole path.

#else
  // This is the same as above, but more elegant.  (Elegance not allowed in CMSSW...)
  std::ostringstream tout;
  tout << "template_summary_zp" << std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
  tempfile = tout.str();

#endif

  //  Open the template file
  //
  std::ifstream in_file(tempfile);
  if (in_file.is_open() && in_file.good()) {
    // Create a local template storage entry

    SiPixelTemplateStore theCurrentTemp;

    // Read-in a header string first and print it

    for (i = 0; (c = in_file.get()) != '\n'; ++i) {
      if (i < 79) {
        theCurrentTemp.head.title[i] = c;
      }
    }
    if (i > 78) {
      i = 78;
    }
    theCurrentTemp.head.title[i + 1] = '\0';
    LOGINFO("SiPixelTemplate") << "Loading Pixel Template File - " << theCurrentTemp.head.title << ENDL;

    // next, the header information

    in_file >> theCurrentTemp.head.ID >> theCurrentTemp.head.templ_version >> theCurrentTemp.head.Bfield >>
        theCurrentTemp.head.NTy >> theCurrentTemp.head.NTyx >> theCurrentTemp.head.NTxx >> theCurrentTemp.head.Dtype >>
        theCurrentTemp.head.Vbias >> theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >>
        theCurrentTemp.head.qscale >> theCurrentTemp.head.s50 >> theCurrentTemp.head.lorywidth >>
        theCurrentTemp.head.lorxwidth >> theCurrentTemp.head.ysize >> theCurrentTemp.head.xsize >>
        theCurrentTemp.head.zsize;

    if (in_file.fail()) {
      LOGERROR("SiPixelTemplate") << "Error reading file 0A, no template load" << ENDL;
      return false;
    }

    if (theCurrentTemp.head.templ_version > 17) {
      in_file >> theCurrentTemp.head.ss50 >> theCurrentTemp.head.lorybias >> theCurrentTemp.head.lorxbias >>
          theCurrentTemp.head.fbin[0] >> theCurrentTemp.head.fbin[1] >> theCurrentTemp.head.fbin[2];

      if (in_file.fail()) {
        LOGERROR("SiPixelTemplate") << "Error reading file 0B, no template load" << ENDL;
        return false;
      }
    } else {
      theCurrentTemp.head.ss50 = theCurrentTemp.head.s50;
      theCurrentTemp.head.lorybias = theCurrentTemp.head.lorywidth / 2.f;
      theCurrentTemp.head.lorxbias = theCurrentTemp.head.lorxwidth / 2.f;
      theCurrentTemp.head.fbin[0] = 1.5f;
      theCurrentTemp.head.fbin[1] = 1.00f;
      theCurrentTemp.head.fbin[2] = 0.85f;
    }

    LOGINFO("SiPixelTemplate") << "Template ID = " << theCurrentTemp.head.ID << ", Template Version "
                               << theCurrentTemp.head.templ_version << ", Bfield = " << theCurrentTemp.head.Bfield
                               << ", NTy = " << theCurrentTemp.head.NTy << ", NTyx = " << theCurrentTemp.head.NTyx
                               << ", NTxx = " << theCurrentTemp.head.NTxx << ", Dtype = " << theCurrentTemp.head.Dtype
                               << ", Bias voltage " << theCurrentTemp.head.Vbias << ", temperature "
                               << theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence
                               << ", Q-scaling factor " << theCurrentTemp.head.qscale << ", 1/2 multi dcol threshold "
                               << theCurrentTemp.head.s50 << ", 1/2 single dcol threshold " << theCurrentTemp.head.ss50
                               << ", y Lorentz Width " << theCurrentTemp.head.lorywidth << ", y Lorentz Bias "
                               << theCurrentTemp.head.lorybias << ", x Lorentz width " << theCurrentTemp.head.lorxwidth
                               << ", x Lorentz Bias " << theCurrentTemp.head.lorxbias
                               << ", Q/Q_avg fractions for Qbin defs " << theCurrentTemp.head.fbin[0] << ", "
                               << theCurrentTemp.head.fbin[1] << ", " << theCurrentTemp.head.fbin[2]
                               << ", pixel x-size " << theCurrentTemp.head.xsize << ", y-size "
                               << theCurrentTemp.head.ysize << ", zsize " << theCurrentTemp.head.zsize << ENDL;

    if (theCurrentTemp.head.templ_version < code_version) {
      LOGERROR("SiPixelTemplate") << "code expects version " << code_version << " finds "
                                  << theCurrentTemp.head.templ_version << ", no template load" << ENDL;
      return false;
    }

#ifdef SI_PIXEL_TEMPLATE_USE_BOOST

    // next, layout the 1-d/2-d structures needed to store template

    theCurrentTemp.cotbetaY = std::vector<float>(theCurrentTemp.head.NTy);
    theCurrentTemp.cotbetaX = std::vector<float>(theCurrentTemp.head.NTyx);
    theCurrentTemp.cotalphaX = std::vector<float>(theCurrentTemp.head.NTxx);

    theCurrentTemp.enty.resize(boost::extents[theCurrentTemp.head.NTy]);

    theCurrentTemp.entx.resize(boost::extents[theCurrentTemp.head.NTyx][theCurrentTemp.head.NTxx]);

#endif

    // next, loop over all y-angle entries

    for (i = 0; i < theCurrentTemp.head.NTy; ++i) {
      in_file >> theCurrentTemp.enty[i].runnum >> theCurrentTemp.enty[i].costrk[0] >>
          theCurrentTemp.enty[i].costrk[1] >> theCurrentTemp.enty[i].costrk[2];

      if (in_file.fail()) {
        LOGERROR("SiPixelTemplate") << "Error reading file 1, no template load, run # " << theCurrentTemp.enty[i].runnum
                                    << ENDL;
        return false;
      }

      // Calculate the alpha, beta, and cot(beta) for this entry

      theCurrentTemp.enty[i].alpha =
          static_cast<float>(atan2((double)theCurrentTemp.enty[i].costrk[2], (double)theCurrentTemp.enty[i].costrk[0]));

      theCurrentTemp.enty[i].cotalpha = theCurrentTemp.enty[i].costrk[0] / theCurrentTemp.enty[i].costrk[2];

      theCurrentTemp.enty[i].beta =
          static_cast<float>(atan2((double)theCurrentTemp.enty[i].costrk[2], (double)theCurrentTemp.enty[i].costrk[1]));

      theCurrentTemp.enty[i].cotbeta = theCurrentTemp.enty[i].costrk[1] / theCurrentTemp.enty[i].costrk[2];

      in_file >> theCurrentTemp.enty[i].qavg >> theCurrentTemp.enty[i].pixmax >> theCurrentTemp.enty[i].symax >>
          theCurrentTemp.enty[i].dyone >> theCurrentTemp.enty[i].syone >> theCurrentTemp.enty[i].sxmax >>
          theCurrentTemp.enty[i].dxone >> theCurrentTemp.enty[i].sxone;

      if (in_file.fail()) {
        LOGERROR("SiPixelTemplate") << "Error reading file 2, no template load, run # " << theCurrentTemp.enty[i].runnum
                                    << ENDL;
        return false;
      }

      in_file >> theCurrentTemp.enty[i].dytwo >> theCurrentTemp.enty[i].sytwo >> theCurrentTemp.enty[i].dxtwo >>
          theCurrentTemp.enty[i].sxtwo >> theCurrentTemp.enty[i].qmin >> theCurrentTemp.enty[i].clsleny >>
          theCurrentTemp.enty[i].clslenx;

      if (in_file.fail()) {
        LOGERROR("SiPixelTemplate") << "Error reading file 3, no template load, run # " << theCurrentTemp.enty[i].runnum
                                    << ENDL;
        return false;
      }

      if (theCurrentTemp.enty[i].qmin <= 0.) {
        LOGERROR("SiPixelTemplate") << "Error in template ID " << theCurrentTemp.head.ID
                                    << " qmin = " << theCurrentTemp.enty[i].qmin << ", run # "
                                    << theCurrentTemp.enty[i].runnum << ENDL;
        return false;
      }

      for (j = 0; j < 2; ++j) {
        in_file >> theCurrentTemp.enty[i].ypar[j][0] >> theCurrentTemp.enty[i].ypar[j][1] >>
            theCurrentTemp.enty[i].ypar[j][2] >> theCurrentTemp.enty[i].ypar[j][3] >> theCurrentTemp.enty[i].ypar[j][4];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 4, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 9; ++j) {
        for (k = 0; k < TYSIZE; ++k) {
          in_file >> theCurrentTemp.enty[i].ytemp[j][k];
        }

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 5, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 2; ++j) {
        in_file >> theCurrentTemp.enty[i].xpar[j][0] >> theCurrentTemp.enty[i].xpar[j][1] >>
            theCurrentTemp.enty[i].xpar[j][2] >> theCurrentTemp.enty[i].xpar[j][3] >> theCurrentTemp.enty[i].xpar[j][4];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 6, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      qavg_avg = 0.f;
      for (j = 0; j < 9; ++j) {
        for (k = 0; k < TXSIZE; ++k) {
          in_file >> theCurrentTemp.enty[i].xtemp[j][k];
          qavg_avg += theCurrentTemp.enty[i].xtemp[j][k];
        }

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 7, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }
      theCurrentTemp.enty[i].qavg_avg = qavg_avg / 9.;

      for (j = 0; j < 4; ++j) {
        in_file >> theCurrentTemp.enty[i].yavg[j] >> theCurrentTemp.enty[i].yrms[j] >> theCurrentTemp.enty[i].ygx0[j] >>
            theCurrentTemp.enty[i].ygsig[j];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 8, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        in_file >> theCurrentTemp.enty[i].yflpar[j][0] >> theCurrentTemp.enty[i].yflpar[j][1] >>
            theCurrentTemp.enty[i].yflpar[j][2] >> theCurrentTemp.enty[i].yflpar[j][3] >>
            theCurrentTemp.enty[i].yflpar[j][4] >> theCurrentTemp.enty[i].yflpar[j][5];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 9, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        in_file >> theCurrentTemp.enty[i].xavg[j] >> theCurrentTemp.enty[i].xrms[j] >> theCurrentTemp.enty[i].xgx0[j] >>
            theCurrentTemp.enty[i].xgsig[j];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 10, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        in_file >> theCurrentTemp.enty[i].xflpar[j][0] >> theCurrentTemp.enty[i].xflpar[j][1] >>
            theCurrentTemp.enty[i].xflpar[j][2] >> theCurrentTemp.enty[i].xflpar[j][3] >>
            theCurrentTemp.enty[i].xflpar[j][4] >> theCurrentTemp.enty[i].xflpar[j][5];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 11, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        in_file >> theCurrentTemp.enty[i].chi2yavg[j] >> theCurrentTemp.enty[i].chi2ymin[j] >>
            theCurrentTemp.enty[i].chi2xavg[j] >> theCurrentTemp.enty[i].chi2xmin[j];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 12, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        in_file >> theCurrentTemp.enty[i].yavgc2m[j] >> theCurrentTemp.enty[i].yrmsc2m[j] >>
            theCurrentTemp.enty[i].chi2yavgc2m[j] >> theCurrentTemp.enty[i].chi2yminc2m[j];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 13, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        in_file >> theCurrentTemp.enty[i].xavgc2m[j] >> theCurrentTemp.enty[i].xrmsc2m[j] >>
            theCurrentTemp.enty[i].chi2xavgc2m[j] >> theCurrentTemp.enty[i].chi2xminc2m[j];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 14, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        in_file >> theCurrentTemp.enty[i].yavggen[j] >> theCurrentTemp.enty[i].yrmsgen[j] >>
            theCurrentTemp.enty[i].ygx0gen[j] >> theCurrentTemp.enty[i].ygsiggen[j];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 14a, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        in_file >> theCurrentTemp.enty[i].xavggen[j] >> theCurrentTemp.enty[i].xrmsgen[j] >>
            theCurrentTemp.enty[i].xgx0gen[j] >> theCurrentTemp.enty[i].xgsiggen[j];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 14b, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      in_file >> theCurrentTemp.enty[i].chi2yavgone >> theCurrentTemp.enty[i].chi2yminone >>
          theCurrentTemp.enty[i].chi2xavgone >> theCurrentTemp.enty[i].chi2xminone >> theCurrentTemp.enty[i].qmin2 >>
          theCurrentTemp.enty[i].mpvvav >> theCurrentTemp.enty[i].sigmavav >> theCurrentTemp.enty[i].kappavav >>
          theCurrentTemp.enty[i].r_qMeas_qTrue >> theCurrentTemp.enty[i].spare[0];

      if (in_file.fail()) {
        LOGERROR("SiPixelTemplate") << "Error reading file 15, no template load, run # "
                                    << theCurrentTemp.enty[i].runnum << ENDL;
        return false;
      }

      in_file >> theCurrentTemp.enty[i].mpvvav2 >> theCurrentTemp.enty[i].sigmavav2 >>
          theCurrentTemp.enty[i].kappavav2 >> theCurrentTemp.enty[i].qbfrac[0] >> theCurrentTemp.enty[i].qbfrac[1] >>
          theCurrentTemp.enty[i].qbfrac[2] >> theCurrentTemp.enty[i].fracyone >> theCurrentTemp.enty[i].fracxone >>
          theCurrentTemp.enty[i].fracytwo >> theCurrentTemp.enty[i].fracxtwo;
      //		theCurrentTemp.enty[i].qbfrac[3] = 1. - theCurrentTemp.enty[i].qbfrac[0] - theCurrentTemp.enty[i].qbfrac[1] - theCurrentTemp.enty[i].qbfrac[2];

      if (in_file.fail()) {
        LOGERROR("SiPixelTemplate") << "Error reading file 16, no template load, run # "
                                    << theCurrentTemp.enty[i].runnum << ENDL;
        return false;
      }
    }

    // next, loop over all barrel x-angle entries

    for (k = 0; k < theCurrentTemp.head.NTyx; ++k) {
      for (i = 0; i < theCurrentTemp.head.NTxx; ++i) {
        in_file >> theCurrentTemp.entx[k][i].runnum >> theCurrentTemp.entx[k][i].costrk[0] >>
            theCurrentTemp.entx[k][i].costrk[1] >> theCurrentTemp.entx[k][i].costrk[2];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 17, no template load, run # "
                                      << theCurrentTemp.entx[k][i].runnum << ENDL;
          return false;
        }

        // Calculate the alpha, beta, and cot(beta) for this entry

        theCurrentTemp.entx[k][i].alpha = static_cast<float>(
            atan2((double)theCurrentTemp.entx[k][i].costrk[2], (double)theCurrentTemp.entx[k][i].costrk[0]));

        theCurrentTemp.entx[k][i].cotalpha = theCurrentTemp.entx[k][i].costrk[0] / theCurrentTemp.entx[k][i].costrk[2];

        theCurrentTemp.entx[k][i].beta = static_cast<float>(
            atan2((double)theCurrentTemp.entx[k][i].costrk[2], (double)theCurrentTemp.entx[k][i].costrk[1]));

        theCurrentTemp.entx[k][i].cotbeta = theCurrentTemp.entx[k][i].costrk[1] / theCurrentTemp.entx[k][i].costrk[2];

        in_file >> theCurrentTemp.entx[k][i].qavg >> theCurrentTemp.entx[k][i].pixmax >>
            theCurrentTemp.entx[k][i].symax >> theCurrentTemp.entx[k][i].dyone >> theCurrentTemp.entx[k][i].syone >>
            theCurrentTemp.entx[k][i].sxmax >> theCurrentTemp.entx[k][i].dxone >> theCurrentTemp.entx[k][i].sxone;

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 18, no template load, run # "
                                      << theCurrentTemp.entx[k][i].runnum << ENDL;
          return false;
        }

        in_file >> theCurrentTemp.entx[k][i].dytwo >> theCurrentTemp.entx[k][i].sytwo >>
            theCurrentTemp.entx[k][i].dxtwo >> theCurrentTemp.entx[k][i].sxtwo >> theCurrentTemp.entx[k][i].qmin >>
            theCurrentTemp.entx[k][i].clsleny >> theCurrentTemp.entx[k][i].clslenx;
        //			   >> theCurrentTemp.entx[k][i].mpvvav >> theCurrentTemp.entx[k][i].sigmavav >> theCurrentTemp.entx[k][i].kappavav;

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 19, no template load, run # "
                                      << theCurrentTemp.entx[k][i].runnum << ENDL;
          return false;
        }

        for (j = 0; j < 2; ++j) {
          in_file >> theCurrentTemp.entx[k][i].ypar[j][0] >> theCurrentTemp.entx[k][i].ypar[j][1] >>
              theCurrentTemp.entx[k][i].ypar[j][2] >> theCurrentTemp.entx[k][i].ypar[j][3] >>
              theCurrentTemp.entx[k][i].ypar[j][4];

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 20, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 9; ++j) {
          for (l = 0; l < TYSIZE; ++l) {
            in_file >> theCurrentTemp.entx[k][i].ytemp[j][l];
          }

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 21, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 2; ++j) {
          in_file >> theCurrentTemp.entx[k][i].xpar[j][0] >> theCurrentTemp.entx[k][i].xpar[j][1] >>
              theCurrentTemp.entx[k][i].xpar[j][2] >> theCurrentTemp.entx[k][i].xpar[j][3] >>
              theCurrentTemp.entx[k][i].xpar[j][4];

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 22, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        qavg_avg = 0.f;
        for (j = 0; j < 9; ++j) {
          for (l = 0; l < TXSIZE; ++l) {
            in_file >> theCurrentTemp.entx[k][i].xtemp[j][l];
            qavg_avg += theCurrentTemp.entx[k][i].xtemp[j][l];
          }

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 23, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }
        theCurrentTemp.entx[k][i].qavg_avg = qavg_avg / 9.;

        for (j = 0; j < 4; ++j) {
          in_file >> theCurrentTemp.entx[k][i].yavg[j] >> theCurrentTemp.entx[k][i].yrms[j] >>
              theCurrentTemp.entx[k][i].ygx0[j] >> theCurrentTemp.entx[k][i].ygsig[j];

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 24, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          in_file >> theCurrentTemp.entx[k][i].yflpar[j][0] >> theCurrentTemp.entx[k][i].yflpar[j][1] >>
              theCurrentTemp.entx[k][i].yflpar[j][2] >> theCurrentTemp.entx[k][i].yflpar[j][3] >>
              theCurrentTemp.entx[k][i].yflpar[j][4] >> theCurrentTemp.entx[k][i].yflpar[j][5];

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 25, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          in_file >> theCurrentTemp.entx[k][i].xavg[j] >> theCurrentTemp.entx[k][i].xrms[j] >>
              theCurrentTemp.entx[k][i].xgx0[j] >> theCurrentTemp.entx[k][i].xgsig[j];

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 26, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          in_file >> theCurrentTemp.entx[k][i].xflpar[j][0] >> theCurrentTemp.entx[k][i].xflpar[j][1] >>
              theCurrentTemp.entx[k][i].xflpar[j][2] >> theCurrentTemp.entx[k][i].xflpar[j][3] >>
              theCurrentTemp.entx[k][i].xflpar[j][4] >> theCurrentTemp.entx[k][i].xflpar[j][5];

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 27, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          in_file >> theCurrentTemp.entx[k][i].chi2yavg[j] >> theCurrentTemp.entx[k][i].chi2ymin[j] >>
              theCurrentTemp.entx[k][i].chi2xavg[j] >> theCurrentTemp.entx[k][i].chi2xmin[j];

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 28, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          in_file >> theCurrentTemp.entx[k][i].yavgc2m[j] >> theCurrentTemp.entx[k][i].yrmsc2m[j] >>
              theCurrentTemp.entx[k][i].chi2yavgc2m[j] >> theCurrentTemp.entx[k][i].chi2yminc2m[j];

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 29, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          in_file >> theCurrentTemp.entx[k][i].xavgc2m[j] >> theCurrentTemp.entx[k][i].xrmsc2m[j] >>
              theCurrentTemp.entx[k][i].chi2xavgc2m[j] >> theCurrentTemp.entx[k][i].chi2xminc2m[j];

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 30, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          in_file >> theCurrentTemp.entx[k][i].yavggen[j] >> theCurrentTemp.entx[k][i].yrmsgen[j] >>
              theCurrentTemp.entx[k][i].ygx0gen[j] >> theCurrentTemp.entx[k][i].ygsiggen[j];

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 30a, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          in_file >> theCurrentTemp.entx[k][i].xavggen[j] >> theCurrentTemp.entx[k][i].xrmsgen[j] >>
              theCurrentTemp.entx[k][i].xgx0gen[j] >> theCurrentTemp.entx[k][i].xgsiggen[j];

          if (in_file.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 30b, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        in_file >> theCurrentTemp.entx[k][i].chi2yavgone >> theCurrentTemp.entx[k][i].chi2yminone >>
            theCurrentTemp.entx[k][i].chi2xavgone >> theCurrentTemp.entx[k][i].chi2xminone >>
            theCurrentTemp.entx[k][i].qmin2 >> theCurrentTemp.entx[k][i].mpvvav >> theCurrentTemp.entx[k][i].sigmavav >>
            theCurrentTemp.entx[k][i].kappavav >> theCurrentTemp.entx[k][i].r_qMeas_qTrue >>
            theCurrentTemp.entx[k][i].spare[0];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 31, no template load, run # "
                                      << theCurrentTemp.entx[k][i].runnum << ENDL;
          return false;
        }

        in_file >> theCurrentTemp.entx[k][i].mpvvav2 >> theCurrentTemp.entx[k][i].sigmavav2 >>
            theCurrentTemp.entx[k][i].kappavav2 >> theCurrentTemp.entx[k][i].qbfrac[0] >>
            theCurrentTemp.entx[k][i].qbfrac[1] >> theCurrentTemp.entx[k][i].qbfrac[2] >>
            theCurrentTemp.entx[k][i].fracyone >> theCurrentTemp.entx[k][i].fracxone >>
            theCurrentTemp.entx[k][i].fracytwo >> theCurrentTemp.entx[k][i].fracxtwo;
        //		theCurrentTemp.entx[k][i].qbfrac[3] = 1. - theCurrentTemp.entx[k][i].qbfrac[0] - theCurrentTemp.entx[k][i].qbfrac[1] - theCurrentTemp.entx[k][i].qbfrac[2];

        if (in_file.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 32, no template load, run # "
                                      << theCurrentTemp.entx[k][i].runnum << ENDL;
          return false;
        }
      }
    }

    in_file.close();

    // Add this template to the store

    pixelTemp.push_back(theCurrentTemp);

    postInit(pixelTemp);
    return true;

  } else {
    // If file didn't open, report this

    LOGERROR("SiPixelTemplate") << "Error opening File" << tempfile << ENDL;
    return false;
  }

}  // TempInit

#ifndef SI_PIXEL_TEMPLATE_STANDALONE

//****************************************************************
//! This routine initializes the global template structures from an
//! external file template_summary_zpNNNN where NNNN are four digits
//! \param dbobject - db storing multiple template calibrations
//****************************************************************
bool SiPixelTemplate::pushfile(const SiPixelTemplateDBObject& dbobject, std::vector<SiPixelTemplateStore>& pixelTemp) {
  // Add template stored in external dbobject to theTemplateStore

  // Local variables
  int i, j, k, l;
  float qavg_avg;
  const int code_version = {17};

  // We must create a new object because dbobject must be a const and our stream must not be
  auto db(dbobject.reader());

  // Create a local template storage entry
  /// SiPixelTemplateStore theCurrentTemp;    // large, don't allocate it on the stack
  auto tmpPtr = std::make_unique<SiPixelTemplateStore>();  // must be allocated on the heap instead
  auto& theCurrentTemp = *tmpPtr;

  // Fill the template storage for each template calibration stored in the db
  for (int m = 0; m < db.numOfTempl(); ++m) {
    // Read-in a header string first and print it

    SiPixelTemplateDBObject::char2float temp;
    for (i = 0; i < 20; ++i) {
      temp.f = db.sVector()[db.index()];
      theCurrentTemp.head.title[4 * i] = temp.c[0];
      theCurrentTemp.head.title[4 * i + 1] = temp.c[1];
      theCurrentTemp.head.title[4 * i + 2] = temp.c[2];
      theCurrentTemp.head.title[4 * i + 3] = temp.c[3];
      db.incrementIndex(1);
    }
    theCurrentTemp.head.title[79] = '\0';
    LOGINFO("SiPixelTemplate") << "Loading Pixel Template File - " << theCurrentTemp.head.title << ENDL;

    // next, the header information

    db >> theCurrentTemp.head.ID >> theCurrentTemp.head.templ_version >> theCurrentTemp.head.Bfield >>
        theCurrentTemp.head.NTy >> theCurrentTemp.head.NTyx >> theCurrentTemp.head.NTxx >> theCurrentTemp.head.Dtype >>
        theCurrentTemp.head.Vbias >> theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >>
        theCurrentTemp.head.qscale >> theCurrentTemp.head.s50 >> theCurrentTemp.head.lorywidth >>
        theCurrentTemp.head.lorxwidth >> theCurrentTemp.head.ysize >> theCurrentTemp.head.xsize >>
        theCurrentTemp.head.zsize;

    if (db.fail()) {
      LOGERROR("SiPixelTemplate") << "Error reading file 0A, no template load" << ENDL;
      return false;
    }

    LOGINFO("SiPixelTemplate") << "Loading Pixel Template File - " << theCurrentTemp.head.title
                               << " code version = " << code_version << " object version "
                               << theCurrentTemp.head.templ_version << ENDL;

    if (theCurrentTemp.head.templ_version > 17) {
      db >> theCurrentTemp.head.ss50 >> theCurrentTemp.head.lorybias >> theCurrentTemp.head.lorxbias >>
          theCurrentTemp.head.fbin[0] >> theCurrentTemp.head.fbin[1] >> theCurrentTemp.head.fbin[2];

      if (db.fail()) {
        LOGERROR("SiPixelTemplate") << "Error reading file 0B, no template load" << ENDL;
        return false;
      }
    } else {
      theCurrentTemp.head.ss50 = theCurrentTemp.head.s50;
      theCurrentTemp.head.lorybias = theCurrentTemp.head.lorywidth / 2.f;
      theCurrentTemp.head.lorxbias = theCurrentTemp.head.lorxwidth / 2.f;
      theCurrentTemp.head.fbin[0] = 1.50f;
      theCurrentTemp.head.fbin[1] = 1.00f;
      theCurrentTemp.head.fbin[2] = 0.85f;
      //std::cout<<" set fbin  "<< theCurrentTemp.head.fbin[0]<<" "<<theCurrentTemp.head.fbin[1]<<" "
      //	     <<theCurrentTemp.head.fbin[2]<<std::endl;
    }

    LOGINFO("SiPixelTemplate") << "Template ID = " << theCurrentTemp.head.ID << ", Template Version "
                               << theCurrentTemp.head.templ_version << ", Bfield = " << theCurrentTemp.head.Bfield
                               << ", NTy = " << theCurrentTemp.head.NTy << ", NTyx = " << theCurrentTemp.head.NTyx
                               << ", NTxx = " << theCurrentTemp.head.NTxx << ", Dtype = " << theCurrentTemp.head.Dtype
                               << ", Bias voltage " << theCurrentTemp.head.Vbias << ", temperature "
                               << theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence
                               << ", Q-scaling factor " << theCurrentTemp.head.qscale << ", 1/2 multi dcol threshold "
                               << theCurrentTemp.head.s50 << ", 1/2 single dcol threshold " << theCurrentTemp.head.ss50
                               << ", y Lorentz Width " << theCurrentTemp.head.lorywidth << ", y Lorentz Bias "
                               << theCurrentTemp.head.lorybias << ", x Lorentz width " << theCurrentTemp.head.lorxwidth
                               << ", x Lorentz Bias " << theCurrentTemp.head.lorxbias
                               << ", Q/Q_avg fractions for Qbin defs " << theCurrentTemp.head.fbin[0] << ", "
                               << theCurrentTemp.head.fbin[1] << ", " << theCurrentTemp.head.fbin[2]
                               << ", pixel x-size " << theCurrentTemp.head.xsize << ", y-size "
                               << theCurrentTemp.head.ysize << ", zsize " << theCurrentTemp.head.zsize << ENDL;

    if (theCurrentTemp.head.templ_version < code_version) {
      LOGINFO("SiPixelTemplate") << "code expects version " << code_version << " finds "
                                 << theCurrentTemp.head.templ_version << ", load anyway " << ENDL;
      //return false; // dk
    }

#ifdef SI_PIXEL_TEMPLATE_USE_BOOST

    // next, layout the 1-d/2-d structures needed to store template
    theCurrentTemp.cotbetaY = std::vector<float>(theCurrentTemp.head.NTy);
    theCurrentTemp.cotbetaX = std::vector<float>(theCurrentTemp.head.NTyx);
    theCurrentTemp.cotalphaX = std::vector<float>(theCurrentTemp.head.NTxx);
    theCurrentTemp.enty.resize(boost::extents[theCurrentTemp.head.NTy]);
    theCurrentTemp.entx.resize(boost::extents[theCurrentTemp.head.NTyx][theCurrentTemp.head.NTxx]);

#endif

    // next, loop over all barrel y-angle entries

    for (i = 0; i < theCurrentTemp.head.NTy; ++i) {
      db >> theCurrentTemp.enty[i].runnum >> theCurrentTemp.enty[i].costrk[0] >> theCurrentTemp.enty[i].costrk[1] >>
          theCurrentTemp.enty[i].costrk[2];

      if (db.fail()) {
        LOGERROR("SiPixelTemplate") << "Error reading file 1, no template load, run # " << theCurrentTemp.enty[i].runnum
                                    << ENDL;
        return false;
      }

      // Calculate the alpha, beta, and cot(beta) for this entry

      theCurrentTemp.enty[i].alpha =
          static_cast<float>(atan2((double)theCurrentTemp.enty[i].costrk[2], (double)theCurrentTemp.enty[i].costrk[0]));

      theCurrentTemp.enty[i].cotalpha = theCurrentTemp.enty[i].costrk[0] / theCurrentTemp.enty[i].costrk[2];

      theCurrentTemp.enty[i].beta =
          static_cast<float>(atan2((double)theCurrentTemp.enty[i].costrk[2], (double)theCurrentTemp.enty[i].costrk[1]));

      theCurrentTemp.enty[i].cotbeta = theCurrentTemp.enty[i].costrk[1] / theCurrentTemp.enty[i].costrk[2];

      db >> theCurrentTemp.enty[i].qavg >> theCurrentTemp.enty[i].pixmax >> theCurrentTemp.enty[i].symax >>
          theCurrentTemp.enty[i].dyone >> theCurrentTemp.enty[i].syone >> theCurrentTemp.enty[i].sxmax >>
          theCurrentTemp.enty[i].dxone >> theCurrentTemp.enty[i].sxone;

      if (db.fail()) {
        LOGERROR("SiPixelTemplate") << "Error reading file 2, no template load, run # " << theCurrentTemp.enty[i].runnum
                                    << ENDL;
        return false;
      }

      db >> theCurrentTemp.enty[i].dytwo >> theCurrentTemp.enty[i].sytwo >> theCurrentTemp.enty[i].dxtwo >>
          theCurrentTemp.enty[i].sxtwo >> theCurrentTemp.enty[i].qmin >> theCurrentTemp.enty[i].clsleny >>
          theCurrentTemp.enty[i].clslenx;
      //			     >> theCurrentTemp.enty[i].mpvvav >> theCurrentTemp.enty[i].sigmavav >> theCurrentTemp.enty[i].kappavav;

      if (db.fail()) {
        LOGERROR("SiPixelTemplate") << "Error reading file 3, no template load, run # " << theCurrentTemp.enty[i].runnum
                                    << ENDL;
        return false;
      }

      if (theCurrentTemp.enty[i].qmin <= 0.) {
        LOGERROR("SiPixelTemplate") << "Error in template ID " << theCurrentTemp.head.ID
                                    << " qmin = " << theCurrentTemp.enty[i].qmin << ", run # "
                                    << theCurrentTemp.enty[i].runnum << ENDL;
        return false;
      }

      for (j = 0; j < 2; ++j) {
        db >> theCurrentTemp.enty[i].ypar[j][0] >> theCurrentTemp.enty[i].ypar[j][1] >>
            theCurrentTemp.enty[i].ypar[j][2] >> theCurrentTemp.enty[i].ypar[j][3] >> theCurrentTemp.enty[i].ypar[j][4];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 4, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 9; ++j) {
        for (k = 0; k < TYSIZE; ++k) {
          db >> theCurrentTemp.enty[i].ytemp[j][k];
        }

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 5, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 2; ++j) {
        db >> theCurrentTemp.enty[i].xpar[j][0] >> theCurrentTemp.enty[i].xpar[j][1] >>
            theCurrentTemp.enty[i].xpar[j][2] >> theCurrentTemp.enty[i].xpar[j][3] >> theCurrentTemp.enty[i].xpar[j][4];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 6, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      qavg_avg = 0.f;
      for (j = 0; j < 9; ++j) {
        for (k = 0; k < TXSIZE; ++k) {
          db >> theCurrentTemp.enty[i].xtemp[j][k];
          qavg_avg += theCurrentTemp.enty[i].xtemp[j][k];
        }

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 7, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }
      theCurrentTemp.enty[i].qavg_avg = qavg_avg / 9.;

      for (j = 0; j < 4; ++j) {
        db >> theCurrentTemp.enty[i].yavg[j] >> theCurrentTemp.enty[i].yrms[j] >> theCurrentTemp.enty[i].ygx0[j] >>
            theCurrentTemp.enty[i].ygsig[j];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 8, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        db >> theCurrentTemp.enty[i].yflpar[j][0] >> theCurrentTemp.enty[i].yflpar[j][1] >>
            theCurrentTemp.enty[i].yflpar[j][2] >> theCurrentTemp.enty[i].yflpar[j][3] >>
            theCurrentTemp.enty[i].yflpar[j][4] >> theCurrentTemp.enty[i].yflpar[j][5];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 9, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        db >> theCurrentTemp.enty[i].xavg[j] >> theCurrentTemp.enty[i].xrms[j] >> theCurrentTemp.enty[i].xgx0[j] >>
            theCurrentTemp.enty[i].xgsig[j];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 10, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        db >> theCurrentTemp.enty[i].xflpar[j][0] >> theCurrentTemp.enty[i].xflpar[j][1] >>
            theCurrentTemp.enty[i].xflpar[j][2] >> theCurrentTemp.enty[i].xflpar[j][3] >>
            theCurrentTemp.enty[i].xflpar[j][4] >> theCurrentTemp.enty[i].xflpar[j][5];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 11, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        db >> theCurrentTemp.enty[i].chi2yavg[j] >> theCurrentTemp.enty[i].chi2ymin[j] >>
            theCurrentTemp.enty[i].chi2xavg[j] >> theCurrentTemp.enty[i].chi2xmin[j];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 12, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        db >> theCurrentTemp.enty[i].yavgc2m[j] >> theCurrentTemp.enty[i].yrmsc2m[j] >>
            theCurrentTemp.enty[i].chi2yavgc2m[j] >> theCurrentTemp.enty[i].chi2yminc2m[j];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 13, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        db >> theCurrentTemp.enty[i].xavgc2m[j] >> theCurrentTemp.enty[i].xrmsc2m[j] >>
            theCurrentTemp.enty[i].chi2xavgc2m[j] >> theCurrentTemp.enty[i].chi2xminc2m[j];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 14, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        db >> theCurrentTemp.enty[i].yavggen[j] >> theCurrentTemp.enty[i].yrmsgen[j] >>
            theCurrentTemp.enty[i].ygx0gen[j] >> theCurrentTemp.enty[i].ygsiggen[j];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 14a, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      for (j = 0; j < 4; ++j) {
        db >> theCurrentTemp.enty[i].xavggen[j] >> theCurrentTemp.enty[i].xrmsgen[j] >>
            theCurrentTemp.enty[i].xgx0gen[j] >> theCurrentTemp.enty[i].xgsiggen[j];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 14b, no template load, run # "
                                      << theCurrentTemp.enty[i].runnum << ENDL;
          return false;
        }
      }

      db >> theCurrentTemp.enty[i].chi2yavgone >> theCurrentTemp.enty[i].chi2yminone >>
          theCurrentTemp.enty[i].chi2xavgone >> theCurrentTemp.enty[i].chi2xminone >> theCurrentTemp.enty[i].qmin2 >>
          theCurrentTemp.enty[i].mpvvav >> theCurrentTemp.enty[i].sigmavav >> theCurrentTemp.enty[i].kappavav >>
          theCurrentTemp.enty[i].r_qMeas_qTrue >> theCurrentTemp.enty[i].spare[0];

      if (db.fail()) {
        LOGERROR("SiPixelTemplate") << "Error reading file 15, no template load, run # "
                                    << theCurrentTemp.enty[i].runnum << ENDL;
        return false;
      }

      db >> theCurrentTemp.enty[i].mpvvav2 >> theCurrentTemp.enty[i].sigmavav2 >> theCurrentTemp.enty[i].kappavav2 >>
          theCurrentTemp.enty[i].qbfrac[0] >> theCurrentTemp.enty[i].qbfrac[1] >> theCurrentTemp.enty[i].qbfrac[2] >>
          theCurrentTemp.enty[i].fracyone >> theCurrentTemp.enty[i].fracxone >> theCurrentTemp.enty[i].fracytwo >>
          theCurrentTemp.enty[i].fracxtwo;
      //			theCurrentTemp.enty[i].qbfrac[3] = 1. - theCurrentTemp.enty[i].qbfrac[0] - theCurrentTemp.enty[i].qbfrac[1] - theCurrentTemp.enty[i].qbfrac[2];

      if (db.fail()) {
        LOGERROR("SiPixelTemplate") << "Error reading file 16, no template load, run # "
                                    << theCurrentTemp.enty[i].runnum << ENDL;
        return false;
      }
    }

    // next, loop over all barrel x-angle entries

    for (k = 0; k < theCurrentTemp.head.NTyx; ++k) {
      for (i = 0; i < theCurrentTemp.head.NTxx; ++i) {
        db >> theCurrentTemp.entx[k][i].runnum >> theCurrentTemp.entx[k][i].costrk[0] >>
            theCurrentTemp.entx[k][i].costrk[1] >> theCurrentTemp.entx[k][i].costrk[2];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 17, no template load, run # "
                                      << theCurrentTemp.entx[k][i].runnum << ENDL;
          return false;
        }

        // Calculate the alpha, beta, and cot(beta) for this entry

        theCurrentTemp.entx[k][i].alpha = static_cast<float>(
            atan2((double)theCurrentTemp.entx[k][i].costrk[2], (double)theCurrentTemp.entx[k][i].costrk[0]));

        theCurrentTemp.entx[k][i].cotalpha = theCurrentTemp.entx[k][i].costrk[0] / theCurrentTemp.entx[k][i].costrk[2];

        theCurrentTemp.entx[k][i].beta = static_cast<float>(
            atan2((double)theCurrentTemp.entx[k][i].costrk[2], (double)theCurrentTemp.entx[k][i].costrk[1]));

        theCurrentTemp.entx[k][i].cotbeta = theCurrentTemp.entx[k][i].costrk[1] / theCurrentTemp.entx[k][i].costrk[2];

        db >> theCurrentTemp.entx[k][i].qavg >> theCurrentTemp.entx[k][i].pixmax >> theCurrentTemp.entx[k][i].symax >>
            theCurrentTemp.entx[k][i].dyone >> theCurrentTemp.entx[k][i].syone >> theCurrentTemp.entx[k][i].sxmax >>
            theCurrentTemp.entx[k][i].dxone >> theCurrentTemp.entx[k][i].sxone;

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 18, no template load, run # "
                                      << theCurrentTemp.entx[k][i].runnum << ENDL;
          return false;
        }

        db >> theCurrentTemp.entx[k][i].dytwo >> theCurrentTemp.entx[k][i].sytwo >> theCurrentTemp.entx[k][i].dxtwo >>
            theCurrentTemp.entx[k][i].sxtwo >> theCurrentTemp.entx[k][i].qmin >> theCurrentTemp.entx[k][i].clsleny >>
            theCurrentTemp.entx[k][i].clslenx;
        //                     >> theCurrentTemp.entx[k][i].mpvvav >> theCurrentTemp.entx[k][i].sigmavav >> theCurrentTemp.entx[k][i].kappavav;

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 19, no template load, run # "
                                      << theCurrentTemp.entx[k][i].runnum << ENDL;
          return false;
        }

        for (j = 0; j < 2; ++j) {
          db >> theCurrentTemp.entx[k][i].ypar[j][0] >> theCurrentTemp.entx[k][i].ypar[j][1] >>
              theCurrentTemp.entx[k][i].ypar[j][2] >> theCurrentTemp.entx[k][i].ypar[j][3] >>
              theCurrentTemp.entx[k][i].ypar[j][4];

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 20, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 9; ++j) {
          for (l = 0; l < TYSIZE; ++l) {
            db >> theCurrentTemp.entx[k][i].ytemp[j][l];
          }

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 21, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 2; ++j) {
          db >> theCurrentTemp.entx[k][i].xpar[j][0] >> theCurrentTemp.entx[k][i].xpar[j][1] >>
              theCurrentTemp.entx[k][i].xpar[j][2] >> theCurrentTemp.entx[k][i].xpar[j][3] >>
              theCurrentTemp.entx[k][i].xpar[j][4];

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 22, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        qavg_avg = 0.f;
        for (j = 0; j < 9; ++j) {
          for (l = 0; l < TXSIZE; ++l) {
            db >> theCurrentTemp.entx[k][i].xtemp[j][l];
            qavg_avg += theCurrentTemp.entx[k][i].xtemp[j][l];
          }

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 23, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }
        theCurrentTemp.entx[k][i].qavg_avg = qavg_avg / 9.;

        for (j = 0; j < 4; ++j) {
          db >> theCurrentTemp.entx[k][i].yavg[j] >> theCurrentTemp.entx[k][i].yrms[j] >>
              theCurrentTemp.entx[k][i].ygx0[j] >> theCurrentTemp.entx[k][i].ygsig[j];

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 24, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          db >> theCurrentTemp.entx[k][i].yflpar[j][0] >> theCurrentTemp.entx[k][i].yflpar[j][1] >>
              theCurrentTemp.entx[k][i].yflpar[j][2] >> theCurrentTemp.entx[k][i].yflpar[j][3] >>
              theCurrentTemp.entx[k][i].yflpar[j][4] >> theCurrentTemp.entx[k][i].yflpar[j][5];

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 25, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          db >> theCurrentTemp.entx[k][i].xavg[j] >> theCurrentTemp.entx[k][i].xrms[j] >>
              theCurrentTemp.entx[k][i].xgx0[j] >> theCurrentTemp.entx[k][i].xgsig[j];

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 26, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          db >> theCurrentTemp.entx[k][i].xflpar[j][0] >> theCurrentTemp.entx[k][i].xflpar[j][1] >>
              theCurrentTemp.entx[k][i].xflpar[j][2] >> theCurrentTemp.entx[k][i].xflpar[j][3] >>
              theCurrentTemp.entx[k][i].xflpar[j][4] >> theCurrentTemp.entx[k][i].xflpar[j][5];

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 27, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          db >> theCurrentTemp.entx[k][i].chi2yavg[j] >> theCurrentTemp.entx[k][i].chi2ymin[j] >>
              theCurrentTemp.entx[k][i].chi2xavg[j] >> theCurrentTemp.entx[k][i].chi2xmin[j];

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 28, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          db >> theCurrentTemp.entx[k][i].yavgc2m[j] >> theCurrentTemp.entx[k][i].yrmsc2m[j] >>
              theCurrentTemp.entx[k][i].chi2yavgc2m[j] >> theCurrentTemp.entx[k][i].chi2yminc2m[j];

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 29, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          db >> theCurrentTemp.entx[k][i].xavgc2m[j] >> theCurrentTemp.entx[k][i].xrmsc2m[j] >>
              theCurrentTemp.entx[k][i].chi2xavgc2m[j] >> theCurrentTemp.entx[k][i].chi2xminc2m[j];

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 30, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          db >> theCurrentTemp.entx[k][i].yavggen[j] >> theCurrentTemp.entx[k][i].yrmsgen[j] >>
              theCurrentTemp.entx[k][i].ygx0gen[j] >> theCurrentTemp.entx[k][i].ygsiggen[j];

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 30a, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        for (j = 0; j < 4; ++j) {
          db >> theCurrentTemp.entx[k][i].xavggen[j] >> theCurrentTemp.entx[k][i].xrmsgen[j] >>
              theCurrentTemp.entx[k][i].xgx0gen[j] >> theCurrentTemp.entx[k][i].xgsiggen[j];

          if (db.fail()) {
            LOGERROR("SiPixelTemplate") << "Error reading file 30b, no template load, run # "
                                        << theCurrentTemp.entx[k][i].runnum << ENDL;
            return false;
          }
        }

        db >> theCurrentTemp.entx[k][i].chi2yavgone >> theCurrentTemp.entx[k][i].chi2yminone >>
            theCurrentTemp.entx[k][i].chi2xavgone >> theCurrentTemp.entx[k][i].chi2xminone >>
            theCurrentTemp.entx[k][i].qmin2 >> theCurrentTemp.entx[k][i].mpvvav >> theCurrentTemp.entx[k][i].sigmavav >>
            theCurrentTemp.entx[k][i].kappavav >> theCurrentTemp.entx[k][i].r_qMeas_qTrue >>
            theCurrentTemp.entx[k][i].spare[0];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 31, no template load, run # "
                                      << theCurrentTemp.entx[k][i].runnum << ENDL;
          return false;
        }

        db >> theCurrentTemp.entx[k][i].mpvvav2 >> theCurrentTemp.entx[k][i].sigmavav2 >>
            theCurrentTemp.entx[k][i].kappavav2 >> theCurrentTemp.entx[k][i].qbfrac[0] >>
            theCurrentTemp.entx[k][i].qbfrac[1] >> theCurrentTemp.entx[k][i].qbfrac[2] >>
            theCurrentTemp.entx[k][i].fracyone >> theCurrentTemp.entx[k][i].fracxone >>
            theCurrentTemp.entx[k][i].fracytwo >> theCurrentTemp.entx[k][i].fracxtwo;
        //				theCurrentTemp.entx[k][i].qbfrac[3] = 1. - theCurrentTemp.entx[k][i].qbfrac[0] - theCurrentTemp.entx[k][i].qbfrac[1] - theCurrentTemp.entx[k][i].qbfrac[2];

        if (db.fail()) {
          LOGERROR("SiPixelTemplate") << "Error reading file 32, no template load, run # "
                                      << theCurrentTemp.entx[k][i].runnum << ENDL;
          return false;
        }
      }
    }

    // Add this template to the store

    pixelTemp.push_back(theCurrentTemp);
  }
  postInit(pixelTemp);
  return true;

}  // TempInit

#endif

void SiPixelTemplate::postInit(std::vector<SiPixelTemplateStore>& thePixelTemp_) {
  /*
    std::cout << "SiPixelTemplate size " << thePixelTemp_.size() << std::endl;
    #ifndef SI_PIXEL_TEMPLATE_USE_BOOST
    std::cout <<"uses C arrays" << std::endl;
    #endif
    
    int i=0;
    for (auto & templ : thePixelTemp_) {
    std::cout << i <<':' << templ.head.ID << ' ' << templ.head.NTy <<','<< templ.head.NTyx <<','<< templ.head.NTxx << std::endl;
    for ( auto iy=1; iy<templ.head.NTy; ++iy  ) { auto & ent = templ.enty[iy]; std::cout << ent.cotbeta <<',' << ent.cotbeta-templ.enty[iy-1].cotbeta << ' '; }
    std::cout << std::endl;
    for ( auto ix=1; ix<templ.head.NTxx; ++ix  ){ auto & ent = templ.entx[0][ix]; std::cout << ent.cotalpha <<','<< ent.cotalpha-templ.entx[0][ix-1].cotalpha << ' ';}
    std::cout << std::endl;
    ++i;
    }
    */

  for (auto& templ : thePixelTemp_) {
    for (auto iy = 0; iy < templ.head.NTy; ++iy)
      templ.cotbetaY[iy] = templ.enty[iy].cotbeta;
    for (auto iy = 0; iy < templ.head.NTyx; ++iy)
      templ.cotbetaX[iy] = templ.entx[iy][0].cotbeta;
    for (auto ix = 0; ix < templ.head.NTxx; ++ix)
      templ.cotalphaX[ix] = templ.entx[0][ix].cotalpha;
  }
}

// ************************************************************************************************************
//! Interpolate input alpha and beta angles to produce a working template for each individual hit.
//! \param id - (input) index of the template to use
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param locBz - (input) the sign of this quantity is used to determine whether to flip cot(beta)<0 quantities from cot(beta)>0 (FPix only)
//!                    for Phase 0 FPix IP-related tracks, locBz < 0 for cot(beta) > 0 and locBz > 0 for cot(beta) < 0
//!                    for Phase 1 FPix IP-related tracks, see next comment
//! \param locBx - (input) the sign of this quantity is used to determine whether to flip cot(alpha/beta)<0 quantities from cot(alpha/beta)>0 (FPix only)
//!                    for Phase 1 FPix IP-related tracks, locBx/locBz > 0 for cot(alpha) > 0 and locBx/locBz < 0 for cot(alpha) < 0
//!                    for Phase 1 FPix IP-related tracks, locBx > 0 for cot(beta) > 0 and locBx < 0 for cot(beta) < 0
// ************************************************************************************************************
bool SiPixelTemplate::interpolate(int id, float cotalpha, float cotbeta, float locBz, float locBx) {
  // Interpolate for a new set of track angles

  // Local variables
  int i, j;
  int ilow, ihigh, iylow, iyhigh, Ny, Nxx, Nyx, imidy, imaxx;
  float yxratio, xxratio, sxmax, qcorrect, qxtempcor, symax, chi2xavgone, chi2xminone, cota, cotb, cotalpha0, cotbeta0;
  float chi2xavg[4], chi2xmin[4], chi2xavgc2m[4], chi2xminc2m[4];

  // If the sideloading is turned on, xtemp_ and ytemp_ are already set to what they need to be.
  // So we'll just exit.
  if (entry_sideloaded_ != nullptr) {
    success_ = true;
    return success_;
  }

  // Check to see if interpolation is valid
  if (id != id_current_ || cotalpha != cota_current_ || cotbeta != cotb_current_) {
    cota_current_ = cotalpha;
    cotb_current_ = cotbeta;
    success_ = true;

    if (id != id_current_) {
      // Find the index corresponding to id

      index_id_ = -1;
      for (i = 0; i < (int)thePixelTemp_.size(); ++i) {
        //std::cout<<i<<" "<<id<<" "<<thePixelTemp_[i].head.ID<<std::endl;

        if (id == thePixelTemp_[i].head.ID) {
          index_id_ = i;
          id_current_ = id;

          // Copy the detector type to the private variable

          dtype_ = thePixelTemp_[index_id_].head.Dtype;

          // Copy the charge scaling factor to the private variable

          qscale_ = thePixelTemp_[index_id_].head.qscale;

          // Copy the pseudopixel signal size to the private variable

          s50_ = thePixelTemp_[index_id_].head.s50;

          // Copy Qbinning info to private variables

          for (j = 0; j < 3; ++j) {
            fbin_[j] = thePixelTemp_[index_id_].head.fbin[j];
          }
          //std::cout<<" set fbin  "<< fbin_[0]<<" "<<fbin_[1]<<" "<<fbin_[2]<<std::endl;

          // Pixel sizes to the private variables

          xsize_ = thePixelTemp_[index_id_].head.xsize;
          ysize_ = thePixelTemp_[index_id_].head.ysize;
          zsize_ = thePixelTemp_[index_id_].head.zsize;

          break;
        }
      }
    }

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (index_id_ < 0 || index_id_ >= (int)thePixelTemp_.size()) {
      throw cms::Exception("DataCorrupt")
          << "SiPixelTemplate::interpolate can't find needed template ID = " << id << std::endl;
    }
#else
    assert(index_id_ >= 0 && index_id_ < (int)thePixelTemp_.size());
#endif

    //	qcorrect corrects the cot(alpha)=0 cluster charge for non-zero cot(alpha)

    cotalpha0 = thePixelTemp_[index_id_].enty[0].cotalpha;
    qcorrect =
        std::sqrt((1.f + cotbeta * cotbeta + cotalpha * cotalpha) / (1.f + cotbeta * cotbeta + cotalpha0 * cotalpha0));

    // for some cosmics, the ususal gymnastics are incorrect
    cota = cotalpha;
    cotb = abs_cotb_ = std::abs(cotbeta);
    flip_x_ = false;
    flip_y_ = false;
    switch (dtype_) {
      case 0:
        if (cotbeta < 0.f) {
          flip_y_ = true;
        }
        break;
      case 1:
        if (locBz < 0.f) {
          cotb = cotbeta;
        } else {
          cotb = -cotbeta;
          flip_y_ = true;
        }
        break;
      case 2:
      case 3:
      case 4:
      case 5:
        if (locBx * locBz < 0.f) {
          cota = -cotalpha;
          flip_x_ = true;
        }
        if (locBx > 0.f) {
          cotb = cotbeta;
        } else {
          cotb = -cotbeta;
          flip_y_ = true;
        }
        break;
      default:
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
        throw cms::Exception("DataCorrupt") << "SiPixelTemplate::illegal subdetector ID = " << dtype_ << std::endl;
#else
        std::cout << "SiPixelTemplate::illegal subdetector ID = " << dtype_ << std::endl;
#endif
    }

    Ny = thePixelTemp_[index_id_].head.NTy;
    Nyx = thePixelTemp_[index_id_].head.NTyx;
    Nxx = thePixelTemp_[index_id_].head.NTxx;

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (Ny < 2 || Nyx < 1 || Nxx < 2) {
      throw cms::Exception("DataCorrupt")
          << "template ID = " << id_current_ << "has too few entries: Ny/Nyx/Nxx = " << Ny << "/" << Nyx << "/" << Nxx
          << std::endl;
    }
#else
    assert(Ny > 1 && Nyx > 0 && Nxx > 1);
#endif
    imaxx = Nyx - 1;
    imidy = Nxx / 2;

    // next, loop over all y-angle entries

    ilow = 0;
    yratio_ = 0.f;

    if (cotb >= thePixelTemp_[index_id_].enty[Ny - 1].cotbeta) {
      ilow = Ny - 2;
      yratio_ = 1.;
      success_ = false;

    } else {
      if (cotb >= thePixelTemp_[index_id_].enty[0].cotbeta) {
        for (i = 0; i < Ny - 1; ++i) {
          if (thePixelTemp_[index_id_].enty[i].cotbeta <= cotb && cotb < thePixelTemp_[index_id_].enty[i + 1].cotbeta) {
            ilow = i;
            yratio_ = (cotb - thePixelTemp_[index_id_].enty[i].cotbeta) /
                      (thePixelTemp_[index_id_].enty[i + 1].cotbeta - thePixelTemp_[index_id_].enty[i].cotbeta);
            break;
          }
        }
      } else {
        success_ = false;
      }
    }

    ihigh = ilow + 1;

    // Use pointers to the three angle pairs used in the interpolation
    //
    enty0_ = &thePixelTemp_[index_id_].enty[ilow];
    enty1_ = &thePixelTemp_[index_id_].enty[ihigh];

    // Interpolate/store all y-related quantities (flip displacements when flip_y_)

    qavg_ = (1.f - yratio_) * enty0_->qavg + yratio_ * enty1_->qavg;
    qavg_ *= qcorrect;
    symax = (1.f - yratio_) * enty0_->symax + yratio_ * enty1_->symax;
    syparmax_ = symax;
    sxmax = (1.f - yratio_) * enty0_->sxmax + yratio_ * enty1_->sxmax;
    dyone_ = (1.f - yratio_) * enty0_->dyone + yratio_ * enty1_->dyone;
    if (flip_y_) {
      dyone_ = -dyone_;
    }
    syone_ = (1.f - yratio_) * enty0_->syone + yratio_ * enty1_->syone;
    dytwo_ = (1.f - yratio_) * enty0_->dytwo + yratio_ * enty1_->dytwo;
    if (flip_y_) {
      dytwo_ = -dytwo_;
    }
    sytwo_ = (1.f - yratio_) * enty0_->sytwo + yratio_ * enty1_->sytwo;
    qmin_ = (1.f - yratio_) * enty0_->qmin + yratio_ * enty1_->qmin;
    qmin_ *= qcorrect;
    qmin2_ = (1.f - yratio_) * enty0_->qmin2 + yratio_ * enty1_->qmin2;
    qmin2_ *= qcorrect;
    mpvvav_ = (1.f - yratio_) * enty0_->mpvvav + yratio_ * enty1_->mpvvav;
    mpvvav_ *= qcorrect;
    sigmavav_ = (1.f - yratio_) * enty0_->sigmavav + yratio_ * enty1_->sigmavav;
    kappavav_ = (1.f - yratio_) * enty0_->kappavav + yratio_ * enty1_->kappavav;
    mpvvav2_ = (1.f - yratio_) * enty0_->mpvvav2 + yratio_ * enty1_->mpvvav2;
    mpvvav2_ *= qcorrect;
    sigmavav2_ = (1.f - yratio_) * enty0_->sigmavav2 + yratio_ * enty1_->sigmavav2;
    kappavav2_ = (1.f - yratio_) * enty0_->kappavav2 + yratio_ * enty1_->kappavav2;
    clsleny_ = fminf(enty0_->clsleny, enty1_->clsleny);
    qavg_avg_ = (1.f - yratio_) * enty0_->qavg_avg + yratio_ * enty1_->qavg_avg;
    qavg_avg_ *= qcorrect;
    for (i = 0; i < 2; ++i) {
      for (j = 0; j < 5; ++j) {
        // Charge loss switches sides when cot(beta) changes sign

        if (flip_y_) {
          yparl_[1 - i][j] = enty0_->ypar[i][j];
          yparh_[1 - i][j] = enty1_->ypar[i][j];
        } else {
          yparl_[i][j] = enty0_->ypar[i][j];
          yparh_[i][j] = enty1_->ypar[i][j];
        }
        if (flip_x_) {
          xparly0_[1 - i][j] = enty0_->xpar[i][j];
          xparhy0_[1 - i][j] = enty1_->xpar[i][j];
        } else {
          xparly0_[i][j] = enty0_->xpar[i][j];
          xparhy0_[i][j] = enty1_->xpar[i][j];
        }
      }
    }

    for (i = 0; i < 4; ++i) {
      yavg_[i] = (1.f - yratio_) * thePixelTemp_[index_id_].enty[ilow].yavg[i] +
                 yratio_ * thePixelTemp_[index_id_].enty[ihigh].yavg[i];
      if (flip_y_) {
        yavg_[i] = -yavg_[i];
      }
      yavg_[i] = (1.f - yratio_) * enty0_->yavg[i] + yratio_ * enty1_->yavg[i];
      if (flip_y_) {
        yavg_[i] = -yavg_[i];
      }
      yrms_[i] = (1.f - yratio_) * enty0_->yrms[i] + yratio_ * enty1_->yrms[i];
      chi2yavg_[i] = (1.f - yratio_) * enty0_->chi2yavg[i] + yratio_ * enty1_->chi2yavg[i];
      chi2ymin_[i] = (1.f - yratio_) * enty0_->chi2ymin[i] + yratio_ * enty1_->chi2ymin[i];
      chi2xavg[i] = (1.f - yratio_) * enty0_->chi2xavg[i] + yratio_ * enty1_->chi2xavg[i];
      chi2xmin[i] = (1.f - yratio_) * enty0_->chi2xmin[i] + yratio_ * enty1_->chi2xmin[i];
      yavgc2m_[i] = (1.f - yratio_) * enty0_->yavgc2m[i] + yratio_ * enty1_->yavgc2m[i];
      if (flip_y_) {
        yavgc2m_[i] = -yavgc2m_[i];
      }
      yrmsc2m_[i] = (1.f - yratio_) * enty0_->yrmsc2m[i] + yratio_ * enty1_->yrmsc2m[i];
      chi2yavgc2m_[i] = (1.f - yratio_) * enty0_->chi2yavgc2m[i] + yratio_ * enty1_->chi2yavgc2m[i];
      //	      if(flip_y_) {chi2yavgc2m_[i] = -chi2yavgc2m_[i];}
      chi2yminc2m_[i] = (1.f - yratio_) * enty0_->chi2yminc2m[i] + yratio_ * enty1_->chi2yminc2m[i];
      //	      xrmsc2m[i]=(1.f - yratio_)*enty0_->xrmsc2m[i] + yratio_*enty1_->xrmsc2m[i];
      chi2xavgc2m[i] = (1.f - yratio_) * enty0_->chi2xavgc2m[i] + yratio_ * enty1_->chi2xavgc2m[i];
      chi2xminc2m[i] = (1.f - yratio_) * enty0_->chi2xminc2m[i] + yratio_ * enty1_->chi2xminc2m[i];
      for (j = 0; j < 6; ++j) {
        yflparl_[i][j] = enty0_->yflpar[i][j];
        yflparh_[i][j] = enty1_->yflpar[i][j];

        // Since Q_fl is odd under cotbeta, it flips qutomatically, change only even terms

        if (flip_y_ && (j == 0 || j == 2 || j == 4)) {
          yflparl_[i][j] = -yflparl_[i][j];
          yflparh_[i][j] = -yflparh_[i][j];
        }
      }
    }

    //// Single pixel cluster probabilities

    chi2yavgone_ = (1.f - yratio_) * enty0_->chi2yavgone + yratio_ * enty1_->chi2yavgone;
    chi2yminone_ = (1.f - yratio_) * enty0_->chi2yminone + yratio_ * enty1_->chi2yminone;
    chi2xavgone = (1.f - yratio_) * enty0_->chi2xavgone + yratio_ * enty1_->chi2xavgone;
    chi2xminone = (1.f - yratio_) * enty0_->chi2xminone + yratio_ * enty1_->chi2xminone;

    fracyone_ = (1.f - yratio_) * enty0_->fracyone + yratio_ * enty1_->fracyone;
    fracytwo_ = (1.f - yratio_) * enty0_->fracytwo + yratio_ * enty1_->fracytwo;
    //       If using y-spares
    //       for(i=0; i<10; ++i) {
    //		    pyspare[i]=(1.f - yratio_)*enty0_->yspare[i] + yratio_*enty1_->yspare[i];
    //       }

    // Interpolate and build the y-template

    for (i = 0; i < 9; ++i) {
      ytemp_[i][0] = 0.f;
      ytemp_[i][1] = 0.f;
      ytemp_[i][BYM2] = 0.f;
      ytemp_[i][BYM1] = 0.f;
      for (j = 0; j < TYSIZE; ++j) {
        // Flip the basic y-template when the cotbeta is negative

        if (flip_y_) {
          ytemp_[8 - i][BYM3 - j] = (1.f - yratio_) * enty0_->ytemp[i][j] + yratio_ * enty1_->ytemp[i][j];
        } else {
          ytemp_[i][j + 2] = (1.f - yratio_) * enty0_->ytemp[i][j] + yratio_ * enty1_->ytemp[i][j];
        }
      }
    }

    // next, loop over all x-angle entries, first, find relevant y-slices

    iylow = 0;
    yxratio = 0.f;

    if (abs_cotb_ >= thePixelTemp_[index_id_].entx[Nyx - 1][0].cotbeta) {
      iylow = Nyx - 2;
      yxratio = 1.f;

    } else if (abs_cotb_ >= thePixelTemp_[index_id_].entx[0][0].cotbeta) {
      for (i = 0; i < Nyx - 1; ++i) {
        if (thePixelTemp_[index_id_].entx[i][0].cotbeta <= abs_cotb_ &&
            abs_cotb_ < thePixelTemp_[index_id_].entx[i + 1][0].cotbeta) {
          iylow = i;
          yxratio = (abs_cotb_ - thePixelTemp_[index_id_].entx[i][0].cotbeta) /
                    (thePixelTemp_[index_id_].entx[i + 1][0].cotbeta - thePixelTemp_[index_id_].entx[i][0].cotbeta);
          break;
        }
      }
    }

    iyhigh = iylow + 1;

    ilow = 0;
    xxratio = 0.f;

    if (cota >= thePixelTemp_[index_id_].entx[0][Nxx - 1].cotalpha) {
      ilow = Nxx - 2;
      xxratio = 1.f;
      success_ = false;

    } else {
      if (cota >= thePixelTemp_[index_id_].entx[0][0].cotalpha) {
        for (i = 0; i < Nxx - 1; ++i) {
          if (thePixelTemp_[index_id_].entx[0][i].cotalpha <= cota &&
              cota < thePixelTemp_[index_id_].entx[0][i + 1].cotalpha) {
            ilow = i;
            xxratio = (cota - thePixelTemp_[index_id_].entx[0][i].cotalpha) /
                      (thePixelTemp_[index_id_].entx[0][i + 1].cotalpha - thePixelTemp_[index_id_].entx[0][i].cotalpha);
            break;
          }
        }
      } else {
        success_ = false;
      }
    }

    ihigh = ilow + 1;

    // Interpolate/store all x-related quantities

    yxratio_ = yxratio;
    xxratio_ = xxratio;

    // sxparmax defines the maximum charge for which the parameters xpar are defined (not rescaled by cotbeta)

    sxparmax_ = (1.f - xxratio) * thePixelTemp_[index_id_].entx[imaxx][ilow].sxmax +
                xxratio * thePixelTemp_[index_id_].entx[imaxx][ihigh].sxmax;
    sxmax_ = sxparmax_;
    if (thePixelTemp_[index_id_].entx[imaxx][imidy].sxmax != 0.f) {
      sxmax_ = sxmax_ / thePixelTemp_[index_id_].entx[imaxx][imidy].sxmax * sxmax;
    }
    symax_ = (1.f - xxratio) * thePixelTemp_[index_id_].entx[imaxx][ilow].symax +
             xxratio * thePixelTemp_[index_id_].entx[imaxx][ihigh].symax;
    if (thePixelTemp_[index_id_].entx[imaxx][imidy].symax != 0.f) {
      symax_ = symax_ / thePixelTemp_[index_id_].entx[imaxx][imidy].symax * symax;
    }
    dxone_ = (1.f - xxratio) * thePixelTemp_[index_id_].entx[0][ilow].dxone +
             xxratio * thePixelTemp_[index_id_].entx[0][ihigh].dxone;
    if (flip_x_) {
      dxone_ = -dxone_;
    }
    sxone_ = (1.f - xxratio) * thePixelTemp_[index_id_].entx[0][ilow].sxone +
             xxratio * thePixelTemp_[index_id_].entx[0][ihigh].sxone;
    dxtwo_ = (1.f - xxratio) * thePixelTemp_[index_id_].entx[0][ilow].dxtwo +
             xxratio * thePixelTemp_[index_id_].entx[0][ihigh].dxtwo;
    if (flip_x_) {
      dxtwo_ = -dxtwo_;
    }
    sxtwo_ = (1.f - xxratio) * thePixelTemp_[index_id_].entx[0][ilow].sxtwo +
             xxratio * thePixelTemp_[index_id_].entx[0][ihigh].sxtwo;
    clslenx_ = fminf(thePixelTemp_[index_id_].entx[0][ilow].clslenx, thePixelTemp_[index_id_].entx[0][ihigh].clslenx);

    for (i = 0; i < 2; ++i) {
      for (j = 0; j < 5; ++j) {
        // Charge loss switches sides when cot(alpha) changes sign
        if (flip_x_) {
          xpar0_[1 - i][j] = thePixelTemp_[index_id_].entx[imaxx][imidy].xpar[i][j];
          xparl_[1 - i][j] = thePixelTemp_[index_id_].entx[imaxx][ilow].xpar[i][j];
          xparh_[1 - i][j] = thePixelTemp_[index_id_].entx[imaxx][ihigh].xpar[i][j];
        } else {
          xpar0_[i][j] = thePixelTemp_[index_id_].entx[imaxx][imidy].xpar[i][j];
          xparl_[i][j] = thePixelTemp_[index_id_].entx[imaxx][ilow].xpar[i][j];
          xparh_[i][j] = thePixelTemp_[index_id_].entx[imaxx][ihigh].xpar[i][j];
        }
      }
    }
    // Pointers to the currently interpolated point.
    entx00_ = &thePixelTemp_[index_id_].entx[iylow][ilow];
    entx02_ = &thePixelTemp_[index_id_].entx[iylow][ihigh];
    entx20_ = &thePixelTemp_[index_id_].entx[iyhigh][ilow];
    entx22_ = &thePixelTemp_[index_id_].entx[iyhigh][ihigh];
    entx21_ = &thePixelTemp_[index_id_].entx[iyhigh][imidy];

    // pixmax is the maximum allowed pixel charge (used for truncation)
    pixmax_ = (1.f - yxratio) * ((1.f - xxratio) * entx00_->pixmax + xxratio * entx02_->pixmax) +
              yxratio * ((1.f - xxratio) * entx20_->pixmax + xxratio * entx22_->pixmax);

    r_qMeas_qTrue_ = (1.f - yxratio) * ((1.f - xxratio) * entx00_->r_qMeas_qTrue + xxratio * entx02_->r_qMeas_qTrue) +
                     yxratio * ((1.f - xxratio) * entx20_->r_qMeas_qTrue + xxratio * entx22_->r_qMeas_qTrue);

    for (i = 0; i < 4; ++i) {
      xavg_[i] = (1.f - yxratio) * ((1.f - xxratio) * entx00_->xavg[i] + xxratio * entx02_->xavg[i]) +
                 yxratio * ((1.f - xxratio) * entx20_->xavg[i] + xxratio * entx22_->xavg[i]);
      if (flip_x_) {
        xavg_[i] = -xavg_[i];
      }

      xrms_[i] = (1.f - yxratio) * ((1.f - xxratio) * entx00_->xrms[i] + xxratio * entx02_->xrms[i]) +
                 yxratio * ((1.f - xxratio) * entx20_->xrms[i] + xxratio * entx22_->xrms[i]);

      xavgc2m_[i] = (1.f - yxratio) * ((1.f - xxratio) * entx00_->xavgc2m[i] + xxratio * entx02_->xavgc2m[i]) +
                    yxratio * ((1.f - xxratio) * entx20_->xavgc2m[i] + xxratio * entx22_->xavgc2m[i]);
      if (flip_x_) {
        xavgc2m_[i] = -xavgc2m_[i];
      }

      xrmsc2m_[i] = (1.f - yxratio) * ((1.f - xxratio) * entx00_->xrmsc2m[i] + xxratio * entx02_->xrmsc2m[i]) +
                    yxratio * ((1.f - xxratio) * entx20_->xrmsc2m[i] + xxratio * entx22_->xrmsc2m[i]);
      //
      //  Try new interpolation scheme instead
      //
      //
      //	      chi2xavgc2m_[i]=(1.f - yxratio)*((1.f - xxratio)*entx00_->chi2xavgc2m[i] + xxratio*entx02_->chi2xavgc2m[i])
      //		          +yxratio*((1.f - xxratio)*entx20_->chi2xavgc2m[i] + xxratio*entx22_->chi2xavgc2m[i]);

      //	      chi2xminc2m_[i]=(1.f - yxratio)*((1.f - xxratio)*entx00_->chi2xminc2m[i] + xxratio*entx02_->chi2xminc2m[i])
      //		          +yxratio*((1.f - xxratio)*entx20_->chi2xminc2m[i] + xxratio*entx22_->chi2xminc2m[i]);
      //
      //	      chi2xavg_[i]=((1.f - xxratio)*thePixelTemp_[index_id_].entx[imaxx][ilow].chi2xavg[i] + xxratio*thePixelTemp_[index_id_].entx[imaxx][ihigh].chi2xavg[i]);
      //		  if(thePixelTemp_[index_id_].entx[imaxx][imidy].chi2xavg[i] != 0.f) {chi2xavg_[i]=chi2xavg_[i]/thePixelTemp_[index_id_].entx[imaxx][imidy].chi2xavg[i]*chi2xavg[i];}
      //
      //	      chi2xmin_[i]=((1.f - xxratio)*thePixelTemp_[index_id_].entx[imaxx][ilow].chi2xmin[i] + xxratio*thePixelTemp_[index_id_].entx[imaxx][ihigh].chi2xmin[i]);
      //		  if(thePixelTemp_[index_id_].entx[imaxx][imidy].chi2xmin[i] != 0.f) {chi2xmin_[i]=chi2xmin_[i]/thePixelTemp_[index_id_].entx[imaxx][imidy].chi2xmin[i]*chi2xmin[i];}
      //
      chi2xavg_[i] = ((1.f - xxratio) * entx20_->chi2xavg[i] + xxratio * entx22_->chi2xavg[i]);
      if (entx21_->chi2xavg[i] != 0.f) {
        chi2xavg_[i] = chi2xavg_[i] / entx21_->chi2xavg[i] * chi2xavg[i];
      }

      chi2xmin_[i] = ((1.f - xxratio) * entx20_->chi2xmin[i] + xxratio * entx22_->chi2xmin[i]);
      if (entx21_->chi2xmin[i] != 0.f) {
        chi2xmin_[i] = chi2xmin_[i] / entx21_->chi2xmin[i] * chi2xmin[i];
      }

      chi2xavgc2m_[i] = ((1.f - xxratio) * entx20_->chi2xavgc2m[i] + xxratio * entx22_->chi2xavgc2m[i]);
      if (entx21_->chi2xavgc2m[i] != 0.f) {
        chi2xavgc2m_[i] = chi2xavgc2m_[i] / entx21_->chi2xavgc2m[i] * chi2xavgc2m[i];
      }

      chi2xminc2m_[i] = ((1.f - xxratio) * entx20_->chi2xminc2m[i] + xxratio * entx22_->chi2xminc2m[i]);
      if (entx21_->chi2xminc2m[i] != 0.f) {
        chi2xminc2m_[i] = chi2xminc2m_[i] / entx21_->chi2xminc2m[i] * chi2xminc2m[i];
      }

      for (j = 0; j < 6; ++j) {
        xflparll_[i][j] = entx00_->xflpar[i][j];
        xflparlh_[i][j] = entx02_->xflpar[i][j];
        xflparhl_[i][j] = entx20_->xflpar[i][j];
        xflparhh_[i][j] = entx22_->xflpar[i][j];
        // Since Q_fl is odd under cotalpha, it flips qutomatically, change only even terms
        if (flip_x_ && (j == 0 || j == 2 || j == 4)) {
          xflparll_[i][j] = -xflparll_[i][j];
          xflparlh_[i][j] = -xflparlh_[i][j];
          xflparhl_[i][j] = -xflparhl_[i][j];
          xflparhh_[i][j] = -xflparhh_[i][j];
        }
      }
    }

    // Do the spares next

    chi2xavgone_ = ((1.f - xxratio) * entx20_->chi2xavgone + xxratio * entx22_->chi2xavgone);
    if (entx21_->chi2xavgone != 0.f) {
      chi2xavgone_ = chi2xavgone_ / entx21_->chi2xavgone * chi2xavgone;
    }

    chi2xminone_ = ((1.f - xxratio) * entx20_->chi2xminone + xxratio * entx22_->chi2xminone);
    if (entx21_->chi2xminone != 0.f) {
      chi2xminone_ = chi2xminone_ / entx21_->chi2xminone * chi2xminone;
    }

    fracxone_ = (1.f - yxratio) * ((1.f - xxratio) * entx00_->fracxone + xxratio * entx02_->fracxone) +
                yxratio * ((1.f - xxratio) * entx20_->fracxone + xxratio * entx22_->fracxone);
    fracxtwo_ = (1.f - yxratio) * ((1.f - xxratio) * entx00_->fracxtwo + xxratio * entx02_->fracxtwo) +
                yxratio * ((1.f - xxratio) * entx20_->fracxtwo + xxratio * entx22_->fracxtwo);

    //       If using x-spares
    //       for(i=0; i<10; ++i) {
    //	      pxspare[i]=(1.f - yxratio)*((1.f - xxratio)*entx00_->xspare[i] + xxratio*entx02_->xspare[i])
    //		          +yxratio*((1.f - xxratio)*entx20_->xspare[i] + xxratio*entx22_->xspare[i]);
    //       }

    // Interpolate and build the x-template

    //	qxtempcor corrects the total charge to the actual track angles (not actually needed for the template fits, but useful for Guofan)

    cotbeta0 = thePixelTemp_[index_id_].entx[iyhigh][0].cotbeta;
    qxtempcor =
        std::sqrt((1.f + cotbeta * cotbeta + cotalpha * cotalpha) / (1.f + cotbeta0 * cotbeta0 + cotalpha * cotalpha));

    for (i = 0; i < 9; ++i) {
      xtemp_[i][0] = 0.f;
      xtemp_[i][1] = 0.f;
      xtemp_[i][BXM2] = 0.f;
      xtemp_[i][BXM1] = 0.f;
      for (j = 0; j < TXSIZE; ++j) {
        //  Take next largest x-slice for the x-template (it reduces bias in the forward direction after irradiation)
        //		   xtemp_[i][j+2]=(1.f - xxratio)*thePixelTemp_[index_id_].entx[imaxx][ilow].xtemp[i][j] + xxratio*thePixelTemp_[index_id_].entx[imaxx][ihigh].xtemp[i][j];
        //		   xtemp_[i][j+2]=(1.f - xxratio)*entx20_->xtemp[i][j] + xxratio*entx22_->xtemp[i][j];
        if (flip_x_) {
          xtemp_[8 - i][BXM3 - j] =
              qxtempcor * ((1.f - xxratio) * entx20_->xtemp[i][j] + xxratio * entx22_->xtemp[i][j]);
        } else {
          xtemp_[i][j + 2] = qxtempcor * ((1.f - xxratio) * entx20_->xtemp[i][j] + xxratio * entx22_->xtemp[i][j]);
        }
      }
    }

    lorywidth_ = thePixelTemp_[index_id_].head.lorywidth;
    lorxwidth_ = thePixelTemp_[index_id_].head.lorxwidth;
    lorybias_ = thePixelTemp_[index_id_].head.lorybias;
    lorxbias_ = thePixelTemp_[index_id_].head.lorxbias;
    if (flip_x_) {
      lorxwidth_ = -lorxwidth_;
      lorxbias_ = -lorxbias_;
    }
    if (flip_y_) {
      lorywidth_ = -lorywidth_;
      lorybias_ = -lorybias_;
    }
  }

  return success_;
}  // interpolate

// ************************************************************************************************************
//! Interpolate input alpha and beta angles to produce a working template for each individual hit.
//! \param id - (input) index of the template to use
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! Use this for Phase 1, IP related hits
// ************************************************************************************************************
bool SiPixelTemplate::interpolate(int id, float cotalpha, float cotbeta) {
  // Interpolate for a new set of track angles

  // Local variables
  float locBx = 1.f;
  if (cotbeta < 0.f) {
    locBx = -1.f;
  }
  float locBz = locBx;
  if (cotalpha < 0.f) {
    locBz = -locBx;
  }
  return SiPixelTemplate::interpolate(id, cotalpha, cotbeta, locBz, locBx);
}

// ************************************************************************************************************
//! Interpolate input alpha and beta angles to produce a working template for each individual hit.
//! \param id - (input) index of the template to use
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! Use this for Phase 0, IP related hits
// ************************************************************************************************************
bool SiPixelTemplate::interpolate(int id, float cotalpha, float cotbeta, float locBz) {
  // Interpolate for a new set of track angles

  // Local variables
  float locBx = 1.f;
  return SiPixelTemplate::interpolate(id, cotalpha, cotbeta, locBz, locBx);
}

// *************************************************************************************************************************************
//! Load template info for single angle point to invoke template reco for template generation
//! \param      entry - (input) pointer to template entry
//! \param      sizex - (input) pixel x-size
//! \param      sizey - (input) pixel y-size
//! \param      sizez - (input) pixel z-size
// *************************************************************************************************************************************
#ifdef SI_PIXEL_TEMPLATE_STANDALONE
void SiPixelTemplate::sideload(SiPixelTemplateEntry* entry,
                               int iDtype,
                               float locBx,
                               float locBz,
                               float lorwdy,
                               float lorwdx,
                               float q50,
                               float fbin[3],
                               float xsize,
                               float ysize,
                               float zsize) {
  // Set class variables to the input parameters
  entry_sideloaded_ = entry;  // will bypass xtemp_[] and ytemp_[] and use those from this Entry.

  enty1_ = entry;
  enty0_ = entry;

  entx00_ = entry;
  entx02_ = entry;
  entx20_ = entry;
  entx22_ = entry;
  entx21_ = entry;

  dtype_ = iDtype;
  lorywidth_ = lorwdy;
  lorxwidth_ = lorwdx;
  xsize_ = xsize;
  ysize_ = ysize;
  zsize_ = zsize;
  s50_ = q50;
  qscale_ = 1.f;
  for (int i = 0; i < 3; ++i) {
    fbin_[i] = fbin[i];
  }

  // Set other class variables

  yratio_ = 0.f;
  yxratio_ = 0.f;
  xxratio_ = 0.f;

  qavg_ = entry->qavg;
  qmin_ = 0.f;
  qmin2_ = 0.f;

  pixmax_ = entry->pixmax;
  sxmax_ = entry->sxmax;
  symax_ = entry->symax;
  clsleny_ = entry->clsleny;
  clslenx_ = entry->clslenx;

  scaleyavg_ = 1.f;
  scalexavg_ = 1.f;
  delyavg_ = 0.f;
  delysig_ = 0.f;

  dyone_ = entry->dyone;
  syone_ = entry->syone;
  dytwo_ = entry->dytwo;
  sytwo_ = entry->sytwo;

  dxone_ = entry->dxone;
  sxone_ = entry->sxone;
  dxtwo_ = entry->dxtwo;
  sxtwo_ = entry->sxtwo;

  chi2yminone_ = 0.f;
  chi2yavgone_ = 0.1;
  chi2xminone_ = 0.f;
  chi2xavgone_ = 0.1;

  for (int i = 0; i < 4; ++i) {
    scalex_[i] = 1.f;
    scaley_[i] = 1.f;
    offsetx_[i] = 0.f;
    offsety_[i] = 0.f;
    xrms_[i] = 0.f;
    yrms_[i] = 0.f;
    xavg_[i] = 0.f;
    yavg_[i] = 0.f;

    chi2yavg_[i] = 0.1;
    chi2xavg_[i] = 0.1;

    chi2xmin_[i] = 0.f;
    chi2ymin_[i] = 0.f;

    for (int j = 0; j < 6; j++) {
      yflparl_[i][j] = yflparh_[i][j] = entry->yflpar[i][j];
      xflparhh_[i][j] = xflparhl_[i][j] = xflparll_[i][j] = xflparlh_[i][j] = entry->xflpar[i][j];
    }
  }

  sxparmax_ = entry->sxmax;
  syparmax_ = entry->symax;

  // This works only for IP-related tracks

  flip_x_ = false;
  flip_y_ = false;
  float cotbeta = entry->cotbeta;
  switch (dtype_) {
    case 0:
      if (cotbeta < 0.f) {
        flip_y_ = true;
      }
      break;
    case 1:
      if (locBz > 0.f) {
        flip_y_ = true;
      }
      break;
    case 2:
    case 3:
    case 4:
    case 5:
      if (locBx * locBz < 0.f) {
        flip_x_ = true;
      }
      if (locBx < 0.f) {
        flip_y_ = true;
      }
      break;
    default:
      std::cout << "SiPixelTemplate:illegal subdetector ID = " << dtype_ << std::endl;
  }

  //  Calculate signed quantities

  //	qxtempcor corrects the total charge to the actual track angles (not actually needed for the template fits, but useful for Guofan)
  // &&& What to do here?
  // qxtempcor=std::sqrt((1.f+cotbeta*cotbeta+cotalpha*cotalpha)/(1.f+cotbeta0*cotbeta0+cotalpha*cotalpha));
  float qxtempcor = 1.f;

  for (int i = 0; i < 9; ++i) {
    xtemp_[i][0] = 0.f;
    xtemp_[i][1] = 0.f;
    xtemp_[i][BXM2] = 0.f;
    xtemp_[i][BXM1] = 0.f;
    for (int j = 0; j < TXSIZE; ++j) {
      if (flip_x_) {
        xtemp_[8 - i][BXM3 - j] = qxtempcor * entry_sideloaded_->xtemp[i][j];
      } else {
        xtemp_[i][j + 2] = qxtempcor * entry_sideloaded_->xtemp[i][j];
      }
    }
  }

  for (int i = 0; i < 9; ++i) {
    ytemp_[i][0] = 0.f;
    ytemp_[i][1] = 0.f;
    ytemp_[i][BYM2] = 0.f;
    ytemp_[i][BYM1] = 0.f;
    for (int j = 0; j < TYSIZE; ++j) {
      // Flip the basic y-template when the cotbeta is negative

      if (flip_y_) {
        ytemp_[8 - i][BYM3 - j] = entry_sideloaded_->ytemp[i][j];
      } else {
        ytemp_[i][j + 2] = entry_sideloaded_->ytemp[i][j];
      }
    }
  }

  // Fitted errors params
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 5; j++) {
      if (flip_y_) {
        yparl_[1 - i][j] = yparh_[1 - i][j] = entry->ypar[i][j];
      } else {
        yparl_[i][j] = yparh_[i][j] = entry->ypar[i][j];
      }

      if (flip_x_) {
        xpar0_[1 - i][j] = xparly0_[1 - i][j] = xparhy0_[1 - i][j] = xparl_[1 - i][j] = xparh_[1 - i][j] =
            entry->xpar[i][j];
      } else {
        xpar0_[i][j] = xparly0_[i][j] = xparhy0_[i][j] = xparl_[i][j] = xparh_[i][j] = entry->xpar[i][j];
      }
    }
  }

  return;
}  // sideload
#endif

// ************************************************************************************************************
//! Return vector of y errors (squared) for an input vector of projected signals
//! Add large Q scaling for use in cluster splitting.
//! \param fypix - (input) index of the first real pixel in the projected cluster (doesn't include pseudopixels)
//! \param lypix - (input) index of the last real pixel in the projected cluster (doesn't include pseudopixels)
//! \param sythr - (input) maximum signal before de-weighting
//! \param ysum - (input) 25-element vector of pixel signals
//! \param ysig2 - (output) 25-element vector of y errors (squared)
// ************************************************************************************************************
void SiPixelTemplate::ysigma2(int fypix, int lypix, float sythr, float ysum[25], float ysig2[25])

{
  // Interpolate using quantities already stored in the private variables

  // Local variables
  int i;
  float sigi, sigi2, sigi3, sigi4, symax, qscale, s25;

  // Make sure that input is OK

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (fypix < 2 || fypix >= BYM2) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ysigma2 called with fypix = " << fypix << std::endl;
  }
#else
  assert(fypix > 1 && fypix < BYM2);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (lypix < fypix || lypix >= BYM2) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ysigma2 called with lypix/fypix = " << lypix << "/"
                                        << fypix << std::endl;
  }
#else
  assert(lypix >= fypix && lypix < BYM2);
#endif

  // Define the maximum signal to use in the parameterization

  symax = symax_;
  s25 = 0.5f * s50_;
  if (symax_ > syparmax_) {
    symax = syparmax_;
  }

  // Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis

  for (i = fypix - 2; i <= lypix + 2; ++i) {
    if (i < fypix || i > lypix) {
      // Nearest pseudopixels have uncertainties of 50% of threshold, next-nearest have 10% of threshold

      ysig2[i] = s50_ * s50_;
    } else {
      if (ysum[i] < symax) {
        sigi = ysum[i];
        qscale = 1.f;
        if (sigi < s25)
          sigi = s25;
      } else {
        sigi = symax;
        qscale = ysum[i] / symax;
      }
      sigi2 = sigi * sigi;
      sigi3 = sigi2 * sigi;
      sigi4 = sigi3 * sigi;
      if (i <= BHY) {
        ysig2[i] = (1.f - yratio_) * (yparl_[0][0] + yparl_[0][1] * sigi + yparl_[0][2] * sigi2 + yparl_[0][3] * sigi3 +
                                      yparl_[0][4] * sigi4) +
                   yratio_ * (yparh_[0][0] + yparh_[0][1] * sigi + yparh_[0][2] * sigi2 + yparh_[0][3] * sigi3 +
                              yparh_[0][4] * sigi4);
      } else {
        ysig2[i] = (1.f - yratio_) * (yparl_[1][0] + yparl_[1][1] * sigi + yparl_[1][2] * sigi2 + yparl_[1][3] * sigi3 +
                                      yparl_[1][4] * sigi4) +
                   yratio_ * (yparh_[1][0] + yparh_[1][1] * sigi + yparh_[1][2] * sigi2 + yparh_[1][3] * sigi3 +
                              yparh_[1][4] * sigi4);
      }
      ysig2[i] *= qscale;
      if (ysum[i] > sythr) {
        ysig2[i] = 1.e8;
      }
      if (ysig2[i] <= 0.f) {
        LOGERROR("SiPixelTemplate") << "neg y-error-squared, id = " << id_current_ << ", index = " << index_id_
                                    << ", cot(alpha) = " << cota_current_ << ", cot(beta) = " << cotb_current_
                                    << ", sigi = " << sigi << ENDL;
      }
    }
  }

  return;

}  // End ysigma2

// ************************************************************************************************************
//! Return y error (squared) for an input signal and yindex
//! Add large Q scaling for use in cluster splitting.
//! \param qpixel - (input) pixel charge
//! \param index - (input) y-index index of pixel
//! \param ysig2 - (output) square error
// ************************************************************************************************************
void SiPixelTemplate::ysigma2(float qpixel, int index, float& ysig2)

{
  // Interpolate using quantities already stored in the private variables

  // Local variables
  float sigi, sigi2, sigi3, sigi4, symax, qscale, err2, s25;

  // Make sure that input is OK

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (index < 2 || index >= BYM2) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ysigma2 called with index = " << index << std::endl;
  }
#else
  assert(index > 1 && index < BYM2);
#endif

  // Define the maximum signal to use in the parameterization

  symax = symax_;
  s25 = 0.5f * s50_;
  if (symax_ > syparmax_) {
    symax = syparmax_;
  }

  // Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis

  if (qpixel < symax) {
    sigi = qpixel;
    qscale = 1.f;
    if (sigi < s25)
      sigi = s25;
  } else {
    sigi = symax;
    qscale = qpixel / symax;
  }
  sigi2 = sigi * sigi;
  sigi3 = sigi2 * sigi;
  sigi4 = sigi3 * sigi;
  if (index <= BHY) {
    err2 =
        (1.f - yratio_) *
            (yparl_[0][0] + yparl_[0][1] * sigi + yparl_[0][2] * sigi2 + yparl_[0][3] * sigi3 + yparl_[0][4] * sigi4) +
        yratio_ *
            (yparh_[0][0] + yparh_[0][1] * sigi + yparh_[0][2] * sigi2 + yparh_[0][3] * sigi3 + yparh_[0][4] * sigi4);
  } else {
    err2 =
        (1.f - yratio_) *
            (yparl_[1][0] + yparl_[1][1] * sigi + yparl_[1][2] * sigi2 + yparl_[1][3] * sigi3 + yparl_[1][4] * sigi4) +
        yratio_ *
            (yparh_[1][0] + yparh_[1][1] * sigi + yparh_[1][2] * sigi2 + yparh_[1][3] * sigi3 + yparh_[1][4] * sigi4);
  }
  ysig2 = qscale * err2;
  if (ysig2 <= 0.f) {
    LOGERROR("SiPixelTemplate") << "neg y-error-squared, id = " << id_current_ << ", index = " << index_id_
                                << ", cot(alpha) = " << cota_current_ << ", cot(beta) = " << cotb_current_
                                << ", sigi = " << sigi << ENDL;
  }

  return;

}  // End ysigma2

// ************************************************************************************************************
//! Return vector of x errors (squared) for an input vector of projected signals
//! Add large Q scaling for use in cluster splitting.
//! \param fxpix - (input) index of the first real pixel in the projected cluster (doesn't include pseudopixels)
//! \param lxpix - (input) index of the last real pixel in the projected cluster (doesn't include pseudopixels)
//! \param sxthr - (input) maximum signal before de-weighting
//! \param xsum - (input) 11-element vector of pixel signals
//! \param xsig2 - (output) 11-element vector of x errors (squared)
// ************************************************************************************************************
void SiPixelTemplate::xsigma2(int fxpix, int lxpix, float sxthr, float xsum[11], float xsig2[11])

{
  // Interpolate using quantities already stored in the private variables

  // Local variables
  int i;
  float sigi, sigi2, sigi3, sigi4, yint, sxmax, x0, qscale, s25;

  // Make sure that input is OK

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (fxpix < 2 || fxpix >= BXM2) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xsigma2 called with fxpix = " << fxpix << std::endl;
  }
#else
  assert(fxpix > 1 && fxpix < BXM2);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (lxpix < fxpix || lxpix >= BXM2) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xsigma2 called with lxpix/fxpix = " << lxpix << "/"
                                        << fxpix << std::endl;
  }
#else
  assert(lxpix >= fxpix && lxpix < BXM2);
#endif

  // Define the maximum signal to use in the parameterization

  sxmax = sxmax_;
  s25 = 0.5f * s50_;
  if (sxmax_ > sxparmax_) {
    sxmax = sxparmax_;
  }

  // Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis

  for (i = fxpix - 2; i <= lxpix + 2; ++i) {
    if (i < fxpix || i > lxpix) {
      // Nearest pseudopixels have uncertainties of 50% of threshold, next-nearest have 10% of threshold

      xsig2[i] = s50_ * s50_;
    } else {
      if (xsum[i] < sxmax) {
        sigi = xsum[i];
        qscale = 1.f;
        if (sigi < s25)
          sigi = s25;
      } else {
        sigi = sxmax;
        qscale = xsum[i] / sxmax;
      }
      sigi2 = sigi * sigi;
      sigi3 = sigi2 * sigi;
      sigi4 = sigi3 * sigi;

      // First, do the cotbeta interpolation

      if (i <= BHX) {
        yint = (1.f - yratio_) * (xparly0_[0][0] + xparly0_[0][1] * sigi + xparly0_[0][2] * sigi2 +
                                  xparly0_[0][3] * sigi3 + xparly0_[0][4] * sigi4) +
               yratio_ * (xparhy0_[0][0] + xparhy0_[0][1] * sigi + xparhy0_[0][2] * sigi2 + xparhy0_[0][3] * sigi3 +
                          xparhy0_[0][4] * sigi4);
      } else {
        yint = (1.f - yratio_) * (xparly0_[1][0] + xparly0_[1][1] * sigi + xparly0_[1][2] * sigi2 +
                                  xparly0_[1][3] * sigi3 + xparly0_[1][4] * sigi4) +
               yratio_ * (xparhy0_[1][0] + xparhy0_[1][1] * sigi + xparhy0_[1][2] * sigi2 + xparhy0_[1][3] * sigi3 +
                          xparhy0_[1][4] * sigi4);
      }

      // Next, do the cotalpha interpolation

      if (i <= BHX) {
        xsig2[i] = (1.f - xxratio_) * (xparl_[0][0] + xparl_[0][1] * sigi + xparl_[0][2] * sigi2 +
                                       xparl_[0][3] * sigi3 + xparl_[0][4] * sigi4) +
                   xxratio_ * (xparh_[0][0] + xparh_[0][1] * sigi + xparh_[0][2] * sigi2 + xparh_[0][3] * sigi3 +
                               xparh_[0][4] * sigi4);
      } else {
        xsig2[i] = (1.f - xxratio_) * (xparl_[1][0] + xparl_[1][1] * sigi + xparl_[1][2] * sigi2 +
                                       xparl_[1][3] * sigi3 + xparl_[1][4] * sigi4) +
                   xxratio_ * (xparh_[1][0] + xparh_[1][1] * sigi + xparh_[1][2] * sigi2 + xparh_[1][3] * sigi3 +
                               xparh_[1][4] * sigi4);
      }

      // Finally, get the mid-point value of the cotalpha function

      if (i <= BHX) {
        x0 = xpar0_[0][0] + xpar0_[0][1] * sigi + xpar0_[0][2] * sigi2 + xpar0_[0][3] * sigi3 + xpar0_[0][4] * sigi4;
      } else {
        x0 = xpar0_[1][0] + xpar0_[1][1] * sigi + xpar0_[1][2] * sigi2 + xpar0_[1][3] * sigi3 + xpar0_[1][4] * sigi4;
      }

      // Finally, rescale the yint value for cotalpha variation

      if (x0 != 0.f) {
        xsig2[i] = xsig2[i] / x0 * yint;
      }
      xsig2[i] *= qscale;
      if (xsum[i] > sxthr) {
        xsig2[i] = 1.e8;
      }
      if (xsig2[i] <= 0.f) {
        LOGERROR("SiPixelTemplate") << "neg x-error-squared, id = " << id_current_ << ", index = " << index_id_
                                    << ", cot(alpha) = " << cota_current_ << ", cot(beta) = " << cotb_current_
                                    << ", sigi = " << sigi << ENDL;
      }
    }
  }

  return;

}  // End xsigma2

// ************************************************************************************************************
//! Return interpolated y-correction for input charge bin and qfly
//! \param binq - (input) charge bin [0-3]
//! \param qfly - (input) (Q_f-Q_l)/(Q_f+Q_l) for this cluster
// ************************************************************************************************************
float SiPixelTemplate::yflcorr(int binq, float qfly)

{
  // Interpolate using quantities already stored in the private variables

  // Local variables
  float qfl, qfl2, qfl3, qfl4, qfl5, dy;

  // Make sure that input is OK

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (binq < 0 || binq > 3) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::yflcorr called with binq = " << binq << std::endl;
  }
#else
  assert(binq >= 0 && binq < 4);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (fabs((double)qfly) > 1.) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::yflcorr called with qfly = " << qfly << std::endl;
  }
#else
  assert(fabs((double)qfly) <= 1.);
#endif

  // Define the maximum signal to allow before de-weighting a pixel

  qfl = qfly;

  if (qfl < -0.9f) {
    qfl = -0.9f;
  }
  if (qfl > 0.9f) {
    qfl = 0.9f;
  }

  // Interpolate between the two polynomials

  qfl2 = qfl * qfl;
  qfl3 = qfl2 * qfl;
  qfl4 = qfl3 * qfl;
  qfl5 = qfl4 * qfl;
  dy = (1.f - yratio_) * (yflparl_[binq][0] + yflparl_[binq][1] * qfl + yflparl_[binq][2] * qfl2 +
                          yflparl_[binq][3] * qfl3 + yflparl_[binq][4] * qfl4 + yflparl_[binq][5] * qfl5) +
       yratio_ * (yflparh_[binq][0] + yflparh_[binq][1] * qfl + yflparh_[binq][2] * qfl2 + yflparh_[binq][3] * qfl3 +
                  yflparh_[binq][4] * qfl4 + yflparh_[binq][5] * qfl5);

  return dy;

}  // End yflcorr

// ************************************************************************************************************
//! Return interpolated x-correction for input charge bin and qflx
//! \param binq - (input) charge bin [0-3]
//! \param qflx - (input) (Q_f-Q_l)/(Q_f+Q_l) for this cluster
// ************************************************************************************************************
float SiPixelTemplate::xflcorr(int binq, float qflx)

{
  // Interpolate using quantities already stored in the private variables

  // Local variables
  float qfl, qfl2, qfl3, qfl4, qfl5, dx;

  // Make sure that input is OK

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (binq < 0 || binq > 3) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xflcorr called with binq = " << binq << std::endl;
  }
#else
  assert(binq >= 0 && binq < 4);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (fabs((double)qflx) > 1.) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xflcorr called with qflx = " << qflx << std::endl;
  }
#else
  assert(fabs((double)qflx) <= 1.);
#endif

  // Define the maximum signal to allow before de-weighting a pixel

  qfl = qflx;

  if (qfl < -0.9f) {
    qfl = -0.9f;
  }
  if (qfl > 0.9f) {
    qfl = 0.9f;
  }

  // Interpolate between the two polynomials

  qfl2 = qfl * qfl;
  qfl3 = qfl2 * qfl;
  qfl4 = qfl3 * qfl;
  qfl5 = qfl4 * qfl;
  dx = (1.f - yxratio_) *
           ((1.f - xxratio_) * (xflparll_[binq][0] + xflparll_[binq][1] * qfl + xflparll_[binq][2] * qfl2 +
                                xflparll_[binq][3] * qfl3 + xflparll_[binq][4] * qfl4 + xflparll_[binq][5] * qfl5) +
            xxratio_ * (xflparlh_[binq][0] + xflparlh_[binq][1] * qfl + xflparlh_[binq][2] * qfl2 +
                        xflparlh_[binq][3] * qfl3 + xflparlh_[binq][4] * qfl4 + xflparlh_[binq][5] * qfl5)) +
       yxratio_ *
           ((1.f - xxratio_) * (xflparhl_[binq][0] + xflparhl_[binq][1] * qfl + xflparhl_[binq][2] * qfl2 +
                                xflparhl_[binq][3] * qfl3 + xflparhl_[binq][4] * qfl4 + xflparhl_[binq][5] * qfl5) +
            xxratio_ * (xflparhh_[binq][0] + xflparhh_[binq][1] * qfl + xflparhh_[binq][2] * qfl2 +
                        xflparhh_[binq][3] * qfl3 + xflparhh_[binq][4] * qfl4 + xflparhh_[binq][5] * qfl5));

  return dx;

}  // End xflcorr

// ************************************************************************************************************
//! Return interpolated y-template in single call
//! \param fybin - (input) index of first bin (0-40) to fill
//! \param fybin - (input) index of last bin (0-40) to fill
//! \param ytemplate - (output) a 41x25 output buffer
// ************************************************************************************************************
void SiPixelTemplate::ytemp(int fybin, int lybin, float ytemplate[41][BYSIZE])

{
  // Retrieve already interpolated quantities

  // Local variables
  int i, j;

  // Verify that input parameters are in valid range

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (fybin < 0 || fybin > 40) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ytemp called with fybin = " << fybin << std::endl;
  }
#else
  assert(fybin >= 0 && fybin < 41);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (lybin < 0 || lybin > 40) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ytemp called with lybin = " << lybin << std::endl;
  }
#else
  assert(lybin >= 0 && lybin < 41);
#endif

  // Build the y-template, the central 25 bins are here in all cases

  for (i = 0; i < 9; ++i) {
    for (j = 0; j < BYSIZE; ++j) {
      ytemplate[i + 16][j] = ytemp_[i][j];
    }
  }
  for (i = 0; i < 8; ++i) {
    ytemplate[i + 8][BYM1] = 0.f;
    for (j = 0; j < BYM1; ++j) {
      ytemplate[i + 8][j] = ytemp_[i][j + 1];
    }
  }
  for (i = 1; i < 9; ++i) {
    ytemplate[i + 24][0] = 0.f;
    for (j = 0; j < BYM1; ++j) {
      ytemplate[i + 24][j + 1] = ytemp_[i][j];
    }
  }

  //  Add	more bins if needed

  if (fybin < 8) {
    for (i = 0; i < 8; ++i) {
      ytemplate[i][BYM2] = 0.f;
      ytemplate[i][BYM1] = 0.f;
      for (j = 0; j < BYM2; ++j) {
        ytemplate[i][j] = ytemp_[i][j + 2];
      }
    }
  }
  if (lybin > 32) {
    for (i = 1; i < 9; ++i) {
      ytemplate[i + 32][0] = 0.f;
      ytemplate[i + 32][1] = 0.f;
      for (j = 0; j < BYM2; ++j) {
        ytemplate[i + 32][j + 2] = ytemp_[i][j];
      }
    }
  }

  return;

}  // End ytemp

// ************************************************************************************************************
//! Return interpolated y-template in single call
//! \param fxbin - (input) index of first bin (0-40) to fill
//! \param fxbin - (input) index of last bin (0-40) to fill
//! \param xtemplate - (output) a 41x11 output buffer
// ************************************************************************************************************
void SiPixelTemplate::xtemp(int fxbin, int lxbin, float xtemplate[41][BXSIZE]) {
  // Retrieve already interpolated quantities

  // Local variables
  int i, j;

  // Verify that input parameters are in valid range

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (fxbin < 0 || fxbin > 40) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xtemp called with fxbin = " << fxbin << std::endl;
  }
#else
  assert(fxbin >= 0 && fxbin < 41);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (lxbin < 0 || lxbin > 40) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xtemp called with lxbin = " << lxbin << std::endl;
  }
#else
  assert(lxbin >= 0 && lxbin < 41);
#endif

  // Build the x-template, the central 25 bins are here in all cases

  for (i = 0; i < 9; ++i) {
    for (j = 0; j < BXSIZE; ++j) {
      xtemplate[i + 16][j] = xtemp_[i][j];
    }
  }
  for (i = 0; i < 8; ++i) {
    xtemplate[i + 8][BXM1] = 0.f;
    for (j = 0; j < BXM1; ++j) {
      xtemplate[i + 8][j] = xtemp_[i][j + 1];
    }
  }
  for (i = 1; i < 9; ++i) {
    xtemplate[i + 24][0] = 0.f;
    for (j = 0; j < BXM1; ++j) {
      xtemplate[i + 24][j + 1] = xtemp_[i][j];
    }
  }
  //  Add more bins if needed
  if (fxbin < 8) {
    for (i = 0; i < 8; ++i) {
      xtemplate[i][BXM2] = 0.f;
      xtemplate[i][BXM1] = 0.f;
      for (j = 0; j < BXM2; ++j) {
        xtemplate[i][j] = xtemp_[i][j + 2];
      }
    }
  }
  if (lxbin > 32) {
    for (i = 1; i < 9; ++i) {
      xtemplate[i + 32][0] = 0.f;
      xtemplate[i + 32][1] = 0.f;
      for (j = 0; j < BXM2; ++j) {
        xtemplate[i + 32][j + 2] = xtemp_[i][j];
      }
    }
  }

  return;

}  // End xtemp

// ************************************************************************************************************
//! Return central pixel of y template pixels above readout threshold
// ************************************************************************************************************
int SiPixelTemplate::cytemp()

{
  // Retrieve already interpolated quantities

  // Local variables
  int j;

  // Analyze only pixels along the central entry
  // First, find the maximum signal and then work out to the edges

  float sigmax = 0.f;
  float qedge = 2. * s50_;
  int jmax = -1;

  for (j = 0; j < BYSIZE; ++j) {
    if (ytemp_[4][j] > sigmax) {
      sigmax = ytemp_[4][j];
      jmax = j;
    }
  }
  if (sigmax < qedge) {
    qedge = s50_;
  }
  if (sigmax < qedge || jmax < 1 || jmax > BYM2) {
    return -1;
  }

  //  Now search forward and backward

  int jend = jmax;

  for (j = jmax + 1; j < BYM1; ++j) {
    if (ytemp_[4][j] < qedge)
      break;
    jend = j;
  }

  int jbeg = jmax;

  for (j = jmax - 1; j > 0; --j) {
    if (ytemp_[4][j] < qedge)
      break;
    jbeg = j;
  }

  return (jbeg + jend) / 2;

}  // End cytemp

// ************************************************************************************************************
//! Return central pixel of x-template pixels above readout threshold
// ************************************************************************************************************
int SiPixelTemplate::cxtemp()

{
  // Retrieve already interpolated quantities

  // Local variables
  int j;

  // Analyze only pixels along the central entry
  // First, find the maximum signal and then work out to the edges

  float sigmax = 0.f;
  float qedge = 2. * s50_;
  int jmax = -1;

  for (j = 0; j < BXSIZE; ++j) {
    if (xtemp_[4][j] > sigmax) {
      sigmax = xtemp_[4][j];
      jmax = j;
    }
  }
  if (sigmax < qedge) {
    qedge = s50_;
  }
  if (sigmax < qedge || jmax < 1 || jmax > BXM2) {
    return -1;
  }

  //  Now search forward and backward

  int jend = jmax;

  for (j = jmax + 1; j < BXM1; ++j) {
    if (xtemp_[4][j] < qedge)
      break;
    jend = j;
  }

  int jbeg = jmax;

  for (j = jmax - 1; j > 0; --j) {
    if (xtemp_[4][j] < qedge)
      break;
    jbeg = j;
  }

  return (jbeg + jend) / 2;

}  // End cxtemp

// ************************************************************************************************************
//! Make interpolated 3d y-template (stored as class variables)
//! \param nypix - (input) number of pixels in cluster (needed to size template)
//! \param nybins - (output) number of bins needed for each template projection
// ************************************************************************************************************
void SiPixelTemplate::ytemp3d_int(int nypix, int& nybins)

{
  // Retrieve already interpolated quantities

  // Local variables
  int i, j, k;
  int ioff0, ioffp, ioffm;

  // Verify that input parameters are in valid range

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (nypix < 1 || nypix >= BYM3) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ytemp3d called with nypix = " << nypix << std::endl;
  }
#else
  assert(nypix > 0 && nypix < BYM3);
#endif

  // Calculate the size of the shift in pixels needed to span the entire cluster

  float diff = fabsf(nypix - clsleny_) / 2. + 1.f;
  int nshift = (int)diff;
  if ((diff - nshift) > 0.5f) {
    ++nshift;
  }

  // Calculate the number of bins needed to specify each hit range

  nybins_ = 9 + 16 * nshift;

  // Create a 2-d working template with the correct size

  temp2dy_.resize(boost::extents[nybins_][BYSIZE]);

  //  The 9 central bins are copied from the interpolated private store

  ioff0 = 8 * nshift;

  for (i = 0; i < 9; ++i) {
    for (j = 0; j < BYSIZE; ++j) {
      temp2dy_[i + ioff0][j] = ytemp_[i][j];
    }
  }

  // Add the +- shifted templates

  for (k = 1; k <= nshift; ++k) {
    ioffm = ioff0 - k * 8;
    for (i = 0; i < 8; ++i) {
      for (j = 0; j < k; ++j) {
        temp2dy_[i + ioffm][BYM1 - j] = 0.f;
      }
      for (j = 0; j < BYSIZE - k; ++j) {
        temp2dy_[i + ioffm][j] = ytemp_[i][j + k];
      }
    }
    ioffp = ioff0 + k * 8;
    for (i = 1; i < 9; ++i) {
      for (j = 0; j < k; ++j) {
        temp2dy_[i + ioffp][j] = 0.f;
      }
      for (j = 0; j < BYSIZE - k; ++j) {
        temp2dy_[i + ioffp][j + k] = ytemp_[i][j];
      }
    }
  }

  nybins = nybins_;
  return;

}  // End ytemp3d_int

// ************************************************************************************************************
//! Return interpolated 3d y-template in single call
//! \param i,j - (input) template indices
//! \param ytemplate - (output) a boost 3d array containing two sets of temlate indices and the combined pixel signals
// ************************************************************************************************************
void SiPixelTemplate::ytemp3d(int i, int j, std::vector<float>& ytemplate)

{
  // Sum two 2-d templates to make the 3-d template
  if (i >= 0 && i < nybins_ && j <= i) {
    for (int k = 0; k < BYSIZE; ++k) {
      ytemplate[k] = temp2dy_[i][k] + temp2dy_[j][k];
    }
  } else {
    for (int k = 0; k < BYSIZE; ++k) {
      ytemplate[k] = 0.;
    }
  }

  return;

}  // End ytemp3d

// ************************************************************************************************************
//! Make interpolated 3d x-template (stored as class variables)
//! \param nxpix - (input) number of pixels in cluster (needed to size template)
//! \param nxbins - (output) number of bins needed for each template projection
// ************************************************************************************************************
void SiPixelTemplate::xtemp3d_int(int nxpix, int& nxbins)

{
  // Retrieve already interpolated quantities

  // Local variables
  int i, j, k;
  int ioff0, ioffp, ioffm;

  // Verify that input parameters are in valid range

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (nxpix < 1 || nxpix >= BXM3) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xtemp3d called with nxpix = " << nxpix << std::endl;
  }
#else
  assert(nxpix > 0 && nxpix < BXM3);
#endif

  // Calculate the size of the shift in pixels needed to span the entire cluster

  float diff = fabsf(nxpix - clslenx_) / 2.f + 1.f;
  int nshift = (int)diff;
  if ((diff - nshift) > 0.5f) {
    ++nshift;
  }

  // Calculate the number of bins needed to specify each hit range

  nxbins_ = 9 + 16 * nshift;

  // Create a 2-d working template with the correct size

  temp2dx_.resize(boost::extents[nxbins_][BXSIZE]);

  //  The 9 central bins are copied from the interpolated private store

  ioff0 = 8 * nshift;

  for (i = 0; i < 9; ++i) {
    for (j = 0; j < BXSIZE; ++j) {
      temp2dx_[i + ioff0][j] = xtemp_[i][j];
    }
  }

  // Add the +- shifted templates

  for (k = 1; k <= nshift; ++k) {
    ioffm = ioff0 - k * 8;
    for (i = 0; i < 8; ++i) {
      for (j = 0; j < k; ++j) {
        temp2dx_[i + ioffm][BXM1 - j] = 0.f;
      }
      for (j = 0; j < BXSIZE - k; ++j) {
        temp2dx_[i + ioffm][j] = xtemp_[i][j + k];
      }
    }
    ioffp = ioff0 + k * 8;
    for (i = 1; i < 9; ++i) {
      for (j = 0; j < k; ++j) {
        temp2dx_[i + ioffp][j] = 0.f;
      }
      for (j = 0; j < BXSIZE - k; ++j) {
        temp2dx_[i + ioffp][j + k] = xtemp_[i][j];
      }
    }
  }

  nxbins = nxbins_;

  return;

}  // End xtemp3d_int

// ************************************************************************************************************
//! Return interpolated 3d x-template in single call
//! \param i,j - (input) template indices
//! \param xtemplate - (output) a boost 3d array containing two sets of temlate indices and the combined pixel signals
// ************************************************************************************************************
void SiPixelTemplate::xtemp3d(int i, int j, std::vector<float>& xtemplate)

{
  // Sum two 2-d templates to make the 3-d template
  if (i >= 0 && i < nxbins_ && j <= i) {
    for (int k = 0; k < BXSIZE; ++k) {
      xtemplate[k] = temp2dx_[i][k] + temp2dx_[j][k];
    }
  } else {
    for (int k = 0; k < BXSIZE; ++k) {
      xtemplate[k] = 0.;
    }
  }

  return;

}  // End xtemp3d

// ************************************************************************************************************
//! Interpolate beta/alpha angles to produce an expected average charge. Return int (0-4) describing the charge
//! of the cluster [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin].
//! \param id - (input) index of the template to use
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param locBz - (input) the sign of this quantity is used to determine whether to flip cot(beta)<0 quantities from cot(beta)>0 (FPix only)
//!                    for Phase 0 FPix IP-related tracks, locBz < 0 for cot(beta) > 0 and locBz > 0 for cot(beta) < 0
//!                    for Phase 1 FPix IP-related tracks, see next comment
//! \param locBx - (input) the sign of this quantity is used to determine whether to flip cot(alpha/beta)<0 quantities from cot(alpha/beta)>0 (FPix only)
//!                    for Phase 1 FPix IP-related tracks, locBx/locBz > 0 for cot(alpha) > 0 and locBx/locBz < 0 for cot(alpha) < 0
//!                    for Phase 1 FPix IP-related tracks, locBx > 0 for cot(beta) > 0 and locBx < 0 for cot(beta) < 0//!
//! \param qclus - (input) the cluster charge in electrons
//! \param pixmax - (output) the maximum pixel charge in electrons (truncation value)
//! \param sigmay - (output) the estimated y-error for CPEGeneric in microns
//! \param deltay - (output) the estimated y-bias for CPEGeneric in microns
//! \param sigmax - (output) the estimated x-error for CPEGeneric in microns
//! \param deltax - (output) the estimated x-bias for CPEGeneric in microns
//! \param sy1 - (output) the estimated y-error for 1 single-pixel clusters in microns
//! \param dy1 - (output) the estimated y-bias for 1 single-pixel clusters in microns
//! \param sy2 - (output) the estimated y-error for 1 double-pixel clusters in microns
//! \param dy2 - (output) the estimated y-bias for 1 double-pixel clusters in microns
//! \param sx1 - (output) the estimated x-error for 1 single-pixel clusters in microns
//! \param dx1 - (output) the estimated x-bias for 1 single-pixel clusters in microns
//! \param sx2 - (output) the estimated x-error for 1 double-pixel clusters in microns
//! \param dx2 - (output) the estimated x-bias for 1 double-pixel clusters in microns
//! \param lorywidth - (output) the estimated y Lorentz width
//! \param lorxwidth - (output) the estimated x Lorentz width
// ************************************************************************************************************
int SiPixelTemplate::qbin(int id,
                          float cotalpha,
                          float cotbeta,
                          float locBz,
                          float locBx,
                          float qclus,
                          float& pixmx,
                          float& sigmay,
                          float& deltay,
                          float& sigmax,
                          float& deltax,
                          float& sy1,
                          float& dy1,
                          float& sy2,
                          float& dy2,
                          float& sx1,
                          float& dx1,
                          float& sx2,
                          float& dx2)  // float& lorywidth, float& lorxwidth)
{
  // Interpolate for a new set of track angles

  // &&& New approach: use cached pointers.

  // Find the index corresponding to id

  int index = -1;
  for (int i = 0; i < (int)thePixelTemp_.size(); ++i) {
    if (id == thePixelTemp_[i].head.ID) {
      index = i;
      break;
    }
  }

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (index < 0 || index >= (int)thePixelTemp_.size()) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::qbin can't find needed template ID = " << id << std::endl;
  }
#else
  assert(index >= 0 && index < (int)thePixelTemp_.size());
#endif

  //

  auto const& templ = thePixelTemp_[index];

  // Interpolate the absolute value of cot(beta)

  auto acotb = std::abs(cotbeta);

  //	qcorrect corrects the cot(alpha)=0 cluster charge for non-zero cot(alpha)

  auto cotalpha0 = thePixelTemp_[index].enty[0].cotalpha;
  auto qcorrect =
      std::sqrt((1.f + cotbeta * cotbeta + cotalpha * cotalpha) / (1.f + cotbeta * cotbeta + cotalpha0 * cotalpha0));

  // for some cosmics, the ususal gymnastics are incorrect

  float cota = cotalpha;
  bool flip_x = false;
  //y flipping already taken care of by interpolate
  switch (dtype_) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
      if (locBx * locBz < 0.f) {
        cota = -cotalpha;
        flip_x = true;
      }
      break;
    default:
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
      throw cms::Exception("DataCorrupt")
          << "SiPixelTemplate::illegal subdetector ID = " << thePixelTemp_[index_id_].head.Dtype << std::endl;
#else
      std::cout << "SiPixelTemplate::illegal subdetector ID = " << thePixelTemp_[index_id_].head.Dtype << std::endl;
#endif
  }

  // Copy the charge scaling factor to the private variable

  auto qscale = thePixelTemp_[index].head.qscale;

  /*
    lorywidth = thePixelTemp_[index].head.lorywidth;
    if(locBz > 0.f) {lorywidth = -lorywidth;}
    lorxwidth = thePixelTemp_[index].head.lorxwidth;
    */

  auto Ny = thePixelTemp_[index].head.NTy;
  auto Nyx = thePixelTemp_[index].head.NTyx;
  auto Nxx = thePixelTemp_[index].head.NTxx;

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (Ny < 2 || Nyx < 1 || Nxx < 2) {
    throw cms::Exception("DataCorrupt") << "template ID = " << id_current_ << "has too few entries: Ny/Nyx/Nxx = " << Ny
                                        << "/" << Nyx << "/" << Nxx << std::endl;
  }
#else
  assert(Ny > 1 && Nyx > 0 && Nxx > 1);
#endif

  // Interpolate/store all y-related quantities (flip displacements when flip_y)

  dy1 = (1.f - yratio_) * enty0_->dyone + yratio_ * enty1_->dyone;
  if (flip_y_) {
    dy1 = -dy1;
  }
  sy1 = (1.f - yratio_) * enty0_->syone + yratio_ * enty1_->syone;
  dy2 = (1.f - yratio_) * enty0_->dytwo + yratio_ * enty1_->dytwo;
  if (flip_y_) {
    dy2 = -dy2;
  }
  sy2 = (1.f - yratio_) * enty0_->sytwo + yratio_ * enty1_->sytwo;

  auto qavg = (1.f - yratio_) * enty0_->qavg + yratio_ * enty1_->qavg;
  qavg *= qcorrect;
  auto qmin = (1.f - yratio_) * enty0_->qmin + yratio_ * enty1_->qmin;
  qmin *= qcorrect;
  auto qmin2 = (1.f - yratio_) * enty0_->qmin2 + yratio_ * enty1_->qmin2;
  qmin2 *= qcorrect;

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (qavg <= 0.f || qmin <= 0.f) {
    throw cms::Exception("DataCorrupt")
        << "SiPixelTemplate::qbin, qavg or qmin <= 0,"
        << " Probably someone called the generic pixel reconstruction with an illegal trajectory state" << std::endl;
  }
#else
  assert(qavg > 0.f && qmin > 0.f);
#endif

  //  Scale the input charge to account for differences between pixelav and CMSSW simulation or data

  auto qtotal = qscale * qclus;

  // uncertainty and final corrections depend upon total charge bin
  auto fq = qtotal / qavg;
  int binq;

  if (fq > fbin_[0]) {
    binq = 0;
    //std::cout<<" fq "<<fq<<" "<<qtotal<<" "<<qavg<<" "<<qclus<<" "<<qscale<<" "
    //       <<fbin_[0]<<" "<<fbin_[1]<<" "<<fbin_[2]<<std::endl;
  } else {
    if (fq > fbin_[1]) {
      binq = 1;
    } else {
      if (fq > fbin_[2]) {
        binq = 2;
      } else {
        binq = 3;
      }
    }
  }

  auto yavggen = (1.f - yratio_) * enty0_->yavggen[binq] + yratio_ * enty1_->yavggen[binq];
  if (flip_y_) {
    yavggen = -yavggen;
  }
  auto yrmsgen = (1.f - yratio_) * enty0_->yrmsgen[binq] + yratio_ * enty1_->yrmsgen[binq];

  // next, loop over all x-angle entries, first, find relevant y-slices

  auto iylow = 0;
  auto iyhigh = 0;
  auto yxratio = 0.f;

  {
    auto j = std::lower_bound(templ.cotbetaX.begin(), templ.cotbetaX.begin() + Nyx, acotb);
    if (j == templ.cotbetaX.begin() + Nyx) {
      --j;
      yxratio = 1.f;
    } else if (j == templ.cotbetaX.begin()) {
      ++j;
      yxratio = 0.f;
    } else {
      yxratio = (acotb - (*(j - 1))) / ((*j) - (*(j - 1)));
    }

    iyhigh = j - templ.cotbetaX.begin();
    iylow = iyhigh - 1;
  }

  auto ilow = 0;
  auto ihigh = 0;
  auto xxratio = 0.f;

  {
    auto j = std::lower_bound(templ.cotalphaX.begin(), templ.cotalphaX.begin() + Nxx, cota);
    if (j == templ.cotalphaX.begin() + Nxx) {
      --j;
      xxratio = 1.f;
    } else if (j == templ.cotalphaX.begin()) {
      ++j;
      xxratio = 0.f;
    } else {
      xxratio = (cota - (*(j - 1))) / ((*j) - (*(j - 1)));
    }

    ihigh = j - templ.cotalphaX.begin();
    ilow = ihigh - 1;
  }

  dx1 =
      (1.f - xxratio) * thePixelTemp_[index].entx[0][ilow].dxone + xxratio * thePixelTemp_[index].entx[0][ihigh].dxone;
  if (flip_x) {
    dx1 = -dx1;
  }
  sx1 =
      (1.f - xxratio) * thePixelTemp_[index].entx[0][ilow].sxone + xxratio * thePixelTemp_[index].entx[0][ihigh].sxone;
  dx2 =
      (1.f - xxratio) * thePixelTemp_[index].entx[0][ilow].dxtwo + xxratio * thePixelTemp_[index].entx[0][ihigh].dxtwo;
  if (flip_x) {
    dx2 = -dx2;
  }
  sx2 =
      (1.f - xxratio) * thePixelTemp_[index].entx[0][ilow].sxtwo + xxratio * thePixelTemp_[index].entx[0][ihigh].sxtwo;

  // pixmax is the maximum allowed pixel charge (used for truncation)

  pixmx = (1.f - yxratio) * ((1.f - xxratio) * thePixelTemp_[index].entx[iylow][ilow].pixmax +
                             xxratio * thePixelTemp_[index].entx[iylow][ihigh].pixmax) +
          yxratio * ((1.f - xxratio) * thePixelTemp_[index].entx[iyhigh][ilow].pixmax +
                     xxratio * thePixelTemp_[index].entx[iyhigh][ihigh].pixmax);

  auto xavggen = (1.f - yxratio) * ((1.f - xxratio) * thePixelTemp_[index].entx[iylow][ilow].xavggen[binq] +
                                    xxratio * thePixelTemp_[index].entx[iylow][ihigh].xavggen[binq]) +
                 yxratio * ((1.f - xxratio) * thePixelTemp_[index].entx[iyhigh][ilow].xavggen[binq] +
                            xxratio * thePixelTemp_[index].entx[iyhigh][ihigh].xavggen[binq]);
  if (flip_x) {
    xavggen = -xavggen;
  }

  auto xrmsgen = (1.f - yxratio) * ((1.f - xxratio) * thePixelTemp_[index].entx[iylow][ilow].xrmsgen[binq] +
                                    xxratio * thePixelTemp_[index].entx[iylow][ihigh].xrmsgen[binq]) +
                 yxratio * ((1.f - xxratio) * thePixelTemp_[index].entx[iyhigh][ilow].xrmsgen[binq] +
                            xxratio * thePixelTemp_[index].entx[iyhigh][ihigh].xrmsgen[binq]);

  //  Take the errors and bias from the correct charge bin

  sigmay = yrmsgen;
  deltay = yavggen;

  sigmax = xrmsgen;
  deltax = xavggen;

  // If the charge is too small (then flag it)

  if (qtotal < 0.95f * qmin) {
    binq = 5;
  } else {
    if (qtotal < 0.95f * qmin2) {
      binq = 4;
    }
  }

  return binq;

}  // qbin

// ************************************************************************************************************
//! Interpolate beta/alpha angles to produce an expected average charge. Return int (0-4) describing the charge
//! of the cluster [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin].
//! \param id - (input) index of the template to use
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param locBz - (input) the sign of this quantity is used to determine whether to flip cot(beta)<0 quantities from cot(beta)>0 (FPix only)
//!                    for Phase 0 FPix IP-related tracks, locBz < 0 for cot(beta) > 0 and locBz > 0 for cot(beta) < 0
//!                    for Phase 1 FPix IP-related tracks, see next comment
//! \param locBx - (input) the sign of this quantity is used to determine whether to flip cot(alpha/beta)<0 quantities from cot(alpha/beta)>0 (FPix only)
//!                    for Phase 1 FPix IP-related tracks, locBx/locBz > 0 for cot(alpha) > 0 and locBx/locBz < 0 for cot(alpha) < 0
//!                    for Phase 1 FPix IP-related tracks, locBx > 0 for cot(beta) > 0 and locBx < 0 for cot(beta) < 0//!
//! \param qclus - (input) the cluster charge in electrons
//! \param pixmax - (output) the maximum pixel charge in electrons (truncation value)
//! \param sigmay - (output) the estimated y-error for CPEGeneric in microns
//! \param deltay - (output) the estimated y-bias for CPEGeneric in microns
//! \param sigmax - (output) the estimated x-error for CPEGeneric in microns
//! \param deltax - (output) the estimated x-bias for CPEGeneric in microns
//! \param sy1 - (output) the estimated y-error for 1 single-pixel clusters in microns
//! \param dy1 - (output) the estimated y-bias for 1 single-pixel clusters in microns
//! \param sy2 - (output) the estimated y-error for 1 double-pixel clusters in microns
//! \param dy2 - (output) the estimated y-bias for 1 double-pixel clusters in microns
//! \param sx1 - (output) the estimated x-error for 1 single-pixel clusters in microns
//! \param dx1 - (output) the estimated x-bias for 1 single-pixel clusters in microns
//! \param sx2 - (output) the estimated x-error for 1 double-pixel clusters in microns
//! \param dx2 - (output) the estimated x-bias for 1 double-pixel clusters in microns
//! \param lorywidth - (output) the estimated y Lorentz width
//! \param lorxwidth - (output) the estimated x Lorentz width
// ************************************************************************************************************
int SiPixelTemplate::qbin(int id,
                          float cotalpha,
                          float cotbeta,
                          float locBz,
                          float qclus,
                          float& pixmx,
                          float& sigmay,
                          float& deltay,
                          float& sigmax,
                          float& deltax,
                          float& sy1,
                          float& dy1,
                          float& sy2,
                          float& dy2,
                          float& sx1,
                          float& dx1,
                          float& sx2,
                          float& dx2)  // float& lorywidth, float& lorxwidth)
{
  // Interpolate for a new set of track angles

  // Local variables
  float locBx = 1.f;  //  lorywidth, lorxwidth;

  return SiPixelTemplate::qbin(id,
                               cotalpha,
                               cotbeta,
                               locBz,
                               locBx,
                               qclus,
                               pixmx,
                               sigmay,
                               deltay,
                               sigmax,
                               deltax,
                               sy1,
                               dy1,
                               sy2,
                               dy2,
                               sx1,
                               dx1,
                               sx2,
                               dx2);  // , lorywidth, lorxwidth);

}  // qbin

// ************************************************************************************************************
//! Interpolate beta/alpha angles to produce an expected average charge. Return int (0-4) describing the charge
//! of the cluster [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin].
//! \param id - (input) index of the template to use
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param locBz - (input) the sign of this quantity is used to determine whether to flip cot(beta)<0 quantities from cot(beta)>0 (FPix only)
//!                    for FPix IP-related tracks, locBz < 0 for cot(beta) > 0 and locBz > 0 for cot(beta) < 0
//! \param qclus - (input) the cluster charge in electrons
//! \param pixmax - (output) the maximum pixel charge in electrons (truncation value)
//! \param sigmay - (output) the estimated y-error for CPEGeneric in microns
//! \param deltay - (output) the estimated y-bias for CPEGeneric in microns
//! \param sigmax - (output) the estimated x-error for CPEGeneric in microns
//! \param deltax - (output) the estimated x-bias for CPEGeneric in microns
//! \param sy1 - (output) the estimated y-error for 1 single-pixel clusters in microns
//! \param dy1 - (output) the estimated y-bias for 1 single-pixel clusters in microns
//! \param sy2 - (output) the estimated y-error for 1 double-pixel clusters in microns
//! \param dy2 - (output) the estimated y-bias for 1 double-pixel clusters in microns
//! \param sx1 - (output) the estimated x-error for 1 single-pixel clusters in microns
//! \param dx1 - (output) the estimated x-bias for 1 single-pixel clusters in microns
//! \param sx2 - (output) the estimated x-error for 1 double-pixel clusters in microns
//! \param dx2 - (output) the estimated x-bias for 1 double-pixel clusters in microns
// ************************************************************************************************************
/*
 int SiPixelTemplate::qbin(int id, float cotalpha, float cotbeta, float locBz, float qclus, float& pixmx, float& sigmay, float& deltay, float& sigmax, float& deltax,
 float& sy1, float& dy1, float& sy2, float& dy2, float& sx1, float& dx1, float& sx2, float& dx2)
 
 {
	float lorywidth, lorxwidth;
	return SiPixelTemplate::qbin(id, cotalpha, cotbeta, locBz, qclus, pixmx, sigmay, deltay, sigmax, deltax,
 sy1, dy1, sy2, dy2, sx1, dx1, sx2, dx2, lorywidth, lorxwidth);
	
 } // qbin
 */

// ************************************************************************************************************
//! Interpolate beta/alpha angles to produce an expected average charge. Return int (0-4) describing the charge
//! of the cluster [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin].
//! \param id - (input) index of the template to use
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param qclus - (input) the cluster charge in electrons
// ************************************************************************************************************
int SiPixelTemplate::qbin(int id, float cotalpha, float cotbeta, float qclus) {
  // Interpolate for a new set of track angles

  // Local variables
  float pixmx, sigmay, deltay, sigmax, deltax, sy1, dy1, sy2, dy2, sx1, dx1, sx2, dx2, locBz;  //  lorywidth, lorxwidth;
  // Local variables
  float locBx = 1.f;
  if (cotbeta < 0.f) {
    locBx = -1.f;
  }
  locBz = locBx;
  if (cotalpha < 0.f) {
    locBz = -locBx;
  }

  return SiPixelTemplate::qbin(id,
                               cotalpha,
                               cotbeta,
                               locBz,
                               locBx,
                               qclus,
                               pixmx,
                               sigmay,
                               deltay,
                               sigmax,
                               deltax,
                               sy1,
                               dy1,
                               sy2,
                               dy2,
                               sx1,
                               dx1,
                               sx2,
                               dx2);  // , lorywidth, lorxwidth);

}  // qbin

// ************************************************************************************************************
//! Interpolate beta/alpha angles to produce an expected average charge. Return int (0-4) describing the charge
//! of the cluster [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin].
//! \param id - (input) index of the template to use
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param qclus - (input) the cluster charge in electrons
// ************************************************************************************************************
int SiPixelTemplate::qbin(int id, float cotbeta, float qclus) {
  // Interpolate for a new set of track angles

  // Local variables
  float pixmx, sigmay, deltay, sigmax, deltax, sy1, dy1, sy2, dy2, sx1, dx1, sx2, dx2, locBz,
      locBx;  //, lorywidth, lorxwidth;
  const float cotalpha = 0.f;
  locBx = 1.f;
  if (cotbeta < 0.f) {
    locBx = -1.f;
  }
  locBz = locBx;
  if (cotalpha < 0.f) {
    locBz = -locBx;
  }
  return SiPixelTemplate::qbin(id,
                               cotalpha,
                               cotbeta,
                               locBz,
                               locBx,
                               qclus,
                               pixmx,
                               sigmay,
                               deltay,
                               sigmax,
                               deltax,
                               sy1,
                               dy1,
                               sy2,
                               dy2,
                               sx1,
                               dx1,
                               sx2,
                               dx2);  // , lorywidth, lorxwidth);

}  // qbin

// ************************************************************************************************************
//! Interpolate beta/alpha angles to produce estimated errors for fastsim
//! \param id - (input) index of the template to use
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param qBin - (input) charge bin from 0-3
//! \param sigmay - (output) the estimated y-error for CPETemplate in microns
//! \param sigmax - (output) the estimated x-error for CPETemplate in microns
//! \param sy1 - (output) the estimated y-error for 1 single-pixel clusters in microns
//! \param sy2 - (output) the estimated y-error for 1 double-pixel clusters in microns
//! \param sx1 - (output) the estimated x-error for 1 single-pixel clusters in microns
//! \param sx2 - (output) the estimated x-error for 1 double-pixel clusters in microns
// ************************************************************************************************************
void SiPixelTemplate::temperrors(int id,
                                 float cotalpha,
                                 float cotbeta,
                                 int qBin,
                                 float& sigmay,
                                 float& sigmax,
                                 float& sy1,
                                 float& sy2,
                                 float& sx1,
                                 float& sx2)

{
  // Interpolate for a new set of track angles

  // Local variables
  int i;
  int ilow, ihigh, iylow, iyhigh, Ny, Nxx, Nyx, index;
  float yxratio, xxratio;
  float acotb;
  float yrms, xrms;
  //bool flip_y;

  // Find the index corresponding to id

  index = -1;
  for (i = 0; i < (int)thePixelTemp_.size(); ++i) {
    if (id == thePixelTemp_[i].head.ID) {
      index = i;
      break;
    }
  }

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (index < 0 || index >= (int)thePixelTemp_.size()) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::temperrors can't find needed template ID = " << id
                                        << std::endl;
  }
#else
  assert(index >= 0 && index < (int)thePixelTemp_.size());
#endif

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (qBin < 0 || qBin > 5) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::temperrors called with illegal qBin = " << qBin
                                        << std::endl;
  }
#else
  assert(qBin >= 0 && qBin < 6);
#endif

  // The error information for qBin > 3 is taken to be the same as qBin=3

  if (qBin > 3) {
    qBin = 3;
  }
  //

  // Interpolate the absolute value of cot(beta)

  acotb = std::abs(cotbeta);

  // Copy the charge scaling factor to the private variable

  Ny = thePixelTemp_[index].head.NTy;
  Nyx = thePixelTemp_[index].head.NTyx;
  Nxx = thePixelTemp_[index].head.NTxx;

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (Ny < 2 || Nyx < 1 || Nxx < 2) {
    throw cms::Exception("DataCorrupt") << "template ID = " << id_current_ << "has too few entries: Ny/Nyx/Nxx = " << Ny
                                        << "/" << Nyx << "/" << Nxx << std::endl;
  }
#else
  assert(Ny > 1 && Nyx > 0 && Nxx > 1);
#endif

  // next, loop over all y-angle entries

  // Interpolate/store all y-related quantities (flip displacements when flip_y)

  sy1 = (1.f - yratio_) * enty0_->syone + yratio_ * enty1_->syone;
  sy2 = (1.f - yratio_) * enty0_->sytwo + yratio_ * enty1_->sytwo;
  yrms = (1.f - yratio_) * enty0_->yrms[qBin] + yratio_ * enty1_->yrms[qBin];

  // next, loop over all x-angle entries, first, find relevant y-slices

  iylow = 0;
  yxratio = 0.f;

  if (acotb >= thePixelTemp_[index].entx[Nyx - 1][0].cotbeta) {
    iylow = Nyx - 2;
    yxratio = 1.f;

  } else if (acotb >= thePixelTemp_[index].entx[0][0].cotbeta) {
    for (i = 0; i < Nyx - 1; ++i) {
      if (thePixelTemp_[index].entx[i][0].cotbeta <= acotb && acotb < thePixelTemp_[index].entx[i + 1][0].cotbeta) {
        iylow = i;
        yxratio = (acotb - thePixelTemp_[index].entx[i][0].cotbeta) /
                  (thePixelTemp_[index].entx[i + 1][0].cotbeta - thePixelTemp_[index].entx[i][0].cotbeta);
        break;
      }
    }
  }

  iyhigh = iylow + 1;

  ilow = 0;
  xxratio = 0.f;

  if (cotalpha >= thePixelTemp_[index].entx[0][Nxx - 1].cotalpha) {
    ilow = Nxx - 2;
    xxratio = 1.f;

  } else {
    if (cotalpha >= thePixelTemp_[index].entx[0][0].cotalpha) {
      for (i = 0; i < Nxx - 1; ++i) {
        if (thePixelTemp_[index].entx[0][i].cotalpha <= cotalpha &&
            cotalpha < thePixelTemp_[index].entx[0][i + 1].cotalpha) {
          ilow = i;
          xxratio = (cotalpha - thePixelTemp_[index].entx[0][i].cotalpha) /
                    (thePixelTemp_[index].entx[0][i + 1].cotalpha - thePixelTemp_[index].entx[0][i].cotalpha);
          break;
        }
      }
    }
  }

  ihigh = ilow + 1;

  sx1 =
      (1.f - xxratio) * thePixelTemp_[index].entx[0][ilow].sxone + xxratio * thePixelTemp_[index].entx[0][ihigh].sxone;
  sx2 =
      (1.f - xxratio) * thePixelTemp_[index].entx[0][ilow].sxtwo + xxratio * thePixelTemp_[index].entx[0][ihigh].sxtwo;

  xrms = (1.f - yxratio) * ((1.f - xxratio) * thePixelTemp_[index].entx[iylow][ilow].xrms[qBin] +
                            xxratio * thePixelTemp_[index].entx[iylow][ihigh].xrms[qBin]) +
         yxratio * ((1.f - xxratio) * thePixelTemp_[index].entx[iyhigh][ilow].xrms[qBin] +
                    xxratio * thePixelTemp_[index].entx[iyhigh][ihigh].xrms[qBin]);

  //  Take the errors and bias from the correct charge bin

  sigmay = yrms;

  sigmax = xrms;

  return;

}  // temperrors

// ************************************************************************************************************
//! Interpolate beta/alpha angles to produce estimated errors for fastsim
//! \param id - (input) index of the template to use
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param qbin_frac[4] - (output) the integrated probability for qbin=0, 0+1, 0+1+2, 0+1+2+3 (1.)
//! \param ny1_frac - (output) the probability for ysize = 1 for a single-size pixel
//! \param ny2_frac - (output) the probability for ysize = 1 for a double-size pixel
//! \param nx1_frac - (output) the probability for xsize = 1 for a single-size pixel
//! \param nx2_frac - (output) the probability for xsize = 1 for a double-size pixel
// ************************************************************************************************************
void SiPixelTemplate::qbin_dist(int id,
                                float cotalpha,
                                float cotbeta,
                                float qbin_frac[4],
                                float& ny1_frac,
                                float& ny2_frac,
                                float& nx1_frac,
                                float& nx2_frac)

{
  // Interpolate for a new set of track angles

  // Local variables
  int i;
  int ilow, ihigh, iylow, iyhigh, Ny, Nxx, Nyx, index;
  float yxratio, xxratio;
  float acotb;
  float qfrac[4];
  //bool flip_y;

  // Find the index corresponding to id

  index = -1;
  for (i = 0; i < (int)thePixelTemp_.size(); ++i) {
    if (id == thePixelTemp_[i].head.ID) {
      index = i;
      //				id_current_ = id;
      break;
    }
  }

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (index < 0 || index >= (int)thePixelTemp_.size()) {
    throw cms::Exception("DataCorrupt") << "SiPixelTemplate::temperrors can't find needed template ID = " << id
                                        << std::endl;
  }
#else
  assert(index >= 0 && index < (int)thePixelTemp_.size());
#endif

  //

  // Interpolate the absolute value of cot(beta)

  acotb = fabs((double)cotbeta);

  // Copy the charge scaling factor to the private variable

  Ny = thePixelTemp_[index].head.NTy;
  Nyx = thePixelTemp_[index].head.NTyx;
  Nxx = thePixelTemp_[index].head.NTxx;

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (Ny < 2 || Nyx < 1 || Nxx < 2) {
    throw cms::Exception("DataCorrupt") << "template ID = " << id_current_ << "has too few entries: Ny/Nyx/Nxx = " << Ny
                                        << "/" << Nyx << "/" << Nxx << std::endl;
  }
#else
  assert(Ny > 1 && Nyx > 0 && Nxx > 1);
#endif

  // Interpolate/store all y-related quantities (flip displacements when flip_y)
  ny1_frac = (1.f - yratio_) * enty0_->fracyone + yratio_ * enty1_->fracyone;
  ny2_frac = (1.f - yratio_) * enty0_->fracytwo + yratio_ * enty1_->fracytwo;

  // next, loop over all x-angle entries, first, find relevant y-slices

  iylow = 0;
  yxratio = 0.f;

  if (acotb >= thePixelTemp_[index].entx[Nyx - 1][0].cotbeta) {
    iylow = Nyx - 2;
    yxratio = 1.f;

  } else if (acotb >= thePixelTemp_[index].entx[0][0].cotbeta) {
    for (i = 0; i < Nyx - 1; ++i) {
      if (thePixelTemp_[index].entx[i][0].cotbeta <= acotb && acotb < thePixelTemp_[index].entx[i + 1][0].cotbeta) {
        iylow = i;
        yxratio = (acotb - thePixelTemp_[index].entx[i][0].cotbeta) /
                  (thePixelTemp_[index].entx[i + 1][0].cotbeta - thePixelTemp_[index].entx[i][0].cotbeta);
        break;
      }
    }
  }

  iyhigh = iylow + 1;

  ilow = 0;
  xxratio = 0.f;

  if (cotalpha >= thePixelTemp_[index].entx[0][Nxx - 1].cotalpha) {
    ilow = Nxx - 2;
    xxratio = 1.f;

  } else {
    if (cotalpha >= thePixelTemp_[index].entx[0][0].cotalpha) {
      for (i = 0; i < Nxx - 1; ++i) {
        if (thePixelTemp_[index].entx[0][i].cotalpha <= cotalpha &&
            cotalpha < thePixelTemp_[index].entx[0][i + 1].cotalpha) {
          ilow = i;
          xxratio = (cotalpha - thePixelTemp_[index].entx[0][i].cotalpha) /
                    (thePixelTemp_[index].entx[0][i + 1].cotalpha - thePixelTemp_[index].entx[0][i].cotalpha);
          break;
        }
      }
    }
  }

  ihigh = ilow + 1;

  for (i = 0; i < 3; ++i) {
    qfrac[i] = (1.f - yxratio) * ((1.f - xxratio) * thePixelTemp_[index].entx[iylow][ilow].qbfrac[i] +
                                  xxratio * thePixelTemp_[index].entx[iylow][ihigh].qbfrac[i]) +
               yxratio * ((1.f - xxratio) * thePixelTemp_[index].entx[iyhigh][ilow].qbfrac[i] +
                          xxratio * thePixelTemp_[index].entx[iyhigh][ihigh].qbfrac[i]);
  }
  nx1_frac = (1.f - yxratio) * ((1.f - xxratio) * thePixelTemp_[index].entx[iylow][ilow].fracxone +
                                xxratio * thePixelTemp_[index].entx[iylow][ihigh].fracxone) +
             yxratio * ((1.f - xxratio) * thePixelTemp_[index].entx[iyhigh][ilow].fracxone +
                        xxratio * thePixelTemp_[index].entx[iyhigh][ihigh].fracxone);
  nx2_frac = (1.f - yxratio) * ((1.f - xxratio) * thePixelTemp_[index].entx[iylow][ilow].fracxtwo +
                                xxratio * thePixelTemp_[index].entx[iylow][ihigh].fracxtwo) +
             yxratio * ((1.f - xxratio) * thePixelTemp_[index].entx[iyhigh][ilow].fracxtwo +
                        xxratio * thePixelTemp_[index].entx[iyhigh][ihigh].fracxtwo);

  qbin_frac[0] = qfrac[0];
  qbin_frac[1] = qbin_frac[0] + qfrac[1];
  qbin_frac[2] = qbin_frac[1] + qfrac[2];
  qbin_frac[3] = 1.f;
  return;

}  // qbin_dist

// *************************************************************************************************************************************
//! Make simple 2-D templates from track angles set in interpolate and hit position.

//! \param       xhit - (input) x-position of hit relative to the lower left corner of pixel[1][1] (to allow for the "padding" of the two-d clusters in the splitter)
//! \param       yhit - (input) y-position of hit relative to the lower left corner of pixel[1][1]
//! \param    ydouble - (input) STL vector of 21 element array to flag a double-pixel starting at cluster[1][1]
//! \param    xdouble - (input) STL vector of 11 element array to flag a double-pixel starting at cluster[1][1]
//! \param template2d - (output) 2d template of size matched to the cluster.  Input must be zeroed since charge is added only.
// *************************************************************************************************************************************

bool SiPixelTemplate::simpletemplate2D(
    float xhit, float yhit, std::vector<bool>& ydouble, std::vector<bool>& xdouble, float template2d[BXM2][BYM2]) {
  // Local variables

  float x0, y0, xf, yf, xi, yi, sf, si, s0, qpix, slopey, slopex, ds;
  int i, j, jpix0, ipix0, jpixf, ipixf, jpix, ipix, nx, ny, anx, any, jmax, imax;
  float qtotal;
  //	double path;
  std::list<SimplePixel> list;
  std::list<SimplePixel>::iterator listIter, listEnd;

  // Calculate the entry and exit points for the line charge from the track

  x0 = xhit - 0.5 * zsize_ * cota_current_;
  y0 = yhit - 0.5 * zsize_ * cotb_current_;

  jpix0 = floor(x0 / xsize_) + 1;
  ipix0 = floor(y0 / ysize_) + 1;

  if (jpix0 < 0 || jpix0 > BXM3) {
    return false;
  }
  if (ipix0 < 0 || ipix0 > BYM3) {
    return false;
  }

  xf = xhit + 0.5 * zsize_ * cota_current_ + lorxwidth_;
  yf = yhit + 0.5 * zsize_ * cotb_current_ + lorywidth_;

  jpixf = floor(xf / xsize_) + 1;
  ipixf = floor(yf / ysize_) + 1;

  if (jpixf < 0 || jpixf > BXM3) {
    return false;
  }
  if (ipixf < 0 || ipixf > BYM3) {
    return false;
  }

  // total charge length

  sf = std::sqrt((xf - x0) * (xf - x0) + (yf - y0) * (yf - y0));
  if ((xf - x0) != 0.f) {
    slopey = (yf - y0) / (xf - x0);
  } else {
    slopey = 1.e10;
  }
  if ((yf - y0) != 0.f) {
    slopex = (xf - x0) / (yf - y0);
  } else {
    slopex = 1.e10;
  }

  // use average charge in this direction

  qtotal = qavg_avg_;

  SimplePixel element;
  element.s = sf;
  element.x = xf;
  element.y = yf;
  element.i = ipixf;
  element.j = jpixf;
  element.btype = 0;
  list.push_back(element);

  //  nx is the number of x interfaces crossed by the line charge

  nx = jpixf - jpix0;
  anx = abs(nx);
  if (anx > 0) {
    if (nx > 0) {
      for (j = jpix0; j < jpixf; ++j) {
        xi = xsize_ * j;
        yi = slopey * (xi - x0) + y0;
        ipix = (int)(yi / ysize_) + 1;
        si = std::sqrt((xi - x0) * (xi - x0) + (yi - y0) * (yi - y0));
        element.s = si;
        element.x = xi;
        element.y = yi;
        element.i = ipix;
        element.j = j;
        element.btype = 1;
        list.push_back(element);
      }
    } else {
      for (j = jpix0; j > jpixf; --j) {
        xi = xsize_ * (j - 1);
        yi = slopey * (xi - x0) + y0;
        ipix = (int)(yi / ysize_) + 1;
        si = std::sqrt((xi - x0) * (xi - x0) + (yi - y0) * (yi - y0));
        element.s = si;
        element.x = xi;
        element.y = yi;
        element.i = ipix;
        element.j = j;
        element.btype = 1;
        list.push_back(element);
      }
    }
  }

  ny = ipixf - ipix0;
  any = abs(ny);
  if (any > 0) {
    if (ny > 0) {
      for (i = ipix0; i < ipixf; ++i) {
        yi = ysize_ * i;
        xi = slopex * (yi - y0) + x0;
        jpix = (int)(xi / xsize_) + 1;
        si = std::sqrt((xi - x0) * (xi - x0) + (yi - y0) * (yi - y0));
        element.s = si;
        element.x = xi;
        element.y = yi;
        element.i = i;
        element.j = jpix;
        element.btype = 2;
        list.push_back(element);
      }
    } else {
      for (i = ipix0; i > ipixf; --i) {
        yi = ysize_ * (i - 1);
        xi = slopex * (yi - y0) + x0;
        jpix = (int)(xi / xsize_) + 1;
        si = std::sqrt((xi - x0) * (xi - x0) + (yi - y0) * (yi - y0));
        element.s = si;
        element.x = xi;
        element.y = yi;
        element.i = i;
        element.j = jpix;
        element.btype = 2;
        list.push_back(element);
      }
    }
  }

  imax = std::max(ipix0, ipixf);
  jmax = std::max(jpix0, jpixf);

  // Sort the list according to the distance from the initial point

  list.sort();

  // Look for double pixels and adjust the list appropriately

  for (i = 1; i < imax; ++i) {
    if (ydouble[i - 1]) {
      listIter = list.begin();
      if (ny > 0) {
        while (listIter != list.end()) {
          if (listIter->i == i && listIter->btype == 2) {
            listIter = list.erase(listIter);
            continue;
          }
          if (listIter->i > i) {
            --(listIter->i);
          }
          ++listIter;
        }
      } else {
        while (listIter != list.end()) {
          if (listIter->i == i + 1 && listIter->btype == 2) {
            listIter = list.erase(listIter);
            continue;
          }
          if (listIter->i > i + 1) {
            --(listIter->i);
          }
          ++listIter;
        }
      }
    }
  }

  for (j = 1; j < jmax; ++j) {
    if (xdouble[j - 1]) {
      listIter = list.begin();
      if (nx > 0) {
        while (listIter != list.end()) {
          if (listIter->j == j && listIter->btype == 1) {
            listIter = list.erase(listIter);
            continue;
          }
          if (listIter->j > j) {
            --(listIter->j);
          }
          ++listIter;
        }
      } else {
        while (listIter != list.end()) {
          if (listIter->j == j + 1 && listIter->btype == 1) {
            listIter = list.erase(listIter);
            continue;
          }
          if (listIter->j > j + 1) {
            --(listIter->j);
          }
          ++listIter;
        }
      }
    }
  }

  // The list now contains the path lengths of the line charge in each pixel from (x0,y0).  Cacluate the lengths of the segments and the charge.

  s0 = 0.f;
  listIter = list.begin();
  listEnd = list.end();
  for (; listIter != listEnd; ++listIter) {
    si = listIter->s;
    ds = si - s0;
    s0 = si;
    j = listIter->j;
    i = listIter->i;
    if (sf > 0.f) {
      qpix = qtotal * ds / sf;
    } else {
      qpix = qtotal;
    }
    template2d[j][i] += qpix;
  }

  return true;

}  // simpletemplate2D

// ************************************************************************************************************
//! Interpolate beta/alpha angles to produce Vavilov parameters for the charge distribution
//! \param mpv   - (output) the Vavilov most probable charge (well, not really the most probable esp at large kappa)
//! \param sigma - (output) the Vavilov sigma parameter
//! \param kappa - (output) the Vavilov kappa parameter [0.01 (Landau-like) < kappa < 10 (Gaussian-like)
// ************************************************************************************************************
void SiPixelTemplate::vavilov_pars(double& mpv, double& sigma, double& kappa)

{
  // Local variables
  int i;
  int ilow, ihigh, Ny;
  float yratio, cotb, cotalpha0, arg;

  // Interpolate in cotbeta only for the correct total path length (converts cotalpha, cotbeta into an effective cotbeta)

  cotalpha0 = thePixelTemp_[index_id_].enty[0].cotalpha;
  arg = cotb_current_ * cotb_current_ + cota_current_ * cota_current_ - cotalpha0 * cotalpha0;
  if (arg < 0.f)
    arg = 0.f;
  cotb = std::sqrt(arg);

  // Copy the charge scaling factor to the private variable

  Ny = thePixelTemp_[index_id_].head.NTy;

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (Ny < 2) {
    throw cms::Exception("DataCorrupt") << "template ID = " << id_current_ << "has too few entries: Ny = " << Ny
                                        << std::endl;
  }
#else
  assert(Ny > 1);
#endif

  // next, loop over all y-angle entries

  ilow = 0;
  yratio = 0.f;

  if (cotb >= thePixelTemp_[index_id_].enty[Ny - 1].cotbeta) {
    ilow = Ny - 2;
    yratio = 1.f;

  } else {
    if (cotb >= thePixelTemp_[index_id_].enty[0].cotbeta) {
      for (i = 0; i < Ny - 1; ++i) {
        if (thePixelTemp_[index_id_].enty[i].cotbeta <= cotb && cotb < thePixelTemp_[index_id_].enty[i + 1].cotbeta) {
          ilow = i;
          yratio = (cotb - thePixelTemp_[index_id_].enty[i].cotbeta) /
                   (thePixelTemp_[index_id_].enty[i + 1].cotbeta - thePixelTemp_[index_id_].enty[i].cotbeta);
          break;
        }
      }
    }
  }

  ihigh = ilow + 1;

  // Interpolate Vavilov parameters

  mpvvav_ = (1.f - yratio) * thePixelTemp_[index_id_].enty[ilow].mpvvav +
            yratio * thePixelTemp_[index_id_].enty[ihigh].mpvvav;
  sigmavav_ = (1.f - yratio) * thePixelTemp_[index_id_].enty[ilow].sigmavav +
              yratio * thePixelTemp_[index_id_].enty[ihigh].sigmavav;
  kappavav_ = (1.f - yratio) * thePixelTemp_[index_id_].enty[ilow].kappavav +
              yratio * thePixelTemp_[index_id_].enty[ihigh].kappavav;

  // Copy to parameter list

  //Avoid rounding difference between floats and doubles causing issues later
  if (kappavav_ <= 0.01f) {
    LOGERROR("SiPixelTemplate") << "Vavilov kappa value is " << kappavav_ << " changing it to be above 0.01" << ENDL;
    kappavav_ = 0.01f + 0.0000001f;
  }

  mpv = (double)mpvvav_;
  sigma = (double)sigmavav_;
  kappa = (double)kappavav_;

  return;

}  // vavilov_pars

// ************************************************************************************************************
//! Interpolate beta/alpha angles to produce Vavilov parameters for the 2-cluster charge distribution
//! \param mpv   - (output) the Vavilov most probable charge (well, not really the most probable esp at large kappa)
//! \param sigma - (output) the Vavilov sigma parameter
//! \param kappa - (output) the Vavilov kappa parameter [0.01 (Landau-like) < kappa < 10 (Gaussian-like)
// ************************************************************************************************************
void SiPixelTemplate::vavilov2_pars(double& mpv, double& sigma, double& kappa)

{
  // Local variables
  int i;
  int ilow, ihigh, Ny;
  float yratio, cotb, cotalpha0, arg;

  // Interpolate in cotbeta only for the correct total path length (converts cotalpha, cotbeta into an effective cotbeta)

  cotalpha0 = thePixelTemp_[index_id_].enty[0].cotalpha;
  arg = cotb_current_ * cotb_current_ + cota_current_ * cota_current_ - cotalpha0 * cotalpha0;
  if (arg < 0.f)
    arg = 0.f;
  cotb = std::sqrt(arg);

  // Copy the charge scaling factor to the private variable

  Ny = thePixelTemp_[index_id_].head.NTy;

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  if (Ny < 2) {
    throw cms::Exception("DataCorrupt") << "template ID = " << id_current_ << "has too few entries: Ny = " << Ny
                                        << std::endl;
  }
#else
  assert(Ny > 1);
#endif

  // next, loop over all y-angle entries

  ilow = 0;
  yratio = 0.f;

  if (cotb >= thePixelTemp_[index_id_].enty[Ny - 1].cotbeta) {
    ilow = Ny - 2;
    yratio = 1.f;

  } else {
    if (cotb >= thePixelTemp_[index_id_].enty[0].cotbeta) {
      for (i = 0; i < Ny - 1; ++i) {
        if (thePixelTemp_[index_id_].enty[i].cotbeta <= cotb && cotb < thePixelTemp_[index_id_].enty[i + 1].cotbeta) {
          ilow = i;
          yratio = (cotb - thePixelTemp_[index_id_].enty[i].cotbeta) /
                   (thePixelTemp_[index_id_].enty[i + 1].cotbeta - thePixelTemp_[index_id_].enty[i].cotbeta);
          break;
        }
      }
    }
  }

  ihigh = ilow + 1;

  // Interpolate Vavilov parameters

  mpvvav2_ = (1.f - yratio) * thePixelTemp_[index_id_].enty[ilow].mpvvav2 +
             yratio * thePixelTemp_[index_id_].enty[ihigh].mpvvav2;
  sigmavav2_ = (1.f - yratio) * thePixelTemp_[index_id_].enty[ilow].sigmavav2 +
               yratio * thePixelTemp_[index_id_].enty[ihigh].sigmavav2;
  kappavav2_ = (1.f - yratio) * thePixelTemp_[index_id_].enty[ilow].kappavav2 +
               yratio * thePixelTemp_[index_id_].enty[ihigh].kappavav2;

  // Copy to parameter list

  mpv = (double)mpvvav2_;
  sigma = (double)sigmavav2_;
  kappa = (double)kappavav2_;

  return;

}  // vavilov2_pars
