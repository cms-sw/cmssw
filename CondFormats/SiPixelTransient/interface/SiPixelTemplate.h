//
//  SiPixelTemplate.h (v10.20)
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
//  Store x and y cluster sizes in fractional pixels to facilitate cluster splitting
//  Keep interpolated central 9 template bins in private storage and expand/shift in the getter functions (faster for speed=2/3) and easier to build 3d templates
//  Store error and bias information for the simple chi^2 min position analysis (no interpolation or Q_{FB} corrections) to use in cluster splitting
//  To save time, the gaussian centers and sigma are not interpolated right now (they aren't currently used).  They can be restored by un-commenting lines in the interpolate method.
//  Add a new method to calculate qbin for input cotbeta and cluster charge.  To be used for error estimation of merged clusters in PixelCPEGeneric.
//  Add bias info for Barrel and FPix separately in the header
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
//  Separate BPix and FPix charge scales and thresholds
//  Fix DB pushfile version number checking bug.
//  Remove assert from qbin method
//  Replace asserts with exceptions in CMSSW
//  Change calling sequence to interpolate method to handle cot(beta)<0 for FPix cosmics
//  Add getter for pixelav Lorentz width estimates to qbin method
//  Add check on template size to interpolate and qbin methods
//  Add qbin population information, charge distribution information
//
//  V7.00 - Decouple BPix and FPix information into separate templates
//  Add methods to facilitate improved cluster splitting
//  Fix small charge scaling bug (affects FPix only)
//  Change y-slice used for the x-template to be closer to the actual cotalpha-cotbeta point
//  (there is some weak breakdown of x-y factorization in the FPix after irradiation)
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
//  V10.10 - Add class variables and methods to correctly calculate the probabilities of single pixel clusters
//  V10.11 - Allow subdetector ID=5 for FPix R2P2 [allows better internal labeling of templates]
//  V10.12 - Enforce minimum signal size in pixel charge uncertainty calculation
//  V10.13 - Update the variable size [SI_PIXEL_TEMPLATE_USE_BOOST] option so that it works with VI's enhancements
//  V10.20 - Add directory path selection to the ascii pushfile method
//  V10.21 - Address runtime issues in pushfile() for gcc 7.X due to using tempfile as char string + misc. cleanup [Petar]
//  V10.22 - Move templateStore to the heap, fix variable name in pushfile() [Petar]

// Created by Morris Swartz on 10/27/06.
//
//

// Build the template storage structure from several pieces

#ifndef SiPixelTemplate_h
#define SiPixelTemplate_h 1

#include "SiPixelTemplateDefs.h"

#include <vector>
#include <cassert>
#include "boost/multi_array.hpp"

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "FWCore/Utilities/interface/Exception.h"
#endif

struct SiPixelTemplateEntry {  //!< Basic template entry corresponding to a single set of track angles
  int runnum;                  //!< number of pixelav run used to generate this entry
  float alpha;                 //!< alpha track angle (defined in CMS CMS IN 2004/014)
  float cotalpha;              //!< cot(alpha) is proportional to cluster length in x and is basis of interpolation
  float beta;                  //!< beta track angle (defined in CMS CMS IN 2004/014)
  float cotbeta;               //!< cot(beta) is proportional to cluster length in y and is basis of interpolation
  float costrk[3];             //!< direction cosines of tracks used to generate this entry
  float qavg;                  //!< average cluster charge for this set of track angles (now includes threshold effects)
  float pixmax;                //!< maximum charge for individual pixels in cluster
  float symax;                 //!< average pixel signal for y-projection of cluster
  float dyone;                 //!< mean offset/correction for one pixel y-clusters
  float syone;                 //!< rms for one pixel y-clusters
  float sxmax;                 //!< average pixel signal for x-projection of cluster
  float dxone;                 //!< mean offset/correction for one pixel x-clusters
  float sxone;                 //!< rms for one pixel x-clusters
  float dytwo;                 //!< mean offset/correction for one double-pixel y-clusters
  float sytwo;                 //!< rms for one double-pixel y-clusters
  float dxtwo;                 //!< mean offset/correction for one double-pixel x-clusters
  float sxtwo;                 //!< rms for one double-pixel x-clusters
  float qmin;                  //!< minimum cluster charge for valid hit (keeps 99.9% of simulated hits)
  float qmin2;                 //!< tighter minimum cluster charge for valid hit (keeps 99.8% of simulated hits)
  float yavggen[4];            //!< generic algorithm: average y-bias of reconstruction binned in 4 charge bins
  float yrmsgen[4];            //!< generic algorithm: average y-rms of reconstruction binned in 4 charge bins
  float xavggen[4];            //!< generic algorithm: average x-bias of reconstruction binned in 4 charge bins
  float xrmsgen[4];            //!< generic algorithm: average x-rms of reconstruction binned in 4 charge bins

  float clsleny;   //!< cluster y-length in pixels at signal height symax/2
  float clslenx;   //!< cluster x-length in pixels at signal height sxmax/2
  float mpvvav;    //!< most probable charge in Vavilov distribution (not actually for larger kappa)
  float sigmavav;  //!< "sigma" scale fctor for Vavilov distribution
  float kappavav;  //!< kappa parameter for Vavilov distribution
  float mpvvav2;  //!< most probable charge in Vavilov distribution for 2 merged clusters (not actually for larger kappa)
  float sigmavav2;         //!< "sigma" scale fctor for Vavilov distribution for 2 merged clusters
  float kappavav2;         //!< kappa parameter for Vavilov distribution for 2 merged clusters
  float ypar[2][5];        //!< projected y-pixel uncertainty parameterization
  float ytemp[9][TYSIZE];  //!< templates for y-reconstruction (binned over 1 central pixel)
  float xpar[2][5];        //!< projected x-pixel uncertainty parameterization
  float xtemp[9][TXSIZE];  //!< templates for x-reconstruction (binned over 1 central pixel)
  float yavg[4];           //!< average y-bias of reconstruction binned in 4 charge bins
  float yrms[4];           //!< average y-rms of reconstruction binned in 4 charge bins
  float ygx0[4];           //!< average y0 from Gaussian fit binned in 4 charge bins
  float ygsig[4];          //!< average sigma_y from Gaussian fit binned in 4 charge bins
  float yflpar[4][6];      //!< Aqfl-parameterized y-correction in 4 charge bins
  float xavg[4];           //!< average x-bias of reconstruction binned in 4 charge bins
  float xrms[4];           //!< average x-rms of reconstruction binned in 4 charge bins
  float xgx0[4];           //!< average x0 from Gaussian fit binned in 4 charge bins
  float xgsig[4];          //!< average sigma_x from Gaussian fit binned in 4 charge bins
  float xflpar[4][6];      //!< Aqfl-parameterized x-correction in 4 charge bins
  float chi2yavg[4];       //!< average y chi^2 in 4 charge bins
  float chi2ymin[4];       //!< minimum of y chi^2 in 4 charge bins
  float chi2xavg[4];       //!< average x chi^2 in 4 charge bins
  float chi2xmin[4];       //!< minimum of x chi^2 in 4 charge bins
  float chi2yavgone;       //!< average y chi^2 for 1 pixel clusters
  float chi2yminone;       //!< minimum of y chi^2 for 1 pixel clusters
  float chi2xavgone;       //!< average x chi^2 for 1 pixel clusters
  float chi2xminone;       //!< minimum of x chi^2 for 1 pixel clusters
  float yavgc2m[4];        //!< 1st pass chi2 min search: average y-bias of reconstruction binned in 4 charge bins
  float yrmsc2m[4];        //!< 1st pass chi2 min search: average y-rms of reconstruction binned in 4 charge bins
  float chi2yavgc2m[4];    //!< 1st pass chi2 min search: average y chi^2 in 4 charge bins (merged clusters)
  float chi2yminc2m[4];    //!< 1st pass chi2 min search: minimum of y chi^2 in 4 charge bins (merged clusters)
  float xavgc2m[4];        //!< 1st pass chi2 min search: average x-bias of reconstruction binned in 4 charge bins
  float xrmsc2m[4];        //!< 1st pass chi2 min search: average x-rms of reconstruction binned in 4 charge bins
  float chi2xavgc2m[4];    //!< 1st pass chi2 min search: average x chi^2 in 4 charge bins (merged clusters)
  float chi2xminc2m[4];    //!< 1st pass chi2 min search: minimum of x chi^2 in 4 charge bins (merged clusters)
  float ygx0gen[4];        //!< generic algorithm: average y0 from Gaussian fit binned in 4 charge bins
  float ygsiggen[4];       //!< generic algorithm: average sigma_y from Gaussian fit binned in 4 charge bins
  float xgx0gen[4];        //!< generic algorithm: average x0 from Gaussian fit binned in 4 charge bins
  float xgsiggen[4];       //!< generic algorithm: average sigma_x from Gaussian fit binned in 4 charge bins
  float qbfrac[3];         //!< fraction of sample in qbin = 0-2 (>=3 is the complement)
  float fracyone;          //!< fraction of sample with ysize = 1
  float fracxone;          //!< fraction of sample with xsize = 1
  float fracytwo;          //!< fraction of double pixel sample with ysize = 1
  float fracxtwo;          //!< fraction of double pixel sample with xsize = 1
  float qavg_avg;       //!< average cluster charge of clusters that are less than qavg (normalize 2-D simple templates)
  float r_qMeas_qTrue;  //!< ratio of measured to true cluster charge
  float spare[1];
};

struct SiPixelTemplateHeader {  //!< template header structure
  int ID;                       //!< template ID number
  int NTy;                      //!< number of Template y entries
  int NTyx;                     //!< number of Template y-slices of x entries
  int NTxx;                     //!< number of Template x-entries in each slice
  int Dtype;                    //!< detector type (0=BPix, 1=FPix)
  float qscale;                 //!< Charge scaling to match cmssw and pixelav
  float lorywidth;              //!< estimate of y-lorentz width for optimal resolution
  float lorxwidth;              //!< estimate of x-lorentz width for optimal resolution
  float lorybias;               //!< estimate of y-lorentz bias
  float lorxbias;               //!< estimate of x-lorentz bias
  float Vbias;                  //!< detector bias potential in Volts
  float temperature;            //!< detector temperature in deg K
  float fluence;                //!< radiation fluence in n_eq/cm^2
  float s50;                    //!< 1/2 of the multihit dcol threshold in electrons
  float ss50;                   //!< 1/2 of the single hit dcol threshold in electrons
  char title[80];               //!< template title
  int templ_version;            //!< Version number of the template to ensure code compatibility
  float Bfield;                 //!< Bfield in Tesla
  float fbin[3];                //!< The QBin definitions in Q_clus/Q_avg
  float xsize;                  //!< pixel size (for future use in upgraded geometry)
  float ysize;                  //!< pixel size (for future use in upgraded geometry)
  float zsize;                  //!< pixel size (for future use in upgraded geometry)
};

struct SiPixelTemplateStore {  //!< template storage structure
  SiPixelTemplateHeader head;
#ifndef SI_PIXEL_TEMPLATE_USE_BOOST
  float cotbetaY[TEMP_ENTRY_SIZEY];
  float cotbetaX[TEMP_ENTRY_SIZEX_B];
  float cotalphaX[TEMP_ENTRY_SIZEX_A];
  //!< 60 y templates spanning cluster lengths from 0px to +18px
  SiPixelTemplateEntry enty[TEMP_ENTRY_SIZEY];
  //!< 60 Barrel x templates spanning cluster lengths from -6px (-1.125Rad) to +6px (+1.125Rad) in each of 60 slices
  SiPixelTemplateEntry entx[TEMP_ENTRY_SIZEX_B][TEMP_ENTRY_SIZEX_A];
  void destroy(){};
#else
  float* cotbetaY = nullptr;
  float* cotbetaX = nullptr;
  float* cotalphaX = nullptr;
  boost::multi_array<SiPixelTemplateEntry, 1> enty;  //!< use 1d entry to store [60] entries
  //!< use 2d entry to store [60][60] entries
  boost::multi_array<SiPixelTemplateEntry, 2> entx;
  void destroy() {  // deletes arrays created by pushfile method of SiPixelTemplate
    if (cotbetaY != nullptr)
      delete[] cotbetaY;
    if (cotbetaX != nullptr)
      delete[] cotbetaX;
    if (cotalphaX != nullptr)
      delete[] cotalphaX;
  }
#endif
};

// ******************************************************************************************
//! \class SiPixelTemplate
//!
//!  A template management class.  SiPixelTemplate contains thePixelTemp
//!  (a std::vector  of SiPixelTemplateStore, each of which is a collection of many
//!  SiPixelTemplateEntries).  Each SiPixelTemplateStore corresponds to a given detector
//!  condition, and is valid for a range of runs.  We allow more than one Store since the
//!  may change over time.
//!
//!  This class reads templates from files via pushfile() method.
//!
//!  The main functionality of SiPixelTemplate is interpolate(), which produces a template
//!  on the fly, given a specific track's alpha and beta.  The results are kept in data
//!  members and accessed via inline getters.
//!
//!  The resulting template is then used by PixelTempReco2D() (a global function) which
//!  get the reference for SiPixelTemplate & templ and uses the current template to
//!  reconstruct the SiPixelRecHit.
// ******************************************************************************************
class SiPixelTemplate {
public:
  SiPixelTemplate(const std::vector<SiPixelTemplateStore>& thePixelTemp) : thePixelTemp_(thePixelTemp) {
    id_current_ = -1;
    index_id_ = -1;
    cota_current_ = 0.;
    cotb_current_ = 0.;
  }  //!< Constructor for cases in which template store already exists

// Load the private store with info from the file with the index (int) filenum from directory dir:
//   ${dir}template_summary_zp${filenum}.out
#ifdef SI_PIXEL_TEMPLATE_STANDALONE
  static bool pushfile(int filenum, std::vector<SiPixelTemplateStore>& pixelTemp, std::string dir = "");
#else
  static bool pushfile(int filenum,
                       std::vector<SiPixelTemplateStore>& pixelTemp,
                       // *&^%$#@!  Different default dir -- remove once FastSim is updated.
                       std::string dir = "CalibTracker/SiPixelESProducers/data/");
  static bool pushfile(const SiPixelTemplateDBObject& dbobject,
                       std::vector<SiPixelTemplateStore>& pixelTemp);  // load the private store with info from db
#endif

  // initialize the rest;
  static void postInit(std::vector<SiPixelTemplateStore>& thePixelTemp_);

  // Interpolate input alpha and beta angles to produce a working template for each individual hit.
  bool interpolate(int id, float cotalpha, float cotbeta, float locBz, float locBx);

  // Interpolate input alpha and beta angles to produce a working template for each individual hit.
  bool interpolate(int id, float cotalpha, float cotbeta, float locBz);

  // overload for compatibility.
  bool interpolate(int id, float cotalpha, float cotbeta);

  // retreive interpolated templates.
  void ytemp(int fybin, int lybin, float ytemplate[41][BYSIZE]);

  void xtemp(int fxbin, int lxbin, float xtemplate[41][BXSIZE]);

  //Method to estimate the central pixel of the interpolated y-template
  int cytemp();

  //Method to estimate the central pixel of the interpolated x-template
  int cxtemp();

  // new methods to build templates from two interpolated clusters (for splitting)
  void ytemp3d_int(int nypix, int& nybins);

  void ytemp3d(int j, int k, std::vector<float>& ytemplate);

  void xtemp3d_int(int nxpix, int& nxbins);

  void xtemp3d(int j, int k, std::vector<float>& xtemplate);

  // Convert vector of projected signals into uncertainties for fitting.
  void ysigma2(int fypix, int lypix, float sythr, float ysum[BYSIZE], float ysig2[BYSIZE]);

  void ysigma2(float qpixel, int index, float& ysig2);

  void xsigma2(int fxpix, int lxpix, float sxthr, float xsum[BXSIZE], float xsig2[BXSIZE]);

  // Interpolate qfl correction in y.
  float yflcorr(int binq, float qfly);

  // Interpolate qfl correction in x.
  float xflcorr(int binq, float qflx);

  int qbin(int id,
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
           float& dx2);

  int qbin(int id,
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
           float& dx2);

  // Overload to use for cluster splitting
  int qbin(int id, float cotalpha, float cotbeta, float qclus);

  // Overload to keep legacy interface
  int qbin(int id, float cotbeta, float qclus);

  // Method to return template errors for fastsim
  void temperrors(int id,
                  float cotalpha,
                  float cotbeta,
                  int qBin,
                  float& sigmay,
                  float& sigmax,
                  float& sy1,
                  float& sy2,
                  float& sx1,
                  float& sx2);

  //Method to return qbin and size probabilities for fastsim
  void qbin_dist(int id,
                 float cotalpha,
                 float cotbeta,
                 float qbin_frac[4],
                 float& ny1_frac,
                 float& ny2_frac,
                 float& nx1_frac,
                 float& nx2_frac);

  //Method to calculate simple 2D templates
  bool simpletemplate2D(
      float xhitp, float yhitp, std::vector<bool>& ydouble, std::vector<bool>& xdouble, float template2d[BXM2][BYM2]);

  //Method to interpolate Vavilov distribution parameters
  void vavilov_pars(double& mpv, double& sigma, double& kappa);

  //Method to interpolate 2-cluster Vavilov distribution parameters
  void vavilov2_pars(double& mpv, double& sigma, double& kappa);

  float qavg() { return qavg_; }      //!< average cluster charge for this set of track angles
  float pixmax() { return pixmax_; }  //!< maximum pixel charge
  float qscale() { return qscale_; }  //!< charge scaling factor
  float s50() { return s50_; }        //!< 1/2 of the pixel threshold signal in electrons
  float ss50() { return ss50_; }      //!< 1/2 of the single pixel per double column threshold in electrons
  float symax() { return symax_; }    //!< average pixel signal for y-projection of cluster
  float dyone() { return dyone_; }    //!< mean offset/correction for one pixel y-clusters
  float syone() { return syone_; }    //!< rms for one pixel y-clusters
  float dytwo() { return dytwo_; }    //!< mean offset/correction for one double-pixel y-clusters
  float sytwo() { return sytwo_; }    //!< rms for one double-pixel y-clusters
  float sxmax() { return sxmax_; }    //!< average pixel signal for x-projection of cluster
  float dxone() { return dxone_; }    //!< mean offset/correction for one pixel x-clusters
  float sxone() { return sxone_; }    //!< rms for one pixel x-clusters
  float dxtwo() { return dxtwo_; }    //!< mean offset/correction for one double-pixel x-clusters
  float sxtwo() { return sxtwo_; }    //!< rms for one double-pixel x-clusters
  float qmin() { return qmin_; }      //!< minimum cluster charge for valid hit (keeps 99.9% of simulated hits)
  float qmin(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 1) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::qmin called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 2);
#endif
    if (i == 0) {
      return qmin_;
    } else {
      return qmin2_;
    }
  }  //!< minimum cluster charge for valid hit (keeps 99.9% or 99.8% of simulated hits)
  float clsleny() { return clsleny_; }  //!< y-size of smaller interpolated template in pixels
  float clslenx() { return clslenx_; }  //!< x-size of smaller interpolated template in pixels
  float yratio() { return yratio_; }    //!< fractional distance in y between cotbeta templates
  float yxratio() { return yxratio_; }  //!< fractional distance in y between cotalpha templates slices
  float xxratio() { return xxratio_; }  //!< fractional distance in x between cotalpha templates
  float yavg(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::yavg called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return yavg_[i];
  }  //!< average y-bias of reconstruction binned in 4 charge bins
  float yrms(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::yrms called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return yrms_[i];
  }  //!< average y-rms of reconstruction binned in 4 charge bins
  float ygx0(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ygx0 called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return ygx0_[i];
  }  //!< average y0 from Gaussian fit binned in 4 charge bins
  float ygsig(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ygsig called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return ygsig_[i];
  }  //!< average sigma_y from Gaussian fit binned in 4 charge bins
  float xavg(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xavg called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return xavg_[i];
  }  //!< average x-bias of reconstruction binned in 4 charge bins
  float xrms(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xrms called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return xrms_[i];
  }  //!< average x-rms of reconstruction binned in 4 charge bins
  float xgx0(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xgx0 called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return xgx0_[i];
  }  //!< average x0 from Gaussian fit binned in 4 charge bins
  float xgsig(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xgsig called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return xgsig_[i];
  }  //!< average sigma_x from Gaussian fit binned in 4 charge bins
  float chi2yavg(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::chi2yavg called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return chi2yavg_[i];
  }  //!< average y chi^2 in 4 charge bins
  float chi2ymin(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::chi2ymin called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return chi2ymin_[i];
  }  //!< minimum y chi^2 in 4 charge bins
  float chi2xavg(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::chi2xavg called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return chi2xavg_[i];
  }  //!< averaage x chi^2 in 4 charge bins
  float chi2xmin(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::chi2xmin called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return chi2xmin_[i];
  }  //!< minimum y chi^2 in 4 charge bins
  float yavgc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::yavgc2m called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return yavgc2m_[i];
  }  //!< 1st pass chi2 min search: average y-bias of reconstruction binned in 4 charge bins
  float yrmsc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::yrmsc2m called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return yrmsc2m_[i];
  }  //!< 1st pass chi2 min search: average y-rms of reconstruction binned in 4 charge bins
  float chi2yavgc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt")
          << "SiPixelTemplate::chi2yavgc2m called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return chi2yavgc2m_[i];
  }  //!< 1st pass chi2 min search: average y-chisq for merged clusters
  float chi2yminc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt")
          << "SiPixelTemplate::chi2yminc2m called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return chi2yminc2m_[i];
  }  //!< 1st pass chi2 min search: minimum y-chisq for merged clusters
  float xavgc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xavgc2m called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return xavgc2m_[i];
  }  //!< 1st pass chi2 min search: average x-bias of reconstruction binned in 4 charge bins
  float xrmsc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xrmsc2m called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return xrmsc2m_[i];
  }  //!< 1st pass chi2 min search: average x-rms of reconstruction binned in 4 charge bins
  float chi2xavgc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt")
          << "SiPixelTemplate::chi2xavgc2m called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return chi2xavgc2m_[i];
  }  //!< 1st pass chi2 min search: average x-chisq for merged clusters
  float chi2xminc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 3) {
      throw cms::Exception("DataCorrupt")
          << "SiPixelTemplate::chi2xminc2m called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 4);
#endif
    return chi2xminc2m_[i];
  }  //!< 1st pass chi2 min search: minimum x-chisq for merged clusters
  float fbin(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > 2) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::fbin called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < 3);
#endif
    return fbin_[i];
  }  //!< Return lower bound of Qbin definition

  float chi2yavgone() { return chi2yavgone_; }  //!< //!< average y chi^2 for 1 pixel clusters
  float chi2yminone() { return chi2yminone_; }  //!< //!< minimum of y chi^2 for 1 pixel clusters
  float chi2xavgone() { return chi2xavgone_; }  //!< //!< average x chi^2 for 1 pixel clusters
  float chi2xminone() { return chi2xminone_; }  //!< //!< minimum of x chi^2 for 1 pixel clusters
  float lorywidth() { return lorywidth_; }      //!< signed lorentz y-width (microns)
  float lorxwidth() { return lorxwidth_; }      //!< signed lorentz x-width (microns)
  //float lorybias() {return lorywidth_;}                            //!< signed lorentz y-width (microns)
  //float lorxbias() {return lorxwidth_;}                            //!< signed lorentz x-width (microns)
  float lorybias() { return lorybias_; }  //!< signed lorentz y-width (microns)
  float lorxbias() { return lorxbias_; }  //!< signed lorentz x-width (microns)
  float mpvvav() { return mpvvav_; }  //!< most probable charge in Vavilov distribution (not actually for larger kappa)
  float sigmavav() { return sigmavav_; }  //!< "sigma" scale fctor for Vavilov distribution
  float kappavav() { return kappavav_; }  //!< kappa parameter for Vavilov distribution
  float mpvvav2() {
    return mpvvav2_;
  }  //!< most probable charge in 2-cluster Vavilov distribution (not actually for larger kappa)
  float sigmavav2() { return sigmavav2_; }          //!< "sigma" scale fctor for 2-cluster Vavilov distribution
  float kappavav2() { return kappavav2_; }          //!< kappa parameter for 2-cluster Vavilov distribution
  float xsize() { return xsize_; }                  //!< pixel x-size (microns)
  float ysize() { return ysize_; }                  //!< pixel y-size (microns)
  float zsize() { return zsize_; }                  //!< pixel z-size or thickness (microns)
  float r_qMeas_qTrue() { return r_qMeas_qTrue_; }  //!< ratio of measured to true cluster charge
  float fracyone() { return fracyone_; }            //!< The simulated fraction of single pixel y-clusters
  float fracxone() { return fracxone_; }            //!< The simulated fraction of single pixel x-clusters
  float fracytwo() { return fracytwo_; }            //!< The simulated fraction of single double-size pixel y-clusters
  float fracxtwo() { return fracxtwo_; }            //!< The simulated fraction of single double-size pixel x-clusters
  //  float yspare(int i) {assert(i>=0 && i<5); return pyspare[i];}    //!< vector of 5 spares interpolated in beta only
  //  float xspare(int i) {assert(i>=0 && i<10); return pxspare[i];}    //!< vector of 10 spares interpolated in alpha and beta

private:
  // Keep current template interpolaion parameters

  int id_current_;      //!< current id
  int index_id_;        //!< current index
  float cota_current_;  //!< current cot alpha
  float cotb_current_;  //!< current cot beta
  float abs_cotb_;      //!< absolute value of cot beta
  bool success_;        //!< true if cotalpha, cotbeta are inside of the acceptance (dynamically loaded)

  // Keep results of last interpolation to return through member functions

  float qavg_;              //!< average cluster charge for this set of track angles
  float pixmax_;            //!< maximum pixel charge
  float qscale_;            //!< charge scaling factor
  float s50_;               //!< 1/2 of the pixel single col threshold signal in electrons
  float ss50_;              //!< 1/2 of the pixel double col threshold signal in electrons
  float symax_;             //!< average pixel signal for y-projection of cluster
  float syparmax_;          //!< maximum pixel signal for parameterization of y uncertainties
  float dyone_;             //!< mean offset/correction for one pixel y-clusters
  float syone_;             //!< rms for one pixel y-clusters
  float dytwo_;             //!< mean offset/correction for one double-pixel y-clusters
  float sytwo_;             //!< rms for one double-pixel y-clusters
  float sxmax_;             //!< average pixel signal for x-projection of cluster
  float sxparmax_;          //!< maximum pixel signal for parameterization of x uncertainties
  float dxone_;             //!< mean offset/correction for one pixel x-clusters
  float sxone_;             //!< rms for one pixel x-clusters
  float dxtwo_;             //!< mean offset/correction for one double-pixel x-clusters
  float sxtwo_;             //!< rms for one double-pixel x-clusters
  float qmin_;              //!< minimum cluster charge for valid hit (keeps 99.9% of simulated hits)
  float clsleny_;           //!< y-cluster length of smaller interpolated template in pixels
  float clslenx_;           //!< x-cluster length of smaller interpolated template in pixels
  float yratio_;            //!< fractional distance in y between cotbeta templates
  float yparl_[2][5];       //!< projected y-pixel uncertainty parameterization for smaller cotbeta
  float yparh_[2][5];       //!< projected y-pixel uncertainty parameterization for larger cotbeta
  float xparly0_[2][5];     //!< projected x-pixel uncertainty parameterization for smaller cotbeta (central alpha)
  float xparhy0_[2][5];     //!< projected x-pixel uncertainty parameterization for larger cotbeta (central alpha)
  float ytemp_[9][BYSIZE];  //!< templates for y-reconstruction (binned over 5 central pixels)
  float yxratio_;           //!< fractional distance in y between x-slices of cotalpha templates
  float xxratio_;           //!< fractional distance in x between cotalpha templates
  float xpar0_[2][5];       //!< projected x-pixel uncertainty parameterization for central cotalpha
  float xparl_[2][5];       //!< projected x-pixel uncertainty parameterization for smaller cotalpha
  float xparh_[2][5];       //!< projected x-pixel uncertainty parameterization for larger cotalpha
  float xtemp_[9][BXSIZE];  //!< templates for x-reconstruction (binned over 5 central pixels)
  float yavg_[4];           //!< average y-bias of reconstruction binned in 4 charge bins
  float yrms_[4];           //!< average y-rms of reconstruction binned in 4 charge bins
  float ygx0_[4];           //!< average y0 from Gaussian fit binned in 4 charge bins
  float ygsig_[4];          //!< average sigma_y from Gaussian fit binned in 4 charge bins
  float yflparl_[4][6];     //!< Aqfl-parameterized y-correction in 4 charge bins for smaller cotbeta
  float yflparh_[4][6];     //!< Aqfl-parameterized y-correction in 4 charge bins for larger cotbeta
  float xavg_[4];           //!< average x-bias of reconstruction binned in 4 charge bins
  float xrms_[4];           //!< average x-rms of reconstruction binned in 4 charge bins
  float xgx0_[4];           //!< average x0 from Gaussian fit binned in 4 charge bins
  float xgsig_[4];          //!< sigma from Gaussian fit binned in 4 charge bins
  float xflparll_[4][6];    //!< Aqfl-parameterized x-correction in 4 charge bins for smaller cotbeta, cotalpha
  float xflparlh_[4][6];    //!< Aqfl-parameterized x-correction in 4 charge bins for smaller cotbeta, larger cotalpha
  float xflparhl_[4][6];    //!< Aqfl-parameterized x-correction in 4 charge bins for larger cotbeta, smaller cotalpha
  float xflparhh_[4][6];    //!< Aqfl-parameterized x-correction in 4 charge bins for larger cotbeta, cotalpha
  float chi2yavg_[4];       //!< average y chi^2 in 4 charge bins
  float chi2ymin_[4];       //!< minimum of y chi^2 in 4 charge bins
  float chi2xavg_[4];       //!< average x chi^2 in 4 charge bins
  float chi2xmin_[4];       //!< minimum of x chi^2 in 4 charge bins
  float yavgc2m_[4];        //!< 1st pass chi2 min search: average y-bias of reconstruction binned in 4 charge bins
  float yrmsc2m_[4];        //!< 1st pass chi2 min search: average y-rms of reconstruction binned in 4 charge bins
  float chi2yavgc2m_[4];    //!< 1st pass chi2 min search: average y-chisq for merged clusters
  float chi2yminc2m_[4];    //!< 1st pass chi2 min search: minimum y-chisq for merged clusters
  float xavgc2m_[4];        //!< 1st pass chi2 min search: average x-bias of reconstruction binned in 4 charge bins
  float xrmsc2m_[4];        //!< 1st pass chi2 min search: average x-rms of reconstruction binned in 4 charge bins
  float chi2xavgc2m_[4];    //!< 1st pass chi2 min search: average x-chisq for merged clusters
  float chi2xminc2m_[4];    //!< 1st pass chi2 min search: minimum x-chisq for merged clusters
  float chi2yavgone_;       //!< average y chi^2 for 1 pixel clusters
  float chi2yminone_;       //!< minimum of y chi^2 for 1 pixel clusters
  float chi2xavgone_;       //!< average x chi^2 for 1 pixel clusters
  float chi2xminone_;       //!< minimum of x chi^2 for 1 pixel clusters
  float qmin2_;             //!< tighter minimum cluster charge for valid hit (keeps 99.8% of simulated hits)
  float mpvvav_;            //!< most probable charge in Vavilov distribution (not actually for larger kappa)
  float sigmavav_;          //!< "sigma" scale fctor for Vavilov distribution
  float kappavav_;          //!< kappa parameter for Vavilov distribution
  float mpvvav2_;           //!< most probable charge in 2-cluster Vavilov distribution (not actually for larger kappa)
  float sigmavav2_;         //!< "sigma" scale fctor for 2-cluster Vavilov distribution
  float kappavav2_;         //!< kappa parameter for 2-cluster Vavilov distribution
  float lorywidth_;         //!< Lorentz y-width (sign corrected for fpix frame)
  float lorxwidth_;         //!< Lorentz x-width
  float lorybias_;          //!< Lorentz y-bias
  float lorxbias_;          //!< Lorentz x-bias
  float xsize_;             //!< Pixel x-size
  float ysize_;             //!< Pixel y-size
  float zsize_;             //!< Pixel z-size (thickness)
  float qavg_avg_;          //!< average of cluster charge less than qavg
  float nybins_;            //!< number of bins in each dimension of the y-splitting template
  float nxbins_;            //!< number of bins in each dimension of the x-splitting template
  float r_qMeas_qTrue_;     //!< ratio of measured to true cluster charges
  float fbin_[3];           //!< The QBin definitions in Q_clus/Q_avg
  float fracyone_;          //!< The simulated fraction of single pixel y-clusters
  float fracxone_;          //!< The simulated fraction of single pixel x-clusters
  float fracytwo_;          //!< The simulated fraction of single double-size pixel y-clusters
  float fracxtwo_;          //!< The simulated fraction of single double-size pixel x-clusters
  boost::multi_array<float, 2> temp2dy_;  //!< 2d-primitive for spltting 3-d template
  boost::multi_array<float, 2> temp2dx_;  //!< 2d-primitive for spltting 3-d template

  // The actual template store is a std::vector container

  const std::vector<SiPixelTemplateStore>& thePixelTemp_;
};

#endif
