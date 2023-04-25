//
//  SiPixelTemplate2D.h (v2.65)
//
//  Full 2-D templates for cluster splitting, simulated cluster reweighting, and improved cluster probability
//
// Created by Morris Swartz on 12/01/09.
// V1.01 - fix qavg_ filling
// V1.02 - Add locBz to test if FPix use is out of range
// V1.03 - Fix edge checking on final template to increase template size and to properly truncate cluster
// v2.00 - Major changes to accommodate 2D reconstruction
// v2.10 - Change chi2 and error scaling information to work with partially reconstructed clusters
// v2.20 - Add cluster charge probability information, side loading for template generation
// v2.21 - Double derivative interval [improves fit convergence]
// v2.25 - Resize template store to accommodate FPix Templates
// v2.30 - Fix bug found by P. Shuetze that compromises sqlite file loading
// v2.35 - Add directory path selection to the ascii pushfile method
// v2.50 - Change template storage to dynamically allocated 2D arrays of SiPixelTemplateEntry2D structs
// v2.51 - Ensure that the derivative arrays are correctly zeroed between calls
// v2.52 - Improve cosmetics for increased style points from judges
// v2.60 - Fix FPix multiframe lookup problem [takes +-cotalpha and +-cotbeta]
// v2.61a - Code 2.60 fix correctly
// v2.65 - change double pixel flags to work with new shifted reco code definition
//

// Build the template storage structure from several pieces

#ifndef SiPixelTemplate2D_h
#define SiPixelTemplate2D_h 1

#include <vector>
#include <cassert>
#include "boost/multi_array.hpp"

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplateDefs.h"
#else
#include "SiPixelTemplateDefs.h"
#endif

struct SiPixelTemplateEntry2D {  //!< Basic template entry corresponding to a single set of track angles
  int runnum;                    //!< number of pixelav run used to generate this entry
  float cotalpha;                //!< cot(alpha) is proportional to cluster length in x and is basis of interpolation
  float cotbeta;                 //!< cot(beta) is proportional to cluster length in y and is basis of interpolation
  float costrk[3];               //!< direction cosines of tracks used to generate this entry
  float qavg;                    //!< average cluster charge for this set of track angles
  float pixmax;                  //!< maximum charge for individual pixels in cluster
  float sxymax;                  //!< average pixel signal for use of the error parameterization
  int iymin;                     //!< the minimum nonzero pixel yindex in template (saves time during interpolation)
  int iymax;                     //!< the maximum nonzero pixel yindex in template (saves time during interpolation)
  int jxmin;                     //!< the minimum nonzero pixel xindex in template (saves time during interpolation)
  int jxmax;                     //!< the maximum nonzero pixel xindex in template (saves time during interpolation)
  float xypar[2][5];             //!< pixel uncertainty parameterization
  float lanpar[2][5];            //!< pixel landau distribution parameters
  short int xytemp[7][7][T2YSIZE][T2XSIZE];  //!< templates for y-reconstruction (binned over 1 central pixel)
  float chi2ppix;                            //!< average chi^2 per pixel
  float chi2scale;                           //!< scale factor for the chi2 distribution
  float chi2avgone;                          //!< average y chi^2 for 1 pixel clusters
  float chi2minone;                          //!< minimum of y chi^2 for 1 pixel clusters
  float clsleny;                             //!< cluster y-length in pixels at signal height symax/2
  float clslenx;                             //!< cluster x-length in pixels at signal height sxmax/2
  float mpvvav;      //!< most probable charge in Vavilov distribution (not actually for larger kappa)
  float sigmavav;    //!< "sigma" scale fctor for Vavilov distribution
  float kappavav;    //!< kappa parameter for Vavilov distribution
  float scalexavg;   //!< average x-error scale factor
  float scaleyavg;   //!< average y-error scale factor
  float delyavg;     //!< average length difference between template and cluster
  float delysig;     //!< rms of length difference between template and cluster
  float scalex[4];   //!< x-error scale factor in 4 charge bins
  float scaley[4];   //!< y-error scale factor in 4 charge bins
  float offsetx[4];  //!< x-offset in 4 charge bins
  float offsety[4];  //!< y-offset in 4 charge bins
  float spare[3];
};

struct SiPixelTemplateHeader2D {  //!< template header structure
  int ID;                         //!< template ID number
  int NTy;                        //!< number of Template y entries
  int NTyx;                       //!< number of Template y-slices of x entries
  int NTxx;                       //!< number of Template x-entries in each slice
  int Dtype;                      //!< detector type (0=BPix, 1=FPix)
  float qscale;                   //!< Charge scaling to match cmssw and pixelav
  float lorywidth;                //!< estimate of y-lorentz width for optimal resolution
  float lorxwidth;                //!< estimate of x-lorentz width for optimal resolution
  float lorybias;                 //!< estimate of y-lorentz bias
  float lorxbias;                 //!< estimate of x-lorentz bias
  float Vbias;                    //!< detector bias potential in Volts
  float temperature;              //!< detector temperature in deg K
  float fluence;                  //!< radiation fluence in n_eq/cm^2
  float s50;                      //!< 1/2 of the multihit dcol threshold in electrons
  float ss50;                     //!< 1/2 of the single hit dcol threshold in electrons
  char title[80];                 //!< template title
  int templ_version;              //!< Version number of the template to ensure code compatibility
  float Bfield;                   //!< Bfield in Tesla
  float fbin[3];                  //!< The QBin definitions in Q_clus/Q_avg
  float xsize;                    //!< pixel size (for future use in upgraded geometry)
  float ysize;                    //!< pixel size (for future use in upgraded geometry)
  float zsize;                    //!< pixel size (for future use in upgraded geometry)
};

struct SiPixelTemplateStore2D {  //!< template storage structure

  void resize(int ny, int nx) {
    entry.resize(ny);
    store.resize(nx * ny);
    int off = 0;
    for (int i = 0; i < ny; ++i) {
      entry[i] = store.data() + off;
      off += nx;
    }
    assert(nx * ny == off);
  }

  SiPixelTemplateHeader2D head;  //!< Header information

  //!< use 2d entry to store BPix and FPix entries [dynamically allocated
  std::vector<SiPixelTemplateEntry2D*> entry;

  std::vector<SiPixelTemplateEntry2D> store;
};

// ******************************************************************************************
//! \class SiPixelTemplate2D
//!
//!  A template management class.  SiPixelTemplate contains thePixelTemp
//!  (a std::vector  of SiPixelTemplateStore, each of which is a collection of many
//!  SiPixelTemplateEntries).  Each SiPixelTemplateStore corresponds to a given detector
//!  condition, and is valid for a range of runs.  We allow more than one Store since the
//!  may change over time.
//!
//!  This class reads templates from files via pushfile() method.
//!
//!  The main functionality of SiPixelTemplate is xytemp(), which produces a template
//!  on the fly, given a specific track's alpha and beta.  The results are kept in data
//!  members and accessed via inline getters.
//!
//!  The resulting template is then used by PixelTempReco2D() (a global function) which
//!  get the reference for SiPixelTemplate & templ and uses the current template to
//!  reconstruct the SiPixelRecHit.
// ******************************************************************************************
class SiPixelTemplate2D {
public:
  SiPixelTemplate2D(const std::vector<SiPixelTemplateStore2D>& thePixelTemp) : thePixelTemp_(thePixelTemp) {
    id_current_ = -1;
    index_id_ = -1;
    cota_current_ = 0.;
    cotb_current_ = 0.;
  }  //!< Default constructor

  // load the private store with info from the
  // file with the index (int) filenum ${dir}template_summary_zp${filenum}.out
#ifdef SI_PIXEL_TEMPLATE_STANDALONE
  static bool pushfile(int filenum, std::vector<SiPixelTemplateStore2D>& pixelTemp, std::string dir = "");

  // For calibrations only: load precalculated values -- no interpolation.
  void sideload(SiPixelTemplateEntry2D* entry,
                int iDtype,
                float locBx,
                float locBz,
                float lorwdy,
                float lorwdx,
                float q50,
                float fbin[3],
                float xsize,
                float ysize,
                float zsize);

#else
  static bool pushfile(int filenum,
                       std::vector<SiPixelTemplateStore2D>& pixelTemp,
                       std::string dir = "CalibTracker/SiPixelESProducers/data/");

  // Load from the DB (the default in CMSSW):
  static bool pushfile(const SiPixel2DTemplateDBObject& dbobject, std::vector<SiPixelTemplateStore2D>& pixelTemp);

#endif

  //  Initialize things before interpolating
  bool getid(int id);

  bool interpolate(int id, float cotalpha, float cotbeta, float locBz, float locBx);

  // Interpolate input alpha and beta angles to produce a working template for each individual hit.

  // Works with Phase 0+1
  bool xytemp(float xhit,
              float yhit,
              bool ydouble[BYM2],
              bool xdouble[BXM2],
              float template2d[BXM2][BYM2],
              bool dervatives,
              float dpdx2d[2][BXM2][BYM2],
              float& QTemplate);

  // Overload for backward compatibility

  bool xytemp(float xhit, float yhit, bool ydouble[BYM2], bool xdouble[BXM2], float template2d[BXM2][BYM2]);

  // Overload for backward compatibility with re-weighting code

  bool xytemp(int id,
              float cotalpha,
              float cotbeta,
              float xhit,
              float yhit,
              std::vector<bool>& ydouble,
              std::vector<bool>& xdouble,
              float template2d[BXM2][BYM2]);

  void xysigma2(float qpixel, int index, float& xysig2);

  // Get the interpolated Landau distribution parameters

  void landau_par(float lanpar[2][5]);

  float qavg() { return qavg_; }      //!< average cluster charge for this set of track angles
  float pixmax() { return pixmax_; }  //!< maximum pixel charge
  float qscale() { return qscale_; }  //!< charge scaling factor
  float s50() { return s50_; }        //!< 1/2 of the pixel threshold signal in adc units
  float sxymax() { return sxymax_; }  //!< max pixel signal for pixel error calculation
  float scalex(int i) {
    if (checkIllegalIndex("scalex", 3, i)) {
      return scalex_[i];
    } else {
      return 0.f;
    }
  }  //!< x-error scale factor in 4 charge bins
  float scaley(int i) {
    if (checkIllegalIndex("scaley", 3, i)) {
      return scaley_[i];
    } else {
      return 0.f;
    }
  }  //!< y-error scale factor in 4 charge bins
  float offsetx(int i) {
    if (checkIllegalIndex("offsetx", 3, i)) {
      return offsetx_[i];
    } else {
      return 0.f;
    }
  }  //!< x-offset in 4 charge bins
  float offsety(int i) {
    if (checkIllegalIndex("offsety", 3, i)) {
      return offsety_[i];
    } else {
      return 0.f;
    }
  }  //!< y-offset in 4 charge bins
  float fbin(int i) {
    if (checkIllegalIndex("fbin", 2, i)) {
      return fbin_[i];
    } else {
      return 0.f;
    }
  }                                           //!< Return lower bound of Qbin definition
  float sizex() { return clslenx_; }          //!< return x size of template cluster
  float sizey() { return clsleny_; }          //!< return y size of template cluster
  float chi2ppix() { return chi2ppix_; }      //!< average chi^2 per struck pixel
  float chi2scale() { return chi2scale_; }    //!< scale factor for chi^2 distribution
  float chi2avgone() { return chi2avgone_; }  //!< average y chi^2 for 1 pixel clusters
  float chi2minone() { return chi2minone_; }  //!< minimum of y chi^2 for 1 pixel clusters
  float mpvvav() { return mpvvav_; }          //!< most probable Q in Vavilov distribution
  float sigmavav() { return sigmavav_; }      //!< scale factor in Vavilov distribution
  float kappavav() { return kappavav_; }      //!< kappa parameter in Vavilov distribution
  float lorydrift() { return lorydrift_; }    //!< signed lorentz y-width (microns)
  float lorxdrift() { return lorxdrift_; }    //!< signed lorentz x-width (microns)
  float clsleny() { return clsleny_; }        //!< cluster y-size
  float clslenx() { return clslenx_; }        //!< cluster x-size
  float scaleyavg() { return scaleyavg_; }    //!< y-reco error scaling factor
  float scalexavg() { return scalexavg_; }    //!< x-reco error scaling factor
  float delyavg() {
    return delyavg_;
  }  //!< average difference between clsleny_ and cluster length [with threshold effects]
  float delysig() { return delysig_; }  //!< rms difference between clsleny_ and cluster length [with threshold effects]
  float xsize() { return xsize_; }      //!< pixel x-size (microns)
  float ysize() { return ysize_; }      //!< pixel y-size (microns)
  float zsize() { return zsize_; }      //!< pixel z-size or thickness (microns)
  int storesize() {
    return (int)thePixelTemp_.size();
  }  //!< return the size of the template store (the number of stored IDs

private:
  bool checkIllegalIndex(const std::string whichMethod, int indMax, int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if (i < 0 || i > indMax) {
      throw cms::Exception("DataCorrupt")
          << "SiPixelTemplate2D::" << whichMethod << " called with illegal index = " << i << std::endl;
    }
#else
    assert(i >= 0 && i < indMax + 1);

#endif
    return true;
  }

  // Keep current template interpolaion parameters

  int id_current_;      //!< current id
  int index_id_;        //!< current index
  float cota_current_;  //!< current cot alpha
  float cotb_current_;  //!< current cot beta
  int Nyx_;             //!< number of cot(beta)-entries (columns) in template
  int Nxx_;             //!< number of cot(alpha)-entries (rows) in template
  int Dtype_;           //!< flags BPix (=0) or FPix (=1)
  float cotbeta0_;      //!< minimum cot(beta) covered
  float cotbeta1_;      //!< maximum cot(beta) covered
  float deltacotb_;     //!< cot(beta) bin size
  float cotalpha0_;     //!< minimum cot(alpha) covered
  float cotalpha1_;     //!< maximum cot(alpha) covered
  float deltacota_;     //!< cot(alpha) bin size
  int iy0_;             //!< index of nearest cot(beta) bin
  int iy1_;             //!< index of next-nearest cot(beta) bin
  float adcotb_;        //!< fractional pixel distance of cot(beta) from iy0_
  int jx0_;             //!< index of nearest cot(alpha) bin
  int jx1_;             //!< index of next-nearest cot(alpha) bin
  float adcota_;        //!< fractional pixel distance of cot(alpha) from jx0_
  int imin_;            //!< min y index of templated cluster
  int imax_;            //!< max y index of templated cluster
  int jmin_;            //!< min x index of templated cluster
  int jmax_;            //!< max x index of templated cluster
  bool flip_y_;         //!< flip y sign-sensitive quantities
  bool flip_x_;         //!< flip x sign-sensitive quantities
  bool success_;        //!< true if cotalpha, cotbeta are inside of the acceptance (dynamically loaded)

  // Keep results of last interpolation to return through member functions

  float qavg_;                //!< average cluster charge for this set of track angles
  float pixmax_;              //!< maximum pixel charge
  float qscale_;              //!< charge scaling factor
  float s50_;                 //!< 1/2 of the pixel threshold signal in adc units
  float sxymax_;              //!< average pixel signal for y-projection of cluster
  float xytemp_[BXM2][BYM2];  //!< template for xy-reconstruction
  float xypary0x0_[2][5];     //!< Polynomial error parameterization at ix0,iy0
  float xypary1x0_[2][5];     //!< Polynomial error parameterization at ix0,iy1
  float xypary0x1_[2][5];     //!< Polynomial error parameterization at ix1,iy0
  float lanpar_[2][5];        //!< Interpolated Landau parameters
  float chi2ppix_;            //!< average chi^2 per struck pixel
  float chi2scale_;           //!< scale factor for chi2 distribution
  float chi2avgone_;          //!< average chi^2 for 1 pixel clusters
  float chi2minone_;          //!< minimum of chi^2 for 1 pixel clusters
  float clsleny_;             //!< projected y-length of cluster
  float clslenx_;             //!< projected x-length of cluster
  float scalexavg_;           //!< average x-error scale factor
  float scaleyavg_;           //!< average y-error scale factor
  float delyavg_;             //!< average difference between clsleny_ and cluster length [with threshold effects]
  float delysig_;             //!< rms of difference between clsleny_ and cluster length [with threshold effects]
  float scalex_[4];           //!< x-error scale factor in charge bins
  float scaley_[4];           //!< y-error scale factor in charge bins
  float offsetx_[4];          //!< x-offset in charge bins
  float offsety_[4];          //!< y-offset in charge bins
  float mpvvav_;              //!< most probable Q in Vavilov distribution
  float sigmavav_;            //!< scale factor in Vavilov distribution
  float kappavav_;            //!< kappa parameter in Vavilov distribution
  float lorywidth_;           //!< Lorentz y-width (sign corrected for fpix frame)
  float lorxwidth_;           //!< Lorentz x-width
  float lorydrift_;           //!< Lorentz y-drift
  float lorxdrift_;           //!< Lorentz x-drift
  float xsize_;               //!< Pixel x-size
  float ysize_;               //!< Pixel y-size
  float zsize_;               //!< Pixel z-size (thickness)
  float fbin_[3];             //!< The QBin definitions in Q_clus/Q_avg
  const SiPixelTemplateEntry2D* entry00_;  // Pointer to presently interpolated point [iy,ix]
  const SiPixelTemplateEntry2D* entry10_;  // Pointer to presently interpolated point [iy+1,ix]
  const SiPixelTemplateEntry2D* entry01_;  // Pointer to presently interpolated point [iy,ix+1]

  // The actual template store is a std::vector container
  const std::vector<SiPixelTemplateStore2D>& thePixelTemp_;
};

#endif
