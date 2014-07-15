//
//  SiPixelTemplate2D.h (v1.03)
//
//  Full 2-D templates for cluster splitting, simulated cluster reweighting, and improved cluster probability
//
// Created by Morris Swartz on 12/01/09.
// V1.01 - fix qavg_ filling
// V1.02 - Add locBz to test if FPix use is out of range
// V1.03 - Fix edge checking on final template to increase template size and to properly truncate cluster
//
//
 
// Build the template storage structure from several pieces 

#ifndef SiPixelTemplate2D_h
#define SiPixelTemplate2D_h 1

#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateDefs.h"

#include<vector>
#include<cassert>
#include "boost/multi_array.hpp"

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "FWCore/Utilities/interface/Exception.h"
#endif

struct SiPixelTemplateEntry2D { //!< Basic template entry corresponding to a single set of track angles 
  int runnum;              //!< number of pixelav run used to generate this entry 
  float cotalpha;          //!< cot(alpha) is proportional to cluster length in x and is basis of interpolation 
  float cotbeta;           //!< cot(beta) is proportional to cluster length in y and is basis of interpolation 
  float costrk[3];         //!< direction cosines of tracks used to generate this entry 
  float qavg;              //!< average cluster charge for this set of track angles 
  float pixmax;            //!< maximum charge for individual pixels in cluster
  float sxymax;            //!< average pixel signal for use of the error parameterization 
  int iymin;               //!< the minimum nonzero pixel yindex in template (saves time during interpolation)
  int iymax;               //!< the maximum nonzero pixel yindex in template (saves time during interpolation)
  int jxmin;               //!< the minimum nonzero pixel xindex in template (saves time during interpolation)
  int jxmax;               //!< the maximum nonzero pixel xindex in template (saves time during interpolation)
  float xypar[2][5];       //!< pixel uncertainty parameterization 
  float lanpar[2][5];      //!< pixel landau distribution parameters
  float xytemp[7][7][T2YSIZE][T2XSIZE];  //!< templates for y-reconstruction (binned over 1 central pixel) 
  float chi2avg[4];        //!< average chi^2 in 4 charge bins
  float chi2min[4];        //!< minimum of chi^2 in 4 charge bins
  float chi2avgone;        //!< average y chi^2 for 1 pixel clusters
  float chi2minone;        //!< minimum of y chi^2 for 1 pixel clusters
  float spare[20];
} ;




struct SiPixelTemplateHeader2D {           //!< template header structure 
  char title[80];         //!< template title 
  int ID;                 //!< template ID number 
  int templ_version;      //!< Version number of the template to ensure code compatibility 
  float Bfield;           //!< Bfield in Tesla
  int NTy;                //!< number of Template y entries (= 0 for 2-D templates)
  int NTyx;               //!< number of Template y-slices of x entries 
  int NTxx;               //!< number of Template x-entries in each slice
  int Dtype;              //!< detector type (0=BPix, 1=FPix)
  float Vbias;            //!< detector bias potential in Volts 
  float temperature;      //!< detector temperature in deg K 
  float fluence;          //!< radiation fluence in n_eq/cm^2 
  float qscale;           //!< Charge scaling to match cmssw and pixelav
  float s50;              //!< 1/2 of the readout threshold in ADC units
  float lorywidth;        //!< estimate of y-lorentz width from single pixel offset
  float lorxwidth;        //!< estimate of x-lorentz width from single pixel offset
  float xsize;            //!< pixel size (for future use in upgraded geometry)
  float ysize;            //!< pixel size (for future use in upgraded geometry)
  float zsize;            //!< pixel size (for future use in upgraded geometry)
} ;



struct SiPixelTemplateStore2D { //!< template storage structure 
  SiPixelTemplateHeader2D head;
  boost::multi_array<SiPixelTemplateEntry2D,2> entry;     //!< use 2d entry to store [47][5] barrel entries or [5][9] fpix entries
} ;





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
  SiPixelTemplate2D(const std::vector< SiPixelTemplateStore2D > & thePixelTemp) : thePixelTemp_(thePixelTemp) {id_current_ = -1; index_id_ = -1; cota_current_ = 0.; cotb_current_ = 0.;} //!< Default constructor
  static bool pushfile(int filenum, std::vector< SiPixelTemplateStore2D > & thePixelTemp_);     // load the private store with info from the 
                                  // file with the index (int) filenum
								  
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  static bool pushfile(const SiPixelTemplateDBObject& dbobject, std::vector< SiPixelTemplateStore2D > & thePixelTemp_);     // load the private store with info from db
#endif
  
	
// Interpolate input alpha and beta angles to produce a working template for each individual hit. 
	
  bool xytemp(int id, float cotalpha, float cotbeta, float locBz, float xhit, float yhit, std::vector<bool>& ydouble, std::vector<bool>& xdouble, float template2d[BXM2][BYM2]);
	
// Overload to allow user to avoid the locBz information
	
  bool xytemp(int id, float cotalpha, float cotbeta, float xhit, float yhit, std::vector<bool>& ydouble, std::vector<bool>& xdouble, float template2d[BXM2][BYM2]);
	
// Get pixel signal uncertainties 
	
  void xysigma2(float qpixel, int index, float& xysig2);

// Get the interpolated Landau distribution parameters
	
  void landau_par(float lanpar[2][5]);
  
  float qavg() {return qavg_;}        //!< average cluster charge for this set of track angles 
  float pixmax() {return pixmax_;}    //!< maximum pixel charge 
  float qscale() {return qscale_;}    //!< charge scaling factor 
  float s50() {return s50_;}          //!< 1/2 of the pixel threshold signal in adc units 
  float sxymax() {return sxymax_;}    //!< max pixel signal for pixel error calculation 
  float xytemp(int j, int i) {        //!< get the 2-d template for pixel j (x), i (y) in BXM2 x BYM2 array (x,y)=(0,0) in lower LH corner of pixel[1][1]
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		if(j < 0 || j > BXM3 || i < 0 || i > BYM3) {throw cms::Exception("DataCorrupt") << "SiPixelTemplate2D::xytemp called with illegal indices = " << j << "," << i << std::endl;}
#else	  
		assert((j>=0 && j<BYM2) && (i>=0 && i<BYM2)); 
#endif
	return xytemp_[j][i];} //!< current 2-d template
	float chi2avg(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiPixelTemplate2D::chi2yavg called with illegal index = " << i << std::endl;}
#else	  
	  assert(i>=0 && i<4); 
#endif
     return chi2avg_[i];} //!< average chi^2 in 4 charge bins 
  float chi2min(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiPixelTemplate2D::chi2ymin called with illegal index = " << i << std::endl;}
#else	  	  
     assert(i>=0 && i<4); 
#endif
  return chi2min_[i];} //!< minimum chi^2 in 4 charge bins 
  float chi2avgone() {return chi2avgone_;}                        //!< //!< average y chi^2 for 1 pixel clusters 
  float chi2minone() {return chi2minone_;}                        //!< //!< minimum of y chi^2 for 1 pixel clusters 
  float lorywidth() {return lorywidth_;}                            //!< signed lorentz y-width (microns)
  float lorxwidth() {return lorxwidth_;}                            //!< signed lorentz x-width (microns)
  float xsize() {return xsize_;}                                    //!< pixel x-size (microns)
  float ysize() {return ysize_;}                                    //!< pixel y-size (microns)
  float zsize() {return zsize_;}                                    //!< pixel z-size or thickness (microns)  
  int storesize() {return (int)thePixelTemp_.size();}                    //!< return the size of the template store (the number of stored IDs
  
 private:
  
  // Keep current template interpolaion parameters	
  
	int id_current_;           //!< current id
	int index_id_;             //!< current index
	float cota_current_;       //!< current cot alpha
	float cotb_current_;       //!< current cot beta
	int Nyx_;                  //!< number of cot(beta)-entries (columns) in template
	int Nxx_;                  //!< number of cot(alpha)-entries (rows) in template
	int Dtype_;                //!< flags BPix (=0) or FPix (=1)
	float cotbeta0_;           //!< minimum cot(beta) covered
	float cotbeta1_;           //!< maximum cot(beta) covered
	float deltacotb_;          //!< cot(beta) bin size
	float cotalpha0_;          //!< minimum cot(alpha) covered
	float cotalpha1_;          //!< maximum cot(alpha) covered
	float deltacota_;          //!< cot(alpha) bin size
	int iy0_;                  //!< index of nearest cot(beta) bin
	int iy1_;                  //!< index of next-nearest cot(beta) bin
	float adcotb_;             //!< fractional pixel distance of cot(beta) from iy0_
	int jx0_;                  //!< index of nearest cot(alpha) bin
	int jx1_;                  //!< index of next-nearest cot(alpha) bin
	float adcota_;             //!< fractional pixel distance of cot(alpha) from jx0_
	bool success_;             //!< true if cotalpha, cotbeta are inside of the acceptance (dynamically loaded)
	
  
  // Keep results of last interpolation to return through member functions
  
  float qavg_;              //!< average cluster charge for this set of track angles 
  float pixmax_;            //!< maximum pixel charge
  float qscale_;            //!< charge scaling factor 
  float s50_;               //!< 1/2 of the pixel threshold signal in adc units 
  float sxymax_;            //!< average pixel signal for y-projection of cluster 
  float xytemp_[BXM2][BYM2];//!< templates for y-reconstruction (binned over 5 central pixels) 
  float xypary0x0_[2][5];   //!< Polynomial error parameterization at ix0,iy0
  float xypary1x0_[2][5];   //!< Polynomial error parameterization at ix0,iy1
  float xypary0x1_[2][5];   //!< Polynomial error parameterization at ix1,iy0
  float lanpar_[2][5];      //!< Interpolated Landau parameters
  float chi2avg_[4];       //!< average chi^2 in 4 charge bins
  float chi2min_[4];       //!< minimum of chi^2 in 4 charge bins
  float chi2avgone_;       //!< average chi^2 for 1 pixel clusters
  float chi2minone_;       //!< minimum of chi^2 for 1 pixel clusters
  float lorywidth_;         //!< Lorentz y-width (sign corrected for fpix frame)
  float lorxwidth_;         //!< Lorentz x-width
  float xsize_;             //!< Pixel x-size
  float ysize_;             //!< Pixel y-size
  float zsize_;             //!< Pixel z-size (thickness)
  
  // The actual template store is a std::vector container

  const std::vector< SiPixelTemplateStore2D > & thePixelTemp_;
} ;


#endif
