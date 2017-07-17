//
//  SiStripTemplate.h (v2.10)  [v1.0 based on SiPixelTemplate v8.20]
//
//  V1.05 - add VI optimizations from pixel template object
//  V1.06 - increase angular acceptance (and structure size)
//  V2.00 - add barycenter interpolation and getters, fix calculation for charge deposition to accommodate cota-offsets in the central cotb entries.
//  V2.01 - fix problem with number of spare entries
//  V2.10 - modify methods for cluster splitting to improve speed
//
// Created by Morris Swartz on 10/11/10.
//
//
 
// Build the template storage structure from several pieces 

#ifndef SiStripTemplate_h
#define SiStripTemplate_h 1

#include "SiStripTemplateDefs.h"

#include<vector>
#include<cassert>
#include "boost/multi_array.hpp"

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "FWCore/Utilities/interface/Exception.h"
#endif

struct SiStripTemplateEntry { //!< Basic template entry corresponding to a single set of track angles 
  int runnum;              //!< number of stripav run used to generate this entry 
  float alpha;             //!< alpha track angle (defined in CMS CMS IN 2004/014) 
  float cotalpha;          //!< cot(alpha) is proportional to cluster length in x and is basis of interpolation 
  float beta;              //!< beta track angle (defined in CMS CMS IN 2004/014) 
  float cotbeta;           //!< cot(beta) is proportional to cluster length in y and is basis of interpolation 
  float costrk[3];            //!< direction cosines of tracks used to generate this entry 
  float qavg;              //!< average cluster charge for this set of track angles (now includes threshold effects)
  float sxmax;             //!< average strip signal for x-projection of cluster 
  float dxone;             //!< mean offset/correction for one strip x-clusters 
  float sxone;             //!< rms for one strip x-clusters 
  float qmin;              //!< minimum cluster charge for valid hit (keeps 99.9% of simulated hits) 
  float qmin2;             //!< tighter minimum cluster charge for valid hit (keeps 99.8% of simulated hits)
  float clslenx;           //!< cluster x-length in strips at signal height sxmax/2
  float mpvvav;            //!< most probable charge in Vavilov distribution (not actually for larger kappa)
  float sigmavav;          //!< "sigma" scale fctor for Vavilov distribution
  float kappavav;          //!< kappa parameter for Vavilov distribution
  float mpvvav2;           //!< most probable charge in Vavilov distribution for 2 merged clusters (not actually for larger kappa)
  float sigmavav2;         //!< "sigma" scale fctor for Vavilov distribution for 2 merged clusters
  float kappavav2;         //!< kappa parameter for Vavilov distribution for 2 merged clusters
  float xpar[2][5];        //!< projected x-strip uncertainty parameterization 
  float xtemp[9][TSXSIZE];  //!< templates for x-reconstruction (binned over 1 central strip) 
  float xavg[4];           //!< average x-bias of reconstruction binned in 4 charge bins 
  float xrms[4];           //!< average x-rms of reconstruction binned in 4 charge bins 
  float xgx0[4];           //!< average x0 from Gaussian fit binned in 4 charge bins 
  float xgsig[4];          //!< average sigma_x from Gaussian fit binned in 4 charge bins 
  float xflpar[4][6];      //!< Aqfl-parameterized x-correction in 4 charge bins 
  float chi2xavg[4];       //!< average x chi^2 in 4 charge bins
  float chi2xmin[4];       //!< minimum of x chi^2 in 4 charge bins 
  float chi2xavgone;       //!< average x chi^2 for 1 strip clusters
  float chi2xminone;       //!< minimum of x chi^2 for 1 strip clusters
  float xavgc2m[4];        //!< 1st pass chi2 min search: average x-bias of reconstruction binned in 4 charge bins 
  float xrmsc2m[4];        //!< 1st pass chi2 min search: average x-rms of reconstruction binned in 4 charge bins 
  float xgx0c2m[4];        //!< 1st pass chi2 min search: average x0 from Gaussian fit binned in 4 charge bins 
  float xgsigc2m[4];       //!< 1st pass chi2 min search: average sigma_x from Gaussian fit binned in 4 charge bins 
  float chi2xavgc2m[4];    //!< 1st pass chi2 min search: average x chi^2 in 4 charge bins (merged clusters) 
  float chi2xminc2m[4];    //!< 1st pass chi2 min search: minimum of x chi^2 in 4 charge bins (merged clusters) 
  float xavggen[4];        //!< generic algorithm: average x-bias of reconstruction binned in 4 charge bins 
  float xrmsgen[4];        //!< generic algorithm: average x-rms of reconstruction binned in 4 charge bins 
  float xgx0gen[4];        //!< generic algorithm: average x0 from Gaussian fit binned in 4 charge bins 
  float xgsiggen[4];       //!< generic algorithm: average sigma_x from Gaussian fit binned in 4 charge bins 
  float xavgbcn[4];        //!< barycenter: average x-bias of reconstruction binned in 4 charge bins 
  float xrmsbcn[4];        //!< barycenter: average x-rms of reconstruction binned in 4 charge bins 
  float xgx0bcn[4];        //!< barycenter: average x0 from Gaussian fit binned in 4 charge bins 
  float xgsigbcn[4];       //!< barycenter: average sigma_x from Gaussian fit binned in 4 charge bins 
  float qbfrac[3];         //!< fraction of sample in qbin = 0-2 (>=3 is the complement)
  float fracxone;          //!< fraction of sample with xsize = 1
  float qavg_avg;          //!< average cluster charge of clusters that are less than qavg (normalize 2-D simple templates)
  float qavg_spare;        //!< spare cluster charge
  float spare[7];
} ;




struct SiStripTemplateHeader {           //!< template header structure 
  char title[80];         //!< template title 
  int ID;                 //!< template ID number 
  int templ_version;      //!< Version number of the template to ensure code compatibility 
  float Bfield;           //!< Bfield in Tesla
  int NTy;                //!< number of Template y entries 
  int NTyx;               //!< number of Template y-slices of x entries 
  int NTxx;               //!< number of Template x-entries in each slice
  int Dtype;              //!< detector type (0=BPix, 1=FPix)
  float Vbias;            //!< detector bias potential in Volts 
  float temperature;      //!< detector temperature in deg K 
  float fluence;          //!< radiation fluence in n_eq/cm^2 
  float qscale;           //!< Charge scaling to match cmssw and stripav
  float s50;              //!< 1/2 of the readout threshold in ADC units
  float lorywidth;        //!< estimate of y-lorentz width from single strip offset
  float lorxwidth;        //!< estimate of x-lorentz width from single strip offset
  float xsize;            //!< strip size (for future use in upgraded geometry)
  float ysize;            //!< strip size (for future use in upgraded geometry)
  float zsize;            //!< strip size (for future use in upgraded geometry)
} ;



struct SiStripTemplateStore { //!< template storage structure 
  SiStripTemplateHeader head;
#ifndef SI_STRIP_TEMPLATE_USE_BOOST 
  SiStripTemplateEntry enty[31];     //!< 60 Barrel y templates spanning cluster lengths from 0px to +18px [28 entries for fstrp]
  SiStripTemplateEntry entx[5][73];  //!< 29 Barrel x templates spanning cluster lengths from -6px (-1.125Rad) to +6px (+1.125Rad) in each of 5 slices [3x29 for fstrp]
#else
  boost::multi_array<SiStripTemplateEntry,1> enty;     //!< use 1d entry to store [60] barrel entries or [28] fstrp entries
  boost::multi_array<SiStripTemplateEntry,2> entx;     //!< use 2d entry to store [5][29] barrel entries or [3][29] fstrp entries	
#endif
} ;


// ******************************************************************************************
//! \class SiStripTemplate 
//!
//!  A template management class.  SiStripTemplate contains theStripTemp 
//!  (a std::vector  of SiStripTemplateStore, each of which is a collection of many 
//!  SiStripTemplateEntries).  Each SiStripTemplateStore corresponds to a given detector 
//!  condition, and is valid for a range of runs.  We allow more than one Store since the 
//!  may change over time.
//!
//!  This class reads templates from files via pushfile() method.
//! 
//!  The main functionality of SiStripTemplate is interpolate(), which produces a template
//!  on the fly, given a specific track's alpha and beta.  The results are kept in data 
//!  members and accessed via inline getters.
//!  
//!  The resulting template is then used by StripTempReco2D() (a global function) which
//!  get the reference for SiStripTemplate & templ and uses the current template to 
//!  reconstruct the SiStripRecHit.
// ******************************************************************************************
class SiStripTemplate {
 public:
  SiStripTemplate(const std::vector< SiStripTemplateStore > & theStripTemp) : theStripTemp_(theStripTemp) {id_current_ = -1; index_id_ = -1; cota_current_ = 0.; cotb_current_ = 0.;} //!< Default constructor
  static bool pushfile(int filenum, std::vector< SiStripTemplateStore > & theStripTemp_);     // load the private store with info from the 
                                  // file with the index (int) filenum
								  
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
  static bool pushfile(const SiPixelTemplateDBObject& dbobject, std::vector< SiStripTemplateStore > & theStripTemp_);     // load the private store with info from db
#endif
  
	
// Interpolate input alpha and beta angles to produce a working template for each individual hit. 
  bool interpolate(int id, float cotalpha, float cotbeta, float locBy);
	
// overload for compatibility. 
  bool interpolate(int id, float cotalpha, float cotbeta);
  
// retreive interpolated templates. 
  void xtemp(int fxbin, int lxbin, float xtemplate[41][BSXSIZE]);
	
// interpolate a scaled cluster shape. 
  void sxtemp(float xhit, std::vector<float>& cluster);
	
//Method to estimate the central strip of the interpolated x-template
  int cxtemp();
	
// new methods to build templates from two interpolated clusters (for splitting) 
  void xtemp3d_int(int nxpix, int& nxbins);
    
  void xtemp3d(int j, int k, std::vector<float>& xtemplate);
  
// Convert vector of projected signals into uncertainties for fitting. 
  void xsigma2(int fxstrp, int lxstrp, float sxthr, float xsum[BSXSIZE], float xsig2[BSXSIZE]);
  
// Interpolate qfl correction in x. 
  float xflcorr(int binq, float qflx);
  
// Interpolate input beta angle to estimate the average charge. return qbin flag for input cluster charge. 
  int qbin(int id, float cotalpha, float cotbeta, float qclus);
	
//Method to interpolate Vavilov distribution parameters
  void vavilov_pars(double& mpv, double& sigma, double& kappa);
	
//Method to interpolate Vavilov distribution parameters for merged clusters
	void vavilov2_pars(double& mpv, double& sigma, double& kappa);
	
   
  float qavg() {return qavg_;}        //!< average cluster charge for this set of track angles 
  float qscale() {return qscale_;}         //!< charge scaling factor 
  float s50() {return s50_;}               //!< 1/2 of the strip threshold signal in electrons
  float sxmax() {return sxmax_;}            //!< average strip signal for x-projection of cluster 
  float dxone() {return dxone_;}             //!< mean offset/correction for one strip x-clusters 
  float sxone() {return sxone_;}             //!< rms for one strip x-clusters 
  float qmin() {return qmin_;}               //!< minimum cluster charge for valid hit (keeps 99.9% of simulated hits)
  float qmin(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 1) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::qmin called with illegal index = " << i << std::endl;}
#else
	  assert(i>=0 && i<2); 
#endif
     if(i==0){return qmin_;}else{return qmin2_;}} //!< minimum cluster charge for valid hit (keeps 99.9% or 99.8% of simulated hits)
  float clslenx() {return clslenx_;}         //!< x-size of smaller interpolated template in strips
  float yratio() {return yratio_;}            //!< fractional distance in y between cotbeta templates 
  float yxratio() {return yxratio_;}           //!< fractional distance in y between cotalpha templates slices
  float xxratio() {return xxratio_;}           //!< fractional distance in x between cotalpha templates 
  float xavg(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::xavg called with illegal index = " << i << std::endl;}
#else	  
	  assert(i>=0 && i<4); 
#endif
     return xavg_[i];}         //!< average x-bias of reconstruction binned in 4 charge bins 
  float xrms(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::xrms called with illegal index = " << i << std::endl;}
#else	  
	  assert(i>=0 && i<4); 
#endif
     return xrms_[i];}         //!< average x-rms of reconstruction binned in 4 charge bins 
  float xgx0(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::xgx0 called with illegal index = " << i << std::endl;}
#else	  
	  assert(i>=0 && i<4); 
#endif
     return xgx0_[i];}         //!< average x0 from Gaussian fit binned in 4 charge bins 
  float xgsig(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::xgsig called with illegal index = " << i << std::endl;}
#else	  
	  assert(i>=0 && i<4); 
#endif
     return xgsig_[i];}       //!< average sigma_x from Gaussian fit binned in 4 charge bins 
  float chi2xavg(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::chi2xavg called with illegal index = " << i << std::endl;}
#else	  
	  assert(i>=0 && i<4); 
#endif
     return chi2xavg_[i];} //!< averaage x chi^2 in 4 charge bins
  float chi2xmin(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::chi2xmin called with illegal index = " << i << std::endl;}
#else	  
	  assert(i>=0 && i<4);
#endif
     return chi2xmin_[i];} //!< minimum y chi^2 in 4 charge bins
  float xavgc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::xavgc2m called with illegal index = " << i << std::endl;}
#else	  
	  assert(i>=0 && i<4); 
#endif
     return xavgc2m_[i];}   //!< 1st pass chi2 min search: average x-bias of reconstruction binned in 4 charge bins 
  float xrmsc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::xrmsc2m called with illegal index = " << i << std::endl;}
#else	  
	  assert(i>=0 && i<4); 
#endif
     return xrmsc2m_[i];}   //!< 1st pass chi2 min search: average x-rms of reconstruction binned in 4 charge bins 
  float xgx0c2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::xgx0cm2 called with illegal index = " << i << std::endl;}
#else	  
	  assert(i>=0 && i<4); 
#endif
     return xgx0c2m_[i];}   //!< 1st pass chi2 min search: average x0 from Gaussian fit binned in 4 charge bins 
  float xgsigc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	  if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::xgsigc2m called with illegal index = " << i << std::endl;}
#else	  
	  assert(i>=0 && i<4); 
#endif
     return xgsigc2m_[i];} //!< 1st pass chi2 min search: average sigma_x from Gaussian fit binned in 4 charge bins 
	float chi2xavgc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiPixelTemplate::chi2xavgc2m called with illegal index = " << i << std::endl;}
#else	  
		assert(i>=0 && i<4); 
#endif
		return chi2xavgc2m_[i];}   //!< 1st pass chi2 min search: average x-chisq for merged clusters 
	float chi2xminc2m(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiPixelTemplate::chi2xminc2m called with illegal index = " << i << std::endl;}
#else	  
		assert(i>=0 && i<4); 
#endif
	return chi2xminc2m_[i];} //!< 1st pass chi2 min search: minimum x-chisq for merged clusters 
	float xavgbcn(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::xavgbcn called with illegal index = " << i << std::endl;}
#else	  
		assert(i>=0 && i<4); 
#endif
        return xavgbcn_[i];}   //!< 1st pass chi2 min search: average x-bias of reconstruction binned in 4 charge bins 
	float xrmsbcn(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::xrmsbcn called with illegal index = " << i << std::endl;}
#else	  
		assert(i>=0 && i<4); 
#endif
        return xrmsbcn_[i];}   //!< 1st pass chi2 min search: average x-rms of reconstruction binned in 4 charge bins 
	float xgx0bcn(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::xgx0cm2 called with illegal index = " << i << std::endl;}
#else	  
		assert(i>=0 && i<4); 
#endif
        return xgx0bcn_[i];}   //!< 1st pass chi2 min search: average x0 from Gaussian fit binned in 4 charge bins 
	float xgsigbcn(int i) {
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		if(i < 0 || i > 3) {throw cms::Exception("DataCorrupt") << "SiStripTemplate::xgsigbcn called with illegal index = " << i << std::endl;}
#else	  
		assert(i>=0 && i<4); 
#endif
        return xgsigbcn_[i];} //!< 1st pass chi2 min search: average sigma_x from Gaussian fit binned in 4 charge bins 
  float chi2xavgone() {return chi2xavgone_;}                        //!< //!< average x chi^2 for 1 strip clusters 
  float chi2xminone() {return chi2xminone_;}                        //!< //!< minimum of x chi^2 for 1 strip clusters 
  float lorxwidth() {return lorxwidth_;}                            //!< signed lorentz x-width (microns)
  float mpvvav() {return mpvvav_;}                                  //!< most probable charge in Vavilov distribution (not actually for larger kappa)
  float sigmavav() {return sigmavav_;}                              //!< "sigma" scale fctor for Vavilov distribution
  float kappavav() {return kappavav_;}                              //!< kappa parameter for Vavilov distribution
  float xsize() {return xsize_;}                                    //!< strip x-size (microns)
  float ysize() {return ysize_;}                                    //!< strip y-size (microns)
  float zsize() {return zsize_;}                                    //!< strip z-size or thickness (microns)
//  float yspare(int i) {assert(i>=0 && i<5); return pyspare[i];}    //!< vector of 5 spares interpolated in beta only
//  float xspare(int i) {assert(i>=0 && i<10); return pxspare[i];}    //!< vector of 10 spares interpolated in alpha and beta
  
  
 private:
  
  // Keep current template interpolaion parameters	
  
  int id_current_;           //!< current id
  int index_id_;             //!< current index
  float cota_current_;       //!< current cot alpha
  float cotb_current_;       //!< current cot beta
  float abs_cotb_;           //!< absolute value of cot beta
  bool success_;             //!< true if cotalpha, cotbeta are inside of the acceptance (dynamically loaded)
  
  
  // Keep results of last interpolation to return through member functions
  
	float qavg_;              //!< average cluster charge for this set of track angles 
	float pixmax_;            //!< maximum strip charge
	float qscale_;            //!< charge scaling factor 
	float s50_;               //!< 1/2 of the strip threshold signal in adc units 
	float sxmax_;             //!< average strip signal for x-projection of cluster 
	float sxparmax_;          //!< maximum strip signal for parameterization of x uncertainties 
	float syparmax_;          //!< maximum strip signal for parameterization of y-slice x uncertainties 
	float dxone_;             //!< mean offset/correction for one strip x-clusters 
	float sxone_;             //!< rms for one strip x-clusters 
	float dxtwo_;             //!< mean offset/correction for one double-strip x-clusters 
	float sxtwo_;             //!< rms for one double-strip x-clusters 
	float qmin_;              //!< minimum cluster charge for valid hit (keeps 99.9% of simulated hits)
	float clslenx_;           //!< x-cluster length of smaller interpolated template in strips 
	float xparly0_[2][5];     //!< projected x-strip uncertainty parameterization for smaller cotbeta (central alpha)
	float xparhy0_[2][5];     //!< projected x-strip uncertainty parameterization for larger cotbeta (central alpha)
	float yratio_;            //!< fractional distance in y between y-slices of cotbeta templates 
	float yxratio_;           //!< fractional distance in y between x-slices of cotalpha templates 
	float xxratio_;           //!< fractional distance in x between cotalpha templates 
	float xpar0_[2][5];       //!< projected x-strip uncertainty parameterization for central cotalpha 
	float xparl_[2][5];       //!< projected x-strip uncertainty parameterization for smaller cotalpha 
	float xparh_[2][5];       //!< projected x-strip uncertainty parameterization for larger cotalpha 
	float xtemp_[9][BSXSIZE];  //!< templates for x-reconstruction (binned over 5 central strips) 
	float xavg_[4];           //!< average x-bias of reconstruction binned in 4 charge bins 
	float xrms_[4];           //!< average x-rms of reconstruction binned in 4 charge bins 
	float xgx0_[4];           //!< average x0 from Gaussian fit binned in 4 charge bins 
	float xgsig_[4];           //!< sigma from Gaussian fit binned in 4 charge bins 
	float xflparll_[4][6];    //!< Aqfl-parameterized x-correction in 4 charge bins for smaller cotbeta, cotalpha
	float xflparlh_[4][6];    //!< Aqfl-parameterized x-correction in 4 charge bins for smaller cotbeta, larger cotalpha
	float xflparhl_[4][6];    //!< Aqfl-parameterized x-correction in 4 charge bins for larger cotbeta, smaller cotalpha
	float xflparhh_[4][6];    //!< Aqfl-parameterized x-correction in 4 charge bins for larger cotbeta, cotalpha 
	float xavgc2m_[4];        //!< 1st pass chi2 min search: average x-bias of reconstruction binned in 4 charge bins 
	float xrmsc2m_[4];        //!< 1st pass chi2 min search: average x-rms of reconstruction binned in 4 charge bins 
	float xgx0c2m_[4];        //!< 1st pass chi2 min search: average x-bias of reconstruction binned in 4 charge bins 
	float xgsigc2m_[4];       //!< 1st pass chi2 min search: average x-rms of reconstruction binned in 4 charge bins 
	float chi2xavg_[4];       //!< average x chi^2 in 4 charge bins
	float chi2xmin_[4];       //!< minimum of x chi^2 in 4 charge bins 
	float chi2xavgc2m_[4];    //!< 1st pass chi2 min search: average x-chisq for merged clusters 
	float chi2xminc2m_[4];    //!< 1st pass chi2 min search: minimum x-chisq for merged clusters 
	float xavgbcn_[4];        //!< barycenter: average x-bias of reconstruction binned in 4 charge bins 
	float xrmsbcn_[4];        //!< barycenter: average x-rms of reconstruction binned in 4 charge bins 
	float xgx0bcn_[4];        //!< barycenter: average x-bias of reconstruction binned in 4 charge bins 
	float xgsigbcn_[4];       //!< barycenter: average x-rms of reconstruction binned in 4 charge bins 
	float chi2xavgone_;       //!< average x chi^2 for 1 strip clusters
	float chi2xminone_;       //!< minimum of x chi^2 for 1 strip clusters
	float qmin2_;             //!< tighter minimum cluster charge for valid hit (keeps 99.8% of simulated hits)
	float mpvvav_;            //!< most probable charge in Vavilov distribution (not actually for larger kappa)
	float sigmavav_;          //!< "sigma" scale fctor for Vavilov distribution
	float kappavav_;          //!< kappa parameter for Vavilov distribution
	float mpvvav2_;           //!< most probable charge in 2-cluster Vavilov distribution (not actually for larger kappa)
	float sigmavav2_;         //!< "sigma" scale fctor for 2-cluster Vavilov distribution
	float kappavav2_;         //!< kappa parameter for 2-cluster Vavilov distribution
	float lorxwidth_;         //!< Lorentz x-width
	float xsize_;             //!< Pixel x-size
	float ysize_;             //!< Pixel y-size
	float zsize_;             //!< Pixel z-size (thickness)
	float qavg_avg_;          //!< average of cluster charge less than qavg
    float nxbins_;            //!< number of bins in each dimension of the x-splitting template
    boost::multi_array<float,2> temp2dx_; //!< 2d-primitive for spltting 3-d template
	
	
	// The actual template store is a std::vector container
	
	const std::vector< SiStripTemplateStore > & theStripTemp_;
} ;


#endif
