//
//  SiPixelTemplate.h (v3.40)
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
//
// Created by Morris Swartz on 10/27/06.
// Copyright 2006 __TheJohnsHopkinsUniversity__. All rights reserved.
//
//
 
// Build the template storage structure from several pieces 

#ifndef SiPixelTemplate_h
#define SiPixelTemplate_h 1

#include <vector>
#include <cassert>

struct SiPixelTemplateEntry { //!< Basic template entry corresponding to a single set of track angles 
  int runnum;              //!< number of pixelav run used to generate this entry 
  float alpha;             //!< alpha track angle (defined in CMS CMS IN 2004/014) 
  float cotalpha;          //!< cot(alpha) is proportional to cluster length in x and is basis of interpolation 
  float beta;              //!< beta track angle (defined in CMS CMS IN 2004/014) 
  float cotbeta;           //!< cot(beta) is proportional to cluster length in y and is basis of interpolation 
  float costrk[3];            //!< direction cosines of tracks used to generate this entry 
  float qavg;              //!< average cluster charge for this set of track angles 
  float symax;             //!< average pixel signal for y-projection of cluster 
  float dyone;             //!< mean offset/correction for one pixel y-clusters 
  float syone;             //!< rms for one pixel y-clusters 
  float sxmax;             //!< average pixel signal for x-projection of cluster 
  float dxone;             //!< mean offset/correction for one pixel x-clusters 
  float sxone;             //!< rms for one pixel x-clusters 
  float dytwo;             //!< mean offset/correction for one double-pixel y-clusters 
  float sytwo;             //!< rms for one double-pixel y-clusters 
  float dxtwo;             //!< mean offset/correction for one double-pixel x-clusters 
  float sxtwo;             //!< rms for one double-pixel x-clusters 
  float qmin;              //!< minimum cluster charge for valid hit (keeps 99.9% of simulated hits) 
  float ypar[2][5];        //!< projected y-pixel uncertainty parameterization 
  float ytemp[9][21];      //!< templates for y-reconstruction (binned over 1 central pixel) 
  float xpar[2][5];        //!< projected x-pixel uncertainty parameterization 
  float xtemp[9][7];       //!< templates for x-reconstruction (binned over 1 central pixel) 
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
  float yspare[10];       //!< spare entries
  float xspare[10];       //!< spare entries
} ;




struct SiPixelTemplateHeader {           //!< template header structure 
  char title[80];         //!< template title 
  int ID;                 //!< template ID number 
  int NBy;                //!< number of Barrel y entries 
  int NByx;               //!< number of Barrel y-slices of x entries 
  int NBxx;               //!< number of Barrel x entries in each slice
  int NFy;                //!< number of FPix y entries 
  int NFyx;               //!< number of FPix y-slices of x entries 
  int NFxx;               //!< number of FPix x entries in each slice
  float vbias;            //!< detector bias potential in Volts 
  float temperature;      //!< detector temperature in deg K 
  float fluence;          //!< radiation fluence in n_eq/cm^2 
  float qscale;           //!< Charge scaling to match cmssw and pixelav 
  float s50;              //!< 1/2 of the readout threshold in ADC units 
  int templ_version;      //!< Version number of the template to ensure code compatibility 
} ;





struct SiPixelTemplateStore { //!< template storage structure 
  SiPixelTemplateHeader head;
  SiPixelTemplateEntry entby[60];     //!< 60 Barrel y templates spanning cluster lengths from 0px to +18px 
  SiPixelTemplateEntry entbx[5][7];   //!< 7 Barrel x templates spanning alpha angles from -225mRad to +225mRad in each of 5 slices
  SiPixelTemplateEntry entfy[3];      //!< 3 FPix y templates spanning cluster lengths from 0.45px to 0.95px 
  SiPixelTemplateEntry entfx[2][9];   //!< 9 FPix x templates spanning alpha angles from 75mRad to 675mRad in each of 2 slices
} ;





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
  SiPixelTemplate() {id_current = -1; index_id = -1; cota_current = 0.; cotb_current = 0.; fpix_current=false;} //!< Default constructor
  bool pushfile(int filenum);     // load the private store with info from the 
                                  // file with the index (int) filenum
  
  // Interpolate input alpha and beta angles to produce a working template for each individual hit. 
  void interpolate(int id, bool fpix, float cotalpha, float cotbeta);
  
  // retreive interpolated templates. 
  void ytemp(int fybin, int lybin, float ytemplate[41][25]);
  
  void xtemp(int fxbin, int fxbin, float xtemplate[41][11]);
  
  // Convert vector of projected signals into uncertainties for fitting. 
  void ysigma2(int fypix, int lypix, float sythr, float ysum[25], float ysig2[25]);
  
  void xsigma2(int fxpix, int lxpix, float sxthr, float xsum[11], float xsig2[11]);
  
  // Interpolate qfl correction in y. 
  float yflcorr(int binq, float qfly);
  
  // Interpolate qfl correction in x. 
  float xflcorr(int binq, float qflx);
  
  float qavg() {return pqavg;}        //!< average cluster charge for this set of track angles 
  float qscale() {return pqscale;}         //!< charge scaling factor 
  float s50() {return ps50;}               //!< 1/2 of the pixel threshold signal in adc units 
  float symax() {return psymax;}             //!< average pixel signal for y-projection of cluster 
  float dyone() {return pdyone;}             //!< mean offset/correction for one pixel y-clusters 
  float syone() {return psyone;}             //!< rms for one pixel y-clusters 
  float dytwo() {return pdytwo;}             //!< mean offset/correction for one double-pixel y-clusters 
  float sytwo() {return psytwo;}            //!< rms for one double-pixel y-clusters 
  float sxmax() {return psxmax;}            //!< average pixel signal for x-projection of cluster 
  float dxone() {return pdxone;}             //!< mean offset/correction for one pixel x-clusters 
  float sxone() {return psxone;}             //!< rms for one pixel x-clusters 
  float dxtwo() {return pdxtwo;}             //!< mean offset/correction for one double-pixel x-clusters 
  float sxtwo() {return psxtwo;}             //!< rms for one double-pixel x-clusters 
  float qmin() {return pqmin;}               //!< minimum cluster charge for valid hit (keeps 99.9% of simulated hits)
  float yratio() {return pyratio;}            //!< fractional distance in y between cotbeta templates 
  float yxratio() {return pyxratio;}           //!< fractional distance in y between cotalpha templates slices
  float xxratio() {return pxxratio;}           //!< fractional distance in x between cotalpha templates 
  float yavg(int i) {assert(i>=0 && i<4); return pyavg[i];}         //!< average y-bias of reconstruction binned in 4 charge bins 
  float yrms(int i) {assert(i>=0 && i<4); return pyrms[i];}         //!< average y-rms of reconstruction binned in 4 charge bins 
  float ygx0(int i) {assert(i>=0 && i<4); return pygx0[i];}         //!< average y0 from Gaussian fit binned in 4 charge bins 
  float ygsig(int i) {assert(i>=0 && i<4); return pygsig[i];}       //!< average sigma_y from Gaussian fit binned in 4 charge bins 
  float xavg(int i) {assert(i>=0 && i<4); return pxavg[i];}         //!< average x-bias of reconstruction binned in 4 charge bins 
  float xrms(int i) {assert(i>=0 && i<4); return pxrms[i];}         //!< average x-rms of reconstruction binned in 4 charge bins 
  float xgx0(int i) {assert(i>=0 && i<4); return pxgx0[i];}         //!< average x0 from Gaussian fit binned in 4 charge bins 
  float xgsig(int i) {assert(i>=0 && i<4); return pxgsig[i];}       //!< average sigma_x from Gaussian fit binned in 4 charge bins 
  float chi2yavg(int i) {assert(i>=0 && i<4); return pchi2yavg[i];} //!< average y chi^2 in 4 charge bins 
  float chi2ymin(int i) {assert(i>=0 && i<4); return pchi2ymin[i];} //!< minimum y chi^2 in 4 charge bins 
  float chi2xavg(int i) {assert(i>=0 && i<4); return pchi2xavg[i];} //!< averaage x chi^2 in 4 charge bins
  float chi2xmin(int i) {assert(i>=0 && i<4); return pchi2xmin[i];} //!< minimum y chi^2 in 4 charge bins
//  float yspare(int i) {assert(i>=0 && i<10); return pyspare[i];}    //!< vector of 10 spares interpolated in beta only
//  float xspare(int i) {assert(i>=0 && i<10); return pxspare[i];}    //!< vector of 10 spares interpolated in alpha and beta
  
  
 private:
  
  // Keep current template interpolaion parameters	
  
  int id_current;           //!< current id
  int index_id;             //!< current index
  float cota_current;       //!< current cot alpha
  float cotb_current;       //!< current cot beta
  float abs_cotb;           //!< absolute value of cot beta
  bool fpix_current;        //!< current pix detector (false for BPix, true for FPix)
  
  
  // Keep results of last interpolation to return through member functions
  
  float pqavg;              //!< average cluster charge for this set of track angles 
  float pqscale;            //!< charge scaling factor 
  float ps50;               //!< 1/2 of the pixel threshold signal in adc units 
  float psymax;             //!< average pixel signal for y-projection of cluster 
  float pdyone;             //!< mean offset/correction for one pixel y-clusters 
  float psyone;             //!< rms for one pixel y-clusters 
  float pdytwo;             //!< mean offset/correction for one double-pixel y-clusters 
  float psytwo;             //!< rms for one double-pixel y-clusters 
  float psxmax;             //!< average pixel signal for x-projection of cluster 
  float psxparmax;          //!< maximum pixel signal for parameterization of x uncertainties 
  float pdxone;             //!< mean offset/correction for one pixel x-clusters 
  float psxone;             //!< rms for one pixel x-clusters 
  float pdxtwo;             //!< mean offset/correction for one double-pixel x-clusters 
  float psxtwo;             //!< rms for one double-pixel x-clusters 
  float pqmin;              //!< minimum cluster charge for valid hit (keeps 99.9% of simulated hits)
  float pyratio;            //!< fractional distance in y between cotbeta templates 
  float pyparl[2][5];       //!< projected y-pixel uncertainty parameterization for smaller cotbeta 
  float pyparh[2][5];       //!< projected y-pixel uncertainty parameterization for larger cotbeta 
  float pxparly0[2][5];     //!< projected x-pixel uncertainty parameterization for smaller cotbeta (central alpha)
  float pxparhy0[2][5];     //!< projected x-pixel uncertainty parameterization for larger cotbeta (central alpha)
  float pytemp[41][25];     //!< templates for y-reconstruction (binned over 5 central pixels) 
  float pyxratio;           //!< fractional distance in y between x-slices of cotalpha templates 
  float pxxratio;           //!< fractional distance in x between cotalpha templates 
  float pxpar0[2][5];       //!< projected x-pixel uncertainty parameterization for central cotalpha 
  float pxparl[2][5];       //!< projected x-pixel uncertainty parameterization for smaller cotalpha 
  float pxparh[2][5];       //!< projected x-pixel uncertainty parameterization for larger cotalpha 
  float pxtemp[41][11];     //!< templates for x-reconstruction (binned over 5 central pixels) 
  float pyavg[4];           //!< average y-bias of reconstruction binned in 4 charge bins 
  float pyrms[4];           //!< average y-rms of reconstruction binned in 4 charge bins 
  float pygx0[4];           //!< average y0 from Gaussian fit binned in 4 charge bins 
  float pygsig[4];          //!< average sigma_y from Gaussian fit binned in 4 charge bins 
  float pyflparl[4][6];    //!< Aqfl-parameterized y-correction in 4 charge bins for smaller cotbeta
  float pyflparh[4][6];    //!< Aqfl-parameterized y-correction in 4 charge bins for larger cotbeta
  float pxavg[4];           //!< average x-bias of reconstruction binned in 4 charge bins 
  float pxrms[4];           //!< average x-rms of reconstruction binned in 4 charge bins 
  float pxgx0[4];           //!< average x0 from Gaussian fit binned in 4 charge bins 
  float pxgsig[4];           //!< sigma from Gaussian fit binned in 4 charge bins 
  float pxflparll[4][6];    //!< Aqfl-parameterized x-correction in 4 charge bins for smaller cotbeta, cotalpha
  float pxflparlh[4][6];    //!< Aqfl-parameterized x-correction in 4 charge bins for smaller cotbeta, larger cotalpha
  float pxflparhl[4][6];    //!< Aqfl-parameterized x-correction in 4 charge bins for larger cotbeta, smaller cotalpha
  float pxflparhh[4][6];    //!< Aqfl-parameterized x-correction in 4 charge bins for larger cotbeta, cotalpha 
  float pchi2yavg[4];       //!< average y chi^2 in 4 charge bins
  float pchi2ymin[4];       //!< minimum of y chi^2 in 4 charge bins
  float pchi2xavg[4];       //!< average x chi^2 in 4 charge bins
  float pchi2xmin[4];       //!< minimum of x chi^2 in 4 charge bins 
  float pyspare[10];        //!< vector of 10 spares interpolated in beta only
  float pxspare[10];        //!< vector of 10 spares interpolated in alpha and beta
  
  // The actual template store is a std::vector container

  std::vector< SiPixelTemplateStore > thePixelTemp;
} ;


#endif
