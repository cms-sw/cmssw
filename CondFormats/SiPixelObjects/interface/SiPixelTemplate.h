//
//  SiPixelTemplate.h
//
//
// Created by Morris Swartz on 10/27/06.
// Copyright 2006 __TheJohnsHopkinsUniversity__. All rights reserved.
//
//
 
// Build the template storage structure from several pieces 

#ifndef SiPixelTemplate_h
#define SiPixelTemplate_h 1

#include<vector>


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
  float ypar[2][5];        //!< projected y-pixel uncertainty parameterization 
  float ytemp[9][21];      //!< templates for y-reconstruction (binned over 1 central pixel) 
  float xpar[2][5];        //!< projected x-pixel uncertainty parameterization 
  float xtemp[9][7];       //!< templates for x-reconstruction (binned over 1 central pixel) 
  float yavg[4];           //!< average y-bias of reconstruction binned in 4 charge bins 
  float yrms[4];           //!< average y-rms of reconstruction binned in 4 charge bins 
  float ygx0[4];           //!< average y0 from Gaussian fit binned in 4 charge bins 
  float ygsig[4];          //!< average sigma_y from Gaussian fit binned in 4 charge bins 
  float yeavg[4];          //!< average y-bias of reconstruction for even length clusters binned in 4 charge bins 
  float yerms[4];          //!< average y-rms of reconstruction for even length clusters binned in 4 charge bins 
  float yegx0[4];          //!< average y0 from Gaussian fit for even length clusters binned in 4 charge bins 
  float yegsig[4];         //!< average sigmay from Gaussian fit for even length clusters binned in 4 charge bins 
  float yoavg[4];          //!< average y-bias of reconstruction for odd length clusters binned in 4 charge bins 
  float yorms[4];          //!< average y-rms of reconstruction for odd length clusters binned in 4 charge bins 
  float yogx0[4];          //!< average y0 from Gaussian fit for odd length clusters binned in 4 charge bins 
  float yogsig[4];         //!< average sigmay from Gaussian fit for odd length clusters binned in 4 charge bins 
  float xavg[4];           //!< average x-bias of reconstruction binned in 4 charge bins 
  float xrms[4];           //!< average x-rms of reconstruction binned in 4 charge bins 
  float xgx0[4];           //!< average x0 from Gaussian fit binned in 4 charge bins 
  float xgsig[4];          //!< average sigma_x from Gaussian fit binned in 4 charge bins 
  float xeavg[4];          //!< average x-bias of reconstruction for even length clusters binned in 4 charge bins 
  float xerms[4];          //!< average x-rms of reconstruction for even length clusters binned in 4 charge bins 
  float xegx0[4];          //!< average x0 from Gaussian fit for even length clusters binned in 4 charge bins 
  float xegsig[4];         //!< average sigmax from Gaussian fit for even length clusters binned in 4 charge bins 
  float xoavg[4];          //!< average x-bias of reconstruction for odd length clusters binned in 4 charge bins 
  float xorms[4];          //!< average x-rms of reconstruction for odd length clusters binned in 4 charge bins 
  float xogx0[4];          //!< average x0 from Gaussian fit for odd length clusters binned in 4 charge bins 
  float xogsig[4];         //!< average sigmax from Gaussian fit for odd length clusters binned in 4 charge bins 
} ;




struct SiPixelTemplateHeader {           //!< template header structure 
  char title[80];         //!< template title 
  int ID;                 //!< template ID number 
  int NBy;                //!< number of Barrel y entries 
  int NBx;                //!< number of Barrel x entries 
  int NFy;                //!< number of FPix y entries 
  int NFx;                //!< number of FPix x entries 
  float vbias;            //!< detector bias potential in Volts 
  float temperature;      //!< detector temperature in deg K 
  float fluence;          //!< radiation fluence in n_eq/cm^2 
  float s50;              //!< 1/2 of the readout threshold in ADC units 
} ;





struct SiPixelTemplateStore { //!< template storage structure 
  SiPixelTemplateHeader head;
  SiPixelTemplateEntry entby[119];     //!< 119 Barrel y templates spanning cluster lengths from -18px to +18px 
  SiPixelTemplateEntry entbx[7];       //!< 7 Barrel x templates spanning alpha angles from -225mRad to +225mRad 
  SiPixelTemplateEntry entfy[6];       //!< 6 FPix y templates spanning cluster lengths from -0.95px t0 -0.45px and +0.45px to 0.95px 
  SiPixelTemplateEntry entfx[9];       //!< 9 FPix x templates spanning alpha angles from 75mRad to 675mRad 
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
  
  float qavg() {return pqavg;}        //!< average cluster charge for this set of track angles 
  float s50() {return ps50;}               //!< 1/2 of the pixel threshold signal in adc units 
  float symax() {return psymax;}             //!< average pixel signal for y-projection of cluster 
  float dyone() {return pdyone;}             //!< mean offset/correction for one pixel y-clusters 
  float syone() {return psyone;}             //!< rms for one pixel y-clusters 
  float dytwo() {return pdytwo;}             //!< mean offset/correction for one double-pixel y-clusters 
  float sytwo() {return psytwo;}            //!< rms for one double-pixel y-clusters 
  float sxmax() {return psxmax;}            //!< average pixel signal for x-projection of cluster 
  float sxparmax() {return psxparmax;}          //!< maximum pixel signal for parameterization of x uncertainties 
  float dxone() {return pdxone;}             //!< mean offset/correction for one pixel x-clusters 
  float sxone() {return psxone;}             //!< rms for one pixel x-clusters 
  float dxtwo() {return pdxtwo;}             //!< mean offset/correction for one double-pixel x-clusters 
  float sxtwo() {return psxtwo;}             //!< rms for one double-pixel x-clusters 
  float yratio() {return pyratio;}            //!< fractional distance in y between cotbeta templates 
  float yparl(int i, int j)
  {assert(i>=0 && i<2 && j>= 0 && j<5); return pyparl[i][j];} //!< projected y-pixel uncertainty parameterization for smaller cotbeta
  float yparh(int i, int j) 
  {assert(i>=0 && i<2 && j>= 0 && j<5); return pyparh[i][j];} //!< projected y-pixel uncertainty parameterization for larger cotbeta
  float ytemp(int i, int j) 
  {assert(i>=0 && i<41 && j>= 0 && j<25); return pytemp[i][j];}     //!< templates for y-reconstruction (binned over 5 central pixels) 
  float xratio() {return pxratio;}           //!< fractional distance in x between cotalpha templates 
  float xparl(int i, int j)
  {assert(i>=0 && i<2 && j>= 0 && j<5); return pxparl[i][j];} //!< projected x-pixel uncertainty parameterization for smaller cotbeta
  float xparh(int i, int j) 
  {assert(i>=0 && i<2 && j>= 0 && j<5); return pxparh[i][j];} //!< projected x-pixel uncertainty parameterization for larger cotbeta
  float xtemp(int i, int j) 
  {assert(i>=0 && i<41 && j>= 0 && j<25); return pxtemp[i][j];}     //!< templates for x-reconstruction (binned over 5 central pixels) 
  float yavg(int i) {assert(i>=0 && i<4); return pyavg[i];}         //!< average y-bias of reconstruction binned in 4 charge bins 
  float yrms(int i) {assert(i>=0 && i<4); return pyrms[i];}         //!< average y-rms of reconstruction binned in 4 charge bins 
  float ygx0(int i) {assert(i>=0 && i<4); return pygx0[i];}         //!< average y0 from Gaussian fit binned in 4 charge bins 
  float ygsig(int i) {assert(i>=0 && i<4); return pygsig[i];}       //!< average sigma_y from Gaussian fit binned in 4 charge bins 
  float yeavg(int i) {assert(i>=0 && i<4); return pyeavg[i];}       //!< average y-bias of reconstruction for even length clusters binned in 4 charge bins 
  float yerms(int i) {assert(i>=0 && i<4); return pyerms[i];}       //!< average y-rms of reconstruction for even length clusters binned in 4 charge bins 
  float yegx0(int i) {assert(i>=0 && i<4); return pyegx0[i];}       //!< average y0 from Gaussian fit for even length clusters binned in 4 charge bins 
  float yegsig(int i) {assert(i>=0 && i<4); return pyegsig[i];}     //!< average sigmay from Gaussian fit for even length clusters binned in 4 charge bins 
  float yoavg(int i) {assert(i>=0 && i<4); return pyoavg[i];}       //!< average y-bias of reconstruction for odd length clusters binned in 4 charge bins 
  float yorms(int i) {assert(i>=0 && i<4); return pyorms[i];}       //!< average y-rms of reconstruction for odd length clusters binned in 4 charge bins 
  float yogx0(int i) {assert(i>=0 && i<4); return pyogx0[i];}       //!< average y0 from Gaussian fit for odd length clusters binned in 4 charge bins 
  float yogsig(int i) {assert(i>=0 && i<4); return pyavg[i];}       //!< average sigmay from Gaussian fit for odd length clusters binned in 4 charge bins 
  float xavg(int i) {assert(i>=0 && i<4); return pxavg[i];}         //!< average x-bias of reconstruction binned in 4 charge bins 
  float xrms(int i) {assert(i>=0 && i<4); return pxrms[i];}         //!< average x-rms of reconstruction binned in 4 charge bins 
  float xgx0(int i) {assert(i>=0 && i<4); return pxgx0[i];}         //!< average x0 from Gaussian fit binned in 4 charge bins 
  float xgsig(int i) {assert(i>=0 && i<4); return pxgsig[i];}       //!< average sigma_x from Gaussian fit binned in 4 charge bins 
  float xeavg(int i) {assert(i>=0 && i<4); return pxeavg[i];}       //!< average x-bias of reconstruction for even length clusters binned in 4 charge bins 
  float xerms(int i) {assert(i>=0 && i<4); return pxerms[i];}       //!< average x-rms of reconstruction for even length clusters binned in 4 charge bins 
  float xegx0(int i) {assert(i>=0 && i<4); return pxegx0[i];}       //!< average x0 from Gaussian fit for even length clusters binned in 4 charge bins 
  float xegsig(int i) {assert(i>=0 && i<4); return pxegsig[i];}     //!< average sigmax from Gaussian fit for even length clusters binned in 4 charge bins 
  float xoavg(int i) {assert(i>=0 && i<4); return pxoavg[i];}       //!< average x-bias of reconstruction for odd length clusters binned in 4 charge bins 
  float xorms(int i) {assert(i>=0 && i<4); return pxorms[i];}       //!< average x-rms of reconstruction for odd length clusters binned in 4 charge bins 
  float xogx0(int i) {assert(i>=0 && i<4); return pxogx0[i];}       //!< average x0 from Gaussian fit for odd length clusters binned in 4 charge bins 
  float xogsig(int i) {assert(i>=0 && i<4); return pxavg[i];}       //!< average sigmax from Gaussian fit for odd length clusters binned in 4 charge bins 
  
  
 private:
  
  // Keep current template interpolaion parameters	
  
  int id_current;           //!< current id
  int index_id;             //!< current index
  float cota_current;       //!< current cot alpha
  float cotb_current;       //!< current cot beta
  bool fpix_current;        //!< current pix detector (false for BPix, true for FPix)
  
  
  // Keep results of last interpolation to return through member functions
  
  float pqavg;              //!< average cluster charge for this set of track angles 
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
  float pyratio;            //!< fractional distance in y between cotbeta templates 
  float pyparl[2][5];       //!< projected y-pixel uncertainty parameterization for smaller cotbeta 
  float pyparh[2][5];       //!< projected y-pixel uncertainty parameterization for larger cotbeta 
  float pytemp[41][25];     //!< templates for y-reconstruction (binned over 5 central pixels) 
  float pxratio;            //!< fractional distance in x between cotalpha templates 
  float pxparl[2][5];       //!< projected x-pixel uncertainty parameterization for smaller cotalpha 
  float pxparh[2][5];       //!< projected x-pixel uncertainty parameterization for larger cotalpha 
  float pxtemp[41][11];     //!< templates for x-reconstruction (binned over 5 central pixels) 
  float pyavg[4];           //!< average y-bias of reconstruction binned in 4 charge bins 
  float pyrms[4];           //!< average y-rms of reconstruction binned in 4 charge bins 
  float pygx0[4];           //!< average y0 from Gaussian fit binned in 4 charge bins 
  float pygsig[4];          //!< average sigma_y from Gaussian fit binned in 4 charge bins 
  float pyeavg[4];          //!< average y-bias of reconstruction for even length clusters binned in 4 charge bins 
  float pyerms[4];          //!< average y-rms of reconstruction for even length clusters binned in 4 charge bins 
  float pyegx0[4];          //!< average y0 from Gaussian fit for even length clusters binned in 4 charge bins 
  float pyegsig[4];         //!< average sigmay from Gaussian fit for even length clusters binned in 4 charge bins 
  float pyoavg[4];          //!< average y-bias of reconstruction for odd length clusters binned in 4 charge bins 
  float pyorms[4];          //!< average y-rms of reconstruction for odd length clusters binned in 4 charge bins 
  float pyogx0[4];          //!< average y0 from Gaussian fit for odd length clusters binned in 4 charge bins 
  float pyogsig[4];         //!< average sigmay from Gaussian fit for odd length clusters binned in 4 charge bins 
  float pxavg[4];           //!< average x-bias of reconstruction binned in 4 charge bins 
  float pxrms[4];           //!< average x-rms of reconstruction binned in 4 charge bins 
  float pxgx0[4];           //!< average x0 from Gaussian fit binned in 4 charge bins 
  float pxgsig[4];          //!< average sigma_x from Gaussian fit binned in 4 charge bins 
  float pxeavg[4];          //!< average x-bias of reconstruction for even length clusters binned in 4 charge bins 
  float pxerms[4];          //!< average x-rms of reconstruction for even length clusters binned in 4 charge bins 
  float pxegx0[4];          //!< average x0 from Gaussian fit for even length clusters binned in 4 charge bins 
  float pxegsig[4];         //!< average sigmax from Gaussian fit for even length clusters binned in 4 charge bins 
  float pxoavg[4];          //!< average x-bias of reconstruction for odd length clusters binned in 4 charge bins 
  float pxorms[4];          //!< average x-rms of reconstruction for odd length clusters binned in 4 charge bins 
  float pxogx0[4];          //!< average x0 from Gaussian fit for odd length clusters binned in 4 charge bins 
  float pxogsig[4];         //!< average sigmax from Gaussian fit for odd length clusters binned in 4 charge bins 
  
  // The actual template store is a std::vector container

  std::vector< SiPixelTemplateStore > thePixelTemp;
} ;


#endif
