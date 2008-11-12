//
//  SiPixelTemplate.cc  Version 5.00 
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
//  Add bias info for Barrel and FPix separately in the header
//  Improve the charge estimation for larger cot(alpha) tracks
//  Change interpolate method to return false boolean if track angles are outside of range
//  Add template info and method for truncation information
//  Change to allow template sizes to be changed at compile time
//
//  Created by Morris Swartz on 10/27/06.
//  Copyright 2006 __TheJohnsHopkinsUniversity__. All rights reserved.
//
//

//#include <stdlib.h> 
//#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <vector>
//#include "boost/multi_array.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>


#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplate.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) LogError(x)
#define LOGINFO(x) LogInfo(x)
#define ENDL " "
using namespace edm;
#else
#include "SiPixelTemplate.h"
#define LOGERROR(x) std::cout << x << ": "
#define LOGINFO(x) std::cout << x << ": "
#define ENDL std::endl
#endif

//**************************************************************** 
//! This routine initializes the global template structures from an
//! external file template_summary_zpNNNN where NNNN are four digits 
//! \param dbobject - db storing multiple template calibrations
//**************************************************************** 
bool SiPixelTemplate::pushfile(const SiPixelTemplateDBObject& dbobject)
{
	// Add template stored in external dbobject to theTemplateStore
    
	// Local variables 
	int i, j, k, l;
	const char *tempfile;
	char title[80];
	char c;
	const int code_version={11};

	// We must create a new object because dbobject must be a const and our stream must not be
	SiPixelTemplateDBObject db = dbobject;

	// Create a local template storage entry
	SiPixelTemplateStore theCurrentTemp;

	// Fill the template storage for each template calibration stored in the db
	for(int m=0; m<db.numOfTempl(); ++m)
	{
			
// Read-in a header string first and print it    
    
		SiPixelTemplateDBObject::char2float temp;
		for (int i=0; i<20; ++i) {
			temp.f = db.sVector()[db.index()];
			theCurrentTemp.head.title[4*i] = temp.c[0];
			theCurrentTemp.head.title[4*i+1] = temp.c[1];
			theCurrentTemp.head.title[4*i+2] = temp.c[2];
			theCurrentTemp.head.title[4*i+3] = temp.c[3];
			db.incrementIndex(1);
		}
		theCurrentTemp.head.title[79] = '\0';
		LOGINFO("SiPixelTemplate") << "Loading Pixel Template File - " << theCurrentTemp.head.title << ENDL;
    
// next, the header information     
    
		db >> theCurrentTemp.head.ID >> theCurrentTemp.head.NBy >> theCurrentTemp.head.NByx >> theCurrentTemp.head.NBxx
			 >> theCurrentTemp.head.NFy >> theCurrentTemp.head.NFyx >> theCurrentTemp.head.NFxx >> theCurrentTemp.head.Bbias 
			 >> theCurrentTemp.head.Fbias >> theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >> theCurrentTemp.head.qscale
			 >> theCurrentTemp.head.s50 >> theCurrentTemp.head.templ_version;
		
		if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
		
		LOGINFO("SiPixelTemplate") << "Template ID = " << theCurrentTemp.head.ID << ", NBy = " << theCurrentTemp.head.NBy << ", NByx = " << theCurrentTemp.head.NByx 
		 << ", NBxx = " << theCurrentTemp.head.NBxx << ", NFy = " << theCurrentTemp.head.NFy << ", NFyx = " << theCurrentTemp.head.NFyx
		 << ", NFxx = " << theCurrentTemp.head.NFxx << ", Barrel bias voltage " << theCurrentTemp.head.Bbias << ", FPix bias voltage " << theCurrentTemp.head.Fbias << ", temperature "
		 << theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence << ", Q-scaling factor " << theCurrentTemp.head.qscale
		 << ", 1/2 threshold " << theCurrentTemp.head.s50 << ", Template Version " << theCurrentTemp.head.templ_version << ENDL;    
			
		if(theCurrentTemp.head.templ_version != code_version) {LOGERROR("SiPixelTemplate") << "code expects version " << code_version << ", no template load" << ENDL; return false;}
		 
// next, loop over all barrel y-angle entries   

		for (i=0; i < theCurrentTemp.head.NBy; ++i) {     
			
			db >> theCurrentTemp.entby[i].runnum >> theCurrentTemp.entby[i].costrk[0] 
				 >> theCurrentTemp.entby[i].costrk[1] >> theCurrentTemp.entby[i].costrk[2]; 
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 1, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			  
// Calculate the alpha, beta, and cot(beta) for this entry 

			theCurrentTemp.entby[i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entby[i].costrk[2], (double)theCurrentTemp.entby[i].costrk[0]));
			
			theCurrentTemp.entby[i].cotalpha = theCurrentTemp.entby[i].costrk[0]/theCurrentTemp.entby[i].costrk[2];
			
			theCurrentTemp.entby[i].beta = static_cast<float>(atan2((double)theCurrentTemp.entby[i].costrk[2], (double)theCurrentTemp.entby[i].costrk[1]));
			
			theCurrentTemp.entby[i].cotbeta = theCurrentTemp.entby[i].costrk[1]/theCurrentTemp.entby[i].costrk[2];
			
			db >> theCurrentTemp.entby[i].qavg >> theCurrentTemp.entby[i].pixmax >> theCurrentTemp.entby[i].symax >> theCurrentTemp.entby[i].dyone
				 >> theCurrentTemp.entby[i].syone >> theCurrentTemp.entby[i].sxmax >> theCurrentTemp.entby[i].dxone >> theCurrentTemp.entby[i].sxone;
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 2, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			
			db >> theCurrentTemp.entby[i].dytwo >> theCurrentTemp.entby[i].sytwo >> theCurrentTemp.entby[i].dxtwo 
				 >> theCurrentTemp.entby[i].sxtwo >> theCurrentTemp.entby[i].qmin >> theCurrentTemp.entby[i].clsleny >> theCurrentTemp.entby[i].clslenx;
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 3, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			
			for (j=0; j<2; ++j) {
				
				db >> theCurrentTemp.entby[i].ypar[j][0] >> theCurrentTemp.entby[i].ypar[j][1] 
					 >> theCurrentTemp.entby[i].ypar[j][2] >> theCurrentTemp.entby[i].ypar[j][3] >> theCurrentTemp.entby[i].ypar[j][4];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 4, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			  
			}
			
			for (j=0; j<9; ++j) {
				
				for (k=0; k<TYSIZE; ++k) {db >> theCurrentTemp.entby[i].ytemp[j][k];}
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 5, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<2; ++j) {
				
				db >> theCurrentTemp.entby[i].xpar[j][0] >> theCurrentTemp.entby[i].xpar[j][1] 
					 >> theCurrentTemp.entby[i].xpar[j][2] >> theCurrentTemp.entby[i].xpar[j][3] >> theCurrentTemp.entby[i].xpar[j][4];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 6, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			  
			}
			
			for (j=0; j<9; ++j) {
				
				for (k=0; k<TXSIZE; ++k) {db >> theCurrentTemp.entby[i].xtemp[j][k];} 
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 7, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entby[i].yavg[j] >> theCurrentTemp.entby[i].yrms[j] >> theCurrentTemp.entby[i].ygx0[j] >> theCurrentTemp.entby[i].ygsig[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 8, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entby[i].yflpar[j][0] >> theCurrentTemp.entby[i].yflpar[j][1] >> theCurrentTemp.entby[i].yflpar[j][2] 
					 >> theCurrentTemp.entby[i].yflpar[j][3] >> theCurrentTemp.entby[i].yflpar[j][4] >> theCurrentTemp.entby[i].yflpar[j][5];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 9, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entby[i].xavg[j] >> theCurrentTemp.entby[i].xrms[j] >> theCurrentTemp.entby[i].xgx0[j] >> theCurrentTemp.entby[i].xgsig[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 10, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entby[i].xflpar[j][0] >> theCurrentTemp.entby[i].xflpar[j][1] >> theCurrentTemp.entby[i].xflpar[j][2] 
					 >> theCurrentTemp.entby[i].xflpar[j][3] >> theCurrentTemp.entby[i].xflpar[j][4] >> theCurrentTemp.entby[i].xflpar[j][5];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 11, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entby[i].chi2yavg[j] >> theCurrentTemp.entby[i].chi2ymin[j] >> theCurrentTemp.entby[i].chi2xavg[j] >> theCurrentTemp.entby[i].chi2xmin[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 12, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entby[i].yavgc2m[j] >> theCurrentTemp.entby[i].yrmsc2m[j] >> theCurrentTemp.entby[i].ygx0c2m[j] >> theCurrentTemp.entby[i].ygsigc2m[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 13, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entby[i].xavgc2m[j] >> theCurrentTemp.entby[i].xrmsc2m[j] >> theCurrentTemp.entby[i].xgx0c2m[j] >> theCurrentTemp.entby[i].xgsigc2m[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 14, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			} 
			
			db >> theCurrentTemp.entby[i].yspare[0] >> theCurrentTemp.entby[i].yspare[1] >> theCurrentTemp.entby[i].yspare[2] >> theCurrentTemp.entby[i].yspare[3] >> theCurrentTemp.entby[i].yspare[4]
				 >> theCurrentTemp.entby[i].yspare[5] >> theCurrentTemp.entby[i].yspare[6] >> theCurrentTemp.entby[i].yspare[7] >> theCurrentTemp.entby[i].yspare[8] >> theCurrentTemp.entby[i].yspare[9];
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 15, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			
			db >> theCurrentTemp.entby[i].xspare[0] >> theCurrentTemp.entby[i].xspare[1] >> theCurrentTemp.entby[i].xspare[2] >> theCurrentTemp.entby[i].xspare[3] >> theCurrentTemp.entby[i].xspare[4]
				 >> theCurrentTemp.entby[i].xspare[5] >> theCurrentTemp.entby[i].xspare[6] >> theCurrentTemp.entby[i].xspare[7] >> theCurrentTemp.entby[i].xspare[8] >> theCurrentTemp.entby[i].xspare[9];
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 16, no template load, run # " << theCurrentTemp.entby[i].runnum << ENDL; return false;}
			
		}
	
// next, loop over all barrel x-angle entries   

		for (k=0; k < theCurrentTemp.head.NByx; ++k) { 
			
			for (i=0; i < theCurrentTemp.head.NBxx; ++i) { 
        
				db >> theCurrentTemp.entbx[k][i].runnum >> theCurrentTemp.entbx[k][i].costrk[0] 
					 >> theCurrentTemp.entbx[k][i].costrk[1] >> theCurrentTemp.entbx[k][i].costrk[2]; 
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 17, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
			  
// Calculate the alpha, beta, and cot(beta) for this entry 

				theCurrentTemp.entbx[k][i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entbx[k][i].costrk[2], (double)theCurrentTemp.entbx[k][i].costrk[0]));
				
				theCurrentTemp.entbx[k][i].cotalpha = theCurrentTemp.entbx[k][i].costrk[0]/theCurrentTemp.entbx[k][i].costrk[2];
				
				theCurrentTemp.entbx[k][i].beta = static_cast<float>(atan2((double)theCurrentTemp.entbx[k][i].costrk[2], (double)theCurrentTemp.entbx[k][i].costrk[1]));
				
				theCurrentTemp.entbx[k][i].cotbeta = theCurrentTemp.entbx[k][i].costrk[1]/theCurrentTemp.entbx[k][i].costrk[2];
				
				db >> theCurrentTemp.entbx[k][i].qavg >> theCurrentTemp.entbx[k][i].pixmax >> theCurrentTemp.entbx[k][i].symax >> theCurrentTemp.entbx[k][i].dyone
					 >> theCurrentTemp.entbx[k][i].syone >> theCurrentTemp.entbx[k][i].sxmax >> theCurrentTemp.entbx[k][i].dxone >> theCurrentTemp.entbx[k][i].sxone;
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 18, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				
				db >> theCurrentTemp.entbx[k][i].dytwo >> theCurrentTemp.entbx[k][i].sytwo >> theCurrentTemp.entbx[k][i].dxtwo 
					 >> theCurrentTemp.entbx[k][i].sxtwo >> theCurrentTemp.entbx[k][i].qmin >> theCurrentTemp.entbx[k][i].clsleny >> theCurrentTemp.entbx[k][i].clslenx;
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 19, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
			  
				for (j=0; j<2; ++j) {
					
					db >> theCurrentTemp.entbx[k][i].ypar[j][0] >> theCurrentTemp.entbx[k][i].ypar[j][1] 
						 >> theCurrentTemp.entbx[k][i].ypar[j][2] >> theCurrentTemp.entbx[k][i].ypar[j][3] >> theCurrentTemp.entbx[k][i].ypar[j][4];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 20, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				}
			  
				for (j=0; j<9; ++j) {
					
					for (l=0; l<TYSIZE; ++l) {db >> theCurrentTemp.entbx[k][i].ytemp[j][l];} 
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 21, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<2; ++j) {
					
					db >> theCurrentTemp.entbx[k][i].xpar[j][0] >> theCurrentTemp.entbx[k][i].xpar[j][1] 
						 >> theCurrentTemp.entbx[k][i].xpar[j][2] >> theCurrentTemp.entbx[k][i].xpar[j][3] >> theCurrentTemp.entbx[k][i].xpar[j][4];
					
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 22, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<9; ++j) {
					
					for (l=0; l<TXSIZE; ++l) {db >> theCurrentTemp.entbx[k][i].xtemp[j][l];} 
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 23, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entbx[k][i].yavg[j] >> theCurrentTemp.entbx[k][i].yrms[j] >> theCurrentTemp.entbx[k][i].ygx0[j] >> theCurrentTemp.entbx[k][i].ygsig[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 24, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entbx[k][i].yflpar[j][0] >> theCurrentTemp.entbx[k][i].yflpar[j][1] >> theCurrentTemp.entbx[k][i].yflpar[j][2] 
						 >> theCurrentTemp.entbx[k][i].yflpar[j][3] >> theCurrentTemp.entbx[k][i].yflpar[j][4] >> theCurrentTemp.entbx[k][i].yflpar[j][5];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 25, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entbx[k][i].xavg[j] >> theCurrentTemp.entbx[k][i].xrms[j] >> theCurrentTemp.entbx[k][i].xgx0[j] >> theCurrentTemp.entbx[k][i].xgsig[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 26, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				}
			  
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entbx[k][i].xflpar[j][0] >> theCurrentTemp.entbx[k][i].xflpar[j][1] >> theCurrentTemp.entbx[k][i].xflpar[j][2] 
						 >> theCurrentTemp.entbx[k][i].xflpar[j][3] >> theCurrentTemp.entbx[k][i].xflpar[j][4] >> theCurrentTemp.entbx[k][i].xflpar[j][5];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 27, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				}
			  
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entbx[k][i].chi2yavg[j] >> theCurrentTemp.entbx[k][i].chi2ymin[j] >> theCurrentTemp.entbx[k][i].chi2xavg[j] >> theCurrentTemp.entbx[k][i].chi2xmin[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 28, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entbx[k][i].yavgc2m[j] >> theCurrentTemp.entbx[k][i].yrmsc2m[j] >> theCurrentTemp.entbx[k][i].ygx0c2m[j] >> theCurrentTemp.entbx[k][i].ygsigc2m[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 29, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entbx[k][i].xavgc2m[j] >> theCurrentTemp.entbx[k][i].xrmsc2m[j] >> theCurrentTemp.entbx[k][i].xgx0c2m[j] >> theCurrentTemp.entbx[k][i].xgsigc2m[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 30, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				}
				
				db >> theCurrentTemp.entbx[k][i].yspare[0] >> theCurrentTemp.entbx[k][i].yspare[1] >> theCurrentTemp.entbx[k][i].yspare[2] >> theCurrentTemp.entbx[k][i].yspare[3] >> theCurrentTemp.entbx[k][i].yspare[4]
					 >> theCurrentTemp.entbx[k][i].yspare[5] >> theCurrentTemp.entbx[k][i].yspare[6] >> theCurrentTemp.entbx[k][i].yspare[7] >> theCurrentTemp.entbx[k][i].yspare[8] >> theCurrentTemp.entbx[k][i].yspare[9];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 31, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				
				db >> theCurrentTemp.entbx[k][i].xspare[0] >> theCurrentTemp.entbx[k][i].xspare[1] >> theCurrentTemp.entbx[k][i].xspare[2] >> theCurrentTemp.entbx[k][i].xspare[3] >> theCurrentTemp.entbx[k][i].xspare[4]
					 >> theCurrentTemp.entbx[k][i].xspare[5] >> theCurrentTemp.entbx[k][i].xspare[6] >> theCurrentTemp.entbx[k][i].xspare[7] >> theCurrentTemp.entbx[k][i].xspare[8] >> theCurrentTemp.entbx[k][i].xspare[9];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 32, no template load, run # " << theCurrentTemp.entbx[k][i].runnum << ENDL; return false;}
				
			}
		}	
    
// next, loop over all forward y-angle entries   
	
		for (i=0; i < theCurrentTemp.head.NFy; ++i) {     
			
			db >> theCurrentTemp.entfy[i].runnum >> theCurrentTemp.entfy[i].costrk[0] 
				 >> theCurrentTemp.entfy[i].costrk[1] >> theCurrentTemp.entfy[i].costrk[2]; 
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 33, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			
// Calculate the alpha, beta, and cot(beta) for this entry 
			
			theCurrentTemp.entfy[i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entfy[i].costrk[2], (double)theCurrentTemp.entfy[i].costrk[0]));
			
			theCurrentTemp.entfy[i].cotalpha = theCurrentTemp.entfy[i].costrk[0]/theCurrentTemp.entfy[i].costrk[2];
			
			theCurrentTemp.entfy[i].beta = static_cast<float>(atan2((double)theCurrentTemp.entfy[i].costrk[2], (double)theCurrentTemp.entfy[i].costrk[1]));
			
			theCurrentTemp.entfy[i].cotbeta = theCurrentTemp.entfy[i].costrk[1]/theCurrentTemp.entfy[i].costrk[2];
			
			db >> theCurrentTemp.entfy[i].qavg >> theCurrentTemp.entfy[i].pixmax >> theCurrentTemp.entfy[i].symax >> theCurrentTemp.entfy[i].dyone
				 >> theCurrentTemp.entfy[i].syone >> theCurrentTemp.entfy[i].sxmax >> theCurrentTemp.entfy[i].dxone >> theCurrentTemp.entfy[i].sxone;
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 34, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			
			db >> theCurrentTemp.entfy[i].dytwo >> theCurrentTemp.entfy[i].sytwo >> theCurrentTemp.entfy[i].dxtwo 
				 >> theCurrentTemp.entfy[i].sxtwo >> theCurrentTemp.entfy[i].qmin >> theCurrentTemp.entfy[i].clsleny >> theCurrentTemp.entfy[i].clslenx;
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 35, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			
			for (j=0; j<2; ++j) {
				
				db >> theCurrentTemp.entfy[i].ypar[j][0] >> theCurrentTemp.entfy[i].ypar[j][1] 
					 >> theCurrentTemp.entfy[i].ypar[j][2] >> theCurrentTemp.entfy[i].ypar[j][3] >> theCurrentTemp.entfy[i].ypar[j][4];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 36, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			  
			}
			
			for (j=0; j<9; ++j) {
				
				for (l=0; l<TYSIZE; ++l) {db >> theCurrentTemp.entfy[i].ytemp[j][l];} 
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 37, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<2; ++j) {
				
				db >> theCurrentTemp.entfy[i].xpar[j][0] >> theCurrentTemp.entfy[i].xpar[j][1] 
					 >> theCurrentTemp.entfy[i].xpar[j][2] >> theCurrentTemp.entfy[i].xpar[j][3] >> theCurrentTemp.entfy[i].xpar[j][4];
			  
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 38, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<9; ++j) {
				
				for (l=0; l<TXSIZE; ++l) {db >> theCurrentTemp.entfy[i].xtemp[j][l];} 
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 39, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entfy[i].yavg[j] >> theCurrentTemp.entfy[i].yrms[j] >> theCurrentTemp.entfy[i].ygx0[j] >> theCurrentTemp.entfy[i].ygsig[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 40, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entfy[i].yflpar[j][0] >> theCurrentTemp.entfy[i].yflpar[j][1] >> theCurrentTemp.entfy[i].yflpar[j][2]
					 >> theCurrentTemp.entfy[i].yflpar[j][3] >> theCurrentTemp.entfy[i].yflpar[j][4] >> theCurrentTemp.entfy[i].yflpar[j][5];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 41, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entfy[i].xavg[j] >> theCurrentTemp.entfy[i].xrms[j] >> theCurrentTemp.entfy[i].xgx0[j] >> theCurrentTemp.entfy[i].xgsig[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 42, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entfy[i].xflpar[j][0] >> theCurrentTemp.entfy[i].xflpar[j][1] >> theCurrentTemp.entfy[i].xflpar[j][2] 
					 >> theCurrentTemp.entfy[i].xflpar[j][3] >> theCurrentTemp.entfy[i].xflpar[j][4] >> theCurrentTemp.entfy[i].xflpar[j][5];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 43, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entfy[i].chi2yavg[j] >> theCurrentTemp.entfy[i].chi2ymin[j] >> theCurrentTemp.entfy[i].chi2xavg[j] >> theCurrentTemp.entfy[i].chi2xmin[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 44, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entfy[i].yavgc2m[j] >> theCurrentTemp.entfy[i].yrmsc2m[j] >> theCurrentTemp.entfy[i].ygx0c2m[j] >> theCurrentTemp.entfy[i].ygsigc2m[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 45, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.entfy[i].xavgc2m[j] >> theCurrentTemp.entfy[i].xrmsc2m[j] >> theCurrentTemp.entfy[i].xgx0c2m[j] >> theCurrentTemp.entfy[i].xgsigc2m[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 46, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			}
			
			db >> theCurrentTemp.entfy[i].yspare[0] >> theCurrentTemp.entfy[i].yspare[1] >> theCurrentTemp.entfy[i].yspare[2] >> theCurrentTemp.entfy[i].yspare[3] >> theCurrentTemp.entfy[i].yspare[4]
				 >> theCurrentTemp.entfy[i].yspare[5] >> theCurrentTemp.entfy[i].yspare[6] >> theCurrentTemp.entfy[i].yspare[7] >> theCurrentTemp.entfy[i].yspare[8] >> theCurrentTemp.entfy[i].yspare[9];
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 47, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			
			db >> theCurrentTemp.entfy[i].xspare[0] >> theCurrentTemp.entfy[i].xspare[1] >> theCurrentTemp.entfy[i].xspare[2] >> theCurrentTemp.entfy[i].xspare[3] >> theCurrentTemp.entfy[i].xspare[4]
				 >> theCurrentTemp.entfy[i].xspare[5] >> theCurrentTemp.entfy[i].xspare[6] >> theCurrentTemp.entfy[i].xspare[7] >> theCurrentTemp.entfy[i].xspare[8] >> theCurrentTemp.entfy[i].xspare[9];
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 48, no template load, run # " << theCurrentTemp.entfy[i].runnum << ENDL; return false;}
			
		}
		
// next, loop over all forward x-angle entries   

		for (k=0; k < theCurrentTemp.head.NFyx; ++k) { 
			
			for (i=0; i < theCurrentTemp.head.NFxx; ++i) {     
				
				db >> theCurrentTemp.entfx[k][i].runnum >> theCurrentTemp.entfx[k][i].costrk[0] 
					 >> theCurrentTemp.entfx[k][i].costrk[1] >> theCurrentTemp.entfx[k][i].costrk[2]; 
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 49, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				
// Calculate the alpha, beta, and cot(beta) for this entry 

				theCurrentTemp.entfx[k][i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entfx[k][i].costrk[2], (double)theCurrentTemp.entfx[k][i].costrk[0]));
				
				theCurrentTemp.entfx[k][i].cotalpha = theCurrentTemp.entfx[k][i].costrk[0]/theCurrentTemp.entfx[k][i].costrk[2];
				
				theCurrentTemp.entfx[k][i].beta = static_cast<float>(atan2((double)theCurrentTemp.entfx[k][i].costrk[2], (double)theCurrentTemp.entfx[k][i].costrk[1]));
				
				theCurrentTemp.entfx[k][i].cotbeta = theCurrentTemp.entfx[k][i].costrk[1]/theCurrentTemp.entfx[k][i].costrk[2];
				
				db >> theCurrentTemp.entfx[k][i].qavg >> theCurrentTemp.entfx[k][i].pixmax >> theCurrentTemp.entfx[k][i].symax >> theCurrentTemp.entfx[k][i].dyone
					 >> theCurrentTemp.entfx[k][i].syone >> theCurrentTemp.entfx[k][i].sxmax >> theCurrentTemp.entfx[k][i].dxone >> theCurrentTemp.entfx[k][i].sxone;
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 50, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				
				db >> theCurrentTemp.entfx[k][i].dytwo >> theCurrentTemp.entfx[k][i].sytwo >> theCurrentTemp.entfx[k][i].dxtwo 
					 >> theCurrentTemp.entfx[k][i].sxtwo >> theCurrentTemp.entfx[k][i].qmin >> theCurrentTemp.entfx[k][i].clsleny >> theCurrentTemp.entfx[k][i].clslenx;
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 51, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
			  
				for (j=0; j<2; ++j) {
					
					db >> theCurrentTemp.entfx[k][i].ypar[j][0] >> theCurrentTemp.entfx[k][i].ypar[j][1] 
						 >> theCurrentTemp.entfx[k][i].ypar[j][2] >> theCurrentTemp.entfx[k][i].ypar[j][3] >> theCurrentTemp.entfx[k][i].ypar[j][4];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 52, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
					
				}
			  
				for (j=0; j<9; ++j) {
					
					for (l=0; l<TYSIZE; ++l) {db >> theCurrentTemp.entfx[k][i].ytemp[j][l];} 
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 53, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<2; ++j) {
					
					db >> theCurrentTemp.entfx[k][i].xpar[j][0] >> theCurrentTemp.entfx[k][i].xpar[j][1] 
						 >> theCurrentTemp.entfx[k][i].xpar[j][2] >> theCurrentTemp.entfx[k][i].xpar[j][3] >> theCurrentTemp.entfx[k][i].xpar[j][4];
					
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 54, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				}
			  
				for (j=0; j<9; ++j) {
					
					for (l=0; l<TXSIZE; ++l) {db >> theCurrentTemp.entfx[k][i].xtemp[j][l];} 
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 55, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entfx[k][i].yavg[j] >> theCurrentTemp.entfx[k][i].yrms[j] >> theCurrentTemp.entfx[k][i].ygx0[j] >> theCurrentTemp.entfx[k][i].ygsig[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 56, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entfx[k][i].yflpar[j][0] >> theCurrentTemp.entfx[k][i].yflpar[j][1] >> theCurrentTemp.entfx[k][i].yflpar[j][2] 
						 >> theCurrentTemp.entfx[k][i].yflpar[j][3] >> theCurrentTemp.entfx[k][i].yflpar[j][4] >> theCurrentTemp.entfx[k][i].yflpar[j][5];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 57, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entfx[k][i].xavg[j] >> theCurrentTemp.entfx[k][i].xrms[j] >> theCurrentTemp.entfx[k][i].xgx0[j] >> theCurrentTemp.entfx[k][i].xgsig[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 58, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				}
			  
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entfx[k][i].xflpar[j][0] >> theCurrentTemp.entfx[k][i].xflpar[j][1] >> theCurrentTemp.entfx[k][i].xflpar[j][2] 
						 >> theCurrentTemp.entfx[k][i].xflpar[j][3] >> theCurrentTemp.entfx[k][i].xflpar[j][4] >> theCurrentTemp.entfx[k][i].xflpar[j][5];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 59, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				}
			  
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entfx[k][i].chi2yavg[j] >> theCurrentTemp.entfx[k][i].chi2ymin[j] >> theCurrentTemp.entfx[k][i].chi2xavg[j] >> theCurrentTemp.entfx[k][i].chi2xmin[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 60, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				}
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entfx[k][i].yavgc2m[j] >> theCurrentTemp.entfx[k][i].yrmsc2m[j] >> theCurrentTemp.entfx[k][i].ygx0c2m[j] >> theCurrentTemp.entfx[k][i].ygsigc2m[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 61, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entfx[k][i].xavgc2m[j] >> theCurrentTemp.entfx[k][i].xrmsc2m[j] >> theCurrentTemp.entfx[k][i].xgx0c2m[j] >> theCurrentTemp.entfx[k][i].xgsigc2m[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 62, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				}
				
				db >> theCurrentTemp.entfx[k][i].yspare[0] >> theCurrentTemp.entfx[k][i].yspare[1] >> theCurrentTemp.entfx[k][i].yspare[2] >> theCurrentTemp.entfx[k][i].yspare[3] >> theCurrentTemp.entfx[k][i].yspare[4]
					 >> theCurrentTemp.entfx[k][i].yspare[5] >> theCurrentTemp.entfx[k][i].yspare[6] >> theCurrentTemp.entfx[k][i].yspare[7] >> theCurrentTemp.entfx[k][i].yspare[8] >> theCurrentTemp.entfx[k][i].yspare[9];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 63, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				
				db >> theCurrentTemp.entfx[k][i].xspare[0] >> theCurrentTemp.entfx[k][i].xspare[1] >> theCurrentTemp.entfx[k][i].xspare[2] >> theCurrentTemp.entfx[k][i].xspare[3] >> theCurrentTemp.entfx[k][i].xspare[4]
					 >> theCurrentTemp.entfx[k][i].xspare[5] >> theCurrentTemp.entfx[k][i].xspare[6] >> theCurrentTemp.entfx[k][i].xspare[7] >> theCurrentTemp.entfx[k][i].xspare[8] >> theCurrentTemp.entfx[k][i].xspare[9];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 64, no template load, run # " << theCurrentTemp.entfx[k][i].runnum << ENDL; return false;}
				
			}	
		}
    
// Add this template to the store
	
		thePixelTemp.push_back(theCurrentTemp);

	}
	return true;

} // TempInit 











// ************************************************************************************************************ 
//! Interpolate input alpha and beta angles to produce a working template for each individual hit. 
//! \param id - (input) index of the template to use
//! \param fpix - (input) logical input indicating whether to use FPix templates (true) 
//!               or Barrel templates (false)
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)	if(filenum == 
// ************************************************************************************************************ 
bool SiPixelTemplate::interpolate(int id, bool fpix, float cotalpha, float cotbeta)
{
    // Interpolate for a new set of track angles 
    
    // Local variables 
    int i, j, ind;
	int ilow, ihigh, iylow, iyhigh, Ny, Nxx, Nyx, imidy, imaxx;
	float yratio, yxratio, xxratio, sxmax, qcorrect, symax;
	bool success;
//	std::vector <float> xrms(4), xgsig(4), xrmsc2m(4), xgsigc2m(4);
	std::vector <float> chi2xavg(4), chi2xmin(4);


// Check to see if interpolation is valid     

if(id != id_current || fpix != fpix_current || cotalpha != cota_current || cotbeta != cotb_current) {

	fpix_current = fpix; cota_current = cotalpha; cotb_current = cotbeta;
	
	if(id != id_current) {

// Find the index corresponding to id

       index_id = -1;
       for(i=0; i<thePixelTemp.size(); ++i) {
	
	      if(id == thePixelTemp[i].head.ID) {
	   
	         index_id = i;
		     id_current = id;
		     break;
          }
	    }
     }
	 
	 assert(index_id >= 0 && index_id < thePixelTemp.size());
	 
// Interpolate the absolute value of cot(beta)     
    
    abs_cotb = fabs((double)cotbeta);
	
//	qcorrect corrects the cot(alpha)=0 cluster charge for non-zero cot(alpha)	

    qcorrect=(float)sqrt((double)((1.+cotbeta*cotbeta+cotalpha*cotalpha)/(1.+cotbeta*cotbeta)));

// Copy the charge scaling factor to the private variable     
    
    pqscale = thePixelTemp[index_id].head.qscale;

// Copy the pseudopixel signal size to the private variable     
    
    ps50 = thePixelTemp[index_id].head.s50;

// success flags whether or not the track angles are inside the interpolation range
    
    success = true;
	
// Decide which template (FPix or BPix) to use 

    if(fpix) {
    
// Begin FPix section, make the index counters easier to use     
    
       Ny = thePixelTemp[index_id].head.NFy;
       Nyx = thePixelTemp[index_id].head.NFyx;
       Nxx = thePixelTemp[index_id].head.NFxx;
	   imaxx = Nyx - 1;
	   imidy = Nxx/2;
        
// next, loop over all y-angle entries   

	   ilow = 0;
	   yratio = 0.;

	   if(abs_cotb >= thePixelTemp[index_id].entfy[Ny-1].cotbeta) {
	
	       ilow = Ny-2;
		   yratio = 1.;
		   success = false;
		
	   } else {
	   
	      if(abs_cotb >= thePixelTemp[index_id].entfy[0].cotbeta) {

             for (i=0; i<Ny-1; ++i) { 
    
                if( thePixelTemp[index_id].entfy[i].cotbeta <= abs_cotb && abs_cotb < thePixelTemp[index_id].entfy[i+1].cotbeta) {
		  
	               ilow = i;
		           yratio = (abs_cotb - thePixelTemp[index_id].entfy[i].cotbeta)/(thePixelTemp[index_id].entfy[i+1].cotbeta - thePixelTemp[index_id].entfy[i].cotbeta);
		           break;			 
		        }
	         }
		  } else { success = false; }
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all y-related quantities (flip displacements when cotbeta < 0)

       pyratio = yratio;
	   pqavg = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].qavg + yratio*thePixelTemp[index_id].entfy[ihigh].qavg;
	   pqavg *= qcorrect;
	   symax = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].symax + yratio*thePixelTemp[index_id].entfy[ihigh].symax;
	   psyparmax = symax;
	   sxmax = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].sxmax + yratio*thePixelTemp[index_id].entfy[ihigh].sxmax;
	   pdyone = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].dyone + yratio*thePixelTemp[index_id].entfy[ihigh].dyone;
	   if(cotbeta < 0.) {pdyone = -pdyone;}
	   psyone = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].syone + yratio*thePixelTemp[index_id].entfy[ihigh].syone;
	   pdytwo = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].dytwo + yratio*thePixelTemp[index_id].entfy[ihigh].dytwo;
	   if(cotbeta < 0.) {pdytwo = -pdytwo;}
	   psytwo = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].sytwo + yratio*thePixelTemp[index_id].entfy[ihigh].sytwo;
	   pqmin = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].qmin + yratio*thePixelTemp[index_id].entfy[ihigh].qmin;
	   pclsleny = fminf(thePixelTemp[index_id].entfy[ilow].clsleny, thePixelTemp[index_id].entfy[ihigh].clsleny);
	   for(i=0; i<2 ; ++i) {
	      for(j=0; j<5 ; ++j) {
// Charge loss switches sides when cot(beta) changes sign
		     if(cotbeta < 0) {
	            pyparl[1-i][j] = thePixelTemp[index_id].entfy[ilow].ypar[i][j];
	            pyparh[1-i][j] = thePixelTemp[index_id].entfy[ihigh].ypar[i][j];
			 } else {
	            pyparl[i][j] = thePixelTemp[index_id].entfy[ilow].ypar[i][j];
	            pyparh[i][j] = thePixelTemp[index_id].entfy[ihigh].ypar[i][j];
			 }
	         pxparly0[i][j] = thePixelTemp[index_id].entfy[ilow].xpar[i][j];
	         pxparhy0[i][j] = thePixelTemp[index_id].entfy[ihigh].xpar[i][j];
	      }
	   }
	   for(i=0; i<4; ++i) {
	      pyavg[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yavg[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yavg[i];
	      if(cotbeta < 0.) {pyavg[i] = -pyavg[i];}
	      pyrms[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yrms[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yrms[i];
//	      pygx0[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ygx0[i] + yratio*thePixelTemp[index_id].entfy[ihigh].ygx0[i];
//	      if(cotbeta < 0.) {pygx0[i] = -pygx0[i];}
//	      pygsig[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ygsig[i] + yratio*thePixelTemp[index_id].entfy[ihigh].ygsig[i];
//	      xrms[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].xrms[i] + yratio*thePixelTemp[index_id].entfy[ihigh].xrms[i];
//	      xgsig[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].xgsig[i] + yratio*thePixelTemp[index_id].entfy[ihigh].xgsig[i];
	      pchi2yavg[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].chi2yavg[i] + yratio*thePixelTemp[index_id].entfy[ihigh].chi2yavg[i];
	      pchi2ymin[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].chi2ymin[i] + yratio*thePixelTemp[index_id].entfy[ihigh].chi2ymin[i];
	      chi2xavg[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].chi2xavg[i] + yratio*thePixelTemp[index_id].entfy[ihigh].chi2xavg[i];
	      chi2xmin[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].chi2xmin[i] + yratio*thePixelTemp[index_id].entfy[ihigh].chi2xmin[i];
	      pyavgc2m[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yavgc2m[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yavgc2m[i];
	      if(cotbeta < 0.) {pyavgc2m[i] = -pyavgc2m[i];}
	      pyrmsc2m[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yrmsc2m[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yrmsc2m[i];
//	      pygx0c2m[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ygx0c2m[i] + yratio*thePixelTemp[index_id].entfy[ihigh].ygx0c2m[i];
//	      if(cotbeta < 0.) {pygx0c2m[i] = -pygx0c2m[i];}
//	      pygsigc2m[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ygsigc2m[i] + yratio*thePixelTemp[index_id].entfy[ihigh].ygsigc2m[i];
//	      xrmsc2m[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].xrmsc2m[i] + yratio*thePixelTemp[index_id].entfy[ihigh].xrmsc2m[i];
//	      xgsigc2m[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].xgsigc2m[i] + yratio*thePixelTemp[index_id].entfy[ihigh].xgsigc2m[i];
		  for(j=0; j<6 ; ++j) {
			 pyflparl[i][j] = thePixelTemp[index_id].entfy[ilow].yflpar[i][j];
			 pyflparh[i][j] = thePixelTemp[index_id].entfy[ihigh].yflpar[i][j];
			 
// Since Q_fl is odd under cotbeta, it flips qutomatically, change only even terms

			 if(cotbeta < 0. && (j == 0 || j == 2 || j == 4)) {
			    pyflparl[i][j] = - pyflparl[i][j];
			    pyflparh[i][j] = - pyflparh[i][j];
			 }
		  }
	   }
	   
//// Do the spares next

//       for(i=0; i<10; ++i) {
//		    pyspare[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yspare[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yspare[i];
//       }
			  
// Interpolate and build the y-template 
	
	   for(i=0; i<9; ++i) {
          pytemp[i][0] = 0.;
          pytemp[i][1] = 0.;
	      pytemp[i][BYM2] = 0.;
	      pytemp[i][BYM1] = 0.;
	      for(j=0; j<TYSIZE; ++j) {
		  
// Flip the basic y-template when the cotbeta is negative

		     if(cotbeta < 0.) {
	            pytemp[8-i][BYM3-j]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ytemp[i][j] + yratio*thePixelTemp[index_id].entfy[ihigh].ytemp[i][j];
			 } else {
	            pytemp[i][j+2]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ytemp[i][j] + yratio*thePixelTemp[index_id].entfy[ihigh].ytemp[i][j];
			 }
	      }
	   }
	
// next, loop over all x-angle entries, first, find relevant y-slices   
	
	   iylow = 0;
	   yxratio = 0.;

	   if(abs_cotb >= thePixelTemp[index_id].entfx[Nyx-1][0].cotbeta) {
	
	       iylow = Nyx-2;
		   yxratio = 1.;
		
	   } else if(abs_cotb >= thePixelTemp[index_id].entfx[0][0].cotbeta) {

          for (i=0; i<Nyx-1; ++i) { 
    
             if( thePixelTemp[index_id].entfx[i][0].cotbeta <= abs_cotb && abs_cotb < thePixelTemp[index_id].entfx[i+1][0].cotbeta) {
		  
	            iylow = i;
		        yxratio = (abs_cotb - thePixelTemp[index_id].entfx[i][0].cotbeta)/(thePixelTemp[index_id].entfx[i+1][0].cotbeta - thePixelTemp[index_id].entfx[i][0].cotbeta);
		        break;			 
		     }
	      }
	   }
	
	   iyhigh=iylow + 1;

	   ilow = 0;
	   xxratio = 0.;

	   if(cotalpha >= thePixelTemp[index_id].entfx[0][Nxx-1].cotalpha) {
	
	       ilow = Nxx-2;
		   xxratio = 1.;
		   success = false;
		
	   } else {
	   
	      if(cotalpha >= thePixelTemp[index_id].entfx[0][0].cotalpha) {

             for (i=0; i<Nxx-1; ++i) { 
    
                if( thePixelTemp[index_id].entfx[0][i].cotalpha <= cotalpha && cotalpha < thePixelTemp[index_id].entfx[0][i+1].cotalpha) {
		  
	               ilow = i;
		           xxratio = (cotalpha - thePixelTemp[index_id].entfx[0][i].cotalpha)/(thePixelTemp[index_id].entfx[0][i+1].cotalpha - thePixelTemp[index_id].entfx[0][i].cotalpha);
		           break;
			    }
		     }
		  } else { success = false; }
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all x-related quantities 

       pyxratio = yxratio;
       pxxratio = xxratio;		
	   		  
// sxparmax defines the maximum charge for which the parameters xpar are defined (not rescaled by cotbeta) 

	   psxparmax = (1. - xxratio)*thePixelTemp[index_id].entfx[imaxx][ilow].sxmax + xxratio*thePixelTemp[index_id].entfx[imaxx][ihigh].sxmax;
	   psxmax = psxparmax;
       if(thePixelTemp[index_id].entfx[imaxx][imidy].sxmax != 0.) {psxmax=psxmax/thePixelTemp[index_id].entfx[imaxx][imidy].sxmax*sxmax;}
	   psymax = (1. - xxratio)*thePixelTemp[index_id].entfx[imaxx][ilow].symax + xxratio*thePixelTemp[index_id].entfx[imaxx][ihigh].symax;
       if(thePixelTemp[index_id].entfx[imaxx][imidy].symax != 0.) {psymax=psymax/thePixelTemp[index_id].entfx[imaxx][imidy].symax*symax;}
	   pdxone = (1. - xxratio)*thePixelTemp[index_id].entfx[0][ilow].dxone + xxratio*thePixelTemp[index_id].entfx[0][ihigh].dxone;
	   psxone = (1. - xxratio)*thePixelTemp[index_id].entfx[0][ilow].sxone + xxratio*thePixelTemp[index_id].entfx[0][ihigh].sxone;
	   pdxtwo = (1. - xxratio)*thePixelTemp[index_id].entfx[0][ilow].dxtwo + xxratio*thePixelTemp[index_id].entfx[0][ihigh].dxtwo;
	   psxtwo = (1. - xxratio)*thePixelTemp[index_id].entfx[0][ilow].sxtwo + xxratio*thePixelTemp[index_id].entfx[0][ihigh].sxtwo;
	   pclslenx = fminf(thePixelTemp[index_id].entfx[0][ilow].clslenx, thePixelTemp[index_id].entfx[0][ihigh].clslenx);
	   for(i=0; i<2 ; ++i) {
	      for(j=0; j<5 ; ++j) {
	         pxpar0[i][j] = thePixelTemp[index_id].entfx[imaxx][imidy].xpar[i][j];
	         pxparl[i][j] = thePixelTemp[index_id].entfx[imaxx][ilow].xpar[i][j];
	         pxparh[i][j] = thePixelTemp[index_id].entfx[imaxx][ihigh].xpar[i][j];
	      }
	   }
	   		  
// pixmax is the maximum allowed pixel charge (used for truncation)

	   ppixmax=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].pixmax + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].pixmax)
			  +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].pixmax + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].pixmax);
			  
	   for(i=0; i<4; ++i) {
	      pxavg[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xavg[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xavg[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xavg[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xavg[i]);
		  
	      pxrms[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xrms[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xrms[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xrms[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xrms[i]);
		  
//	      pxgx0[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xgx0[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xgx0[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xgx0[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xgx0[i]);
							
//	      pxgsig[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xgsig[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xgsig[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xgsig[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xgsig[i]);
				  
	      pxavgc2m[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xavgc2m[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xavgc2m[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xavgc2m[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xavgc2m[i]);
		  
	      pxrmsc2m[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xrmsc2m[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xrmsc2m[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xrmsc2m[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xrmsc2m[i]);
		  
//	      pxgx0c2m[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xgx0c2m[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xgx0c2m[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xgx0c2m[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xgx0c2m[i]);
							
//	      pxgsigc2m[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xgsigc2m[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xgsigc2m[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xgsigc2m[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xgsigc2m[i]);
//
//  Try new interpolation scheme
//	  														
//	      pchi2xavg[i]=((1. - xxratio)*thePixelTemp[index_id].entfx[imaxx][ilow].chi2xavg[i] + xxratio*thePixelTemp[index_id].entfx[imaxx][ihigh].chi2xavg[i]);
//		  if(thePixelTemp[index_id].entfx[imaxx][imidy].chi2xavg[i] != 0.) {pchi2xavg[i]=pchi2xavg[i]/thePixelTemp[index_id].entfx[imaxx][imidy].chi2xavg[i]*chi2xavg[i];}
//							
//	      pchi2xmin[i]=((1. - xxratio)*thePixelTemp[index_id].entfx[imaxx][ilow].chi2xmin[i] + xxratio*thePixelTemp[index_id].entfx[imaxx][ihigh].chi2xmin[i]);
//		  if(thePixelTemp[index_id].entfx[imaxx][imidy].chi2xmin[i] != 0.) {pchi2xmin[i]=pchi2xmin[i]/thePixelTemp[index_id].entfx[imaxx][imidy].chi2xmin[i]*chi2xmin[i];}
//		  
	      pchi2xavg[i]=((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].chi2xavg[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].chi2xavg[i]);
		  if(thePixelTemp[index_id].entfx[iyhigh][imidy].chi2xavg[i] != 0.) {pchi2xavg[i]=pchi2xavg[i]/thePixelTemp[index_id].entfx[iyhigh][imidy].chi2xavg[i]*chi2xavg[i];}
							
	      pchi2xmin[i]=((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].chi2xmin[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].chi2xmin[i]);
		  if(thePixelTemp[index_id].entfx[iyhigh][imidy].chi2xmin[i] != 0.) {pchi2xmin[i]=pchi2xmin[i]/thePixelTemp[index_id].entfx[iyhigh][imidy].chi2xmin[i]*chi2xmin[i];}
		  
	      for(j=0; j<6 ; ++j) {
	         pxflparll[i][j] = thePixelTemp[index_id].entfx[iylow][ilow].xflpar[i][j];
	         pxflparlh[i][j] = thePixelTemp[index_id].entfx[iylow][ihigh].xflpar[i][j];
	         pxflparhl[i][j] = thePixelTemp[index_id].entfx[iyhigh][ilow].xflpar[i][j];
	         pxflparhh[i][j] = thePixelTemp[index_id].entfx[iyhigh][ihigh].xflpar[i][j];
		  }
	   }
	   
// Do the spares next

//       for(i=0; i<10; ++i) {
//	      pxspare[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xspare[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xspare[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xspare[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xspare[i]);
//       }
			  
// Interpolate and build the x-template 
	
	   for(i=0; i<9; ++i) {
          pxtemp[i][0] = 0.;
          pxtemp[i][1] = 0.;
	      pxtemp[i][BXM2] = 0.;
	      pxtemp[i][BXM1] = 0.;
	      for(j=0; j<TXSIZE; ++j) {
	        pxtemp[i][j+2]=(1. - xxratio)*thePixelTemp[index_id].entfx[imaxx][ilow].xtemp[i][j] + xxratio*thePixelTemp[index_id].entfx[imaxx][ihigh].xtemp[i][j];
	      }
	   }

	} else {
	
    
// Begin BPix section, make the index counters easier to use     
    
       Ny = thePixelTemp[index_id].head.NBy;
       Nyx = thePixelTemp[index_id].head.NByx;
       Nxx = thePixelTemp[index_id].head.NBxx;
	   imaxx = Nyx - 1;
	   imidy = Nxx/2;
        
// next, loop over all y-angle entries   

	   ilow = 0;
	   yratio = 0.;

	   if(abs_cotb >= thePixelTemp[index_id].entby[Ny-1].cotbeta) {
	
	       ilow = Ny-2;
		   yratio = 1.;
		   success = false;
		
	   } else {
	   
	      if(abs_cotb >= thePixelTemp[index_id].entby[0].cotbeta) {

             for (i=0; i<Ny-1; ++i) { 
    
                if( thePixelTemp[index_id].entby[i].cotbeta <= abs_cotb && abs_cotb < thePixelTemp[index_id].entby[i+1].cotbeta) {
		  
	               ilow = i;
		           yratio = (abs_cotb - thePixelTemp[index_id].entby[i].cotbeta)/(thePixelTemp[index_id].entby[i+1].cotbeta - thePixelTemp[index_id].entby[i].cotbeta);
		           break;			 
		        }
	         }
		  } else { success = false;}
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all y-related quantities (flip displacements when cotbeta < 0)

       pyratio = yratio;
	   pqavg = (1. - yratio)*thePixelTemp[index_id].entby[ilow].qavg + yratio*thePixelTemp[index_id].entby[ihigh].qavg;
	   pqavg *= qcorrect;
	   symax = (1. - yratio)*thePixelTemp[index_id].entby[ilow].symax + yratio*thePixelTemp[index_id].entby[ihigh].symax;
	   psyparmax = symax;
	   sxmax = (1. - yratio)*thePixelTemp[index_id].entby[ilow].sxmax + yratio*thePixelTemp[index_id].entby[ihigh].sxmax;
	   pdyone = (1. - yratio)*thePixelTemp[index_id].entby[ilow].dyone + yratio*thePixelTemp[index_id].entby[ihigh].dyone;
	   if(cotbeta < 0.) {pdyone = -pdyone;}
	   psyone = (1. - yratio)*thePixelTemp[index_id].entby[ilow].syone + yratio*thePixelTemp[index_id].entby[ihigh].syone;
	   pdytwo = (1. - yratio)*thePixelTemp[index_id].entby[ilow].dytwo + yratio*thePixelTemp[index_id].entby[ihigh].dytwo;
	   if(cotbeta < 0.) {pdytwo = -pdytwo;}
	   psytwo = (1. - yratio)*thePixelTemp[index_id].entby[ilow].sytwo + yratio*thePixelTemp[index_id].entby[ihigh].sytwo;
	   pqmin = (1. - yratio)*thePixelTemp[index_id].entby[ilow].qmin + yratio*thePixelTemp[index_id].entby[ihigh].qmin;
	   pclsleny = fminf(thePixelTemp[index_id].entby[ilow].clsleny, thePixelTemp[index_id].entby[ihigh].clsleny);
	   for(i=0; i<2 ; ++i) {
	      for(j=0; j<5 ; ++j) {
// Charge loss switches sides when cot(beta) changes sign
		     if(cotbeta < 0) {
	            pyparl[1-i][j] = thePixelTemp[index_id].entby[ilow].ypar[i][j];
	            pyparh[1-i][j] = thePixelTemp[index_id].entby[ihigh].ypar[i][j];
			 } else {
	            pyparl[i][j] = thePixelTemp[index_id].entby[ilow].ypar[i][j];
	            pyparh[i][j] = thePixelTemp[index_id].entby[ihigh].ypar[i][j];
			 }
	         pxparly0[i][j] = thePixelTemp[index_id].entby[ilow].xpar[i][j];
	         pxparhy0[i][j] = thePixelTemp[index_id].entby[ihigh].xpar[i][j];
	      }
	   }
	   for(i=0; i<4; ++i) {
	      pyavg[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yavg[i] + yratio*thePixelTemp[index_id].entby[ihigh].yavg[i];
	      if(cotbeta < 0.) {pyavg[i] = -pyavg[i];}
	      pyrms[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yrms[i] + yratio*thePixelTemp[index_id].entby[ihigh].yrms[i];
//	      pygx0[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ygx0[i] + yratio*thePixelTemp[index_id].entby[ihigh].ygx0[i];
//	      if(cotbeta < 0.) {pygx0[i] = -pygx0[i];}
//	      pygsig[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ygsig[i] + yratio*thePixelTemp[index_id].entby[ihigh].ygsig[i];
//	      xrms[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].xrms[i] + yratio*thePixelTemp[index_id].entby[ihigh].xrms[i];
//	      xgsig[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].xgsig[i] + yratio*thePixelTemp[index_id].entby[ihigh].xgsig[i];
	      pchi2yavg[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].chi2yavg[i] + yratio*thePixelTemp[index_id].entby[ihigh].chi2yavg[i];
	      pchi2ymin[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].chi2ymin[i] + yratio*thePixelTemp[index_id].entby[ihigh].chi2ymin[i];
	      chi2xavg[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].chi2xavg[i] + yratio*thePixelTemp[index_id].entby[ihigh].chi2xavg[i];
	      chi2xmin[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].chi2xmin[i] + yratio*thePixelTemp[index_id].entby[ihigh].chi2xmin[i];
	      pyavgc2m[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yavgc2m[i] + yratio*thePixelTemp[index_id].entby[ihigh].yavgc2m[i];
	      if(cotbeta < 0.) {pyavgc2m[i] = -pyavgc2m[i];}
	      pyrmsc2m[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yrmsc2m[i] + yratio*thePixelTemp[index_id].entby[ihigh].yrmsc2m[i];
//	      pygx0c2m[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ygx0c2m[i] + yratio*thePixelTemp[index_id].entby[ihigh].ygx0c2m[i];
//	      if(cotbeta < 0.) {pygx0c2m[i] = -pygx0c2m[i];}
//	      pygsigc2m[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ygsigc2m[i] + yratio*thePixelTemp[index_id].entby[ihigh].ygsigc2m[i];
//	      xrmsc2m[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].xrmsc2m[i] + yratio*thePixelTemp[index_id].entby[ihigh].xrmsc2m[i];
//	      xgsigc2m[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].xgsigc2m[i] + yratio*thePixelTemp[index_id].entby[ihigh].xgsigc2m[i];
		  for(j=0; j<6 ; ++j) {
			 pyflparl[i][j] = thePixelTemp[index_id].entby[ilow].yflpar[i][j];
			 pyflparh[i][j] = thePixelTemp[index_id].entby[ihigh].yflpar[i][j];
			 
// Since Q_fl is odd under cotbeta, it flips qutomatically, change only even terms

			 if(cotbeta < 0. && (j == 0 || j == 2 || j == 4)) {
			    pyflparl[i][j] = - pyflparl[i][j];
			    pyflparh[i][j] = - pyflparh[i][j];
			 }
		  }
	   }
	   
// Do the spares next

//       for(i=0; i<10; ++i) {
//		  pyspare[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yspare[i] + yratio*thePixelTemp[index_id].entby[ihigh].yspare[i];
//       }
			  
// Interpolate and build the y-template 
	
	   for(i=0; i<9; ++i) {
          pytemp[i][0] = 0.;
          pytemp[i][1] = 0.;
	      pytemp[i][BYM2] = 0.;
	      pytemp[i][BYM1] = 0.;
	      for(j=0; j<TYSIZE; ++j) {
		  
// Flip the basic y-template when the cotbeta is negative

		     if(cotbeta < 0.) {
	            pytemp[8-i][BYM3-j]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ytemp[i][j] + yratio*thePixelTemp[index_id].entby[ihigh].ytemp[i][j];
			 } else {
	            pytemp[i][j+2]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ytemp[i][j] + yratio*thePixelTemp[index_id].entby[ihigh].ytemp[i][j];
			 }
	      }
	   }
	
// next, loop over all x-angle entries, first, find relevant y-slices   

	   iylow = 0;
	   yxratio = 0.;

	   if(abs_cotb >= thePixelTemp[index_id].entbx[Nyx-1][0].cotbeta) {
	
	       iylow = Nyx-2;
		   yxratio = 1.;
		
	   } else if(abs_cotb >= thePixelTemp[index_id].entbx[0][0].cotbeta) {

          for (i=0; i<Nyx-1; ++i) { 
    
             if( thePixelTemp[index_id].entbx[i][0].cotbeta <= abs_cotb && abs_cotb < thePixelTemp[index_id].entbx[i+1][0].cotbeta) {
		  
	            iylow = i;
		        yxratio = (abs_cotb - thePixelTemp[index_id].entbx[i][0].cotbeta)/(thePixelTemp[index_id].entbx[i+1][0].cotbeta - thePixelTemp[index_id].entbx[i][0].cotbeta);
		        break;			 
		     }
	      }
	   }
	
	   iyhigh=iylow + 1;

	   ilow = 0;
	   xxratio = 0.;

	   if(cotalpha >= thePixelTemp[index_id].entbx[0][Nxx-1].cotalpha) {
	
	       ilow = Nxx-2;
		   xxratio = 1.;
		   success = false;
		
	   } else {
	      if(cotalpha >= thePixelTemp[index_id].entbx[0][0].cotalpha) {

             for (i=0; i<Nxx-1; ++i) { 
    
                if( thePixelTemp[index_id].entbx[0][i].cotalpha <= cotalpha && cotalpha < thePixelTemp[index_id].entbx[0][i+1].cotalpha) {
		  
	               ilow = i;
		           xxratio = (cotalpha - thePixelTemp[index_id].entbx[0][i].cotalpha)/(thePixelTemp[index_id].entbx[0][i+1].cotalpha - thePixelTemp[index_id].entbx[0][i].cotalpha);
		           break;
			    }
		     }
		  } else { success = false; }
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all x-related quantities 

       pyxratio = yxratio;
       pxxratio = xxratio;		
	   		  
// sxparmax defines the maximum charge for which the parameters xpar are defined (not rescaled by cotbeta) 

	   psxparmax = (1. - xxratio)*thePixelTemp[index_id].entbx[imaxx][ilow].sxmax + xxratio*thePixelTemp[index_id].entbx[imaxx][ihigh].sxmax;
	   psxmax = psxparmax;
       if(thePixelTemp[index_id].entbx[imaxx][imidy].sxmax != 0.) {psxmax=psxmax/thePixelTemp[index_id].entbx[imaxx][imidy].sxmax*sxmax;}
	   psymax = (1. - xxratio)*thePixelTemp[index_id].entbx[imaxx][ilow].symax + xxratio*thePixelTemp[index_id].entbx[imaxx][ihigh].symax;
       if(thePixelTemp[index_id].entbx[imaxx][imidy].symax != 0.) {psymax=psymax/thePixelTemp[index_id].entbx[imaxx][imidy].symax*symax;}
	   pdxone = (1. - xxratio)*thePixelTemp[index_id].entbx[0][ilow].dxone + xxratio*thePixelTemp[index_id].entbx[0][ihigh].dxone;
	   psxone = (1. - xxratio)*thePixelTemp[index_id].entbx[0][ilow].sxone + xxratio*thePixelTemp[index_id].entbx[0][ihigh].sxone;
	   pdxtwo = (1. - xxratio)*thePixelTemp[index_id].entbx[0][ilow].dxtwo + xxratio*thePixelTemp[index_id].entbx[0][ihigh].dxtwo;
	   psxtwo = (1. - xxratio)*thePixelTemp[index_id].entbx[0][ilow].sxtwo + xxratio*thePixelTemp[index_id].entbx[0][ihigh].sxtwo;
	   pclslenx = fminf(thePixelTemp[index_id].entbx[0][ilow].clslenx, thePixelTemp[index_id].entbx[0][ihigh].clslenx);
	   for(i=0; i<2 ; ++i) {
	      for(j=0; j<5 ; ++j) {
	         pxpar0[i][j] = thePixelTemp[index_id].entbx[imaxx][imidy].xpar[i][j];
	         pxparl[i][j] = thePixelTemp[index_id].entbx[imaxx][ilow].xpar[i][j];
	         pxparh[i][j] = thePixelTemp[index_id].entbx[imaxx][ihigh].xpar[i][j];
	      }
	   }
	   		  
// pixmax is the maximum allowed pixel charge (used for truncation)

	   ppixmax=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].pixmax + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].pixmax)
			  +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].pixmax + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].pixmax);
			  
	   for(i=0; i<4; ++i) {
	      pxavg[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xavg[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xavg[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xavg[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xavg[i]);
		  
	      pxrms[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xrms[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xrms[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xrms[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xrms[i]);
		  
//	      pxgx0[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xgx0[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xgx0[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xgx0[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xgx0[i]);
							
//	      pxgsig[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xgsig[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xgsig[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xgsig[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xgsig[i]);
				  
	      pxavgc2m[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xavgc2m[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xavgc2m[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xavgc2m[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xavgc2m[i]);
		  
	      pxrmsc2m[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xrmsc2m[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xrmsc2m[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xrmsc2m[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xrmsc2m[i]);
		  
//	      pxgx0c2m[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xgx0c2m[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xgx0c2m[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xgx0c2m[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xgx0c2m[i]);
							
//	      pxgsigc2m[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xgsigc2m[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xgsigc2m[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xgsigc2m[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xgsigc2m[i]);
//
//  Try new interpolation scheme
//	  															  														
//	      pchi2xavg[i]=((1. - xxratio)*thePixelTemp[index_id].entbx[imaxx][ilow].chi2xavg[i] + xxratio*thePixelTemp[index_id].entbx[imaxx][ihigh].chi2xavg[i]);
//		  if(thePixelTemp[index_id].entbx[imaxx][imidy].chi2xavg[i] != 0.) {pchi2xavg[i]=pchi2xavg[i]/thePixelTemp[index_id].entbx[imaxx][imidy].chi2xavg[i]*chi2xavg[i];}
//							
//	      pchi2xmin[i]=((1. - xxratio)*thePixelTemp[index_id].entbx[imaxx][ilow].chi2xmin[i] + xxratio*thePixelTemp[index_id].entbx[imaxx][ihigh].chi2xmin[i]);
//		  if(thePixelTemp[index_id].entbx[imaxx][imidy].chi2xmin[i] != 0.) {pchi2xmin[i]=pchi2xmin[i]/thePixelTemp[index_id].entbx[imaxx][imidy].chi2xmin[i]*chi2xmin[i];}
		  
	      pchi2xavg[i]=((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].chi2xavg[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].chi2xavg[i]);
		  if(thePixelTemp[index_id].entbx[iyhigh][imidy].chi2xavg[i] != 0.) {pchi2xavg[i]=pchi2xavg[i]/thePixelTemp[index_id].entbx[iyhigh][imidy].chi2xavg[i]*chi2xavg[i];}
							
	      pchi2xmin[i]=((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].chi2xmin[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].chi2xmin[i]);
		  if(thePixelTemp[index_id].entbx[iyhigh][imidy].chi2xmin[i] != 0.) {pchi2xmin[i]=pchi2xmin[i]/thePixelTemp[index_id].entbx[iyhigh][imidy].chi2xmin[i]*chi2xmin[i];}
		  
	      for(j=0; j<6 ; ++j) {
	         pxflparll[i][j] = thePixelTemp[index_id].entbx[iylow][ilow].xflpar[i][j];
	         pxflparlh[i][j] = thePixelTemp[index_id].entbx[iylow][ihigh].xflpar[i][j];
	         pxflparhl[i][j] = thePixelTemp[index_id].entbx[iyhigh][ilow].xflpar[i][j];
	         pxflparhh[i][j] = thePixelTemp[index_id].entbx[iyhigh][ihigh].xflpar[i][j];
		  }
	   }
	   
// Do the spares next

//       for(i=0; i<10; ++i) {
//	      pxspare[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xspare[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xspare[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xspare[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xspare[i]);
//       }
			  
// Interpolate and build the x-template 
	
	   for(i=0; i<9; ++i) {
          pxtemp[i][0] = 0.;
          pxtemp[i][1] = 0.;
	      pxtemp[i][BXM2] = 0.;
	      pxtemp[i][BXM1] = 0.;
	      for(j=0; j<TXSIZE; ++j) {
	        pxtemp[i][j+2]=(1. - xxratio)*thePixelTemp[index_id].entbx[imaxx][ilow].xtemp[i][j] + xxratio*thePixelTemp[index_id].entbx[imaxx][ihigh].xtemp[i][j];
	      }
	   }
	}
  }
  return success;
} // interpolate





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
	float sigi, sigi2, sigi3, sigi4, symax, qscale;
	
    // Make sure that input is OK
    
	assert(fypix > 1 && fypix < BYM2);
	assert(lypix >= fypix && lypix < BYM2);
	   	     
// Define the maximum signal to use in the parameterization 

       symax = psymax;
	   if(psymax > psyparmax) {symax = psyparmax;}
	   
// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 

	   for(i=fypix-2; i<=lypix+2; ++i) {
		  if(i < fypix || i > lypix) {
	   
// Nearest pseudopixels have uncertainties of 50% of threshold, next-nearest have 10% of threshold

			 ysig2[i] = ps50*ps50;
		  } else {
			 if(ysum[i] < symax) {
				sigi = ysum[i];
				qscale = 1.;
			 } else {
				sigi = symax;
				qscale = ysum[i]/symax;
			 }
			 sigi2 = sigi*sigi; sigi3 = sigi2*sigi; sigi4 = sigi3*sigi;
			 if(i <= BHY) {
				ysig2[i] = (1.-pyratio)*
				(pyparl[0][0]+pyparl[0][1]*sigi+pyparl[0][2]*sigi2+pyparl[0][3]*sigi3+pyparl[0][4]*sigi4)
				+ pyratio*
				(pyparh[0][0]+pyparh[0][1]*sigi+pyparh[0][2]*sigi2+pyparh[0][3]*sigi3+pyparh[0][4]*sigi4);
			 } else {
				ysig2[i] = (1.-pyratio)*
				(pyparl[1][0]+pyparl[1][1]*sigi+pyparl[1][2]*sigi2+pyparl[1][3]*sigi3+pyparl[1][4]*sigi4)
				+ pyratio*
			    (pyparh[1][0]+pyparh[1][1]*sigi+pyparh[1][2]*sigi2+pyparh[1][3]*sigi3+pyparh[1][4]*sigi4);
			 }
			 ysig2[i] *=qscale;
		     if(ysum[i] > sythr) {ysig2[i] = 1.e8;}
			 if(ysig2[i] <= 0.) {LOGERROR("SiPixelTemplate") << "neg y-error-squared, id = " << id_current << ", index = " << index_id << 
			 ", cot(alpha) = " << cota_current << ", cot(beta) = " << cotb_current << ", fpix = " << fpix_current << " sigi = " << sigi << ENDL;}
	      }
	   }
	
	return;
	
} // End ysigma2







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
	float sigi, sigi2, sigi3, sigi4, yint, sxmax, x0, qscale;
	
    // Make sure that input is OK
    
	assert(fxpix > 1 && fxpix < BXM2);
	assert(lxpix >= fxpix && lxpix < BXM2);
	   	     
// Define the maximum signal to use in the parameterization 

       sxmax = psxmax;
	   if(psxmax > psxparmax) {sxmax = psxparmax;}
	   
// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 

	   for(i=fxpix-2; i<=lxpix+2; ++i) {
		  if(i < fxpix || i > lxpix) {
	   
// Nearest pseudopixels have uncertainties of 50% of threshold, next-nearest have 10% of threshold

			 xsig2[i] = ps50*ps50;
		  } else {
			 if(xsum[i] < sxmax) {
				sigi = xsum[i];
				qscale = 1.;
			 } else {
				sigi = sxmax;
				qscale = xsum[i]/sxmax;
			 }
			 sigi2 = sigi*sigi; sigi3 = sigi2*sigi; sigi4 = sigi3*sigi;
			 
// First, do the cotbeta interpolation			 
			 
			 if(i <= BHX) {
				yint = (1.-pyratio)*
				(pxparly0[0][0]+pxparly0[0][1]*sigi+pxparly0[0][2]*sigi2+pxparly0[0][3]*sigi3+pxparly0[0][4]*sigi4)
				+ pyratio*
				(pxparhy0[0][0]+pxparhy0[0][1]*sigi+pxparhy0[0][2]*sigi2+pxparhy0[0][3]*sigi3+pxparhy0[0][4]*sigi4);
			 } else {
				yint = (1.-pyratio)*
				(pxparly0[1][0]+pxparly0[1][1]*sigi+pxparly0[1][2]*sigi2+pxparly0[1][3]*sigi3+pxparly0[1][4]*sigi4)
				+ pyratio*
			    (pxparhy0[1][0]+pxparhy0[1][1]*sigi+pxparhy0[1][2]*sigi2+pxparhy0[1][3]*sigi3+pxparhy0[1][4]*sigi4);
			 }
			 
// Next, do the cotalpha interpolation			 
			 
			 if(i <= BHX) {
				xsig2[i] = (1.-pxxratio)*
				(pxparl[0][0]+pxparl[0][1]*sigi+pxparl[0][2]*sigi2+pxparl[0][3]*sigi3+pxparl[0][4]*sigi4)
				+ pxxratio*
				(pxparh[0][0]+pxparh[0][1]*sigi+pxparh[0][2]*sigi2+pxparh[0][3]*sigi3+pxparh[0][4]*sigi4);
			 } else {
				xsig2[i] = (1.-pxxratio)*
				(pxparl[1][0]+pxparl[1][1]*sigi+pxparl[1][2]*sigi2+pxparl[1][3]*sigi3+pxparl[1][4]*sigi4)
				+ pxxratio*
			    (pxparh[1][0]+pxparh[1][1]*sigi+pxparh[1][2]*sigi2+pxparh[1][3]*sigi3+pxparh[1][4]*sigi4);
			 }
			 
// Finally, get the mid-point value of the cotalpha function			 
			 
			 if(i <= BHX) {
				x0 = pxpar0[0][0]+pxpar0[0][1]*sigi+pxpar0[0][2]*sigi2+pxpar0[0][3]*sigi3+pxpar0[0][4]*sigi4;
			 } else {
				x0 = pxpar0[1][0]+pxpar0[1][1]*sigi+pxpar0[1][2]*sigi2+pxpar0[1][3]*sigi3+pxpar0[1][4]*sigi4;
			 }
			 
// Finally, rescale the yint value for cotalpha variation			 
			 
			 if(x0 != 0.) {xsig2[i] = xsig2[i]/x0 * yint;}
			 xsig2[i] *=qscale;
		     if(xsum[i] > sxthr) {xsig2[i] = 1.e8;}
			 if(xsig2[i] <= 0.) {LOGERROR("SiPixelTemplate") << "neg x-error-squared, id = " << id_current << ", index = " << index_id << 
			 ", cot(alpha) = " << cota_current << ", cot(beta) = " << cotb_current << ", fpix = " << fpix_current << " sigi = " << sigi << ENDL;}
	      }
	   }
	
	return;
	
} // End xsigma2






// ************************************************************************************************************ 
//! Return interpolated y-correction for input charge bin and qfly
//! \param binq - (input) charge bin [0-3]
//! \param qfly - (input) (Q_f-Q_l)/(Q_f+Q_l) for this cluster
// ************************************************************************************************************ 
  float SiPixelTemplate::yflcorr(int binq, float qfly)
  
{
    // Interpolate using quantities already stored in the private variables
    
    // Local variables 
    int i;
	float qfl, qfl2, qfl3, qfl4, qfl5, dy;
	
    // Make sure that input is OK
    
	assert(binq >= 0 && binq < 4);
	assert(fabs((double)qfly) <= 1.);
	   	     
// Define the maximum signal to allow before de-weighting a pixel 

       qfl = qfly;

       if(qfl < -0.9) {qfl = -0.9;}
	   if(qfl > 0.9) {qfl = 0.9;}
	   
// Interpolate between the two polynomials

	   qfl2 = qfl*qfl; qfl3 = qfl2*qfl; qfl4 = qfl3*qfl; qfl5 = qfl4*qfl;
	   dy = (1.-pyratio)*(pyflparl[binq][0]+pyflparl[binq][1]*qfl+pyflparl[binq][2]*qfl2+pyflparl[binq][3]*qfl3+pyflparl[binq][4]*qfl4+pyflparl[binq][5]*qfl5)
		  + pyratio*(pyflparh[binq][0]+pyflparh[binq][1]*qfl+pyflparh[binq][2]*qfl2+pyflparh[binq][3]*qfl3+pyflparh[binq][4]*qfl4+pyflparh[binq][5]*qfl5);
	
	return dy;
	
} // End yflcorr






// ************************************************************************************************************ 
//! Return interpolated x-correction for input charge bin and qflx
//! \param binq - (input) charge bin [0-3]
//! \param qflx - (input) (Q_f-Q_l)/(Q_f+Q_l) for this cluster
// ************************************************************************************************************ 
  float SiPixelTemplate::xflcorr(int binq, float qflx)
  
{
    // Interpolate using quantities already stored in the private variables
    
    // Local variables 
    int i;
	float qfl, qfl2, qfl3, qfl4, qfl5, dx;
	
    // Make sure that input is OK
    
	assert(binq >= 0 && binq < 4);
	assert(fabs((double)qflx) <= 1.);
	   	     
// Define the maximum signal to allow before de-weighting a pixel 

       qfl = qflx;

       if(qfl < -0.9) {qfl = -0.9;}
	   if(qfl > 0.9) {qfl = 0.9;}
	   
// Interpolate between the two polynomials

	   qfl2 = qfl*qfl; qfl3 = qfl2*qfl; qfl4 = qfl3*qfl; qfl5 = qfl4*qfl;
	   dx = (1. - pyxratio)*((1.-pxxratio)*(pxflparll[binq][0]+pxflparll[binq][1]*qfl+pxflparll[binq][2]*qfl2+pxflparll[binq][3]*qfl3+pxflparll[binq][4]*qfl4+pxflparll[binq][5]*qfl5)
		  + pxxratio*(pxflparlh[binq][0]+pxflparlh[binq][1]*qfl+pxflparlh[binq][2]*qfl2+pxflparlh[binq][3]*qfl3+pxflparlh[binq][4]*qfl4+pxflparlh[binq][5]*qfl5))
	      + pyxratio*((1.-pxxratio)*(pxflparhl[binq][0]+pxflparhl[binq][1]*qfl+pxflparhl[binq][2]*qfl2+pxflparhl[binq][3]*qfl3+pxflparhl[binq][4]*qfl4+pxflparhl[binq][5]*qfl5)
		  + pxxratio*(pxflparhh[binq][0]+pxflparhh[binq][1]*qfl+pxflparhh[binq][2]*qfl2+pxflparhh[binq][3]*qfl3+pxflparhh[binq][4]*qfl4+pxflparhh[binq][5]*qfl5));
	
	return dx;
	
} // End xflcorr



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

	assert(fybin >= 0 && fybin < 41);
	assert(lybin >= 0 && lybin < 41);

// Build the y-template, the central 25 bins are here in all cases
	
	for(i=0; i<9; ++i) {
	   for(j=0; j<BYSIZE; ++j) {
			ytemplate[i+16][j]=pytemp[i][j];
	   }
	}
	for(i=0; i<8; ++i) {
	   ytemplate[i+8][BYM1] = 0.;
	   for(j=0; j<BYM1; ++j) {
	      ytemplate[i+8][j]=pytemp[i][j+1];
	   }
	}
	for(i=1; i<9; ++i) {
	   ytemplate[i+24][0] = 0.;
	   for(j=0; j<BYM1; ++j) {
	      ytemplate[i+24][j+1]=pytemp[i][j];
	   }
	}
	
//  Add	more bins if needed

	if(fybin < 8) {
	   for(i=0; i<8; ++i) {
	      ytemplate[i][BYM2] = 0.;
	      ytemplate[i][BYM1] = 0.;
	      for(j=0; j<BYM2; ++j) {
	        ytemplate[i][j]=pytemp[i][j+2];
	      }
	   }
	}
	if(lybin > 32) {
  	   for(i=1; i<9; ++i) {
          ytemplate[i+32][0] = 0.;
	      ytemplate[i+32][1] = 0.;
	      for(j=0; j<BYM2; ++j) {
	         ytemplate[i+32][j+2]=pytemp[i][j];
	      }
	   }
	}
	
	return;
	
} // End ytemp



// ************************************************************************************************************ 
//! Return interpolated y-template in single call
//! \param fxbin - (input) index of first bin (0-40) to fill
//! \param fxbin - (input) index of last bin (0-40) to fill
//! \param xtemplate - (output) a 41x11 output buffer
// ************************************************************************************************************ 
  void SiPixelTemplate::xtemp(int fxbin, int lxbin, float xtemplate[41][BXSIZE])
  
{
    // Retrieve already interpolated quantities
    
    // Local variables 
    int i, j;

   // Verify that input parameters are in valid range

	assert(fxbin >= 0 && fxbin < 41);
	assert(lxbin >= 0 && lxbin < 41);

// Build the x-template, the central 25 bins are here in all cases
	
	for(i=0; i<9; ++i) {
	   for(j=0; j<BXSIZE; ++j) {
	      xtemplate[i+16][j]=pxtemp[i][j];
	   }
	}
	for(i=0; i<8; ++i) {
	   xtemplate[i+8][BXM1] = 0.;
	   for(j=0; j<BXM1; ++j) {
	      xtemplate[i+8][j]=pxtemp[i][j+1];
	   }
	}
	for(i=1; i<9; ++i) {
	   xtemplate[i+24][0] = 0.;
	   for(j=0; j<BXM1; ++j) {
	      xtemplate[i+24][j+1]=pxtemp[i][j];
	   }
	}
	
//  Add more bins if needed	
	
	if(fxbin < 8) {
	   for(i=0; i<8; ++i) {
          xtemplate[i][BXM2] = 0.;
	      xtemplate[i][BXM1] = 0.;
	      for(j=0; j<BXM2; ++j) {
	        xtemplate[i][j]=pxtemp[i][j+2];
	      }
	   }
	}
	if(lxbin > 32) {
	   for(i=1; i<9; ++i) {
          xtemplate[i+32][0] = 0.;
	      xtemplate[i+32][1] = 0.;
	      for(j=0; j<BXM2; ++j) {
	        xtemplate[i+32][j+2]=pxtemp[i][j];
	      }
	   }
	}
	
 	return;
	
} // End xtemp


// ************************************************************************************************************ 
//! Return interpolated 3d y-template in single call
//! \param nypix - (input) number of pixels in cluster (needed to size template)
//! \param ytemplate - (output) a boost 3d array containing two sets of temlate indices and the combined pixel signals
// ************************************************************************************************************ 
  void SiPixelTemplate::ytemp3d(int nypix, array_3d& ytemplate)
  
{
    typedef boost::multi_array<float, 2> array_2d;
	
    // Retrieve already interpolated quantities
    
    // Local variables 
    int i, j, k;
	int ioff0, ioffp, ioffm;

   // Verify that input parameters are in valid range

	assert(nypix > 0 && nypix < BYM3);
	
// Calculate the size of the shift in pixels needed to span the entire cluster

    float diff = fabsf(nypix - pclsleny)/2. + 1.;
	int nshift = (int) diff;
	if((diff - nshift) > 0.5) {++nshift;}

// Calculate the number of bins needed to specify each hit range

    int nbins = 9 + 16*nshift;
	
// Create a 2-d working template with the correct size

	array_2d temp2d(boost::extents[nbins][BYSIZE]);
	
//  The 9 central bins are copied from the interpolated private store

	ioff0 = 8*nshift;
	
	for(i=0; i<9; ++i) {
	   for(j=0; j<BYSIZE; ++j) {
          temp2d[i+ioff0][j]=pytemp[i][j];
	   }
	}
	
// Add the +- shifted templates	

	for(k=1; k<=nshift; ++k) {
	  ioffm=ioff0-k*8;
	  for(i=0; i<8; ++i) {
	     for(j=0; j<k; ++j) {
	        temp2d[i+ioffm][BYM1-j] = 0.;
		 }
	     for(j=0; j<BYSIZE-k; ++j) {
	        temp2d[i+ioffm][j]=pytemp[i][j+k];
	     }
	   }
	   ioffp=ioff0+k*8;
	   for(i=1; i<9; ++i) {
	      for(j=0; j<k; ++j) {
	         temp2d[i+ioffp][j] = 0.;
	      }
	      for(j=0; j<BYSIZE-k; ++j) {
	         temp2d[i+ioffp][j+k]=pytemp[i][j];
	      }
	   }
	}
		
// Resize the 3d template container

    ytemplate.resize(boost::extents[nbins][nbins][BYSIZE]);
	
// Sum two 2-d templates to make the 3-d template
	
   for(i=0; i<nbins; ++i) {
      for(j=0; j<=i; ++j) {
         for(k=0; k<BYSIZE; ++k) {
            ytemplate[i][j][k]=temp2d[i][k]+temp2d[j][k];	
		 }
	  }
   }
	
	return;
	
} // End ytemp3d


// ************************************************************************************************************ 
//! Return interpolated 3d y-template in single call
//! \param nxpix - (input) number of pixels in cluster (needed to size template)
//! \param xtemplate - (output) a boost 3d array containing two sets of temlate indices and the combined pixel signals
// ************************************************************************************************************ 
  void SiPixelTemplate::xtemp3d(int nxpix, array_3d& xtemplate)
  
{
    typedef boost::multi_array<float, 2> array_2d;
	
    // Retrieve already interpolated quantities
    
    // Local variables 
    int i, j, k;
	int ioff0, ioffp, ioffm;

   // Verify that input parameters are in valid range

	assert(nxpix > 0 && nxpix < BXM3);
	
// Calculate the size of the shift in pixels needed to span the entire cluster

    float diff = fabsf(nxpix - pclslenx)/2. + 1.;
	int nshift = (int) diff;
	if((diff - nshift) > 0.5) {++nshift;}

// Calculate the number of bins needed to specify each hit range

    int nbins = 9 + 16*nshift;
	
// Create a 2-d working template with the correct size

	array_2d temp2d(boost::extents[nbins][BXSIZE]);
	
//  The 9 central bins are copied from the interpolated private store

	ioff0 = 8*nshift;
	
	for(i=0; i<9; ++i) {
	   for(j=0; j<BXSIZE; ++j) {
          temp2d[i+ioff0][j]=pxtemp[i][j];
	   }
	}
	
// Add the +- shifted templates	

	for(k=1; k<=nshift; ++k) {
	  ioffm=ioff0-k*8;
	  for(i=0; i<8; ++i) {
	     for(j=0; j<k; ++j) {
	        temp2d[i+ioffm][BXM1-j] = 0.;
		 }
	     for(j=0; j<BXSIZE-k; ++j) {
	        temp2d[i+ioffm][j]=pxtemp[i][j+k];
	     }
	   }
	   ioffp=ioff0+k*8;
	   for(i=1; i<9; ++i) {
	      for(j=0; j<k; ++j) {
	         temp2d[i+ioffp][j] = 0.;
	      }
	      for(j=0; j<BXSIZE-k; ++j) {
	         temp2d[i+ioffp][j+k]=pxtemp[i][j];
	      }
	   }
	}
				
// Resize the 3d template container

    xtemplate.resize(boost::extents[nbins][nbins][BXSIZE]);
	
// Sum two 2-d templates to make the 3-d template
	
   for(i=0; i<nbins; ++i) {
      for(j=0; j<=i; ++j) {
         for(k=0; k<BXSIZE; ++k) {
            xtemplate[i][j][k]=temp2d[i][k]+temp2d[j][k];	
		 }
	  }
   }
	
	return;
	
} // End xtemp3d



// ************************************************************************************************************ 
//! Interpolate beta angles to produce an expected average charge. Return int (0-4) describing the charge 
//! of the cluster [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin].
//! \param id - (input) index of the template to use
//! \param fpix - (input) logical input indicating whether to use FPix templates (true) 
//!               or Barrel templates (false)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param qclus - (input) the cluster charge in electrons 
// ************************************************************************************************************ 
int SiPixelTemplate::qbin(int id, bool fpix, float cotbeta, float qclus)
{
    // Interpolate for a new set of track angles 
    
    // Local variables 
    int i, j, binq;
	int ilow, ihigh, iylow, iyhigh, Ny, Nxx, Nyx, imidy, imaxx;
	float yratio, yxratio, xxratio, sxmax;
	float acotb, qscale, qavg, qmin, fq, qtotal;
	
	if(id != id_current) {

// Find the index corresponding to id

       index_id = -1;
       for(i=0; i<thePixelTemp.size(); ++i) {
	
	      if(id == thePixelTemp[i].head.ID) {
	   
	         index_id = i;
		     id_current = id;
		     break;
          }
	    }
     }
	 
	 assert(index_id >= 0 && index_id < thePixelTemp.size());
	 
//		

// Interpolate the absolute value of cot(beta)     
    
    acotb = fabs((double)cotbeta);

// Copy the charge scaling factor to the private variable     
    
    qscale = thePixelTemp[index_id].head.qscale;
	
// Decide which template (FPix or BPix) to use 

    if(fpix) {
    
// Begin FPix section, make the index counters easier to use     
    
       Ny = thePixelTemp[index_id].head.NFy;
       Nyx = thePixelTemp[index_id].head.NFyx;
       Nxx = thePixelTemp[index_id].head.NFxx;
	   imaxx = Nyx - 1;
	   imidy = Nxx/2;
        
// next, loop over all y-angle entries   

	   ilow = 0;
	   yratio = 0.;

	   if(acotb >= thePixelTemp[index_id].entfy[Ny-1].cotbeta) {
	
	       ilow = Ny-2;
		   yratio = 1.;
		
	   } else if(acotb >= thePixelTemp[index_id].entfy[0].cotbeta) {

          for (i=0; i<Ny-1; ++i) { 
    
             if( thePixelTemp[index_id].entfy[i].cotbeta <= acotb && acotb < thePixelTemp[index_id].entfy[i+1].cotbeta) {
		  
	            ilow = i;
		        yratio = (acotb - thePixelTemp[index_id].entfy[i].cotbeta)/(thePixelTemp[index_id].entfy[i+1].cotbeta - thePixelTemp[index_id].entfy[i].cotbeta);
		        break;			 
		     }
	      }
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all y-related quantities (flip displacements when cotbeta < 0)

 	   qavg = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].qavg + yratio*thePixelTemp[index_id].entfy[ihigh].qavg;
	   qmin = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].qmin + yratio*thePixelTemp[index_id].entfy[ihigh].qmin;

	} else {
	
    
// Begin BPix section, make the index counters easier to use     
    
       Ny = thePixelTemp[index_id].head.NBy;
       Nyx = thePixelTemp[index_id].head.NByx;
       Nxx = thePixelTemp[index_id].head.NBxx;
	   imaxx = Nyx - 1;
	   imidy = Nxx/2;
        
// next, loop over all y-angle entries   

	   ilow = 0;
	   yratio = 0.;

	   if(acotb >= thePixelTemp[index_id].entby[Ny-1].cotbeta) {
	
	       ilow = Ny-2;
		   yratio = 1.;
		
	   } else if(acotb >= thePixelTemp[index_id].entby[0].cotbeta) {

          for (i=0; i<Ny-1; ++i) { 
    
             if( thePixelTemp[index_id].entby[i].cotbeta <= acotb && abs_cotb < thePixelTemp[index_id].entby[i+1].cotbeta) {
		  
	            ilow = i;
		        yratio = (acotb - thePixelTemp[index_id].entby[i].cotbeta)/(thePixelTemp[index_id].entby[i+1].cotbeta - thePixelTemp[index_id].entby[i].cotbeta);
		        break;			 
		     }
	      }
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all y-related quantities (flip displacements when cotbeta < 0)

	   qavg = (1. - yratio)*thePixelTemp[index_id].entby[ilow].qavg + yratio*thePixelTemp[index_id].entby[ihigh].qavg;
	   qmin = (1. - yratio)*thePixelTemp[index_id].entby[ilow].qmin + yratio*thePixelTemp[index_id].entby[ihigh].qmin;
	}
	
	assert(qavg > 0. && qmin > 0.);
	
//  Scale the input charge to account for differences between pixelav and CMSSW simulation or data	
	
	qtotal = qscale*qclus;
	
// uncertainty and final corrections depend upon total charge bin 	   
	   
	fq = qtotal/qavg;
	if(fq > 1.5) {
	   binq=0;
	} else {
	   if(fq > 1.0) {
	      binq=1;
	   } else {
		  if(fq > 0.85) {
			 binq=2;
		  } else {
			 binq=3;
		  }
	   }
	}
	
// If the charge is too small (then flag it)
	
	if(qtotal < 0.95*qmin) {binq = 4;}
		
    return binq;
  
} // qbin





