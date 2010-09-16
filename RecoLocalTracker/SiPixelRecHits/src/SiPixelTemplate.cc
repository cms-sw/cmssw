//
//  SiPixelTemplate.cc  Version 8.12 
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
#include <list>



#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SimplePixel.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) LogError(x)
#define LOGINFO(x) LogInfo(x)
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
bool SiPixelTemplate::pushfile(int filenum)
{
    // Add template stored in external file numbered filenum to theTemplateStore
    
    // Local variables 
    int i, j, k, l;
	float qavg_avg;
	const char *tempfile;
	//	char title[80]; remove this
    char c;
	const int code_version={16};
	
	
	
	//  Create a filename for this run 
	
	std::ostringstream tout;
	
	//  Create different path in CMSSW than standalone
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	tout << "CalibTracker/SiPixelESProducers/data/template_summary_zp" 
	<< std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
	std::string tempf = tout.str();
	edm::FileInPath file( tempf.c_str() );
	tempfile = (file.fullPath()).c_str();
#else
	tout << "template_summary_zp" << std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
	std::string tempf = tout.str();
	tempfile = tempf.c_str();
#endif
	
	//  open the template file 
	
	std::ifstream in_file(tempfile, std::ios::in);
	
	if(in_file.is_open()) {
		
		// Create a local template storage entry
		
		SiPixelTemplateStore theCurrentTemp;
		
		// Read-in a header string first and print it    
		
		for (i=0; (c=in_file.get()) != '\n'; ++i) {
			if(i < 79) {theCurrentTemp.head.title[i] = c;}
		}
		if(i > 78) {i=78;}
		theCurrentTemp.head.title[i+1] ='\0';
		LOGINFO("SiPixelTemplate") << "Loading Pixel Template File - " << theCurrentTemp.head.title << ENDL;
		
		// next, the header information     
		
		in_file >> theCurrentTemp.head.ID  >> theCurrentTemp.head.templ_version >> theCurrentTemp.head.Bfield >> theCurrentTemp.head.NTy >> theCurrentTemp.head.NTyx >> theCurrentTemp.head.NTxx
		>> theCurrentTemp.head.Dtype >> theCurrentTemp.head.Vbias >> theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >> theCurrentTemp.head.qscale
		>> theCurrentTemp.head.s50 >> theCurrentTemp.head.lorywidth >> theCurrentTemp.head.lorxwidth >> theCurrentTemp.head.ysize >> theCurrentTemp.head.xsize >> theCurrentTemp.head.zsize;
		
		if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
		
		LOGINFO("SiPixelTemplate") << "Template ID = " << theCurrentTemp.head.ID << ", Template Version " << theCurrentTemp.head.templ_version << ", Bfield = " << theCurrentTemp.head.Bfield 
		<< ", NTy = " << theCurrentTemp.head.NTy << ", NTyx = " << theCurrentTemp.head.NTyx<< ", NTxx = " << theCurrentTemp.head.NTxx << ", Dtype = " << theCurrentTemp.head.Dtype
		<< ", Bias voltage " << theCurrentTemp.head.Vbias << ", temperature "
		<< theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence << ", Q-scaling factor " << theCurrentTemp.head.qscale
		<< ", 1/2 threshold " << theCurrentTemp.head.s50 << ", y Lorentz Width " << theCurrentTemp.head.lorywidth << ", x Lorentz width " << theCurrentTemp.head.lorxwidth    
		<< ", pixel x-size " << theCurrentTemp.head.xsize << ", y-size " << theCurrentTemp.head.ysize << ", zsize " << theCurrentTemp.head.zsize << ENDL;
		
		if(theCurrentTemp.head.templ_version < code_version) {LOGERROR("SiPixelTemplate") << "code expects version " << code_version << ", no template load" << ENDL; return false;}
		
		// next, loop over all y-angle entries   
		
		for (i=0; i < theCurrentTemp.head.NTy; ++i) {     
			
			in_file >> theCurrentTemp.enty[i].runnum >> theCurrentTemp.enty[i].costrk[0] 
			>> theCurrentTemp.enty[i].costrk[1] >> theCurrentTemp.enty[i].costrk[2]; 
			
			if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 1, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			// Calculate the alpha, beta, and cot(beta) for this entry 
			
			theCurrentTemp.enty[i].alpha = static_cast<float>(atan2((double)theCurrentTemp.enty[i].costrk[2], (double)theCurrentTemp.enty[i].costrk[0]));
			
			theCurrentTemp.enty[i].cotalpha = theCurrentTemp.enty[i].costrk[0]/theCurrentTemp.enty[i].costrk[2];
			
			theCurrentTemp.enty[i].beta = static_cast<float>(atan2((double)theCurrentTemp.enty[i].costrk[2], (double)theCurrentTemp.enty[i].costrk[1]));
			
			theCurrentTemp.enty[i].cotbeta = theCurrentTemp.enty[i].costrk[1]/theCurrentTemp.enty[i].costrk[2];
			
			in_file >> theCurrentTemp.enty[i].qavg >> theCurrentTemp.enty[i].pixmax >> theCurrentTemp.enty[i].symax >> theCurrentTemp.enty[i].dyone
			>> theCurrentTemp.enty[i].syone >> theCurrentTemp.enty[i].sxmax >> theCurrentTemp.enty[i].dxone >> theCurrentTemp.enty[i].sxone;
			
			if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 2, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			in_file >> theCurrentTemp.enty[i].dytwo >> theCurrentTemp.enty[i].sytwo >> theCurrentTemp.enty[i].dxtwo 
			>> theCurrentTemp.enty[i].sxtwo >> theCurrentTemp.enty[i].qmin >> theCurrentTemp.enty[i].clsleny >> theCurrentTemp.enty[i].clslenx;
			
			if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 3, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			for (j=0; j<2; ++j) {
				
				in_file >> theCurrentTemp.enty[i].ypar[j][0] >> theCurrentTemp.enty[i].ypar[j][1] 
				>> theCurrentTemp.enty[i].ypar[j][2] >> theCurrentTemp.enty[i].ypar[j][3] >> theCurrentTemp.enty[i].ypar[j][4];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 4, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
				
			}
			
			for (j=0; j<9; ++j) {
				
				for (k=0; k<TYSIZE; ++k) {in_file >> theCurrentTemp.enty[i].ytemp[j][k];}
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 5, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<2; ++j) {
				
				in_file >> theCurrentTemp.enty[i].xpar[j][0] >> theCurrentTemp.enty[i].xpar[j][1] 
				>> theCurrentTemp.enty[i].xpar[j][2] >> theCurrentTemp.enty[i].xpar[j][3] >> theCurrentTemp.enty[i].xpar[j][4];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 6, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
				
			}
			
			qavg_avg = 0.;
			for (j=0; j<9; ++j) {
				
				for (k=0; k<TXSIZE; ++k) {in_file >> theCurrentTemp.enty[i].xtemp[j][k]; qavg_avg += theCurrentTemp.enty[i].xtemp[j][k];} 
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 7, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			theCurrentTemp.enty[i].qavg_avg = qavg_avg/9.;
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].yavg[j] >> theCurrentTemp.enty[i].yrms[j] >> theCurrentTemp.enty[i].ygx0[j] >> theCurrentTemp.enty[i].ygsig[j];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 8, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].yflpar[j][0] >> theCurrentTemp.enty[i].yflpar[j][1] >> theCurrentTemp.enty[i].yflpar[j][2] 
				>> theCurrentTemp.enty[i].yflpar[j][3] >> theCurrentTemp.enty[i].yflpar[j][4] >> theCurrentTemp.enty[i].yflpar[j][5];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 9, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].xavg[j] >> theCurrentTemp.enty[i].xrms[j] >> theCurrentTemp.enty[i].xgx0[j] >> theCurrentTemp.enty[i].xgsig[j];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 10, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].xflpar[j][0] >> theCurrentTemp.enty[i].xflpar[j][1] >> theCurrentTemp.enty[i].xflpar[j][2] 
				>> theCurrentTemp.enty[i].xflpar[j][3] >> theCurrentTemp.enty[i].xflpar[j][4] >> theCurrentTemp.enty[i].xflpar[j][5];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 11, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].chi2yavg[j] >> theCurrentTemp.enty[i].chi2ymin[j] >> theCurrentTemp.enty[i].chi2xavg[j] >> theCurrentTemp.enty[i].chi2xmin[j];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 12, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].yavgc2m[j] >> theCurrentTemp.enty[i].yrmsc2m[j] >> theCurrentTemp.enty[i].ygx0c2m[j] >> theCurrentTemp.enty[i].ygsigc2m[j];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 13, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].xavgc2m[j] >> theCurrentTemp.enty[i].xrmsc2m[j] >> theCurrentTemp.enty[i].xgx0c2m[j] >> theCurrentTemp.enty[i].xgsigc2m[j];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 14, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			} 
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].yavggen[j] >> theCurrentTemp.enty[i].yrmsgen[j] >> theCurrentTemp.enty[i].ygx0gen[j] >> theCurrentTemp.enty[i].ygsiggen[j];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 14a, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].xavggen[j] >> theCurrentTemp.enty[i].xrmsgen[j] >> theCurrentTemp.enty[i].xgx0gen[j] >> theCurrentTemp.enty[i].xgsiggen[j];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 14b, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			} 
			
			in_file >> theCurrentTemp.enty[i].chi2yavgone >> theCurrentTemp.enty[i].chi2yminone >> theCurrentTemp.enty[i].chi2xavgone >> theCurrentTemp.enty[i].chi2xminone >> theCurrentTemp.enty[i].qmin2
			>> theCurrentTemp.enty[i].mpvvav >> theCurrentTemp.enty[i].sigmavav >> theCurrentTemp.enty[i].kappavav >> theCurrentTemp.enty[i].qavg_spare >> theCurrentTemp.enty[i].spare[0];
			
			if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 15, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			in_file >> theCurrentTemp.enty[i].spare[1] >> theCurrentTemp.enty[i].spare[2] >> theCurrentTemp.enty[i].spare[3] >> theCurrentTemp.enty[i].qbfrac[0] >> theCurrentTemp.enty[i].qbfrac[1]
			>> theCurrentTemp.enty[i].qbfrac[2] >> theCurrentTemp.enty[i].fracyone >> theCurrentTemp.enty[i].fracxone >> theCurrentTemp.enty[i].fracytwo >> theCurrentTemp.enty[i].fracxtwo;
			//		theCurrentTemp.enty[i].qbfrac[3] = 1. - theCurrentTemp.enty[i].qbfrac[0] - theCurrentTemp.enty[i].qbfrac[1] - theCurrentTemp.enty[i].qbfrac[2];
			
			if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 16, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
		}
		
		// next, loop over all barrel x-angle entries   
		
		for (k=0; k < theCurrentTemp.head.NTyx; ++k) { 
			
			for (i=0; i < theCurrentTemp.head.NTxx; ++i) { 
				
				in_file >> theCurrentTemp.entx[k][i].runnum >> theCurrentTemp.entx[k][i].costrk[0] 
				>> theCurrentTemp.entx[k][i].costrk[1] >> theCurrentTemp.entx[k][i].costrk[2]; 
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 17, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				// Calculate the alpha, beta, and cot(beta) for this entry 
				
				theCurrentTemp.entx[k][i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entx[k][i].costrk[2], (double)theCurrentTemp.entx[k][i].costrk[0]));
				
				theCurrentTemp.entx[k][i].cotalpha = theCurrentTemp.entx[k][i].costrk[0]/theCurrentTemp.entx[k][i].costrk[2];
				
				theCurrentTemp.entx[k][i].beta = static_cast<float>(atan2((double)theCurrentTemp.entx[k][i].costrk[2], (double)theCurrentTemp.entx[k][i].costrk[1]));
				
				theCurrentTemp.entx[k][i].cotbeta = theCurrentTemp.entx[k][i].costrk[1]/theCurrentTemp.entx[k][i].costrk[2];
				
				in_file >> theCurrentTemp.entx[k][i].qavg >> theCurrentTemp.entx[k][i].pixmax >> theCurrentTemp.entx[k][i].symax >> theCurrentTemp.entx[k][i].dyone
				>> theCurrentTemp.entx[k][i].syone >> theCurrentTemp.entx[k][i].sxmax >> theCurrentTemp.entx[k][i].dxone >> theCurrentTemp.entx[k][i].sxone;
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 18, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				in_file >> theCurrentTemp.entx[k][i].dytwo >> theCurrentTemp.entx[k][i].sytwo >> theCurrentTemp.entx[k][i].dxtwo 
				>> theCurrentTemp.entx[k][i].sxtwo >> theCurrentTemp.entx[k][i].qmin >> theCurrentTemp.entx[k][i].clsleny >> theCurrentTemp.entx[k][i].clslenx;
				//			   >> theCurrentTemp.entx[k][i].mpvvav >> theCurrentTemp.entx[k][i].sigmavav >> theCurrentTemp.entx[k][i].kappavav;
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 19, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				for (j=0; j<2; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].ypar[j][0] >> theCurrentTemp.entx[k][i].ypar[j][1] 
					>> theCurrentTemp.entx[k][i].ypar[j][2] >> theCurrentTemp.entx[k][i].ypar[j][3] >> theCurrentTemp.entx[k][i].ypar[j][4];
					
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 20, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<9; ++j) {
					
					for (l=0; l<TYSIZE; ++l) {in_file >> theCurrentTemp.entx[k][i].ytemp[j][l];} 
		  			
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 21, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<2; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].xpar[j][0] >> theCurrentTemp.entx[k][i].xpar[j][1] 
					>> theCurrentTemp.entx[k][i].xpar[j][2] >> theCurrentTemp.entx[k][i].xpar[j][3] >> theCurrentTemp.entx[k][i].xpar[j][4];
					
					
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 22, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				qavg_avg = 0.;
				for (j=0; j<9; ++j) {
					
					for (l=0; l<TXSIZE; ++l) {in_file >> theCurrentTemp.entx[k][i].xtemp[j][l]; qavg_avg += theCurrentTemp.entx[k][i].xtemp[j][l];} 
					
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 23, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				theCurrentTemp.entx[k][i].qavg_avg = qavg_avg/9.;
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].yavg[j] >> theCurrentTemp.entx[k][i].yrms[j] >> theCurrentTemp.entx[k][i].ygx0[j] >> theCurrentTemp.entx[k][i].ygsig[j];
					
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 24, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].yflpar[j][0] >> theCurrentTemp.entx[k][i].yflpar[j][1] >> theCurrentTemp.entx[k][i].yflpar[j][2] 
					>> theCurrentTemp.entx[k][i].yflpar[j][3] >> theCurrentTemp.entx[k][i].yflpar[j][4] >> theCurrentTemp.entx[k][i].yflpar[j][5];
					
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 25, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].xavg[j] >> theCurrentTemp.entx[k][i].xrms[j] >> theCurrentTemp.entx[k][i].xgx0[j] >> theCurrentTemp.entx[k][i].xgsig[j];
					
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 26, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].xflpar[j][0] >> theCurrentTemp.entx[k][i].xflpar[j][1] >> theCurrentTemp.entx[k][i].xflpar[j][2] 
					>> theCurrentTemp.entx[k][i].xflpar[j][3] >> theCurrentTemp.entx[k][i].xflpar[j][4] >> theCurrentTemp.entx[k][i].xflpar[j][5];
					
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 27, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].chi2yavg[j] >> theCurrentTemp.entx[k][i].chi2ymin[j] >> theCurrentTemp.entx[k][i].chi2xavg[j] >> theCurrentTemp.entx[k][i].chi2xmin[j];
					
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 28, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].yavgc2m[j] >> theCurrentTemp.entx[k][i].yrmsc2m[j] >> theCurrentTemp.entx[k][i].ygx0c2m[j] >> theCurrentTemp.entx[k][i].ygsigc2m[j];
					
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 29, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].xavgc2m[j] >> theCurrentTemp.entx[k][i].xrmsc2m[j] >> theCurrentTemp.entx[k][i].xgx0c2m[j] >> theCurrentTemp.entx[k][i].xgsigc2m[j];
					
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 30, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].yavggen[j] >> theCurrentTemp.entx[k][i].yrmsgen[j] >> theCurrentTemp.entx[k][i].ygx0gen[j] >> theCurrentTemp.entx[k][i].ygsiggen[j];
					
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 30a, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].xavggen[j] >> theCurrentTemp.entx[k][i].xrmsgen[j] >> theCurrentTemp.entx[k][i].xgx0gen[j] >> theCurrentTemp.entx[k][i].xgsiggen[j];
					
					if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 30b, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				in_file >> theCurrentTemp.entx[k][i].chi2yavgone >> theCurrentTemp.entx[k][i].chi2yminone >> theCurrentTemp.entx[k][i].chi2xavgone >> theCurrentTemp.entx[k][i].chi2xminone >> theCurrentTemp.entx[k][i].qmin2
				>> theCurrentTemp.entx[k][i].mpvvav >> theCurrentTemp.entx[k][i].sigmavav >> theCurrentTemp.entx[k][i].kappavav >> theCurrentTemp.entx[k][i].qavg_spare >> theCurrentTemp.entx[k][i].spare[0];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 31, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				in_file >> theCurrentTemp.entx[k][i].spare[1] >> theCurrentTemp.entx[k][i].spare[2] >> theCurrentTemp.entx[k][i].spare[3] >> theCurrentTemp.entx[k][i].qbfrac[0] >> theCurrentTemp.entx[k][i].qbfrac[1]
				>> theCurrentTemp.entx[k][i].qbfrac[2] >> theCurrentTemp.entx[k][i].fracyone >> theCurrentTemp.entx[k][i].fracxone >> theCurrentTemp.entx[k][i].fracytwo >> theCurrentTemp.entx[k][i].fracxtwo;
				//		theCurrentTemp.entx[k][i].qbfrac[3] = 1. - theCurrentTemp.entx[k][i].qbfrac[0] - theCurrentTemp.entx[k][i].qbfrac[1] - theCurrentTemp.entx[k][i].qbfrac[2];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 32, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
			}
		}	
		
		
		in_file.close();
		
		// Add this template to the store
		
		thePixelTemp.push_back(theCurrentTemp);
		
		return true;
		
	} else {
		
		// If file didn't open, report this
		
		LOGERROR("SiPixelTemplate") << "Error opening File" << tempfile << ENDL;
		return false;
		
	}
	
} // TempInit 


#ifndef SI_PIXEL_TEMPLATE_STANDALONE

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
	float qavg_avg;
	//	const char *tempfile;
	const int code_version={16};
	
	// We must create a new object because dbobject must be a const and our stream must not be
	SiPixelTemplateDBObject db = dbobject;
	
	// Create a local template storage entry
	SiPixelTemplateStore theCurrentTemp;
	
	// Fill the template storage for each template calibration stored in the db
	for(int m=0; m<db.numOfTempl(); ++m)
	{
		
		// Read-in a header string first and print it    
		
		SiPixelTemplateDBObject::char2float temp;
		for (i=0; i<20; ++i) {
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
		
		db >> theCurrentTemp.head.ID  >> theCurrentTemp.head.templ_version >> theCurrentTemp.head.Bfield >> theCurrentTemp.head.NTy >> theCurrentTemp.head.NTyx >> theCurrentTemp.head.NTxx
		>> theCurrentTemp.head.Dtype >> theCurrentTemp.head.Vbias >> theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >> theCurrentTemp.head.qscale
		>> theCurrentTemp.head.s50 >> theCurrentTemp.head.lorywidth >> theCurrentTemp.head.lorxwidth >> theCurrentTemp.head.ysize >> theCurrentTemp.head.xsize >> theCurrentTemp.head.zsize;
		
		if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
		
		LOGINFO("SiPixelTemplate") << "Template ID = " << theCurrentTemp.head.ID << ", Template Version " << theCurrentTemp.head.templ_version << ", Bfield = " << theCurrentTemp.head.Bfield 
		<< ", NTy = " << theCurrentTemp.head.NTy << ", NTyx = " << theCurrentTemp.head.NTyx<< ", NTxx = " << theCurrentTemp.head.NTxx << ", Dtype = " << theCurrentTemp.head.Dtype
		<< ", Bias voltage " << theCurrentTemp.head.Vbias << ", temperature "
		<< theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence << ", Q-scaling factor " << theCurrentTemp.head.qscale
		<< ", 1/2 threshold " << theCurrentTemp.head.s50 << ", y Lorentz Width " << theCurrentTemp.head.lorywidth << ", x Lorentz width " << theCurrentTemp.head.lorxwidth    
		<< ", pixel x-size " << theCurrentTemp.head.xsize << ", y-size " << theCurrentTemp.head.ysize << ", zsize " << theCurrentTemp.head.zsize << ENDL;
		
		if(theCurrentTemp.head.templ_version < code_version) {LOGERROR("SiPixelTemplate") << "code expects version " << code_version << ", no template load" << ENDL; return false;}
		
		// next, loop over all barrel y-angle entries   
		
		for (i=0; i < theCurrentTemp.head.NTy; ++i) {     
			
			db >> theCurrentTemp.enty[i].runnum >> theCurrentTemp.enty[i].costrk[0] 
			>> theCurrentTemp.enty[i].costrk[1] >> theCurrentTemp.enty[i].costrk[2]; 
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 1, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			// Calculate the alpha, beta, and cot(beta) for this entry 
			
			theCurrentTemp.enty[i].alpha = static_cast<float>(atan2((double)theCurrentTemp.enty[i].costrk[2], (double)theCurrentTemp.enty[i].costrk[0]));
			
			theCurrentTemp.enty[i].cotalpha = theCurrentTemp.enty[i].costrk[0]/theCurrentTemp.enty[i].costrk[2];
			
			theCurrentTemp.enty[i].beta = static_cast<float>(atan2((double)theCurrentTemp.enty[i].costrk[2], (double)theCurrentTemp.enty[i].costrk[1]));
			
			theCurrentTemp.enty[i].cotbeta = theCurrentTemp.enty[i].costrk[1]/theCurrentTemp.enty[i].costrk[2];
			
			db >> theCurrentTemp.enty[i].qavg >> theCurrentTemp.enty[i].pixmax >> theCurrentTemp.enty[i].symax >> theCurrentTemp.enty[i].dyone
			>> theCurrentTemp.enty[i].syone >> theCurrentTemp.enty[i].sxmax >> theCurrentTemp.enty[i].dxone >> theCurrentTemp.enty[i].sxone;
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 2, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			db >> theCurrentTemp.enty[i].dytwo >> theCurrentTemp.enty[i].sytwo >> theCurrentTemp.enty[i].dxtwo 
			>> theCurrentTemp.enty[i].sxtwo >> theCurrentTemp.enty[i].qmin >> theCurrentTemp.enty[i].clsleny >> theCurrentTemp.enty[i].clslenx;
			//			     >> theCurrentTemp.enty[i].mpvvav >> theCurrentTemp.enty[i].sigmavav >> theCurrentTemp.enty[i].kappavav;
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 3, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			for (j=0; j<2; ++j) {
				
				db >> theCurrentTemp.enty[i].ypar[j][0] >> theCurrentTemp.enty[i].ypar[j][1] 
				>> theCurrentTemp.enty[i].ypar[j][2] >> theCurrentTemp.enty[i].ypar[j][3] >> theCurrentTemp.enty[i].ypar[j][4];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 4, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
				
			}
			
			for (j=0; j<9; ++j) {
				
				for (k=0; k<TYSIZE; ++k) {db >> theCurrentTemp.enty[i].ytemp[j][k];}
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 5, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<2; ++j) {
				
				db >> theCurrentTemp.enty[i].xpar[j][0] >> theCurrentTemp.enty[i].xpar[j][1] 
				>> theCurrentTemp.enty[i].xpar[j][2] >> theCurrentTemp.enty[i].xpar[j][3] >> theCurrentTemp.enty[i].xpar[j][4];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 6, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
				
			}
			
			qavg_avg = 0.;
			for (j=0; j<9; ++j) {
				
				for (k=0; k<TXSIZE; ++k) {db >> theCurrentTemp.enty[i].xtemp[j][k]; qavg_avg += theCurrentTemp.enty[i].xtemp[j][k];} 
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 7, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			theCurrentTemp.enty[i].qavg_avg = qavg_avg/9.;
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].yavg[j] >> theCurrentTemp.enty[i].yrms[j] >> theCurrentTemp.enty[i].ygx0[j] >> theCurrentTemp.enty[i].ygsig[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 8, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].yflpar[j][0] >> theCurrentTemp.enty[i].yflpar[j][1] >> theCurrentTemp.enty[i].yflpar[j][2] 
				>> theCurrentTemp.enty[i].yflpar[j][3] >> theCurrentTemp.enty[i].yflpar[j][4] >> theCurrentTemp.enty[i].yflpar[j][5];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 9, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].xavg[j] >> theCurrentTemp.enty[i].xrms[j] >> theCurrentTemp.enty[i].xgx0[j] >> theCurrentTemp.enty[i].xgsig[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 10, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].xflpar[j][0] >> theCurrentTemp.enty[i].xflpar[j][1] >> theCurrentTemp.enty[i].xflpar[j][2] 
				>> theCurrentTemp.enty[i].xflpar[j][3] >> theCurrentTemp.enty[i].xflpar[j][4] >> theCurrentTemp.enty[i].xflpar[j][5];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 11, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].chi2yavg[j] >> theCurrentTemp.enty[i].chi2ymin[j] >> theCurrentTemp.enty[i].chi2xavg[j] >> theCurrentTemp.enty[i].chi2xmin[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 12, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].yavgc2m[j] >> theCurrentTemp.enty[i].yrmsc2m[j] >> theCurrentTemp.enty[i].ygx0c2m[j] >> theCurrentTemp.enty[i].ygsigc2m[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 13, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].xavgc2m[j] >> theCurrentTemp.enty[i].xrmsc2m[j] >> theCurrentTemp.enty[i].xgx0c2m[j] >> theCurrentTemp.enty[i].xgsigc2m[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 14, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			} 
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].yavggen[j] >> theCurrentTemp.enty[i].yrmsgen[j] >> theCurrentTemp.enty[i].ygx0gen[j] >> theCurrentTemp.enty[i].ygsiggen[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 14a, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].xavggen[j] >> theCurrentTemp.enty[i].xrmsgen[j] >> theCurrentTemp.enty[i].xgx0gen[j] >> theCurrentTemp.enty[i].xgsiggen[j];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 14b, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			} 
			
			
			db >> theCurrentTemp.enty[i].chi2yavgone >> theCurrentTemp.enty[i].chi2yminone >> theCurrentTemp.enty[i].chi2xavgone >> theCurrentTemp.enty[i].chi2xminone >> theCurrentTemp.enty[i].qmin2
			>> theCurrentTemp.enty[i].mpvvav >> theCurrentTemp.enty[i].sigmavav >> theCurrentTemp.enty[i].kappavav >> theCurrentTemp.enty[i].qavg_spare >> theCurrentTemp.enty[i].spare[0];
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 15, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			db >> theCurrentTemp.enty[i].spare[1] >> theCurrentTemp.enty[i].spare[2] >> theCurrentTemp.enty[i].spare[3] >> theCurrentTemp.enty[i].qbfrac[0] >> theCurrentTemp.enty[i].qbfrac[1]
			>> theCurrentTemp.enty[i].qbfrac[2] >> theCurrentTemp.enty[i].fracyone >> theCurrentTemp.enty[i].fracxone >> theCurrentTemp.enty[i].fracytwo >> theCurrentTemp.enty[i].fracxtwo;
			//			theCurrentTemp.enty[i].qbfrac[3] = 1. - theCurrentTemp.enty[i].qbfrac[0] - theCurrentTemp.enty[i].qbfrac[1] - theCurrentTemp.enty[i].qbfrac[2];
			
			if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 16, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
		}
		
		// next, loop over all barrel x-angle entries   
		
		for (k=0; k < theCurrentTemp.head.NTyx; ++k) { 
			
			for (i=0; i < theCurrentTemp.head.NTxx; ++i) { 
				
				db >> theCurrentTemp.entx[k][i].runnum >> theCurrentTemp.entx[k][i].costrk[0] 
				>> theCurrentTemp.entx[k][i].costrk[1] >> theCurrentTemp.entx[k][i].costrk[2]; 
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 17, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				// Calculate the alpha, beta, and cot(beta) for this entry 
				
				theCurrentTemp.entx[k][i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entx[k][i].costrk[2], (double)theCurrentTemp.entx[k][i].costrk[0]));
				
				theCurrentTemp.entx[k][i].cotalpha = theCurrentTemp.entx[k][i].costrk[0]/theCurrentTemp.entx[k][i].costrk[2];
				
				theCurrentTemp.entx[k][i].beta = static_cast<float>(atan2((double)theCurrentTemp.entx[k][i].costrk[2], (double)theCurrentTemp.entx[k][i].costrk[1]));
				
				theCurrentTemp.entx[k][i].cotbeta = theCurrentTemp.entx[k][i].costrk[1]/theCurrentTemp.entx[k][i].costrk[2];
				
				db >> theCurrentTemp.entx[k][i].qavg >> theCurrentTemp.entx[k][i].pixmax >> theCurrentTemp.entx[k][i].symax >> theCurrentTemp.entx[k][i].dyone
				>> theCurrentTemp.entx[k][i].syone >> theCurrentTemp.entx[k][i].sxmax >> theCurrentTemp.entx[k][i].dxone >> theCurrentTemp.entx[k][i].sxone;
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 18, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				db >> theCurrentTemp.entx[k][i].dytwo >> theCurrentTemp.entx[k][i].sytwo >> theCurrentTemp.entx[k][i].dxtwo 
				>> theCurrentTemp.entx[k][i].sxtwo >> theCurrentTemp.entx[k][i].qmin >> theCurrentTemp.entx[k][i].clsleny >> theCurrentTemp.entx[k][i].clslenx;
				//                     >> theCurrentTemp.entx[k][i].mpvvav >> theCurrentTemp.entx[k][i].sigmavav >> theCurrentTemp.entx[k][i].kappavav;
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 19, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				for (j=0; j<2; ++j) {
					
					db >> theCurrentTemp.entx[k][i].ypar[j][0] >> theCurrentTemp.entx[k][i].ypar[j][1] 
					>> theCurrentTemp.entx[k][i].ypar[j][2] >> theCurrentTemp.entx[k][i].ypar[j][3] >> theCurrentTemp.entx[k][i].ypar[j][4];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 20, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<9; ++j) {
					
					for (l=0; l<TYSIZE; ++l) {db >> theCurrentTemp.entx[k][i].ytemp[j][l];} 
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 21, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<2; ++j) {
					
					db >> theCurrentTemp.entx[k][i].xpar[j][0] >> theCurrentTemp.entx[k][i].xpar[j][1] 
					>> theCurrentTemp.entx[k][i].xpar[j][2] >> theCurrentTemp.entx[k][i].xpar[j][3] >> theCurrentTemp.entx[k][i].xpar[j][4];
					
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 22, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				qavg_avg = 0.;
				for (j=0; j<9; ++j) {
					
					for (l=0; l<TXSIZE; ++l) {db >> theCurrentTemp.entx[k][i].xtemp[j][l]; qavg_avg += theCurrentTemp.entx[k][i].xtemp[j][l];} 
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 23, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				theCurrentTemp.entx[k][i].qavg_avg = qavg_avg/9.;
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].yavg[j] >> theCurrentTemp.entx[k][i].yrms[j] >> theCurrentTemp.entx[k][i].ygx0[j] >> theCurrentTemp.entx[k][i].ygsig[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 24, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].yflpar[j][0] >> theCurrentTemp.entx[k][i].yflpar[j][1] >> theCurrentTemp.entx[k][i].yflpar[j][2] 
					>> theCurrentTemp.entx[k][i].yflpar[j][3] >> theCurrentTemp.entx[k][i].yflpar[j][4] >> theCurrentTemp.entx[k][i].yflpar[j][5];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 25, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].xavg[j] >> theCurrentTemp.entx[k][i].xrms[j] >> theCurrentTemp.entx[k][i].xgx0[j] >> theCurrentTemp.entx[k][i].xgsig[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 26, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].xflpar[j][0] >> theCurrentTemp.entx[k][i].xflpar[j][1] >> theCurrentTemp.entx[k][i].xflpar[j][2] 
					>> theCurrentTemp.entx[k][i].xflpar[j][3] >> theCurrentTemp.entx[k][i].xflpar[j][4] >> theCurrentTemp.entx[k][i].xflpar[j][5];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 27, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].chi2yavg[j] >> theCurrentTemp.entx[k][i].chi2ymin[j] >> theCurrentTemp.entx[k][i].chi2xavg[j] >> theCurrentTemp.entx[k][i].chi2xmin[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 28, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].yavgc2m[j] >> theCurrentTemp.entx[k][i].yrmsc2m[j] >> theCurrentTemp.entx[k][i].ygx0c2m[j] >> theCurrentTemp.entx[k][i].ygsigc2m[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 29, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].xavgc2m[j] >> theCurrentTemp.entx[k][i].xrmsc2m[j] >> theCurrentTemp.entx[k][i].xgx0c2m[j] >> theCurrentTemp.entx[k][i].xgsigc2m[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 30, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].yavggen[j] >> theCurrentTemp.entx[k][i].yrmsgen[j] >> theCurrentTemp.entx[k][i].ygx0gen[j] >> theCurrentTemp.entx[k][i].ygsiggen[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 30a, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].xavggen[j] >> theCurrentTemp.entx[k][i].xrmsgen[j] >> theCurrentTemp.entx[k][i].xgx0gen[j] >> theCurrentTemp.entx[k][i].xgsiggen[j];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 30b, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				
				db >> theCurrentTemp.entx[k][i].chi2yavgone >> theCurrentTemp.entx[k][i].chi2yminone >> theCurrentTemp.entx[k][i].chi2xavgone >> theCurrentTemp.entx[k][i].chi2xminone >> theCurrentTemp.entx[k][i].qmin2
				>> theCurrentTemp.entx[k][i].mpvvav >> theCurrentTemp.entx[k][i].sigmavav >> theCurrentTemp.entx[k][i].kappavav >> theCurrentTemp.entx[k][i].qavg_spare >> theCurrentTemp.entx[k][i].spare[0];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 31, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				db >> theCurrentTemp.entx[k][i].spare[1] >> theCurrentTemp.entx[k][i].spare[2] >> theCurrentTemp.entx[k][i].spare[3] >> theCurrentTemp.entx[k][i].qbfrac[0] >> theCurrentTemp.entx[k][i].qbfrac[1]
				>> theCurrentTemp.entx[k][i].qbfrac[2] >> theCurrentTemp.entx[k][i].fracyone >> theCurrentTemp.entx[k][i].fracxone >> theCurrentTemp.entx[k][i].fracytwo >> theCurrentTemp.entx[k][i].fracxtwo;
				//				theCurrentTemp.entx[k][i].qbfrac[3] = 1. - theCurrentTemp.entx[k][i].qbfrac[0] - theCurrentTemp.entx[k][i].qbfrac[1] - theCurrentTemp.entx[k][i].qbfrac[2];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file 32, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
			}
		}	
		
		
		// Add this template to the store
		
		thePixelTemp.push_back(theCurrentTemp);
		
	}
	return true;
	
} // TempInit 

#endif


// ************************************************************************************************************ 
//! Interpolate input alpha and beta angles to produce a working template for each individual hit. 
//! \param id - (input) index of the template to use
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param locBz - (input) the sign of this quantity is used to determine whether to flip cot(beta)<0 quantities from cot(beta)>0 (FPix only)
//!                    for FPix IP-related tracks, locBz < 0 for cot(beta) > 0 and locBz > 0 for cot(beta) < 0
// ************************************************************************************************************ 
bool SiPixelTemplate::interpolate(int id, float cotalpha, float cotbeta, float locBz)
{
    // Interpolate for a new set of track angles 
    
    // Local variables 
    int i, j;
	int ilow, ihigh, iylow, iyhigh, Ny, Nxx, Nyx, imidy, imaxx;
	float yratio, yxratio, xxratio, sxmax, qcorrect, qxtempcor, symax, chi2xavgone, chi2xminone, cotb, cotalpha0, cotbeta0;
	bool flip_y;
//	std::vector <float> xrms(4), xgsig(4), xrmsc2m(4), xgsigc2m(4);
	std::vector <float> chi2xavg(4), chi2xmin(4);


// Check to see if interpolation is valid     

if(id != id_current || cotalpha != cota_current || cotbeta != cotb_current) {

	cota_current = cotalpha; cotb_current = cotbeta; success = true;
	
	if(id != id_current) {

// Find the index corresponding to id

       index_id = -1;
       for(i=0; i<(int)thePixelTemp.size(); ++i) {
	
	      if(id == thePixelTemp[i].head.ID) {
	   
	         index_id = i;
		      id_current = id;
				
// Copy the charge scaling factor to the private variable     
				
				pqscale = thePixelTemp[index_id].head.qscale;
				
// Copy the pseudopixel signal size to the private variable     
				
				ps50 = thePixelTemp[index_id].head.s50;
				
// Pixel sizes to the private variables     
				
				pxsize = thePixelTemp[index_id].head.xsize;
				pysize = thePixelTemp[index_id].head.ysize;
				pzsize = thePixelTemp[index_id].head.zsize;
				
				break;
          }
	    }
     }
	 
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if(index_id < 0 || index_id >= (int)thePixelTemp.size()) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::interpolate can't find needed template ID = " << id << std::endl;
	}
#else
	assert(index_id >= 0 && index_id < (int)thePixelTemp.size());
#endif
	 
// Interpolate the absolute value of cot(beta)     
    
    abs_cotb = fabs((double)cotbeta);
	
//	qcorrect corrects the cot(alpha)=0 cluster charge for non-zero cot(alpha)	

	cotalpha0 =  thePixelTemp[index_id].enty[0].cotalpha;
    qcorrect=(float)sqrt((double)((1.+cotbeta*cotbeta+cotalpha*cotalpha)/(1.+cotbeta*cotbeta+cotalpha0*cotalpha0)));
	
// for some cosmics, the ususal gymnastics are incorrect   
	if(thePixelTemp[index_id].head.Dtype == 0) {
		cotb = abs_cotb;
		flip_y = false;
		if(cotbeta < 0.) {flip_y = true;}
	} else {
	    if(locBz < 0.) {
			cotb = cotbeta;
			flip_y = false;
		} else {
			cotb = -cotbeta;
			flip_y = true;
		}	
	}
		
	Ny = thePixelTemp[index_id].head.NTy;
	Nyx = thePixelTemp[index_id].head.NTyx;
	Nxx = thePixelTemp[index_id].head.NTxx;
		
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(Ny < 2 || Nyx < 1 || Nxx < 2) {
		throw cms::Exception("DataCorrupt") << "template ID = " << id_current << "has too few entries: Ny/Nyx/Nxx = " << Ny << "/" << Nyx << "/" << Nxx << std::endl;
	}
#else
	assert(Ny > 1 && Nyx > 0 && Nxx > 1);
#endif
	imaxx = Nyx - 1;
	imidy = Nxx/2;
        
// next, loop over all y-angle entries   

	ilow = 0;
	yratio = 0.;

	if(cotb >= thePixelTemp[index_id].enty[Ny-1].cotbeta) {
	
		ilow = Ny-2;
		yratio = 1.;
		success = false;
		
	} else {
	   
		if(cotb >= thePixelTemp[index_id].enty[0].cotbeta) {

			for (i=0; i<Ny-1; ++i) { 
    
			if( thePixelTemp[index_id].enty[i].cotbeta <= cotb && cotb < thePixelTemp[index_id].enty[i+1].cotbeta) {
		  
				ilow = i;
				yratio = (cotb - thePixelTemp[index_id].enty[i].cotbeta)/(thePixelTemp[index_id].enty[i+1].cotbeta - thePixelTemp[index_id].enty[i].cotbeta);
				break;			 
			}
		}
	} else { success = false; }
  }
	
	ihigh=ilow + 1;
			  
// Interpolate/store all y-related quantities (flip displacements when flip_y)

	pyratio = yratio;
	pqavg = (1. - yratio)*thePixelTemp[index_id].enty[ilow].qavg + yratio*thePixelTemp[index_id].enty[ihigh].qavg;
	pqavg *= qcorrect;
	symax = (1. - yratio)*thePixelTemp[index_id].enty[ilow].symax + yratio*thePixelTemp[index_id].enty[ihigh].symax;
	psyparmax = symax;
	sxmax = (1. - yratio)*thePixelTemp[index_id].enty[ilow].sxmax + yratio*thePixelTemp[index_id].enty[ihigh].sxmax;
	pdyone = (1. - yratio)*thePixelTemp[index_id].enty[ilow].dyone + yratio*thePixelTemp[index_id].enty[ihigh].dyone;
	if(flip_y) {pdyone = -pdyone;}
	psyone = (1. - yratio)*thePixelTemp[index_id].enty[ilow].syone + yratio*thePixelTemp[index_id].enty[ihigh].syone;
	pdytwo = (1. - yratio)*thePixelTemp[index_id].enty[ilow].dytwo + yratio*thePixelTemp[index_id].enty[ihigh].dytwo;
	if(flip_y) {pdytwo = -pdytwo;}
	psytwo = (1. - yratio)*thePixelTemp[index_id].enty[ilow].sytwo + yratio*thePixelTemp[index_id].enty[ihigh].sytwo;
	pqmin = (1. - yratio)*thePixelTemp[index_id].enty[ilow].qmin + yratio*thePixelTemp[index_id].enty[ihigh].qmin;
	pqmin *= qcorrect;
	pqmin2 = (1. - yratio)*thePixelTemp[index_id].enty[ilow].qmin2 + yratio*thePixelTemp[index_id].enty[ihigh].qmin2;
	pqmin2 *= qcorrect;
	pmpvvav = (1. - yratio)*thePixelTemp[index_id].enty[ilow].mpvvav + yratio*thePixelTemp[index_id].enty[ihigh].mpvvav;
	pmpvvav *= qcorrect;
	psigmavav = (1. - yratio)*thePixelTemp[index_id].enty[ilow].sigmavav + yratio*thePixelTemp[index_id].enty[ihigh].sigmavav;
	pkappavav = (1. - yratio)*thePixelTemp[index_id].enty[ilow].kappavav + yratio*thePixelTemp[index_id].enty[ihigh].kappavav;
	pclsleny = fminf(thePixelTemp[index_id].enty[ilow].clsleny, thePixelTemp[index_id].enty[ihigh].clsleny);
	pqavg_avg = (1. - yratio)*thePixelTemp[index_id].enty[ilow].qavg_avg + yratio*thePixelTemp[index_id].enty[ihigh].qavg_avg;
	pqavg_avg *= qcorrect;
	for(i=0; i<2 ; ++i) {
		for(j=0; j<5 ; ++j) {
// Charge loss switches sides when cot(beta) changes sign
			if(flip_y) {
	            pyparl[1-i][j] = thePixelTemp[index_id].enty[ilow].ypar[i][j];
	            pyparh[1-i][j] = thePixelTemp[index_id].enty[ihigh].ypar[i][j];
			} else {
	            pyparl[i][j] = thePixelTemp[index_id].enty[ilow].ypar[i][j];
	            pyparh[i][j] = thePixelTemp[index_id].enty[ihigh].ypar[i][j];
			}
			pxparly0[i][j] = thePixelTemp[index_id].enty[ilow].xpar[i][j];
			pxparhy0[i][j] = thePixelTemp[index_id].enty[ihigh].xpar[i][j];
		}
	}
	for(i=0; i<4; ++i) {
		pyavg[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].yavg[i] + yratio*thePixelTemp[index_id].enty[ihigh].yavg[i];
		if(flip_y) {pyavg[i] = -pyavg[i];}
		pyrms[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].yrms[i] + yratio*thePixelTemp[index_id].enty[ihigh].yrms[i];
//	      pygx0[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].ygx0[i] + yratio*thePixelTemp[index_id].enty[ihigh].ygx0[i];
//	      if(flip_y) {pygx0[i] = -pygx0[i];}
//	      pygsig[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].ygsig[i] + yratio*thePixelTemp[index_id].enty[ihigh].ygsig[i];
//	      xrms[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].xrms[i] + yratio*thePixelTemp[index_id].enty[ihigh].xrms[i];
//	      xgsig[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].xgsig[i] + yratio*thePixelTemp[index_id].enty[ihigh].xgsig[i];
		pchi2yavg[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].chi2yavg[i] + yratio*thePixelTemp[index_id].enty[ihigh].chi2yavg[i];
		pchi2ymin[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].chi2ymin[i] + yratio*thePixelTemp[index_id].enty[ihigh].chi2ymin[i];
		chi2xavg[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].chi2xavg[i] + yratio*thePixelTemp[index_id].enty[ihigh].chi2xavg[i];
		chi2xmin[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].chi2xmin[i] + yratio*thePixelTemp[index_id].enty[ihigh].chi2xmin[i];
		pyavgc2m[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].yavgc2m[i] + yratio*thePixelTemp[index_id].enty[ihigh].yavgc2m[i];
		if(flip_y) {pyavgc2m[i] = -pyavgc2m[i];}
	      pyrmsc2m[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].yrmsc2m[i] + yratio*thePixelTemp[index_id].enty[ihigh].yrmsc2m[i];
//	      pygx0c2m[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].ygx0c2m[i] + yratio*thePixelTemp[index_id].enty[ihigh].ygx0c2m[i];
//	      if(flip_y) {pygx0c2m[i] = -pygx0c2m[i];}
//	      pygsigc2m[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].ygsigc2m[i] + yratio*thePixelTemp[index_id].enty[ihigh].ygsigc2m[i];
//	      xrmsc2m[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].xrmsc2m[i] + yratio*thePixelTemp[index_id].enty[ihigh].xrmsc2m[i];
//	      xgsigc2m[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].xgsigc2m[i] + yratio*thePixelTemp[index_id].enty[ihigh].xgsigc2m[i];
		for(j=0; j<6 ; ++j) {
			pyflparl[i][j] = thePixelTemp[index_id].enty[ilow].yflpar[i][j];
			pyflparh[i][j] = thePixelTemp[index_id].enty[ihigh].yflpar[i][j];
			 
// Since Q_fl is odd under cotbeta, it flips qutomatically, change only even terms

			if(flip_y && (j == 0 || j == 2 || j == 4)) {
			   pyflparl[i][j] = - pyflparl[i][j];
			   pyflparh[i][j] = - pyflparh[i][j];
			}
		}
	}
	   
//// Do the spares next

	pchi2yavgone=(1. - yratio)*thePixelTemp[index_id].enty[ilow].chi2yavgone + yratio*thePixelTemp[index_id].enty[ihigh].chi2yavgone;
	pchi2yminone=(1. - yratio)*thePixelTemp[index_id].enty[ilow].chi2yminone + yratio*thePixelTemp[index_id].enty[ihigh].chi2yminone;
	chi2xavgone=(1. - yratio)*thePixelTemp[index_id].enty[ilow].chi2xavgone + yratio*thePixelTemp[index_id].enty[ihigh].chi2xavgone;
	chi2xminone=(1. - yratio)*thePixelTemp[index_id].enty[ilow].chi2xminone + yratio*thePixelTemp[index_id].enty[ihigh].chi2xminone;
		//       for(i=0; i<10; ++i) {
//		    pyspare[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].yspare[i] + yratio*thePixelTemp[index_id].enty[ihigh].yspare[i];
//       }
			  
// Interpolate and build the y-template 
	
	for(i=0; i<9; ++i) {
		pytemp[i][0] = 0.;
		pytemp[i][1] = 0.;
		pytemp[i][BYM2] = 0.;
		pytemp[i][BYM1] = 0.;
		for(j=0; j<TYSIZE; ++j) {
		  
// Flip the basic y-template when the cotbeta is negative

			if(flip_y) {
			   pytemp[8-i][BYM3-j]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].ytemp[i][j] + yratio*thePixelTemp[index_id].enty[ihigh].ytemp[i][j];
			} else {
			   pytemp[i][j+2]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].ytemp[i][j] + yratio*thePixelTemp[index_id].enty[ihigh].ytemp[i][j];
			}
		}
	}
	
// next, loop over all x-angle entries, first, find relevant y-slices   
	
	iylow = 0;
	yxratio = 0.;

	if(abs_cotb >= thePixelTemp[index_id].entx[Nyx-1][0].cotbeta) {
	
		iylow = Nyx-2;
		yxratio = 1.;
		
	} else if(abs_cotb >= thePixelTemp[index_id].entx[0][0].cotbeta) {

		for (i=0; i<Nyx-1; ++i) { 
    
			if( thePixelTemp[index_id].entx[i][0].cotbeta <= abs_cotb && abs_cotb < thePixelTemp[index_id].entx[i+1][0].cotbeta) {
		  
			   iylow = i;
			   yxratio = (abs_cotb - thePixelTemp[index_id].entx[i][0].cotbeta)/(thePixelTemp[index_id].entx[i+1][0].cotbeta - thePixelTemp[index_id].entx[i][0].cotbeta);
			   break;			 
			}
		}
	}
	
	iyhigh=iylow + 1;

	ilow = 0;
	xxratio = 0.;

	if(cotalpha >= thePixelTemp[index_id].entx[0][Nxx-1].cotalpha) {
	
		ilow = Nxx-2;
		xxratio = 1.;
		success = false;
		
	} else {
	   
		if(cotalpha >= thePixelTemp[index_id].entx[0][0].cotalpha) {

			for (i=0; i<Nxx-1; ++i) { 
    
			   if( thePixelTemp[index_id].entx[0][i].cotalpha <= cotalpha && cotalpha < thePixelTemp[index_id].entx[0][i+1].cotalpha) {
		  
				  ilow = i;
				  xxratio = (cotalpha - thePixelTemp[index_id].entx[0][i].cotalpha)/(thePixelTemp[index_id].entx[0][i+1].cotalpha - thePixelTemp[index_id].entx[0][i].cotalpha);
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

	psxparmax = (1. - xxratio)*thePixelTemp[index_id].entx[imaxx][ilow].sxmax + xxratio*thePixelTemp[index_id].entx[imaxx][ihigh].sxmax;
	psxmax = psxparmax;
	if(thePixelTemp[index_id].entx[imaxx][imidy].sxmax != 0.) {psxmax=psxmax/thePixelTemp[index_id].entx[imaxx][imidy].sxmax*sxmax;}
	psymax = (1. - xxratio)*thePixelTemp[index_id].entx[imaxx][ilow].symax + xxratio*thePixelTemp[index_id].entx[imaxx][ihigh].symax;
	if(thePixelTemp[index_id].entx[imaxx][imidy].symax != 0.) {psymax=psymax/thePixelTemp[index_id].entx[imaxx][imidy].symax*symax;}
	pdxone = (1. - xxratio)*thePixelTemp[index_id].entx[0][ilow].dxone + xxratio*thePixelTemp[index_id].entx[0][ihigh].dxone;
	psxone = (1. - xxratio)*thePixelTemp[index_id].entx[0][ilow].sxone + xxratio*thePixelTemp[index_id].entx[0][ihigh].sxone;
	pdxtwo = (1. - xxratio)*thePixelTemp[index_id].entx[0][ilow].dxtwo + xxratio*thePixelTemp[index_id].entx[0][ihigh].dxtwo;
	psxtwo = (1. - xxratio)*thePixelTemp[index_id].entx[0][ilow].sxtwo + xxratio*thePixelTemp[index_id].entx[0][ihigh].sxtwo;
	pclslenx = fminf(thePixelTemp[index_id].entx[0][ilow].clslenx, thePixelTemp[index_id].entx[0][ihigh].clslenx);
	for(i=0; i<2 ; ++i) {
		for(j=0; j<5 ; ++j) {
	         pxpar0[i][j] = thePixelTemp[index_id].entx[imaxx][imidy].xpar[i][j];
	         pxparl[i][j] = thePixelTemp[index_id].entx[imaxx][ilow].xpar[i][j];
	         pxparh[i][j] = thePixelTemp[index_id].entx[imaxx][ihigh].xpar[i][j];
		}
	}
	   		  
// pixmax is the maximum allowed pixel charge (used for truncation)

	ppixmax=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].pixmax + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].pixmax)
			+yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].pixmax + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].pixmax);
			  
	for(i=0; i<4; ++i) {
		pxavg[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].xavg[i] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].xavg[i])
				+yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xavg[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xavg[i]);
		  
		pxrms[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].xrms[i] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].xrms[i])
				+yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xrms[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xrms[i]);
		  
//	      pxgx0[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].xgx0[i] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].xgx0[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xgx0[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xgx0[i]);
							
//	      pxgsig[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].xgsig[i] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].xgsig[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xgsig[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xgsig[i]);
				  
		pxavgc2m[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].xavgc2m[i] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].xavgc2m[i])
				+yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xavgc2m[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xavgc2m[i]);
		  
		pxrmsc2m[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].xrmsc2m[i] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].xrmsc2m[i])
				+yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xrmsc2m[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xrmsc2m[i]);
		  
//	      pxgx0c2m[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].xgx0c2m[i] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].xgx0c2m[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xgx0c2m[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xgx0c2m[i]);
							
//	      pxgsigc2m[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].xgsigc2m[i] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].xgsigc2m[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xgsigc2m[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xgsigc2m[i]);
//
//  Try new interpolation scheme
//	  														
//	      pchi2xavg[i]=((1. - xxratio)*thePixelTemp[index_id].entx[imaxx][ilow].chi2xavg[i] + xxratio*thePixelTemp[index_id].entx[imaxx][ihigh].chi2xavg[i]);
//		  if(thePixelTemp[index_id].entx[imaxx][imidy].chi2xavg[i] != 0.) {pchi2xavg[i]=pchi2xavg[i]/thePixelTemp[index_id].entx[imaxx][imidy].chi2xavg[i]*chi2xavg[i];}
//							
//	      pchi2xmin[i]=((1. - xxratio)*thePixelTemp[index_id].entx[imaxx][ilow].chi2xmin[i] + xxratio*thePixelTemp[index_id].entx[imaxx][ihigh].chi2xmin[i]);
//		  if(thePixelTemp[index_id].entx[imaxx][imidy].chi2xmin[i] != 0.) {pchi2xmin[i]=pchi2xmin[i]/thePixelTemp[index_id].entx[imaxx][imidy].chi2xmin[i]*chi2xmin[i];}
//		  
		pchi2xavg[i]=((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].chi2xavg[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].chi2xavg[i]);
		if(thePixelTemp[index_id].entx[iyhigh][imidy].chi2xavg[i] != 0.) {pchi2xavg[i]=pchi2xavg[i]/thePixelTemp[index_id].entx[iyhigh][imidy].chi2xavg[i]*chi2xavg[i];}
							
		pchi2xmin[i]=((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].chi2xmin[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].chi2xmin[i]);
		if(thePixelTemp[index_id].entx[iyhigh][imidy].chi2xmin[i] != 0.) {pchi2xmin[i]=pchi2xmin[i]/thePixelTemp[index_id].entx[iyhigh][imidy].chi2xmin[i]*chi2xmin[i];}
		  
		for(j=0; j<6 ; ++j) {
	         pxflparll[i][j] = thePixelTemp[index_id].entx[iylow][ilow].xflpar[i][j];
	         pxflparlh[i][j] = thePixelTemp[index_id].entx[iylow][ihigh].xflpar[i][j];
	         pxflparhl[i][j] = thePixelTemp[index_id].entx[iyhigh][ilow].xflpar[i][j];
	         pxflparhh[i][j] = thePixelTemp[index_id].entx[iyhigh][ihigh].xflpar[i][j];
		}
	}
	   
// Do the spares next

	pchi2xavgone=((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].chi2xavgone + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].chi2xavgone);
	if(thePixelTemp[index_id].entx[iyhigh][imidy].chi2xavgone != 0.) {pchi2xavgone=pchi2xavgone/thePixelTemp[index_id].entx[iyhigh][imidy].chi2xavgone*chi2xavgone;}
		
	pchi2xminone=((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].chi2xminone + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].chi2xminone);
	if(thePixelTemp[index_id].entx[iyhigh][imidy].chi2xminone != 0.) {pchi2xminone=pchi2xminone/thePixelTemp[index_id].entx[iyhigh][imidy].chi2xminone*chi2xminone;}
		//       for(i=0; i<10; ++i) {
//	      pxspare[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].xspare[i] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].xspare[i])
//		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xspare[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xspare[i]);
//       }
			  
// Interpolate and build the x-template 
	
//	qxtempcor corrects the total charge to the actual track angles (not actually needed for the template fits, but useful for Guofan)
	
	cotbeta0 =  thePixelTemp[index_id].entx[iyhigh][0].cotbeta;
	qxtempcor=(float)sqrt((double)((1.+cotbeta*cotbeta+cotalpha*cotalpha)/(1.+cotbeta0*cotbeta0+cotalpha*cotalpha)));
	
	for(i=0; i<9; ++i) {
		pxtemp[i][0] = 0.;
		pxtemp[i][1] = 0.;
		pxtemp[i][BXM2] = 0.;
		pxtemp[i][BXM1] = 0.;
		for(j=0; j<TXSIZE; ++j) {
//  Take next largest x-slice for the x-template (it reduces bias in the forward direction after irradiation)
//		   pxtemp[i][j+2]=(1. - xxratio)*thePixelTemp[index_id].entx[imaxx][ilow].xtemp[i][j] + xxratio*thePixelTemp[index_id].entx[imaxx][ihigh].xtemp[i][j];
//		   pxtemp[i][j+2]=(1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xtemp[i][j] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xtemp[i][j];
         pxtemp[i][j+2]=qxtempcor*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xtemp[i][j] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xtemp[i][j]);
		}
	}
	
	plorywidth = thePixelTemp[index_id].head.lorywidth;
	if(locBz > 0.) {plorywidth = -plorywidth;}
	plorxwidth = thePixelTemp[index_id].head.lorxwidth;
	
  }
	
  return success;
} // interpolate





// ************************************************************************************************************ 
//! Interpolate input alpha and beta angles to produce a working template for each individual hit. 
//! \param id - (input) index of the template to use
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
// ************************************************************************************************************ 
bool SiPixelTemplate::interpolate(int id, float cotalpha, float cotbeta)
{
    // Interpolate for a new set of track angles 
    
    // Local variables 
    float locBz;
	locBz = -1.;
	if(cotbeta < 0.) {locBz = -locBz;}
    return SiPixelTemplate::interpolate(id, cotalpha, cotbeta, locBz);
}




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
    
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if(fypix < 2 || fypix >= BYM2) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ysigma2 called with fypix = " << fypix << std::endl;
	}
#else
	assert(fypix > 1 && fypix < BYM2);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	   if(lypix < fypix || lypix >= BYM2) {
		  throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ysigma2 called with lypix/fypix = " << lypix << "/" << fypix << std::endl;
		}
#else
		assert(lypix >= fypix && lypix < BYM2);
#endif
	   	     
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
			 ", cot(alpha) = " << cota_current << ", cot(beta) = " << cotb_current <<  ", sigi = " << sigi << ENDL;}
	      }
	   }
	
	return;
	
} // End ysigma2


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
	float sigi, sigi2, sigi3, sigi4, symax, qscale, err2;
	
    // Make sure that input is OK
    
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if(index < 2 || index >= BYM2) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ysigma2 called with index = " << index << std::endl;
	}
#else
	assert(index > 1 && index < BYM2);
#endif
	
	// Define the maximum signal to use in the parameterization 
	
	symax = psymax;
	if(psymax > psyparmax) {symax = psyparmax;}
	
	// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 
	
			if(qpixel < symax) {
				sigi = qpixel;
				qscale = 1.;
			} else {
				sigi = symax;
				qscale = qpixel/symax;
			}
			sigi2 = sigi*sigi; sigi3 = sigi2*sigi; sigi4 = sigi3*sigi;
			if(index <= BHY) {
				err2 = (1.-pyratio)*
				(pyparl[0][0]+pyparl[0][1]*sigi+pyparl[0][2]*sigi2+pyparl[0][3]*sigi3+pyparl[0][4]*sigi4)
				+ pyratio*
				(pyparh[0][0]+pyparh[0][1]*sigi+pyparh[0][2]*sigi2+pyparh[0][3]*sigi3+pyparh[0][4]*sigi4);
			} else {
				err2 = (1.-pyratio)*
				(pyparl[1][0]+pyparl[1][1]*sigi+pyparl[1][2]*sigi2+pyparl[1][3]*sigi3+pyparl[1][4]*sigi4)
				+ pyratio*
			    (pyparh[1][0]+pyparh[1][1]*sigi+pyparh[1][2]*sigi2+pyparh[1][3]*sigi3+pyparh[1][4]*sigi4);
			}
			ysig2 =qscale*err2;
			if(ysig2 <= 0.) {LOGERROR("SiPixelTemplate") << "neg y-error-squared, id = " << id_current << ", index = " << index_id << 
			", cot(alpha) = " << cota_current << ", cot(beta) = " << cotb_current <<  ", sigi = " << sigi << ENDL;}
	
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
    
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		  if(fxpix < 2 || fxpix >= BXM2) {
			 throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xsigma2 called with fxpix = " << fxpix << std::endl;
		   }
#else
		   assert(fxpix > 1 && fxpix < BXM2);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
			 if(lxpix < fxpix || lxpix >= BXM2) {
				throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xsigma2 called with lxpix/fxpix = " << lxpix << "/" << fxpix << std::endl;
			 }
#else
			 assert(lxpix >= fxpix && lxpix < BXM2);
#endif
	   	     
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
			 ", cot(alpha) = " << cota_current << ", cot(beta) = " << cotb_current  << ", sigi = " << sigi << ENDL;}
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
	float qfl, qfl2, qfl3, qfl4, qfl5, dy;
	
    // Make sure that input is OK
    
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(binq < 0 || binq > 3) {
	   throw cms::Exception("DataCorrupt") << "SiPixelTemplate::yflcorr called with binq = " << binq << std::endl;
	}
#else
	 assert(binq >= 0 && binq < 4);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	 if(fabs((double)qfly) > 1.) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::yflcorr called with qfly = " << qfly << std::endl;
	 }
#else
	 assert(fabs((double)qfly) <= 1.);
#endif
	   	     
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
	float qfl, qfl2, qfl3, qfl4, qfl5, dx;
	
    // Make sure that input is OK
    
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(binq < 0 || binq > 3) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xflcorr called with binq = " << binq << std::endl;
	}
#else
	assert(binq >= 0 && binq < 4);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(fabs((double)qflx) > 1.) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xflcorr called with qflx = " << qflx << std::endl;
	}
#else
	assert(fabs((double)qflx) <= 1.);
#endif
	   	     
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

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(fybin < 0 || fybin > 40) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ytemp called with fybin = " << fybin << std::endl;
	}
#else
	assert(fybin >= 0 && fybin < 41);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(lybin < 0 || lybin > 40) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ytemp called with lybin = " << lybin << std::endl;
	}
#else
	assert(lybin >= 0 && lybin < 41);
#endif

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

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(fxbin < 0 || fxbin > 40) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xtemp called with fxbin = " << fxbin << std::endl;
	}
#else
	assert(fxbin >= 0 && fxbin < 41);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(lxbin < 0 || lxbin > 40) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xtemp called with lxbin = " << lxbin << std::endl;
	}
#else
	assert(lxbin >= 0 && lxbin < 41);
#endif

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

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(nypix < 1 || nypix >= BYM3) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::ytemp3d called with nypix = " << nypix << std::endl;
	}
#else
	assert(nypix > 0 && nypix < BYM3);
#endif
	
// Calculate the size of the shift in pixels needed to span the entire cluster

    float diff = fabsf(nypix - pclsleny)/2. + 1.;
	int nshift = (int)diff;
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

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(nxpix < 1 || nxpix >= BXM3) {
	   throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xtemp3d called with nxpix = " << nxpix << std::endl;
	}
#else
	assert(nxpix > 0 && nxpix < BXM3);
#endif
	
// Calculate the size of the shift in pixels needed to span the entire cluster

    float diff = fabsf(nxpix - pclslenx)/2. + 1.;
	int nshift = (int)diff;
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
//! \param lorywidth - (output) the estimated y Lorentz width
//! \param lorxwidth - (output) the estimated x Lorentz width
// ************************************************************************************************************ 
int SiPixelTemplate::qbin(int id, float cotalpha, float cotbeta, float locBz, float qclus, float& pixmx, float& sigmay, float& deltay, float& sigmax, float& deltax, 
                          float& sy1, float& dy1, float& sy2, float& dy2, float& sx1, float& dx1, float& sx2, float& dx2, float& lorywidth, float& lorxwidth)
		 
{
    // Interpolate for a new set of track angles 
    
    // Local variables 
    int i, binq;
	int ilow, ihigh, iylow, iyhigh, Ny, Nxx, Nyx, imidy, imaxx;
	float yratio, yxratio, xxratio;
	float acotb, qscale, qavg, qmin, qmin2, fq, qtotal, qcorrect, cotb, cotalpha0;
	float yavggen[4], yrmsgen[4], xavggen[4], xrmsgen[4];
	bool flip_y;
	
	if(id != id_current) {

// Find the index corresponding to id

       index_id = -1;
       for(i=0; i<(int)thePixelTemp.size(); ++i) {
	
	      if(id == thePixelTemp[i].head.ID) {
	   
	         index_id = i;
		     id_current = id;
		     break;
          }
	    }
     }
	 
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	   if(index_id < 0 || index_id >= (int)thePixelTemp.size()) {
	      throw cms::Exception("DataCorrupt") << "SiPixelTemplate::qbin can't find needed template ID = " << id << std::endl;
	   }
#else
	   assert(index_id >= 0 && index_id < (int)thePixelTemp.size());
#endif
	 
//		

// Interpolate the absolute value of cot(beta)     
    
    acotb = fabs((double)cotbeta);
	
//	qcorrect corrects the cot(alpha)=0 cluster charge for non-zero cot(alpha)	

	//	qcorrect corrects the cot(alpha)=0 cluster charge for non-zero cot(alpha)	
	
	cotalpha0 =  thePixelTemp[index_id].enty[0].cotalpha;
    qcorrect=(float)sqrt((double)((1.+cotbeta*cotbeta+cotalpha*cotalpha)/(1.+cotbeta*cotbeta+cotalpha0*cotalpha0)));
				
	// for some cosmics, the ususal gymnastics are incorrect   
	
	if(thePixelTemp[index_id].head.Dtype == 0) {
		cotb = acotb;
		flip_y = false;
		if(cotbeta < 0.) {flip_y = true;}
	} else {
	    if(locBz < 0.) {
			cotb = cotbeta;
			flip_y = false;
		} else {
			cotb = -cotbeta;
			flip_y = true;
		}	
	}
	
	// Copy the charge scaling factor to the private variable     
		
	   qscale = thePixelTemp[index_id].head.qscale;
		
       Ny = thePixelTemp[index_id].head.NTy;
       Nyx = thePixelTemp[index_id].head.NTyx;
       Nxx = thePixelTemp[index_id].head.NTxx;
		
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		if(Ny < 2 || Nyx < 1 || Nxx < 2) {
			throw cms::Exception("DataCorrupt") << "template ID = " << id_current << "has too few entries: Ny/Nyx/Nxx = " << Ny << "/" << Nyx << "/" << Nxx << std::endl;
		}
#else
		assert(Ny > 1 && Nyx > 0 && Nxx > 1);
#endif
	   imaxx = Nyx - 1;
	   imidy = Nxx/2;
        
// next, loop over all y-angle entries   

	   ilow = 0;
	   yratio = 0.;

	   if(cotb >= thePixelTemp[index_id].enty[Ny-1].cotbeta) {
	
	       ilow = Ny-2;
		   yratio = 1.;
		
	   } else {
	   
	      if(cotb >= thePixelTemp[index_id].enty[0].cotbeta) {

             for (i=0; i<Ny-1; ++i) { 
    
                if( thePixelTemp[index_id].enty[i].cotbeta <= cotb && cotb < thePixelTemp[index_id].enty[i+1].cotbeta) {
		  
	               ilow = i;
		           yratio = (cotb - thePixelTemp[index_id].enty[i].cotbeta)/(thePixelTemp[index_id].enty[i+1].cotbeta - thePixelTemp[index_id].enty[i].cotbeta);
		           break;			 
		        }
	         }
		  } 
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all y-related quantities (flip displacements when flip_y)

	   qavg = (1. - yratio)*thePixelTemp[index_id].enty[ilow].qavg + yratio*thePixelTemp[index_id].enty[ihigh].qavg;
	   qavg *= qcorrect;
	   dy1 = (1. - yratio)*thePixelTemp[index_id].enty[ilow].dyone + yratio*thePixelTemp[index_id].enty[ihigh].dyone;
	   if(flip_y) {dy1 = -dy1;}
	   sy1 = (1. - yratio)*thePixelTemp[index_id].enty[ilow].syone + yratio*thePixelTemp[index_id].enty[ihigh].syone;
	   dy2 = (1. - yratio)*thePixelTemp[index_id].enty[ilow].dytwo + yratio*thePixelTemp[index_id].enty[ihigh].dytwo;
	   if(flip_y) {dy2 = -dy2;}
	   sy2 = (1. - yratio)*thePixelTemp[index_id].enty[ilow].sytwo + yratio*thePixelTemp[index_id].enty[ihigh].sytwo;
	   qmin = (1. - yratio)*thePixelTemp[index_id].enty[ilow].qmin + yratio*thePixelTemp[index_id].enty[ihigh].qmin;
	   qmin *= qcorrect;
	   qmin2 = (1. - yratio)*thePixelTemp[index_id].enty[ilow].qmin2 + yratio*thePixelTemp[index_id].enty[ihigh].qmin2;
	   qmin2 *= qcorrect;
	   for(i=0; i<4; ++i) {
	      yavggen[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].yavggen[i] + yratio*thePixelTemp[index_id].enty[ihigh].yavggen[i];
	      if(flip_y) {yavggen[i] = -yavggen[i];}
	      yrmsgen[i]=(1. - yratio)*thePixelTemp[index_id].enty[ilow].yrmsgen[i] + yratio*thePixelTemp[index_id].enty[ihigh].yrmsgen[i];
	   }
	   
	
// next, loop over all x-angle entries, first, find relevant y-slices   
	
	   iylow = 0;
	   yxratio = 0.;

	   if(acotb >= thePixelTemp[index_id].entx[Nyx-1][0].cotbeta) {
	
	       iylow = Nyx-2;
		   yxratio = 1.;
		
	   } else if(acotb >= thePixelTemp[index_id].entx[0][0].cotbeta) {

          for (i=0; i<Nyx-1; ++i) { 
    
             if( thePixelTemp[index_id].entx[i][0].cotbeta <= acotb && acotb < thePixelTemp[index_id].entx[i+1][0].cotbeta) {
		  
	            iylow = i;
		        yxratio = (acotb - thePixelTemp[index_id].entx[i][0].cotbeta)/(thePixelTemp[index_id].entx[i+1][0].cotbeta - thePixelTemp[index_id].entx[i][0].cotbeta);
		        break;			 
		     }
	      }
	   }
	
	   iyhigh=iylow + 1;

	   ilow = 0;
	   xxratio = 0.;

	   if(cotalpha >= thePixelTemp[index_id].entx[0][Nxx-1].cotalpha) {
	
	       ilow = Nxx-2;
		   xxratio = 1.;
		
	   } else {
	   
	      if(cotalpha >= thePixelTemp[index_id].entx[0][0].cotalpha) {

             for (i=0; i<Nxx-1; ++i) { 
    
                if( thePixelTemp[index_id].entx[0][i].cotalpha <= cotalpha && cotalpha < thePixelTemp[index_id].entx[0][i+1].cotalpha) {
		  
	               ilow = i;
		           xxratio = (cotalpha - thePixelTemp[index_id].entx[0][i].cotalpha)/(thePixelTemp[index_id].entx[0][i+1].cotalpha - thePixelTemp[index_id].entx[0][i].cotalpha);
		           break;
			    }
		     }
		  } 
	   }
	
	   ihigh=ilow + 1;
			  
	   dx1 = (1. - xxratio)*thePixelTemp[index_id].entx[0][ilow].dxone + xxratio*thePixelTemp[index_id].entx[0][ihigh].dxone;
	   sx1 = (1. - xxratio)*thePixelTemp[index_id].entx[0][ilow].sxone + xxratio*thePixelTemp[index_id].entx[0][ihigh].sxone;
	   dx2 = (1. - xxratio)*thePixelTemp[index_id].entx[0][ilow].dxtwo + xxratio*thePixelTemp[index_id].entx[0][ihigh].dxtwo;
	   sx2 = (1. - xxratio)*thePixelTemp[index_id].entx[0][ilow].sxtwo + xxratio*thePixelTemp[index_id].entx[0][ihigh].sxtwo;
	   		  
// pixmax is the maximum allowed pixel charge (used for truncation)

	   pixmx=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].pixmax + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].pixmax)
			  +yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].pixmax + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].pixmax);
			  
	   for(i=0; i<4; ++i) {
				  
	      xavggen[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].xavggen[i] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].xavggen[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xavggen[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xavggen[i]);
		  
	      xrmsgen[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].xrmsgen[i] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].xrmsgen[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xrmsgen[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xrmsgen[i]);		  
	   }
	   
		lorywidth = thePixelTemp[index_id].head.lorywidth;
	    if(locBz > 0.) {lorywidth = -lorywidth;}
		lorxwidth = thePixelTemp[index_id].head.lorxwidth;
		
	
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(qavg <= 0. || qmin <= 0.) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::qbin, qavg or qmin <= 0," 
		<< " Probably someone called the generic pixel reconstruction with an illegal trajectory state" << std::endl;
	}
#else
	assert(qavg > 0. && qmin > 0.);
#endif
	
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
	
//  Take the errors and bias from the correct charge bin
	
	sigmay = yrmsgen[binq]; deltay = yavggen[binq];
	
	sigmax = xrmsgen[binq]; deltax = xavggen[binq];
	
// If the charge is too small (then flag it)
	
	if(qtotal < 0.95*qmin) {binq = 5;} else {if(qtotal < 0.95*qmin2) {binq = 4;}}
		
    return binq;
  
} // qbin

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
int SiPixelTemplate::qbin(int id, float cotalpha, float cotbeta, float locBz, float qclus, float& pixmx, float& sigmay, float& deltay, float& sigmax, float& deltax, 
                          float& sy1, float& dy1, float& sy2, float& dy2, float& sx1, float& dx1, float& sx2, float& dx2)

{
	float lorywidth, lorxwidth;
	return SiPixelTemplate::qbin(id, cotalpha, cotbeta, locBz, qclus, pixmx, sigmay, deltay, sigmax, deltax, 
								 sy1, dy1, sy2, dy2, sx1, dx1, sx2, dx2, lorywidth, lorxwidth);
	
} // qbin

// ************************************************************************************************************ 
//! Interpolate beta/alpha angles to produce an expected average charge. Return int (0-4) describing the charge 
//! of the cluster [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin].
//! \param id - (input) index of the template to use
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param qclus - (input) the cluster charge in electrons 
// ************************************************************************************************************ 
int SiPixelTemplate::qbin(int id, float cotbeta, float qclus)
{
// Interpolate for a new set of track angles 
				
// Local variables 
	float pixmx, sigmay, deltay, sigmax, deltax, sy1, dy1, sy2, dy2, sx1, dx1, sx2, dx2, locBz, lorywidth, lorxwidth;
	const float cotalpha = 0.;
	locBz = -1.;
	if(cotbeta < 0.) {locBz = -locBz;}
	return SiPixelTemplate::qbin(id, cotalpha, cotbeta, locBz, qclus, pixmx, sigmay, deltay, sigmax, deltax, 
								sy1, dy1, sy2, dy2, sx1, dx1, sx2, dx2, lorywidth, lorxwidth);
				
} // qbin
				


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
void SiPixelTemplate::temperrors(int id, float cotalpha, float cotbeta, int qBin, float& sigmay, float& sigmax, float& sy1, float& sy2, float& sx1, float& sx2)

{
    // Interpolate for a new set of track angles 
    
    // Local variables 
    int i;
	int ilow, ihigh, iylow, iyhigh, Ny, Nxx, Nyx, imidy, imaxx;
	float yratio, yxratio, xxratio;
	float acotb, cotb;
	float yrms, xrms;
	bool flip_y;
	
	if(id != id_current) {
		
		// Find the index corresponding to id
		
		index_id = -1;
		for(i=0; i<(int)thePixelTemp.size(); ++i) {
			
			if(id == thePixelTemp[i].head.ID) {
				
				index_id = i;
				id_current = id;
				break;
			}
	    }
	}
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(index_id < 0 || index_id >= (int)thePixelTemp.size()) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::temperrors can't find needed template ID = " << id << std::endl;
	}
#else
	assert(index_id >= 0 && index_id < (int)thePixelTemp.size());
#endif
	
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(qBin < 0 || qBin > 5) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::temperrors called with illegal qBin = " << qBin << std::endl;
	}
#else
	assert(qBin >= 0 && qBin < 6);
#endif
	
// The error information for qBin > 3 is taken to be the same as qBin=3	
	
	if(qBin > 3) {qBin = 3;}
	//		
	
	// Interpolate the absolute value of cot(beta)     
    
    acotb = fabs((double)cotbeta);
	cotb = cotbeta;
	
	// for some cosmics, the ususal gymnastics are incorrect   
	
//	if(thePixelTemp[index_id].head.Dtype == 0) {
		cotb = acotb;
		flip_y = false;
		if(cotbeta < 0.) {flip_y = true;}
//	} else {
//	    if(locBz < 0.) {
//			cotb = cotbeta;
//			flip_y = false;
//		} else {
//			cotb = -cotbeta;
//			flip_y = true;
//		}	
//	}
				
		// Copy the charge scaling factor to the private variable     
		
		Ny = thePixelTemp[index_id].head.NTy;
		Nyx = thePixelTemp[index_id].head.NTyx;
		Nxx = thePixelTemp[index_id].head.NTxx;
		
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		if(Ny < 2 || Nyx < 1 || Nxx < 2) {
			throw cms::Exception("DataCorrupt") << "template ID = " << id_current << "has too few entries: Ny/Nyx/Nxx = " << Ny << "/" << Nyx << "/" << Nxx << std::endl;
		}
#else
		assert(Ny > 1 && Nyx > 0 && Nxx > 1);
#endif
		imaxx = Nyx - 1;
		imidy = Nxx/2;
        
		// next, loop over all y-angle entries   
		
		ilow = 0;
		yratio = 0.;
		
		if(cotb >= thePixelTemp[index_id].enty[Ny-1].cotbeta) {
			
			ilow = Ny-2;
			yratio = 1.;
			
		} else {
			
			if(cotb >= thePixelTemp[index_id].enty[0].cotbeta) {
				
				for (i=0; i<Ny-1; ++i) { 
					
					if( thePixelTemp[index_id].enty[i].cotbeta <= cotb && cotb < thePixelTemp[index_id].enty[i+1].cotbeta) {
						
						ilow = i;
						yratio = (cotb - thePixelTemp[index_id].enty[i].cotbeta)/(thePixelTemp[index_id].enty[i+1].cotbeta - thePixelTemp[index_id].enty[i].cotbeta);
						break;			 
					}
				}
			} 
		}
		
		ihigh=ilow + 1;
		
		// Interpolate/store all y-related quantities (flip displacements when flip_y)
		
		sy1 = (1. - yratio)*thePixelTemp[index_id].enty[ilow].syone + yratio*thePixelTemp[index_id].enty[ihigh].syone;
		sy2 = (1. - yratio)*thePixelTemp[index_id].enty[ilow].sytwo + yratio*thePixelTemp[index_id].enty[ihigh].sytwo;
		yrms=(1. - yratio)*thePixelTemp[index_id].enty[ilow].yrms[qBin] + yratio*thePixelTemp[index_id].enty[ihigh].yrms[qBin];
		
		
		// next, loop over all x-angle entries, first, find relevant y-slices   
		
		iylow = 0;
		yxratio = 0.;
		
		if(acotb >= thePixelTemp[index_id].entx[Nyx-1][0].cotbeta) {
			
			iylow = Nyx-2;
			yxratio = 1.;
			
		} else if(acotb >= thePixelTemp[index_id].entx[0][0].cotbeta) {
			
			for (i=0; i<Nyx-1; ++i) { 
				
				if( thePixelTemp[index_id].entx[i][0].cotbeta <= acotb && acotb < thePixelTemp[index_id].entx[i+1][0].cotbeta) {
					
					iylow = i;
					yxratio = (acotb - thePixelTemp[index_id].entx[i][0].cotbeta)/(thePixelTemp[index_id].entx[i+1][0].cotbeta - thePixelTemp[index_id].entx[i][0].cotbeta);
					break;			 
				}
			}
		}
		
		iyhigh=iylow + 1;
		
		ilow = 0;
		xxratio = 0.;
		
		if(cotalpha >= thePixelTemp[index_id].entx[0][Nxx-1].cotalpha) {
			
			ilow = Nxx-2;
			xxratio = 1.;
			
		} else {
			
			if(cotalpha >= thePixelTemp[index_id].entx[0][0].cotalpha) {
				
				for (i=0; i<Nxx-1; ++i) { 
					
					if( thePixelTemp[index_id].entx[0][i].cotalpha <= cotalpha && cotalpha < thePixelTemp[index_id].entx[0][i+1].cotalpha) {
						
						ilow = i;
						xxratio = (cotalpha - thePixelTemp[index_id].entx[0][i].cotalpha)/(thePixelTemp[index_id].entx[0][i+1].cotalpha - thePixelTemp[index_id].entx[0][i].cotalpha);
						break;
					}
				}
			} 
		}
		
		ihigh=ilow + 1;
		
		sx1 = (1. - xxratio)*thePixelTemp[index_id].entx[0][ilow].sxone + xxratio*thePixelTemp[index_id].entx[0][ihigh].sxone;
		sx2 = (1. - xxratio)*thePixelTemp[index_id].entx[0][ilow].sxtwo + xxratio*thePixelTemp[index_id].entx[0][ihigh].sxtwo;
		
		xrms=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].xrms[qBin] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].xrms[qBin])
			+yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].xrms[qBin] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].xrms[qBin]);		  
		
	
	
	
	//  Take the errors and bias from the correct charge bin
	
	sigmay = yrms;
	
	sigmax = xrms;
		
    return;
	
} // temperrors

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
void SiPixelTemplate::qbin_dist(int id, float cotalpha, float cotbeta, float qbin_frac[4], float& ny1_frac, float& ny2_frac, float& nx1_frac, float& nx2_frac)

{
    // Interpolate for a new set of track angles 
    
    // Local variables 
    int i;
	int ilow, ihigh, iylow, iyhigh, Ny, Nxx, Nyx, imidy, imaxx;
	float yratio, yxratio, xxratio;
	float acotb, cotb;
	float qfrac[4];
	bool flip_y;
	
	if(id != id_current) {
		
		// Find the index corresponding to id
		
		index_id = -1;
		for(i=0; i<(int)thePixelTemp.size(); ++i) {
			
			if(id == thePixelTemp[i].head.ID) {
				
				index_id = i;
				id_current = id;
				break;
			}
	    }
	}
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(index_id < 0 || index_id >= (int)thePixelTemp.size()) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate::temperrors can't find needed template ID = " << id << std::endl;
	}
#else
	assert(index_id >= 0 && index_id < (int)thePixelTemp.size());
#endif
	
	//		
	
	// Interpolate the absolute value of cot(beta)     
    
    acotb = fabs((double)cotbeta);
	cotb = cotbeta;
	
	
	// for some cosmics, the ususal gymnastics are incorrect   
	
//	if(thePixelTemp[index_id].head.Dtype == 0) {
	    cotb = acotb;
		flip_y = false;
		if(cotbeta < 0.) {flip_y = true;}
//	} else {
//	    if(locBz < 0.) {
//			cotb = cotbeta;
//			flip_y = false;
//		} else {
//			cotb = -cotbeta;
//			flip_y = true;
//		}	
//	}
		
		// Copy the charge scaling factor to the private variable     
		
		Ny = thePixelTemp[index_id].head.NTy;
		Nyx = thePixelTemp[index_id].head.NTyx;
		Nxx = thePixelTemp[index_id].head.NTxx;
		
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		if(Ny < 2 || Nyx < 1 || Nxx < 2) {
			throw cms::Exception("DataCorrupt") << "template ID = " << id_current << "has too few entries: Ny/Nyx/Nxx = " << Ny << "/" << Nyx << "/" << Nxx << std::endl;
		}
#else
		assert(Ny > 1 && Nyx > 0 && Nxx > 1);
#endif
		imaxx = Nyx - 1;
		imidy = Nxx/2;
        
		// next, loop over all y-angle entries   
		
		ilow = 0;
		yratio = 0.;
		
		if(cotb >= thePixelTemp[index_id].enty[Ny-1].cotbeta) {
			
			ilow = Ny-2;
			yratio = 1.;
			
		} else {
			
			if(cotb >= thePixelTemp[index_id].enty[0].cotbeta) {
				
				for (i=0; i<Ny-1; ++i) { 
					
					if( thePixelTemp[index_id].enty[i].cotbeta <= cotb && cotb < thePixelTemp[index_id].enty[i+1].cotbeta) {
						
						ilow = i;
						yratio = (cotb - thePixelTemp[index_id].enty[i].cotbeta)/(thePixelTemp[index_id].enty[i+1].cotbeta - thePixelTemp[index_id].enty[i].cotbeta);
						break;			 
					}
				}
			} 
		}
		
		ihigh=ilow + 1;
		
		// Interpolate/store all y-related quantities (flip displacements when flip_y)
		ny1_frac = (1. - yratio)*thePixelTemp[index_id].enty[ilow].fracyone + yratio*thePixelTemp[index_id].enty[ihigh].fracyone;
	    ny2_frac = (1. - yratio)*thePixelTemp[index_id].enty[ilow].fracytwo + yratio*thePixelTemp[index_id].enty[ihigh].fracytwo;
		
		// next, loop over all x-angle entries, first, find relevant y-slices   
		
		iylow = 0;
		yxratio = 0.;
		
		if(acotb >= thePixelTemp[index_id].entx[Nyx-1][0].cotbeta) {
			
			iylow = Nyx-2;
			yxratio = 1.;
			
		} else if(acotb >= thePixelTemp[index_id].entx[0][0].cotbeta) {
			
			for (i=0; i<Nyx-1; ++i) { 
				
				if( thePixelTemp[index_id].entx[i][0].cotbeta <= acotb && acotb < thePixelTemp[index_id].entx[i+1][0].cotbeta) {
					
					iylow = i;
					yxratio = (acotb - thePixelTemp[index_id].entx[i][0].cotbeta)/(thePixelTemp[index_id].entx[i+1][0].cotbeta - thePixelTemp[index_id].entx[i][0].cotbeta);
					break;			 
				}
			}
		}
		
		iyhigh=iylow + 1;
		
		ilow = 0;
		xxratio = 0.;
		
		if(cotalpha >= thePixelTemp[index_id].entx[0][Nxx-1].cotalpha) {
			
			ilow = Nxx-2;
			xxratio = 1.;
			
		} else {
			
			if(cotalpha >= thePixelTemp[index_id].entx[0][0].cotalpha) {
				
				for (i=0; i<Nxx-1; ++i) { 
					
					if( thePixelTemp[index_id].entx[0][i].cotalpha <= cotalpha && cotalpha < thePixelTemp[index_id].entx[0][i+1].cotalpha) {
						
						ilow = i;
						xxratio = (cotalpha - thePixelTemp[index_id].entx[0][i].cotalpha)/(thePixelTemp[index_id].entx[0][i+1].cotalpha - thePixelTemp[index_id].entx[0][i].cotalpha);
						break;
					}
				}
			} 
		}
		
		ihigh=ilow + 1;
		
		for(i=0; i<3; ++i) {
		   qfrac[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].qbfrac[i] + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].qbfrac[i])
		   +yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].qbfrac[i] + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].qbfrac[i]);		  
		}
		nx1_frac = (1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].fracxone + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].fracxone)
		   +yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].fracxone + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].fracxone);		  
		nx2_frac = (1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entx[iylow][ilow].fracxtwo + xxratio*thePixelTemp[index_id].entx[iylow][ihigh].fracxtwo)
		   +yxratio*((1. - xxratio)*thePixelTemp[index_id].entx[iyhigh][ilow].fracxtwo + xxratio*thePixelTemp[index_id].entx[iyhigh][ihigh].fracxtwo);		  
		
	

	qbin_frac[0] = qfrac[0];
	qbin_frac[1] = qbin_frac[0] + qfrac[1];
	qbin_frac[2] = qbin_frac[1] + qfrac[2];
	qbin_frac[3] = 1.;
    return;
	
} // qbin

// *************************************************************************************************************************************
//! Make simple 2-D templates from track angles set in interpolate and hit position.      

//! \param       xhit - (input) x-position of hit relative to the lower left corner of pixel[1][1] (to allow for the "padding" of the two-d clusters in the splitter)
//! \param       yhit - (input) y-position of hit relative to the lower left corner of pixel[1][1] 
//! \param    ydouble - (input) STL vector of 21 element array to flag a double-pixel starting at cluster[1][1]
//! \param    xdouble - (input) STL vector of 11 element array to flag a double-pixel starting at cluster[1][1]
//! \param template2d - (output) 2d template of size matched to the cluster.  Input must be zeroed since charge is added only.
// *************************************************************************************************************************************

bool SiPixelTemplate::simpletemplate2D(float xhit, float yhit, std::vector<bool>& ydouble, std::vector<bool>& xdouble, float template2d[BXM2][BYM2])
{
	
	// Local variables
	
	static float x0, y0, xf, yf, xi, yi, sf, si, s0, qpix, slopey, slopex, ds;
	static int i, j, jpix0, ipix0, jpixf, ipixf, jpix, ipix, nx, ny, anx, any, jmax, imax;
	float qtotal;
	//	double path;
	std::list<SimplePixel> list;
	std::list<SimplePixel>::iterator listIter, listEnd;	
	
	// Calculate the entry and exit points for the line charge from the track
	
	x0 = xhit - 0.5*pzsize*cota_current;
	y0 = yhit - 0.5*pzsize*cotb_current;
	
	jpix0 = floor(x0/pxsize)+1;
	ipix0 = floor(y0/pysize)+1;
	
	if(jpix0 < 0 || jpix0 > BXM3) {return false;}
	if(ipix0 < 0 || ipix0 > BYM3) {return false;}
	
	xf = xhit + 0.5*pzsize*cota_current + plorxwidth;
	yf = yhit + 0.5*pzsize*cotb_current + plorywidth;
	
	jpixf = floor(xf/pxsize)+1;
	ipixf = floor(yf/pysize)+1;
	
	if(jpixf < 0 || jpixf > BXM3) {return false;}
	if(ipixf < 0 || ipixf > BYM3) {return false;}
	
// total charge length 
	
	sf = sqrt((xf-x0)*(xf-x0) + (yf-y0)*(yf-y0));
	if((xf-x0) != 0.) {slopey = (yf-y0)/(xf-x0);} else { slopey = 1.e10;}
	if((yf-y0) != 0.) {slopex = (xf-x0)/(yf-y0);} else { slopex = 1.e10;}
	
// use average charge in this direction
	
	qtotal = pqavg_avg;
	
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
	if(anx > 0) {
		if(nx > 0) {
			for(j=jpix0; j<jpixf; ++j) {
				xi = pxsize*j;
				yi = slopey*(xi-x0) + y0;
				ipix = (int)(yi/pysize)+1;
				si = sqrt((xi-x0)*(xi-x0) + (yi-y0)*(yi-y0));
				element.s = si;
				element.x = xi;
				element.y = yi;
				element.i = ipix;
				element.j = j;
				element.btype = 1;
				list.push_back(element);
			}
		} else {
			for(j=jpix0; j>jpixf; --j) {
				xi = pxsize*(j-1);
				yi = slopey*(xi-x0) + y0;
				ipix = (int)(yi/pysize)+1;
				si = sqrt((xi-x0)*(xi-x0) + (yi-y0)*(yi-y0));
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
	if(any > 0) {
		if(ny > 0) {
			for(i=ipix0; i<ipixf; ++i) {
				yi = pysize*i;
				xi = slopex*(yi-y0) + x0;
				jpix = (int)(xi/pxsize)+1;
				si = sqrt((xi-x0)*(xi-x0) + (yi-y0)*(yi-y0));
				element.s = si;
				element.x = xi;
				element.y = yi;
				element.i = i;
				element.j = jpix;
				element.btype = 2;
				list.push_back(element);
			}
		} else {
			for(i=ipix0; i>ipixf; --i) {
				yi = pysize*(i-1);
				xi = slopex*(yi-y0) + x0;
				jpix = (int)(xi/pxsize)+1;
				si = sqrt((xi-x0)*(xi-x0) + (yi-y0)*(yi-y0));
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
	
	for(i=1; i<imax; ++i) {
		if(ydouble[i-1]) {
			listIter = list.begin();
			if(ny > 0) {
				while(listIter != list.end()) {
					if(listIter->i == i && listIter->btype == 2) {
						listIter = list.erase(listIter);
						continue;
					}
					if(listIter->i > i) {
						--(listIter->i);
					}
					++listIter;
				}
			} else {
				while(listIter != list.end()) {
					if(listIter->i == i+1 && listIter->btype == 2) {
						listIter = list.erase(listIter);
						continue;
					}
					if(listIter->i > i+1) {
						--(listIter->i);
					}
					++listIter;
				}				
			}
		}
	}
	
	for(j=1; j<jmax; ++j) {
		if(xdouble[j-1]) {
			listIter = list.begin();
			if(nx > 0) {
				while(listIter != list.end()) {
					if(listIter->j == j && listIter->btype == 1) {
						listIter = list.erase(listIter);
						continue;
					}
					if(listIter->j > j) {
						--(listIter->j);
					}
					++listIter;
				}
			} else {
				while(listIter != list.end()) {
					if(listIter->j == j+1 && listIter->btype == 1) {
						listIter = list.erase(listIter);
						continue;
					}
					if(listIter->j > j+1) {
						--(listIter->j);
					}
					++listIter;
				}				
			}
		}
	}
	
	// The list now contains the path lengths of the line charge in each pixel from (x0,y0).  Cacluate the lengths of the segments and the charge. 
	
	s0 = 0.;
	listIter = list.begin();
	listEnd = list.end();
	for( ;listIter != listEnd; ++listIter) {
		si = listIter->s;
		ds = si - s0;
		s0 = si;
		j = listIter->j;
		i = listIter->i;
		if(sf > 0.) { qpix = qtotal*ds/sf;} else {qpix = qtotal;}
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
	float yratio, cotb;
	
// Interpolate in cotbeta only for the correct total path length (converts cotalpha, cotbeta into an effective cotbeta) 
	
	cotb = sqrt(cotb_current*cotb_current + cota_current*cota_current);
		
// Copy the charge scaling factor to the private variable     
	
	Ny = thePixelTemp[index_id].head.NTy;
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(Ny < 2) {
		throw cms::Exception("DataCorrupt") << "template ID = " << id_current << "has too few entries: Ny = " << Ny << std::endl;
	}
#else
	assert(Ny > 1);
#endif
	
// next, loop over all y-angle entries   
	
	ilow = 0;
	yratio = 0.;
	
	if(cotb >= thePixelTemp[index_id].enty[Ny-1].cotbeta) {
		
		ilow = Ny-2;
		yratio = 1.;
		
	} else {
		
		if(cotb >= thePixelTemp[index_id].enty[0].cotbeta) {
			
			for (i=0; i<Ny-1; ++i) { 
				
				if( thePixelTemp[index_id].enty[i].cotbeta <= cotb && cotb < thePixelTemp[index_id].enty[i+1].cotbeta) {
					
					ilow = i;
					yratio = (cotb - thePixelTemp[index_id].enty[i].cotbeta)/(thePixelTemp[index_id].enty[i+1].cotbeta - thePixelTemp[index_id].enty[i].cotbeta);
					break;			 
				}
			}
		} 
	}
	
	ihigh=ilow + 1;
	
// Interpolate Vavilov parameters
	
	pmpvvav = (1. - yratio)*thePixelTemp[index_id].enty[ilow].mpvvav + yratio*thePixelTemp[index_id].enty[ihigh].mpvvav;
	psigmavav = (1. - yratio)*thePixelTemp[index_id].enty[ilow].sigmavav + yratio*thePixelTemp[index_id].enty[ihigh].sigmavav;
	pkappavav = (1. - yratio)*thePixelTemp[index_id].enty[ilow].kappavav + yratio*thePixelTemp[index_id].enty[ihigh].kappavav;
	
// Copy to parameter list
	
	
	mpv = (double)pmpvvav;
	sigma = (double)psigmavav;
	kappa = (double)pkappavav;
	
	return;
	
} // vavilov_pars

