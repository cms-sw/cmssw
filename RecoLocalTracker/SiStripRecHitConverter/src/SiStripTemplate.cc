//
//  SiStripTemplate.cc  Version 1.00 (based on SiPixelTemplate v8.20)
//
//  V1.05 - add VI optimizations from pixel template object
//  V1.06 - increase angular acceptance (and structure size)
//  V2.00 - add barycenter interpolation and getters, fix calculation for charge deposition to accommodate cota-offsets in the central cotb entries.
//  V2.01 - fix problem with number of spare entries
//

//  Created by Morris Swartz on 2/2/11.
//
//

//#include <stdlib.h> 
//#include <stdio.h>
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
//#include <cmath.h>
#else
#include <math.h>
#endif
#include <algorithm>
#include <vector>
//#include "boost/multi_array.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <list>



#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripTemplate.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) LogError(x)
#define LOGINFO(x) LogInfo(x)
#define ENDL " "
#include "FWCore/Utilities/interface/Exception.h"
using namespace edm;
#else
#include "SiStripTemplate.h"
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
bool SiStripTemplate::pushfile(int filenum, std::vector< SiStripTemplateStore > & theStripTemp_)
{
    // Add template stored in external file numbered filenum to theTemplateStore
    
    // Local variables 
    int i, j, k, l;
	float qavg_avg;
	const char *tempfile;
	//	char title[80]; remove this
    char c;
	const int code_version={18};
	
	
	
	//  Create a filename for this run 
	
	std::ostringstream tout;
	
	//  Create different path in CMSSW than standalone
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	tout << "CalibTracker/SiPixelESProducers/data/stemplate_summary_p" << std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
	std::string tempf = tout.str();
	edm::FileInPath file( tempf.c_str() );
	tempfile = (file.fullPath()).c_str();
#else
	tout << "stemplate_summary_p" << std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
	std::string tempf = tout.str();
	tempfile = tempf.c_str();
#endif
	
	//  open the template file 
	
	std::ifstream in_file(tempfile, std::ios::in);
	
	if(in_file.is_open()) {
		
		// Create a local template storage entry
		
		SiStripTemplateStore theCurrentTemp;
		
		// Read-in a header string first and print it    
		
		for (i=0; (c=in_file.get()) != '\n'; ++i) {
			if(i < 79) {theCurrentTemp.head.title[i] = c;}
		}
		if(i > 78) {i=78;}
		theCurrentTemp.head.title[i+1] ='\0';
		LOGINFO("SiStripTemplate") << "Loading Strip Template File - " << theCurrentTemp.head.title << ENDL;
		
		// next, the header information     
		
		in_file >> theCurrentTemp.head.ID  >> theCurrentTemp.head.templ_version >> theCurrentTemp.head.Bfield >> theCurrentTemp.head.NTy >> theCurrentTemp.head.NTyx >> theCurrentTemp.head.NTxx
		>> theCurrentTemp.head.Dtype >> theCurrentTemp.head.Vbias >> theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >> theCurrentTemp.head.qscale
		>> theCurrentTemp.head.s50 >> theCurrentTemp.head.lorywidth >> theCurrentTemp.head.lorxwidth >> theCurrentTemp.head.ysize >> theCurrentTemp.head.xsize >> theCurrentTemp.head.zsize;
		
		if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file, no template load" << ENDL; return false;}
		
		LOGINFO("SiStripTemplate") << "Template ID = " << theCurrentTemp.head.ID << ", Template Version " << theCurrentTemp.head.templ_version << ", Bfield = " << theCurrentTemp.head.Bfield 
		<< ", NTy = " << theCurrentTemp.head.NTy << ", NTyx = " << theCurrentTemp.head.NTyx<< ", NTxx = " << theCurrentTemp.head.NTxx << ", Dtype = " << theCurrentTemp.head.Dtype
		<< ", Bias voltage " << theCurrentTemp.head.Vbias << ", temperature "
		<< theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence << ", Q-scaling factor " << theCurrentTemp.head.qscale
		<< ", 1/2 threshold " << theCurrentTemp.head.s50 << ", y Lorentz Width " << theCurrentTemp.head.lorywidth << ", x Lorentz width " << theCurrentTemp.head.lorxwidth    
		<< ", pixel x-size " << theCurrentTemp.head.xsize << ", y-size " << theCurrentTemp.head.ysize << ", zsize " << theCurrentTemp.head.zsize << ENDL;
		
		if(theCurrentTemp.head.templ_version < code_version) {LOGERROR("SiStripTemplate") << "code expects version " << code_version << ", no template load" << ENDL; return false;}
		
#ifdef SI_STRIP_TEMPLATE_USE_BOOST 
		
// next, layout the 1-d/2-d structures needed to store template
				
		theCurrentTemp.enty.resize(boost::extents[theCurrentTemp.head.NTy]);

		theCurrentTemp.entx.resize(boost::extents[theCurrentTemp.head.NTyx][theCurrentTemp.head.NTxx]);
		
#endif
		
// next, loop over all y-angle entries   
		
		for (i=0; i < theCurrentTemp.head.NTy; ++i) {     
			
			in_file >> theCurrentTemp.enty[i].runnum >> theCurrentTemp.enty[i].costrk[0] 
			>> theCurrentTemp.enty[i].costrk[1] >> theCurrentTemp.enty[i].costrk[2]; 
			
			if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 1, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			// Calculate the alpha, beta, and cot(beta) for this entry 
			
			theCurrentTemp.enty[i].alpha = static_cast<float>(atan2((double)theCurrentTemp.enty[i].costrk[2], (double)theCurrentTemp.enty[i].costrk[0]));
			
			theCurrentTemp.enty[i].cotalpha = theCurrentTemp.enty[i].costrk[0]/theCurrentTemp.enty[i].costrk[2];
			
			theCurrentTemp.enty[i].beta = static_cast<float>(atan2((double)theCurrentTemp.enty[i].costrk[2], (double)theCurrentTemp.enty[i].costrk[1]));
			
			theCurrentTemp.enty[i].cotbeta = theCurrentTemp.enty[i].costrk[1]/theCurrentTemp.enty[i].costrk[2];
			
			in_file >> theCurrentTemp.enty[i].qavg >> theCurrentTemp.enty[i].sxmax >> theCurrentTemp.enty[i].dxone >> theCurrentTemp.enty[i].sxone >> theCurrentTemp.enty[i].qmin >> theCurrentTemp.enty[i].clslenx;
			
			if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 2, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			for (j=0; j<2; ++j) {
				
				in_file >> theCurrentTemp.enty[i].xpar[j][0] >> theCurrentTemp.enty[i].xpar[j][1] 
				>> theCurrentTemp.enty[i].xpar[j][2] >> theCurrentTemp.enty[i].xpar[j][3] >> theCurrentTemp.enty[i].xpar[j][4];
				
				if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 6, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
				
			}
			
			qavg_avg = 0.f;
			for (j=0; j<9; ++j) {
				
				for (k=0; k<TSXSIZE; ++k) {in_file >> theCurrentTemp.enty[i].xtemp[j][k]; qavg_avg += theCurrentTemp.enty[i].xtemp[j][k];} 
				
				if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 7, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			theCurrentTemp.enty[i].qavg_avg = qavg_avg/9.;
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].xavg[j] >> theCurrentTemp.enty[i].xrms[j] >> theCurrentTemp.enty[i].xgx0[j] >> theCurrentTemp.enty[i].xgsig[j];
				
				if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 10, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].xflpar[j][0] >> theCurrentTemp.enty[i].xflpar[j][1] >> theCurrentTemp.enty[i].xflpar[j][2] 
				>> theCurrentTemp.enty[i].xflpar[j][3] >> theCurrentTemp.enty[i].xflpar[j][4] >> theCurrentTemp.enty[i].xflpar[j][5];
				
				if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 11, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].chi2xavg[j] >> theCurrentTemp.enty[i].chi2xmin[j] >> theCurrentTemp.enty[i].chi2xavgc2m[j] >> theCurrentTemp.enty[i].chi2xminc2m[j];
				
				if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 12, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
				for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].xavgc2m[j] >> theCurrentTemp.enty[i].xrmsc2m[j] >> theCurrentTemp.enty[i].xgx0c2m[j] >> theCurrentTemp.enty[i].xgsigc2m[j];
				
				if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 14, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			} 
			
				for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].xavggen[j] >> theCurrentTemp.enty[i].xrmsgen[j] >> theCurrentTemp.enty[i].xgx0gen[j] >> theCurrentTemp.enty[i].xgsiggen[j];
				
				if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 14b, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			} 
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].xavgbcn[j] >> theCurrentTemp.enty[i].xrmsbcn[j] >> theCurrentTemp.enty[i].xgx0bcn[j] >> theCurrentTemp.enty[i].xgsigbcn[j];
				
				if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 14c, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			} 
			
			in_file >> theCurrentTemp.enty[i].chi2xavgone >> theCurrentTemp.enty[i].chi2xminone >> theCurrentTemp.enty[i].qmin2
			>> theCurrentTemp.enty[i].mpvvav >> theCurrentTemp.enty[i].sigmavav >> theCurrentTemp.enty[i].kappavav 
			>> theCurrentTemp.enty[i].mpvvav2 >> theCurrentTemp.enty[i].sigmavav2 >> theCurrentTemp.enty[i].kappavav2 >> theCurrentTemp.enty[i].spare[0];
			
			if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 15, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			in_file >> theCurrentTemp.enty[i].qbfrac[0] >> theCurrentTemp.enty[i].qbfrac[1] >> theCurrentTemp.enty[i].qbfrac[2] >> theCurrentTemp.enty[i].fracxone
			>> theCurrentTemp.enty[i].spare[1] >> theCurrentTemp.enty[i].spare[2] >> theCurrentTemp.enty[i].spare[3] >> theCurrentTemp.enty[i].spare[4]
			>> theCurrentTemp.enty[i].spare[5] >> theCurrentTemp.enty[i].spare[6];
			
			if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 16, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
		}
		
		// next, loop over all barrel x-angle entries   
		
		for (k=0; k < theCurrentTemp.head.NTyx; ++k) { 
			
			for (i=0; i < theCurrentTemp.head.NTxx; ++i) { 
				
				in_file >> theCurrentTemp.entx[k][i].runnum >> theCurrentTemp.entx[k][i].costrk[0] 
				>> theCurrentTemp.entx[k][i].costrk[1] >> theCurrentTemp.entx[k][i].costrk[2]; 
				
				if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 17, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				// Calculate the alpha, beta, and cot(beta) for this entry 
				
				theCurrentTemp.entx[k][i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entx[k][i].costrk[2], (double)theCurrentTemp.entx[k][i].costrk[0]));
				
				theCurrentTemp.entx[k][i].cotalpha = theCurrentTemp.entx[k][i].costrk[0]/theCurrentTemp.entx[k][i].costrk[2];
				
				theCurrentTemp.entx[k][i].beta = static_cast<float>(atan2((double)theCurrentTemp.entx[k][i].costrk[2], (double)theCurrentTemp.entx[k][i].costrk[1]));
				
				theCurrentTemp.entx[k][i].cotbeta = theCurrentTemp.entx[k][i].costrk[1]/theCurrentTemp.entx[k][i].costrk[2];
				
				in_file >> theCurrentTemp.entx[k][i].qavg >> theCurrentTemp.entx[k][i].sxmax >> theCurrentTemp.entx[k][i].dxone >> theCurrentTemp.entx[k][i].sxone >> theCurrentTemp.entx[k][i].qmin >> theCurrentTemp.entx[k][i].clslenx;
				
				if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 18, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				for (j=0; j<2; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].xpar[j][0] >> theCurrentTemp.entx[k][i].xpar[j][1] 
					>> theCurrentTemp.entx[k][i].xpar[j][2] >> theCurrentTemp.entx[k][i].xpar[j][3] >> theCurrentTemp.entx[k][i].xpar[j][4];
					
					if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 19, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
					
				}
				
				qavg_avg = 0.f;
				for (j=0; j<9; ++j) {
					
					for (l=0; l<TSXSIZE; ++l) {in_file >> theCurrentTemp.entx[k][i].xtemp[j][l]; qavg_avg += theCurrentTemp.entx[k][i].xtemp[j][l];} 
					
					if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 20, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				theCurrentTemp.entx[k][i].qavg_avg = qavg_avg/9.;
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].xavg[j] >> theCurrentTemp.entx[k][i].xrms[j] >> theCurrentTemp.entx[k][i].xgx0[j] >> theCurrentTemp.entx[k][i].xgsig[j];
					
					if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 21, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].xflpar[j][0] >> theCurrentTemp.entx[k][i].xflpar[j][1] >> theCurrentTemp.entx[k][i].xflpar[j][2] 
					>> theCurrentTemp.entx[k][i].xflpar[j][3] >> theCurrentTemp.entx[k][i].xflpar[j][4] >> theCurrentTemp.entx[k][i].xflpar[j][5];
					
					if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 22, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].chi2xavg[j] >> theCurrentTemp.entx[k][i].chi2xmin[j] >> theCurrentTemp.entx[k][i].chi2xavgc2m[j] >> theCurrentTemp.entx[k][i].chi2xminc2m[j];
					
					if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 23, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].xavgc2m[j] >> theCurrentTemp.entx[k][i].xrmsc2m[j] >> theCurrentTemp.entx[k][i].xgx0c2m[j] >> theCurrentTemp.entx[k][i].xgsigc2m[j];
					
					if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 24, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				} 
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].xavggen[j] >> theCurrentTemp.entx[k][i].xrmsgen[j] >> theCurrentTemp.entx[k][i].xgx0gen[j] >> theCurrentTemp.entx[k][i].xgsiggen[j];
					
					if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 25, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				} 
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].xavgbcn[j] >> theCurrentTemp.entx[k][i].xrmsbcn[j] >> theCurrentTemp.entx[k][i].xgx0bcn[j] >> theCurrentTemp.entx[k][i].xgsigbcn[j];
					
					if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 26, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				} 
				
				in_file >> theCurrentTemp.entx[k][i].chi2xavgone >> theCurrentTemp.entx[k][i].chi2xminone >> theCurrentTemp.entx[k][i].qmin2
				>> theCurrentTemp.entx[k][i].mpvvav >> theCurrentTemp.entx[k][i].sigmavav >> theCurrentTemp.entx[k][i].kappavav 
				>> theCurrentTemp.entx[k][i].mpvvav2 >> theCurrentTemp.entx[k][i].sigmavav2 >> theCurrentTemp.entx[k][i].kappavav2 >> theCurrentTemp.entx[k][i].spare[0];
				
				if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 27, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				in_file >> theCurrentTemp.entx[k][i].qbfrac[0] >> theCurrentTemp.entx[k][i].qbfrac[1] >> theCurrentTemp.entx[k][i].qbfrac[2] >> theCurrentTemp.entx[k][i].fracxone
				>> theCurrentTemp.entx[k][i].spare[1] >> theCurrentTemp.entx[k][i].spare[2] >> theCurrentTemp.entx[k][i].spare[3] >> theCurrentTemp.entx[k][i].spare[4]
				>> theCurrentTemp.entx[k][i].spare[5] >> theCurrentTemp.entx[k][i].spare[6];
				
				if(in_file.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 28, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				

			}
		}	
		
		
		in_file.close();
		
		// Add this template to the store
		
		theStripTemp_.push_back(theCurrentTemp);
		
		return true;
		
	} else {
		
		// If file didn't open, report this
		
		LOGERROR("SiStripTemplate") << "Error opening File " << tempfile << ENDL;
		return false;
		
	}
	
} // TempInit 


#ifndef SI_PIXEL_TEMPLATE_STANDALONE

//**************************************************************** 
//! This routine initializes the global template structures from an
//! external file template_summary_zpNNNN where NNNN are four digits 
//! \param dbobject - db storing multiple template calibrations
//**************************************************************** 
bool SiStripTemplate::pushfile(const SiPixelTemplateDBObject& dbobject, std::vector< SiStripTemplateStore > & theStripTemp_)
{
	// Add template stored in external dbobject to theTemplateStore
    
	// Local variables 
	int i, j, k, l;
	float qavg_avg;
	//	const char *tempfile;
	const int code_version={17};
	
	// We must create a new object because dbobject must be a const and our stream must not be
	SiPixelTemplateDBObject db = dbobject;
	
	// Create a local template storage entry
	SiStripTemplateStore theCurrentTemp;
	
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
		LOGINFO("SiStripTemplate") << "Loading Strip Template File - " << theCurrentTemp.head.title << ENDL;
		
		// next, the header information     
		
		db >> theCurrentTemp.head.ID  >> theCurrentTemp.head.templ_version >> theCurrentTemp.head.Bfield >> theCurrentTemp.head.NTy >> theCurrentTemp.head.NTyx >> theCurrentTemp.head.NTxx
		>> theCurrentTemp.head.Dtype >> theCurrentTemp.head.Vbias >> theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >> theCurrentTemp.head.qscale
		>> theCurrentTemp.head.s50 >> theCurrentTemp.head.lorywidth >> theCurrentTemp.head.lorxwidth >> theCurrentTemp.head.ysize >> theCurrentTemp.head.xsize >> theCurrentTemp.head.zsize;
		
		if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file, no template load" << ENDL; return false;}
		
		LOGINFO("SiStripTemplate") << "Template ID = " << theCurrentTemp.head.ID << ", Template Version " << theCurrentTemp.head.templ_version << ", Bfield = " << theCurrentTemp.head.Bfield 
		<< ", NTy = " << theCurrentTemp.head.NTy << ", NTyx = " << theCurrentTemp.head.NTyx<< ", NTxx = " << theCurrentTemp.head.NTxx << ", Dtype = " << theCurrentTemp.head.Dtype
		<< ", Bias voltage " << theCurrentTemp.head.Vbias << ", temperature "
		<< theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence << ", Q-scaling factor " << theCurrentTemp.head.qscale
		<< ", 1/2 threshold " << theCurrentTemp.head.s50 << ", y Lorentz Width " << theCurrentTemp.head.lorywidth << ", x Lorentz width " << theCurrentTemp.head.lorxwidth    
		<< ", pixel x-size " << theCurrentTemp.head.xsize << ", y-size " << theCurrentTemp.head.ysize << ", zsize " << theCurrentTemp.head.zsize << ENDL;
		
		if(theCurrentTemp.head.templ_version < code_version) {LOGERROR("SiStripTemplate") << "code expects version " << code_version << ", no template load" << ENDL; return false;}
		
		
#ifdef SI_PIXEL_TEMPLATE_USE_BOOST 
		
// next, layout the 1-d/2-d structures needed to store template
		
		theCurrentTemp.enty.resize(boost::extents[theCurrentTemp.head.NTy]);
		
		theCurrentTemp.entx.resize(boost::extents[theCurrentTemp.head.NTyx][theCurrentTemp.head.NTxx]);
		
#endif
				
		// next, loop over all y-angle entries   
		
		for (i=0; i < theCurrentTemp.head.NTy; ++i) {     
			
			db >> theCurrentTemp.enty[i].runnum >> theCurrentTemp.enty[i].costrk[0] 
			>> theCurrentTemp.enty[i].costrk[1] >> theCurrentTemp.enty[i].costrk[2]; 
			
			if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 1, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			// Calculate the alpha, beta, and cot(beta) for this entry 
			
			theCurrentTemp.enty[i].alpha = static_cast<float>(atan2((double)theCurrentTemp.enty[i].costrk[2], (double)theCurrentTemp.enty[i].costrk[0]));
			
			theCurrentTemp.enty[i].cotalpha = theCurrentTemp.enty[i].costrk[0]/theCurrentTemp.enty[i].costrk[2];
			
			theCurrentTemp.enty[i].beta = static_cast<float>(atan2((double)theCurrentTemp.enty[i].costrk[2], (double)theCurrentTemp.enty[i].costrk[1]));
			
			theCurrentTemp.enty[i].cotbeta = theCurrentTemp.enty[i].costrk[1]/theCurrentTemp.enty[i].costrk[2];
			
			db >> theCurrentTemp.enty[i].qavg >> theCurrentTemp.enty[i].sxmax >> theCurrentTemp.enty[i].dxone >> theCurrentTemp.enty[i].sxone >> theCurrentTemp.enty[i].qmin >> theCurrentTemp.enty[i].clslenx;
			
			if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 2, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			for (j=0; j<2; ++j) {
				
				db >> theCurrentTemp.enty[i].xpar[j][0] >> theCurrentTemp.enty[i].xpar[j][1] 
				>> theCurrentTemp.enty[i].xpar[j][2] >> theCurrentTemp.enty[i].xpar[j][3] >> theCurrentTemp.enty[i].xpar[j][4];
				
				if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 6, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
				
			}
			
			qavg_avg = 0.f;
			for (j=0; j<9; ++j) {
				
				for (k=0; k<TSXSIZE; ++k) {db >> theCurrentTemp.enty[i].xtemp[j][k]; qavg_avg += theCurrentTemp.enty[i].xtemp[j][k];} 
				
				if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 7, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			theCurrentTemp.enty[i].qavg_avg = qavg_avg/9.;
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].xavg[j] >> theCurrentTemp.enty[i].xrms[j] >> theCurrentTemp.enty[i].xgx0[j] >> theCurrentTemp.enty[i].xgsig[j];
				
				if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 10, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].xflpar[j][0] >> theCurrentTemp.enty[i].xflpar[j][1] >> theCurrentTemp.enty[i].xflpar[j][2] 
				>> theCurrentTemp.enty[i].xflpar[j][3] >> theCurrentTemp.enty[i].xflpar[j][4] >> theCurrentTemp.enty[i].xflpar[j][5];
				
				if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 11, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].chi2xavg[j] >> theCurrentTemp.enty[i].chi2xmin[j] >> theCurrentTemp.enty[i].chi2xavgc2m[j] >> theCurrentTemp.enty[i].chi2xminc2m[j];
				
				if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 12, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].xavgc2m[j] >> theCurrentTemp.enty[i].xrmsc2m[j] >> theCurrentTemp.enty[i].xgx0c2m[j] >> theCurrentTemp.enty[i].xgsigc2m[j];
				
				if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 14, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			} 
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].xavggen[j] >> theCurrentTemp.enty[i].xrmsgen[j] >> theCurrentTemp.enty[i].xgx0gen[j] >> theCurrentTemp.enty[i].xgsiggen[j];
				
				if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 14b, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			} 
			
			for (j=0; j<4; ++j) {
				
				db >> theCurrentTemp.enty[i].xavgbcn[j] >> theCurrentTemp.enty[i].xrmsbcn[j] >> theCurrentTemp.enty[i].xgx0bcn[j] >> theCurrentTemp.enty[i].xgsigbcn[j];
				
				if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 14c, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			} 
			
			db >> theCurrentTemp.enty[i].chi2xavgone >> theCurrentTemp.enty[i].chi2xminone >> theCurrentTemp.enty[i].qmin2
			>> theCurrentTemp.enty[i].mpvvav >> theCurrentTemp.enty[i].sigmavav >> theCurrentTemp.enty[i].kappavav 
			>> theCurrentTemp.enty[i].mpvvav2 >> theCurrentTemp.enty[i].sigmavav2 >> theCurrentTemp.enty[i].kappavav2 >> theCurrentTemp.enty[i].spare[0];
			
			if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 15, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			db >> theCurrentTemp.enty[i].qbfrac[0] >> theCurrentTemp.enty[i].qbfrac[1] >> theCurrentTemp.enty[i].qbfrac[2] >> theCurrentTemp.enty[i].fracxone
			>> theCurrentTemp.enty[i].spare[1] >> theCurrentTemp.enty[i].spare[2] >> theCurrentTemp.enty[i].spare[3] >> theCurrentTemp.enty[i].spare[4]
			>> theCurrentTemp.enty[i].spare[5] >> theCurrentTemp.enty[i].spare[6];
			
			if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 16, no template load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
		}
		
		// next, loop over all barrel x-angle entries   
		
		for (k=0; k < theCurrentTemp.head.NTyx; ++k) { 
			
			for (i=0; i < theCurrentTemp.head.NTxx; ++i) { 
				
				db >> theCurrentTemp.entx[k][i].runnum >> theCurrentTemp.entx[k][i].costrk[0] 
				>> theCurrentTemp.entx[k][i].costrk[1] >> theCurrentTemp.entx[k][i].costrk[2]; 
				
				if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 17, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				// Calculate the alpha, beta, and cot(beta) for this entry 
				
				theCurrentTemp.entx[k][i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entx[k][i].costrk[2], (double)theCurrentTemp.entx[k][i].costrk[0]));
				
				theCurrentTemp.entx[k][i].cotalpha = theCurrentTemp.entx[k][i].costrk[0]/theCurrentTemp.entx[k][i].costrk[2];
				
				theCurrentTemp.entx[k][i].beta = static_cast<float>(atan2((double)theCurrentTemp.entx[k][i].costrk[2], (double)theCurrentTemp.entx[k][i].costrk[1]));
				
				theCurrentTemp.entx[k][i].cotbeta = theCurrentTemp.entx[k][i].costrk[1]/theCurrentTemp.entx[k][i].costrk[2];
				
				db >> theCurrentTemp.entx[k][i].qavg >> theCurrentTemp.entx[k][i].sxmax >> theCurrentTemp.entx[k][i].dxone >> theCurrentTemp.entx[k][i].sxone >> theCurrentTemp.entx[k][i].qmin >> theCurrentTemp.entx[k][i].clslenx;
				
				if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 18, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				for (j=0; j<2; ++j) {
					
					db >> theCurrentTemp.entx[k][i].xpar[j][0] >> theCurrentTemp.entx[k][i].xpar[j][1] 
					>> theCurrentTemp.entx[k][i].xpar[j][2] >> theCurrentTemp.entx[k][i].xpar[j][3] >> theCurrentTemp.entx[k][i].xpar[j][4];
					
					if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 19, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
					
				}
				
				qavg_avg = 0.f;
				for (j=0; j<9; ++j) {
					
					for (l=0; l<TSXSIZE; ++k) {db >> theCurrentTemp.entx[k][i].xtemp[j][l]; qavg_avg += theCurrentTemp.entx[k][i].xtemp[j][l];} 
					
					if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 20, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				theCurrentTemp.entx[k][i].qavg_avg = qavg_avg/9.;
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].xavg[j] >> theCurrentTemp.entx[k][i].xrms[j] >> theCurrentTemp.entx[k][i].xgx0[j] >> theCurrentTemp.entx[k][i].xgsig[j];
					
					if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 21, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].xflpar[j][0] >> theCurrentTemp.entx[k][i].xflpar[j][1] >> theCurrentTemp.entx[k][i].xflpar[j][2] 
					>> theCurrentTemp.entx[k][i].xflpar[j][3] >> theCurrentTemp.entx[k][i].xflpar[j][4] >> theCurrentTemp.entx[k][i].xflpar[j][5];
					
					if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 22, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].chi2xavg[j] >> theCurrentTemp.entx[k][i].chi2xmin[j] >> theCurrentTemp.entx[k][i].chi2xavgc2m[j] >> theCurrentTemp.entx[k][i].chi2xminc2m[j];
					
					if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 23, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].xavgc2m[j] >> theCurrentTemp.entx[k][i].xrmsc2m[j] >> theCurrentTemp.entx[k][i].xgx0c2m[j] >> theCurrentTemp.entx[k][i].xgsigc2m[j];
					
					if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 24, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				} 
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].xavggen[j] >> theCurrentTemp.entx[k][i].xrmsgen[j] >> theCurrentTemp.entx[k][i].xgx0gen[j] >> theCurrentTemp.entx[k][i].xgsiggen[j];
					
					if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 25, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				} 
				
				for (j=0; j<4; ++j) {
					
					db >> theCurrentTemp.entx[k][i].xavgbcn[j] >> theCurrentTemp.entx[k][i].xrmsbcn[j] >> theCurrentTemp.entx[k][i].xgx0bcn[j] >> theCurrentTemp.entx[k][i].xgsigbcn[j];
					
					if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 26, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				} 
				
				db >> theCurrentTemp.entx[k][i].chi2xavgone >> theCurrentTemp.entx[k][i].chi2xminone >> theCurrentTemp.entx[k][i].qmin2
				>> theCurrentTemp.entx[k][i].mpvvav >> theCurrentTemp.entx[k][i].sigmavav >> theCurrentTemp.entx[k][i].kappavav 
				>> theCurrentTemp.entx[k][i].mpvvav2 >> theCurrentTemp.entx[k][i].sigmavav2 >> theCurrentTemp.entx[k][i].kappavav2 >> theCurrentTemp.entx[k][i].spare[0];
				
				if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 27, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				db >> theCurrentTemp.entx[k][i].qbfrac[0] >> theCurrentTemp.entx[k][i].qbfrac[1] >> theCurrentTemp.entx[k][i].qbfrac[2] >> theCurrentTemp.entx[k][i].fracxone
				>> theCurrentTemp.entx[k][i].spare[1] >> theCurrentTemp.entx[k][i].spare[2] >> theCurrentTemp.entx[k][i].spare[3] >> theCurrentTemp.entx[k][i].spare[4]
				>> theCurrentTemp.entx[k][i].spare[5] >> theCurrentTemp.entx[k][i].spare[6];
				
				if(db.fail()) {LOGERROR("SiStripTemplate") << "Error reading file 28, no template load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				
				
			}
		}	
		
		
		// Add this template to the store
		
		theStripTemp_.push_back(theCurrentTemp);
		
	}
	return true;
	
} // TempInit 

#endif


// ************************************************************************************************************ 
//! Interpolate input alpha and beta angles to produce a working template for each individual hit. 
//! \param id - (input) index of the template to use
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param locBy - (input) the sign of the y-component of the local magnetic field (if positive, flip things)
// ************************************************************************************************************ 
bool SiStripTemplate::interpolate(int id, float cotalpha, float cotbeta, float locBy)
{
    // Interpolate for a new set of track angles 
    
    // Local variables 
    int i, j;
	int ilow, ihigh, iylow, iyhigh, Ny, Nxx, Nyx, imidy, imaxx;
	float yratio, yxratio, xxratio, sxmax, qcorrect, qxtempcor, chi2xavgone, chi2xminone, cota, cotb, cotalpha0, cotbeta0;
	bool flip_x;
//	std::vector <float> xrms(4), xgsig(4), xrmsc2m(4), xgsigc2m(4);
	std::vector <float> chi2xavg(4), chi2xmin(4), chi2xavgc2m(4), chi2xminc2m(4);


// Check to see if interpolation is valid     

if(id != id_current_ || cotalpha != cota_current_ || cotbeta != cotb_current_) {

	cota_current_ = cotalpha; cotb_current_ = cotbeta; success_ = true;
	
	if(id != id_current_) {

// Find the index corresponding to id

       index_id_ = -1;
       for(i=0; i<(int)theStripTemp_.size(); ++i) {
	
	      if(id == theStripTemp_[i].head.ID) {
	   
	         index_id_ = i;
		      id_current_ = id;
				
// Copy the charge scaling factor to the private variable     
				
				qscale_ = theStripTemp_[index_id_].head.qscale;
				
// Copy the pseudopixel signal size to the private variable     
				
				s50_ = theStripTemp_[index_id_].head.s50;
			
// Pixel sizes to the private variables     
				
				xsize_ = theStripTemp_[index_id_].head.xsize;
				ysize_ = theStripTemp_[index_id_].head.ysize;
				zsize_ = theStripTemp_[index_id_].head.zsize;
				
				break;
          }
	    }
     }
	 
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if(index_id_ < 0 || index_id_ >= (int)theStripTemp_.size()) {
		throw cms::Exception("DataCorrupt") << "SiStripTemplate::interpolate can't find needed template ID = " << id << std::endl;
	}
#else
	assert(index_id_ >= 0 && index_id_ < (int)theStripTemp_.size());
#endif
	 
// Interpolate the absolute value of cot(beta)     
    
	abs_cotb_ = std::abs(cotbeta);
	cotb = abs_cotb_; 
	
//	qcorrect corrects the cot(alpha)=0 cluster charge for non-zero cot(alpha)	

	cotalpha0 =  theStripTemp_[index_id_].enty[0].cotalpha;
	qcorrect=std::sqrt((1.f+cotbeta*cotbeta+cotalpha*cotalpha)/(1.f+cotbeta*cotbeta+cotalpha0*cotalpha0));	
// flip quantities when the magnetic field in in the positive y local direction  
	if(locBy > 0.f) {
		flip_x = true;
	} else {
		flip_x = false;
	}
		
	Ny = theStripTemp_[index_id_].head.NTy;
	Nyx = theStripTemp_[index_id_].head.NTyx;
	Nxx = theStripTemp_[index_id_].head.NTxx;
		
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(Ny < 2 || Nyx < 1 || Nxx < 2) {
		throw cms::Exception("DataCorrupt") << "template ID = " << id_current_ << "has too few entries: Ny/Nyx/Nxx = " << Ny << "/" << Nyx << "/" << Nxx << std::endl;
	}
#else
	assert(Ny > 1 && Nyx > 0 && Nxx > 1);
#endif
	imaxx = Nyx - 1;
	imidy = Nxx/2;
        
// next, loop over all y-angle entries   

	ilow = 0;
	yratio = 0.f;

	if(cotb >= theStripTemp_[index_id_].enty[Ny-1].cotbeta) {
	
		ilow = Ny-2;
		yratio = 1.f;
		success_ = false;
		
	} else {
	   
		if(cotb >= theStripTemp_[index_id_].enty[0].cotbeta) {

			for (i=0; i<Ny-1; ++i) { 
    
			if( theStripTemp_[index_id_].enty[i].cotbeta <= cotb && cotb < theStripTemp_[index_id_].enty[i+1].cotbeta) {
		  
				ilow = i;
				yratio = (cotb - theStripTemp_[index_id_].enty[i].cotbeta)/(theStripTemp_[index_id_].enty[i+1].cotbeta - theStripTemp_[index_id_].enty[i].cotbeta);
				break;			 
			}
		}
	} else { success_ = false; }
  }
	
	ihigh=ilow + 1;
			  
// Interpolate/store all y-related quantities (flip displacements when flip_y)

	yratio_ = yratio;
	qavg_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].qavg + yratio*theStripTemp_[index_id_].enty[ihigh].qavg;
	qavg_ *= qcorrect;
	sxmax = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].sxmax + yratio*theStripTemp_[index_id_].enty[ihigh].sxmax;
	syparmax_ = sxmax;
	qmin_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].qmin + yratio*theStripTemp_[index_id_].enty[ihigh].qmin;
	qmin_ *= qcorrect;
	qmin2_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].qmin2 + yratio*theStripTemp_[index_id_].enty[ihigh].qmin2;
	qmin2_ *= qcorrect;
	mpvvav_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].mpvvav + yratio*theStripTemp_[index_id_].enty[ihigh].mpvvav;
	mpvvav_ *= qcorrect;
	sigmavav_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].sigmavav + yratio*theStripTemp_[index_id_].enty[ihigh].sigmavav;
	kappavav_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].kappavav + yratio*theStripTemp_[index_id_].enty[ihigh].kappavav;
	mpvvav2_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].mpvvav2 + yratio*theStripTemp_[index_id_].enty[ihigh].mpvvav2;
	mpvvav2_ *= qcorrect;
	sigmavav2_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].sigmavav2 + yratio*theStripTemp_[index_id_].enty[ihigh].sigmavav2;
	kappavav2_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].kappavav2 + yratio*theStripTemp_[index_id_].enty[ihigh].kappavav2;
	qavg_avg_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].qavg_avg + yratio*theStripTemp_[index_id_].enty[ihigh].qavg_avg;
	qavg_avg_ *= qcorrect;
	for(i=0; i<2 ; ++i) {
		for(j=0; j<5 ; ++j) {
// Charge loss switches sides when cot(alpha) changes sign
			if(flip_x) {
				xparly0_[1-i][j] = theStripTemp_[index_id_].enty[ilow].xpar[i][j];
				xparhy0_[1-i][j] = theStripTemp_[index_id_].enty[ihigh].xpar[i][j];
			} else {
				xparly0_[i][j] = theStripTemp_[index_id_].enty[ilow].xpar[i][j];
				xparhy0_[i][j] = theStripTemp_[index_id_].enty[ihigh].xpar[i][j];
			}
		}
	}
	for(i=0; i<4; ++i) {
		chi2xavg[i]=(1.f - yratio)*theStripTemp_[index_id_].enty[ilow].chi2xavg[i] + yratio*theStripTemp_[index_id_].enty[ihigh].chi2xavg[i];
		chi2xmin[i]=(1.f - yratio)*theStripTemp_[index_id_].enty[ilow].chi2xmin[i] + yratio*theStripTemp_[index_id_].enty[ihigh].chi2xmin[i];
		chi2xavgc2m[i]=(1.f - yratio)*theStripTemp_[index_id_].enty[ilow].chi2xavgc2m[i] + yratio*theStripTemp_[index_id_].enty[ihigh].chi2xavgc2m[i];
		chi2xminc2m[i]=(1.f - yratio)*theStripTemp_[index_id_].enty[ilow].chi2xminc2m[i] + yratio*theStripTemp_[index_id_].enty[ihigh].chi2xminc2m[i];
	}
	   
//// Do the spares next

	chi2xavgone=(1.f - yratio)*theStripTemp_[index_id_].enty[ilow].chi2xavgone + yratio*theStripTemp_[index_id_].enty[ihigh].chi2xavgone;
	chi2xminone=(1.f - yratio)*theStripTemp_[index_id_].enty[ilow].chi2xminone + yratio*theStripTemp_[index_id_].enty[ihigh].chi2xminone;
		//       for(i=0; i<10; ++i) {
//		    pyspare[i]=(1.f - yratio)*theStripTemp_[index_id_].enty[ilow].yspare[i] + yratio*theStripTemp_[index_id_].enty[ihigh].yspare[i];
//       }
			  
	
// next, loop over all x-angle entries, first, find relevant y-slices   
	
	iylow = 0;
	yxratio = 0.f;

	if(abs_cotb_ >= theStripTemp_[index_id_].entx[Nyx-1][0].cotbeta) {
	
		iylow = Nyx-2;
		yxratio = 1.f;
		
	} else if(abs_cotb_ >= theStripTemp_[index_id_].entx[0][0].cotbeta) {

		for (i=0; i<Nyx-1; ++i) { 
    
			if( theStripTemp_[index_id_].entx[i][0].cotbeta <= abs_cotb_ && abs_cotb_ < theStripTemp_[index_id_].entx[i+1][0].cotbeta) {
		  
			   iylow = i;
			   yxratio = (abs_cotb_ - theStripTemp_[index_id_].entx[i][0].cotbeta)/(theStripTemp_[index_id_].entx[i+1][0].cotbeta - theStripTemp_[index_id_].entx[i][0].cotbeta);
			   break;			 
			}
		}
	}
	
	iyhigh=iylow + 1;

	ilow = 0;
	xxratio = 0.f;
	if(flip_x) {cota = -cotalpha;} else {cota = cotalpha;}

	if(cota >= theStripTemp_[index_id_].entx[0][Nxx-1].cotalpha) {
	
		ilow = Nxx-2;
		xxratio = 1.f;
		success_ = false;
		
	} else {
	   
		if(cota >= theStripTemp_[index_id_].entx[0][0].cotalpha) {

			for (i=0; i<Nxx-1; ++i) { 
    
			   if( theStripTemp_[index_id_].entx[0][i].cotalpha <= cota && cota < theStripTemp_[index_id_].entx[0][i+1].cotalpha) {
		  
				  ilow = i;
				  xxratio = (cota - theStripTemp_[index_id_].entx[0][i].cotalpha)/(theStripTemp_[index_id_].entx[0][i+1].cotalpha - theStripTemp_[index_id_].entx[0][i].cotalpha);
				  break;
			}
		  }
		} else { success_ = false; }
	}
	
	ihigh=ilow + 1;
			  
// Interpolate/store all x-related quantities 

	yxratio_ = yxratio;
	xxratio_ = xxratio;		
	   		  
// sxparmax defines the maximum charge for which the parameters xpar are defined (not rescaled by cotbeta) 

	sxparmax_ = (1.f - xxratio)*theStripTemp_[index_id_].entx[imaxx][ilow].sxmax + xxratio*theStripTemp_[index_id_].entx[imaxx][ihigh].sxmax;
	sxmax_ = sxparmax_;
	if(theStripTemp_[index_id_].entx[imaxx][imidy].sxmax != 0.f) {sxmax_=sxmax_/theStripTemp_[index_id_].entx[imaxx][imidy].sxmax*sxmax;}
	dxone_ = (1.f - xxratio)*theStripTemp_[index_id_].entx[0][ilow].dxone + xxratio*theStripTemp_[index_id_].entx[0][ihigh].dxone;
	if(flip_x) {dxone_ = -dxone_;}
	sxone_ = (1.f - xxratio)*theStripTemp_[index_id_].entx[0][ilow].sxone + xxratio*theStripTemp_[index_id_].entx[0][ihigh].sxone;
	clslenx_ = fminf(theStripTemp_[index_id_].entx[0][ilow].clslenx, theStripTemp_[index_id_].entx[0][ihigh].clslenx);
	for(i=0; i<2 ; ++i) {
		for(j=0; j<5 ; ++j) {
			if(flip_x) {
	         xpar0_[1-i][j] = theStripTemp_[index_id_].entx[imaxx][imidy].xpar[i][j];
	         xparl_[1-i][j] = theStripTemp_[index_id_].entx[imaxx][ilow].xpar[i][j];
	         xparh_[1-i][j] = theStripTemp_[index_id_].entx[imaxx][ihigh].xpar[i][j];
			} else {
	         xpar0_[i][j] = theStripTemp_[index_id_].entx[imaxx][imidy].xpar[i][j];
	         xparl_[i][j] = theStripTemp_[index_id_].entx[imaxx][ilow].xpar[i][j];
	         xparh_[i][j] = theStripTemp_[index_id_].entx[imaxx][ihigh].xpar[i][j];
			}
		}
	}
	   		  
// sxmax is the maximum allowed strip charge (used for truncation)

	sxmax_=(1.f - yxratio)*((1.f - xxratio)*theStripTemp_[index_id_].entx[iylow][ilow].sxmax + xxratio*theStripTemp_[index_id_].entx[iylow][ihigh].sxmax)
			+yxratio*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].sxmax + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].sxmax);
			  
	for(i=0; i<4; ++i) {
		xavg_[i]=(1.f - yxratio)*((1.f - xxratio)*theStripTemp_[index_id_].entx[iylow][ilow].xavg[i] + xxratio*theStripTemp_[index_id_].entx[iylow][ihigh].xavg[i])
				+yxratio*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xavg[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xavg[i]);
		if(flip_x) {xavg_[i] = -xavg_[i];}
		  
		xrms_[i]=(1.f - yxratio)*((1.f - xxratio)*theStripTemp_[index_id_].entx[iylow][ilow].xrms[i] + xxratio*theStripTemp_[index_id_].entx[iylow][ihigh].xrms[i])
				+yxratio*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xrms[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xrms[i]);
		  
//	      xgx0_[i]=(1.f - yxratio)*((1.f - xxratio)*theStripTemp_[index_id_].entx[iylow][ilow].xgx0[i] + xxratio*theStripTemp_[index_id_].entx[iylow][ihigh].xgx0[i])
//		          +yxratio*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xgx0[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xgx0[i]);
							
//	      xgsig_[i]=(1.f - yxratio)*((1.f - xxratio)*theStripTemp_[index_id_].entx[iylow][ilow].xgsig[i] + xxratio*theStripTemp_[index_id_].entx[iylow][ihigh].xgsig[i])
//		          +yxratio*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xgsig[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xgsig[i]);
				  
		xavgc2m_[i]=(1.f - yxratio)*((1.f - xxratio)*theStripTemp_[index_id_].entx[iylow][ilow].xavgc2m[i] + xxratio*theStripTemp_[index_id_].entx[iylow][ihigh].xavgc2m[i])
				+yxratio*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xavgc2m[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xavgc2m[i]);
		if(flip_x) {xavgc2m_[i] = -xavgc2m_[i];}
		  
		xrmsc2m_[i]=(1.f - yxratio)*((1.f - xxratio)*theStripTemp_[index_id_].entx[iylow][ilow].xrmsc2m[i] + xxratio*theStripTemp_[index_id_].entx[iylow][ihigh].xrmsc2m[i])
				+yxratio*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xrmsc2m[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xrmsc2m[i]);
		  
		xavgbcn_[i]=(1.f - yxratio)*((1.f - xxratio)*theStripTemp_[index_id_].entx[iylow][ilow].xavgbcn[i] + xxratio*theStripTemp_[index_id_].entx[iylow][ihigh].xavgbcn[i])
        +yxratio*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xavgbcn[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xavgbcn[i]);
		if(flip_x) {xavgbcn_[i] = -xavgbcn_[i];}
        
		xrmsbcn_[i]=(1.f - yxratio)*((1.f - xxratio)*theStripTemp_[index_id_].entx[iylow][ilow].xrmsbcn[i] + xxratio*theStripTemp_[index_id_].entx[iylow][ihigh].xrmsbcn[i])
        +yxratio*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xrmsbcn[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xrmsbcn[i]);
        
//	      xgx0c2m_[i]=(1.f - yxratio)*((1.f - xxratio)*theStripTemp_[index_id_].entx[iylow][ilow].xgx0c2m[i] + xxratio*theStripTemp_[index_id_].entx[iylow][ihigh].xgx0c2m[i])
//		          +yxratio*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xgx0c2m[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xgx0c2m[i]);
							
//	      xgsigc2m_[i]=(1.f - yxratio)*((1.f - xxratio)*theStripTemp_[index_id_].entx[iylow][ilow].xgsigc2m[i] + xxratio*theStripTemp_[index_id_].entx[iylow][ihigh].xgsigc2m[i])
//		          +yxratio*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xgsigc2m[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xgsigc2m[i]);
//
//  Try new interpolation scheme
//	  														
//	      chi2xavg_[i]=((1.f - xxratio)*theStripTemp_[index_id_].entx[imaxx][ilow].chi2xavg[i] + xxratio*theStripTemp_[index_id_].entx[imaxx][ihigh].chi2xavg[i]);
//		  if(theStripTemp_[index_id_].entx[imaxx][imidy].chi2xavg[i] != 0.f) {chi2xavg_[i]=chi2xavg_[i]/theStripTemp_[index_id_].entx[imaxx][imidy].chi2xavg[i]*chi2xavg[i];}
//							
//	      chi2xmin_[i]=((1.f - xxratio)*theStripTemp_[index_id_].entx[imaxx][ilow].chi2xmin[i] + xxratio*theStripTemp_[index_id_].entx[imaxx][ihigh].chi2xmin[i]);
//		  if(theStripTemp_[index_id_].entx[imaxx][imidy].chi2xmin[i] != 0.f) {chi2xmin_[i]=chi2xmin_[i]/theStripTemp_[index_id_].entx[imaxx][imidy].chi2xmin[i]*chi2xmin[i];}
//		  
		chi2xavg_[i]=((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].chi2xavg[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].chi2xavg[i]);
		if(theStripTemp_[index_id_].entx[iyhigh][imidy].chi2xavg[i] != 0.f) {chi2xavg_[i]=chi2xavg_[i]/theStripTemp_[index_id_].entx[iyhigh][imidy].chi2xavg[i]*chi2xavg[i];}
							
		chi2xmin_[i]=((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].chi2xmin[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].chi2xmin[i]);
		if(theStripTemp_[index_id_].entx[iyhigh][imidy].chi2xmin[i] != 0.f) {chi2xmin_[i]=chi2xmin_[i]/theStripTemp_[index_id_].entx[iyhigh][imidy].chi2xmin[i]*chi2xmin[i];}

		chi2xavgc2m_[i]=((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].chi2xavgc2m[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].chi2xavgc2m[i]);
		if(theStripTemp_[index_id_].entx[iyhigh][imidy].chi2xavgc2m[i] != 0.f) {chi2xavgc2m_[i]=chi2xavgc2m_[i]/theStripTemp_[index_id_].entx[iyhigh][imidy].chi2xavgc2m[i]*chi2xavgc2m[i];}
		
		chi2xminc2m_[i]=((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].chi2xminc2m[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].chi2xminc2m[i]);
		if(theStripTemp_[index_id_].entx[iyhigh][imidy].chi2xminc2m[i] != 0.f) {chi2xminc2m_[i]=chi2xminc2m_[i]/theStripTemp_[index_id_].entx[iyhigh][imidy].chi2xminc2m[i]*chi2xminc2m[i];}	
		
		for(j=0; j<6 ; ++j) {
	         xflparll_[i][j] = theStripTemp_[index_id_].entx[iylow][ilow].xflpar[i][j];
	         xflparlh_[i][j] = theStripTemp_[index_id_].entx[iylow][ihigh].xflpar[i][j];
	         xflparhl_[i][j] = theStripTemp_[index_id_].entx[iyhigh][ilow].xflpar[i][j];
	         xflparhh_[i][j] = theStripTemp_[index_id_].entx[iyhigh][ihigh].xflpar[i][j];
			// Since Q_fl is odd under cotbeta, it flips qutomatically, change only even terms
			
			if(flip_x && (j == 0 || j == 2 || j == 4)) {
	         xflparll_[i][j] = -xflparll_[i][j];
	         xflparlh_[i][j] = -xflparlh_[i][j];
	         xflparhl_[i][j] = -xflparhl_[i][j];
	         xflparhh_[i][j] = -xflparhh_[i][j];
			}
		}
	}
	   
// Do the spares next

	chi2xavgone_=((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].chi2xavgone + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].chi2xavgone);
	if(theStripTemp_[index_id_].entx[iyhigh][imidy].chi2xavgone != 0.f) {chi2xavgone_=chi2xavgone_/theStripTemp_[index_id_].entx[iyhigh][imidy].chi2xavgone*chi2xavgone;}
		
	chi2xminone_=((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].chi2xminone + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].chi2xminone);
	if(theStripTemp_[index_id_].entx[iyhigh][imidy].chi2xminone != 0.f) {chi2xminone_=chi2xminone_/theStripTemp_[index_id_].entx[iyhigh][imidy].chi2xminone*chi2xminone;}
		//       for(i=0; i<10; ++i) {
//	      pxspare[i]=(1.f - yxratio)*((1.f - xxratio)*theStripTemp_[index_id_].entx[iylow][ilow].xspare[i] + xxratio*theStripTemp_[index_id_].entx[iylow][ihigh].xspare[i])
//		          +yxratio*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xspare[i] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xspare[i]);
//       }
			  
// Interpolate and build the x-template 
	
//	qxtempcor corrects the total charge to the actual track angles (not actually needed for the template fits, but useful for Guofan)
	
	cotbeta0 =  theStripTemp_[index_id_].entx[iyhigh][0].cotbeta;
	qxtempcor=std::sqrt((1.f+cotbeta*cotbeta+cotalpha*cotalpha)/(1.f+cotbeta0*cotbeta0+cotalpha*cotalpha));
	
	for(i=0; i<9; ++i) {
		xtemp_[i][0] = 0.f;
		xtemp_[i][1] = 0.f;
		xtemp_[i][BSXM2] = 0.f;
		xtemp_[i][BSXM1] = 0.f;
		for(j=0; j<TSXSIZE; ++j) {
//  Take next largest x-slice for the x-template (it reduces bias in the forward direction after irradiation)
//		   xtemp_[i][j+2]=(1.f - xxratio)*theStripTemp_[index_id_].entx[imaxx][ilow].xtemp[i][j] + xxratio*theStripTemp_[index_id_].entx[imaxx][ihigh].xtemp[i][j];
         if(flip_x) {
				xtemp_[8-i][BSXM3-j]=qxtempcor*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xtemp[i][j] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xtemp[i][j]);
			} else {
            xtemp_[i][j+2]=qxtempcor*((1.f - xxratio)*theStripTemp_[index_id_].entx[iyhigh][ilow].xtemp[i][j] + xxratio*theStripTemp_[index_id_].entx[iyhigh][ihigh].xtemp[i][j]);
			}
		}
	}
	
	lorxwidth_ = theStripTemp_[index_id_].head.lorxwidth;
	if(locBy > 0.f) {lorxwidth_ = -lorxwidth_;}
	
  }
	
  return success_;
} // interpolate



// ************************************************************************************************************ 
//! Return vector of x errors (squared) for an input vector of projected signals 
//! Add large Q scaling for use in cluster splitting.
//! \param fxpix - (input) index of the first real pixel in the projected cluster (doesn't include pseudopixels)
//! \param lxpix - (input) index of the last real pixel in the projected cluster (doesn't include pseudopixels)
//! \param sxthr - (input) maximum signal before de-weighting
//! \param xsum - (input) 11-element vector of pixel signals
//! \param xsig2 - (output) 11-element vector of x errors (squared)
// ************************************************************************************************************ 
  void SiStripTemplate::xsigma2(int fxpix, int lxpix, float sxthr, float xsum[11], float xsig2[11])
  
{
    // Interpolate using quantities already stored in the private variables
    
    // Local variables 
    int i;
	float sigi, sigi2, sigi3, sigi4, yint, sxmax, x0, qscale;
	float sigiy, sigiy2, sigiy3, sigiy4;
	
    // Make sure that input is OK
    
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		  if(fxpix < 2 || fxpix >= BSXM2) {
			 throw cms::Exception("DataCorrupt") << "SiStripTemplate::xsigma2 called with fxpix = " << fxpix << std::endl;
		   }
#else
		   assert(fxpix > 1 && fxpix < BSXM2);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
			 if(lxpix < fxpix || lxpix >= BSXM2) {
				throw cms::Exception("DataCorrupt") << "SiStripTemplate::xsigma2 called with lxpix/fxpix = " << lxpix << "/" << fxpix << std::endl;
			 }
#else
			 assert(lxpix >= fxpix && lxpix < BSXM2);
#endif
	   	     
// Define the maximum signal to use in the parameterization 

       sxmax = sxmax_;
	   if(sxmax_ > sxparmax_) {sxmax = sxparmax_;}
	   
// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 

	   for(i=fxpix-2; i<=lxpix+2; ++i) {
		  if(i < fxpix || i > lxpix) {
	   
// Nearest pseudopixels have uncertainties of 50% of threshold, next-nearest have 10% of threshold

			 xsig2[i] = s50_*s50_;
		  } else {
			 if(xsum[i] < sxmax) {
				sigi = xsum[i];
				qscale = 1.f;
			 } else {
				sigi = sxmax;
				qscale = xsum[i]/sxmax;
			 }
			 sigi2 = sigi*sigi; sigi3 = sigi2*sigi; sigi4 = sigi3*sigi;
			 
			  if(xsum[i] < syparmax_) {
				  sigiy = xsum[i];
			  } else {
				  sigiy = syparmax_;
			  }
			  sigiy2 = sigiy*sigiy; sigiy3 = sigiy2*sigiy; sigiy4 = sigiy3*sigiy;
			  
// First, do the cotbeta interpolation			 
			 
			 if(i <= BSHX) {
				yint = (1.f-yratio_)*
				(xparly0_[0][0]+xparly0_[0][1]*sigiy+xparly0_[0][2]*sigiy2+xparly0_[0][3]*sigiy3+xparly0_[0][4]*sigiy4)
				+ yratio_*
				(xparhy0_[0][0]+xparhy0_[0][1]*sigiy+xparhy0_[0][2]*sigiy2+xparhy0_[0][3]*sigiy3+xparhy0_[0][4]*sigiy4);
			 } else {
				yint = (1.f-yratio_)*
				(xparly0_[1][0]+xparly0_[1][1]*sigiy+xparly0_[1][2]*sigiy2+xparly0_[1][3]*sigiy3+xparly0_[1][4]*sigiy4)
				+ yratio_*
				(xparhy0_[1][0]+xparhy0_[1][1]*sigiy+xparhy0_[1][2]*sigiy2+xparhy0_[1][3]*sigiy3+xparhy0_[1][4]*sigiy4);
			 }
			 
// Next, do the cotalpha interpolation			 
			 
			 if(i <= BSHX) {
				xsig2[i] = (1.f-xxratio_)*
				(xparl_[0][0]+xparl_[0][1]*sigi+xparl_[0][2]*sigi2+xparl_[0][3]*sigi3+xparl_[0][4]*sigi4)
				+ xxratio_*
				(xparh_[0][0]+xparh_[0][1]*sigi+xparh_[0][2]*sigi2+xparh_[0][3]*sigi3+xparh_[0][4]*sigi4);
			 } else {
				xsig2[i] = (1.f-xxratio_)*
				(xparl_[1][0]+xparl_[1][1]*sigi+xparl_[1][2]*sigi2+xparl_[1][3]*sigi3+xparl_[1][4]*sigi4)
				+ xxratio_*
			    (xparh_[1][0]+xparh_[1][1]*sigi+xparh_[1][2]*sigi2+xparh_[1][3]*sigi3+xparh_[1][4]*sigi4);
			 }
			 
// Finally, get the mid-point value of the cotalpha function			 
			 
			 if(i <= BSHX) {
				x0 = xpar0_[0][0]+xpar0_[0][1]*sigi+xpar0_[0][2]*sigi2+xpar0_[0][3]*sigi3+xpar0_[0][4]*sigi4;
			 } else {
				x0 = xpar0_[1][0]+xpar0_[1][1]*sigi+xpar0_[1][2]*sigi2+xpar0_[1][3]*sigi3+xpar0_[1][4]*sigi4;
			 }
			 
// Finally, rescale the yint value for cotalpha variation			 
			 
			 if(x0 != 0.f) {xsig2[i] = xsig2[i]/x0 * yint;}
			 xsig2[i] *=qscale;
		     if(xsum[i] > sxthr) {xsig2[i] = 1.e8f;}
			 if(xsig2[i] <= 0.f) {LOGERROR("SiStripTemplate") << "neg x-error-squared = " << xsig2[i] << ", id = " << id_current_ << ", index = " << index_id_ << 
			 ", cot(alpha) = " << cota_current_ << ", cot(beta) = " << cotb_current_  << ", sigi = " << sigi << ", sxparmax = " << sxparmax_ << ", sxmax = " << sxmax_ << ENDL;}
	      }
	   }
	
	return;
	
} // End xsigma2




// ************************************************************************************************************ 
//! Return interpolated x-correction for input charge bin and qflx
//! \param binq - (input) charge bin [0-3]
//! \param qflx - (input) (Q_f-Q_l)/(Q_f+Q_l) for this cluster
// ************************************************************************************************************ 
  float SiStripTemplate::xflcorr(int binq, float qflx)
  
{
    // Interpolate using quantities already stored in the private variables
    
    // Local variables 
	float qfl, qfl2, qfl3, qfl4, qfl5, dx;
	
    // Make sure that input is OK
    
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(binq < 0 || binq > 3) {
		throw cms::Exception("DataCorrupt") << "SiStripTemplate::xflcorr called with binq = " << binq << std::endl;
	}
#else
	assert(binq >= 0 && binq < 4);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(fabs((double)qflx) > 1.) {
		throw cms::Exception("DataCorrupt") << "SiStripTemplate::xflcorr called with qflx = " << qflx << std::endl;
	}
#else
	assert(fabs((double)qflx) <= 1.);
#endif
	   	     
// Define the maximum signal to allow before de-weighting a pixel 

       qfl = qflx;

       if(qfl < -0.9f) {qfl = -0.9f;}
	   if(qfl > 0.9f) {qfl = 0.9f;}
	   
// Interpolate between the two polynomials

	   qfl2 = qfl*qfl; qfl3 = qfl2*qfl; qfl4 = qfl3*qfl; qfl5 = qfl4*qfl;
	   dx = (1.f - yxratio_)*((1.f-xxratio_)*(xflparll_[binq][0]+xflparll_[binq][1]*qfl+xflparll_[binq][2]*qfl2+xflparll_[binq][3]*qfl3+xflparll_[binq][4]*qfl4+xflparll_[binq][5]*qfl5)
		  + xxratio_*(xflparlh_[binq][0]+xflparlh_[binq][1]*qfl+xflparlh_[binq][2]*qfl2+xflparlh_[binq][3]*qfl3+xflparlh_[binq][4]*qfl4+xflparlh_[binq][5]*qfl5))
	      + yxratio_*((1.f-xxratio_)*(xflparhl_[binq][0]+xflparhl_[binq][1]*qfl+xflparhl_[binq][2]*qfl2+xflparhl_[binq][3]*qfl3+xflparhl_[binq][4]*qfl4+xflparhl_[binq][5]*qfl5)
		  + xxratio_*(xflparhh_[binq][0]+xflparhh_[binq][1]*qfl+xflparhh_[binq][2]*qfl2+xflparhh_[binq][3]*qfl3+xflparhh_[binq][4]*qfl4+xflparhh_[binq][5]*qfl5));
	
	return dx;
	
} // End xflcorr


// ************************************************************************************************************ 
//! Return interpolated y-template in single call
//! \param fxbin - (input) index of first bin (0-40) to fill
//! \param fxbin - (input) index of last bin (0-40) to fill
//! \param xtemplate - (output) a 41x11 output buffer
// ************************************************************************************************************ 
  void SiStripTemplate::xtemp(int fxbin, int lxbin, float xtemplate[41][BSXSIZE])
  
{
    // Retrieve already interpolated quantities
    
    // Local variables 
    int i, j;

   // Verify that input parameters are in valid range

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(fxbin < 0 || fxbin > 40) {
		throw cms::Exception("DataCorrupt") << "SiStripTemplate::xtemp called with fxbin = " << fxbin << std::endl;
	}
#else
	assert(fxbin >= 0 && fxbin < 41);
#endif
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(lxbin < 0 || lxbin > 40) {
		throw cms::Exception("DataCorrupt") << "SiStripTemplate::xtemp called with lxbin = " << lxbin << std::endl;
	}
#else
	assert(lxbin >= 0 && lxbin < 41);
#endif

// Build the x-template, the central 25 bins are here in all cases
	
	for(i=0; i<9; ++i) {
	   for(j=0; j<BSXSIZE; ++j) {
	      xtemplate[i+16][j]=xtemp_[i][j];
	   }
	}
	for(i=0; i<8; ++i) {
	   xtemplate[i+8][BSXM1] = 0.f;
	   for(j=0; j<BSXM1; ++j) {
	      xtemplate[i+8][j]=xtemp_[i][j+1];
	   }
	}
	for(i=1; i<9; ++i) {
	   xtemplate[i+24][0] = 0.f;
	   for(j=0; j<BSXM1; ++j) {
	      xtemplate[i+24][j+1]=xtemp_[i][j];
	   }
	}
	
//  Add more bins if needed	
	
	if(fxbin < 8) {
	   for(i=0; i<8; ++i) {
          xtemplate[i][BSXM2] = 0.f;
	      xtemplate[i][BSXM1] = 0.f;
	      for(j=0; j<BSXM2; ++j) {
	        xtemplate[i][j]=xtemp_[i][j+2];
	      }
	   }
	}
	if(lxbin > 32) {
	   for(i=1; i<9; ++i) {
          xtemplate[i+32][0] = 0.f;
	      xtemplate[i+32][1] = 0.f;
	      for(j=0; j<BSXM2; ++j) {
	        xtemplate[i+32][j+2]=xtemp_[i][j];
	      }
	   }
	}
	
 	return;
	
} // End xtemp


// ************************************************************************************************************ 
//! Interpolate the template in xhit and return scaled charges (ADC units) in the vector container
//! \param xhit - (input) coordinate of the hit (0 at center of first strip: cluster[0])
//! \param cluster - (output) vector array of TSXSIZE (or any other)
// ************************************************************************************************************ 
void SiStripTemplate::sxtemp(float xhit, std::vector<float>& cluster)

{
	// Retrieve already interpolated quantities
	
	// Local variables 
	int i, j;
	
	// Extract x template based upon the hit position 
   
	float xpix = xhit/xsize_ + 0.5f;
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(xpix < 0.f) {
		throw cms::Exception("DataCorrupt") << "SiStripTemplate::2xtemp called with xhit = " << xhit << std::endl;
	}
#else
	assert(xpix >= 0.f);
#endif
	
// cpix is struck pixel(strip) of the cluster	
	
	int cpix = (int)xpix;
	int shift = BSHX - cpix;
	
// xbin the floating bin number and cbin is the bin number (between 0 and 7) of the interpolated template
	
	float xbin = 8.*(xpix-(float)cpix);
	int cbin = (int)xbin;
	
	float xfrac = xbin-(float)cbin;
	
	int sizex = std::min((int)cluster.size(), BSXSIZE);
	
// shift and interpolate the correct cluster shape	
	
	for(i=0; i<sizex; ++i) {
		j = i+shift;
		if(j < 0 || j > sizex-1) {cluster[i] = 0.f;} else {
	      cluster[i]=(1.f-xfrac)*xtemp_[cbin][j]+xfrac*xtemp_[cbin+1][j];
			if(cluster[i] < s50_) cluster[i] = 0.f;
		}
		
// Return cluster in same charge units		
		
		cluster[i] /= qscale_;
	}

	
 	return;
	
} // End sxtemp

// ************************************************************************************************************ 
//! Return central pixel of x-template pixels above readout threshold
// ************************************************************************************************************ 
int SiStripTemplate::cxtemp()

{
	// Retrieve already interpolated quantities
	
	// Local variables 
	int j;
	
	// Analyze only pixels along the central entry
	// First, find the maximum signal and then work out to the edges
	
	float sigmax = 0.f;
	int jmax = -1;
	
	for(j=0; j<BSXSIZE; ++j) {
		if(xtemp_[4][j] > sigmax) {
			sigmax = xtemp_[4][j];
			jmax = j;
		}	
	}
	if(sigmax < 2.*s50_ || jmax<1 || jmax>BSXM2) {return -1;}
	
	//  Now search forward and backward
	
	int jend = jmax;
	
	for(j=jmax+1; j<BSXM1; ++j) {
		if(xtemp_[4][j] < 2.*s50_) break;
	   jend = j;
   }

   int jbeg = jmax;

   for(j=jmax-1; j>0; --j) {
	   if(xtemp_[4][j] < 2.*s50_) break;
	   jbeg = j;
   }

   return (jbeg+jend)/2;

} // End cxtemp


// ************************************************************************************************************ 
//! Make interpolated 3d x-template (stored as class variables)
//! \param nxpix - (input) number of pixels in cluster (needed to size template)
//! \param nxbins - (output) number of bins needed for each template projection
// ************************************************************************************************************ 
void SiStripTemplate::xtemp3d_int(int nxpix, int& nxbins)

{	
    // Retrieve already interpolated quantities
    
    // Local variables 
    int i, j, k;
	int ioff0, ioffp, ioffm;
    
    // Verify that input parameters are in valid range
    
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(nxpix < 1 || nxpix >= BSXM3) {
        throw cms::Exception("DataCorrupt") << "SiPixelTemplate::xtemp3d called with nxpix = " << nxpix << std::endl;
	}
#else
	assert(nxpix > 0 && nxpix < BSXM3);
#endif
	
    // Calculate the size of the shift in pixels needed to span the entire cluster
    
    float diff = fabsf(nxpix - clslenx_)/2. + 1.f;
	int nshift = (int)diff;
	if((diff - nshift) > 0.5f) {++nshift;}
    
    // Calculate the number of bins needed to specify each hit range
    
    nxbins_ = 9 + 16*nshift;
	
    // Create a 2-d working template with the correct size
    
	temp2dx_.resize(boost::extents[nxbins_][BSXSIZE]);
	
    //  The 9 central bins are copied from the interpolated private store
    
	ioff0 = 8*nshift;
	
	for(i=0; i<9; ++i) {
        for(j=0; j<BSXSIZE; ++j) {
            temp2dx_[i+ioff0][j]=xtemp_[i][j];
        }
	}
	
    // Add the +- shifted templates	
    
	for(k=1; k<=nshift; ++k) {
        ioffm=ioff0-k*8;
        for(i=0; i<8; ++i) {
            for(j=0; j<k; ++j) {
                temp2dx_[i+ioffm][BSXM1-j] = 0.f;
            }
            for(j=0; j<BSXSIZE-k; ++j) {
                temp2dx_[i+ioffm][j]=xtemp_[i][j+k];
            }
        }
        ioffp=ioff0+k*8;
        for(i=1; i<9; ++i) {
            for(j=0; j<k; ++j) {
                temp2dx_[i+ioffp][j] = 0.f;
            }
            for(j=0; j<BSXSIZE-k; ++j) {
                temp2dx_[i+ioffp][j+k]=xtemp_[i][j];
            }
        }
	}
    
    nxbins = nxbins_;					
	
	return;
	
} // End xtemp3d_int



// ************************************************************************************************************ 
//! Return interpolated 3d x-template in single call
//! \param i,j - (input) template indices
//! \param xtemplate - (output) a boost 3d array containing two sets of temlate indices and the combined pixel signals
// ************************************************************************************************************ 
void SiStripTemplate::xtemp3d(int i, int j, std::vector<float>& xtemplate)

{	
    // Sum two 2-d templates to make the 3-d template
	if(i >= 0 && i < nxbins_ && j <= i) {
        for(int k=0; k<BSXSIZE; ++k) {
            xtemplate[k]=temp2dx_[i][k]+temp2dx_[j][k];	
        }
    } else {
        for(int k=0; k<BSXSIZE; ++k) {
            xtemplate[k]=0.;	
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
//! \param qclus - (input) the cluster charge in electrons 
// ************************************************************************************************************ 
int SiStripTemplate::qbin(int id, float cotalpha, float cotbeta, float qclus)
		 
{
    // Interpolate for a new set of track angles 
    
    // Local variables 
    int i, binq;
	int ilow, ihigh, Ny, Nxx, Nyx, index;
	float yratio;
	float acotb, qscale, qavg, qmin, qmin2, fq, qtotal, qcorrect, cotalpha0;
	

// Find the index corresponding to id

       index = -1;
       for(i=0; i<(int)theStripTemp_.size(); ++i) {
	
	      if(id == theStripTemp_[i].head.ID) {
	   
	         index = i;
		     break;
          }
	    }
	 
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	   if(index < 0 || index >= (int)theStripTemp_.size()) {
	      throw cms::Exception("DataCorrupt") << "SiStripTemplate::qbin can't find needed template ID = " << id << std::endl;
	   }
#else
	   assert(index >= 0 && index < (int)theStripTemp_.size());
#endif
	 
//		

// Interpolate the absolute value of cot(beta)     
    
    acotb = fabs((double)cotbeta);
	
//	qcorrect corrects the cot(alpha)=0 cluster charge for non-zero cot(alpha)	

	//	qcorrect corrects the cot(alpha)=0 cluster charge for non-zero cot(alpha)	
	
	cotalpha0 =  theStripTemp_[index].enty[0].cotalpha;
	qcorrect=std::sqrt((1.f+cotbeta*cotbeta+cotalpha*cotalpha)/(1.f+cotbeta*cotbeta+cotalpha0*cotalpha0));
					
	// Copy the charge scaling factor to the private variable     
		
	   qscale = theStripTemp_[index].head.qscale;
		
       Ny = theStripTemp_[index].head.NTy;
       Nyx = theStripTemp_[index].head.NTyx;
       Nxx = theStripTemp_[index].head.NTxx;
		
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		if(Ny < 2 || Nyx < 1 || Nxx < 2) {
			throw cms::Exception("DataCorrupt") << "template ID = " << id_current_ << "has too few entries: Ny/Nyx/Nxx = " << Ny << "/" << Nyx << "/" << Nxx << std::endl;
		}
#else
		assert(Ny > 1 && Nyx > 0 && Nxx > 1);
#endif
        
// next, loop over all y-angle entries   

	   ilow = 0;
	   yratio = 0.f;

	   if(acotb >= theStripTemp_[index].enty[Ny-1].cotbeta) {
	
	       ilow = Ny-2;
		   yratio = 1.f;
		
	   } else {
	   
	      if(acotb >= theStripTemp_[index].enty[0].cotbeta) {

             for (i=0; i<Ny-1; ++i) { 
    
                if( theStripTemp_[index].enty[i].cotbeta <= acotb && acotb < theStripTemp_[index].enty[i+1].cotbeta) {
		  
	               ilow = i;
		           yratio = (acotb - theStripTemp_[index].enty[i].cotbeta)/(theStripTemp_[index].enty[i+1].cotbeta - theStripTemp_[index].enty[i].cotbeta);
		           break;			 
		        }
	         }
		  } 
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all y-related quantities (flip displacements when flip_y)

	   qavg = (1.f - yratio)*theStripTemp_[index].enty[ilow].qavg + yratio*theStripTemp_[index].enty[ihigh].qavg;
	   qavg *= qcorrect;
	   qmin = (1.f - yratio)*theStripTemp_[index].enty[ilow].qmin + yratio*theStripTemp_[index].enty[ihigh].qmin;
	   qmin *= qcorrect;
	   qmin2 = (1.f - yratio)*theStripTemp_[index].enty[ilow].qmin2 + yratio*theStripTemp_[index].enty[ihigh].qmin2;
	   qmin2 *= qcorrect;
	   
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(qavg <= 0.f || qmin <= 0.f) {
		throw cms::Exception("DataCorrupt") << "SiStripTemplate::qbin, qavg or qmin <= 0," 
		<< " Probably someone called the generic pixel reconstruction with an illegal trajectory state" << std::endl;
	}
#else
	assert(qavg > 0.f && qmin > 0.f);
#endif
	
//  Scale the input charge to account for differences between pixelav and CMSSW simulation or data	
	
	qtotal = qscale*qclus;
	
// uncertainty and final corrections depend upon total charge bin 	   
	   
	fq = qtotal/qavg;
	if(fq > 1.5f) {
	   binq=0;
	} else {
	   if(fq > 1.0f) {
	      binq=1;
	   } else {
		  if(fq > 0.85f) {
			 binq=2;
		  } else {
			 binq=3;
		  }
	   }
	}
	
// If the charge is too small (then flag it)
	
	if(qtotal < 0.95f*qmin) {binq = 5;} else {if(qtotal < 0.95f*qmin2) {binq = 4;}}
		
    return binq;
  
} // qbin


// ************************************************************************************************************ 
//! Interpolate beta/alpha angles to produce Vavilov parameters for the charge distribution 
//! \param mpv   - (output) the Vavilov most probable charge (well, not really the most probable esp at large kappa)
//! \param sigma - (output) the Vavilov sigma parameter
//! \param kappa - (output) the Vavilov kappa parameter [0.01 (Landau-like) < kappa < 10 (Gaussian-like)
// ************************************************************************************************************ 
void SiStripTemplate::vavilov_pars(double& mpv, double& sigma, double& kappa)

{
	// Local variables 
	int i;
	int ilow, ihigh, Ny;
	float yratio, cotb, cotalpha0, arg;
	
// Interpolate in cotbeta only for the correct total path length (converts cotalpha, cotbeta into an effective cotbeta) 
	
	cotalpha0 =  theStripTemp_[index_id_].enty[0].cotalpha;
    arg = cotb_current_*cotb_current_ + cota_current_*cota_current_ - cotalpha0*cotalpha0;
    if(arg < 0.f) arg = 0.f;
	cotb = std::sqrt(arg);
		
// Copy the charge scaling factor to the private variable     
	
	Ny = theStripTemp_[index_id_].head.NTy;
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(Ny < 2) {
		throw cms::Exception("DataCorrupt") << "template ID = " << id_current_ << "has too few entries: Ny = " << Ny << std::endl;
	}
#else
	assert(Ny > 1);
#endif
	
// next, loop over all y-angle entries   
	
	ilow = 0;
	yratio = 0.f;
	
	if(cotb >= theStripTemp_[index_id_].enty[Ny-1].cotbeta) {
		
		ilow = Ny-2;
		yratio = 1.f;
		
	} else {
		
		if(cotb >= theStripTemp_[index_id_].enty[0].cotbeta) {
			
			for (i=0; i<Ny-1; ++i) { 
				
				if( theStripTemp_[index_id_].enty[i].cotbeta <= cotb && cotb < theStripTemp_[index_id_].enty[i+1].cotbeta) {
					
					ilow = i;
					yratio = (cotb - theStripTemp_[index_id_].enty[i].cotbeta)/(theStripTemp_[index_id_].enty[i+1].cotbeta - theStripTemp_[index_id_].enty[i].cotbeta);
					break;			 
				}
			}
		} 
	}
	
	ihigh=ilow + 1;
	
// Interpolate Vavilov parameters
	
	mpvvav_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].mpvvav + yratio*theStripTemp_[index_id_].enty[ihigh].mpvvav;
	sigmavav_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].sigmavav + yratio*theStripTemp_[index_id_].enty[ihigh].sigmavav;
	kappavav_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].kappavav + yratio*theStripTemp_[index_id_].enty[ihigh].kappavav;
	
// Copy to parameter list
	
	
	mpv = (double)mpvvav_;
	sigma = (double)sigmavav_;
	kappa = (double)kappavav_;
	
	return;
	
} // vavilov_pars

// ************************************************************************************************************ 
//! Interpolate beta/alpha angles to produce Vavilov parameters for the 2-cluster charge distribution 
//! \param mpv   - (output) the Vavilov most probable charge (well, not really the most probable esp at large kappa)
//! \param sigma - (output) the Vavilov sigma parameter
//! \param kappa - (output) the Vavilov kappa parameter [0.01 (Landau-like) < kappa < 10 (Gaussian-like)
// ************************************************************************************************************ 
void SiStripTemplate::vavilov2_pars(double& mpv, double& sigma, double& kappa)

{
	// Local variables 
	int i;
	int ilow, ihigh, Ny;
	float yratio, cotb, cotalpha0, arg;
	
// Interpolate in cotbeta only for the correct total path length (converts cotalpha, cotbeta into an effective cotbeta) 
	
	cotalpha0 =  theStripTemp_[index_id_].enty[0].cotalpha;
    arg = cotb_current_*cotb_current_ + cota_current_*cota_current_ - cotalpha0*cotalpha0;
    if(arg < 0.f) arg = 0.f;
	cotb = std::sqrt(arg);
	
	// Copy the charge scaling factor to the private variable     
	
	Ny = theStripTemp_[index_id_].head.NTy;
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(Ny < 2) {
		throw cms::Exception("DataCorrupt") << "template ID = " << id_current_ << "has too few entries: Ny = " << Ny << std::endl;
	}
#else
	assert(Ny > 1);
#endif
	
	// next, loop over all y-angle entries   
	
	ilow = 0;
	yratio = 0.f;
	
	if(cotb >= theStripTemp_[index_id_].enty[Ny-1].cotbeta) {
		
		ilow = Ny-2;
		yratio = 1.f;
		
	} else {
		
		if(cotb >= theStripTemp_[index_id_].enty[0].cotbeta) {
			
			for (i=0; i<Ny-1; ++i) { 
				
				if( theStripTemp_[index_id_].enty[i].cotbeta <= cotb && cotb < theStripTemp_[index_id_].enty[i+1].cotbeta) {
					
					ilow = i;
					yratio = (cotb - theStripTemp_[index_id_].enty[i].cotbeta)/(theStripTemp_[index_id_].enty[i+1].cotbeta - theStripTemp_[index_id_].enty[i].cotbeta);
					break;			 
				}
			}
		} 
	}
	
	ihigh=ilow + 1;
	
	// Interpolate Vavilov parameters
	
	mpvvav2_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].mpvvav2 + yratio*theStripTemp_[index_id_].enty[ihigh].mpvvav2;
	sigmavav2_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].sigmavav2 + yratio*theStripTemp_[index_id_].enty[ihigh].sigmavav2;
	kappavav2_ = (1.f - yratio)*theStripTemp_[index_id_].enty[ilow].kappavav2 + yratio*theStripTemp_[index_id_].enty[ihigh].kappavav2;
	
	// Copy to parameter list
	
	mpv = (double)mpvvav2_;
	sigma = (double)sigmavav2_;
	kappa = (double)kappavav2_;
	
	return;
	
} // vavilov2_pars



