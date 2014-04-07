//
//  SiPixelTemplate2D.cc  Version 1.03 
//
//  Full 2-D templates for cluster splitting, simulated cluster reweighting, and improved cluster probability
//
// Created by Morris Swartz on 12/01/09.
// 2009 __TheJohnsHopkinsUniversity__. 
//
// V1.01 - fix qavg_ filling
// V1.02 - Add locBz to test if FPix use is out of range
// V1.03 - Fix edge checking on final template to increase template size and to properly truncate cluster
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


#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate2D.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) LogError(x)
#define LOGINFO(x) LogInfo(x)
#define ENDL " "
#include "FWCore/Utilities/interface/Exception.h"
using namespace edm;
#else
#include "SiPixelTemplate2D.h"
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
bool SiPixelTemplate2D::pushfile(int filenum, std::vector< SiPixelTemplateStore2D > & thePixelTemp_)
{
    // Add template stored in external file numbered filenum to theTemplateStore
    
    // Local variables 
    int i, j, k, l, iy, jx;
	const char *tempfile;
	//	char title[80]; remove this
    char c;
	const int code_version={16};
	
	
	
	//  Create a filename for this run 
	
	std::ostringstream tout;
	
	//  Create different path in CMSSW than standalone
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	tout << "CalibTracker/SiPixelESProducers/data/template_summary2D_zp" 
	<< std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
	std::string tempf = tout.str();
	edm::FileInPath file( tempf.c_str() );
	tempfile = (file.fullPath()).c_str();
#else
	tout << "template_summary2D_zp" << std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
	std::string tempf = tout.str();
	tempfile = tempf.c_str();
#endif
	
	//  open the template file 
	
	std::ifstream in_file(tempfile, std::ios::in);
	
	if(in_file.is_open()) {
		
		// Create a local template storage entry
		
		SiPixelTemplateStore2D theCurrentTemp;
		
		// Read-in a header string first and print it    
		
		for (i=0; (c=in_file.get()) != '\n'; ++i) {
			if(i < 79) {theCurrentTemp.head.title[i] = c;}
		}
		if(i > 78) {i=78;}
		theCurrentTemp.head.title[i+1] ='\0';
		LOGINFO("SiPixelTemplate2D") << "Loading Pixel Template File - " << theCurrentTemp.head.title << ENDL;
		
		// next, the header information     
		
		in_file >> theCurrentTemp.head.ID  >> theCurrentTemp.head.templ_version >> theCurrentTemp.head.Bfield >> theCurrentTemp.head.NTy >> theCurrentTemp.head.NTyx >> theCurrentTemp.head.NTxx
		>> theCurrentTemp.head.Dtype >> theCurrentTemp.head.Vbias >> theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >> theCurrentTemp.head.qscale
		>> theCurrentTemp.head.s50 >> theCurrentTemp.head.lorywidth >> theCurrentTemp.head.lorxwidth >> theCurrentTemp.head.ysize >> theCurrentTemp.head.xsize >> theCurrentTemp.head.zsize;
		
		if(in_file.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file, no template load" << ENDL; return false;}
		
		LOGINFO("SiPixelTemplate2D") << "Template ID = " << theCurrentTemp.head.ID << ", Template Version " << theCurrentTemp.head.templ_version << ", Bfield = " << theCurrentTemp.head.Bfield 
		<< ", NTy = " << theCurrentTemp.head.NTy << ", NTyx = " << theCurrentTemp.head.NTyx<< ", NTxx = " << theCurrentTemp.head.NTxx << ", Dtype = " << theCurrentTemp.head.Dtype
		<< ", Bias voltage " << theCurrentTemp.head.Vbias << ", temperature "
		<< theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence << ", Q-scaling factor " << theCurrentTemp.head.qscale
		<< ", 1/2 threshold " << theCurrentTemp.head.s50 << ", y Lorentz Width " << theCurrentTemp.head.lorywidth << ", x Lorentz width " << theCurrentTemp.head.lorxwidth    
		<< ", pixel x-size " << theCurrentTemp.head.xsize << ", y-size " << theCurrentTemp.head.ysize << ", zsize " << theCurrentTemp.head.zsize << ENDL;
		
		if(theCurrentTemp.head.templ_version != code_version) {LOGERROR("SiPixelTemplate2D") << "code expects version " << code_version << ", no template load" << ENDL; return false;}
		
		if(theCurrentTemp.head.NTy != 0) {LOGERROR("SiPixelTemplate2D") << "Trying to load 1-d template info into the 2-d template object, check your DB/global tag!" << ENDL; return false;}
		
		// next, layout the 2-d structure needed to store template
		
		theCurrentTemp.entry.resize(boost::extents[theCurrentTemp.head.NTyx][theCurrentTemp.head.NTxx]);
		
		// Read in the file info
		
		for (iy=0; iy < theCurrentTemp.head.NTyx; ++iy) {    
			for(jx=0; jx < theCurrentTemp.head.NTxx; ++jx) {
			
			   in_file >> theCurrentTemp.entry[iy][jx].runnum >> theCurrentTemp.entry[iy][jx].costrk[0] 
			   >> theCurrentTemp.entry[iy][jx].costrk[1] >> theCurrentTemp.entry[iy][jx].costrk[2]; 
			
			   if(in_file.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 1, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
			
			// Calculate cot(alpha) and cot(beta) for this entry 
			
			   theCurrentTemp.entry[iy][jx].cotalpha = theCurrentTemp.entry[iy][jx].costrk[0]/theCurrentTemp.entry[iy][jx].costrk[2];
			
			   theCurrentTemp.entry[iy][jx].cotbeta = theCurrentTemp.entry[iy][jx].costrk[1]/theCurrentTemp.entry[iy][jx].costrk[2];
			
			   in_file >> theCurrentTemp.entry[iy][jx].qavg >> theCurrentTemp.entry[iy][jx].pixmax >> theCurrentTemp.entry[iy][jx].sxymax >> theCurrentTemp.entry[iy][jx].iymin
			   >> theCurrentTemp.entry[iy][jx].iymax >> theCurrentTemp.entry[iy][jx].jxmin >> theCurrentTemp.entry[iy][jx].jxmax;
			
			   if(in_file.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 2, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
			
			   for (k=0; k<2; ++k) {
				
				  in_file >> theCurrentTemp.entry[iy][jx].xypar[k][0] >> theCurrentTemp.entry[iy][jx].xypar[k][1] 
				  >> theCurrentTemp.entry[iy][jx].xypar[k][2] >> theCurrentTemp.entry[iy][jx].xypar[k][3] >> theCurrentTemp.entry[iy][jx].xypar[k][4];
				
				  if(in_file.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 3, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
				
			   }
				
			   for (k=0; k<2; ++k) {
					
				  in_file >> theCurrentTemp.entry[iy][jx].lanpar[k][0] >> theCurrentTemp.entry[iy][jx].lanpar[k][1] 
				  >> theCurrentTemp.entry[iy][jx].lanpar[k][2] >> theCurrentTemp.entry[iy][jx].lanpar[k][3] >> theCurrentTemp.entry[iy][jx].lanpar[k][4];
					
				  if(in_file.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 4, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
					
			   }
				
			   for (l=0; l<7; ++l) {	
			      for (k=0; k<7; ++k) {
						for (j=0; j<T2XSIZE; ++j) {
				
				        for (i=0; i<T2YSIZE; ++i) {in_file >> theCurrentTemp.entry[iy][jx].xytemp[k][l][i][j];}
				
				        if(in_file.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 5, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
				      }
			      }
			   }
						
				
			   in_file >> theCurrentTemp.entry[iy][jx].chi2minone >> theCurrentTemp.entry[iy][jx].chi2avgone >> theCurrentTemp.entry[iy][jx].chi2min[0] >> theCurrentTemp.entry[iy][jx].chi2avg[0]
				   >> theCurrentTemp.entry[iy][jx].chi2min[1] >> theCurrentTemp.entry[iy][jx].chi2avg[1]>> theCurrentTemp.entry[iy][jx].chi2min[2] >> theCurrentTemp.entry[iy][jx].chi2avg[2]
				   >> theCurrentTemp.entry[iy][jx].chi2min[3] >> theCurrentTemp.entry[iy][jx].chi2avg[3];
				
			   if(in_file.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 6, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
			
			   in_file >> theCurrentTemp.entry[iy][jx].spare[0] >> theCurrentTemp.entry[iy][jx].spare[1] >> theCurrentTemp.entry[iy][jx].spare[2] >> theCurrentTemp.entry[iy][jx].spare[3]
				    >> theCurrentTemp.entry[iy][jx].spare[4] >> theCurrentTemp.entry[iy][jx].spare[5] >> theCurrentTemp.entry[iy][jx].spare[6] >> theCurrentTemp.entry[iy][jx].spare[7]
				    >> theCurrentTemp.entry[iy][jx].spare[8]  >> theCurrentTemp.entry[iy][jx].spare[9];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 7, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
					
			   in_file >> theCurrentTemp.entry[iy][jx].spare[10] >> theCurrentTemp.entry[iy][jx].spare[11] >> theCurrentTemp.entry[iy][jx].spare[12] >> theCurrentTemp.entry[iy][jx].spare[13]
				>> theCurrentTemp.entry[iy][jx].spare[14] >> theCurrentTemp.entry[iy][jx].spare[15] >> theCurrentTemp.entry[iy][jx].spare[16] >> theCurrentTemp.entry[iy][jx].spare[17]
				>> theCurrentTemp.entry[iy][jx].spare[18]  >> theCurrentTemp.entry[iy][jx].spare[19];
				
				if(in_file.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 8, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
		   }
			
		}
				
		
		in_file.close();
		
		// Add this template to the store
		
		thePixelTemp_.push_back(theCurrentTemp);
		
		return true;
		
	} else {
		
		// If file didn't open, report this
		
		LOGERROR("SiPixelTemplate2D") << "Error opening File" << tempfile << ENDL;
		return false;
		
   }
	
} // TempInit 


#ifndef SI_PIXEL_TEMPLATE_STANDALONE

//**************************************************************** 
//! This routine initializes the global template structures from an
//! external file template_summary_zpNNNN where NNNN are four digits 
//! \param dbobject - db storing multiple template calibrations
//**************************************************************** 
bool SiPixelTemplate2D::pushfile(const SiPixelTemplateDBObject& dbobject, std::vector< SiPixelTemplateStore2D > & thePixelTemp_)
{
	// Add template stored in external dbobject to theTemplateStore
    
	// Local variables 
	int i, j, k, l, iy, jx;
	//	const char *tempfile;
	const int code_version={16};
	
	// We must create a new object because dbobject must be a const and our stream must not be
	SiPixelTemplateDBObject db = dbobject;
	
	// Create a local template storage entry
	SiPixelTemplateStore2D theCurrentTemp;
	
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
		LOGINFO("SiPixelTemplate2D") << "Loading Pixel Template File - " << theCurrentTemp.head.title << ENDL;
		
		// next, the header information     
		
		db >> theCurrentTemp.head.ID  >> theCurrentTemp.head.templ_version >> theCurrentTemp.head.Bfield >> theCurrentTemp.head.NTy >> theCurrentTemp.head.NTyx >> theCurrentTemp.head.NTxx
		>> theCurrentTemp.head.Dtype >> theCurrentTemp.head.Vbias >> theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >> theCurrentTemp.head.qscale
		>> theCurrentTemp.head.s50 >> theCurrentTemp.head.lorywidth >> theCurrentTemp.head.lorxwidth >> theCurrentTemp.head.ysize >> theCurrentTemp.head.xsize >> theCurrentTemp.head.zsize;
		
		if(db.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file, no template load" << ENDL; return false;}
		
		LOGINFO("SiPixelTemplate2D") << "Template ID = " << theCurrentTemp.head.ID << ", Template Version " << theCurrentTemp.head.templ_version << ", Bfield = " << theCurrentTemp.head.Bfield 
		<< ", NTy = " << theCurrentTemp.head.NTy << ", NTyx = " << theCurrentTemp.head.NTyx<< ", NTxx = " << theCurrentTemp.head.NTxx << ", Dtype = " << theCurrentTemp.head.Dtype
		<< ", Bias voltage " << theCurrentTemp.head.Vbias << ", temperature "
		<< theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence << ", Q-scaling factor " << theCurrentTemp.head.qscale
		<< ", 1/2 threshold " << theCurrentTemp.head.s50 << ", y Lorentz Width " << theCurrentTemp.head.lorywidth << ", x Lorentz width " << theCurrentTemp.head.lorxwidth    
		<< ", pixel x-size " << theCurrentTemp.head.xsize << ", y-size " << theCurrentTemp.head.ysize << ", zsize " << theCurrentTemp.head.zsize << ENDL;
		
		if(theCurrentTemp.head.templ_version != code_version) {LOGERROR("SiPixelTemplate2D") << "code expects version " << code_version << ", no template load" << ENDL; return false;}
		
		if(theCurrentTemp.head.NTy != 0) {LOGERROR("SiPixelTemplate2D") << "Trying to load 1-d template info into the 2-d template object, check your DB/global tag!" << ENDL; return false;}
		
		// next, layout the 2-d structure needed to store template
		
		theCurrentTemp.entry.resize(boost::extents[theCurrentTemp.head.NTyx][theCurrentTemp.head.NTxx]);
		
		// Read in the file info
		
		for (iy=0; iy < theCurrentTemp.head.NTyx; ++iy) {    
			for(jx=0; jx < theCurrentTemp.head.NTxx; ++jx) {
				
				db >> theCurrentTemp.entry[iy][jx].runnum >> theCurrentTemp.entry[iy][jx].costrk[0] 
				>> theCurrentTemp.entry[iy][jx].costrk[1] >> theCurrentTemp.entry[iy][jx].costrk[2]; 
				
				if(db.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 1, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
				
				// Calculate cot(alpha) and cot(beta) for this entry 
				
				theCurrentTemp.entry[iy][jx].cotalpha = theCurrentTemp.entry[iy][jx].costrk[0]/theCurrentTemp.entry[iy][jx].costrk[2];
				
				theCurrentTemp.entry[iy][jx].cotbeta = theCurrentTemp.entry[iy][jx].costrk[1]/theCurrentTemp.entry[iy][jx].costrk[2];
				
				db >> theCurrentTemp.entry[iy][jx].qavg >> theCurrentTemp.entry[iy][jx].pixmax >> theCurrentTemp.entry[iy][jx].sxymax >> theCurrentTemp.entry[iy][jx].iymin
				>> theCurrentTemp.entry[iy][jx].iymax >> theCurrentTemp.entry[iy][jx].jxmin >> theCurrentTemp.entry[iy][jx].jxmax;
				
				if(db.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 2, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
				
				for (k=0; k<2; ++k) {
					
					db >> theCurrentTemp.entry[iy][jx].xypar[k][0] >> theCurrentTemp.entry[iy][jx].xypar[k][1] 
					>> theCurrentTemp.entry[iy][jx].xypar[k][2] >> theCurrentTemp.entry[iy][jx].xypar[k][3] >> theCurrentTemp.entry[iy][jx].xypar[k][4];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 3, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
					
				}
				
				for (k=0; k<2; ++k) {
					
					db >> theCurrentTemp.entry[iy][jx].lanpar[k][0] >> theCurrentTemp.entry[iy][jx].lanpar[k][1] 
					>> theCurrentTemp.entry[iy][jx].lanpar[k][2] >> theCurrentTemp.entry[iy][jx].lanpar[k][3] >> theCurrentTemp.entry[iy][jx].lanpar[k][4];
					
					if(db.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 4, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
					
				}
				
				for (l=0; l<7; ++l) {	
					for (k=0; k<7; ++k) {
						for (j=0; j<T2XSIZE; ++j) {
							
							for (i=0; i<T2YSIZE; ++i) {db >> theCurrentTemp.entry[iy][jx].xytemp[k][l][i][j];}
							
							if(db.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 5, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
						}
					}
				}
				
				
				db >> theCurrentTemp.entry[iy][jx].chi2minone >> theCurrentTemp.entry[iy][jx].chi2avgone >> theCurrentTemp.entry[iy][jx].chi2min[0] >> theCurrentTemp.entry[iy][jx].chi2avg[0]
				>> theCurrentTemp.entry[iy][jx].chi2min[1] >> theCurrentTemp.entry[iy][jx].chi2avg[1]>> theCurrentTemp.entry[iy][jx].chi2min[2] >> theCurrentTemp.entry[iy][jx].chi2avg[2]
				>> theCurrentTemp.entry[iy][jx].chi2min[3] >> theCurrentTemp.entry[iy][jx].chi2avg[3];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 6, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
				
			   db >> theCurrentTemp.entry[iy][jx].spare[0] >> theCurrentTemp.entry[iy][jx].spare[1] >> theCurrentTemp.entry[iy][jx].spare[2] >> theCurrentTemp.entry[iy][jx].spare[3]
				>> theCurrentTemp.entry[iy][jx].spare[4] >> theCurrentTemp.entry[iy][jx].spare[5] >> theCurrentTemp.entry[iy][jx].spare[6] >> theCurrentTemp.entry[iy][jx].spare[7]
				>> theCurrentTemp.entry[iy][jx].spare[8]  >> theCurrentTemp.entry[iy][jx].spare[9];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 7, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
				
			   db >> theCurrentTemp.entry[iy][jx].spare[10] >> theCurrentTemp.entry[iy][jx].spare[11] >> theCurrentTemp.entry[iy][jx].spare[12] >> theCurrentTemp.entry[iy][jx].spare[13]
				>> theCurrentTemp.entry[iy][jx].spare[14] >> theCurrentTemp.entry[iy][jx].spare[15] >> theCurrentTemp.entry[iy][jx].spare[16] >> theCurrentTemp.entry[iy][jx].spare[17]
				>> theCurrentTemp.entry[iy][jx].spare[18]  >> theCurrentTemp.entry[iy][jx].spare[19];
				
				if(db.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 8, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; return false;}
					
				}
			}
			
		}
		
				
// Add this template to the store
		
		thePixelTemp_.push_back(theCurrentTemp);
		
	return true;
	
} // TempInit 

#endif



// *************************************************************************************************************************************
//! Interpolate stored 2-D information for input angles and hit position to make a 2-D template   
//! \param         id - (input) the id of the template 
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014) 
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)  
//! \param      locBz - (input) the sign of this quantity is used to determine whether to flip cot(beta)<0 quantities from cot(beta)>0 (FPix only)
//!                             for FPix IP-related tracks, locBz < 0 for cot(beta) > 0 and locBz > 0 for cot(beta) < 0
//! \param       xhit - (input) x-position of hit relative to the lower left corner of pixel[1][1] (to allow for the "padding" of the two-d clusters in the splitter)
//! \param       yhit - (input) y-position of hit relative to the lower left corner of pixel[1][1] 
//! \param    ydouble - (input) STL vector of 21 element array to flag a double-pixel starting at cluster[1][1]
//! \param    xdouble - (input) STL vector of 11 element array to flag a double-pixel starting at cluster[1][1]
//! \param template2d - (output) 2d template of size matched to the cluster.  Input must be zeroed since charge is added only.
// *************************************************************************************************************************************

bool SiPixelTemplate2D::xytemp(int id, float cotalpha, float cotbeta, float locBz, float xhit, float yhit, std::vector<bool>& ydouble, std::vector<bool>& xdouble, float template2d[BXM2][BYM2])
{
    // Interpolate for a new set of track angles 
    
    // Local variables 
	int i, j;
	int pixx, pixy, k0, k1, l0, l1, imidx, deltax, deltay, iflipy, imin, imax, jmin, jmax;
	int m, n;
	float acotb, dcota, dcotb, dx, dy, ddx, ddy, adx, ady, tmpxy;
	bool flip_y;
//	std::vector <float> xrms(4), xgsig(4), xrmsc2m(4), xgsigc2m(4);
	std::vector <float> chi2xavg(4), chi2xmin(4);


// Check to see if interpolation is valid     

	if(id != id_current_ || cotalpha != cota_current_ || cotbeta != cotb_current_) {
		
		cota_current_ = cotalpha; cotb_current_ = cotbeta; success_ = true;
		
		if(id != id_current_) {
			
			// Find the index corresponding to id
			
			index_id_ = -1;
			for(i=0; i<(int)thePixelTemp_.size(); ++i) {
				
				if(id == thePixelTemp_[i].head.ID) {
					
					index_id_ = i;
					id_current_ = id;
					
// Copy the charge scaling factor to the private variable     
					
					Dtype_ = thePixelTemp_[index_id_].head.Dtype;
					
// Copy the charge scaling factor to the private variable     
					
					qscale_ = thePixelTemp_[index_id_].head.qscale;
					
// Copy the pseudopixel signal size to the private variable     
					
					s50_ = thePixelTemp_[index_id_].head.s50;
					
// Copy the Lorentz widths to private variables     
					
					lorywidth_ = thePixelTemp_[index_id_].head.lorywidth;
					lorxwidth_ = thePixelTemp_[index_id_].head.lorxwidth;
					
// Copy the pixel sizes private variables    
					
					xsize_ = thePixelTemp_[index_id_].head.xsize;
					ysize_ = thePixelTemp_[index_id_].head.ysize;
					zsize_ = thePixelTemp_[index_id_].head.zsize;
					
// Determine the size of this template    
					
					Nyx_ = thePixelTemp_[index_id_].head.NTyx;
					Nxx_ = thePixelTemp_[index_id_].head.NTxx;
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
					if(Nyx_ < 2 || Nxx_ < 2) {
						throw cms::Exception("DataCorrupt") << "template ID = " << id_current_ << "has too few entries: Nyx/Nxx = " << Nyx_ << "/" << Nxx_ << std::endl;
					}
#else
					assert(Nyx_ > 1 && Nxx_ > 1);
#endif
					imidx = Nxx_/2;
					
					cotalpha0_ =  thePixelTemp_[index_id_].entry[0][0].cotalpha;
					cotalpha1_ =  thePixelTemp_[index_id_].entry[0][Nxx_-1].cotalpha;
					deltacota_ = (cotalpha1_-cotalpha0_)/(float)(Nxx_-1);
					
					cotbeta0_ =  thePixelTemp_[index_id_].entry[0][imidx].cotbeta;
					cotbeta1_ =  thePixelTemp_[index_id_].entry[Nyx_-1][imidx].cotbeta;
					deltacotb_ = (cotbeta1_-cotbeta0_)/(float)(Nyx_-1);
					
					break;
				}
			}
		}
	}
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(index_id_ < 0 || index_id_ >= (int)thePixelTemp_.size()) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate2D::interpolate can't find needed template ID = " << id 
		<< ", Are you using the correct global tag?" << std::endl;
	}
#else
	assert(index_id_ >= 0 && index_id_ < (int)thePixelTemp_.size());
#endif
	
// Check angle limits and et up interpolation parameters  
	
	if(cotalpha < cotalpha0_) {
		success_ = false;
		jx0_ = 0;
		jx1_ = 1;
		adcota_ = 0.f;
	} else if(cotalpha > cotalpha1_) {
	    success_ = false;
		jx0_ = Nxx_ - 1;
		jx1_ = jx0_ - 1;
		adcota_ = 0.f;
	} else {
	   jx0_ = (int)((cotalpha-cotalpha0_)/deltacota_+0.5f);
	   dcota = (cotalpha - (cotalpha0_ + jx0_*deltacota_))/deltacota_;
	   adcota_ = fabs(dcota);
	   if(dcota > 0.f) {jx1_ = jx0_ + 1;if(jx1_ > Nxx_-1) jx1_ = jx0_-1;} else {jx1_ = jx0_ - 1; if(jx1_ < 0) jx1_ = jx0_+1;}		
	}
		
// Interpolate the absolute value of cot(beta)     
    
	acotb = std::abs(cotbeta);
	
	if(acotb < cotbeta0_) {
	    success_ = false;
		iy0_ = 0;
		iy1_ = 1;
		adcotb_ = 0.f;
	} else if(acotb > cotbeta1_) {
	    success_ = false;
		iy0_ = Nyx_ - 1;
		iy1_ = iy0_ - 1;
		adcotb_ = 0.f;
	} else {
		iy0_ = (int)((acotb-cotbeta0_)/deltacotb_+0.5f);
		dcotb = (acotb - (cotbeta0_ + iy0_*deltacotb_))/deltacotb_;
		adcotb_ = fabs(dcotb);
		if(dcotb > 0.f) {iy1_ = iy0_ + 1; if(iy1_ > Nyx_-1) iy1_ = iy0_-1;} else {iy1_ = iy0_ - 1; if(iy1_ < 0) iy1_ = iy0_+1;}		
    }	
	
// This works only for IP-related tracks 
	
	flip_y = false;
	if(cotbeta < 0.f) {flip_y = true;}
	
// If Fpix-related track has wrong cotbeta-field correlation, return false to trigger simple template instead	
	
	if(Dtype_ == 1) {
		if(cotbeta*locBz > 0.f) success_ = false;
	}
	
// Interpolate things in cot(alpha)-cot(beta)	
	
	qavg_ = thePixelTemp_[index_id_].entry[iy0_][jx0_].qavg
	+adcota_*(thePixelTemp_[index_id_].entry[iy0_][jx1_].qavg - thePixelTemp_[index_id_].entry[iy0_][jx0_].qavg)
	+adcotb_*(thePixelTemp_[index_id_].entry[iy1_][jx0_].qavg - thePixelTemp_[index_id_].entry[iy0_][jx0_].qavg);
	
	pixmax_ = thePixelTemp_[index_id_].entry[iy0_][jx0_].pixmax
	+adcota_*(thePixelTemp_[index_id_].entry[iy0_][jx1_].pixmax - thePixelTemp_[index_id_].entry[iy0_][jx0_].pixmax)
	+adcotb_*(thePixelTemp_[index_id_].entry[iy1_][jx0_].pixmax - thePixelTemp_[index_id_].entry[iy0_][jx0_].pixmax);
	
	sxymax_ = thePixelTemp_[index_id_].entry[iy0_][jx0_].sxymax
	+adcota_*(thePixelTemp_[index_id_].entry[iy0_][jx1_].sxymax - thePixelTemp_[index_id_].entry[iy0_][jx0_].sxymax)
	+adcotb_*(thePixelTemp_[index_id_].entry[iy1_][jx0_].sxymax - thePixelTemp_[index_id_].entry[iy0_][jx0_].sxymax);
	
	chi2avgone_ = thePixelTemp_[index_id_].entry[iy0_][jx0_].chi2avgone
	+adcota_*(thePixelTemp_[index_id_].entry[iy0_][jx1_].chi2avgone - thePixelTemp_[index_id_].entry[iy0_][jx0_].chi2avgone)
	+adcotb_*(thePixelTemp_[index_id_].entry[iy1_][jx0_].chi2avgone - thePixelTemp_[index_id_].entry[iy0_][jx0_].chi2avgone);
	
	chi2minone_ = thePixelTemp_[index_id_].entry[iy0_][jx0_].chi2minone
	+adcota_*(thePixelTemp_[index_id_].entry[iy0_][jx1_].chi2minone - thePixelTemp_[index_id_].entry[iy0_][jx0_].chi2minone)
	+adcotb_*(thePixelTemp_[index_id_].entry[iy1_][jx0_].chi2minone - thePixelTemp_[index_id_].entry[iy0_][jx0_].chi2minone);
	
	for(i=0; i<4 ; ++i) {
		chi2avg_[i] = thePixelTemp_[index_id_].entry[iy0_][jx0_].chi2avg[i]
		+adcota_*(thePixelTemp_[index_id_].entry[iy0_][jx1_].chi2avg[i] - thePixelTemp_[index_id_].entry[iy0_][jx0_].chi2avg[i])
		+adcotb_*(thePixelTemp_[index_id_].entry[iy1_][jx0_].chi2avg[i] - thePixelTemp_[index_id_].entry[iy0_][jx0_].chi2avg[i]);
		
		chi2min_[i] = thePixelTemp_[index_id_].entry[iy0_][jx0_].chi2min[i]
		+adcota_*(thePixelTemp_[index_id_].entry[iy0_][jx1_].chi2min[i] - thePixelTemp_[index_id_].entry[iy0_][jx0_].chi2min[i])
		+adcotb_*(thePixelTemp_[index_id_].entry[iy1_][jx0_].chi2min[i] - thePixelTemp_[index_id_].entry[iy0_][jx0_].chi2min[i]);
	}
	
	for(i=0; i<2 ; ++i) {
		for(j=0; j<5 ; ++j) {
			// Charge loss switches sides when cot(beta) changes sign
			if(flip_y) {
				xypary0x0_[1-i][j] = thePixelTemp_[index_id_].entry[iy0_][jx0_].xypar[i][j];
				xypary1x0_[1-i][j] = thePixelTemp_[index_id_].entry[iy1_][jx0_].xypar[i][j];
				xypary0x1_[1-i][j] = thePixelTemp_[index_id_].entry[iy0_][jx1_].xypar[i][j];
				lanpar_[1-i][j] = thePixelTemp_[index_id_].entry[iy0_][jx0_].lanpar[i][j]
				+adcota_*(thePixelTemp_[index_id_].entry[iy0_][jx1_].lanpar[i][j] - thePixelTemp_[index_id_].entry[iy0_][jx0_].lanpar[i][j])
				+adcotb_*(thePixelTemp_[index_id_].entry[iy1_][jx0_].lanpar[i][j] - thePixelTemp_[index_id_].entry[iy0_][jx0_].lanpar[i][j]);
			} else {
				xypary0x0_[i][j] = thePixelTemp_[index_id_].entry[iy0_][jx0_].xypar[i][j];
				xypary1x0_[i][j] = thePixelTemp_[index_id_].entry[iy1_][jx0_].xypar[i][j];
				xypary0x1_[i][j] = thePixelTemp_[index_id_].entry[iy0_][jx1_].xypar[i][j];
				lanpar_[i][j] = thePixelTemp_[index_id_].entry[iy0_][jx0_].lanpar[i][j]
				+adcota_*(thePixelTemp_[index_id_].entry[iy0_][jx1_].lanpar[i][j] - thePixelTemp_[index_id_].entry[iy0_][jx0_].lanpar[i][j])
				+adcotb_*(thePixelTemp_[index_id_].entry[iy1_][jx0_].lanpar[i][j] - thePixelTemp_[index_id_].entry[iy0_][jx0_].lanpar[i][j]);
			}
		}
	}
	
// next, determine the indices of the closest point in k (y-displacement), l (x-displacement)
// pixy and pixx are the indices of the struck pixel in the (Ty,Tx) system
// k0,k1 are the k-indices of the closest and next closest point
// l0,l1 are the l-indices of the closest and next closest point

	pixy = (int)floorf(yhit/ysize_);
	dy = yhit-(pixy+0.5f)*ysize_;
	if(flip_y) {dy = -dy;}
	k0 = (int)(dy/ysize_*6.f+3.5f);
	if(k0 < 0) k0 = 0;
	if(k0 > 6) k0 = 6;
	ddy = 6.f*dy/ysize_ - (k0-3);
	ady = fabs(ddy);
	if(ddy > 0.f) {k1 = k0 + 1; if(k1 > 6) k1 = k0-1;} else {k1 = k0 - 1; if(k1 < 0) k1 = k0+1;}
	pixx = (int)floorf(xhit/xsize_);
	dx = xhit-(pixx+0.5f)*xsize_;
	l0 = (int)(dx/xsize_*6.f+3.5f);
	if(l0 < 0) l0 = 0;
	if(l0 > 6) l0 = 6;
	ddx = 6.f*dx/xsize_ - (l0-3);
	adx = fabs(ddx);
	if(ddx > 0.f) {l1 = l0 + 1; if(l1 > 6) l1 = l0-1;} else {l1 = l0 - 1; if(l1 < 0) l1 = l0+1;}
	
// OK, lets do the template interpolation.  
	
// First find the limits of the indices for non-zero pixels
	
	imin = std::min(thePixelTemp_[index_id_].entry[iy0_][jx0_].iymin,thePixelTemp_[index_id_].entry[iy1_][jx0_].iymin);
	imin = std::min(imin,thePixelTemp_[index_id_].entry[iy0_][jx1_].iymin);
	
	jmin = std::min(thePixelTemp_[index_id_].entry[iy0_][jx0_].jxmin,thePixelTemp_[index_id_].entry[iy1_][jx0_].jxmin);
	jmin = std::min(jmin,thePixelTemp_[index_id_].entry[iy0_][jx1_].jxmin);
	
	imax = std::max(thePixelTemp_[index_id_].entry[iy0_][jx0_].iymax,thePixelTemp_[index_id_].entry[iy1_][jx0_].iymax);
	imax = std::max(imax,thePixelTemp_[index_id_].entry[iy0_][jx1_].iymax);
	
	jmax = std::max(thePixelTemp_[index_id_].entry[iy0_][jx0_].jxmax,thePixelTemp_[index_id_].entry[iy1_][jx0_].jxmax);
	jmax = std::max(jmax,thePixelTemp_[index_id_].entry[iy0_][jx1_].jxmax);
		
// Calculate the x and y offsets to make the new template
	
// First, shift the struck pixel coordinates to the (Ty+2, Tx+2) system
	
	++pixy; ++pixx;
	
// In the template store, the struck pixel is always (THy,THx)
	
	deltax = pixx - T2HX;
	deltay = pixy - T2HY;
	
//  First zero the local 2-d template
	
	for(j=0; j<BXM2; ++j) {for(i=0; i<BYM2; ++i) {xytemp_[j][i] = 0.f;}}
	
// Loop over the non-zero part of the template index space and interpolate
	
	for(j=jmin; j<=jmax; ++j) {
	   for(i=imin; i<=imax; ++i) {
			m = deltax+j;
			
// If cot(beta) < 0, we must flip the cluster, iflipy is the flipped y-index
			
			if(flip_y) {iflipy=T2YSIZE-1-i; n = deltay+iflipy;} else {n = deltay+i;}
			if(m>=0 && m<=BXM3 && n>=0 && n<=BYM3) {
			   tmpxy = thePixelTemp_[index_id_].entry[iy0_][jx0_].xytemp[k0][l0][i][j] 
				   + adx*(thePixelTemp_[index_id_].entry[iy0_][jx0_].xytemp[k0][l1][i][j] - thePixelTemp_[index_id_].entry[iy0_][jx0_].xytemp[k0][l0][i][j])
				   + ady*(thePixelTemp_[index_id_].entry[iy0_][jx0_].xytemp[k1][l0][i][j] - thePixelTemp_[index_id_].entry[iy0_][jx0_].xytemp[k0][l0][i][j])
				   + adcota_*(thePixelTemp_[index_id_].entry[iy0_][jx1_].xytemp[k0][l0][i][j] - thePixelTemp_[index_id_].entry[iy0_][jx0_].xytemp[k0][l0][i][j])
				   + adcotb_*(thePixelTemp_[index_id_].entry[iy1_][jx0_].xytemp[k0][l0][i][j] - thePixelTemp_[index_id_].entry[iy0_][jx0_].xytemp[k0][l0][i][j]);
			   if(tmpxy > 0.f) {xytemp_[m][n] = tmpxy;} else {xytemp_[m][n] = 0.f;}
			}
		}
	}
	
//combine rows and columns to simulate double pixels
	
	for(n=1; n<BYM3; ++n) {
		if(ydouble[n-1]) {
//  Combine the y-columns
			for(m=1; m<BXM3; ++m) {
				xytemp_[m][n] += xytemp_[m][n+1];
			}
//  Now shift the remaining pixels over by one column
			for(i=n+1; i<BYM3; ++i) {
			   for(m=1; m<BXM3; ++m) {
				   xytemp_[m][i] = xytemp_[m][i+1];
			   }
			}
		}
	}

//combine rows and columns to simulate double pixels
	
	for(m=1; m<BXM3; ++m) {
		if(xdouble[m-1]) {
//  Combine the x-rows
			for(n=1; n<BYM3; ++n) {
				xytemp_[m][n] += xytemp_[m+1][n];
			}
//  Now shift the remaining pixels over by one row
			for(j=m+1; j<BXM3; ++j) {
			   for(n=1; n<BYM3; ++n) {
				   xytemp_[j][n] = xytemp_[j+1][n];
			   }
			}
		}
	}
	
//  Finally, loop through and increment the external template

	for(n=1; n<BYM3; ++n) {
		for(m=1; m<BXM3; ++m) {
			if(xytemp_[m][n] > 0.f) {template2d[m][n] += xytemp_[m][n];}
		}
	}
	
  return success_;
} // xytemp



// *************************************************************************************************************************************
//! Interpolate stored 2-D information for input angles and hit position to make a 2-D template   
//! \param         id - (input) the id of the template 
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014) 
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)  
//! \param       xhit - (input) x-position of hit relative to the lower left corner of pixel[1][1] (to allow for the "padding" of the two-d clusters in the splitter)
//! \param       yhit - (input) y-position of hit relative to the lower left corner of pixel[1][1] 
//! \param    ydouble - (input) STL vector of 21 element array to flag a double-pixel starting at cluster[1][1]
//! \param    xdouble - (input) STL vector of 11 element array to flag a double-pixel starting at cluster[1][1]
//! \param template2d - (output) 2d template of size matched to the cluster.  Input must be zeroed since charge is added only.
// *************************************************************************************************************************************

bool SiPixelTemplate2D::xytemp(int id, float cotalpha, float cotbeta, float xhit, float yhit, std::vector<bool>& ydouble, std::vector<bool>& xdouble, float template2d[BXM2][BYM2])
{
	// Interpolate for a new set of track angles 
	
	// Local variables 
	float locBz = -1;
	if(cotbeta < 0.f) {locBz = -locBz;}

	return SiPixelTemplate2D::xytemp(id, cotalpha, cotbeta, locBz, xhit, yhit, ydouble, xdouble, template2d);
	
} // xytemp



// ************************************************************************************************************ 
//! Return y error (squared) for an input signal and yindex
//! Add large Q scaling for use in cluster splitting.
//! \param qpixel - (input) pixel charge
//! \param index - (input) y-index index of pixel
//! \param xysig2 - (output) square error
// ************************************************************************************************************ 
void SiPixelTemplate2D::xysigma2(float qpixel, int index, float& xysig2)

{
    // Interpolate using quantities already stored in the private variables
    
    // Local variables 
	float sigi, sigi2, sigi3, sigi4, qscale, err2, err00;
	
    // Make sure that input is OK
    
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if(index < 2 || index >= BYM2) {
		throw cms::Exception("DataCorrupt") << "SiPixelTemplate2D::ysigma2 called with index = " << index << std::endl;
	}
#else
	assert(index > 1 && index < BYM2);
#endif
	
	// Define the maximum signal to use in the parameterization 
	
	// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 
	
			if(qpixel < sxymax_) {
				sigi = qpixel;
				qscale = 1.f;
			} else {
				sigi = sxymax_;
				qscale = qpixel/sxymax_;
			}
			sigi2 = sigi*sigi; sigi3 = sigi2*sigi; sigi4 = sigi3*sigi;
			if(index <= THXP1) {
				err00 = xypary0x0_[0][0]+xypary0x0_[0][1]*sigi+xypary0x0_[0][2]*sigi2+xypary0x0_[0][3]*sigi3+xypary0x0_[0][4]*sigi4;
				err2 = err00
				     +adcota_*(xypary0x1_[0][0]+xypary0x1_[0][1]*sigi+xypary0x1_[0][2]*sigi2+xypary0x1_[0][3]*sigi3+xypary0x1_[0][4]*sigi4 - err00)
				     +adcotb_*(xypary1x0_[0][0]+xypary1x0_[0][1]*sigi+xypary1x0_[0][2]*sigi2+xypary1x0_[0][3]*sigi3+xypary1x0_[0][4]*sigi4 - err00);
			} else {
				err00 = xypary0x0_[1][0]+xypary0x0_[1][1]*sigi+xypary0x0_[1][2]*sigi2+xypary0x0_[1][3]*sigi3+xypary0x0_[1][4]*sigi4;
				err2 = err00
				     +adcota_*(xypary0x1_[1][0]+xypary0x1_[1][1]*sigi+xypary0x1_[1][2]*sigi2+xypary0x1_[1][3]*sigi3+xypary0x1_[1][4]*sigi4 - err00)
				     +adcotb_*(xypary1x0_[1][0]+xypary1x0_[1][1]*sigi+xypary1x0_[1][2]*sigi2+xypary1x0_[1][3]*sigi3+xypary1x0_[1][4]*sigi4 - err00);
			}
			xysig2 =qscale*err2;
			if(xysig2 <= 0.f) {LOGERROR("SiPixelTemplate2D") << "neg y-error-squared, id = " << id_current_ << ", index = " << index_id_ << 
			", cot(alpha) = " << cota_current_ << ", cot(beta) = " << cotb_current_ <<  ", sigi = " << sigi << ENDL;}
	
	return;
	
} // End xysigma2


// ************************************************************************************************************ 
//! Return the Landau probability parameters for this set of cot(alpha, cot(beta)
// ************************************************************************************************************ 
void SiPixelTemplate2D::landau_par(float lanpar[2][5])

{
	// Interpolate using quantities already stored in the private variables
	
	// Local variables 
	int i,j;
	for(i=0; i<2; ++i) {
		for(j=0; j<5; ++j) {
			lanpar[i][j] = lanpar_[i][j];
		}
	}
	return;
	
} // End lan_par



