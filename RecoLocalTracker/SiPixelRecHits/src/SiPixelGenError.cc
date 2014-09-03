//
//  SiPixelGenError.cc  Version 1.00
//
//  Object stores Lorentz widths, bias corrections, and errors for the Generic Algorithm
//  
//  Created by Morris Swartz on 10/27/06.
//  Add some debugging messages. d.k. 5/14
//

//#include <stdlib.h>
//#include <stdio.h>
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include <math.h>
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
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelGenError.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) LogError(x)
#define LOGWARNING(x) LogWarning(x)
#define LOGINFO(x) LogInfo(x)
#define ENDL " "
#include "FWCore/Utilities/interface/Exception.h"
using namespace edm;
#else
#include "SiPixelGenError.h"
#define LOGERROR(x) std::cout << x << ": "
#define LOGINFO(x) std::cout << x << ": "
#define LOGWARNING(x) std::cout << x << ": "
#define ENDL std::endl
#endif

//**************************************************************** 
//! This routine initializes the global GenError structures from
//! an external file generror_summary_zpNNNN where NNNN are four
//! digits of filenum.                                           
//! \param filenum - an integer NNNN used in the filename generror_summary_zpNNNN
//**************************************************************** 
bool SiPixelGenError::pushfile(int filenum, std::vector< SiPixelGenErrorStore > & thePixelTemp_)
{
    // Add info stored in external file numbered filenum to theGenErrorStore
    
    // Local variables 
    int i, j, k;
	float costrk[3]={0,0,0};
	const char *tempfile;
	//	char title[80]; remove this
    char c;
	const int code_version={1};
	
	
	
	//  Create a filename for this run 
	
	std::ostringstream tout;
	
	//  Create different path in CMSSW than standalone
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	tout << "CalibTracker/SiPixelESProducers/data/generror_summary_zp"
	<< std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
	std::string tempf = tout.str();
	edm::FileInPath file( tempf.c_str() );
	tempfile = (file.fullPath()).c_str();
#else
	tout << "generror_summary_zp" << std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
	std::string tempf = tout.str();
	tempfile = tempf.c_str();
#endif
	
	//  open the Generic file 
	
	std::ifstream in_file(tempfile, std::ios::in);
	
	if(in_file.is_open()) {
		
		// Create a local GenError storage entry
		
		SiPixelGenErrorStore theCurrentTemp;
		
		// Read-in a header string first and print it    
		
		for (i=0; (c=in_file.get()) != '\n'; ++i) {
			if(i < 79) {theCurrentTemp.head.title[i] = c;}
		}
		if(i > 78) {i=78;}
		theCurrentTemp.head.title[i+1] ='\0';
		LOGINFO("SiPixelGenError") << "Loading Pixel GenError File - " << theCurrentTemp.head.title << ENDL;
		
		// next, the header information     
		
		in_file >> theCurrentTemp.head.ID  >> theCurrentTemp.head.templ_version >> theCurrentTemp.head.Bfield >> theCurrentTemp.head.NTy >> theCurrentTemp.head.NTyx >> theCurrentTemp.head.NTxx >> theCurrentTemp.head.Dtype >> theCurrentTemp.head.Vbias >>
          theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >> theCurrentTemp.head.qscale >> theCurrentTemp.head.s50 >>
          theCurrentTemp.head.lorywidth >> theCurrentTemp.head.lorxwidth >> theCurrentTemp.head.ysize >> theCurrentTemp.head.xsize >>
          theCurrentTemp.head.zsize >> theCurrentTemp.head.ss50 >> theCurrentTemp.head.lorybias >> theCurrentTemp.head.lorxbias >>
          theCurrentTemp.head.fbin[0] >> theCurrentTemp.head.fbin[1] >> theCurrentTemp.head.fbin[2];
		
		if(in_file.fail()) {LOGERROR("SiPixelGenError") << "Error reading file, no GenError load" << ENDL; return false;}
				
		LOGINFO("SiPixelGenError") << "GenError ID = " << theCurrentTemp.head.ID << ", GenError Version " << theCurrentTemp.head.templ_version << ", Bfield = " << theCurrentTemp.head.Bfield
		<< ", NTy = " << theCurrentTemp.head.NTy << ", NTyx = " << theCurrentTemp.head.NTyx<< ", NTxx = " << theCurrentTemp.head.NTxx << ", Dtype = " << theCurrentTemp.head.Dtype
		<< ", Bias voltage " << theCurrentTemp.head.Vbias << ", temperature "
		<< theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence << ", Q-scaling factor " << theCurrentTemp.head.qscale
		<< ", 1/2 multi dcol threshold " << theCurrentTemp.head.s50 << ", 1/2 single dcol threshold " << theCurrentTemp.head.ss50
      << ", y Lorentz Width " << theCurrentTemp.head.lorywidth << ", y Lorentz Bias " << theCurrentTemp.head.lorybias
      << ", x Lorentz width " << theCurrentTemp.head.lorxwidth << ", x Lorentz Bias " << theCurrentTemp.head.lorxbias
      << ", Q/Q_avg fractions for Qbin defs " << theCurrentTemp.head.fbin[0] << ", " << theCurrentTemp.head.fbin[1]
      << ", " << theCurrentTemp.head.fbin[2]
		<< ", pixel x-size " << theCurrentTemp.head.xsize << ", y-size " << theCurrentTemp.head.ysize << ", zsize " << theCurrentTemp.head.zsize << ENDL;
		
		if(theCurrentTemp.head.templ_version < code_version) {LOGERROR("SiPixelGenError") << "code expects version " << code_version << ", no GenError load" << ENDL; return false;}
		
#ifdef SI_PIXEL_TEMPLATE_USE_BOOST 
		
// next, layout the 1-d/2-d structures needed to store GenError info
				
		theCurrentTemp.enty.resize(boost::extents[theCurrentTemp.head.NTy]);

		theCurrentTemp.entx.resize(boost::extents[theCurrentTemp.head.NTyx][theCurrentTemp.head.NTxx]);
		
#endif
		
// next, loop over all y-angle entries   
		
		for (i=0; i < theCurrentTemp.head.NTy; ++i) {     
			
			in_file >> theCurrentTemp.enty[i].runnum >> costrk[0] >> costrk[1] >> costrk[2];
			
			if(in_file.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 1, no GenError load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			// Calculate the alpha, beta, and cot(beta) for this entry
			
			theCurrentTemp.enty[i].cotalpha = costrk[0]/costrk[2];
						
			theCurrentTemp.enty[i].cotbeta = costrk[1]/costrk[2];
			
			in_file >> theCurrentTemp.enty[i].qavg >> theCurrentTemp.enty[i].pixmax >> theCurrentTemp.enty[i].dyone
			>> theCurrentTemp.enty[i].syone >> theCurrentTemp.enty[i].dxone >> theCurrentTemp.enty[i].sxone;
			
			if(in_file.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 2, no GenError load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			in_file >> theCurrentTemp.enty[i].dytwo >> theCurrentTemp.enty[i].sytwo >> theCurrentTemp.enty[i].dxtwo 
			>> theCurrentTemp.enty[i].sxtwo >> theCurrentTemp.enty[i].qmin >>  theCurrentTemp.enty[i].qmin2;
			
			if(in_file.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 3, no GenError load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			if(theCurrentTemp.enty[i].qmin <= 0.) {LOGERROR("SiPixelGenError") << "Error in GenError ID " << theCurrentTemp.head.ID << " qmin = " << theCurrentTemp.enty[i].qmin << ", run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			
			
			for (j=0; j<4; ++j) {
				
				in_file >> theCurrentTemp.enty[i].yavggen[j] >> theCurrentTemp.enty[i].yrmsgen[j] >> theCurrentTemp.enty[i].xavggen[j] >> theCurrentTemp.enty[i].xrmsgen[j];
				
				if(in_file.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 14a, no GenError load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
			}
			
						
		}
		
		// next, loop over all barrel x-angle entries   
		
		for (k=0; k < theCurrentTemp.head.NTyx; ++k) { 
			
			for (i=0; i < theCurrentTemp.head.NTxx; ++i) { 
				
				in_file >> theCurrentTemp.entx[k][i].runnum >> costrk[0] >> costrk[1] >> costrk[2];
				
				if(in_file.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 17, no GenError load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				// Calculate the alpha, beta, and cot(beta) for this entry 
				
				theCurrentTemp.entx[k][i].cotalpha = costrk[0]/costrk[2];
				
				theCurrentTemp.entx[k][i].cotbeta = costrk[1]/costrk[2];
				
				in_file >> theCurrentTemp.entx[k][i].qavg >> theCurrentTemp.entx[k][i].pixmax >> theCurrentTemp.entx[k][i].dyone
				>> theCurrentTemp.entx[k][i].syone >> theCurrentTemp.entx[k][i].dxone >> theCurrentTemp.entx[k][i].sxone;
				
				if(in_file.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 18, no GenError load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				in_file >> theCurrentTemp.entx[k][i].dytwo >> theCurrentTemp.entx[k][i].sytwo >> theCurrentTemp.entx[k][i].dxtwo 
				>> theCurrentTemp.entx[k][i].sxtwo >> theCurrentTemp.entx[k][i].qmin >> theCurrentTemp.entx[k][i].qmin2;
				//			   >> theCurrentTemp.entx[k][i].mpvvav >> theCurrentTemp.entx[k][i].sigmavav >> theCurrentTemp.entx[k][i].kappavav;
				
				if(in_file.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 19, no GenError load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				
				
				for (j=0; j<4; ++j) {
					
					in_file >> theCurrentTemp.entx[k][i].yavggen[j] >> theCurrentTemp.entx[k][i].yrmsgen[j] >> theCurrentTemp.entx[k][i].xavggen[j] >> theCurrentTemp.entx[k][i].xrmsgen[j];
					
					if(in_file.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 30a, no GenError load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
				}
												
			}
		}	
		
		
		in_file.close();
		
		// Add this info to the store
		
		thePixelTemp_.push_back(theCurrentTemp);
      
		postInit(thePixelTemp_);
		
		return true;
		
	} else {
		
		// If file didn't open, report this
		
		LOGERROR("SiPixelGenError") << "Error opening File" << tempfile << ENDL;
		return false;
		
	}
	
} // TempInit 


#ifndef SI_PIXEL_TEMPLATE_STANDALONE

//**************************************************************** 
//! This routine initializes the global GenError structures from an
//! SiPixelGenErrorDBObject
//! \param dbobject - db storing multiple generic calibrations
//**************************************************************** 
bool SiPixelGenError::pushfile(const SiPixelGenErrorDBObject& dbobject, 
			       std::vector< SiPixelGenErrorStore > & thePixelTemp_) {
  // Add GenError stored in external dbobject to theGenErrorStore
    
  // Local variables 
  int i, j, k;
  float costrk[3]={0,0,0};
  //	const char *tempfile;
  const int code_version={1};
  
  // We must create a new object because dbobject must be a const and our stream must not be
  SiPixelGenErrorDBObject db = dbobject;
  
  // Create a local GenError storage entry
  SiPixelGenErrorStore theCurrentTemp;
  
  // Fill the GenError storage for each GenError calibration stored in the db
  for(int m=0; m<db.numOfTempl(); ++m) {
		
    // Read-in a header string first and print it    
    
    SiPixelGenErrorDBObject::char2float temp;
    for (i=0; i<20; ++i) {
      temp.f = db.sVector()[db.index()];
      theCurrentTemp.head.title[4*i] = temp.c[0];
      theCurrentTemp.head.title[4*i+1] = temp.c[1];
      theCurrentTemp.head.title[4*i+2] = temp.c[2];
      theCurrentTemp.head.title[4*i+3] = temp.c[3];
      db.incrementIndex(1);
    }

    theCurrentTemp.head.title[79] = '\0';
    LOGINFO("SiPixelGenError") << "Loading Pixel GenError File - " 
			       << theCurrentTemp.head.title << ENDL;
    
    // next, the header information     
		
    db >> theCurrentTemp.head.ID  >> theCurrentTemp.head.templ_version >> theCurrentTemp.head.Bfield >> theCurrentTemp.head.NTy >> theCurrentTemp.head.NTyx >> theCurrentTemp.head.NTxx >> theCurrentTemp.head.Dtype >> theCurrentTemp.head.Vbias >>
      theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >> theCurrentTemp.head.qscale >> theCurrentTemp.head.s50 >>
      theCurrentTemp.head.lorywidth >> theCurrentTemp.head.lorxwidth >> theCurrentTemp.head.ysize >> theCurrentTemp.head.xsize >>
      theCurrentTemp.head.zsize >> theCurrentTemp.head.ss50 >> theCurrentTemp.head.lorybias >> theCurrentTemp.head.lorxbias >>
      theCurrentTemp.head.fbin[0] >> theCurrentTemp.head.fbin[1] >> theCurrentTemp.head.fbin[2];
    
    if(db.fail()) {LOGERROR("SiPixelGenError") 
	<< "Error reading file, no GenError load" << ENDL; return false;}
      
    LOGINFO("SiPixelGenError") << "GenError ID = " << theCurrentTemp.head.ID << ", GenError Version " << theCurrentTemp.head.templ_version << ", Bfield = " << theCurrentTemp.head.Bfield
			       << ", NTy = " << theCurrentTemp.head.NTy << ", NTyx = " << theCurrentTemp.head.NTyx<< ", NTxx = " << theCurrentTemp.head.NTxx << ", Dtype = " << theCurrentTemp.head.Dtype
			       << ", Bias voltage " << theCurrentTemp.head.Vbias << ", temperature "
			       << theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence << ", Q-scaling factor " << theCurrentTemp.head.qscale
			       << ", 1/2 multi dcol threshold " << theCurrentTemp.head.s50 << ", 1/2 single dcol threshold " << theCurrentTemp.head.ss50
			       << ", y Lorentz Width " << theCurrentTemp.head.lorywidth << ", y Lorentz Bias " << theCurrentTemp.head.lorybias
			       << ", x Lorentz width " << theCurrentTemp.head.lorxwidth << ", x Lorentz Bias " << theCurrentTemp.head.lorxbias
			       << ", Q/Q_avg fractions for Qbin defs " << theCurrentTemp.head.fbin[0] << ", " << theCurrentTemp.head.fbin[1]
			       << ", " << theCurrentTemp.head.fbin[2]
			       << ", pixel x-size " << theCurrentTemp.head.xsize << ", y-size " << theCurrentTemp.head.ysize << ", zsize " << theCurrentTemp.head.zsize << ENDL;
    
    LOGINFO("SiPixelGenError") << "Loading Pixel GenError - " 
				  << theCurrentTemp.head.title << " version "
				  <<theCurrentTemp.head.templ_version <<" code v."
				  <<code_version<<ENDL;
    if(theCurrentTemp.head.templ_version < code_version) {
      LOGERROR("SiPixelGenError") << "code expects version " << code_version 
				  << ", no GenError load" << ENDL; return false;}
    		
#ifdef SI_PIXEL_TEMPLATE_USE_BOOST 
		
// next, layout the 1-d/2-d structures needed to store GenError
		
    theCurrentTemp.enty.resize(boost::extents[theCurrentTemp.head.NTy]);
    
    theCurrentTemp.entx.resize(boost::extents[theCurrentTemp.head.NTyx][theCurrentTemp.head.NTxx]);
    
#endif
    
    // next, loop over all barrel y-angle entries   
    
    for (i=0; i < theCurrentTemp.head.NTy; ++i) {     
      
      db >> theCurrentTemp.enty[i].runnum >> costrk[0] >> costrk[1] >> costrk[2];
      
      if(db.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 1, no GenError load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
      
      // Calculate the alpha, beta, and cot(beta) for this entry 
      
      theCurrentTemp.enty[i].cotalpha = costrk[0]/costrk[2];
      
      theCurrentTemp.enty[i].cotbeta = costrk[1]/costrk[2];
      
      db >> theCurrentTemp.enty[i].qavg >> theCurrentTemp.enty[i].pixmax >> theCurrentTemp.enty[i].dyone
	 >> theCurrentTemp.enty[i].syone >> theCurrentTemp.enty[i].dxone >> theCurrentTemp.enty[i].sxone;
      
      if(db.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 2, no GenError load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
      
      db >> theCurrentTemp.enty[i].dytwo >> theCurrentTemp.enty[i].sytwo >> theCurrentTemp.enty[i].dxtwo 
	 >> theCurrentTemp.enty[i].sxtwo >> theCurrentTemp.enty[i].qmin >> theCurrentTemp.enty[i].qmin2;
      
      for (j=0; j<4; ++j) {
	
	db >> theCurrentTemp.enty[i].yavggen[j] >> theCurrentTemp.enty[i].yrmsgen[j] >> theCurrentTemp.enty[i].xavggen[j] >> theCurrentTemp.enty[i].xrmsgen[j];
	
	if(db.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 14a, no GenError load, run # " << theCurrentTemp.enty[i].runnum << ENDL; return false;}
      }
      
    }
    
    // next, loop over all barrel x-angle entries   
    
    for (k=0; k < theCurrentTemp.head.NTyx; ++k) { 
      
      for (i=0; i < theCurrentTemp.head.NTxx; ++i) { 
	
	db >> theCurrentTemp.entx[k][i].runnum >> costrk[0] >> costrk[1] >> costrk[2];
	
	if(db.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 17, no GenError load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
	
	// Calculate the alpha, beta, and cot(beta) for this entry 
	
	theCurrentTemp.entx[k][i].cotalpha = costrk[0]/costrk[2];
	
	theCurrentTemp.entx[k][i].cotbeta = costrk[1]/costrk[2];
	
	db >> theCurrentTemp.entx[k][i].qavg >> theCurrentTemp.entx[k][i].pixmax >> theCurrentTemp.entx[k][i].dyone
	   >> theCurrentTemp.entx[k][i].syone >> theCurrentTemp.entx[k][i].dxone >> theCurrentTemp.entx[k][i].sxone;
	
	if(db.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 18, no GenError load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
	
	db >> theCurrentTemp.entx[k][i].dytwo >> theCurrentTemp.entx[k][i].sytwo >> theCurrentTemp.entx[k][i].dxtwo 
	   >> theCurrentTemp.entx[k][i].sxtwo >> theCurrentTemp.entx[k][i].qmin >> theCurrentTemp.entx[k][i].qmin2;
	
	for (j=0; j<4; ++j) {
	  
	  db >> theCurrentTemp.entx[k][i].yavggen[j] >> theCurrentTemp.entx[k][i].yrmsgen[j] >> theCurrentTemp.entx[k][i].xavggen[j] >> theCurrentTemp.entx[k][i].xrmsgen[j];
	  
	  if(db.fail()) {LOGERROR("SiPixelGenError") << "Error reading file 30a, no GenError load, run # " << theCurrentTemp.entx[k][i].runnum << ENDL; return false;}
	}
	
      }
    }	
    
		
    // Add this GenError to the store
		
    thePixelTemp_.push_back(theCurrentTemp);
      
    postInit(thePixelTemp_);
    
  }
  return true;
  
} // TempInit 

#endif

void SiPixelGenError::postInit(std::vector< SiPixelGenErrorStore > & thePixelTemp_) {
   
   for (auto & templ : thePixelTemp_) {
      for ( auto iy=0; iy<templ.head.NTy; ++iy ) templ.cotbetaY[iy]=templ.enty[iy].cotbeta;
      for ( auto iy=0; iy<templ.head.NTyx; ++iy )  templ.cotbetaX[iy]= templ.entx[iy][0].cotbeta;
      for ( auto ix=0; ix<templ.head.NTxx; ++ix ) templ.cotalphaX[ix]=templ.entx[0][ix].cotalpha;
   }
   
}




// ************************************************************************************************************
//! Interpolate beta/alpha angles to produce an expected average charge. Return int (0-4) describing the charge
//! of the cluster [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin].
//! \param id - (input) index of the GenError to use
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
// a simpler method just to return the LA
int SiPixelGenError::qbin(int id) {
   // Find the index corresponding to id
   
  if(id != id_current_) {
    
    index_id_ = -1;
    for( int i=0; i<(int)thePixelTemp_.size(); ++i) {
      if(id == thePixelTemp_[i].head.ID) {
	index_id_ = i;
	id_current_ = id;
	// 
	lorywidth_ = thePixelTemp_[i].head.lorywidth;
	lorxwidth_ = thePixelTemp_[i].head.lorxwidth;
	lorybias_ = thePixelTemp_[i].head.lorybias;
	lorxbias_ = thePixelTemp_[i].head.lorxbias;
	
	//for(int j=0; j<3; ++j) {fbin_[j] = thePixelTemp_[i].head.fbin[j];}	

	// Pixel sizes to the private variables        
	xsize_ = thePixelTemp_[i].head.xsize;
	ysize_ = thePixelTemp_[i].head.ysize;
	zsize_ = thePixelTemp_[i].head.zsize;
        
	break;
      }
    }
  }
  return index_id_;
}
  //-----------------------------------------------------------------------
  // Full method 
int SiPixelGenError::qbin(int id, float cotalpha, float cotbeta, float locBz, float qclus, 
			  float& pixmx, float& sigmay, float& deltay, float& sigmax, float& deltax,
                          float& sy1, float& dy1, float& sy2, float& dy2, float& sx1, float& dx1, 
			  float& sx2, float& dx2)
{
   // Interpolate for a new set of track angles
   
   
   // Find the index corresponding to id
   
   
   if(id != id_current_) {
         
      index_id_ = -1;
      for( int i=0; i<(int)thePixelTemp_.size(); ++i) {
         if(id == thePixelTemp_[i].head.ID) {
            index_id_ = i;
            id_current_ = id;
            lorywidth_ = thePixelTemp_[i].head.lorywidth;
            lorxwidth_ = thePixelTemp_[i].head.lorxwidth;
            lorybias_ = thePixelTemp_[i].head.lorybias;
            lorxbias_ = thePixelTemp_[i].head.lorxbias;
            for(int j=0; j<3; ++j) {fbin_[j] = thePixelTemp_[i].head.fbin[j];}
			
         // Pixel sizes to the private variables
         
            xsize_ = thePixelTemp_[i].head.xsize;
            ysize_ = thePixelTemp_[i].head.ysize;
            zsize_ = thePixelTemp_[i].head.zsize;
         
            break;
         }
      }
   }
   
   int index = index_id_;
   
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
   if(index < 0 || index >= (int)thePixelTemp_.size()) {
      throw cms::Exception("DataCorrupt") << "SiPixelGenError::qbin can't find needed GenError ID = " << id << std::endl;
   }
#else
   assert(index >= 0 && index < (int)thePixelTemp_.size());
#endif
   
   //
   
   
   auto const & templ = thePixelTemp_[index];
   
   // Interpolate the absolute value of cot(beta)
   
   auto acotb = std::abs(cotbeta);
   
   //	qcorrect corrects the cot(alpha)=0 cluster charge for non-zero cot(alpha)
	
   auto cotalpha0 =  thePixelTemp_[index].enty[0].cotalpha;
   auto qcorrect=std::sqrt((1.f+cotbeta*cotbeta+cotalpha*cotalpha)/(1.f+cotbeta*cotbeta+cotalpha0*cotalpha0));
   
   // for some cosmics, the ususal gymnastics are incorrect
   
   float cotb; bool flip_y;
   if(thePixelTemp_[index].head.Dtype == 0) {
      cotb = acotb;
      flip_y = false;
      if(cotbeta < 0.f) {flip_y = true;}
   } else {
      if(locBz < 0.f) {
         cotb = cotbeta;
         flip_y = false;
      } else {
         cotb = -cotbeta;
         flip_y = true;
      }
   }
   
   // Copy the charge scaling factor to the private variable
   
   auto qscale = thePixelTemp_[index].head.qscale;
   
   
   /*
    lorywidth = thePixelTemp_[index].head.lorywidth;
    if(locBz > 0.f) {lorywidth = -lorywidth;}
    lorxwidth = thePixelTemp_[index].head.lorxwidth;
    */
   
   
   auto Ny = thePixelTemp_[index].head.NTy;
   auto Nyx = thePixelTemp_[index].head.NTyx;
   auto Nxx = thePixelTemp_[index].head.NTxx;
   
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
   if(Ny < 2 || Nyx < 1 || Nxx < 2) {
      throw cms::Exception("DataCorrupt") << "GenError ID = " << id_current_ << "has too few entries: Ny/Nyx/Nxx = " << Ny << "/" << Nyx << "/" << Nxx << std::endl;
   }
#else
   assert(Ny > 1 && Nyx > 0 && Nxx > 1);
#endif
   
   // next, loop over all y-angle entries
   
   auto ilow = 0;
   auto ihigh = 0;
   auto yratio = 0.f;
   
   {
      auto j = std::lower_bound(templ.cotbetaY,templ.cotbetaY+Ny,cotb);
      if (j==templ.cotbetaY+Ny) { --j;  yratio = 1.f; }
      else if (j==templ.cotbetaY) { ++j; yratio = 0.f;}
      else { yratio = (cotb - (*(j-1)) )/( (*j) - (*(j-1)) ) ; }
      
      ihigh = j-templ.cotbetaY;
      ilow = ihigh-1;
   }
   
   
   
   // Interpolate/store all y-related quantities (flip displacements when flip_y)
   
   dy1 = (1.f - yratio)*thePixelTemp_[index].enty[ilow].dyone + yratio*thePixelTemp_[index].enty[ihigh].dyone;
   if(flip_y) {dy1 = -dy1;}
   sy1 = (1.f - yratio)*thePixelTemp_[index].enty[ilow].syone + yratio*thePixelTemp_[index].enty[ihigh].syone;
   dy2 = (1.f - yratio)*thePixelTemp_[index].enty[ilow].dytwo + yratio*thePixelTemp_[index].enty[ihigh].dytwo;
   if(flip_y) {dy2 = -dy2;}
   sy2 = (1.f - yratio)*thePixelTemp_[index].enty[ilow].sytwo + yratio*thePixelTemp_[index].enty[ihigh].sytwo;
   
   auto qavg = (1.f - yratio)*thePixelTemp_[index].enty[ilow].qavg + yratio*thePixelTemp_[index].enty[ihigh].qavg;
   qavg *= qcorrect;
   auto qmin = (1.f - yratio)*thePixelTemp_[index].enty[ilow].qmin + yratio*thePixelTemp_[index].enty[ihigh].qmin;
   qmin *= qcorrect;
   auto qmin2 = (1.f - yratio)*thePixelTemp_[index].enty[ilow].qmin2 + yratio*thePixelTemp_[index].enty[ihigh].qmin2;
   qmin2 *= qcorrect;
   
   
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
   if(qavg <= 0.f || qmin <= 0.f) {
      throw cms::Exception("DataCorrupt") << "SiPixelGenError::qbin, qavg or qmin <= 0,"
      << " Probably someone called the generic pixel reconstruction with an illegal trajectory state" << std::endl;
   }
#else
   assert(qavg > 0.f && qmin > 0.f);
#endif
   
   //  Scale the input charge to account for differences between pixelav and CMSSW simulation or data
   
   auto qtotal = qscale*qclus;
   
   // uncertainty and final corrections depend upon total charge bin
   
   auto fq = qtotal/qavg;
   int  binq;
   if(fq > fbin_[0]) {
      binq=0;
   } else {
      if(fq > fbin_[1]) {
         binq=1;
      } else {
         if(fq > fbin_[2]) {
            binq=2;
         } else {
            binq=3;
         }
      }
   }
   
   auto yavggen =(1.f - yratio)*thePixelTemp_[index].enty[ilow].yavggen[binq] + yratio*thePixelTemp_[index].enty[ihigh].yavggen[binq];
   if(flip_y) {yavggen = -yavggen;}
   auto yrmsgen =(1.f - yratio)*thePixelTemp_[index].enty[ilow].yrmsgen[binq] + yratio*thePixelTemp_[index].enty[ihigh].yrmsgen[binq];
   
   
   // next, loop over all x-angle entries, first, find relevant y-slices
   
   auto iylow = 0;
   auto iyhigh = 0;
   auto yxratio = 0.f;
   
   
   {
      auto j = std::lower_bound(templ.cotbetaX,templ.cotbetaX+Nyx,acotb);
      if (j==templ.cotbetaX+Nyx) { --j;  yxratio = 1.f; }
      else if (j==templ.cotbetaX) { ++j; yxratio = 0.f;}
      else { yxratio = (acotb - (*(j-1)) )/( (*j) - (*(j-1)) ) ; }
      
      iyhigh = j-templ.cotbetaX;
      iylow = iyhigh -1;
   }
   
   
   
   ilow = ihigh = 0;
   auto xxratio = 0.f;
   
   {
      auto j = std::lower_bound(templ.cotalphaX,templ.cotalphaX+Nxx,cotalpha);
      if (j==templ.cotalphaX+Nxx) { --j;  xxratio = 1.f; }
      else if (j==templ.cotalphaX) { ++j; xxratio = 0.f;}
      else { xxratio = (cotalpha - (*(j-1)) )/( (*j) - (*(j-1)) ) ; }
      
      ihigh = j-templ.cotalphaX;
      ilow = ihigh-1;
   }
   
   
   
   dx1 = (1.f - xxratio)*thePixelTemp_[index].entx[0][ilow].dxone + xxratio*thePixelTemp_[index].entx[0][ihigh].dxone;
   sx1 = (1.f - xxratio)*thePixelTemp_[index].entx[0][ilow].sxone + xxratio*thePixelTemp_[index].entx[0][ihigh].sxone;
   dx2 = (1.f - xxratio)*thePixelTemp_[index].entx[0][ilow].dxtwo + xxratio*thePixelTemp_[index].entx[0][ihigh].dxtwo;
   sx2 = (1.f - xxratio)*thePixelTemp_[index].entx[0][ilow].sxtwo + xxratio*thePixelTemp_[index].entx[0][ihigh].sxtwo;
   
   // pixmax is the maximum allowed pixel charge (used for truncation)
   
   pixmx=(1.f - yxratio)*((1.f - xxratio)*thePixelTemp_[index].entx[iylow][ilow].pixmax + xxratio*thePixelTemp_[index].entx[iylow][ihigh].pixmax)
   +yxratio*((1.f - xxratio)*thePixelTemp_[index].entx[iyhigh][ilow].pixmax + xxratio*thePixelTemp_[index].entx[iyhigh][ihigh].pixmax);
   
   auto xavggen = (1.f - yxratio)*((1.f - xxratio)*thePixelTemp_[index].entx[iylow][ilow].xavggen[binq] + xxratio*thePixelTemp_[index].entx[iylow][ihigh].xavggen[binq])
   +yxratio*((1.f - xxratio)*thePixelTemp_[index].entx[iyhigh][ilow].xavggen[binq] + xxratio*thePixelTemp_[index].entx[iyhigh][ihigh].xavggen[binq]);
   
   auto xrmsgen = (1.f - yxratio)*((1.f - xxratio)*thePixelTemp_[index].entx[iylow][ilow].xrmsgen[binq] + xxratio*thePixelTemp_[index].entx[iylow][ihigh].xrmsgen[binq])
   +yxratio*((1.f - xxratio)*thePixelTemp_[index].entx[iyhigh][ilow].xrmsgen[binq] + xxratio*thePixelTemp_[index].entx[iyhigh][ihigh].xrmsgen[binq]);
   
   
   
   //  Take the errors and bias from the correct charge bin
	
   sigmay = yrmsgen; deltay = yavggen;
   
   sigmax = xrmsgen; deltax = xavggen;
   
   // If the charge is too small (then flag it)
   
   if(qtotal < 0.95f*qmin) {binq = 5;} else {if(qtotal < 0.95f*qmin2) {binq = 4;}}
   
   return binq;
   
} // qbin
