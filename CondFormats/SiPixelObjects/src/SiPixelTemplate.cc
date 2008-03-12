//
//  SiPixelTemplate.cc  Version 3.43 
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
//! This routine initializes the global template structures from 
//! an external file template_summary_zpNNNN where NNNN are four  
//! digits of filenum.                                           
//! \param filenum - an integer NNNN used in the filename template_summary_zpNNNN
//**************************************************************** 
bool SiPixelTemplate::pushfile(int filenum)
{
    // Add template stored in external file numbered filenum to theTemplateStore
    
    // Local variables 
    int i, j, k;
	const char *tempfile;
	char title[80];
    char c;
	const int code_version={8};
	


//  Create a filename for this run 

 //std::ostringstream tout;
 //tout << "template_summary_zp" << std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
 //std::string tempf = tout.str();
 //tempfile = tempf.c_str();
	
 
 std::ostringstream tout;
 tout << "RecoLocalTracker/SiPixelRecHits/data/template_summary_zp" 
      << std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
 std::string tempf = tout.str();
 
 // std::cout << "tempf = " << tempf << std::endl;
 edm::FileInPath file( tempf.c_str() );
 tempfile = (file.fullPath()).c_str();
 // std::cout << "tempfile = " << tempfile << std::endl;
 

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
    
    in_file >> theCurrentTemp.head.ID >> theCurrentTemp.head.NBy >> theCurrentTemp.head.NByx >> theCurrentTemp.head.NBxx
	        >> theCurrentTemp.head.NFy >> theCurrentTemp.head.NFyx >> theCurrentTemp.head.NFxx >> theCurrentTemp.head.vbias 
			>> theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >> theCurrentTemp.head.qscale
			>> theCurrentTemp.head.s50 >> theCurrentTemp.head.templ_version;
			
	if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	
    LOGINFO("SiPixelTemplate") << "Template ID = " << theCurrentTemp.head.ID << ", NBy = " << theCurrentTemp.head.NBy << ", NByx = " << theCurrentTemp.head.NByx 
		 << ", NBxx = " << theCurrentTemp.head.NBxx << ", NFy = " << theCurrentTemp.head.NFy << ", NFyx = " << theCurrentTemp.head.NFyx
		 << ", NFxx = " << theCurrentTemp.head.NFxx << ", bias voltage " << theCurrentTemp.head.vbias << ", temperature "
		 << theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence << ", Q-scaling factor " << theCurrentTemp.head.qscale
		 << ", 1/2 threshold " << theCurrentTemp.head.s50 << ", Template Version " << theCurrentTemp.head.templ_version << ENDL;    
			
	if(theCurrentTemp.head.templ_version != code_version) {LOGERROR("SiPixelTemplate") << "code expects version " << code_version << ", no template load" << ENDL; return false;}
		 
// next, loop over all barrel y-angle entries   

    for (i=0; i < theCurrentTemp.head.NBy; ++i) {     
    
       in_file >> theCurrentTemp.entby[i].runnum >> theCurrentTemp.entby[i].costrk[0] 
	           >> theCurrentTemp.entby[i].costrk[1] >> theCurrentTemp.entby[i].costrk[2]; 
			
	   if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
			  
// Calculate the alpha, beta, and cot(beta) for this entry 

       theCurrentTemp.entby[i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entby[i].costrk[2], (double)theCurrentTemp.entby[i].costrk[0]));
	   
	   theCurrentTemp.entby[i].cotalpha = theCurrentTemp.entby[i].costrk[0]/theCurrentTemp.entby[i].costrk[2];

       theCurrentTemp.entby[i].beta = static_cast<float>(atan2((double)theCurrentTemp.entby[i].costrk[2], (double)theCurrentTemp.entby[i].costrk[1]));
	   
	   theCurrentTemp.entby[i].cotbeta = theCurrentTemp.entby[i].costrk[1]/theCurrentTemp.entby[i].costrk[2];
    
       in_file >> theCurrentTemp.entby[i].qavg >> theCurrentTemp.entby[i].symax >> theCurrentTemp.entby[i].dyone
	           >> theCurrentTemp.entby[i].syone >> theCurrentTemp.entby[i].sxmax >> theCurrentTemp.entby[i].dxone >> theCurrentTemp.entby[i].sxone;
			
       if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
    
       in_file >> theCurrentTemp.entby[i].dytwo >> theCurrentTemp.entby[i].sytwo >> theCurrentTemp.entby[i].dxtwo 
	           >> theCurrentTemp.entby[i].sxtwo >> theCurrentTemp.entby[i].qmin;
			
       if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
			  
	   for (j=0; j<2; ++j) {
    
          in_file >> theCurrentTemp.entby[i].ypar[j][0] >> theCurrentTemp.entby[i].ypar[j][1] 
	              >> theCurrentTemp.entby[i].ypar[j][2] >> theCurrentTemp.entby[i].ypar[j][3] >> theCurrentTemp.entby[i].ypar[j][4];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
			  
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entby[i].ytemp[j][0] >> theCurrentTemp.entby[i].ytemp[j][1] >> theCurrentTemp.entby[i].ytemp[j][2]
	              >> theCurrentTemp.entby[i].ytemp[j][3] >> theCurrentTemp.entby[i].ytemp[j][4] >> theCurrentTemp.entby[i].ytemp[j][5]
	              >> theCurrentTemp.entby[i].ytemp[j][6] >> theCurrentTemp.entby[i].ytemp[j][7] >> theCurrentTemp.entby[i].ytemp[j][8]
	              >> theCurrentTemp.entby[i].ytemp[j][9] >> theCurrentTemp.entby[i].ytemp[j][10] >> theCurrentTemp.entby[i].ytemp[j][11]
	              >> theCurrentTemp.entby[i].ytemp[j][12] >> theCurrentTemp.entby[i].ytemp[j][13] >> theCurrentTemp.entby[i].ytemp[j][14]
	              >> theCurrentTemp.entby[i].ytemp[j][15] >> theCurrentTemp.entby[i].ytemp[j][16] >> theCurrentTemp.entby[i].ytemp[j][17]
	              >> theCurrentTemp.entby[i].ytemp[j][18] >> theCurrentTemp.entby[i].ytemp[j][19] >> theCurrentTemp.entby[i].ytemp[j][20];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
   			  
	   for (j=0; j<2; ++j) {
    
		  in_file >> theCurrentTemp.entby[i].xpar[j][0] >> theCurrentTemp.entby[i].xpar[j][1] 
	              >> theCurrentTemp.entby[i].xpar[j][2] >> theCurrentTemp.entby[i].xpar[j][3] >> theCurrentTemp.entby[i].xpar[j][4];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
			  
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entby[i].xtemp[j][0] >> theCurrentTemp.entby[i].xtemp[j][1] >> theCurrentTemp.entby[i].xtemp[j][2]
	              >> theCurrentTemp.entby[i].xtemp[j][3] >> theCurrentTemp.entby[i].xtemp[j][4] >> theCurrentTemp.entby[i].xtemp[j][5]
	              >> theCurrentTemp.entby[i].xtemp[j][6];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entby[i].yavg[j] >> theCurrentTemp.entby[i].yrms[j] >> theCurrentTemp.entby[i].ygx0[j] >> theCurrentTemp.entby[i].ygsig[j];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entby[i].yflpar[j][0] >> theCurrentTemp.entby[i].yflpar[j][1] >> theCurrentTemp.entby[i].yflpar[j][2] 
				  >> theCurrentTemp.entby[i].yflpar[j][3] >> theCurrentTemp.entby[i].yflpar[j][4] >> theCurrentTemp.entby[i].yflpar[j][5];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
  	   }
	   
	  	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entby[i].xavg[j] >> theCurrentTemp.entby[i].xrms[j] >> theCurrentTemp.entby[i].xgx0[j] >> theCurrentTemp.entby[i].xgsig[j];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entby[i].xflpar[j][0] >> theCurrentTemp.entby[i].xflpar[j][1] >> theCurrentTemp.entby[i].xflpar[j][2] 
		          >> theCurrentTemp.entby[i].xflpar[j][3] >> theCurrentTemp.entby[i].xflpar[j][4] >> theCurrentTemp.entby[i].xflpar[j][5];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entby[i].chi2yavg[j] >> theCurrentTemp.entby[i].chi2ymin[j] >> theCurrentTemp.entby[i].chi2xavg[j] >> theCurrentTemp.entby[i].chi2xmin[j];

          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   
	   in_file >> theCurrentTemp.entby[i].yspare[0] >> theCurrentTemp.entby[i].yspare[1] >> theCurrentTemp.entby[i].yspare[2] >> theCurrentTemp.entby[i].yspare[3] >> theCurrentTemp.entby[i].yspare[4]
	    >> theCurrentTemp.entby[i].yspare[5] >> theCurrentTemp.entby[i].yspare[6] >> theCurrentTemp.entby[i].yspare[7] >> theCurrentTemp.entby[i].yspare[8] >> theCurrentTemp.entby[i].yspare[9];

	   if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}

	   in_file >> theCurrentTemp.entby[i].xspare[0] >> theCurrentTemp.entby[i].xspare[1] >> theCurrentTemp.entby[i].xspare[2] >> theCurrentTemp.entby[i].xspare[3] >> theCurrentTemp.entby[i].xspare[4]
	    >> theCurrentTemp.entby[i].xspare[5] >> theCurrentTemp.entby[i].xspare[6] >> theCurrentTemp.entby[i].xspare[7] >> theCurrentTemp.entby[i].xspare[8] >> theCurrentTemp.entby[i].xspare[9];

	   if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
    	   
	}
	
// next, loop over all barrel x-angle entries   

  for (k=0; k < theCurrentTemp.head.NByx; ++k) { 

    for (i=0; i < theCurrentTemp.head.NBxx; ++i) { 
        
       in_file >> theCurrentTemp.entbx[k][i].runnum >> theCurrentTemp.entbx[k][i].costrk[0] 
	           >> theCurrentTemp.entbx[k][i].costrk[1] >> theCurrentTemp.entbx[k][i].costrk[2]; 
			
       if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
			  
// Calculate the alpha, beta, and cot(beta) for this entry 

       theCurrentTemp.entbx[k][i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entbx[k][i].costrk[2], (double)theCurrentTemp.entbx[k][i].costrk[0]));
	   
	   theCurrentTemp.entbx[k][i].cotalpha = theCurrentTemp.entbx[k][i].costrk[0]/theCurrentTemp.entbx[k][i].costrk[2];

       theCurrentTemp.entbx[k][i].beta = static_cast<float>(atan2((double)theCurrentTemp.entbx[k][i].costrk[2], (double)theCurrentTemp.entbx[k][i].costrk[1]));
	   
	   theCurrentTemp.entbx[k][i].cotbeta = theCurrentTemp.entbx[k][i].costrk[1]/theCurrentTemp.entbx[k][i].costrk[2];
    
       in_file >> theCurrentTemp.entbx[k][i].qavg >> theCurrentTemp.entbx[k][i].symax >> theCurrentTemp.entbx[k][i].dyone
	           >> theCurrentTemp.entbx[k][i].syone >> theCurrentTemp.entbx[k][i].sxmax >> theCurrentTemp.entbx[k][i].dxone >> theCurrentTemp.entbx[k][i].sxone;
			
       if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
    
       in_file >> theCurrentTemp.entbx[k][i].dytwo >> theCurrentTemp.entbx[k][i].sytwo >> theCurrentTemp.entbx[k][i].dxtwo 
	           >> theCurrentTemp.entbx[k][i].sxtwo >> theCurrentTemp.entbx[k][i].qmin;
			
       if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
			  
	   for (j=0; j<2; ++j) {
    
          in_file >> theCurrentTemp.entbx[k][i].ypar[j][0] >> theCurrentTemp.entbx[k][i].ypar[j][1] 
	              >> theCurrentTemp.entbx[k][i].ypar[j][2] >> theCurrentTemp.entbx[k][i].ypar[j][3] >> theCurrentTemp.entbx[k][i].ypar[j][4];
			  			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entbx[k][i].ytemp[j][0] >> theCurrentTemp.entbx[k][i].ytemp[j][1] >> theCurrentTemp.entbx[k][i].ytemp[j][2]
	              >> theCurrentTemp.entbx[k][i].ytemp[j][3] >> theCurrentTemp.entbx[k][i].ytemp[j][4] >> theCurrentTemp.entbx[k][i].ytemp[j][5]
	              >> theCurrentTemp.entbx[k][i].ytemp[j][6] >> theCurrentTemp.entbx[k][i].ytemp[j][7] >> theCurrentTemp.entbx[k][i].ytemp[j][8]
	              >> theCurrentTemp.entbx[k][i].ytemp[j][9] >> theCurrentTemp.entbx[k][i].ytemp[j][10] >> theCurrentTemp.entbx[k][i].ytemp[j][11]
	              >> theCurrentTemp.entbx[k][i].ytemp[j][12] >> theCurrentTemp.entbx[k][i].ytemp[j][13] >> theCurrentTemp.entbx[k][i].ytemp[j][14]
	              >> theCurrentTemp.entbx[k][i].ytemp[j][15] >> theCurrentTemp.entbx[k][i].ytemp[j][16] >> theCurrentTemp.entbx[k][i].ytemp[j][17]
	              >> theCurrentTemp.entbx[k][i].ytemp[j][18] >> theCurrentTemp.entbx[k][i].ytemp[j][19] >> theCurrentTemp.entbx[k][i].ytemp[j][20];
			
		  if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
   			  
	   for (j=0; j<2; ++j) {
    
		  in_file >> theCurrentTemp.entbx[k][i].xpar[j][0] >> theCurrentTemp.entbx[k][i].xpar[j][1] 
	              >> theCurrentTemp.entbx[k][i].xpar[j][2] >> theCurrentTemp.entbx[k][i].xpar[j][3] >> theCurrentTemp.entbx[k][i].xpar[j][4];
			  
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entbx[k][i].xtemp[j][0] >> theCurrentTemp.entbx[k][i].xtemp[j][1] >> theCurrentTemp.entbx[k][i].xtemp[j][2]
	              >> theCurrentTemp.entbx[k][i].xtemp[j][3] >> theCurrentTemp.entbx[k][i].xtemp[j][4] >> theCurrentTemp.entbx[k][i].xtemp[j][5]
	              >> theCurrentTemp.entbx[k][i].xtemp[j][6];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entbx[k][i].yavg[j] >> theCurrentTemp.entbx[k][i].yrms[j] >> theCurrentTemp.entbx[k][i].ygx0[j] >> theCurrentTemp.entbx[k][i].ygsig[j];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entbx[k][i].yflpar[j][0] >> theCurrentTemp.entbx[k][i].yflpar[j][1] >> theCurrentTemp.entbx[k][i].yflpar[j][2] 
				  >> theCurrentTemp.entbx[k][i].yflpar[j][3] >> theCurrentTemp.entbx[k][i].yflpar[j][4] >> theCurrentTemp.entbx[k][i].yflpar[j][5];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entbx[k][i].xavg[j] >> theCurrentTemp.entbx[k][i].xrms[j] >> theCurrentTemp.entbx[k][i].xgx0[j] >> theCurrentTemp.entbx[k][i].xgsig[j];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entbx[k][i].xflpar[j][0] >> theCurrentTemp.entbx[k][i].xflpar[j][1] >> theCurrentTemp.entbx[k][i].xflpar[j][2] 
		          >> theCurrentTemp.entbx[k][i].xflpar[j][3] >> theCurrentTemp.entbx[k][i].xflpar[j][4] >> theCurrentTemp.entbx[k][i].xflpar[j][5];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entbx[k][i].chi2yavg[j] >> theCurrentTemp.entbx[k][i].chi2ymin[j] >> theCurrentTemp.entbx[k][i].chi2xavg[j] >> theCurrentTemp.entbx[k][i].chi2xmin[j];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   
	   in_file >> theCurrentTemp.entbx[k][i].yspare[0] >> theCurrentTemp.entbx[k][i].yspare[1] >> theCurrentTemp.entbx[k][i].yspare[2] >> theCurrentTemp.entbx[k][i].yspare[3] >> theCurrentTemp.entbx[k][i].yspare[4]
	    >> theCurrentTemp.entbx[k][i].yspare[5] >> theCurrentTemp.entbx[k][i].yspare[6] >> theCurrentTemp.entbx[k][i].yspare[7] >> theCurrentTemp.entbx[k][i].yspare[8] >> theCurrentTemp.entbx[k][i].yspare[9];
			
	   if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}

	   in_file >> theCurrentTemp.entbx[k][i].xspare[0] >> theCurrentTemp.entbx[k][i].xspare[1] >> theCurrentTemp.entbx[k][i].xspare[2] >> theCurrentTemp.entbx[k][i].xspare[3] >> theCurrentTemp.entbx[k][i].xspare[4]
	    >> theCurrentTemp.entbx[k][i].xspare[5] >> theCurrentTemp.entbx[k][i].xspare[6] >> theCurrentTemp.entbx[k][i].xspare[7] >> theCurrentTemp.entbx[k][i].xspare[8] >> theCurrentTemp.entbx[k][i].xspare[9];
			
	   if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
    	   
	}
  }	
    
// next, loop over all forward y-angle entries   

    for (i=0; i < theCurrentTemp.head.NFy; ++i) {     
    
       in_file >> theCurrentTemp.entfy[i].runnum >> theCurrentTemp.entfy[i].costrk[0] 
	           >> theCurrentTemp.entfy[i].costrk[1] >> theCurrentTemp.entfy[i].costrk[2]; 
			
       if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
			  
// Calculate the alpha, beta, and cot(beta) for this entry 

       theCurrentTemp.entfy[i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entfy[i].costrk[2], (double)theCurrentTemp.entfy[i].costrk[0]));
	   
	   theCurrentTemp.entfy[i].cotalpha = theCurrentTemp.entfy[i].costrk[0]/theCurrentTemp.entfy[i].costrk[2];

       theCurrentTemp.entfy[i].beta = static_cast<float>(atan2((double)theCurrentTemp.entfy[i].costrk[2], (double)theCurrentTemp.entfy[i].costrk[1]));
	   
	   theCurrentTemp.entfy[i].cotbeta = theCurrentTemp.entfy[i].costrk[1]/theCurrentTemp.entfy[i].costrk[2];
    
       in_file >> theCurrentTemp.entfy[i].qavg >> theCurrentTemp.entfy[i].symax >> theCurrentTemp.entfy[i].dyone
	           >> theCurrentTemp.entfy[i].syone >> theCurrentTemp.entfy[i].sxmax >> theCurrentTemp.entfy[i].dxone >> theCurrentTemp.entfy[i].sxone;
    			
       if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   
       in_file >> theCurrentTemp.entfy[i].dytwo >> theCurrentTemp.entfy[i].sytwo >> theCurrentTemp.entfy[i].dxtwo 
	           >> theCurrentTemp.entfy[i].sxtwo >> theCurrentTemp.entfy[i].qmin;
			
       if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
			  
	   for (j=0; j<2; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].ypar[j][0] >> theCurrentTemp.entfy[i].ypar[j][1] 
	              >> theCurrentTemp.entfy[i].ypar[j][2] >> theCurrentTemp.entfy[i].ypar[j][3] >> theCurrentTemp.entfy[i].ypar[j][4];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
			  
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].ytemp[j][0] >> theCurrentTemp.entfy[i].ytemp[j][1] >> theCurrentTemp.entfy[i].ytemp[j][2]
	              >> theCurrentTemp.entfy[i].ytemp[j][3] >> theCurrentTemp.entfy[i].ytemp[j][4] >> theCurrentTemp.entfy[i].ytemp[j][5]
	              >> theCurrentTemp.entfy[i].ytemp[j][6] >> theCurrentTemp.entfy[i].ytemp[j][7] >> theCurrentTemp.entfy[i].ytemp[j][8]
	              >> theCurrentTemp.entfy[i].ytemp[j][9] >> theCurrentTemp.entfy[i].ytemp[j][10] >> theCurrentTemp.entfy[i].ytemp[j][11]
	              >> theCurrentTemp.entfy[i].ytemp[j][12] >> theCurrentTemp.entfy[i].ytemp[j][13] >> theCurrentTemp.entfy[i].ytemp[j][14]
	              >> theCurrentTemp.entfy[i].ytemp[j][15] >> theCurrentTemp.entfy[i].ytemp[j][16] >> theCurrentTemp.entfy[i].ytemp[j][17]
	              >> theCurrentTemp.entfy[i].ytemp[j][18] >> theCurrentTemp.entfy[i].ytemp[j][19] >> theCurrentTemp.entfy[i].ytemp[j][20];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
   			  
	   for (j=0; j<2; ++j) {
    
		  in_file >> theCurrentTemp.entfy[i].xpar[j][0] >> theCurrentTemp.entfy[i].xpar[j][1] 
	              >> theCurrentTemp.entfy[i].xpar[j][2] >> theCurrentTemp.entfy[i].xpar[j][3] >> theCurrentTemp.entfy[i].xpar[j][4];
			  
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].xtemp[j][0] >> theCurrentTemp.entfy[i].xtemp[j][1] >> theCurrentTemp.entfy[i].xtemp[j][2]
	              >> theCurrentTemp.entfy[i].xtemp[j][3] >> theCurrentTemp.entfy[i].xtemp[j][4] >> theCurrentTemp.entfy[i].xtemp[j][5]
	              >> theCurrentTemp.entfy[i].xtemp[j][6];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].yavg[j] >> theCurrentTemp.entfy[i].yrms[j] >> theCurrentTemp.entfy[i].ygx0[j] >> theCurrentTemp.entfy[i].ygsig[j];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].yflpar[j][0] >> theCurrentTemp.entfy[i].yflpar[j][1] >> theCurrentTemp.entfy[i].yflpar[j][2]
				  >> theCurrentTemp.entfy[i].yflpar[j][3] >> theCurrentTemp.entfy[i].yflpar[j][4] >> theCurrentTemp.entfy[i].yflpar[j][5];
 			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].xavg[j] >> theCurrentTemp.entfy[i].xrms[j] >> theCurrentTemp.entfy[i].xgx0[j] >> theCurrentTemp.entfy[i].xgsig[j];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].xflpar[j][0] >> theCurrentTemp.entfy[i].xflpar[j][1] >> theCurrentTemp.entfy[i].xflpar[j][2] 
		          >> theCurrentTemp.entfy[i].xflpar[j][3] >> theCurrentTemp.entfy[i].xflpar[j][4] >> theCurrentTemp.entfy[i].xflpar[j][5];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].chi2yavg[j] >> theCurrentTemp.entfy[i].chi2ymin[j] >> theCurrentTemp.entfy[i].chi2xavg[j] >> theCurrentTemp.entfy[i].chi2xmin[j];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   
	   in_file >> theCurrentTemp.entfy[i].yspare[0] >> theCurrentTemp.entfy[i].yspare[1] >> theCurrentTemp.entfy[i].yspare[2] >> theCurrentTemp.entfy[i].yspare[3] >> theCurrentTemp.entfy[i].yspare[4]
	    >> theCurrentTemp.entfy[i].yspare[5] >> theCurrentTemp.entfy[i].yspare[6] >> theCurrentTemp.entfy[i].yspare[7] >> theCurrentTemp.entfy[i].yspare[8] >> theCurrentTemp.entfy[i].yspare[9];
			
	   if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}

	   in_file >> theCurrentTemp.entfy[i].xspare[0] >> theCurrentTemp.entfy[i].xspare[1] >> theCurrentTemp.entfy[i].xspare[2] >> theCurrentTemp.entfy[i].xspare[3] >> theCurrentTemp.entfy[i].xspare[4]
	    >> theCurrentTemp.entfy[i].xspare[5] >> theCurrentTemp.entfy[i].xspare[6] >> theCurrentTemp.entfy[i].xspare[7] >> theCurrentTemp.entfy[i].xspare[8] >> theCurrentTemp.entfy[i].xspare[9];
			
	   if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
    	   
	}
	
// next, loop over all forward x-angle entries   

  for (k=0; k < theCurrentTemp.head.NFyx; ++k) { 
  
    for (i=0; i < theCurrentTemp.head.NFxx; ++i) {     
    
       in_file >> theCurrentTemp.entfx[k][i].runnum >> theCurrentTemp.entfx[k][i].costrk[0] 
	           >> theCurrentTemp.entfx[k][i].costrk[1] >> theCurrentTemp.entfx[k][i].costrk[2]; 
			
       if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
			  
// Calculate the alpha, beta, and cot(beta) for this entry 

       theCurrentTemp.entfx[k][i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entfx[k][i].costrk[2], (double)theCurrentTemp.entfx[k][i].costrk[0]));
	   
	   theCurrentTemp.entfx[k][i].cotalpha = theCurrentTemp.entfx[k][i].costrk[0]/theCurrentTemp.entfx[k][i].costrk[2];

       theCurrentTemp.entfx[k][i].beta = static_cast<float>(atan2((double)theCurrentTemp.entfx[k][i].costrk[2], (double)theCurrentTemp.entfx[k][i].costrk[1]));
	   
	   theCurrentTemp.entfx[k][i].cotbeta = theCurrentTemp.entfx[k][i].costrk[1]/theCurrentTemp.entfx[k][i].costrk[2];
    
       in_file >> theCurrentTemp.entfx[k][i].qavg >> theCurrentTemp.entfx[k][i].symax >> theCurrentTemp.entfx[k][i].dyone
	           >> theCurrentTemp.entfx[k][i].syone >> theCurrentTemp.entfx[k][i].sxmax >> theCurrentTemp.entfx[k][i].dxone >> theCurrentTemp.entfx[k][i].sxone;
			
       if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
    
       in_file >> theCurrentTemp.entfx[k][i].dytwo >> theCurrentTemp.entfx[k][i].sytwo >> theCurrentTemp.entfx[k][i].dxtwo 
	           >> theCurrentTemp.entfx[k][i].sxtwo >> theCurrentTemp.entfx[k][i].qmin;
			
       if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
			  
	   for (j=0; j<2; ++j) {
    
          in_file >> theCurrentTemp.entfx[k][i].ypar[j][0] >> theCurrentTemp.entfx[k][i].ypar[j][1] 
	              >> theCurrentTemp.entfx[k][i].ypar[j][2] >> theCurrentTemp.entfx[k][i].ypar[j][3] >> theCurrentTemp.entfx[k][i].ypar[j][4];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
			  
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entfx[k][i].ytemp[j][0] >> theCurrentTemp.entfx[k][i].ytemp[j][1] >> theCurrentTemp.entfx[k][i].ytemp[j][2]
	              >> theCurrentTemp.entfx[k][i].ytemp[j][3] >> theCurrentTemp.entfx[k][i].ytemp[j][4] >> theCurrentTemp.entfx[k][i].ytemp[j][5]
	              >> theCurrentTemp.entfx[k][i].ytemp[j][6] >> theCurrentTemp.entfx[k][i].ytemp[j][7] >> theCurrentTemp.entfx[k][i].ytemp[j][8]
	              >> theCurrentTemp.entfx[k][i].ytemp[j][9] >> theCurrentTemp.entfx[k][i].ytemp[j][10] >> theCurrentTemp.entfx[k][i].ytemp[j][11]
	              >> theCurrentTemp.entfx[k][i].ytemp[j][12] >> theCurrentTemp.entfx[k][i].ytemp[j][13] >> theCurrentTemp.entfx[k][i].ytemp[j][14]
	              >> theCurrentTemp.entfx[k][i].ytemp[j][15] >> theCurrentTemp.entfx[k][i].ytemp[j][16] >> theCurrentTemp.entfx[k][i].ytemp[j][17]
	              >> theCurrentTemp.entfx[k][i].ytemp[j][18] >> theCurrentTemp.entfx[k][i].ytemp[j][19] >> theCurrentTemp.entfx[k][i].ytemp[j][20];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
   			  
	   for (j=0; j<2; ++j) {
    
		  in_file >> theCurrentTemp.entfx[k][i].xpar[j][0] >> theCurrentTemp.entfx[k][i].xpar[j][1] 
	              >> theCurrentTemp.entfx[k][i].xpar[j][2] >> theCurrentTemp.entfx[k][i].xpar[j][3] >> theCurrentTemp.entfx[k][i].xpar[j][4];
			  
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entfx[k][i].xtemp[j][0] >> theCurrentTemp.entfx[k][i].xtemp[j][1] >> theCurrentTemp.entfx[k][i].xtemp[j][2]
	              >> theCurrentTemp.entfx[k][i].xtemp[j][3] >> theCurrentTemp.entfx[k][i].xtemp[j][4] >> theCurrentTemp.entfx[k][i].xtemp[j][5]
	              >> theCurrentTemp.entfx[k][i].xtemp[j][6];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfx[k][i].yavg[j] >> theCurrentTemp.entfx[k][i].yrms[j] >> theCurrentTemp.entfx[k][i].ygx0[j] >> theCurrentTemp.entfx[k][i].ygsig[j];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfx[k][i].yflpar[j][0] >> theCurrentTemp.entfx[k][i].yflpar[j][1] >> theCurrentTemp.entfx[k][i].yflpar[j][2] 
		          >> theCurrentTemp.entfx[k][i].yflpar[j][3] >> theCurrentTemp.entfx[k][i].yflpar[j][4] >> theCurrentTemp.entfx[k][i].yflpar[j][5];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfx[k][i].xavg[j] >> theCurrentTemp.entfx[k][i].xrms[j] >> theCurrentTemp.entfx[k][i].xgx0[j] >> theCurrentTemp.entfx[k][i].xgsig[j];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfx[k][i].xflpar[j][0] >> theCurrentTemp.entfx[k][i].xflpar[j][1] >> theCurrentTemp.entfx[k][i].xflpar[j][2] 
		          >> theCurrentTemp.entfx[k][i].xflpar[j][3] >> theCurrentTemp.entfx[k][i].xflpar[j][4] >> theCurrentTemp.entfx[k][i].xflpar[j][5];
			
          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfx[k][i].chi2yavg[j] >> theCurrentTemp.entfx[k][i].chi2ymin[j] >> theCurrentTemp.entfx[k][i].chi2xavg[j] >> theCurrentTemp.entfx[k][i].chi2xmin[j];

          if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
	   }
	   
	   in_file >> theCurrentTemp.entfx[k][i].yspare[0] >> theCurrentTemp.entfx[k][i].yspare[1] >> theCurrentTemp.entfx[k][i].yspare[2] >> theCurrentTemp.entfx[k][i].yspare[3] >> theCurrentTemp.entfx[k][i].yspare[4]
	    >> theCurrentTemp.entfx[k][i].yspare[5] >> theCurrentTemp.entfx[k][i].yspare[6] >> theCurrentTemp.entfx[k][i].yspare[7] >> theCurrentTemp.entfx[k][i].yspare[8] >> theCurrentTemp.entfx[k][i].yspare[9];

	   if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}

	   in_file >> theCurrentTemp.entfx[k][i].xspare[0] >> theCurrentTemp.entfx[k][i].xspare[1] >> theCurrentTemp.entfx[k][i].xspare[2] >> theCurrentTemp.entfx[k][i].xspare[3] >> theCurrentTemp.entfx[k][i].xspare[4]
	    >> theCurrentTemp.entfx[k][i].xspare[5] >> theCurrentTemp.entfx[k][i].xspare[6] >> theCurrentTemp.entfx[k][i].xspare[7] >> theCurrentTemp.entfx[k][i].xspare[8] >> theCurrentTemp.entfx[k][i].xspare[9];

	   if(in_file.fail()) {LOGERROR("SiPixelTemplate") << "Error reading file, no template load" << ENDL; return false;}
    	   
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











// ************************************************************************************************************ 
//! Interpolate input alpha and beta angles to produce a working template for each individual hit. 
//! \param id - (input) index of the template to use
//! \param fpix - (input) logical input indicating whether to use FPix templates (true) 
//!               or Barrel templates (false)
//! \param cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
// ************************************************************************************************************ 
void SiPixelTemplate::interpolate(int id, bool fpix, float cotalpha, float cotbeta)
{
    // Interpolate for a new set of track angles 
    
    // Local variables 
    int i, j, ind;
	int ilow, ihigh, iylow, iyhigh, Ny, Nxx, Nyx, imidy, imaxx;
	float yratio, yxratio, xxratio, sxmax;
	std::vector <float> xrms(4), xgsig(4);
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
	 
//		

// Interpolate the absolute value of cot(beta)     
    
    abs_cotb = fabs((double)cotbeta);

// Copy the charge scaling factor to the private variable     
    
    pqscale = thePixelTemp[index_id].head.qscale;

// Copy the pseudopixel signal size to the private variable     
    
    ps50 = thePixelTemp[index_id].head.s50;
	
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
		
	   } else if(abs_cotb >= thePixelTemp[index_id].entfy[0].cotbeta) {

          for (i=0; i<Ny-1; ++i) { 
    
             if( thePixelTemp[index_id].entfy[i].cotbeta <= abs_cotb && abs_cotb < thePixelTemp[index_id].entfy[i+1].cotbeta) {
		  
	            ilow = i;
		        yratio = (abs_cotb - thePixelTemp[index_id].entfy[i].cotbeta)/(thePixelTemp[index_id].entfy[i+1].cotbeta - thePixelTemp[index_id].entfy[i].cotbeta);
		        break;			 
		     }
	      }
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all y-related quantities (flip displacements when cotbeta < 0)

       pyratio = yratio;
	   pqavg = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].qavg + yratio*thePixelTemp[index_id].entfy[ihigh].qavg;
	   psymax = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].symax + yratio*thePixelTemp[index_id].entfy[ihigh].symax;
	   sxmax = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].sxmax + yratio*thePixelTemp[index_id].entfy[ihigh].sxmax;
	   pdyone = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].dyone + yratio*thePixelTemp[index_id].entfy[ihigh].dyone;
	   if(cotbeta < 0.) {pdyone = -pdyone;}
	   psyone = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].syone + yratio*thePixelTemp[index_id].entfy[ihigh].syone;
	   pdytwo = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].dytwo + yratio*thePixelTemp[index_id].entfy[ihigh].dytwo;
	   if(cotbeta < 0.) {pdytwo = -pdytwo;}
	   psytwo = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].sytwo + yratio*thePixelTemp[index_id].entfy[ihigh].sytwo;
	   pqmin = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].qmin + yratio*thePixelTemp[index_id].entfy[ihigh].qmin;
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
	      pygx0[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ygx0[i] + yratio*thePixelTemp[index_id].entfy[ihigh].ygx0[i];
	      if(cotbeta < 0.) {pygx0[i] = -pygx0[i];}
	      pygsig[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ygsig[i] + yratio*thePixelTemp[index_id].entfy[ihigh].ygsig[i];
	      xrms[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].xrms[i] + yratio*thePixelTemp[index_id].entfy[ihigh].xrms[i];
	      xgsig[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].xgsig[i] + yratio*thePixelTemp[index_id].entfy[ihigh].xgsig[i];
	      pchi2yavg[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].chi2yavg[i] + yratio*thePixelTemp[index_id].entfy[ihigh].chi2yavg[i];
	      pchi2ymin[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].chi2ymin[i] + yratio*thePixelTemp[index_id].entfy[ihigh].chi2ymin[i];
	      chi2xavg[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].chi2xavg[i] + yratio*thePixelTemp[index_id].entfy[ihigh].chi2xavg[i];
	      chi2xmin[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].chi2xmin[i] + yratio*thePixelTemp[index_id].entfy[ihigh].chi2xmin[i];
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
          pytemp[i+16][0] = 0.;
          pytemp[i+16][1] = 0.;
	      pytemp[i+16][23] = 0.;
	      pytemp[i+16][24] = 0.;
	      for(j=0; j<21; ++j) {
		  
// Flip the basic y-template when the cotbeta is negative

		     if(cotbeta < 0.) {
	            pytemp[24-i][22-j]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ytemp[i][j] + yratio*thePixelTemp[index_id].entfy[ihigh].ytemp[i][j];
			 } else {
	            pytemp[i+16][j+2]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ytemp[i][j] + yratio*thePixelTemp[index_id].entfy[ihigh].ytemp[i][j];
			 }
	      }
	   }
	   for(i=0; i<8; ++i) {
          pytemp[i+8][0] = 0.;
          pytemp[i+8][22] = 0.;
	      pytemp[i+8][23] = 0.;
	      pytemp[i+8][24] = 0.;
	      for(j=0; j<21; ++j) {
	        pytemp[i+8][j+1]=pytemp[i+16][j+2];
	      }
	   }
	   for(i=0; i<8; ++i) {
          pytemp[i][21] = 0.;
          pytemp[i][22] = 0.;
	      pytemp[i][23] = 0.;
	      pytemp[i][24] = 0.;
	      for(j=0; j<21; ++j) {
	        pytemp[i][j]=pytemp[i+16][j+2];
	      }
	   }
  	   for(i=1; i<9; ++i) {
          pytemp[i+24][0] = 0.;
	      pytemp[i+24][1] = 0.;
	      pytemp[i+24][2] = 0.;
	      pytemp[i+24][24] = 0.;
	      for(j=0; j<21; ++j) {
	         pytemp[i+24][j+3]=pytemp[i+16][j+2];
	      }
	   }
  	   for(i=1; i<9; ++i) {
          pytemp[i+32][0] = 0.;
	      pytemp[i+32][1] = 0.;
	      pytemp[i+32][2] = 0.;
	      pytemp[i+32][3] = 0.;
	      for(j=0; j<21; ++j) {
	         pytemp[i+32][j+4]=pytemp[i+16][j+2];
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
		
	   } else if(cotalpha >= thePixelTemp[index_id].entfx[0][0].cotalpha) {

          for (i=0; i<Nxx-1; ++i) { 
    
             if( thePixelTemp[index_id].entfx[0][i].cotalpha <= cotalpha && cotalpha < thePixelTemp[index_id].entfx[0][i+1].cotalpha) {
		  
	            ilow = i;
		        xxratio = (cotalpha - thePixelTemp[index_id].entfx[0][i].cotalpha)/(thePixelTemp[index_id].entfx[0][i+1].cotalpha - thePixelTemp[index_id].entfx[0][i].cotalpha);
		        break;
			 
		     }
	      }
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all x-related quantities 

       pyxratio = yxratio;
       pxxratio = xxratio;		
	   		  
// sxparmax defines the maximum charge for which the parameters xpar are defined (not rescaled by cotbeta) 

	   psxparmax = (1. - xxratio)*thePixelTemp[index_id].entfx[imaxx][ilow].sxmax + xxratio*thePixelTemp[index_id].entfx[imaxx][ihigh].sxmax;
	   psxmax = psxparmax;
       if(thePixelTemp[index_id].entfx[imaxx][imidy].sxmax != 0.) {psxmax=psxmax/thePixelTemp[index_id].entfx[imaxx][imidy].sxmax*sxmax;}
	   pdxone = (1. - xxratio)*thePixelTemp[index_id].entfx[0][ilow].dxone + xxratio*thePixelTemp[index_id].entfx[0][ihigh].dxone;
	   psxone = (1. - xxratio)*thePixelTemp[index_id].entfx[0][ilow].sxone + xxratio*thePixelTemp[index_id].entfx[0][ihigh].sxone;
	   pdxtwo = (1. - xxratio)*thePixelTemp[index_id].entfx[0][ilow].dxtwo + xxratio*thePixelTemp[index_id].entfx[0][ihigh].dxtwo;
	   psxtwo = (1. - xxratio)*thePixelTemp[index_id].entfx[0][ilow].sxtwo + xxratio*thePixelTemp[index_id].entfx[0][ihigh].sxtwo;
	   for(i=0; i<2 ; ++i) {
	      for(j=0; j<5 ; ++j) {
	         pxpar0[i][j] = thePixelTemp[index_id].entfx[imaxx][imidy].xpar[i][j];
	         pxparl[i][j] = thePixelTemp[index_id].entfx[imaxx][ilow].xpar[i][j];
	         pxparh[i][j] = thePixelTemp[index_id].entfx[imaxx][ihigh].xpar[i][j];
	      }
	   }
	   for(i=0; i<4; ++i) {
	      pxavg[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xavg[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xavg[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xavg[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xavg[i]);
		  
	      pxrms[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xrms[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xrms[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xrms[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xrms[i]);
		  
	      pxgx0[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xgx0[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xgx0[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xgx0[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xgx0[i]);
							
	      pxgsig[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entfx[iylow][ilow].xgsig[i] + xxratio*thePixelTemp[index_id].entfx[iylow][ihigh].xgsig[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entfx[iyhigh][ilow].xgsig[i] + xxratio*thePixelTemp[index_id].entfx[iyhigh][ihigh].xgsig[i]);
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
          pxtemp[i+16][0] = 0.;
          pxtemp[i+16][1] = 0.;
	      pxtemp[i+16][9] = 0.;
	      pxtemp[i+16][10] = 0.;
	      for(j=0; j<7; ++j) {
	        pxtemp[i+16][j+2]=(1. - xxratio)*thePixelTemp[index_id].entfx[imaxx][ilow].xtemp[i][j] + xxratio*thePixelTemp[index_id].entfx[imaxx][ihigh].xtemp[i][j];
	      }
	   }
	   for(i=0; i<8; ++i) {
          pxtemp[i+8][0] = 0.;
	      pxtemp[i+8][8] = 0.;
          pxtemp[i+8][9] = 0.;
	      pxtemp[i+8][10] = 0.;
	      for(j=0; j<7; ++j) {
	        pxtemp[i+8][j+1]=pxtemp[i+16][j+2];
	      }
	   }
	   for(i=0; i<8; ++i) {
          pxtemp[i][7] = 0.;
	      pxtemp[i][8] = 0.;
          pxtemp[i][9] = 0.;
	      pxtemp[i][10] = 0.;
	      for(j=0; j<7; ++j) {
	        pxtemp[i][j]=pxtemp[i+16][j+2];
	      }
	   }
	   for(i=1; i<9; ++i) {
          pxtemp[i+24][0] = 0.;
	      pxtemp[i+24][1] = 0.;
          pxtemp[i+24][2] = 0.;
	      pxtemp[i+24][10] = 0.;
	      for(j=0; j<7; ++j) {
	        pxtemp[i+24][j+3]=pxtemp[i+16][j+2];
	      }
	   }
	   for(i=1; i<9; ++i) {
          pxtemp[i+32][0] = 0.;
	      pxtemp[i+32][1] = 0.;
          pxtemp[i+32][2] = 0.;
	      pxtemp[i+32][3] = 0.;
	      for(j=0; j<7; ++j) {
	        pxtemp[i+32][j+4]=pxtemp[i+16][j+2];
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
		
	   } else if(abs_cotb >= thePixelTemp[index_id].entby[0].cotbeta) {

          for (i=0; i<Ny-1; ++i) { 
    
             if( thePixelTemp[index_id].entby[i].cotbeta <= abs_cotb && abs_cotb < thePixelTemp[index_id].entby[i+1].cotbeta) {
		  
	            ilow = i;
		        yratio = (abs_cotb - thePixelTemp[index_id].entby[i].cotbeta)/(thePixelTemp[index_id].entby[i+1].cotbeta - thePixelTemp[index_id].entby[i].cotbeta);
		        break;			 
		     }
	      }
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all y-related quantities (flip displacements when cotbeta < 0)

       pyratio = yratio;
	   pqavg = (1. - yratio)*thePixelTemp[index_id].entby[ilow].qavg + yratio*thePixelTemp[index_id].entby[ihigh].qavg;
	   psymax = (1. - yratio)*thePixelTemp[index_id].entby[ilow].symax + yratio*thePixelTemp[index_id].entby[ihigh].symax;
	   sxmax = (1. - yratio)*thePixelTemp[index_id].entby[ilow].sxmax + yratio*thePixelTemp[index_id].entby[ihigh].sxmax;
	   pdyone = (1. - yratio)*thePixelTemp[index_id].entby[ilow].dyone + yratio*thePixelTemp[index_id].entby[ihigh].dyone;
	   if(cotbeta < 0.) {pdyone = -pdyone;}
	   psyone = (1. - yratio)*thePixelTemp[index_id].entby[ilow].syone + yratio*thePixelTemp[index_id].entby[ihigh].syone;
	   pdytwo = (1. - yratio)*thePixelTemp[index_id].entby[ilow].dytwo + yratio*thePixelTemp[index_id].entby[ihigh].dytwo;
	   if(cotbeta < 0.) {pdytwo = -pdytwo;}
	   psytwo = (1. - yratio)*thePixelTemp[index_id].entby[ilow].sytwo + yratio*thePixelTemp[index_id].entby[ihigh].sytwo;
	   pqmin = (1. - yratio)*thePixelTemp[index_id].entby[ilow].qmin + yratio*thePixelTemp[index_id].entby[ihigh].qmin;
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
	      pygx0[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ygx0[i] + yratio*thePixelTemp[index_id].entby[ihigh].ygx0[i];
	      if(cotbeta < 0.) {pygx0[i] = -pygx0[i];}
	      pygsig[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ygsig[i] + yratio*thePixelTemp[index_id].entby[ihigh].ygsig[i];
	      xrms[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].xrms[i] + yratio*thePixelTemp[index_id].entby[ihigh].xrms[i];
	      xgsig[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].xgsig[i] + yratio*thePixelTemp[index_id].entby[ihigh].xgsig[i];
	      pchi2yavg[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].chi2yavg[i] + yratio*thePixelTemp[index_id].entby[ihigh].chi2yavg[i];
	      pchi2ymin[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].chi2ymin[i] + yratio*thePixelTemp[index_id].entby[ihigh].chi2ymin[i];
	      chi2xavg[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].chi2xavg[i] + yratio*thePixelTemp[index_id].entby[ihigh].chi2xavg[i];
	      chi2xmin[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].chi2xmin[i] + yratio*thePixelTemp[index_id].entby[ihigh].chi2xmin[i];
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
          pytemp[i+16][0] = 0.;
          pytemp[i+16][1] = 0.;
	      pytemp[i+16][23] = 0.;
	      pytemp[i+16][24] = 0.;
	      for(j=0; j<21; ++j) {
		  
// Flip the basic y-template when the cotbeta is negative

		     if(cotbeta < 0.) {
	            pytemp[24-i][22-j]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ytemp[i][j] + yratio*thePixelTemp[index_id].entby[ihigh].ytemp[i][j];
			 } else {
	            pytemp[i+16][j+2]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ytemp[i][j] + yratio*thePixelTemp[index_id].entby[ihigh].ytemp[i][j];
			 }
	      }
	   }
	   for(i=0; i<8; ++i) {
          pytemp[i+8][0] = 0.;
          pytemp[i+8][22] = 0.;
	      pytemp[i+8][23] = 0.;
	      pytemp[i+8][24] = 0.;
	      for(j=0; j<21; ++j) {
	        pytemp[i+8][j+1]=pytemp[i+16][j+2];
	      }
	   }
	   for(i=0; i<8; ++i) {
          pytemp[i][21] = 0.;
          pytemp[i][22] = 0.;
	      pytemp[i][23] = 0.;
	      pytemp[i][24] = 0.;
	      for(j=0; j<21; ++j) {
	        pytemp[i][j]=pytemp[i+16][j+2];
	      }
	   }
  	   for(i=1; i<9; ++i) {
          pytemp[i+24][0] = 0.;
	      pytemp[i+24][1] = 0.;
	      pytemp[i+24][2] = 0.;
	      pytemp[i+24][24] = 0.;
	      for(j=0; j<21; ++j) {
	         pytemp[i+24][j+3]=pytemp[i+16][j+2];
	      }
	   }
  	   for(i=1; i<9; ++i) {
          pytemp[i+32][0] = 0.;
	      pytemp[i+32][1] = 0.;
	      pytemp[i+32][2] = 0.;
	      pytemp[i+32][3] = 0.;
	      for(j=0; j<21; ++j) {
	         pytemp[i+32][j+4]=pytemp[i+16][j+2];
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
		
	   } else if(cotalpha >= thePixelTemp[index_id].entbx[0][0].cotalpha) {

          for (i=0; i<Nxx-1; ++i) { 
    
             if( thePixelTemp[index_id].entbx[0][i].cotalpha <= cotalpha && cotalpha < thePixelTemp[index_id].entbx[0][i+1].cotalpha) {
		  
	            ilow = i;
		        xxratio = (cotalpha - thePixelTemp[index_id].entbx[0][i].cotalpha)/(thePixelTemp[index_id].entbx[0][i+1].cotalpha - thePixelTemp[index_id].entbx[0][i].cotalpha);
		        break;
			 
		     }
	      }
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all x-related quantities 

       pyxratio = yxratio;
       pxxratio = xxratio;		
	   		  
// sxparmax defines the maximum charge for which the parameters xpar are defined (not rescaled by cotbeta) 

	   psxparmax = (1. - xxratio)*thePixelTemp[index_id].entbx[imaxx][ilow].sxmax + xxratio*thePixelTemp[index_id].entbx[imaxx][ihigh].sxmax;
	   psxmax = psxparmax;
       if(thePixelTemp[index_id].entbx[imaxx][imidy].sxmax != 0.) {psxmax=psxmax/thePixelTemp[index_id].entbx[imaxx][imidy].sxmax*sxmax;}
	   pdxone = (1. - xxratio)*thePixelTemp[index_id].entbx[0][ilow].dxone + xxratio*thePixelTemp[index_id].entbx[0][ihigh].dxone;
	   psxone = (1. - xxratio)*thePixelTemp[index_id].entbx[0][ilow].sxone + xxratio*thePixelTemp[index_id].entbx[0][ihigh].sxone;
	   pdxtwo = (1. - xxratio)*thePixelTemp[index_id].entbx[0][ilow].dxtwo + xxratio*thePixelTemp[index_id].entbx[0][ihigh].dxtwo;
	   psxtwo = (1. - xxratio)*thePixelTemp[index_id].entbx[0][ilow].sxtwo + xxratio*thePixelTemp[index_id].entbx[0][ihigh].sxtwo;
	   for(i=0; i<2 ; ++i) {
	      for(j=0; j<5 ; ++j) {
	         pxpar0[i][j] = thePixelTemp[index_id].entbx[imaxx][imidy].xpar[i][j];
	         pxparl[i][j] = thePixelTemp[index_id].entbx[imaxx][ilow].xpar[i][j];
	         pxparh[i][j] = thePixelTemp[index_id].entbx[imaxx][ihigh].xpar[i][j];
	      }
	   }
	   for(i=0; i<4; ++i) {
	      pxavg[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xavg[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xavg[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xavg[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xavg[i]);
		  
	      pxrms[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xrms[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xrms[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xrms[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xrms[i]);
		  
	      pxgx0[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xgx0[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xgx0[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xgx0[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xgx0[i]);
							
	      pxgsig[i]=(1. - yxratio)*((1. - xxratio)*thePixelTemp[index_id].entbx[iylow][ilow].xgsig[i] + xxratio*thePixelTemp[index_id].entbx[iylow][ihigh].xgsig[i])
		          +yxratio*((1. - xxratio)*thePixelTemp[index_id].entbx[iyhigh][ilow].xgsig[i] + xxratio*thePixelTemp[index_id].entbx[iyhigh][ihigh].xgsig[i]);
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
          pxtemp[i+16][0] = 0.;
          pxtemp[i+16][1] = 0.;
	      pxtemp[i+16][9] = 0.;
	      pxtemp[i+16][10] = 0.;
	      for(j=0; j<7; ++j) {
	        pxtemp[i+16][j+2]=(1. - xxratio)*thePixelTemp[index_id].entbx[imaxx][ilow].xtemp[i][j] + xxratio*thePixelTemp[index_id].entbx[imaxx][ihigh].xtemp[i][j];
	      }
	   }
	   for(i=0; i<8; ++i) {
          pxtemp[i+8][0] = 0.;
	      pxtemp[i+8][8] = 0.;
          pxtemp[i+8][9] = 0.;
	      pxtemp[i+8][10] = 0.;
	      for(j=0; j<7; ++j) {
	        pxtemp[i+8][j+1]=pxtemp[i+16][j+2];
	      }
	   }
	   for(i=0; i<8; ++i) {
          pxtemp[i][7] = 0.;
	      pxtemp[i][8] = 0.;
          pxtemp[i][9] = 0.;
	      pxtemp[i][10] = 0.;
	      for(j=0; j<7; ++j) {
	        pxtemp[i][j]=pxtemp[i+16][j+2];
	      }
	   }
	   for(i=1; i<9; ++i) {
          pxtemp[i+24][0] = 0.;
	      pxtemp[i+24][1] = 0.;
          pxtemp[i+24][2] = 0.;
	      pxtemp[i+24][10] = 0.;
	      for(j=0; j<7; ++j) {
	        pxtemp[i+24][j+3]=pxtemp[i+16][j+2];
	      }
	   }
	   for(i=1; i<9; ++i) {
          pxtemp[i+32][0] = 0.;
	      pxtemp[i+32][1] = 0.;
          pxtemp[i+32][2] = 0.;
	      pxtemp[i+32][3] = 0.;
	      for(j=0; j<7; ++j) {
	        pxtemp[i+32][j+4]=pxtemp[i+16][j+2];
	      }
	   }
	}
  }
  return;
} // interpolate





// ************************************************************************************************************ 
//! Return vector of y errors (squared) for an input vector of projected signals 
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
	float sigi, sigi2, sigi3, sigi4;
	
    // Make sure that input is OK
    
	assert(fypix > 1 && fypix < 23);
	assert(lypix >= fypix && lypix < 23);
	   	     
// Define the maximum signal to allow before de-weighting a pixel 

//       sythr = 1.1*psymax;
	   
// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 

	   for(i=fypix-2; i<=lypix+2; ++i) {
		  if(i < fypix || i > lypix) {
	   
// Nearest pseudopixels have uncertainties of 50% of threshold, next-nearest have 10% of threshold

			 ysig2[i] = ps50*ps50;
		  } else {
			 if(ysum[i] < psymax) {
				sigi = ysum[i];
			 } else {
				sigi = psymax;
			 }
			 sigi2 = sigi*sigi; sigi3 = sigi2*sigi; sigi4 = sigi3*sigi;
			 if(i <= 12) {
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
		     if(ysum[i] > sythr) {ysig2[i] = 1.e8;}
			 if(ysig2[i] <= 0.) {LOGERROR("SiPixelTemplate") << "neg y-error-squared, id = " << id_current << ", index = " << index_id << 
			 ", cot(alpha) = " << cota_current << ", cot(beta) = " << cotb_current << ", fpix = " << fpix_current << " sigi = " << sigi << ENDL;}
	      }
	   }
	
	return;
	
} // End ysigma2







// ************************************************************************************************************ 
//! Return vector of x errors (squared) for an input vector of projected signals 
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
	float sigi, sigi2, sigi3, sigi4, yint, sxmax, x0;
	
    // Make sure that input is OK
    
	assert(fxpix > 1 && fxpix < 9);
	assert(lxpix >= fxpix && lxpix < 9);
	   	     
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
			 } else {
				sigi = sxmax;
			 }
			 sigi2 = sigi*sigi; sigi3 = sigi2*sigi; sigi4 = sigi3*sigi;
			 
// First, do the cotbeta interpolation			 
			 
			 if(i <= 5) {
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
			 
			 if(i <= 5) {
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
			 
			 if(i <= 5) {
				x0 = pxpar0[0][0]+pxpar0[0][1]*sigi+pxpar0[0][2]*sigi2+pxpar0[0][3]*sigi3+pxpar0[0][4]*sigi4;
			 } else {
				x0 = pxpar0[1][0]+pxpar0[1][1]*sigi+pxpar0[1][2]*sigi2+pxpar0[1][3]*sigi3+pxpar0[1][4]*sigi4;
			 }
			 
// Finally, rescale the yint value for cotalpha variation			 
			 
			 if(x0 != 0.) {xsig2[i] = xsig2[i]/x0 * yint;}
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
  void SiPixelTemplate::ytemp(int fybin, int lybin, float ytemplate[41][25])
  
{
    // Retrieve already interpolated quantities
    
    // Local variables 
    int i, j;

   // Verify that input parameters are in valid range

	assert(fybin >= 0 && fybin < 41);
	assert(lybin >= 0 && lybin < 41);

	for(i=fybin; i<=lybin; ++i) {
	   for(j=0; j<25; ++j) {
	      ytemplate[i][j] = pytemp[i][j];
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
  void SiPixelTemplate::xtemp(int fxbin, int lxbin, float xtemplate[41][11])
  
{
    // Retrieve already interpolated quantities
    
    // Local variables 
    int i, j;

   // Verify that input parameters are in valid range

	assert(fxbin >= 0 && fxbin < 41);
	assert(lxbin >= 0 && lxbin < 41);

	for(i=fxbin; i<=lxbin; ++i) {
	   for(j=0; j<11; ++j) {
	      xtemplate[i][j] = pxtemp[i][j];
	   }
	}
	
 	return;
	
} // End xtemp

