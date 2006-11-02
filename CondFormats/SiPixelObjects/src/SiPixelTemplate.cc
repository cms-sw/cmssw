/*
 *  SiPixelTemplate.cc
 *  
 *
 *  Created by Morris Swartz on 10/27/06.
 *  Copyright 2006 __TheJohnsHopkinsUniversity__. All rights reserved.
 *
 */

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplate.h"
#else
#include "SiPixelTemplate.h"
#endif

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


//**************************************************************** 
//! This routine initializes the global template structures from 
//! an external file template_summary_zpNNNN where NNNN are four  
//! digits of filenum.                                           
//! \param filenum - an integer NNNN used in the filename template_summary_zpNNNN
//**************************************************************** 
void SiPixelTemplate::pushfile(int filenum)
{
    // Add template stored in external file numbered filenum to theTemplateStore
    
    // Local variables 
    int i, j;
	const char *tempfile;
	char title[80];
    char c;
	


//  Create a filename for this run 

 std::ostringstream tout;
 tout << "template_summary_zp" << std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
 std::string tempf = tout.str();
 tempfile = tempf.c_str();
	
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
    std::cout << theCurrentTemp.head.title << std::endl;
    
// next, the header information     
    
    in_file >> theCurrentTemp.head.ID >> theCurrentTemp.head.NBy >> theCurrentTemp.head.NBx
	        >> theCurrentTemp.head.NFy >> theCurrentTemp.head.NFx >> theCurrentTemp.head.vbias >> theCurrentTemp.head.temperature 
		    >> theCurrentTemp.head.fluence >> theCurrentTemp.head.s50;
    
    std::cout << "Template ID = " << theCurrentTemp.head.ID << ", NBy = " << theCurrentTemp.head.NBy << ", NBx = " << theCurrentTemp.head.NBx << ", NFy = "
	     << theCurrentTemp.head.NFy << ", NFx = " << theCurrentTemp.head.NFx << ", bias voltage " << theCurrentTemp.head.vbias << ", temperature "
		 << theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence << ", 1/2 threshold " << theCurrentTemp.head.s50 << std::endl; 
    
// next, loop over all barrel y-angle entries   

    for (i=0; i < theCurrentTemp.head.NBy; ++i) {     
    
       in_file >> theCurrentTemp.entby[i].runnum >> theCurrentTemp.entby[i].costrk[0] 
	           >> theCurrentTemp.entby[i].costrk[1] >> theCurrentTemp.entby[i].costrk[2]; 
			  
// Calculate the alpha, beta, and cot(beta) for this entry 

       theCurrentTemp.entby[i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entby[i].costrk[2], (double)theCurrentTemp.entby[i].costrk[0]));
	   
	   theCurrentTemp.entby[i].cotalpha = theCurrentTemp.entby[i].costrk[0]/theCurrentTemp.entby[i].costrk[2];

       theCurrentTemp.entby[i].beta = static_cast<float>(atan2((double)theCurrentTemp.entby[i].costrk[2], (double)theCurrentTemp.entby[i].costrk[1]));
	   
	   theCurrentTemp.entby[i].cotbeta = theCurrentTemp.entby[i].costrk[1]/theCurrentTemp.entby[i].costrk[2];
    
       in_file >> theCurrentTemp.entby[i].qavg >> theCurrentTemp.entby[i].symax >> theCurrentTemp.entby[i].dyone
	           >> theCurrentTemp.entby[i].syone >> theCurrentTemp.entby[i].sxmax >> theCurrentTemp.entby[i].dxone >> theCurrentTemp.entby[i].sxone;
    
       in_file >> theCurrentTemp.entby[i].dytwo >> theCurrentTemp.entby[i].sytwo >> theCurrentTemp.entby[i].dxtwo 
	           >> theCurrentTemp.entby[i].sxtwo;
			  
	   for (j=0; j<2; ++j) {
    
          in_file >> theCurrentTemp.entby[i].ypar[j][0] >> theCurrentTemp.entby[i].ypar[j][1] 
	              >> theCurrentTemp.entby[i].ypar[j][2] >> theCurrentTemp.entby[i].ypar[j][3] >> theCurrentTemp.entby[i].ypar[j][4];
			  
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entby[i].ytemp[j][0] >> theCurrentTemp.entby[i].ytemp[j][1] >> theCurrentTemp.entby[i].ytemp[j][2]
	              >> theCurrentTemp.entby[i].ytemp[j][3] >> theCurrentTemp.entby[i].ytemp[j][4] >> theCurrentTemp.entby[i].ytemp[j][5]
	              >> theCurrentTemp.entby[i].ytemp[j][6] >> theCurrentTemp.entby[i].ytemp[j][7] >> theCurrentTemp.entby[i].ytemp[j][8]
	              >> theCurrentTemp.entby[i].ytemp[j][9] >> theCurrentTemp.entby[i].ytemp[j][10] >> theCurrentTemp.entby[i].ytemp[j][11]
	              >> theCurrentTemp.entby[i].ytemp[j][12] >> theCurrentTemp.entby[i].ytemp[j][13] >> theCurrentTemp.entby[i].ytemp[j][14]
	              >> theCurrentTemp.entby[i].ytemp[j][15] >> theCurrentTemp.entby[i].ytemp[j][16] >> theCurrentTemp.entby[i].ytemp[j][17]
	              >> theCurrentTemp.entby[i].ytemp[j][18] >> theCurrentTemp.entby[i].ytemp[j][19] >> theCurrentTemp.entby[i].ytemp[j][20];
	   }
   			  
	   for (j=0; j<2; ++j) {
    
		  in_file >> theCurrentTemp.entby[i].xpar[j][0] >> theCurrentTemp.entby[i].xpar[j][1] 
	              >> theCurrentTemp.entby[i].xpar[j][2] >> theCurrentTemp.entby[i].xpar[j][3] >> theCurrentTemp.entby[i].xpar[j][4];
			  
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entby[i].xtemp[j][0] >> theCurrentTemp.entby[i].xtemp[j][1] >> theCurrentTemp.entby[i].xtemp[j][2]
	              >> theCurrentTemp.entby[i].xtemp[j][3] >> theCurrentTemp.entby[i].xtemp[j][4] >> theCurrentTemp.entby[i].xtemp[j][5]
	              >> theCurrentTemp.entby[i].xtemp[j][6];
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entby[i].yavg[j] >> theCurrentTemp.entby[i].yrms[j] >> theCurrentTemp.entby[i].ygx0[j] >> theCurrentTemp.entby[i].ygsig[j];
	   }
	   			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entby[i].yeavg[j] >> theCurrentTemp.entby[i].yerms[j] >> theCurrentTemp.entby[i].yegx0[j] >> theCurrentTemp.entby[i].yegsig[j];
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entby[i].yoavg[j] >> theCurrentTemp.entby[i].yorms[j] >> theCurrentTemp.entby[i].yogx0[j] >> theCurrentTemp.entby[i].yogsig[j];
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entby[i].xavg[j] >> theCurrentTemp.entby[i].xrms[j] >> theCurrentTemp.entby[i].xgx0[j] >> theCurrentTemp.entby[i].xgsig[j];
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entby[i].xeavg[j] >> theCurrentTemp.entby[i].xerms[j] >> theCurrentTemp.entby[i].xegx0[j] >> theCurrentTemp.entby[i].xegsig[j];
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entby[i].xoavg[j] >> theCurrentTemp.entby[i].xorms[j] >> theCurrentTemp.entby[i].xogx0[j] >> theCurrentTemp.entby[i].xogsig[j];
	   }
    	   
	}
	
// next, loop over all barrel x-angle entries   

    for (i=0; i < theCurrentTemp.head.NBx; ++i) { 
        
       in_file >> theCurrentTemp.entbx[i].runnum >> theCurrentTemp.entbx[i].costrk[0] 
	           >> theCurrentTemp.entbx[i].costrk[1] >> theCurrentTemp.entbx[i].costrk[2]; 
			  
// Calculate the alpha, beta, and cot(beta) for this entry 

       theCurrentTemp.entbx[i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entbx[i].costrk[2], (double)theCurrentTemp.entbx[i].costrk[0]));
	   
	   theCurrentTemp.entbx[i].cotalpha = theCurrentTemp.entbx[i].costrk[0]/theCurrentTemp.entbx[i].costrk[2];

       theCurrentTemp.entbx[i].beta = static_cast<float>(atan2((double)theCurrentTemp.entbx[i].costrk[2], (double)theCurrentTemp.entbx[i].costrk[1]));
	   
	   theCurrentTemp.entbx[i].cotbeta = theCurrentTemp.entbx[i].costrk[1]/theCurrentTemp.entbx[i].costrk[2];
    
       in_file >> theCurrentTemp.entbx[i].qavg >> theCurrentTemp.entbx[i].symax >> theCurrentTemp.entbx[i].dyone
	           >> theCurrentTemp.entbx[i].syone >> theCurrentTemp.entbx[i].sxmax >> theCurrentTemp.entbx[i].dxone >> theCurrentTemp.entbx[i].sxone;
    
       in_file >> theCurrentTemp.entbx[i].dytwo >> theCurrentTemp.entbx[i].sytwo >> theCurrentTemp.entbx[i].dxtwo 
	           >> theCurrentTemp.entbx[i].sxtwo;
			  
	   for (j=0; j<2; ++j) {
    
          in_file >> theCurrentTemp.entbx[i].ypar[j][0] >> theCurrentTemp.entbx[i].ypar[j][1] 
	              >> theCurrentTemp.entbx[i].ypar[j][2] >> theCurrentTemp.entbx[i].ypar[j][3] >> theCurrentTemp.entbx[i].ypar[j][4];
			  
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entbx[i].ytemp[j][0] >> theCurrentTemp.entbx[i].ytemp[j][1] >> theCurrentTemp.entbx[i].ytemp[j][2]
	              >> theCurrentTemp.entbx[i].ytemp[j][3] >> theCurrentTemp.entbx[i].ytemp[j][4] >> theCurrentTemp.entbx[i].ytemp[j][5]
	              >> theCurrentTemp.entbx[i].ytemp[j][6] >> theCurrentTemp.entbx[i].ytemp[j][7] >> theCurrentTemp.entbx[i].ytemp[j][8]
	              >> theCurrentTemp.entbx[i].ytemp[j][9] >> theCurrentTemp.entbx[i].ytemp[j][10] >> theCurrentTemp.entbx[i].ytemp[j][11]
	              >> theCurrentTemp.entbx[i].ytemp[j][12] >> theCurrentTemp.entbx[i].ytemp[j][13] >> theCurrentTemp.entbx[i].ytemp[j][14]
	              >> theCurrentTemp.entbx[i].ytemp[j][15] >> theCurrentTemp.entbx[i].ytemp[j][16] >> theCurrentTemp.entbx[i].ytemp[j][17]
	              >> theCurrentTemp.entbx[i].ytemp[j][18] >> theCurrentTemp.entbx[i].ytemp[j][19] >> theCurrentTemp.entbx[i].ytemp[j][20];
	   }
   			  
	   for (j=0; j<2; ++j) {
    
		  in_file >> theCurrentTemp.entbx[i].xpar[j][0] >> theCurrentTemp.entbx[i].xpar[j][1] 
	              >> theCurrentTemp.entbx[i].xpar[j][2] >> theCurrentTemp.entbx[i].xpar[j][3] >> theCurrentTemp.entbx[i].xpar[j][4];
			  
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entbx[i].xtemp[j][0] >> theCurrentTemp.entbx[i].xtemp[j][1] >> theCurrentTemp.entbx[i].xtemp[j][2]
	              >> theCurrentTemp.entbx[i].xtemp[j][3] >> theCurrentTemp.entbx[i].xtemp[j][4] >> theCurrentTemp.entbx[i].xtemp[j][5]
	              >> theCurrentTemp.entbx[i].xtemp[j][6];
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entbx[i].yavg[j] >> theCurrentTemp.entbx[i].yrms[j] >> theCurrentTemp.entbx[i].ygx0[j] >> theCurrentTemp.entbx[i].ygsig[j];
	   }
	   			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entbx[i].yeavg[j] >> theCurrentTemp.entbx[i].yerms[j] >> theCurrentTemp.entbx[i].yegx0[j] >> theCurrentTemp.entbx[i].yegsig[j];
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entbx[i].yoavg[j] >> theCurrentTemp.entbx[i].yorms[j] >> theCurrentTemp.entbx[i].yogx0[j] >> theCurrentTemp.entbx[i].yogsig[j];
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entbx[i].xavg[j] >> theCurrentTemp.entbx[i].xrms[j] >> theCurrentTemp.entbx[i].xgx0[j] >> theCurrentTemp.entbx[i].xgsig[j];
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entbx[i].xeavg[j] >> theCurrentTemp.entbx[i].xerms[j] >> theCurrentTemp.entbx[i].xegx0[j] >> theCurrentTemp.entbx[i].xegsig[j];
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entbx[i].xoavg[j] >> theCurrentTemp.entbx[i].xorms[j] >> theCurrentTemp.entbx[i].xogx0[j] >> theCurrentTemp.entbx[i].xogsig[j];
	   }
    	   
	}
    
// next, loop over all forward y-angle entries   

    for (i=0; i < theCurrentTemp.head.NFy; ++i) {     
    
       in_file >> theCurrentTemp.entfy[i].runnum >> theCurrentTemp.entfy[i].costrk[0] 
	           >> theCurrentTemp.entfy[i].costrk[1] >> theCurrentTemp.entfy[i].costrk[2]; 
			  
// Calculate the alpha, beta, and cot(beta) for this entry 

       theCurrentTemp.entfy[i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entfy[i].costrk[2], (double)theCurrentTemp.entfy[i].costrk[0]));
	   
	   theCurrentTemp.entfy[i].cotalpha = theCurrentTemp.entfy[i].costrk[0]/theCurrentTemp.entfy[i].costrk[2];

       theCurrentTemp.entfy[i].beta = static_cast<float>(atan2((double)theCurrentTemp.entfy[i].costrk[2], (double)theCurrentTemp.entfy[i].costrk[1]));
	   
	   theCurrentTemp.entfy[i].cotbeta = theCurrentTemp.entfy[i].costrk[1]/theCurrentTemp.entfy[i].costrk[2];
    
       in_file >> theCurrentTemp.entfy[i].qavg >> theCurrentTemp.entfy[i].symax >> theCurrentTemp.entfy[i].dyone
	           >> theCurrentTemp.entfy[i].syone >> theCurrentTemp.entfy[i].sxmax >> theCurrentTemp.entfy[i].dxone >> theCurrentTemp.entfy[i].sxone;
    
       in_file >> theCurrentTemp.entfy[i].dytwo >> theCurrentTemp.entfy[i].sytwo >> theCurrentTemp.entfy[i].dxtwo 
	           >> theCurrentTemp.entfy[i].sxtwo;
			  
	   for (j=0; j<2; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].ypar[j][0] >> theCurrentTemp.entfy[i].ypar[j][1] 
	              >> theCurrentTemp.entfy[i].ypar[j][2] >> theCurrentTemp.entfy[i].ypar[j][3] >> theCurrentTemp.entfy[i].ypar[j][4];
			  
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].ytemp[j][0] >> theCurrentTemp.entfy[i].ytemp[j][1] >> theCurrentTemp.entfy[i].ytemp[j][2]
	              >> theCurrentTemp.entfy[i].ytemp[j][3] >> theCurrentTemp.entfy[i].ytemp[j][4] >> theCurrentTemp.entfy[i].ytemp[j][5]
	              >> theCurrentTemp.entfy[i].ytemp[j][6] >> theCurrentTemp.entfy[i].ytemp[j][7] >> theCurrentTemp.entfy[i].ytemp[j][8]
	              >> theCurrentTemp.entfy[i].ytemp[j][9] >> theCurrentTemp.entfy[i].ytemp[j][10] >> theCurrentTemp.entfy[i].ytemp[j][11]
	              >> theCurrentTemp.entfy[i].ytemp[j][12] >> theCurrentTemp.entfy[i].ytemp[j][13] >> theCurrentTemp.entfy[i].ytemp[j][14]
	              >> theCurrentTemp.entfy[i].ytemp[j][15] >> theCurrentTemp.entfy[i].ytemp[j][16] >> theCurrentTemp.entfy[i].ytemp[j][17]
	              >> theCurrentTemp.entfy[i].ytemp[j][18] >> theCurrentTemp.entfy[i].ytemp[j][19] >> theCurrentTemp.entfy[i].ytemp[j][20];
	   }
   			  
	   for (j=0; j<2; ++j) {
    
		  in_file >> theCurrentTemp.entfy[i].xpar[j][0] >> theCurrentTemp.entfy[i].xpar[j][1] 
	              >> theCurrentTemp.entfy[i].xpar[j][2] >> theCurrentTemp.entfy[i].xpar[j][3] >> theCurrentTemp.entfy[i].xpar[j][4];
			  
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].xtemp[j][0] >> theCurrentTemp.entfy[i].xtemp[j][1] >> theCurrentTemp.entfy[i].xtemp[j][2]
	              >> theCurrentTemp.entfy[i].xtemp[j][3] >> theCurrentTemp.entfy[i].xtemp[j][4] >> theCurrentTemp.entfy[i].xtemp[j][5]
	              >> theCurrentTemp.entfy[i].xtemp[j][6];
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].yavg[j] >> theCurrentTemp.entfy[i].yrms[j] >> theCurrentTemp.entfy[i].ygx0[j] >> theCurrentTemp.entfy[i].ygsig[j];
	   }
	   			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].yeavg[j] >> theCurrentTemp.entfy[i].yerms[j] >> theCurrentTemp.entfy[i].yegx0[j] >> theCurrentTemp.entfy[i].yegsig[j];
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].yoavg[j] >> theCurrentTemp.entfy[i].yorms[j] >> theCurrentTemp.entfy[i].yogx0[j] >> theCurrentTemp.entfy[i].yogsig[j];
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].xavg[j] >> theCurrentTemp.entfy[i].xrms[j] >> theCurrentTemp.entfy[i].xgx0[j] >> theCurrentTemp.entfy[i].xgsig[j];
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].xeavg[j] >> theCurrentTemp.entfy[i].xerms[j] >> theCurrentTemp.entfy[i].xegx0[j] >> theCurrentTemp.entfy[i].xegsig[j];
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfy[i].xoavg[j] >> theCurrentTemp.entfy[i].xorms[j] >> theCurrentTemp.entfy[i].xogx0[j] >> theCurrentTemp.entfy[i].xogsig[j];
	   }
    	   
	}
	
// next, loop over all forward x-angle entries   

    for (i=0; i < theCurrentTemp.head.NFx; ++i) {     
    
       in_file >> theCurrentTemp.entfx[i].runnum >> theCurrentTemp.entfx[i].costrk[0] 
	           >> theCurrentTemp.entfx[i].costrk[1] >> theCurrentTemp.entfx[i].costrk[2]; 
			  
// Calculate the alpha, beta, and cot(beta) for this entry 

       theCurrentTemp.entfx[i].alpha = static_cast<float>(atan2((double)theCurrentTemp.entfx[i].costrk[2], (double)theCurrentTemp.entfx[i].costrk[0]));
	   
	   theCurrentTemp.entfx[i].cotalpha = theCurrentTemp.entfx[i].costrk[0]/theCurrentTemp.entfx[i].costrk[2];

       theCurrentTemp.entfx[i].beta = static_cast<float>(atan2((double)theCurrentTemp.entfx[i].costrk[2], (double)theCurrentTemp.entfx[i].costrk[1]));
	   
	   theCurrentTemp.entfx[i].cotbeta = theCurrentTemp.entfx[i].costrk[1]/theCurrentTemp.entfx[i].costrk[2];
    
       in_file >> theCurrentTemp.entfx[i].qavg >> theCurrentTemp.entfx[i].symax >> theCurrentTemp.entfx[i].dyone
	           >> theCurrentTemp.entfx[i].syone >> theCurrentTemp.entfx[i].sxmax >> theCurrentTemp.entfx[i].dxone >> theCurrentTemp.entfx[i].sxone;
    
       in_file >> theCurrentTemp.entfx[i].dytwo >> theCurrentTemp.entfx[i].sytwo >> theCurrentTemp.entfx[i].dxtwo 
	           >> theCurrentTemp.entfx[i].sxtwo;
			  
	   for (j=0; j<2; ++j) {
    
          in_file >> theCurrentTemp.entfx[i].ypar[j][0] >> theCurrentTemp.entfx[i].ypar[j][1] 
	              >> theCurrentTemp.entfx[i].ypar[j][2] >> theCurrentTemp.entfx[i].ypar[j][3] >> theCurrentTemp.entfx[i].ypar[j][4];
			  
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entfx[i].ytemp[j][0] >> theCurrentTemp.entfx[i].ytemp[j][1] >> theCurrentTemp.entfx[i].ytemp[j][2]
	              >> theCurrentTemp.entfx[i].ytemp[j][3] >> theCurrentTemp.entfx[i].ytemp[j][4] >> theCurrentTemp.entfx[i].ytemp[j][5]
	              >> theCurrentTemp.entfx[i].ytemp[j][6] >> theCurrentTemp.entfx[i].ytemp[j][7] >> theCurrentTemp.entfx[i].ytemp[j][8]
	              >> theCurrentTemp.entfx[i].ytemp[j][9] >> theCurrentTemp.entfx[i].ytemp[j][10] >> theCurrentTemp.entfx[i].ytemp[j][11]
	              >> theCurrentTemp.entfx[i].ytemp[j][12] >> theCurrentTemp.entfx[i].ytemp[j][13] >> theCurrentTemp.entfx[i].ytemp[j][14]
	              >> theCurrentTemp.entfx[i].ytemp[j][15] >> theCurrentTemp.entfx[i].ytemp[j][16] >> theCurrentTemp.entfx[i].ytemp[j][17]
	              >> theCurrentTemp.entfx[i].ytemp[j][18] >> theCurrentTemp.entfx[i].ytemp[j][19] >> theCurrentTemp.entfx[i].ytemp[j][20];
	   }
   			  
	   for (j=0; j<2; ++j) {
    
		  in_file >> theCurrentTemp.entfx[i].xpar[j][0] >> theCurrentTemp.entfx[i].xpar[j][1] 
	              >> theCurrentTemp.entfx[i].xpar[j][2] >> theCurrentTemp.entfx[i].xpar[j][3] >> theCurrentTemp.entfx[i].xpar[j][4];
			  
	   }
			  
	   for (j=0; j<9; ++j) {
    
          in_file >> theCurrentTemp.entfx[i].xtemp[j][0] >> theCurrentTemp.entfx[i].xtemp[j][1] >> theCurrentTemp.entfx[i].xtemp[j][2]
	              >> theCurrentTemp.entfx[i].xtemp[j][3] >> theCurrentTemp.entfx[i].xtemp[j][4] >> theCurrentTemp.entfx[i].xtemp[j][5]
	              >> theCurrentTemp.entfx[i].xtemp[j][6];
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfx[i].yavg[j] >> theCurrentTemp.entfx[i].yrms[j] >> theCurrentTemp.entfx[i].ygx0[j] >> theCurrentTemp.entfx[i].ygsig[j];
	   }
	   			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfx[i].yeavg[j] >> theCurrentTemp.entfx[i].yerms[j] >> theCurrentTemp.entfx[i].yegx0[j] >> theCurrentTemp.entfx[i].yegsig[j];
	   }
	   
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfx[i].yoavg[j] >> theCurrentTemp.entfx[i].yorms[j] >> theCurrentTemp.entfx[i].yogx0[j] >> theCurrentTemp.entfx[i].yogsig[j];
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfx[i].xavg[j] >> theCurrentTemp.entfx[i].xrms[j] >> theCurrentTemp.entfx[i].xgx0[j] >> theCurrentTemp.entfx[i].xgsig[j];
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfx[i].xeavg[j] >> theCurrentTemp.entfx[i].xerms[j] >> theCurrentTemp.entfx[i].xegx0[j] >> theCurrentTemp.entfx[i].xegsig[j];
	   }
			  
	   for (j=0; j<4; ++j) {
    
          in_file >> theCurrentTemp.entfx[i].xoavg[j] >> theCurrentTemp.entfx[i].xorms[j] >> theCurrentTemp.entfx[i].xogx0[j] >> theCurrentTemp.entfx[i].xogsig[j];
	   }
    	   
	}
    
    in_file.close();
	
// Add this template to the store
	
	thePixelTemp.push_back(theCurrentTemp);
	
 } else {
 
 // If file didn't open, report this
 
    std::cout << "Error opening File" << tempfile << std::endl;
	
 }
	
	return;
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
	int ilow, ihigh, Nx, Ny, imidy;
	float yratio, xratio, qscale, sxmax0;
	std::vector <float> xrms(4), xgsig(4), xerms(4), xegsig(4), xorms(4), xogsig(4);


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

// Copy the pseudopixel signal size to the template     
    
    ps50 = thePixelTemp[index_id].head.s50;
	
// Decide which template (FPix or BPix) to use 

    if(fpix) {
    
// Begin FPix section, make the index counters easier to use     
    
       Ny = thePixelTemp[index_id].head.NFy;
       Nx = thePixelTemp[index_id].head.NFx;
	   imidy = Nx/2;
        
// next, loop over all y-angle entries   

       if(cotbeta < thePixelTemp[index_id].entfy[0].cotbeta) {
	
	       ilow = 0;
		   yratio = 0.;

	   } else if(cotbeta > thePixelTemp[index_id].entfy[Ny-1].cotbeta) {
	
	       ilow = Ny-2;
		   yratio = 1.;
		
	   } else {

          for (i=0; i<Ny-2; ++i) { 
    
             if( thePixelTemp[index_id].entfy[i].cotbeta <= cotbeta && cotbeta < thePixelTemp[index_id].entfy[i+1].cotbeta) {
		  
	            ilow = i;
		        yratio = (cotbeta - thePixelTemp[index_id].entfy[i].cotbeta)/(thePixelTemp[index_id].entfy[i+1].cotbeta - thePixelTemp[index_id].entfy[i].cotbeta);
		        break;			 
		     }
	      }
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all y-related quantities 

       pyratio = yratio;
	   pqavg = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].qavg + yratio*thePixelTemp[index_id].entfy[ihigh].qavg;
	   psymax = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].symax + yratio*thePixelTemp[index_id].entfy[ihigh].symax;
	   sxmax0 = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].sxmax + yratio*thePixelTemp[index_id].entfy[ihigh].sxmax;
	   pdyone = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].dyone + yratio*thePixelTemp[index_id].entfy[ihigh].dyone;
	   psyone = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].syone + yratio*thePixelTemp[index_id].entfy[ihigh].syone;
	   pdytwo = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].dytwo + yratio*thePixelTemp[index_id].entfy[ihigh].dytwo;
	   psytwo = (1. - yratio)*thePixelTemp[index_id].entfy[ilow].sytwo + yratio*thePixelTemp[index_id].entfy[ihigh].sytwo;
	   for(i=0; i<2 ; ++i) {
	      for(j=0; j<5 ; ++j) {
	         pyparl[i][j] = thePixelTemp[index_id].entfy[ilow].ypar[i][j];
	         pyparh[i][j] = thePixelTemp[index_id].entfy[ihigh].ypar[i][j];
	      }
	   }
	   for(i=0; i<4; ++i) {
	      pyavg[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yavg[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yavg[i];
	      pyrms[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yrms[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yrms[i];
	      pygx0[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ygx0[i] + yratio*thePixelTemp[index_id].entfy[ihigh].ygx0[i];
	      pygsig[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ygsig[i] + yratio*thePixelTemp[index_id].entfy[ihigh].ygsig[i];
	      pyeavg[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yeavg[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yeavg[i];
	      pyerms[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yerms[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yerms[i];
	      pyegx0[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yegx0[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yegx0[i];
	      pyegsig[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yegsig[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yegsig[i];
	      pyoavg[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yoavg[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yoavg[i];
	      pyorms[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yorms[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yorms[i];
	      pyogx0[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yogx0[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yogx0[i];
	      pyogsig[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].yogsig[i] + yratio*thePixelTemp[index_id].entfy[ihigh].yogsig[i];
	      xrms[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].xrms[i] + yratio*thePixelTemp[index_id].entfy[ihigh].xrms[i];
	      xgsig[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].xgsig[i] + yratio*thePixelTemp[index_id].entfy[ihigh].xgsig[i];
	      xerms[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].xerms[i] + yratio*thePixelTemp[index_id].entfy[ihigh].xerms[i];
	      xegsig[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].xegsig[i] + yratio*thePixelTemp[index_id].entfy[ihigh].xegsig[i];
	      xorms[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].xorms[i] + yratio*thePixelTemp[index_id].entfy[ihigh].xorms[i];
	      xogsig[i]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].xogsig[i] + yratio*thePixelTemp[index_id].entfy[ihigh].xogsig[i];
	   }
			  
// Interpolate and build the y-template 
	
	   for(i=0; i<9; ++i) {
          pytemp[i+16][0] = 0.;
          pytemp[i+16][1] = 0.;
	      pytemp[i+16][23] = 0.;
	      pytemp[i+16][24] = 0.;
	      for(j=0; j<21; ++j) {
	        pytemp[i+16][j+2]=(1. - yratio)*thePixelTemp[index_id].entfy[ilow].ytemp[i][j] + yratio*thePixelTemp[index_id].entfy[ihigh].ytemp[i][j];
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
	
// next, loop over all x-angle entries   

       if(cotalpha < thePixelTemp[index_id].entfx[0].cotalpha) {
	
	       ilow = 0;
		   xratio = 0.;

	   } else if(cotalpha > thePixelTemp[index_id].entfx[Nx-1].cotalpha) {
	
	       ilow = Nx-2;
		   xratio = 1.;
		
	   } else {

          for (i=0; i<Nx-2; ++i) { 
    
             if( thePixelTemp[index_id].entfx[i].cotalpha <= cotalpha && cotalpha < thePixelTemp[index_id].entfx[i+1].cotalpha) {
		  
	            ilow = i;
		        xratio = (cotalpha - thePixelTemp[index_id].entfx[i].cotalpha)/(thePixelTemp[index_id].entfx[i+1].cotalpha - thePixelTemp[index_id].entfx[i].cotalpha);
		        break;
			 
		     }
	      }
	   }
	
	   ihigh=ilow + 1;
	
// Define some charge scaling factors 

       qscale = sxmax0/thePixelTemp[index_id].entfx[imidy].sxmax;
			  
// Interpolate/store all x-related quantities 

       pxratio = xratio;				  
// sxparmax defines the maximum charge for which the parameters xpar are defined (not scaled by sqscale) 
	   psxparmax = (1. - xratio)*thePixelTemp[index_id].entfx[ilow].sxmax + xratio*thePixelTemp[index_id].entfx[ihigh].sxmax;
	   psxmax = qscale*psxparmax;
	   pdxone = (1. - xratio)*thePixelTemp[index_id].entfx[ilow].dxone + xratio*thePixelTemp[index_id].entfx[ihigh].dxone;
	   psxone = (1. - xratio)*thePixelTemp[index_id].entfx[ilow].sxone + xratio*thePixelTemp[index_id].entfx[ihigh].sxone;
	   pdxtwo = (1. - xratio)*thePixelTemp[index_id].entfx[ilow].dxtwo + xratio*thePixelTemp[index_id].entfx[ihigh].dxtwo;
	   psxtwo = (1. - xratio)*thePixelTemp[index_id].entfx[ilow].sxtwo + xratio*thePixelTemp[index_id].entfx[ihigh].sxtwo;
	   for(i=0; i<2 ; ++i) {
	      for(j=0; j<5 ; ++j) {
	         pxparl[i][j] = qscale*thePixelTemp[index_id].entfx[ilow].xpar[i][j];
	         pxparh[i][j] = qscale*thePixelTemp[index_id].entfx[ihigh].xpar[i][j];
	      }
	   }
	   for(i=0; i<4; ++i) {
	      pxavg[i]=((1. - xratio)*thePixelTemp[index_id].entfx[ilow].xavg[i] + xratio*thePixelTemp[index_id].entfx[ihigh].xavg[i])
	                        /thePixelTemp[index_id].entfx[imidy].xrms[i]*xrms[i];
	      pxrms[i]=((1. - xratio)*thePixelTemp[index_id].entfx[ilow].xrms[i] + xratio*thePixelTemp[index_id].entfx[ihigh].xrms[i])
	                        /thePixelTemp[index_id].entfx[imidy].xrms[i]*xrms[i];
	      pxgx0[i]=((1. - xratio)*thePixelTemp[index_id].entfx[ilow].xgx0[i] + xratio*thePixelTemp[index_id].entfx[ihigh].xgx0[i])
	                        /thePixelTemp[index_id].entfx[imidy].xgsig[i]*xgsig[i];
	      pxgsig[i]=((1. - xratio)*thePixelTemp[index_id].entfx[ilow].xgsig[i] + xratio*thePixelTemp[index_id].entfx[ihigh].xgsig[i])
	                        /thePixelTemp[index_id].entfx[imidy].xgsig[i]*xgsig[i];
	      pxeavg[i]=((1. - xratio)*thePixelTemp[index_id].entfx[ilow].xeavg[i] + xratio*thePixelTemp[index_id].entfx[ihigh].xeavg[i])
	                        /thePixelTemp[index_id].entfx[imidy].xerms[i]*xerms[i];
	      pxerms[i]=((1. - xratio)*thePixelTemp[index_id].entfx[ilow].xerms[i] + xratio*thePixelTemp[index_id].entfx[ihigh].xerms[i])
	                        /thePixelTemp[index_id].entfx[imidy].xerms[i]*xerms[i];
	      pxegx0[i]=((1. - xratio)*thePixelTemp[index_id].entfx[ilow].xegx0[i] + xratio*thePixelTemp[index_id].entfx[ihigh].xegx0[i])
	                        /thePixelTemp[index_id].entfx[imidy].xegsig[i]*xegsig[i];
	      pxegsig[i]=((1. - xratio)*thePixelTemp[index_id].entfx[ilow].xegsig[i] + xratio*thePixelTemp[index_id].entfx[ihigh].xegsig[i])
	                        /thePixelTemp[index_id].entfx[imidy].xegsig[i]*xegsig[i];
	      pxoavg[i]=((1. - xratio)*thePixelTemp[index_id].entfx[ilow].xoavg[i] + xratio*thePixelTemp[index_id].entfx[ihigh].xoavg[i])
	                        /thePixelTemp[index_id].entfx[imidy].xorms[i]*xorms[i];
	      pxorms[i]=((1. - xratio)*thePixelTemp[index_id].entfx[ilow].xorms[i] + xratio*thePixelTemp[index_id].entfx[ihigh].xorms[i])
	                        /thePixelTemp[index_id].entfx[imidy].xorms[i]*xorms[i];
	      pxogx0[i]=((1. - xratio)*thePixelTemp[index_id].entfx[ilow].xogx0[i] + xratio*thePixelTemp[index_id].entfx[ihigh].xogx0[i])
	                        /thePixelTemp[index_id].entfx[imidy].xogsig[i]*xogsig[i];
	      pxogsig[i]=((1. - xratio)*thePixelTemp[index_id].entfx[ilow].xogsig[i] + xratio*thePixelTemp[index_id].entfx[ihigh].xogsig[i])
	                        /thePixelTemp[index_id].entfx[imidy].xogsig[i]*xogsig[i];
	   }
			  
// Interpolate and build the x-template 
	
	   for(i=0; i<9; ++i) {
          pxtemp[i+16][0] = 0.;
          pxtemp[i+16][1] = 0.;
	      pxtemp[i+16][9] = 0.;
	      pxtemp[i+16][10] = 0.;
	      for(j=0; j<7; ++j) {
	        pxtemp[i+16][j+2]=(1. - xratio)*thePixelTemp[index_id].entfx[ilow].xtemp[i][j] + xratio*thePixelTemp[index_id].entfx[ihigh].xtemp[i][j];
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
       Nx = thePixelTemp[index_id].head.NBx;
	   imidy = Nx/2;
        
// next, loop over all y-angle entries   

       if(cotbeta < thePixelTemp[index_id].entby[0].cotbeta) {
	
	       ilow = 0;
		   yratio = 0.;

	   } else if(cotbeta > thePixelTemp[index_id].entby[Ny-1].cotbeta) {
	
	       ilow = Ny-2;
		   yratio = 1.;
		
	   } else {

          for (i=0; i<Ny-2; ++i) { 
    
             if( thePixelTemp[index_id].entby[i].cotbeta <= cotbeta && cotbeta < thePixelTemp[index_id].entby[i+1].cotbeta) {
		  
	            ilow = i;
		        yratio = (cotbeta - thePixelTemp[index_id].entby[i].cotbeta)/(thePixelTemp[index_id].entby[i+1].cotbeta - thePixelTemp[index_id].entby[i].cotbeta);
		        break;			 
		     }
	      }
	   }
	
	   ihigh=ilow + 1;
			  
// Interpolate/store all y-related quantities 

       pyratio = yratio;
	   pqavg = (1. - yratio)*thePixelTemp[index_id].entby[ilow].qavg + yratio*thePixelTemp[index_id].entby[ihigh].qavg;
	   psymax = (1. - yratio)*thePixelTemp[index_id].entby[ilow].symax + yratio*thePixelTemp[index_id].entby[ihigh].symax;
	   sxmax0 = (1. - yratio)*thePixelTemp[index_id].entby[ilow].sxmax + yratio*thePixelTemp[index_id].entby[ihigh].sxmax;
	   pdyone = (1. - yratio)*thePixelTemp[index_id].entby[ilow].dyone + yratio*thePixelTemp[index_id].entby[ihigh].dyone;
	   psyone = (1. - yratio)*thePixelTemp[index_id].entby[ilow].syone + yratio*thePixelTemp[index_id].entby[ihigh].syone;
	   pdytwo = (1. - yratio)*thePixelTemp[index_id].entby[ilow].dytwo + yratio*thePixelTemp[index_id].entby[ihigh].dytwo;
	   psytwo = (1. - yratio)*thePixelTemp[index_id].entby[ilow].sytwo + yratio*thePixelTemp[index_id].entby[ihigh].sytwo;
	   for(i=0; i<2 ; ++i) {
	      for(j=0; j<5 ; ++j) {
	         pyparl[i][j] = thePixelTemp[index_id].entby[ilow].ypar[i][j];
	         pyparh[i][j] = thePixelTemp[index_id].entby[ihigh].ypar[i][j];
	      }
	   }
	   for(i=0; i<4; ++i) {
	      pyavg[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yavg[i] + yratio*thePixelTemp[index_id].entby[ihigh].yavg[i];
	      pyrms[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yrms[i] + yratio*thePixelTemp[index_id].entby[ihigh].yrms[i];
	      pygx0[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ygx0[i] + yratio*thePixelTemp[index_id].entby[ihigh].ygx0[i];
	      pygsig[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ygsig[i] + yratio*thePixelTemp[index_id].entby[ihigh].ygsig[i];
	      pyeavg[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yeavg[i] + yratio*thePixelTemp[index_id].entby[ihigh].yeavg[i];
	      pyerms[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yerms[i] + yratio*thePixelTemp[index_id].entby[ihigh].yerms[i];
	      pyegx0[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yegx0[i] + yratio*thePixelTemp[index_id].entby[ihigh].yegx0[i];
	      pyegsig[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yegsig[i] + yratio*thePixelTemp[index_id].entby[ihigh].yegsig[i];
	      pyoavg[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yoavg[i] + yratio*thePixelTemp[index_id].entby[ihigh].yoavg[i];
	      pyorms[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yorms[i] + yratio*thePixelTemp[index_id].entby[ihigh].yorms[i];
	      pyogx0[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yogx0[i] + yratio*thePixelTemp[index_id].entby[ihigh].yogx0[i];
	      pyogsig[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].yogsig[i] + yratio*thePixelTemp[index_id].entby[ihigh].yogsig[i];
	      xrms[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].xrms[i] + yratio*thePixelTemp[index_id].entby[ihigh].xrms[i];
	      xgsig[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].xgsig[i] + yratio*thePixelTemp[index_id].entby[ihigh].xgsig[i];
	      xerms[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].xerms[i] + yratio*thePixelTemp[index_id].entby[ihigh].xerms[i];
	      xegsig[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].xegsig[i] + yratio*thePixelTemp[index_id].entby[ihigh].xegsig[i];
	      xorms[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].xorms[i] + yratio*thePixelTemp[index_id].entby[ihigh].xorms[i];
	      xogsig[i]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].xogsig[i] + yratio*thePixelTemp[index_id].entby[ihigh].xogsig[i];
	   }
			  
// Interpolate and build the y-template 
	
	   for(i=0; i<9; ++i) {
          pytemp[i+16][0] = 0.;
          pytemp[i+16][1] = 0.;
	      pytemp[i+16][23] = 0.;
	      pytemp[i+16][24] = 0.;
	      for(j=0; j<21; ++j) {
	        pytemp[i+16][j+2]=(1. - yratio)*thePixelTemp[index_id].entby[ilow].ytemp[i][j] + yratio*thePixelTemp[index_id].entby[ihigh].ytemp[i][j];
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
	
// next, loop over all x-angle entries   

       if(cotalpha < thePixelTemp[index_id].entbx[0].cotalpha) {
	
	       ilow = 0;
		   xratio = 0.;

	   } else if(cotalpha > thePixelTemp[index_id].entbx[Nx-1].cotalpha) {
	
	       ilow = Nx-2;
		   xratio = 1.;
		
	   } else {

          for (i=0; i<Nx-2; ++i) { 
    
             if( thePixelTemp[index_id].entbx[i].cotalpha <= cotalpha && cotalpha < thePixelTemp[index_id].entbx[i+1].cotalpha) {
		  
	            ilow = i;
		        xratio = (cotalpha - thePixelTemp[index_id].entbx[i].cotalpha)/(thePixelTemp[index_id].entbx[i+1].cotalpha - thePixelTemp[index_id].entbx[i].cotalpha);
		        break;
			 
		     }
	      }
	   }
	
	   ihigh=ilow + 1;
	
// Define some charge scaling factors 

       qscale = sxmax0/thePixelTemp[index_id].entbx[imidy].sxmax;
			  
// Interpolate/store all x-related quantities 

       pxratio = xratio;				  
// sxparmax defines the maximum charge for which the parameters xpar are defined (not scaled by sqscale) 
	   psxparmax = (1. - xratio)*thePixelTemp[index_id].entbx[ilow].sxmax + xratio*thePixelTemp[index_id].entbx[ihigh].sxmax;
	   psxmax = qscale*psxparmax;
	   pdxone = (1. - xratio)*thePixelTemp[index_id].entbx[ilow].dxone + xratio*thePixelTemp[index_id].entbx[ihigh].dxone;
	   psxone = (1. - xratio)*thePixelTemp[index_id].entbx[ilow].sxone + xratio*thePixelTemp[index_id].entbx[ihigh].sxone;
	   pdxtwo = (1. - xratio)*thePixelTemp[index_id].entbx[ilow].dxtwo + xratio*thePixelTemp[index_id].entbx[ihigh].dxtwo;
	   psxtwo = (1. - xratio)*thePixelTemp[index_id].entbx[ilow].sxtwo + xratio*thePixelTemp[index_id].entbx[ihigh].sxtwo;
	   for(i=0; i<2 ; ++i) {
	      for(j=0; j<5 ; ++j) {
	         pxparl[i][j] = qscale*thePixelTemp[index_id].entbx[ilow].xpar[i][j];
	         pxparh[i][j] = qscale*thePixelTemp[index_id].entbx[ihigh].xpar[i][j];
	      }
	   }
	   for(i=0; i<4; ++i) {
	      pxavg[i]=((1. - xratio)*thePixelTemp[index_id].entbx[ilow].xavg[i] + xratio*thePixelTemp[index_id].entbx[ihigh].xavg[i])
	                        /thePixelTemp[index_id].entbx[imidy].xrms[i]*xrms[i];
	      pxrms[i]=((1. - xratio)*thePixelTemp[index_id].entbx[ilow].xrms[i] + xratio*thePixelTemp[index_id].entbx[ihigh].xrms[i])
	                        /thePixelTemp[index_id].entbx[imidy].xrms[i]*xrms[i];
	      pxgx0[i]=((1. - xratio)*thePixelTemp[index_id].entbx[ilow].xgx0[i] + xratio*thePixelTemp[index_id].entbx[ihigh].xgx0[i])
	                        /thePixelTemp[index_id].entbx[imidy].xgsig[i]*xgsig[i];
	      pxgsig[i]=((1. - xratio)*thePixelTemp[index_id].entbx[ilow].xgsig[i] + xratio*thePixelTemp[index_id].entbx[ihigh].xgsig[i])
	                        /thePixelTemp[index_id].entbx[imidy].xgsig[i]*xgsig[i];
	      pxeavg[i]=((1. - xratio)*thePixelTemp[index_id].entbx[ilow].xeavg[i] + xratio*thePixelTemp[index_id].entbx[ihigh].xeavg[i])
	                        /thePixelTemp[index_id].entbx[imidy].xerms[i]*xerms[i];
	      pxerms[i]=((1. - xratio)*thePixelTemp[index_id].entbx[ilow].xerms[i] + xratio*thePixelTemp[index_id].entbx[ihigh].xerms[i])
	                        /thePixelTemp[index_id].entbx[imidy].xerms[i]*xerms[i];
	      pxegx0[i]=((1. - xratio)*thePixelTemp[index_id].entbx[ilow].xegx0[i] + xratio*thePixelTemp[index_id].entbx[ihigh].xegx0[i])
	                        /thePixelTemp[index_id].entbx[imidy].xegsig[i]*xegsig[i];
	      pxegsig[i]=((1. - xratio)*thePixelTemp[index_id].entbx[ilow].xegsig[i] + xratio*thePixelTemp[index_id].entbx[ihigh].xegsig[i])
	                        /thePixelTemp[index_id].entbx[imidy].xegsig[i]*xegsig[i];
	      pxoavg[i]=((1. - xratio)*thePixelTemp[index_id].entbx[ilow].xoavg[i] + xratio*thePixelTemp[index_id].entbx[ihigh].xoavg[i])
	                        /thePixelTemp[index_id].entbx[imidy].xorms[i]*xorms[i];
	      pxorms[i]=((1. - xratio)*thePixelTemp[index_id].entbx[ilow].xorms[i] + xratio*thePixelTemp[index_id].entbx[ihigh].xorms[i])
	                        /thePixelTemp[index_id].entbx[imidy].xorms[i]*xorms[i];
	      pxogx0[i]=((1. - xratio)*thePixelTemp[index_id].entbx[ilow].xogx0[i] + xratio*thePixelTemp[index_id].entbx[ihigh].xogx0[i])
	                        /thePixelTemp[index_id].entbx[imidy].xogsig[i]*xogsig[i];
	      pxogsig[i]=((1. - xratio)*thePixelTemp[index_id].entbx[ilow].xogsig[i] + xratio*thePixelTemp[index_id].entbx[ihigh].xogsig[i])
	                        /thePixelTemp[index_id].entbx[imidy].xogsig[i]*xogsig[i];
	   }
			  
// Interpolate and build the x-template 
	
	   for(i=0; i<9; ++i) {
          pxtemp[i+16][0] = 0.;
          pxtemp[i+16][1] = 0.;
	      pxtemp[i+16][9] = 0.;
	      pxtemp[i+16][10] = 0.;
	      for(j=0; j<7; ++j) {
	        pxtemp[i+16][j+2]=(1. - xratio)*thePixelTemp[index_id].entbx[ilow].xtemp[i][j] + xratio*thePixelTemp[index_id].entbx[ihigh].xtemp[i][j];
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
