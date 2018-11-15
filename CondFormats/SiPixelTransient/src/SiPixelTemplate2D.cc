//
//  SiPixelTemplate2D.cc  Version 2.65
//
//  Full 2-D templates for cluster splitting, simulated cluster reweighting, and improved cluster probability
//
// Created by Morris Swartz on 12/01/09.
// 2009 __TheJohnsHopkinsUniversity__.
//
// V1.01 - fix qavg_ filling
// V1.02 - Add locBz to test if FPix use is out of range
// V1.03 - Fix edge checking on final template to increase template size and to properly truncate cluster
// v2.00 - Major changes to accommodate 2D reconstruction
// v2.10 - Change chi2 and error scaling information to work with partially reconstructed clusters
// v2.20 - Add cluster charge probability information, side loading for template generation
// v2.21 - Double derivative interval [improves fit convergence]
// v2.25 - Resize template store to accommodate FPix Templates
// v2.30 - Fix bug found by P. Shuetze that compromises sqlite file loading
// v2.35 - Add directory path selection to the ascii pushfile method
// v2.50 - Change template storage to dynamically allocated 2D arrays of SiPixelTemplateEntry2D structs
// v2.51 - Ensure that the derivative arrays are correctly zeroed between calls
// v2.52 - Improve cosmetics for increased style points from judges
// v2.60 - Fix FPix multiframe lookup problem [takes +-cotalpha and +-cotbeta]
// v2.61a - Code 2.60 fix correctly
// v2.65 - change double pixel flags to work with new shifted reco code definition
//

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
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
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate2D.h"
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
bool SiPixelTemplate2D::pushfile(int filenum, std::vector< SiPixelTemplateStore2D > & pixelTemp, std::string dir)
{
   // Add template stored in external file numbered filenum to theTemplateStore
   
   // Local variables
   const int code_version={21};
   
   //  Create a filename for this run
   std::string tempfile = std::to_string(filenum);
   
   
   //  Create different path in CMSSW than standalone
   
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
   // If integer filenum has less than 4 digits, prepend 0's until we have four numerical characters, e.g. "0292"
   int nzeros = 4-tempfile.length();
   if (nzeros > 0)
     tempfile = std::string(nzeros, '0') + tempfile;
   /// Alt implementation: for (unsigned cnt=4-tempfile.length(); cnt > 0; cnt-- ){ tempfile = "0" + tempfile; }

   tempfile = dir + "template_summary2D_zp" + tempfile + ".out";
   edm::FileInPath file( tempfile );         // Find the file in CMSSW
   tempfile = file.fullPath();               // Put it back with the whole path.

#else
   // This is the same as above, but more elegant.
   std::ostringstream tout;
   tout << "template_summary2D_zp" << std::setw(4) << std::setfill('0') << std::right << filenum << ".out" << std::ends;
   tempfile = tout.str();

#endif
   
   //  Open the template file
   //
   std::ifstream in_file(tempfile);
   if(in_file.is_open() && in_file.good()) {

      // Create a local template storage entry
      SiPixelTemplateStore2D theCurrentTemp;
      
      // Read-in a header string first and print it
      char c;
      for (int i=0; (c=in_file.get()) != '\n'; ++i) {
         if(i < 79) {theCurrentTemp.head.title[i] = c;}
         else       {theCurrentTemp.head.title[79] ='\0';}
      }
      LOGINFO("SiPixelTemplate2D") << "Loading Pixel Template File - " << theCurrentTemp.head.title << ENDL;
      
      // next, the header information
      in_file >> theCurrentTemp.head.ID  >> theCurrentTemp.head.templ_version >> theCurrentTemp.head.Bfield >> theCurrentTemp.head.NTy >> theCurrentTemp.head.NTyx >> theCurrentTemp.head.NTxx
      >> theCurrentTemp.head.Dtype >> theCurrentTemp.head.Vbias >> theCurrentTemp.head.temperature >> theCurrentTemp.head.fluence >> theCurrentTemp.head.qscale
      >> theCurrentTemp.head.s50 >> theCurrentTemp.head.lorywidth >> theCurrentTemp.head.lorxwidth >> theCurrentTemp.head.ysize >> theCurrentTemp.head.xsize >> theCurrentTemp.head.zsize;
      
      if(in_file.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 0A, no template load" << ENDL; return false;}
      
      if(theCurrentTemp.head.templ_version > 17) {
         in_file >> theCurrentTemp.head.ss50 >> theCurrentTemp.head.lorybias
         >> theCurrentTemp.head.lorxbias >> theCurrentTemp.head.fbin[0]
         >> theCurrentTemp.head.fbin[1] >> theCurrentTemp.head.fbin[2];
         
         if(in_file.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 0B, no template load"
            << ENDL; return false;}
      } else {
// This is for older [legacy] payloads 
         theCurrentTemp.head.ss50 = theCurrentTemp.head.s50;
         theCurrentTemp.head.lorybias = theCurrentTemp.head.lorywidth/2.f;
         theCurrentTemp.head.lorxbias = theCurrentTemp.head.lorxwidth/2.f;
         theCurrentTemp.head.fbin[0] = 1.5f;
         theCurrentTemp.head.fbin[1] = 1.00f;
         theCurrentTemp.head.fbin[2] = 0.85f;
      }
      
      LOGINFO("SiPixelTemplate2D") << "Template ID = " << theCurrentTemp.head.ID << ", Template Version " << theCurrentTemp.head.templ_version << ", Bfield = " << theCurrentTemp.head.Bfield
      << ", NTy = " << theCurrentTemp.head.NTy << ", NTyx = " << theCurrentTemp.head.NTyx<< ", NTxx = " << theCurrentTemp.head.NTxx << ", Dtype = " << theCurrentTemp.head.Dtype
      << ", Bias voltage " << theCurrentTemp.head.Vbias << ", temperature "
      << theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence << ", Q-scaling factor " << theCurrentTemp.head.qscale
      << ", 1/2 multi dcol threshold " << theCurrentTemp.head.s50 << ", 1/2 single dcol threshold " << theCurrentTemp.head.ss50
      << ", y Lorentz Width " << theCurrentTemp.head.lorywidth << ", y Lorentz Bias " << theCurrentTemp.head.lorybias
      << ", x Lorentz width " << theCurrentTemp.head.lorxwidth << ", x Lorentz Bias " << theCurrentTemp.head.lorxbias
      << ", Q/Q_avg fractions for Qbin defs " << theCurrentTemp.head.fbin[0] << ", " << theCurrentTemp.head.fbin[1]
      << ", " << theCurrentTemp.head.fbin[2]
      << ", pixel x-size " << theCurrentTemp.head.xsize << ", y-size " << theCurrentTemp.head.ysize << ", zsize " << theCurrentTemp.head.zsize << ENDL;
      
      if(theCurrentTemp.head.templ_version < code_version) {LOGERROR("SiPixelTemplate2D") << "code expects version " << code_version << ", no template load" << ENDL; return false;}
      
      if(theCurrentTemp.head.NTy != 0) {LOGERROR("SiPixelTemplate2D") << "Trying to load 1-d template info into the 2-d template object, check your DB/global tag!" << ENDL; return false;}
      
      // next, layout the 2-d structure needed to store template
      
      theCurrentTemp.entry = new SiPixelTemplateEntry2D*[theCurrentTemp.head.NTyx];
      theCurrentTemp.entry[0] = new SiPixelTemplateEntry2D[theCurrentTemp.head.NTyx*theCurrentTemp.head.NTxx];
      for(int i = 1; i < theCurrentTemp.head.NTyx; ++i) theCurrentTemp.entry[i] = theCurrentTemp.entry[i-1] + theCurrentTemp.head.NTxx;      

      // Read in the file info
      
      for (int iy=0; iy < theCurrentTemp.head.NTyx; ++iy) {
         for(int jx=0; jx < theCurrentTemp.head.NTxx; ++jx) {
            
            in_file >> theCurrentTemp.entry[iy][jx].runnum >> theCurrentTemp.entry[iy][jx].costrk[0]
            >> theCurrentTemp.entry[iy][jx].costrk[1] >> theCurrentTemp.entry[iy][jx].costrk[2];
            
            if(in_file.fail()) {
	      LOGERROR("SiPixelTemplate2D") << "Error reading file 1, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL;
	      delete [] theCurrentTemp.entry[0];
	      delete [] theCurrentTemp.entry;
	      return false;
	    }
            
            // Calculate cot(alpha) and cot(beta) for this entry
            
            theCurrentTemp.entry[iy][jx].cotalpha = theCurrentTemp.entry[iy][jx].costrk[0]/theCurrentTemp.entry[iy][jx].costrk[2];
            
            theCurrentTemp.entry[iy][jx].cotbeta = theCurrentTemp.entry[iy][jx].costrk[1]/theCurrentTemp.entry[iy][jx].costrk[2];
            
            in_file >> theCurrentTemp.entry[iy][jx].qavg >> theCurrentTemp.entry[iy][jx].pixmax >> theCurrentTemp.entry[iy][jx].sxymax >> theCurrentTemp.entry[iy][jx].iymin
            >> theCurrentTemp.entry[iy][jx].iymax >> theCurrentTemp.entry[iy][jx].jxmin >> theCurrentTemp.entry[iy][jx].jxmax;
            
            if(in_file.fail()) {
	      LOGERROR("SiPixelTemplate2D") << "Error reading file 2, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL;
	      delete [] theCurrentTemp.entry[0];
	      delete [] theCurrentTemp.entry; 
	      return false;
	    }
            
            for (int k=0; k<2; ++k) {
               
               in_file >> theCurrentTemp.entry[iy][jx].xypar[k][0] >> theCurrentTemp.entry[iy][jx].xypar[k][1]
               >> theCurrentTemp.entry[iy][jx].xypar[k][2] >> theCurrentTemp.entry[iy][jx].xypar[k][3] >> theCurrentTemp.entry[iy][jx].xypar[k][4];
               
               if(in_file.fail()) {
		 LOGERROR("SiPixelTemplate2D") << "Error reading file 3, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
		 delete [] theCurrentTemp.entry[0];
		 delete [] theCurrentTemp.entry;
		 return false;
	       }               
            }
            
            for (int k=0; k<2; ++k) {
               
               in_file >> theCurrentTemp.entry[iy][jx].lanpar[k][0] >> theCurrentTemp.entry[iy][jx].lanpar[k][1]
               >> theCurrentTemp.entry[iy][jx].lanpar[k][2] >> theCurrentTemp.entry[iy][jx].lanpar[k][3] >> theCurrentTemp.entry[iy][jx].lanpar[k][4];
               
               if(in_file.fail()) {
		 LOGERROR("SiPixelTemplate2D") << "Error reading file 4, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
		 delete [] theCurrentTemp.entry[0];
		 delete [] theCurrentTemp.entry;
		 return false;
	       }
               
            }
            
      
//  Read the 2D template entries as floats [they are formatted that way] and cast to short ints

            float dummy[T2YSIZE];
            for (int l=0; l<7; ++l) {
               for (int k=0; k<7; ++k) {
                  for (int j=0; j<T2XSIZE; ++j) {
                     for (int i=0; i<T2YSIZE; ++i) {
                        in_file >> dummy[i];                       
                     }                     
                     if(in_file.fail()) {
		       LOGERROR("SiPixelTemplate2D") << "Error reading file 5, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
		       delete [] theCurrentTemp.entry[0];
		       delete [] theCurrentTemp.entry;
		       return false;
		     }
                     for (int i=0; i<T2YSIZE; ++i) {
                        theCurrentTemp.entry[iy][jx].xytemp[k][l][i][j] = (short int)dummy[i];                        
                     }                    
                  }
               }
            }
            
            
            in_file >> theCurrentTemp.entry[iy][jx].chi2ppix >> theCurrentTemp.entry[iy][jx].chi2scale >> theCurrentTemp.entry[iy][jx].offsetx[0] >> theCurrentTemp.entry[iy][jx].offsetx[1]
            >> theCurrentTemp.entry[iy][jx].offsetx[2] >> theCurrentTemp.entry[iy][jx].offsetx[3]>> theCurrentTemp.entry[iy][jx].offsety[0] >> theCurrentTemp.entry[iy][jx].offsety[1]
            >> theCurrentTemp.entry[iy][jx].offsety[2] >> theCurrentTemp.entry[iy][jx].offsety[3];
            
            if(in_file.fail()) {
	      LOGERROR("SiPixelTemplate2D") << "Error reading file 6, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
	      delete [] theCurrentTemp.entry[0];
	      delete [] theCurrentTemp.entry;
	      return false;
	    }
            
            in_file >> theCurrentTemp.entry[iy][jx].clsleny >> theCurrentTemp.entry[iy][jx].clslenx >> theCurrentTemp.entry[iy][jx].mpvvav >> theCurrentTemp.entry[iy][jx].sigmavav
            >> theCurrentTemp.entry[iy][jx].kappavav >> theCurrentTemp.entry[iy][jx].scalexavg >> theCurrentTemp.entry[iy][jx].scaleyavg >> theCurrentTemp.entry[iy][jx].delyavg >> theCurrentTemp.entry[iy][jx].delysig >> theCurrentTemp.entry[iy][jx].spare[0];
            
            if(in_file.fail()) {
	      LOGERROR("SiPixelTemplate2D") << "Error reading file 7, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
	      delete [] theCurrentTemp.entry[0];
	      delete [] theCurrentTemp.entry;	      
	      return false;
	    }
            
            in_file >> theCurrentTemp.entry[iy][jx].scalex[0] >> theCurrentTemp.entry[iy][jx].scalex[1] >> theCurrentTemp.entry[iy][jx].scalex[2] >> theCurrentTemp.entry[iy][jx].scalex[3]
            >> theCurrentTemp.entry[iy][jx].scaley[0] >> theCurrentTemp.entry[iy][jx].scaley[1] >> theCurrentTemp.entry[iy][jx].scaley[2] >> theCurrentTemp.entry[iy][jx].scaley[3]
            >> theCurrentTemp.entry[iy][jx].spare[1]  >> theCurrentTemp.entry[iy][jx].spare[2];
            
            if(in_file.fail()) {
	      LOGERROR("SiPixelTemplate2D") << "Error reading file 8, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
	      delete [] theCurrentTemp.entry[0];
	      delete [] theCurrentTemp.entry;	      
	      return false;
	    }
            
         }
         
      }
      
      
      in_file.close();
      
      // Add this template to the store
      
      pixelTemp.push_back(theCurrentTemp);
      
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
bool SiPixelTemplate2D::pushfile(const SiPixel2DTemplateDBObject& dbobject, std::vector< SiPixelTemplateStore2D > & pixelTemp)
{
   // Add template stored in external dbobject to theTemplateStore
   
   const int code_version={21};
   
   // We must create a new object because dbobject must be a const and our stream must not be
   SiPixel2DTemplateDBObject db = dbobject;
   
   // Create a local template storage entry
   SiPixelTemplateStore2D theCurrentTemp;
   
   // Fill the template storage for each template calibration stored in the db
   for(int m=0; m<db.numOfTempl(); ++m)
   {
      
      // Read-in a header string first and print it
      
      SiPixel2DTemplateDBObject::char2float temp;
      for (int i=0; i<20; ++i) {
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
      
      if(db.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 0A, no template load" << ENDL; return false;}
      
      LOGINFO("SiPixelTemplate2D") << "Loading Pixel Template File - " << theCurrentTemp.head.title
      <<" code version = "<<code_version
      <<" object version "<<theCurrentTemp.head.templ_version
      << ENDL;
      
      if(theCurrentTemp.head.templ_version > 17) {
         db >> theCurrentTemp.head.ss50 >> theCurrentTemp.head.lorybias >> theCurrentTemp.head.lorxbias >> theCurrentTemp.head.fbin[0] >> theCurrentTemp.head.fbin[1] >> theCurrentTemp.head.fbin[2];
         
         if(db.fail()) {LOGERROR("SiPixelTemplate2D") << "Error reading file 0B, no template load"
            << ENDL; return false;}
      } else {
// This is for older [legacy] payloads and the numbers are indeed magic [they are part of the payload for v>17]
         theCurrentTemp.head.ss50 = theCurrentTemp.head.s50;
         theCurrentTemp.head.lorybias = theCurrentTemp.head.lorywidth/2.f;
         theCurrentTemp.head.lorxbias = theCurrentTemp.head.lorxwidth/2.f;
         theCurrentTemp.head.fbin[0] = 1.50f;
         theCurrentTemp.head.fbin[1] = 1.00f;
         theCurrentTemp.head.fbin[2] = 0.85f;
      }
      
      LOGINFO("SiPixelTemplate2D")
      << "Template ID = " << theCurrentTemp.head.ID << ", Template Version "
      << theCurrentTemp.head.templ_version << ", Bfield = "
      << theCurrentTemp.head.Bfield<< ", NTy = " << theCurrentTemp.head.NTy << ", NTyx = "
      << theCurrentTemp.head.NTyx<< ", NTxx = " << theCurrentTemp.head.NTxx << ", Dtype = "
      << theCurrentTemp.head.Dtype<< ", Bias voltage " << theCurrentTemp.head.Vbias << ", temperature "
      << theCurrentTemp.head.temperature << ", fluence " << theCurrentTemp.head.fluence
      << ", Q-scaling factor " << theCurrentTemp.head.qscale
      << ", 1/2 multi dcol threshold " << theCurrentTemp.head.s50 << ", 1/2 single dcol threshold "
      << theCurrentTemp.head.ss50<< ", y Lorentz Width " << theCurrentTemp.head.lorywidth
      << ", y Lorentz Bias " << theCurrentTemp.head.lorybias
      << ", x Lorentz width " << theCurrentTemp.head.lorxwidth
      << ", x Lorentz Bias " << theCurrentTemp.head.lorxbias
      << ", Q/Q_avg fractions for Qbin defs " << theCurrentTemp.head.fbin[0]
      << ", " << theCurrentTemp.head.fbin[1]
      << ", " << theCurrentTemp.head.fbin[2]
      << ", pixel x-size " << theCurrentTemp.head.xsize
      << ", y-size " << theCurrentTemp.head.ysize << ", zsize " << theCurrentTemp.head.zsize << ENDL;
      
      if(theCurrentTemp.head.templ_version < code_version) {
         LOGINFO("SiPixelTemplate2D") << "code expects version "<< code_version << " finds "
         << theCurrentTemp.head.templ_version <<", load anyway " << ENDL;
      }
      
      if(theCurrentTemp.head.NTy != 0) {LOGERROR("SiPixelTemplate2D") << "Trying to load 1-d template info into the 2-d template object, check your DB/global tag!" << ENDL; return false;}
      
      
      // next, layout the 2-d structure needed to store template
      
      theCurrentTemp.entry = new SiPixelTemplateEntry2D*[theCurrentTemp.head.NTyx];
      theCurrentTemp.entry[0] = new SiPixelTemplateEntry2D[theCurrentTemp.head.NTyx*theCurrentTemp.head.NTxx];
      for(int i = 1; i < theCurrentTemp.head.NTyx; ++i) theCurrentTemp.entry[i] = theCurrentTemp.entry[i-1] + theCurrentTemp.head.NTxx;
      
      // Read in the file info
      
      for (int iy=0; iy < theCurrentTemp.head.NTyx; ++iy) {
         for(int jx=0; jx < theCurrentTemp.head.NTxx; ++jx) {
            
            db >> theCurrentTemp.entry[iy][jx].runnum >> theCurrentTemp.entry[iy][jx].costrk[0]
            >> theCurrentTemp.entry[iy][jx].costrk[1] >> theCurrentTemp.entry[iy][jx].costrk[2];
            
            if(db.fail()) {
	      LOGERROR("SiPixelTemplate2D") << "Error reading file 1, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
	      delete [] theCurrentTemp.entry[0];
	      delete [] theCurrentTemp.entry;	      
	      return false;
	    }
            
            // Calculate cot(alpha) and cot(beta) for this entry
            
            theCurrentTemp.entry[iy][jx].cotalpha = theCurrentTemp.entry[iy][jx].costrk[0]/theCurrentTemp.entry[iy][jx].costrk[2];
            
            theCurrentTemp.entry[iy][jx].cotbeta = theCurrentTemp.entry[iy][jx].costrk[1]/theCurrentTemp.entry[iy][jx].costrk[2];
            
            db >> theCurrentTemp.entry[iy][jx].qavg >> theCurrentTemp.entry[iy][jx].pixmax >> theCurrentTemp.entry[iy][jx].sxymax >> theCurrentTemp.entry[iy][jx].iymin
            >> theCurrentTemp.entry[iy][jx].iymax >> theCurrentTemp.entry[iy][jx].jxmin >> theCurrentTemp.entry[iy][jx].jxmax;
            
            if(db.fail()) {
	      LOGERROR("SiPixelTemplate2D") << "Error reading file 2, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
	      delete [] theCurrentTemp.entry[0];
	      delete [] theCurrentTemp.entry;	      
	      return false;
	    }
            
            for (int k=0; k<2; ++k) {
               
               db >> theCurrentTemp.entry[iy][jx].xypar[k][0] >> theCurrentTemp.entry[iy][jx].xypar[k][1]
               >> theCurrentTemp.entry[iy][jx].xypar[k][2] >> theCurrentTemp.entry[iy][jx].xypar[k][3] >> theCurrentTemp.entry[iy][jx].xypar[k][4];
               
               if(db.fail()) {
		 LOGERROR("SiPixelTemplate2D") << "Error reading file 3, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
		 delete [] theCurrentTemp.entry[0];
		 delete [] theCurrentTemp.entry;	      
		 return false;
	       }
               
            }
            
            for (int k=0; k<2; ++k) {
               
               db >> theCurrentTemp.entry[iy][jx].lanpar[k][0] >> theCurrentTemp.entry[iy][jx].lanpar[k][1]
               >> theCurrentTemp.entry[iy][jx].lanpar[k][2] >> theCurrentTemp.entry[iy][jx].lanpar[k][3] >> theCurrentTemp.entry[iy][jx].lanpar[k][4];
               
               if(db.fail()) {
		 LOGERROR("SiPixelTemplate2D") << "Error reading file 4, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
		 delete [] theCurrentTemp.entry[0];
		 delete [] theCurrentTemp.entry;	      
		 return false;
	       }
               
            }
            
//  Read the 2D template entries as floats [they are formatted that way] and cast to short ints

            float dummy[T2YSIZE];
            for (int l=0; l<7; ++l) {
               for (int k=0; k<7; ++k) {
                  for (int j=0; j<T2XSIZE; ++j) {
                     for (int i=0; i<T2YSIZE; ++i) {
                        db >> dummy[i];                       
                     }                     
                     if(db.fail()) {
		       LOGERROR("SiPixelTemplate2D") << "Error reading file 5, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
		       delete [] theCurrentTemp.entry[0];
		       delete [] theCurrentTemp.entry;	      
		       return false;
		     }
                     for (int i=0; i<T2YSIZE; ++i) {
                        theCurrentTemp.entry[iy][jx].xytemp[k][l][i][j] = (short int)dummy[i];                        
                     }                    
                  }
               }
            }
            
            db >> theCurrentTemp.entry[iy][jx].chi2ppix >> theCurrentTemp.entry[iy][jx].chi2scale >> theCurrentTemp.entry[iy][jx].offsetx[0] >> theCurrentTemp.entry[iy][jx].offsetx[1]
            >> theCurrentTemp.entry[iy][jx].offsetx[2] >> theCurrentTemp.entry[iy][jx].offsetx[3]>> theCurrentTemp.entry[iy][jx].offsety[0] >> theCurrentTemp.entry[iy][jx].offsety[1]
            >> theCurrentTemp.entry[iy][jx].offsety[2] >> theCurrentTemp.entry[iy][jx].offsety[3];
            
            if(db.fail()) {
	      LOGERROR("SiPixelTemplate2D") << "Error reading file 6, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
	      delete [] theCurrentTemp.entry[0];
	      delete [] theCurrentTemp.entry;	      
	      return false;
	    }
            
            db >> theCurrentTemp.entry[iy][jx].clsleny >> theCurrentTemp.entry[iy][jx].clslenx >> theCurrentTemp.entry[iy][jx].mpvvav >> theCurrentTemp.entry[iy][jx].sigmavav
            >> theCurrentTemp.entry[iy][jx].kappavav >> theCurrentTemp.entry[iy][jx].scalexavg >> theCurrentTemp.entry[iy][jx].scaleyavg >> theCurrentTemp.entry[iy][jx].delyavg >> theCurrentTemp.entry[iy][jx].delysig >> theCurrentTemp.entry[iy][jx].spare[0];
            
            if(db.fail()) {
	      LOGERROR("SiPixelTemplate2D") << "Error reading file 7, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
	      delete [] theCurrentTemp.entry[0];
	      delete [] theCurrentTemp.entry;	      
	      return false;
	    }
            
            db >> theCurrentTemp.entry[iy][jx].scalex[0] >> theCurrentTemp.entry[iy][jx].scalex[1] >> theCurrentTemp.entry[iy][jx].scalex[2] >> theCurrentTemp.entry[iy][jx].scalex[3]
            >> theCurrentTemp.entry[iy][jx].scaley[0] >> theCurrentTemp.entry[iy][jx].scaley[1] >> theCurrentTemp.entry[iy][jx].scaley[2] >> theCurrentTemp.entry[iy][jx].scaley[3]
            >> theCurrentTemp.entry[iy][jx].spare[1]  >> theCurrentTemp.entry[iy][jx].spare[2];
            
            if(db.fail()) {
	      LOGERROR("SiPixelTemplate2D") << "Error reading file 8, no template load, run # " << theCurrentTemp.entry[iy][jx].runnum << ENDL; 
	      delete [] theCurrentTemp.entry[0];
	      delete [] theCurrentTemp.entry;	      
	      return false;
	    }
            
         }
      }
         
// Add this template to the store
   
   pixelTemp.push_back(theCurrentTemp);    
     
   }
   
   
   return true;
   
} // TempInit

#endif


bool SiPixelTemplate2D::getid(int id)
{
   if(id != id_current_) {
      
      // Find the index corresponding to id
      
      index_id_ = -1;
      for(int i=0; i<(int)thePixelTemp_.size(); ++i) {
         
         if(id == thePixelTemp_[i].head.ID) {
            
            index_id_ = i;
            id_current_ = id;
            
            // Copy the charge scaling factor to the private variable
            
            Dtype_ = thePixelTemp_[index_id_].head.Dtype;
            
            // Copy the charge scaling factor to the private variable
            
            qscale_ = thePixelTemp_[index_id_].head.qscale;
            
            // Copy the pseudopixel signal size to the private variable
            
            s50_ = thePixelTemp_[index_id_].head.s50;
            
            // Copy Qbinning info to private variables
            
            for(int j=0; j<3; ++j) {fbin_[j] = thePixelTemp_[index_id_].head.fbin[j];}
            
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
            int imidx = Nxx_/2;
            
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

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
   if(index_id_ < 0 || index_id_ >= (int)thePixelTemp_.size()) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate2D::interpolate can't find needed template ID = " << id
      << ", Are you using the correct global tag?" << std::endl;
   }
#else
   assert(index_id_ >= 0 && index_id_ < (int)thePixelTemp_.size());
#endif
   return true;
}

// *************************************************************************************************************************************
//! Interpolate stored 2-D information for input angles
//! \param         id - (input) the id of the template
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014)
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)
//! \param locBz - (input) the sign of this quantity is used to determine whether to flip cot(beta)<0 quantities from cot(beta)>0 (FPix only)
//!                    for Phase 0 FPix IP-related tracks, locBz < 0 for cot(beta) > 0 and locBz > 0 for cot(beta) < 0
//!                    for Phase 1 FPix IP-related tracks, see next comment
//! \param locBx - (input) the sign of this quantity is used to determine whether to flip cot(alpha/beta)<0 quantities from cot(alpha/beta)>0 (FPix only)
//!                    for Phase 1 FPix IP-related tracks, locBx/locBz > 0 for cot(alpha) > 0 and locBx/locBz < 0 for cot(alpha) < 0
//!                    for Phase 1 FPix IP-related tracks, locBx > 0 for cot(beta) > 0 and locBx < 0 for cot(beta) < 0
// *************************************************************************************************************************************

bool SiPixelTemplate2D::interpolate(int id, float cotalpha, float cotbeta, float locBz, float locBx)
{
   
   
   // Interpolate for a new set of track angles
   
   // Local variables

   float acotb, dcota, dcotb;
   
   // Check to see if interpolation is valid
   
   if(id != id_current_ || cotalpha != cota_current_ || cotbeta != cotb_current_) {
      cota_current_ = cotalpha; cotb_current_ = cotbeta;
      success_ = getid(id);
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
   
   float cota = cotalpha;
   flip_x_ = false;
   flip_y_ = false;
   switch(Dtype_) {
      case 0:
         if(cotbeta < 0.f) {flip_y_ = true;}
         break;
      case 1:
         if(locBz > 0.f) {flip_y_ = true;}
         break;
      case 2:
      case 3:
      case 4:
      case 5:
         if(locBx*locBz < 0.f) {
            cota = fabs(cotalpha);
            flip_x_ = true;
         }
         if(locBx < 0.f) {flip_y_ = true;}
         break;
      default:
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
         throw cms::Exception("DataCorrupt") << "SiPixelTemplate2D::illegal subdetector ID = " << thePixelTemp_[index_id_].head.Dtype << std::endl;
#else
         std::cout << "SiPixelTemplate:2D:illegal subdetector ID = " << thePixelTemp_[index_id_].head.Dtype << std::endl;
#endif
      }
   
   if(cota < cotalpha0_) {
      success_ = false;
      jx0_ = 0;
      jx1_ = 1;
      adcota_ = 0.f;
   } else if(cota > cotalpha1_) {
      success_ = false;
      jx0_ = Nxx_ - 1;
      jx1_ = jx0_ - 1;
      adcota_ = 0.f;
   } else {
      jx0_ = (int)((cota-cotalpha0_)/deltacota_+0.5f);
      dcota = (cota - (cotalpha0_ + jx0_*deltacota_))/deltacota_;
      adcota_ = fabs(dcota);
      if(dcota > 0.f) {jx1_ = jx0_ + 1;if(jx1_ > Nxx_-1) jx1_ = jx0_-1;} else {jx1_ = jx0_ - 1; if(jx1_ < 0) jx1_ = jx0_+1;}
   }
   
   // Interpolate the absolute value of cot(beta)
   
   acotb = fabs(cotbeta);
   
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
   
   //  Calculate signed quantities
   
   lorydrift_ = lorywidth_/2.;
   if(flip_y_) lorydrift_ = -lorydrift_;
   lorxdrift_ = lorxwidth_/2.;
   if(flip_x_) lorxdrift_ = -lorxdrift_;
   
   // Use pointers to the three angle pairs used in the interpolation
   
   
   entry00_ = &thePixelTemp_[index_id_].entry[iy0_][jx0_];
   entry10_ = &thePixelTemp_[index_id_].entry[iy1_][jx0_];
   entry01_ = &thePixelTemp_[index_id_].entry[iy0_][jx1_];
   
   // Interpolate things in cot(alpha)-cot(beta)
   
   qavg_ = entry00_->qavg
   +adcota_*(entry01_->qavg - entry00_->qavg)
   +adcotb_*(entry10_->qavg - entry00_->qavg);
   
   pixmax_ = entry00_->pixmax
   +adcota_*(entry01_->pixmax - entry00_->pixmax)
   +adcotb_*(entry10_->pixmax - entry00_->pixmax);
   
   sxymax_ = entry00_->sxymax
   +adcota_*(entry01_->sxymax - entry00_->sxymax)
   +adcotb_*(entry10_->sxymax - entry00_->sxymax);
   
   chi2avgone_ = entry00_->chi2avgone
   +adcota_*(entry01_->chi2avgone - entry00_->chi2avgone)
   +adcotb_*(entry10_->chi2avgone - entry00_->chi2avgone);
   
   chi2minone_ = entry00_->chi2minone
   +adcota_*(entry01_->chi2minone - entry00_->chi2minone)
   +adcotb_*(entry10_->chi2minone - entry00_->chi2minone);
   
   clsleny_ = entry00_->clsleny
   +adcota_*(entry01_->clsleny - entry00_->clsleny)
   +adcotb_*(entry10_->clsleny - entry00_->clsleny);
   
   clslenx_ = entry00_->clslenx
   +adcota_*(entry01_->clslenx - entry00_->clslenx)
   +adcotb_*(entry10_->clslenx - entry00_->clslenx);
   
   
   chi2ppix_ = entry00_->chi2ppix
   +adcota_*(entry01_->chi2ppix - entry00_->chi2ppix)
   +adcotb_*(entry10_->chi2ppix - entry00_->chi2ppix);
   
   chi2scale_ = entry00_->chi2scale
   +adcota_*(entry01_->chi2scale - entry00_->chi2scale)
   +adcotb_*(entry10_->chi2scale - entry00_->chi2scale);
   
   scaleyavg_ = entry00_->scaleyavg
   +adcota_*(entry01_->scaleyavg - entry00_->scaleyavg)
   +adcotb_*(entry10_->scaleyavg - entry00_->scaleyavg);
   
   scalexavg_ = entry00_->scalexavg
   +adcota_*(entry01_->scalexavg - entry00_->scalexavg)
   +adcotb_*(entry10_->scalexavg - entry00_->scalexavg);
   
   delyavg_ = entry00_->delyavg
   +adcota_*(entry01_->delyavg - entry00_->delyavg)
   +adcotb_*(entry10_->delyavg - entry00_->delyavg);
   
   delysig_ = entry00_->delysig
   +adcota_*(entry01_->delysig - entry00_->delysig)
   +adcotb_*(entry10_->delysig - entry00_->delysig);
   
   mpvvav_ = entry00_->mpvvav
   +adcota_*(entry01_->mpvvav - entry00_->mpvvav)
   +adcotb_*(entry10_->mpvvav - entry00_->mpvvav);
   
   sigmavav_ = entry00_->sigmavav
   +adcota_*(entry01_->sigmavav - entry00_->sigmavav)
   +adcotb_*(entry10_->sigmavav - entry00_->sigmavav);
   
   kappavav_ = entry00_->kappavav
   +adcota_*(entry01_->kappavav - entry00_->kappavav)
   +adcotb_*(entry10_->kappavav - entry00_->kappavav);
   
   for(int i=0; i<4 ; ++i) {
      scalex_[i] = entry00_->scalex[i]
      +adcota_*(entry01_->scalex[i] - entry00_->scalex[i])
      +adcotb_*(entry10_->scalex[i] - entry00_->scalex[i]);
      
      scaley_[i] = entry00_->scaley[i]
      +adcota_*(entry01_->scaley[i] - entry00_->scaley[i])
      +adcotb_*(entry10_->scaley[i] - entry00_->scaley[i]);
      
      offsetx_[i] = entry00_->offsetx[i]
      +adcota_*(entry01_->offsetx[i] - entry00_->offsetx[i])
      +adcotb_*(entry10_->offsetx[i] - entry00_->offsetx[i]);
      if(flip_x_) offsetx_[i] = -offsetx_[i];
      
      offsety_[i] = entry00_->offsety[i]
      +adcota_*(entry01_->offsety[i] - entry00_->offsety[i])
      +adcotb_*(entry10_->offsety[i] - entry00_->offsety[i]);
      if(flip_y_) offsety_[i] = -offsety_[i];
   }
   
   for(int i=0; i<2 ; ++i) {
      for(int j=0; j<5 ; ++j) {
         // Charge loss switches sides when cot(beta) changes sign
         if(flip_y_) {
            xypary0x0_[1-i][j] = (float)entry00_->xypar[i][j];
            xypary1x0_[1-i][j] = (float)entry10_->xypar[i][j];
            xypary0x1_[1-i][j] = (float)entry01_->xypar[i][j];
            lanpar_[1-i][j] = entry00_->lanpar[i][j]
            +adcota_*(entry01_->lanpar[i][j] - entry00_->lanpar[i][j])
            +adcotb_*(entry10_->lanpar[i][j] - entry00_->lanpar[i][j]);
         } else {
            xypary0x0_[i][j] = (float)entry00_->xypar[i][j];
            xypary1x0_[i][j] = (float)entry10_->xypar[i][j];
            xypary0x1_[i][j] = (float)entry01_->xypar[i][j];
            lanpar_[i][j] = entry00_->lanpar[i][j]
            +adcota_*(entry01_->lanpar[i][j] - entry00_->lanpar[i][j])
            +adcotb_*(entry10_->lanpar[i][j] - entry00_->lanpar[i][j]);
         }
      }
   }
   
   return success_;
} // interpolate


// *************************************************************************************************************************************
//! Load template info for single angle point to invoke template reco for template generation
//! \param      entry - (input) pointer to template entry
//! \param      sizex - (input) pixel x-size
//! \param      sizey - (input) pixel y-size
//! \param      sizez - (input) pixel z-size
// *************************************************************************************************************************************

void SiPixelTemplate2D::sideload(SiPixelTemplateEntry2D* entry, int iDtype, float locBx, float locBz, float lorwdy, float lorwdx, float q50, float fbin[3], float xsize, float ysize, float zsize)
{
   // Set class variables to the input parameters
   
   entry00_ = entry;
   entry01_ = entry;
   entry10_ = entry;
   Dtype_ = iDtype;
   lorywidth_ = lorwdy;
   lorxwidth_ = lorwdx;
   xsize_ = xsize;
   ysize_ = ysize;
   zsize_ = zsize;
   s50_ = q50;
   qscale_ = 1.f;
   for(int i=0; i<3; ++i) {fbin_[i] = fbin[i];}
   
   // Set other class variables
   
   adcota_ = 0.f;
   adcotb_ = 0.f;
   
   // Interpolate things in cot(alpha)-cot(beta)
   
   qavg_ = entry00_->qavg;
   
   pixmax_ = entry00_->pixmax;
   
   sxymax_ = entry00_->sxymax;
   
   clsleny_ = entry00_->clsleny;
   
   clslenx_ = entry00_->clslenx;
   
   scaleyavg_ = 1.f;
   
   scalexavg_ = 1.f;
   
   delyavg_ = 0.f;
   
   delysig_ = 0.f;
   
   for(int i=0; i<4 ; ++i) {
      scalex_[i] = 1.f;
      scaley_[i] = 1.f;
      offsetx_[i] = 0.f;
      offsety_[i] = 0.f;
   }
   
   // This works only for IP-related tracks
   
   flip_x_ = false;
   flip_y_ = false;
   float cotbeta = entry00_->cotbeta;
   switch(Dtype_) {
      case 0:
         if(cotbeta < 0.f) {flip_y_ = true;}
         break;
      case 1:
         if(locBz > 0.f) {
            flip_y_ = true;
         }
         break;
      case 2:
      case 3:
      case 4:
      case 5:
         if(locBx*locBz < 0.f) {
            flip_x_ = true;
         }
         if(locBx < 0.f) {
            flip_y_ = true;
         }
         break;
      default:
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
         throw cms::Exception("DataCorrupt") << "SiPixelTemplate2D::illegal subdetector ID = " << iDtype << std::endl;
#else
         std::cout << "SiPixelTemplate:2D:illegal subdetector ID = " << iDtype << std::endl;
#endif
   }
   
   //  Calculate signed quantities
   
   lorydrift_ = lorywidth_/2.;
   if(flip_y_) lorydrift_ = -lorydrift_;
   lorxdrift_ = lorxwidth_/2.;
   if(flip_x_) lorxdrift_ = -lorxdrift_;
   
   for(int i=0; i<2 ; ++i) {
      for(int j=0; j<5 ; ++j) {
         // Charge loss switches sides when cot(beta) changes sign
         if(flip_y_) {
            xypary0x0_[1-i][j] = (float)entry00_->xypar[i][j];
            xypary1x0_[1-i][j] = (float)entry00_->xypar[i][j];
            xypary0x1_[1-i][j] = (float)entry00_->xypar[i][j];
            lanpar_[1-i][j] = entry00_->lanpar[i][j];
         } else {
            xypary0x0_[i][j] = (float)entry00_->xypar[i][j];
            xypary1x0_[i][j] = (float)entry00_->xypar[i][j];
            xypary0x1_[i][j] = (float)entry00_->xypar[i][j];
            lanpar_[i][j] = entry00_->lanpar[i][j];
         }
      }
   }
   return;
}


// *************************************************************************************************************************************
//! \param       xhit - (input) x-position of hit relative to the lower left corner of pixel[1][1] (to allow for the "padding" of the two-d clusters in the splitter)
//! \param       yhit - (input) y-position of hit relative to the lower left corner of pixel[1][1]
//! \param    ydouble - (input) STL vector of 21 element array to flag a double-pixel starting at cluster[1][1]
//! \param    xdouble - (input) STL vector of 11 element array to flag a double-pixel starting at cluster[1][1]
//! \param template2d - (output) 2d template of size matched to the cluster.  Input must be zeroed since charge is added only.
// *************************************************************************************************************************************

bool SiPixelTemplate2D::xytemp(float xhit, float yhit, bool ydouble[BYM2], bool xdouble[BXM2], float template2d[BXM2][BYM2], bool derivatives, float dpdx2d[2][BXM2][BYM2], float& QTemplate)
{
   // Interpolate for a new set of track angles
   
   // Local variables
   int pixx, pixy, k0, k1, l0, l1, deltax, deltay, iflipy, jflipx, imin, imax, jmin, jmax;
   int m, n;
   float dx, dy, ddx, ddy, adx, ady;
   //   const float deltaxy[2] = {8.33f, 12.5f};
   const float deltaxy[2] = {16.67f, 25.0f};
   
   // Check to see if interpolation is valid
   
   
   // next, determine the indices of the closest point in k (y-displacement), l (x-displacement)
   // pixy and pixx are the indices of the struck pixel in the (Ty,Tx) system
   // k0,k1 are the k-indices of the closest and next closest point
   // l0,l1 are the l-indices of the closest and next closest point
   
   pixy = (int)floorf(yhit/ysize_);
   dy = yhit-(pixy+0.5f)*ysize_;
   if(flip_y_) {dy = -dy;}
   k0 = (int)(dy/ysize_*6.f+3.5f);
   if(k0 < 0) k0 = 0;
   if(k0 > 6) k0 = 6;
   ddy = 6.f*dy/ysize_ - (k0-3);
   ady = fabs(ddy);
   if(ddy > 0.f) {k1 = k0 + 1; if(k1 > 6) k1 = k0-1;} else {k1 = k0 - 1; if(k1 < 0) k1 = k0+1;}
   pixx = (int)floorf(xhit/xsize_);
   dx = xhit-(pixx+0.5f)*xsize_;
   if(flip_x_) {dx = -dx;}
   l0 = (int)(dx/xsize_*6.f+3.5f);
   if(l0 < 0) l0 = 0;
   if(l0 > 6) l0 = 6;
   ddx = 6.f*dx/xsize_ - (l0-3);
   adx = fabs(ddx);
   if(ddx > 0.f) {l1 = l0 + 1; if(l1 > 6) l1 = l0-1;} else {l1 = l0 - 1; if(l1 < 0) l1 = l0+1;}
   
   // OK, lets do the template interpolation.
   
   // First find the limits of the indices for non-zero pixels
   
   imin = std::min(entry00_->iymin,entry10_->iymin);
   imin_ = std::min(imin,entry01_->iymin);
   
   jmin = std::min(entry00_->jxmin,entry10_->jxmin);
   jmin_ = std::min(jmin,entry01_->jxmin);
   
   imax = std::max(entry00_->iymax,entry10_->iymax);
   imax_ = std::max(imax,entry01_->iymax);
   
   jmax = std::max(entry00_->jxmax,entry10_->jxmax);
   jmax_ = std::max(jmax,entry01_->jxmax);
   
   // Calculate the x and y offsets to make the new template
   
   // First, shift the struck pixel coordinates to the (Ty+2, Tx+2) system
   
   ++pixy; ++pixx;
   
   // In the template store, the struck pixel is always (THy,THx)
   
   deltax = pixx - T2HX;
   deltay = pixy - T2HY;
   
   //  First zero the local 2-d template
   
   for(int j=0; j<BXM2; ++j) {for(int i=0; i<BYM2; ++i) {xytemp_[j][i] = 0.f;}}
   
   // Loop over the non-zero part of the template index space and interpolate
   
   for(int j=jmin_; j<=jmax_; ++j) {
      // Flip indices as needed
      if(flip_x_) {jflipx=T2XSIZE-1-j; m = deltax+jflipx;} else {m = deltax+j;}
      for(int i=imin_; i<=imax_; ++i) {
         if(flip_y_) {iflipy=T2YSIZE-1-i; n = deltay+iflipy;} else {n = deltay+i;}
         if(m>=0 && m<=BXM3 && n>=0 && n<=BYM3) {
            xytemp_[m][n] = (float)entry00_->xytemp[k0][l0][i][j]
            + adx*(float)(entry00_->xytemp[k0][l1][i][j] - entry00_->xytemp[k0][l0][i][j])
            + ady*(float)(entry00_->xytemp[k1][l0][i][j] - entry00_->xytemp[k0][l0][i][j])
            + adcota_*(float)(entry01_->xytemp[k0][l0][i][j] - entry00_->xytemp[k0][l0][i][j])
            + adcotb_*(float)(entry10_->xytemp[k0][l0][i][j] - entry00_->xytemp[k0][l0][i][j]);
         }
      }
   }
   
   //combine rows and columns to simulate double pixels
   
   for(int n=1; n<BYM3; ++n) {
      if(ydouble[n]) {
         //  Combine the y-columns
         for(int m=1; m<BXM3; ++m) {
            xytemp_[m][n] += xytemp_[m][n+1];
         }
         //  Now shift the remaining pixels over by one column
         for(int i=n+1; i<BYM3; ++i) {
            for(int m=1; m<BXM3; ++m) {
               xytemp_[m][i] = xytemp_[m][i+1];
            }
         }
      }
   }
   
   //combine rows and columns to simulate double pixels
   
   for(int m=1; m<BXM3; ++m) {
      if(xdouble[m]) {
         //  Combine the x-rows
         for(int n=1; n<BYM3; ++n) {
            xytemp_[m][n] += xytemp_[m+1][n];
         }
         //  Now shift the remaining pixels over by one row
         for(int j=m+1; j<BXM3; ++j) {
            for(n=1; n<BYM3; ++n) {
               xytemp_[j][n] = xytemp_[j+1][n];
            }
         }
      }
   }
   
   //  Finally, loop through and increment the external template
   
   float qtemptot = 0.f;
   
   for(int n=1; n<BYM3; ++n) {
      for(int m=1; m<BXM3; ++m) {
         if(xytemp_[m][n] != 0.f) {template2d[m][n] += xytemp_[m][n]; qtemptot += xytemp_[m][n];}
      }
   }
   
   QTemplate = qtemptot;
   
   if(derivatives) {
      
      float dxytempdx[2][BXM2][BYM2], dxytempdy[2][BXM2][BYM2];
      
      for(int k = 0; k<2; ++k) {
         for(int i = 0; i<BXM2; ++i) {
            for(int j = 0; j<BYM2; ++j) {
               dxytempdx[k][i][j] = 0.f;
               dxytempdy[k][i][j] = 0.f;
               dpdx2d[k][i][j] = 0.f;
            }
         }
      }
      
      // First do shifted +x template
      
      pixx = (int)floorf((xhit+deltaxy[0])/xsize_);
      dx = (xhit+deltaxy[0])-(pixx+0.5f)*xsize_;
      if(flip_x_) {dx = -dx;}
      l0 = (int)(dx/xsize_*6.f+3.5f);
      if(l0 < 0) l0 = 0;
      if(l0 > 6) l0 = 6;
      ddx = 6.f*dx/xsize_ - (l0-3);
      adx = fabs(ddx);
      if(ddx > 0.f) {l1 = l0 + 1; if(l1 > 6) l1 = l0-1;} else {l1 = l0 - 1; if(l1 < 0) l1 = l0+1;}
      
      // OK, lets do the template interpolation.
      
      // Calculate the x and y offsets to make the new template
      
      // First, shift the struck pixel coordinates to the (Ty+2, Tx+2) system
      
      ++pixx;
      
      // In the template store, the struck pixel is always (THy,THx)
      
      deltax = pixx - T2HX;
      
      // Loop over the non-zero part of the template index space and interpolate
      
      for(int j=jmin_; j<=jmax_; ++j) {
         // Flip indices as needed
         if(flip_x_) {jflipx=T2XSIZE-1-j; m = deltax+jflipx;} else {m = deltax+j;}
         for(int i=imin_; i<=imax_; ++i) {
            if(flip_y_) {iflipy=T2YSIZE-1-i; n = deltay+iflipy;} else {n = deltay+i;}
            if(m>=0 && m<=BXM3 && n>=0 && n<=BYM3) {
               dxytempdx[1][m][n] = (float)entry00_->xytemp[k0][l0][i][j]
               + adx*(float)(entry00_->xytemp[k0][l1][i][j] -   entry00_->xytemp[k0][l0][i][j])
               + ady*(float)(entry00_->xytemp[k1][l0][i][j] - entry00_->xytemp[k0][l0][i][j])
               + adcota_*(float)(entry01_->xytemp[k0][l0][i][j] - entry00_->xytemp[k0][l0][i][j])
               + adcotb_*(float)(entry10_->xytemp[k0][l0][i][j] - entry00_->xytemp[k0][l0][i][j]);
            }
         }
      }
      
      //combine rows and columns to simulate double pixels
      
      for(int n=1; n<BYM3; ++n) {
         if(ydouble[n]) {
            //  Combine the y-columns
            for(int m=1; m<BXM3; ++m) {
               dxytempdx[1][m][n] += dxytempdx[1][m][n+1];
            }
            //  Now shift the remaining pixels over by one column
            for(int i=n+1; i<BYM3; ++i) {
               for(int m=1; m<BXM3; ++m) {
                  dxytempdx[1][m][i] = dxytempdx[1][m][i+1];
               }
            }
         }
      }
      
      //combine rows and columns to simulate double pixels
      
      for(int m=1; m<BXM3; ++m) {
         if(xdouble[m]) {
            //  Combine the x-rows
            for(int n=1; n<BYM3; ++n) {
               dxytempdx[1][m][n] += dxytempdx[1][m+1][n];
            }
            //  Now shift the remaining pixels over by one row
            for(int j=m+1; j<BXM3; ++j) {
               for(int n=1; n<BYM3; ++n) {
                  dxytempdx[1][j][n] = dxytempdx[1][j+1][n];
               }
            }
         }
      }
      
      // Next do shifted -x template
      
      pixx = (int)floorf((xhit-deltaxy[0])/xsize_);
      dx = (xhit-deltaxy[0])-(pixx+0.5f)*xsize_;
      if(flip_x_) {dx = -dx;}
      l0 = (int)(dx/xsize_*6.f+3.5f);
      if(l0 < 0) l0 = 0;
      if(l0 > 6) l0 = 6;
      ddx = 6.f*dx/xsize_ - (l0-3);
      adx = fabs(ddx);
      if(ddx > 0.f) {l1 = l0 + 1; if(l1 > 6) l1 = l0-1;} else {l1 = l0 - 1; if(l1 < 0) l1 = l0+1;}
      
      // OK, lets do the template interpolation.
      
      // Calculate the x and y offsets to make the new template
      
      // First, shift the struck pixel coordinates to the (Ty+2, Tx+2) system
      
      ++pixx;
      
      // In the template store, the struck pixel is always (THy,THx)
      
      deltax = pixx - T2HX;
      
      // Loop over the non-zero part of the template index space and interpolate
      
      for(int j=jmin_; j<=jmax_; ++j) {
         // Flip indices as needed
         if(flip_x_) {jflipx=T2XSIZE-1-j; m = deltax+jflipx;} else {m = deltax+j;}
         for(int i=imin_; i<=imax_; ++i) {
            if(flip_y_) {iflipy=T2YSIZE-1-i; n = deltay+iflipy;} else {n = deltay+i;}
            if(m>=0 && m<=BXM3 && n>=0 && n<=BYM3) {
               dxytempdx[0][m][n]  = (float)entry00_->xytemp[k0][l0][i][j]
               + adx*(float)(entry00_->xytemp[k0][l1][i][j] -   entry00_->xytemp[k0][l0][i][j])
               + ady*(float)(entry00_->xytemp[k1][l0][i][j] - entry00_->xytemp[k0][l0][i][j])
               + adcota_*(float)(entry01_->xytemp[k0][l0][i][j] - entry00_->xytemp[k0][l0][i][j])
               + adcotb_*(float)(entry10_->xytemp[k0][l0][i][j] - entry00_->xytemp[k0][l0][i][j]);
            }
         }
      }
      
      //combine rows and columns to simulate double pixels
      
      for(int n=1; n<BYM3; ++n) {
         if(ydouble[n]) {
            //  Combine the y-columns
            for(int m=1; m<BXM3; ++m) {
               dxytempdx[0][m][n] += dxytempdx[0][m][n+1];
            }
            //  Now shift the remaining pixels over by one column
            for(int i=n+1; i<BYM3; ++i) {
               for(int m=1; m<BXM3; ++m) {
                  dxytempdx[0][m][i] = dxytempdx[0][m][i+1];
               }
            }
         }
      }
      
      //combine rows and columns to simulate double pixels
      
      for(int m=1; m<BXM3; ++m) {
         if(xdouble[m]) {
            //  Combine the x-rows
            for(int n=1; n<BYM3; ++n) {
               dxytempdx[0][m][n] += dxytempdx[0][m+1][n];
            }
            //  Now shift the remaining pixels over by one row
            for(int j=m+1; j<BXM3; ++j) {
               for(int n=1; n<BYM3; ++n) {
                  dxytempdx[0][j][n] = dxytempdx[0][j+1][n];
               }
            }
         }
      }
      
      
      //  Finally, normalize the derivatives and copy the results to the output array
      
      for(int n=1; n<BYM3; ++n) {
         for(int m=1; m<BXM3; ++m) {
            dpdx2d[0][m][n] = (dxytempdx[1][m][n] - dxytempdx[0][m][n])/(2.*deltaxy[0]);
         }
      }
      
      // Next, do shifted y template
      
      pixy = (int)floorf((yhit+deltaxy[1])/ysize_);
      dy = (yhit+deltaxy[1])-(pixy+0.5f)*ysize_;
      if(flip_y_) {dy = -dy;}
      k0 = (int)(dy/ysize_*6.f+3.5f);
      if(k0 < 0) k0 = 0;
      if(k0 > 6) k0 = 6;
      ddy = 6.f*dy/ysize_ - (k0-3);
      ady = fabs(ddy);
      if(ddy > 0.f) {k1 = k0 + 1; if(k1 > 6) k1 = k0-1;} else {k1 = k0 - 1; if(k1 < 0) k1 = k0+1;}
      pixx = (int)floorf(xhit/xsize_);
      dx = xhit-(pixx+0.5f)*xsize_;
      if(flip_x_) {dx = -dx;}
      l0 = (int)(dx/xsize_*6.f+3.5f);
      if(l0 < 0) l0 = 0;
      if(l0 > 6) l0 = 6;
      ddx = 6.f*dx/xsize_ - (l0-3);
      adx = fabs(ddx);
      if(ddx > 0.f) {l1 = l0 + 1; if(l1 > 6) l1 = l0-1;} else {l1 = l0 - 1; if(l1 < 0) l1 = l0+1;}
      
      // OK, lets do the template interpolation.
      
      // Calculate the x and y offsets to make the new template
      
      // First, shift the struck pixel coordinates to the (Ty+2, Tx+2) system
      
      ++pixy; ++pixx;
      
      // In the template store, the struck pixel is always (THy,THx)
      
      deltax = pixx - T2HX;
      deltay = pixy - T2HY;
      
      // Loop over the non-zero part of the template index space and interpolate
      
      for(int j=jmin_; j<=jmax_; ++j) {
         // Flip indices as needed
         if(flip_x_) {jflipx=T2XSIZE-1-j; m = deltax+jflipx;} else {m = deltax+j;}
         for(int i=imin_; i<=imax_; ++i) {
            if(flip_y_) {iflipy=T2YSIZE-1-i; n = deltay+iflipy;} else {n = deltay+i;}
            if(m>=0 && m<=BXM3 && n>=0 && n<=BYM3) {
               dxytempdy[1][m][n] = (float)entry00_->xytemp[k0][l0][i][j]
               + adx*(float)(entry00_->xytemp[k0][l1][i][j] - entry00_->xytemp[k0][l0][i][j])
               + ady*(float)(entry00_->xytemp[k1][l0][i][j] - entry00_->xytemp[k0][l0][i][j])
               + adcota_*(float)(entry01_->xytemp[k0][l0][i][j] - entry00_->xytemp[k0][l0][i][j])
               + adcotb_*(float)(entry10_->xytemp[k0][l0][i][j] - entry00_->xytemp[k0][l0][i][j]);
            }
         }
      }
      
      //combine rows and columns to simulate double pixels
      
      for(int n=1; n<BYM3; ++n) {
         if(ydouble[n]) {
            //  Combine the y-columns
            for(int m=1; m<BXM3; ++m) {
               dxytempdy[1][m][n] += dxytempdy[1][m][n+1];
            }
            //  Now shift the remaining pixels over by one column
            for(int i=n+1; i<BYM3; ++i) {
               for(int m=1; m<BXM3; ++m) {
                  dxytempdy[1][m][i] = dxytempdy[1][m][i+1];
               }
            }
         }
      }
      
      //combine rows and columns to simulate double pixels
      
      for(int m=1; m<BXM3; ++m) {
         if(xdouble[m]) {
            //  Combine the x-rows
            for(int n=1; n<BYM3; ++n) {
               dxytempdy[1][m][n] += dxytempdy[1][m+1][n];
            }
            //  Now shift the remaining pixels over by one row
            for(int j=m+1; j<BXM3; ++j) {
               for(int n=1; n<BYM3; ++n) {
                  dxytempdy[1][j][n] = dxytempdy[1][j+1][n];
               }
            }
         }
      }
      
      // Next, do shifted -y template
      
      pixy = (int)floorf((yhit-deltaxy[1])/ysize_);
      dy = (yhit-deltaxy[1])-(pixy+0.5f)*ysize_;
      if(flip_y_) {dy = -dy;}
      k0 = (int)(dy/ysize_*6.f+3.5f);
      if(k0 < 0) k0 = 0;
      if(k0 > 6) k0 = 6;
      ddy = 6.f*dy/ysize_ - (k0-3);
      ady = fabs(ddy);
      if(ddy > 0.f) {k1 = k0 + 1; if(k1 > 6) k1 = k0-1;} else {k1 = k0 - 1; if(k1 < 0) k1 = k0+1;}
      
      // OK, lets do the template interpolation.
      
      // Calculate the x and y offsets to make the new template
      
      // First, shift the struck pixel coordinates to the (Ty+2, Tx+2) system
      
      ++pixy;
      
      // In the template store, the struck pixel is always (THy,THx)
      
      deltay = pixy - T2HY;
      
      // Loop over the non-zero part of the template index space and interpolate
      
      for(int j=jmin_; j<=jmax_; ++j) {
         // Flip indices as needed
         if(flip_x_) {jflipx=T2XSIZE-1-j; m = deltax+jflipx;} else {m = deltax+j;}
         for(int i=imin_; i<=imax_; ++i) {
            if(flip_y_) {iflipy=T2YSIZE-1-i; n = deltay+iflipy;} else {n = deltay+i;}
            if(m>=0 && m<=BXM3 && n>=0 && n<=BYM3) {
               dxytempdy[0][m][n] = (float)entry00_->xytemp[k0][l0][i][j]
               + adx*(float)(entry00_->xytemp[k0][l1][i][j] - entry00_->xytemp[k0][l0][i][j])
               + ady*(float)(entry00_->xytemp[k1][l0][i][j] - entry00_->xytemp[k0][l0][i][j])
               + adcota_*(float)(entry01_->xytemp[k0][l0][i][j] - entry00_->xytemp[k0][l0][i][j])
               + adcotb_*(float)(entry10_->xytemp[k0][l0][i][j] - entry00_->xytemp[k0][l0][i][j]);
            }
         }
      }
      
      //combine rows and columns to simulate double pixels
      
      for(int n=1; n<BYM3; ++n) {
         if(ydouble[n]) {
            //  Combine the y-columns
            for(int m=1; m<BXM3; ++m) {
               dxytempdy[0][m][n] += dxytempdy[0][m][n+1];
            }
            //  Now shift the remaining pixels over by one column
            for(int i=n+1; i<BYM3; ++i) {
               for(int m=1; m<BXM3; ++m) {
                  dxytempdy[0][m][i] = dxytempdy[0][m][i+1];
               }
            }
         }
      }
      
      //combine rows and columns to simulate double pixels
      
      for(int m=1; m<BXM3; ++m) {
         if(xdouble[m]) {
            //  Combine the x-rows
            for(int n=1; n<BYM3; ++n) {
               dxytempdy[0][m][n] += dxytempdy[0][m+1][n];
            }
            //  Now shift the remaining pixels over by one row
            for(int j=m+1; j<BXM3; ++j) {
               for(int n=1; n<BYM3; ++n) {
                  dxytempdy[0][j][n] = dxytempdy[0][j+1][n];
               }
            }
         }
      }
      
      //  Finally, normalize the derivatives and copy the results to the output array
      
      for(int n=1; n<BYM3; ++n) {
         for(int m=1; m<BXM3; ++m) {
            dpdx2d[1][m][n] = (dxytempdy[1][m][n] - dxytempdy[0][m][n])/(2.*deltaxy[1]);
         }
      }
   }
   
   return success_;
} // xytemp



// *************************************************************************************************************************************
//! Interpolate stored 2-D information for input angles and hit position to make a 2-D template
//! \param       xhit - (input) x-position of hit relative to the lower left corner of pixel[1][1] (to allow for the "padding" of the two-d clusters in the splitter)
//! \param       yhit - (input) y-position of hit relative to the lower left corner of pixel[1][1]
//! \param    ydouble - (input) STL vector of 21 element array to flag a double-pixel starting at cluster[1][1]
//! \param    xdouble - (input) STL vector of 11 element array to flag a double-pixel starting at cluster[1][1]
//! \param template2d - (output) 2d template of size matched to the cluster.  Input must be zeroed since charge is added only.
// *************************************************************************************************************************************

bool SiPixelTemplate2D::xytemp(float xhit, float yhit, bool ydouble[BYM2], bool xdouble[BXM2], float template2d[BXM2][BYM2])
{
   // Interpolate for a new set of track angles
   
   bool derivatives = false;
   float dpdx2d[2][BXM2][BYM2];
   float QTemplate;
   
   return SiPixelTemplate2D::xytemp(xhit, yhit, ydouble, xdouble, template2d, derivatives, dpdx2d, QTemplate);
   
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
   
   // Local variables
   
   bool derivatives = false;
   float dpdx2d[2][BXM2][BYM2];
   float QTemplate;
   float locBx = 1.f;
   if(cotbeta < 0.f) {locBx = -1.f;}
   float locBz = locBx;
   if(cotalpha < 0.f) {locBz = -locBx;}
   
   bool yd[BYM2], xd[BXM2];
   
   yd[0] = false; yd[BYM2-1] = false;
   for(int i=0; i<TYSIZE; ++i) { yd[i+1] = ydouble[i];}
   xd[0] = false; xd[BXM2-1] = false;
   for(int j=0; j<TXSIZE; ++j) { xd[j+1] = xdouble[j];}
   
   
   // Interpolate for a new set of track angles
   
   if(SiPixelTemplate2D::interpolate(id, cotalpha, cotbeta, locBz, locBx)) {
      return SiPixelTemplate2D::xytemp(xhit, yhit, yd, xd, template2d, derivatives, dpdx2d, QTemplate);
   } else {
      return false;
   }
   
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
   if(index < 1 || index >= BYM2) {
      throw cms::Exception("DataCorrupt") << "SiPixelTemplate2D::ysigma2 called with index = " << index << std::endl;
   }
#else
   assert(index > 0 && index < BYM2);
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
   if(index <= T2HYP1) {
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
   if(xysig2 <= 0.f) {xysig2 = s50_*s50_;}
   
   return;
   
} // End xysigma2


// ************************************************************************************************************
//! Return the Landau probability parameters for this set of cot(alpha, cot(beta)
// ************************************************************************************************************
void SiPixelTemplate2D::landau_par(float lanpar[2][5])

{
   // Interpolate using quantities already stored in the private variables

   for(int i=0; i<2; ++i) {
      for(int j=0; j<5; ++j) {
         lanpar[i][j] = lanpar_[i][j];
      }
   }
   return;
   
} // End lan_par



