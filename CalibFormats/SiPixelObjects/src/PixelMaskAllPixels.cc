//
// This class provide a base class for the
// pixel mask data for the pixel FEC configuration
// This is a pure interface (abstract class) that
// needs to have an implementation.
//
// All applications should just use this 
// interface and not care about the specific
// implementation
//
//
#include <sstream>
#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaskAllPixels.h"
#include <fstream>
#include <map>
#include <iostream>
#include <assert.h>

using namespace pos;
using namespace std;

PixelMaskAllPixels::PixelMaskAllPixels(std::vector< std::vector<std::string> >& tableMat) : PixelMaskBase("","","")
{

//std::cout<<"Table Size in const:"<<tableMat.size()<<std::endl;

 std::stringstream currentRocName;
 std::vector< std::string > ins = tableMat[0];
 std::map<std::string , int > colM;
 std::vector<std::string > colNames;
 colNames.push_back("ROC_NAME");//0
 colNames.push_back("TRIM_BLOB");//1

  
 for(unsigned int c = 0 ; c < ins.size() ; c++){
   for(unsigned int n=0; n<colNames.size(); n++){
     if(tableMat[0][c] == colNames[n]){
       colM[colNames[n]] = c;
       break;
     }
   }
 }//end for
 for(unsigned int n=0; n<colNames.size(); n++){
   if(colM.find(colNames[n]) == colM.end()){
     std::cerr << "[PixelMaskAllPixels::PixelMaskAllPixels()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
     assert(0);
   }
 }

	
std::string bits;
	
  for(unsigned int r = 1 ; r < tableMat.size() ; r++){   //Goes to every row of the Matrix
    currentRocName.str("");
    currentRocName <<  tableMat[r][colM[colNames[0]]]  ;               
		                 
		   
		
		
		 PixelROCName rocid(currentRocName.str());
		 
		 
	         PixelROCMaskBits tmp;     // Have to add like this  PixelROCTrimBits tmp(rocid , bits ); 
		 
		 bits = tableMat[r][colM[colNames[1]]];
		// std::cout<<rocid<<std::endl;
		// std::cout<<bits.size()<<std::endl;
		 
		 tmp.setROCMaskBits(rocid, bits);
		 
		 maskbits_.push_back(tmp);
		  
		 
  }//end for r 

//std::cout<<maskbits_.size()<<std::endl;
}

// modified by MR on 18-04-2008 10:02:00
PixelMaskAllPixels::PixelMaskAllPixels():PixelMaskBase("","",""){;}

void PixelMaskAllPixels::addROCMaskBits(PixelROCMaskBits bits)
{
  maskbits_.push_back(bits);
}

//**********************************************************************



PixelMaskAllPixels::PixelMaskAllPixels(std::string filename):
  PixelMaskBase("","",""){

    if (filename[filename.size()-1]=='t'){


	std::ifstream in(filename.c_str());

	if (!in.good()){
	    std::cout << "Could not open:"<<filename<<std::endl;
	    assert(0);
	}
	
	std::string tag;
	in >> tag;

	maskbits_.clear();

	while (!in.eof()) {

	    PixelROCName rocid(in);

	    PixelROCMaskBits tmp;
	    
	    tmp.read(rocid,in);
	    
	    maskbits_.push_back(tmp);
	    
	    in >> tag;
	    
	}
	
	in.close();

    }
    else{

	std::ifstream in(filename.c_str(),std::ios::binary);

        char nchar;

	in.read(&nchar,1);

	//in >> nchar;

       	std::string s1;

        //wrote these lines of code without ref. needs to be fixed
	for(int i=0;i< nchar; i++){
	    char c;
	    in >>c;
	    s1.push_back(c);
	}

	//std::cout << "READ ROC name:"<<s1<<std::endl;
	
	maskbits_.clear();


	while (!in.eof()){

	    //std::cout << "PixelMaskAllPixels::PixelMaskAllPixels read s1:"<<s1<<std::endl;

	    PixelROCName rocid(s1);

	    //std::cout << "PixelMaskAllPixels::PixelMaskAllPixels read rocid:"<<rocid<<std::endl;
	    
	    PixelROCMaskBits tmp;
      
	    tmp.readBinary(rocid, in);

	    maskbits_.push_back(tmp);


	    in.read(&nchar,1);

	    s1.clear();

	    if (in.eof()) continue;
	    
	    //std::cout << "Will read:"<<(int)nchar<<" characters."<<std::endl;

	    //wrote these lines of code without ref. needs to be fixed
	    for(int i=0;i< nchar; i++){
		char c;
		in >>c;
		//std::cout <<" "<<c;
		s1.push_back(c);
	    }
	    //std::cout << std::endl;


	}

	in.close();



    }


    //std::cout << "Read maskbits for "<<maskbits_.size()<<" ROCs"<<std::endl;
	
    }
    
const PixelROCMaskBits& PixelMaskAllPixels::getMaskBits(int ROCId) const {

  return maskbits_[ROCId];

}

PixelROCMaskBits* PixelMaskAllPixels::getMaskBits(PixelROCName name) {

  for(unsigned int i=0;i<maskbits_.size();i++){
    if (maskbits_[i].name()==name) return &(maskbits_[i]);
  }

  return 0;

}

void PixelMaskAllPixels::writeBinary(std::string filename) const{

  
    std::ofstream out(filename.c_str(),std::ios::binary);

    for(unsigned int i=0;i<maskbits_.size();i++){
	maskbits_[i].writeBinary(out);
    }


}


void PixelMaskAllPixels::writeASCII(std::string dir) const{

  if (dir!="") dir+="/";
  PixelModuleName module(maskbits_[0].name().rocname());
  std::string filename=dir+"ROC_Masks_module_"+module.modulename()+".dat";
  
    std::ofstream out(filename.c_str());

    for(unsigned int i=0;i<maskbits_.size();i++){
	maskbits_[i].writeASCII(out);
    }


}

