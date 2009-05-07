//
// This class provide a base class for the
// pixel trim data for the pixel FEC configuration
// This is a pure interface (abstract class) that
// needs to have an implementation.
//
// Need to figure out what is 'VMEcommand' below! 
//
// All applications should just use this 
// interface and not care about the specific
// implementation.
//

#include <sstream>
#include <iostream>
#include <ios>
#include <assert.h>
#include "CalibFormats/SiPixelObjects/interface/PixelTrimAllPixels.h"

using namespace pos;

PixelTrimAllPixels::PixelTrimAllPixels( std::vector <std::vector<std::string> >& tableMat):
  PixelTrimBase("","","")
{

    std::stringstream currentRocName;
    std::map<std::string , int > colM;
    std::vector<std::string > colNames;
    /**
       View's name: CONF_KEY_ROC_TRIMS_MV

       Name                                      Null?    Type
       ----------------------------------------- -------- ----------------------------
       CONFIG_KEY_ID                                      NUMBER(38)
       CONFG_KEY                                          VARCHAR2(80)
       VERSION                                            VARCHAR2(40)
       KIND_OF_COND                                       VARCHAR2(40)
       ROC_NAME                                           VARCHAR2(187)
       HUB_ADDRS                                          NUMBER(38)
       PORT_NUMBER                                        NUMBER(10)
       ROC_I2C_ADDR                                       NUMBER
       GEOM_ROC_NUM                                       NUMBER(10)
       DATA_FILE                                          VARCHAR2(200)
       TRIM_CLOB                                          CLOB
    */

    colNames.push_back("CONFIG_KEY_ID"  );
    colNames.push_back("CONFG_KEY"	);
    colNames.push_back("VERSION"	);
    colNames.push_back("KIND_OF_COND"	);
    colNames.push_back("ROC_NAME"	);
    colNames.push_back("HUB_ADDRS"	);
    colNames.push_back("PORT_NUMBER"	);
    colNames.push_back("ROC_I2C_ADDR"   );
    colNames.push_back("GEOM_ROC_NUM"   );
    colNames.push_back("DATA_FILE"      );
    colNames.push_back("TRIM_CLOB"      );
 

    for(unsigned int c = 0 ; c < tableMat[0].size() ; c++)
      {
	for(unsigned int n=0; n<colNames.size(); n++)
	  {
	    if(tableMat[0][c] == colNames[n])
	      {
		colM[colNames[n]] = c;
		break;
	      }
	  }
      }//end for
    for(unsigned int n=0; n<colNames.size(); n++)
      {
      if(colM.find(colNames[n]) == colM.end())
	{
	  std::cerr << "[PixelTrimAllPixels::PixelTrimAllPixels()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
      }
    
    //unsigned char *bits ;        /// supose to be " unsigned  char bits[tableMat[1][colM["TRIM_BLOB"]].size()] ;  "
    //char c[2080];
    std::string bits;
	
    for(unsigned int r = 1 ; r < tableMat.size() ; r++)    //Goes to every row of the Matrix
      {
	PixelROCName rocid( tableMat[r][colM["ROC_NAME"]] );
	// tableMat[r][colM["TRIM_BLOB"]].copy(c , 2080 );
	// unsigned char *bits = (unsigned char* )(tableMat[r][colM["TRIM_BLOB"]].c_str());
	//bits = (unsigned char)(tableMat[r][colM["TRIM_BLOB"]].c_str());
	PixelROCTrimBits tmp;     // Have to add like this  PixelROCTrimBits tmp(rocid , bits ); 
	std::istringstream istring ;
	istring.str(tableMat[r][colM["TRIM_CLOB"]]) ;
	tmp.read(rocid, istring) ;
// 	bits = tableMat[r][colM["TRIM_CLOB"]];
	//std::cout<<rocid<<std::endl;
	// std::cout<<bits.size()<<std::endl;
// 	tmp.setROCTrimBits(rocid, bits);
	trimbits_.push_back(tmp);
	//std::cout<<"Pase por aqui:"<<r<<std::endl;
	// dacValue = atoi(tableMat[r][colM["VALUE"]].c_str());
	// pDSM.insert(pair<string,pair<string,int> >(currentRocName.str(),pair<string,int>(dacName,dacValue)));
      }//end for r 
    //std::cout<<trimbits_.size()<<std::endl;
} //end contructor with databasa table

 
PixelTrimAllPixels::PixelTrimAllPixels(std::string filename):
  PixelTrimBase("","",""){

    if (filename[filename.size()-1]=='t'){

      std::ifstream in(filename.c_str());
	
      //	std::cout << "filename =" << filename << std::endl;

      std::string s1;
      in >> s1; 

      trimbits_.clear();


      while (in.good()){
	  
	std::string s2;
	in>>s2;

	//	    std::cout << "PixelTrimAllPixels::PixelTrimAllPixels read s1:"<<s1<< " s2:" << s2 << std::endl;

	assert( s1 == "ROC:" );

	PixelROCName rocid(s2);

	//std::cout << "PixelTrimAllPixels::PixelTrimAllPixels read rocid:"<<rocid<<std::endl;
	    
	PixelROCTrimBits tmp;
      
	tmp.read(rocid, in);

	trimbits_.push_back(tmp);

	in >> s1;

      }

      in.close();

    }
    else{

      std::ifstream in(filename.c_str(),std::ios::binary);

      char nchar;

      in.read(&nchar,1);

      std::string s1;

      //wrote these lines of code without ref. needs to be fixed
      for(int i=0;i< nchar; i++){
	char c;
	in >>c;
	s1.push_back(c);
      }

      //std::cout << "READ ROC name:"<<s1<<std::endl;

      trimbits_.clear();


      while (!in.eof()){

	//std::cout << "PixelTrimAllPixels::PixelTrimAllPixels read s1:"<<s1<<std::endl;

	PixelROCName rocid(s1);

	//std::cout << "PixelTrimAllPixels::PixelTrimAllPixels read rocid:"<<rocid<<std::endl;
	    
	PixelROCTrimBits tmp;
      
	tmp.readBinary(rocid, in);

	trimbits_.push_back(tmp);


	in.read(&nchar,1);

	s1.clear();

	if (in.eof()) continue;

	//wrote these lines of code without ref. needs to be fixed
	for(int i=0;i< nchar; i++){
	  char c;
	  in >>c;
	  s1.push_back(c);
	}


      }

      in.close();



    }

    //std::cout << "Read trimbits for "<<trimbits_.size()<<" ROCs"<<std::endl;

  }


//std::string PixelTrimAllPixels::getConfigCommand(PixelMaskBase& pixelMask){
//
//  std::string s;
//  return s;
//
//}

PixelROCTrimBits PixelTrimAllPixels::getTrimBits(int ROCId) const {

  return trimbits_[ROCId];

}

PixelROCTrimBits* PixelTrimAllPixels::getTrimBits(PixelROCName name){

  for(unsigned int i=0;i<trimbits_.size();i++){
    if (trimbits_[i].name()==name) return &(trimbits_[i]);
  }

  return 0;

}



void PixelTrimAllPixels::generateConfiguration(PixelFECConfigInterface* pixelFEC,
					       PixelNameTranslation* trans,
					       const PixelMaskBase& pixelMask) const{

  for(unsigned int i=0;i<trimbits_.size();i++){

    std::vector<unsigned char> trimAndMasks(4160);

    const PixelROCMaskBits& maskbits=pixelMask.getMaskBits(i);

    for (unsigned int col=0;col<52;col++){
      for (unsigned int row=0;row<80;row++){
	unsigned char tmp=trimbits_[i].trim(col,row);
	if (maskbits.mask(col,row)!=0) tmp|=0x80;
	trimAndMasks[col*80+row]=tmp;
      }
    }

    // the slow way, one pixel at a time
    //pixelFEC->setMaskAndTrimAll(*(trans->getHdwAddress(trimbits_[i].name())),trimAndMasks);
    // the fast way, a full roc in column mode (& block xfer)
    const PixelHdwAddress* theROC = trans->getHdwAddress(trimbits_[i].name());
    pixelFEC->roctrimload(theROC->mfec(),
			  theROC->mfecchannel(),
			  theROC->hubaddress(),
			  theROC->portaddress(),
			  theROC->rocid(),
			  trimAndMasks);
  }
}

void PixelTrimAllPixels::writeBinary(std::string filename) const{

  
  std::ofstream out(filename.c_str(),std::ios::binary);

  for(unsigned int i=0;i<trimbits_.size();i++){
    trimbits_[i].writeBinary(out);
  }


}


void PixelTrimAllPixels::writeASCII(std::string dir) const{

  if (dir!="") dir+="/";
  PixelModuleName module(trimbits_[0].name().rocname());
  std::string filename=dir+"ROC_Trims_module_"+module.modulename()+".dat";
  

  std::ofstream out(filename.c_str());

  for(unsigned int i=0;i<trimbits_.size();i++){
    trimbits_[i].writeASCII(out);
  }


}
