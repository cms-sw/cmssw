//
// This class provide a base class for the
// pixel ROC DAC data for the pixel FEC configuration
//
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelTBMSettings.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <ios>
#include <assert.h>
#include <stdexcept>


using namespace pos;


PixelTBMSettings::PixelTBMSettings(std::vector < std::vector< std::string> > &tableMat):PixelConfigBase("","",""){
  std::string mthn = "]\t[PixelTBMSettings::PixelTBMSettings()]\t\t\t    " ;
  std::vector< std::string > ins = tableMat[0];
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;

  /**

     EXTENSION_TABLE_NAME:     (VIEW:)

     CONFIG_KEY 			       NOT NULL VARCHAR2(80)
     KEY_TYPE				       NOT NULL VARCHAR2(80)
     KEY_ALIAS  			       NOT NULL VARCHAR2(80)
     VERSION						VARCHAR2(40)
     KIND_OF_COND			       NOT NULL VARCHAR2(40)
     TBM_NAME						VARCHAR2(200)
     MODULE_NAME			       NOT NULL VARCHAR2(200)
     HUB_ADDRS  					NUMBER(38)
     TBM_MODE						VARCHAR2(200)
     ANLG_INBIAS_ADDR					NUMBER(38)
     ANLG_INBIAS_VAL			       NOT NULL NUMBER(38)
     ANLG_OUTBIAS_ADDR  				NUMBER(38)
     ANLG_OUTBIAS_VAL			       NOT NULL NUMBER(38)
     ANLG_OUTGAIN_ADDR  				NUMBER(38)
     ANLG_OUTGAIN_VAL			       NOT NULL NUMBER(38)

     N.B.: Here we should (MUST) get a single row referring to a particula module for a particula version.
  */

  colNames.push_back("CONFIG_KEY" 	);
  colNames.push_back("KEY_TYPE"   	);
  colNames.push_back("KEY_ALIAS"  	);
  colNames.push_back("VERSION"    	);
  colNames.push_back("KIND_OF_COND"	);
  colNames.push_back("TBM_NAME"   	);
  colNames.push_back("MODULE_NAME"      );
  colNames.push_back("HUB_ADDRS"  	);
  colNames.push_back("TBM_MODE"   	);
  colNames.push_back("ANLG_INBIAS_ADDR" );
  colNames.push_back("ANLG_INBIAS_VAL"  );
  colNames.push_back("ANLG_OUTBIAS_ADDR");
  colNames.push_back("ANLG_OUTBIAS_VAL" );
  colNames.push_back("ANLG_OUTGAIN_ADDR");
  colNames.push_back("ANLG_OUTGAIN_VAL" );

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
      std::cerr << __LINE__ << mthn << "Couldn't find in the database the column with name " << colNames[n] << std::endl;
      assert(0);
    }
  }
 
  if(tableMat.size() >1)
    {
      //std::cout << __LINE__ << mthn << "Module from DB: " << tableMat[1][colM["MODULE_NAME"]]<< std::endl ;
      PixelROCName tmp(tableMat[1][colM["MODULE_NAME"]]);
      rocid_ = tmp ;
      //std::cout << __LINE__ << mthn << "Built ROCNAME: " << rocid_.rocname()<< std::endl ;
      
      analogInputBias_  = atoi(tableMat[1][colM["ANLG_INBIAS_VAL"]].c_str());
      analogOutputBias_ = atoi(tableMat[1][colM["ANLG_OUTBIAS_VAL"]].c_str());
      analogOutputGain_ = atoi(tableMat[1][colM["ANLG_OUTGAIN_VAL"]].c_str());
      
      if( tableMat[1][colM["TBM_MODE"]] == "SingleMode"){
	singlemode_=true;
      }
      else{
	singlemode_=false;
      }
    }
}//end contructor

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PixelTBMSettings::PixelTBMSettings(std::string filename):
  PixelConfigBase("","",""){

    std::string mthn = "]\t[PixelTBMSettings::PixelTBMSettings()]\t\t\t    " ;
    if (filename[filename.size()-1]=='t'){

	std::ifstream in(filename.c_str());

	if (!in.good()){
	    std::cout << __LINE__ << mthn << "Could not open:"<<filename<<std::endl;
	    throw std::runtime_error("Failed to open file "+filename);
	}
	else {
	  // std::cout << "Opened:"<<filename<<std::endl;
	}
	
       	std::string tag;

	PixelROCName tmp(in);

	rocid_=tmp;

        unsigned int tmpint;	

	in >> tag;
	//std::cout << "Tag="<<tag<<std::endl;
	assert(tag=="AnalogInputBias:");
	in >> tmpint;
	analogInputBias_=tmpint;

	in >> tag;
	//std::cout << "Tag="<<tag<<std::endl;
	assert(tag=="AnalogOutputBias:");
	in >> tmpint;
	analogOutputBias_=tmpint;

	in >> tag;
	//std::cout << "Tag="<<tag<<std::endl;
	assert(tag=="AnalogOutputGain:");
	in >> tmpint;
	analogOutputGain_=tmpint;

	in >> tag;
	//std::cout << "Tag="<<tag<<std::endl;
	assert(tag=="Mode:");
	in >> tag;
	assert(tag=="SingleMode"||tag=="DualMode");
	
	singlemode_=true;

	if (tag=="DualMode") singlemode_=false;

	in.close();

    }
    else{

	std::ifstream in(filename.c_str(),std::ios::binary);

	if (!in.good()){
	    std::cout << __LINE__ << mthn << "Could not open:"<<filename<<std::endl;
	    assert(0);
	}
	else {
	    std::cout << __LINE__ << mthn << "Opened:"<<filename<<std::endl;
	}

	char nchar;
	std::string s1;

	in.read(&nchar,1);

	s1.clear();

	//wrote these lines of code without ref. needs to be fixed
	for(int i=0;i< nchar; i++){
	    char c;
	    in >>c;
	    s1.push_back(c);
	}

	PixelROCName tmp(s1);

	rocid_=tmp;

	in >> analogInputBias_;
	in >> analogOutputBias_;
	in >> analogOutputGain_;
	in >> singlemode_;

	in.close();



    }


}

void PixelTBMSettings::setTBMGenericValue(std::string what, int value) 
{
 if(      what == "analogInputBias" )  {analogInputBias_  = (unsigned char)value;}
 else if( what == "analogOutputBias" ) {analogOutputBias_ = (unsigned char)value;}
 else if( what == "analogOutputGain" ) {analogOutputGain_ = (unsigned char)value;}
 else if( what == "Mode" )             {singlemode_       = (bool)value;         }
 else 
 {
   std::cout << __LINE__ << "]\t[PixelTBMSettings::setTBMGenericValue()]\t\tFATAL: invalid key/value pair: " << what << "/" << value << std::endl ; 
   assert(0);
 }
}
 
void PixelTBMSettings::writeBinary(std::string filename) const {

    std::ofstream out(filename.c_str(),std::ios::binary);

    out << (char)rocid_.rocname().size();
    out.write(rocid_.rocname().c_str(),rocid_.rocname().size());

    out <<analogInputBias_;
    out <<analogOutputBias_;
    out <<analogOutputGain_;
    out << singlemode_;


}

void PixelTBMSettings::writeASCII(std::string dir) const {

  PixelModuleName module(rocid_.rocname());
  
  if (dir!="") dir+="/";
  std::string filename=dir+"TBM_module_"+module.modulename()+".dat";

    std::ofstream out(filename.c_str());

    out << rocid_.rocname() << std::endl;

    out << "AnalogInputBias: "<<(int)analogInputBias_<<std::endl;
    out << "AnalogOutputBias: "<<(int)analogOutputBias_<<std::endl;
    out << "AnalogOutputGain: "<<(int)analogOutputGain_<<std::endl;
    out << "Mode: ";
    if (singlemode_) {
      out << "SingleMode" << std::endl;
    }
    else{
      out << "DualMode" << std::endl;
    }
}

void PixelTBMSettings::generateConfiguration(PixelFECConfigInterface* pixelFEC,
					     PixelNameTranslation* trans,
					     bool physics, bool doResets) const{

    PixelHdwAddress theROC=*(trans->getHdwAddress(rocid_));


    int mfec=theROC.mfec();
    int mfecchannel=theROC.mfecchannel();
    int tbmchannel=14; 
    int tbmchannelB=15; 
    int hubaddress=theROC.hubaddress();

    if (doResets) {
      pixelFEC->injectrsttbm(mfec, 1);
      pixelFEC->injectrstroc(mfec,1);
    }
    pixelFEC->enablecallatency(mfec,0);
    pixelFEC->disableexttrigger(mfec,0);
    pixelFEC->injecttrigger(mfec,0);
    pixelFEC->callatencycount(mfec,79);

    //pixelFEC->synccontrolregister(mfec);

    //Reset TBM and reset ROC
    if (doResets)    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 2, 0x14, 0);
    //setting speed to 40MHz
    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 0, 1, 0);
    // setting the mode, we should always stay in the CAL mode
    // since the EventNumberClear Mode does not work correctly
    //if (physics) {  // comment out, stau always in the CAL mode,  d.k. 27/09/09
    //pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 1, 0x80, 0);
    //} else {
    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 1, 0xc0, 0);
    //}
    //Enable token and analog output
    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 4, 0x0, 0);

    //Analog input bias
    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 5, 
		     analogInputBias_, 0);
    //Analog output bias
    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 6, 
		     analogOutputBias_, 0);
    //Analog output gain
    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 7, 
		     analogOutputGain_, 0);


    //setting speed to 40MHz
    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannelB, hubaddress, 4, 0, 1, 0);
    //pre-calibration, stay always in this mode
    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannelB, hubaddress, 4, 1, 0xc0, 0);
    //Reset TBM and reset ROC
    if (doResets)    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannelB, hubaddress, 4, 2, 0x14, 0);
    //Enable token and analog output
    if (singlemode_){
      pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannelB, hubaddress, 4, 4, 0x3, 0);
    }
    else{
      pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannelB, hubaddress, 4, 4, 0x0, 0);
    }
} 


std::ostream& pos::operator<<(std::ostream& s, const PixelTBMSettings& tbm){

    s << "Module          :"<<tbm.rocid_.rocname() <<std::endl; 
    s << "analogInputBias :"<<tbm.analogInputBias_<<std::endl;
    s << "analogOutputBias:"<<tbm.analogOutputBias_<<std::endl;
    s << "analogOutputGain:"<<tbm.analogOutputGain_<<std::endl;
    if (tbm.singlemode_){
      s << "mode            :Singlemode"<<std::endl;
    }
    else{
      s << "mode            :Dualmode"<<std::endl;
    }

    return s;

}
//=============================================================================================
void PixelTBMSettings::writeXMLHeader(pos::PixelConfigKey key, 
                                      int version, 
                                      std::string path, 
                                      std::ofstream *outstream,
                                      std::ofstream *out1stream,
                                      std::ofstream *out2stream) const
{
  std::string mthn = "]\t[PixelTBMSettings::writeXMLHeader()]\t\t\t    " ;
  std::stringstream fullPath ;
  fullPath << path << "/Pixel_TbmParameters_" << PixelTimeFormatter::getmSecTime() << ".xml" ;
  std::cout << __LINE__ << mthn << "Writing to: " << fullPath.str() << std::endl ;
  
  outstream->open(fullPath.str().c_str()) ;
  
  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"			 	     << std::endl ;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 		 	             << std::endl ;
  *outstream << " <HEADER>"								         	     << std::endl ;
  *outstream << "  <TYPE>"								         	     << std::endl ;
  *outstream << "   <EXTENSION_TABLE_NAME>PIXEL_TBM_PARAMETERS</EXTENSION_TABLE_NAME>"          	     << std::endl ;
  *outstream << "   <NAME>Pixel TBM Parameters</NAME>"				         	             << std::endl ;
  *outstream << "  </TYPE>"								         	     << std::endl ;
  *outstream << "  <RUN>"								         	     << std::endl ;
  *outstream << "   <RUN_TYPE>Pixel TBM Parameters</RUN_TYPE>" 		                                     << std::endl ;
  *outstream << "   <RUN_NUMBER>1</RUN_NUMBER>"					         	             << std::endl ;
  *outstream << "   <RUN_BEGIN_TIMESTAMP>" << pos::PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << std::endl ;
  *outstream << "   <LOCATION>CERN P5</LOCATION>"                                                            << std::endl ; 
  *outstream << "  </RUN>"								         	     << std::endl ;
  *outstream << " </HEADER>"								         	     << std::endl ;
  *outstream << ""										 	     << std::endl ;
  *outstream << " <DATA_SET>"								         	     << std::endl ;
  *outstream << "  <PART>"                                                                                   << std::endl ;
  *outstream << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"                                                 << std::endl ;
  *outstream << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"                                              << std::endl ;
  *outstream << "  </PART>"                                                                                  << std::endl ;
  *outstream << "  <VERSION>"             << version      << "</VERSION>"				     << std::endl ;
  *outstream << "  <COMMENT_DESCRIPTION>" << getComment() << "</COMMENT_DESCRIPTION>"			     << std::endl ;
  *outstream << "  <CREATED_BY_USER>"     << getAuthor()  << "</CREATED_BY_USER>"  			     << std::endl ;
  *outstream << " "				         	                                             << std::endl ;
}

//=============================================================================================
void PixelTBMSettings::writeXML(std::ofstream *outstream,
                                std::ofstream *out1stream,
                                std::ofstream *out2stream) const 
{
  std::string mthn = "]\t[PixelTBMSettings::writeXML()]\t\t\t    " ;

  PixelModuleName module(rocid_.rocname());
  													     
  *outstream << "  <DATA>"										     << std::endl ;
  *outstream << "   <MODULE_NAME>"	<< rocid_.rocname()	 << "</MODULE_NAME>"			     << std::endl ;
  *outstream << "   <ANLG_INBIAS_VAL>"  <<(int)analogInputBias_  << "</ANLG_INBIAS_VAL>"		     << std::endl ;
  *outstream << "   <ANLG_OUTBIAS_VAL>" <<(int)analogOutputBias_ << "</ANLG_OUTBIAS_VAL>"		     << std::endl ;
  *outstream << "   <ANLG_OUTGAIN_VAL>" <<(int)analogOutputGain_ << "</ANLG_OUTGAIN_VAL>"		     << std::endl ;
  if (singlemode_) {											     	
    *outstream << "  <TBM_MODE>SingleMode</TBM_MODE>" 					 		     << std::endl ;
  }
  else{ 												     
    *outstream << "  <TBM_MODE>DualMode</TBM_MODE>"   					 		     << std::endl ;
  }													     
  *outstream << "  </DATA>"                                                               		     << std::endl ;
}

//=============================================================================================
void PixelTBMSettings::writeXMLTrailer(std::ofstream *outstream,
                                       std::ofstream *out1stream,
                                       std::ofstream *out2stream) const 
{
  std::string mthn = "]\t[PixelTBMSettings::writeXMLTrailer()]\t\t\t    " ;
  
  *outstream << " " 						    	 	              	             << std::endl ;
  *outstream << " </DATA_SET>" 						    	 	              	     << std::endl ;
  *outstream << "</ROOT> "								              	     << std::endl ;

  outstream->close() ;
}
