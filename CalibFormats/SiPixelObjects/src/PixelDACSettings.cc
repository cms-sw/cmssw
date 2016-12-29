//
// This class provide a base class for the
// pixel ROC DAC data for the pixel FEC configuration
//
//
//
//


#include "CalibFormats/SiPixelObjects/interface/PixelDACSettings.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCDACSettings.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDACNames.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include <fstream>
#include <iostream>
#include <ios>
#include <assert.h>
#include <stdexcept>
#include <map>
#include <sstream>
#include <sys/time.h>
#include <cstdlib>

using namespace pos;

namespace {
  const bool readTemperatures = false;
  //const bool readTemperatures = true;
  //const int temperatureReg = 0x9; // hardwire to fixed reference voltage 0x8 + 0x1
  const int temperatureReg = 0x1; // hardwire to the usefull range, change to range 1, Marco's request, 25/10/11
}

PixelDACSettings::PixelDACSettings(std::string filename):
  PixelConfigBase("","",""){

  std::string mthn = "[PixelDACSettings::PixelDACSettings()]\t\t\t    " ;
  
  if (filename[filename.size()-1]=='t'){

    std::ifstream in(filename.c_str());

    if (!in.good()){
      std::cout << __LINE__ << "]\t" << mthn << "Could not open: " << filename << std::endl;
      // assert(0); //in case of failure, we don't want POS to die here
      throw std::runtime_error("Failed to open file "+filename);
    }
    else {
      // std::cout << "Opened:"<<filename<<std::endl;
    }
	
	
    dacsettings_.clear();
	
    std::string tag;
	
    in >> tag;

    while (!in.eof()){
	    
	    
      PixelROCName rocid(in);

      //	    std::cout << "[PixelDACSettings::PixelDACSettings()] DAC setting ROC id:"<<rocid<<std::endl;
	    
      PixelROCDACSettings tmp;

      tmp.read(in,rocid);
	    
      //	    std::cout << "[PixelDACSettings::PixelDACSettings()] DACSetting: " << std::endl << tmp << std::endl ;
      dacsettings_.push_back(tmp);
	    
      in >> tag;
	    
      assert(dacsettings_.size()<100);
	    
    }
	
    in.close();

  }
  else{

    std::ifstream in(filename.c_str(),std::ios::binary);

    if (!in.good()){
      std::cout << __LINE__ << "]\t" << mthn << "Could not open: " << filename << std::endl;
      assert(0);
    }
    else {
      std::cout << __LINE__ << "]\t" << mthn << "Opened: "	   << filename << std::endl;
    }

    char nchar;

    in.read(&nchar,1);
	
    std::string s1;

    //wrote these lines of code without ref. needs to be fixed
    for(int i=0;i< nchar; i++){
      char c;
      in >>c;
      s1.push_back(c);
    }

    //std::cout << __LINE__ << "]\t" << mthn << "READ ROC name: " << s1 << std::endl;

    dacsettings_.clear();


    while (!in.eof()){

      //std::cout << __LINE__ << "]\t" << mthn << "read s1   : " << s1    << std::endl;

      PixelROCName rocid(s1);

      //std::cout << __LINE__ << "]\t" << mthn << "read rocid: " << rocid << std::endl;
	    
      PixelROCDACSettings tmp;
      
      tmp.readBinary(in, rocid);

      dacsettings_.push_back(tmp);


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


  //std::cout << __LINE__ << "]\t" << mthn << "Read dac settings for "<<dacsettings_.size()<<" ROCs"<<std::endl;


}
// modified by MR on 10-01-2008 14:48:19
PixelDACSettings::PixelDACSettings(PixelROCDACSettings &rocname):
  PixelConfigBase("","","") {
  dacsettings_.push_back(rocname) ;

}

// modified by MR on 24-01-2008 14:27:35a
void PixelDACSettings::addROC(PixelROCDACSettings &rocname)
{
  dacsettings_.push_back(rocname) ;

}


PixelDACSettings::PixelDACSettings(std::vector< std::vector<std::string> > &tableMat): PixelConfigBase("","","")
{



/*
 EXTENSION_TABLE_NAME: ROC_DAC_SETTINGS_COL (VIEW: CONF_KEY_ROCDAC_COL_V)
 
 CONFIG_KEY				   NOT NULL VARCHAR2(80)
 KEY_TYPE				   NOT NULL VARCHAR2(80)
 KEY_ALIAS				   NOT NULL VARCHAR2(80)
 VERSION					    VARCHAR2(40)
 KIND_OF_COND				   NOT NULL VARCHAR2(40)
 ROC_NAME					    VARCHAR2(200)
 VDD					   NOT NULL NUMBER(38)
 VANA					   NOT NULL NUMBER(38)
 VSF					   NOT NULL NUMBER(38)
 VCOMP					   NOT NULL NUMBER(38)
 VLEAK					   NOT NULL NUMBER(38)
 VRGPR					   NOT NULL NUMBER(38)
 VWLLPR 				   NOT NULL NUMBER(38)
 VRGSH					   NOT NULL NUMBER(38)
 VWLLSH 				   NOT NULL NUMBER(38)
 VHLDDEL				   NOT NULL NUMBER(38)
 VTRIM					   NOT NULL NUMBER(38)
 VCTHR					   NOT NULL NUMBER(38)
 VIBIAS_BUS				   NOT NULL NUMBER(38)
 VIBIAS_SF				   NOT NULL NUMBER(38)
 VOFFSETOP				   NOT NULL NUMBER(38)
 VBIASOP				   NOT NULL NUMBER(38)
 VOFFSETRO				   NOT NULL NUMBER(38)
 VION					   NOT NULL NUMBER(38)
 VIBIAS_PH				   NOT NULL NUMBER(38)
 VIBIAS_DAC				   NOT NULL NUMBER(38)
 VIBIAS_ROC				   NOT NULL NUMBER(38)
 VICOLOR				   NOT NULL NUMBER(38)
 VNPIX					   NOT NULL NUMBER(38)
 VSUMCOL				   NOT NULL NUMBER(38)
 VCAL					   NOT NULL NUMBER(38)
 CALDEL 				   NOT NULL NUMBER(38)
 TEMPRANGE				   NOT NULL NUMBER(38)
 WBC					   NOT NULL NUMBER(38)
 CHIPCONTREG				   NOT NULL NUMBER(38)
*/
//   std::multimap<std::string,std::pair<std::string,int > > pDSM;
  //  std::stringstream currentRocName;
  std::vector< std::string > ins = tableMat[0];
  std::string mthn("[PixelDACSettings::PixelDACSettings()] ") ;
  std::string dacName;
  std::istringstream dbin ;
//   int dacValue;
  int skipColumns = 0 ;
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  std::map<std::string, std::string> nameTranslation ;

  colNames.push_back("CONFIG_KEY"  );
  colNames.push_back("KEY_TYPE"    );
  colNames.push_back("KEY_ALIAS"   );
  colNames.push_back("VERSION"     );
  colNames.push_back("KIND_OF_COND");
  colNames.push_back("ROC_NAME"    );
  colNames.push_back("VDD"	   );
  colNames.push_back("VANA"	   );
  colNames.push_back("VSF"	   );
  colNames.push_back("VCOMP"	   );
  colNames.push_back("VLEAK"	   );
  colNames.push_back("VRGPR"	   );
  colNames.push_back("VWLLPR"	   );
  colNames.push_back("VRGSH"	   );
  colNames.push_back("VWLLSH"	   );
  colNames.push_back("VHLDDEL"     );
  colNames.push_back("VTRIM"	   );
  colNames.push_back("VCTHR"	   );
  colNames.push_back("VIBIAS_BUS"  );
  colNames.push_back("VIBIAS_SF"   );
  colNames.push_back("VOFFSETOP"   );
  colNames.push_back("VBIASOP"     );
  colNames.push_back("VOFFSETRO"   );
  colNames.push_back("VION"	   );
  colNames.push_back("VIBIAS_PH"   );
  colNames.push_back("VIBIAS_DAC"  );
  colNames.push_back("VIBIAS_ROC"  );
  colNames.push_back("VICOLOR"     );
  colNames.push_back("VNPIX"	   );
  colNames.push_back("VSUMCOL"     );
  colNames.push_back("VCAL"	   );
  colNames.push_back("CALDEL"	   );
  colNames.push_back("TEMPRANGE"   );
  colNames.push_back("WBC"	   );
  colNames.push_back("CHIPCONTREG" );

  nameTranslation["VDD"]          = k_DACName_Vdd ;
  nameTranslation["VANA"]         = k_DACName_Vana;               
  nameTranslation["VSF"]          = k_DACName_Vsf;                 
  nameTranslation["VCOMP"]        = k_DACName_Vcomp;             
  nameTranslation["VLEAK"]        = k_DACName_Vleak;             
  nameTranslation["VRGPR"]        = k_DACName_VrgPr;             
  nameTranslation["VWLLPR"]       = k_DACName_VwllPr;           
  nameTranslation["VRGSH"]        = k_DACName_VrgSh;             
  nameTranslation["VWLLSH"]       = k_DACName_VwllSh;           
  nameTranslation["VHLDDEL"]      = k_DACName_VHldDel;         
  nameTranslation["VTRIM"]        = k_DACName_Vtrim;             
  nameTranslation["VCTHR"]        = k_DACName_VcThr;             
  nameTranslation["VIBIAS_BUS"]   = k_DACName_VIbias_bus;  
  nameTranslation["VIBIAS_SF"]    = k_DACName_VIbias_sf;     
  nameTranslation["VOFFSETOP"]    = k_DACName_VOffsetOp;     
  nameTranslation["VBIASOP"]      = k_DACName_VbiasOp;         
  nameTranslation["VOFFSETRO"]    = k_DACName_VOffsetRO;     
  nameTranslation["VION"]         = k_DACName_VIon;               
  nameTranslation["VIBIAS_PH"]    = k_DACName_VIbias_PH;     
  nameTranslation["VIBIAS_DAC"]   = k_DACName_VIbias_DAC;   
  nameTranslation["VIBIAS_ROC"]   = k_DACName_VIbias_roc;   
  nameTranslation["VICOLOR"]      = k_DACName_VIColOr;         
  nameTranslation["VNPIX"]        = k_DACName_Vnpix;             
  nameTranslation["VSUMCOL"]      = k_DACName_VsumCol;         
  nameTranslation["VCAL"]         = k_DACName_Vcal;               
  nameTranslation["CALDEL"]       = k_DACName_CalDel;           
  nameTranslation["TEMPRANGE"]    = k_DACName_TempRange;     
  nameTranslation["WBC"]          = k_DACName_WBC;                 
  nameTranslation["CHIPCONTREG"]  = k_DACName_ChipContReg; 

  // modified by MR on 25-02-2008 10:00:45
  // colM stores the index (referred to tableMat) where the specified dac setting is store!!!
  for(unsigned int c = skipColumns ; c < ins.size() ; c++){
    for(unsigned int n=0; n<colNames.size(); n++){
      if(tableMat[0][c] == colNames[n]){
	colM[colNames[n]] = c;
	break;
      }
    }
  }//end for
  for(unsigned int n=skipColumns; n<colNames.size(); n++){
    if(colM.find(colNames[n]) == colM.end()){
      std::cerr << "[PixelDACSettings::PixelDACSettings()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
      assert(0);
    }
  }

	
  dacsettings_.clear();
//   struct timeval  start_time  ;
//   struct timeval  end_time    ;
//   gettimeofday(&start_time, (struct timezone *)0 );
  for(unsigned int r = 1 ; r < tableMat.size() ; r++){    //Goes to every row of the Matrix
    // currentRocName.str("");
    // currentRocName << tableMat[r][colM["NAME"]] ; 
    //currentRocName << "FPix_BmI_D" << tableMat[r][colM["HDISK_POSN"]]                 
    //	   << "_BLD"       << tableMat[r][colM["BLD_POSN"]]                  
    //	   << "_PNL"       << tableMat[r][colM["PANEL_POSITION"]]            
    //	   << "_PLQ"       << tableMat[r][colM["PLAQ_POS"]]                 
    //	   << "_ROC"       << tableMat[r][colM["ROC_POSN"]];                
		   
    // modified by MR on 25-02-2008 10:04:55
    PixelROCName rocid(tableMat[r][colM["ROC_NAME"]]);
    PixelROCDACSettings tmp(rocid);
    std::ostringstream dacs("") ;
    // +6 to get rid of the first 5 columns not pertaining DAC Settings...
    for(unsigned int n=skipColumns+6; n<colNames.size(); n++)
      {
	dacs << nameTranslation[colNames[n]] <<": "<< atoi(tableMat[r][colM[colNames[n]]].c_str()) << std::endl ;
	//       dacName  = colNames[n];
	//       dacValue = atoi(tableMat[r][colM[colNames[n]]].c_str());
	//       pDSM.insert(std::pair<std::string,std::pair<std::string,int> >(tableMat[r][colM["ROC_NAME"]],std::pair<std::string,int>(dacName,dacValue)));
	//       std::cout << "On " << tableMat[r][colM["ROC_NAME"]] << " DAC:\t" << dacName << " value:\t" << dacValue<< std::endl ;
	//       tmp.setDac(dacName, dacValue) ;
      }
//     tmp.setDACs(tmpDACs) ;
    dbin.str(dacs.str()) ;
    tmp.read(dbin, rocid) ;
    dacsettings_.push_back(tmp) ;
  }//end for r
//   gettimeofday(&end_time, (struct timezone *)0 );
//   int total_usecs = (end_time.tv_sec - start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec);
//   std::cout << mthn << "Time taken : " << total_usecs / 1000000.  << " secs" << std::endl;
  
//   dacsettings_.clear();
//   std::string currentRocName2 = "";
//   for(std::multimap<std::string,std::pair<std::string,int> >::iterator tableMapIt=pDSM.begin(); tableMapIt!= pDSM.end(); tableMapIt++){
//     if(currentRocName2 != tableMapIt->first){
//       std::cout << tableMapIt->first << std::endl;
//       std::cout << tableMapIt->second.first << std::endl;
//       std::cout << tableMapIt->second.second << std::endl;
//       currentRocName2 = tableMapIt->first;
//       PixelROCName rocid(currentRocName2);
      
//       // std::cout << "DAC setting ROC id:"<<rocid<<std::endl;
  
//       PixelROCDACSettings tmp(rocid);
      
//       //       tmp.read(in,rocid);
	    
//       dacsettings_.push_back(tmp);
//     }//end if
//     dacsettings_[dacsettings_.size()-1].setDac(tableMapIt->second.first,tableMapIt->second.second);
//   }//end for 
  
  
//   for(unsigned int w = 0 ; w < dacsettings_.size() ; w++)
//     {
  
//       PixelROCDACSettings tmp2 = dacsettings_[w];
//       //   std::cout<<tmp2<<std::endl;
//     }   
  //  std::cout<<"Number of ROCs in the PixelDACSettings::PixelDACSettings(vector <vector<string> >):"<<dacsettings_.size()<<std::endl; 
  //  std::cout << "[PixelDACSettings::PixelDACSettings(std::vector)] before end of constructor" << std::endl ;
}//end PDSMatrix constructor
//end added by Umesh

PixelROCDACSettings PixelDACSettings::getDACSettings(int ROCId) const {

  return dacsettings_[ROCId];

}
 
PixelROCDACSettings* PixelDACSettings::getDACSettings(PixelROCName name){


  for(unsigned int i=0;i<dacsettings_.size();i++){
    if (dacsettings_[i].getROCName()==name) return &(dacsettings_[i]);
  }

  return 0;

}
 
void PixelDACSettings::writeBinary(std::string filename) const {

  std::ofstream out(filename.c_str(),std::ios::binary);

  for(unsigned int i=0;i<dacsettings_.size();i++){
    dacsettings_[i].writeBinary(out);
  }

}


void PixelDACSettings::writeASCII(std::string dir) const {

  std::string mthn = "[PixelDACSettings::writeASCII()]\t\t\t    " ;
  PixelModuleName module(dacsettings_[0].getROCName().rocname());

  std::string filename=dir+"/ROC_DAC_module_"+module.modulename()+".dat";
  std::cout << __LINE__ << "]\t" << mthn << "Writing to file " << filename << std::endl ; 
  std::ofstream out(filename.c_str());
  
  for(unsigned int i=0;i<dacsettings_.size();i++){
    dacsettings_[i].writeASCII(out);
  }

}

//=============================================================================================
void PixelDACSettings::writeXMLHeader(pos::PixelConfigKey key, 
				      int version, 
				      std::string path, 
				      std::ofstream *outstream,
				      std::ofstream *out1stream,
				      std::ofstream *out2stream) const {
  std::string mthn = "[PixelDACSettings::writeXMLHeader()]\t\t\t    " ;
  std::stringstream fullPath ;

  fullPath << path << "/Pixel_RocDacSettings_" << PixelTimeFormatter::getmSecTime() << ".xml" ;
  std::cout << __LINE__ << "]\t" << mthn << "Writing to: " << fullPath.str()  << std::endl ;

  outstream->open(fullPath.str().c_str()) ;
  
  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"		         	<< std::endl ;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 	         	        << std::endl ;
  *outstream << ""                                                                                      << std::endl ; 
  *outstream << " <!-- " << mthn << "-->"                                                               << std::endl ; 
  *outstream << ""                                                                                      << std::endl ; 
  *outstream << " <HEADER>"										<< std::endl ;
  *outstream << "  <TYPE>"										<< std::endl ;
  *outstream << "   <EXTENSION_TABLE_NAME>ROC_DAC_SETTINGS_COL</EXTENSION_TABLE_NAME>"		  	<< std::endl ;
  *outstream << "   <NAME>ROC DAC Settings Col</NAME>"						  	<< std::endl ;
  *outstream << "  </TYPE>"										<< std::endl ;
  *outstream << "  <RUN>"										<< std::endl ;
  *outstream << "   <RUN_TYPE>ROC DAC Settings</RUN_TYPE>"						<< std::endl ;
  *outstream << "   <RUN_NUMBER>1</RUN_NUMBER>"							  	<< std::endl ;
  *outstream << "   <RUN_BEGIN_TIMESTAMP>" << PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << std::endl ;
  *outstream << "   <LOCATION>CERN P5</LOCATION>"                                                       << std::endl ; 
  *outstream << "  </RUN>"										<< std::endl ;
  *outstream << " </HEADER>"										<< std::endl ;
  *outstream << ""											<< std::endl ;
  *outstream << " <DATA_SET>" 									        << std::endl ;
  *outstream << "  <VERSION>" 		  << version      << "</VERSION>"				<< std::endl ;
  *outstream << "  <COMMENT_DESCRIPTION>" << getComment() << "</COMMENT_DESCRIPTION>"		        << std::endl ;
  *outstream << "  <CREATED_BY_USER>"     << getAuthor()  << "</CREATED_BY_USER>"  		        << std::endl ;
  *outstream << " "											<< std::endl ;
  *outstream << "  <PART>"										<< std::endl ;
  *outstream << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"  					  	<< std::endl ;
  *outstream << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"					  	<< std::endl ;
  *outstream << "  </PART>"										<< std::endl ;
  *outstream << " "                                                                       		<< std::endl ;

  std::cout << __LINE__ << "]\t" << mthn << "Header written" << std::endl ;
}

//=============================================================================================
void PixelDACSettings::writeXML( std::ofstream *outstream,
				 std::ofstream *out1stream,
				 std::ofstream *out2stream) const {
  std::string mthn = "[PixelDACSettings::writeXML()]\t\t\t    " ;

  for(unsigned int i=0;i<dacsettings_.size();i++){
    dacsettings_[i].writeXML(outstream);
  }
}

//=============================================================================================
void PixelDACSettings::writeXMLTrailer(std::ofstream *outstream,
				       std::ofstream *out1stream,
				       std::ofstream *out2stream) const {
  std::string mthn = "[PixelDACSettings::writeXMLTrailer()]\t\t\t    " ;

  *outstream << " </DATA_SET>"              							       << std::endl ;
  *outstream << "</ROOT>"                   							       << std::endl ;

  outstream->close() ;
  std::cout << __LINE__ << "]\t" << mthn << "Data written"       				       << std::endl ;
}

/* O B S O L E T E -----

//=============================================================================================
void PixelDACSettings::writeXML(pos::PixelConfigKey key, int version, std::string path) const {
  std::string mthn = "[PixelDACSettings::writeXML()]\t\t\t    " ;
  std::stringstream fullPath ;

  PixelModuleName module(dacsettings_[0].getROCName().rocname());
  fullPath << path << "/dacsettings_" << module.modulename() << ".xml" ;
  std::cout << mthn << "Writing to: |" << fullPath.str()  << "|" << std::endl ;
  

  std::ofstream outstream(fullPath.str().c_str()) ;
  
  out << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"		         	 << std::endl ;
  out << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 	         	 << std::endl ;
  out << " <HEADER>"								         	 << std::endl ;
  out << "  <TYPE>"								         	 << std::endl ;
  out << "   <EXTENSION_TABLE_NAME>ROC_DAC_SETTINGS_COL</EXTENSION_TABLE_NAME>" 	 	 << std::endl ;
  out << "   <NAME>ROC DAC Settings Col</NAME>" 					 	 << std::endl ;
  out << "  </TYPE>"								         	 << std::endl ;
  out << "  <RUN>"								         	 << std::endl ;
  out << "   <RUN_TYPE>test</RUN_TYPE>" 					         	 << std::endl ;
  out << "   <RUN_NUMBER>1</RUN_NUMBER>"					         	 << std::endl ;
  out << "   <RUN_BEGIN_TIMESTAMP>" << PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << std::endl ;
  out << "   <COMMENT_DESCRIPTION>Test of DAC Settings xml</COMMENT_DESCRIPTION>"	 	 << std::endl ;
  out << "   <LOCATION>CERN TAC</LOCATION>"					         	 << std::endl ;
  out << "   <CREATED_BY_USER>Dario Menasce</CREATED_BY_USER>"  		         	 << std::endl ;
  out << "  </RUN>"								         	 << std::endl ;
  out << " </HEADER>"								         	 << std::endl ;
  out << ""									         	 << std::endl ;
  out << " <DATA_SET>"                                                             		 << std::endl ;
  out << "  <VERSION>" << version << "</VERSION>"                                  		 << std::endl ;
  out << " "                                                                       		 << std::endl ;
  out << "  <PART>"                                                                		 << std::endl ;
  out << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"                              		 << std::endl ;
  out << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"                           		 << std::endl ;
  out << "  </PART>"                                                               		 << std::endl ;
  out << " "                                                                       		 << std::endl ;

  for(unsigned int i=0;i<dacsettings_.size();i++){
//    dacsettings_[i].writeXML(out, key, version, path);
  }

  out << " </DATA_SET>"                                                                          << std::endl ;
  out << "</ROOT>"                                                                               << std::endl ;

  out.close() ;
  std::cout << mthn << "Data written"                                                            << std::endl ;
}
*/

//=============================================================================================
void PixelDACSettings::generateConfiguration(PixelFECConfigInterface* pixelFEC,
					     PixelNameTranslation* trans, PixelDetectorConfig* detconfig, bool HVon) const{

  bool bufferData=true; 

  std::vector<unsigned int> dacs;

  //pixelFEC->fecDebug(1);  //FIXME someday maybe don't want to take the time

  for(unsigned int i=0;i<dacsettings_.size();i++){  // loop over ROCs

    bool disableRoc = rocIsDisabled(detconfig, dacsettings_[i].getROCName());

    dacsettings_[i].getDACs(dacs);

    PixelHdwAddress theROC=*(trans->getHdwAddress(dacsettings_[i].getROCName()));

    //Need to set readout speed (40MHz) and Vcal range (0-1800 mV) and enable the chip



    // This is not needed. The ControlReg is programmed in setAllDAC(). d.k. 21.01.11
//     pixelFEC->progdac(theROC.mfec(),
// 		      theROC.mfecchannel(),
// 		      theROC.hubaddress(),
// 		      theROC.portaddress(),
// 		      theROC.rocid(),
// 		      0xfd,
// 		      controlreg,
// 		      bufferData);

    if (!HVon || disableRoc)    dacs[11]=0; //set Vcthr DAC to 0 (Vcthr is DAC 12=11+1)
    //    std::cout<<" ; setting VcThr to "<<dacs[11]<<std::endl; //for debugging
    pixelFEC->setAllDAC(theROC,dacs,bufferData);

    // start with no pixels on for calibration
    pixelFEC->clrcal(theROC.mfec(), 
		     theROC.mfecchannel(), 
		     theROC.hubaddress(), 
		     theROC.portaddress(),  
		     theROC.rocid(),
		     bufferData);
    
    const bool kmeKLUDGE=false;
    if(kmeKLUDGE) //enable one pixel per ROC for calibration (all the time!)
      {
	unsigned int col=0;
	//for(unsigned int col=0;col<52;col+=50) //try 0, 50
	  {
	    pixelFEC->calpix(theROC.mfec(),
			     theROC.mfecchannel(),
			     theROC.hubaddress(),
			     theROC.portaddress(),
			     theROC.rocid(),
			     col, //column
			     0, //row
			     1, //caldata
			     bufferData);
	  }
      }

    // enable all the double columns
    for(int dcol=0;dcol<26;dcol++){
      pixelFEC->dcolenable(theROC.mfec(),
			   theROC.mfecchannel(),
			   theROC.hubaddress(),
			   theROC.portaddress(),
			   theROC.rocid(),
			   dcol,
			   1,
			   bufferData);
    }

    if (!HVon || disableRoc) { //HV off
      int controlreg=dacsettings_[i].getControlRegister();
      //      std::cout << "[PixelDACSettings::generateConfiguration] HV off! ROC control reg to be set to: " <<  (controlreg|0x2) <<std::endl;
      pixelFEC->progdac(theROC.mfec(),
			theROC.mfecchannel(),
			theROC.hubaddress(),
			theROC.portaddress(),
			theROC.rocid(),
			0xfd,
			controlreg | 0x2, //=010 in binary. should disable the chip
			bufferData);
    } //HV off


    // Now program (again) the temperature register to make sure it is the last one
    // and appears in the LastDAC
    if(readTemperatures) { 
      //     std::cout<<"ROC="<<dacsettings_[i].getROCName()<<" ; VcThr set to "<<dacs[11]
      //       << " ROC control reg to be set to: " <<  dacs[28] <<" LastDAC=Temp"<<std::endl;
      if( (theROC.mfec()==1) && (theROC.mfecchannel()==1) &&  (theROC.hubaddress()==0) && 
	  (theROC.portaddress()==0) &&  (theROC.rocid()) ) 
	std::cout<<"ROC="<<dacsettings_[i].getROCName()<< " ROC control reg to be set to: " 
		 <<  dacs[28] <<" LastDAC=Temp "<<temperatureReg<<std::endl;
      //int temperatureReg = dacs[26];  // overwrite with the number from DB
      pixelFEC->progdac(theROC.mfec(),
			theROC.mfecchannel(),
			theROC.hubaddress(),
			theROC.portaddress(),
			theROC.rocid(),
			0x1B,
			temperatureReg,bufferData);
    } else {
      //      std::cout<<"ROC="<<dacsettings_[i].getROCName()<<" ; VcThr set to "<<dacs[11]
      //	       << " ROC control reg to be set to: " <<  dacs[28] <<" LastDAC=Vcal"<<std::endl;
      if( (theROC.mfec()==1) && (theROC.mfecchannel()==1) &&  (theROC.hubaddress()==0) && 
	  (theROC.portaddress()==0) &&  (theROC.rocid()) )
	std::cout<<"ROC="<<dacsettings_[i].getROCName()
		 << " ROC control reg to be set to: " <<  dacs[28] <<" LastDAC=Vcal"<<std::endl;
      // VCAL
      pixelFEC->progdac(theROC.mfec(),
			theROC.mfecchannel(),
			theROC.hubaddress(),
			theROC.portaddress(),
			theROC.rocid(),
			0x19,
			200,
			bufferData);
    }
    
  } // end ROC loop 
  
  if (bufferData) {  // Send data to the FEC
    pixelFEC->qbufsend();
  }
  
}

void PixelDACSettings::setVcthrDisable(PixelFECConfigInterface* pixelFEC, PixelNameTranslation* trans ) const {
  //the point here is to set Vcthr to 0
  //then disable the ROCs

  //note -- no need to look at the detconfig here, because we're going to disable everything no matter what

  bool bufferData=true;

  std::vector<unsigned int> dacs;

  for(unsigned int i=0;i<dacsettings_.size();i++){ //loop over the ROCs

    dacsettings_[i].getDACs(dacs);
    int controlreg=dacsettings_[i].getControlRegister();
    
    PixelHdwAddress theROC=*(trans->getHdwAddress(dacsettings_[i].getROCName()));

    //std::cout<<"disabling ROC="<<dacsettings_[i].getROCName()<<std::endl;
    pixelFEC->progdac(theROC.mfec(),
		      theROC.mfecchannel(),
		      theROC.hubaddress(),
		      theROC.portaddress(),
		      theROC.rocid(),
		      12, //12 == Vcthr
		      0, //set Vcthr to 0
                      bufferData);

    //this should disable the roc
    pixelFEC->progdac(theROC.mfec(),
		      theROC.mfecchannel(),
		      theROC.hubaddress(),
		      theROC.portaddress(),
		      theROC.rocid(),
		      0xfd,
		      controlreg | 0x2,
		      bufferData);

    // Now program (again) the temperature register to make sure it is the last one
    // and appears in the LastDAC
    if(readTemperatures) {
      //int temperatureReg = dacs[26];  // value from DB
      pixelFEC->progdac(theROC.mfec(),
			theROC.mfecchannel(),
			theROC.hubaddress(),
			theROC.portaddress(),
			theROC.rocid(),
			0x1B,
			temperatureReg,
			bufferData);
    } else {
    // VCAL
      pixelFEC->progdac(theROC.mfec(),
			theROC.mfecchannel(),
			theROC.hubaddress(),
			theROC.portaddress(),
			theROC.rocid(),
			0x19,
			200,
			bufferData);
    }
    
  }

  if (bufferData) { //just copying the way it was done in the existing method
    pixelFEC->qbufsend();
  }
}

void PixelDACSettings::setVcthrEnable(PixelFECConfigInterface* pixelFEC, PixelNameTranslation* trans, PixelDetectorConfig* detconfig) const {
  //the point here is to set Vcthr to the nominal values
  //then enable the ROCs

  bool bufferData=true;

  std::vector<unsigned int> dacs;

  for(unsigned int i=0;i<dacsettings_.size();i++){ //loop over the ROCs

    bool disableRoc = rocIsDisabled(detconfig, dacsettings_[i].getROCName()); //don't enable ROCs that are disabled in the detconfig


    //std::cout<<"ROC="<<dacsettings_[i].getROCName()<<" ; VcThr set to "<<dacs[11]
    //	     << " ; ROC control reg to be set to: " <<  controlreg <<std::endl;

    if (!disableRoc) {  // Disable

      dacsettings_[i].getDACs(dacs);
      int controlreg=dacsettings_[i].getControlRegister();
      
      PixelHdwAddress theROC=*(trans->getHdwAddress(dacsettings_[i].getROCName()));

      pixelFEC->progdac(theROC.mfec(),
			theROC.mfecchannel(),
			theROC.hubaddress(),
			theROC.portaddress(),
			theROC.rocid(),
			12, //12 == Vcthr
			dacs[11],
			bufferData);
      
      //enable the roc (assuming controlreg was set for the roc to be enabled)
      
      pixelFEC->progdac(theROC.mfec(),
			theROC.mfecchannel(),
			theROC.hubaddress(),
			theROC.portaddress(),
			theROC.rocid(),
			0xfd,
			controlreg,
			bufferData);

      // Now program (again) the temperature register to make sure it is the last one
      // and appears in the LastDAC
      if(readTemperatures) {
	//int temperatureReg = dacs[26];  // value from DB
	pixelFEC->progdac(theROC.mfec(),
			  theROC.mfecchannel(),
			  theROC.hubaddress(),
			  theROC.portaddress(),
			  theROC.rocid(),
			  0x1B,
			  temperatureReg,
			  bufferData);
      } else {
	// VCAL
	pixelFEC->progdac(theROC.mfec(),
			  theROC.mfecchannel(),
			  theROC.hubaddress(),
			  theROC.portaddress(),
			  theROC.rocid(),
			  0x19,
			  200,
			  bufferData);
      }

    }  // end disable 
    
  } // loop over ROCs


  if (bufferData) {  // Send data to FEC
    pixelFEC->qbufsend();
  }

}

bool PixelDACSettings::rocIsDisabled(const PixelDetectorConfig* detconfig, const PixelROCName rocname) const {

  const std::map<PixelROCName, PixelROCStatus> & roclist=detconfig->getROCsList();
  const std::map<PixelROCName, PixelROCStatus>::const_iterator iroc = roclist.find(rocname);
  assert(iroc != roclist.end());    // the roc name should always be found
  PixelROCStatus thisROCstatus = iroc->second;

  return thisROCstatus.get(PixelROCStatus::noAnalogSignal);

}

std::ostream& operator<<(std::ostream& s, const PixelDACSettings& dacs){

  s << dacs.getDACSettings(0) <<std::endl; 

  return s;

}

