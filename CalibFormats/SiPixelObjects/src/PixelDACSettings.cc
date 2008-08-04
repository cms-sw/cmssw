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
#include <map>
#include <sstream>
#include <sys/time.h>

using namespace pos;

PixelDACSettings::PixelDACSettings(std::string filename):
  PixelConfigBase("","",""){


  if (filename[filename.size()-1]=='t'){

    std::ifstream in(filename.c_str());

    if (!in.good()){
      std::cout << "Could not open:"<<filename<<std::endl;
      assert(0);
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
      std::cout << "Could not open:"<<filename<<std::endl;
      assert(0);
    }
    else {
      std::cout << "Opened:"<<filename<<std::endl;
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

    //std::cout << "READ ROC name:"<<s1<<std::endl;

    dacsettings_.clear();


    while (!in.eof()){

      //std::cout << "PixelDACSettings::PixelDACSettings read s1:"<<s1<<std::endl;

      PixelROCName rocid(s1);

      //td::cout << "PixelDACSettings::PixelDACSettings read rocid:"<<rocid<<std::endl;
	    
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


  //std::cout << "Read dac settings for "<<dacsettings_.size()<<" ROCs"<<std::endl;


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
  //   colNames.push_back("CONFIG_KEY_ID");
  //   colNames.push_back("CONFIG_KEY");
  //   colNames.push_back("VERSION");
  //   colNames.push_back("KIND_OF_COND");
  colNames.push_back("ROC_NAME");
  //   colNames.push_back("HUB_ADDRS");
  //   colNames.push_back("PORT_NUMBER");
  //   colNames.push_back("I2C_ADDR");
  //   colNames.push_back("GEOM_ROC_NUM");
  colNames.push_back("VDD");
  colNames.push_back("VANA");
  colNames.push_back("VSF");
  colNames.push_back("VCOMP");
  colNames.push_back("VLEAK");
  colNames.push_back("VRGPR");
  colNames.push_back("VWLLPR");
  colNames.push_back("VRGSH");
  colNames.push_back("VWLLSH");
  colNames.push_back("VHLDDEL");
  colNames.push_back("VTRIM");
  colNames.push_back("VCTHR");
  colNames.push_back("VIBIAS_BUS");
  colNames.push_back("VIBIAS_SF");
  colNames.push_back("VOFFSETOP");
  colNames.push_back("VBIASOP");
  colNames.push_back("VOFFSETRO");
  colNames.push_back("VION");
  colNames.push_back("VIBIAS_PH");
  colNames.push_back("VIBIAS_DAC");
  colNames.push_back("VIBIAS_ROC");
  colNames.push_back("VICOLOR");
  colNames.push_back("VNPIX");
  colNames.push_back("VSUMCOL");
  colNames.push_back("VCAL");
  colNames.push_back("CALDEL");
  colNames.push_back("TEMPRANGE");
  colNames.push_back("WBC");
  colNames.push_back("CHIPCONTREG");

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
    // +1 to get rid of the unwanted ROC_NAME...
    PixelROCName rocid(tableMat[r][colM["ROC_NAME"]]);
    PixelROCDACSettings tmp(rocid);
//     std::map<std::string, unsigned int> tmpDACs ;
    std::ostringstream dacs("") ;
    for(unsigned int n=skipColumns+1; n<colNames.size(); n++)
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

  PixelModuleName module(dacsettings_[0].getROCName().rocname());

  std::string filename=dir+"/ROC_DAC_module_"+module.modulename()+".dat";
  std::cout << "[PixelDACSettings::writeASCII()] Writing to file " << filename << std::endl ; 
  std::ofstream out(filename.c_str());
  
  for(unsigned int i=0;i<dacsettings_.size();i++){
    dacsettings_[i].writeASCII(out);
  }

}

//=============================================================================================
void PixelDACSettings::writeXMLHeader(pos::PixelConfigKey key, int version, std::string path, std::ofstream *out) const {
  std::string mthn = "[PixelDACSettings::writeXMLHeader()]\t\t\t    " ;
  std::stringstream fullPath ;

  fullPath << path << "/dacsettings.xml" ;
  std::cout << mthn << "Writing to: |" << fullPath.str()  << "|" << std::endl ;

  out->open(fullPath.str().c_str()) ;
  
  *out << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"		         	  << std::endl ;
  *out << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 	         	  << std::endl ;
  *out << " <HEADER>"										  << std::endl ;
  *out << "  <TYPE>"										  << std::endl ;
  *out << "   <EXTENSION_TABLE_NAME>ROC_DAC_SETTINGS_COL</EXTENSION_TABLE_NAME>"		  << std::endl ;
  *out << "   <NAME>ROC DAC Settings Col</NAME>"						  << std::endl ;
  *out << "  </TYPE>"										  << std::endl ;
  *out << "  <RUN>"										  << std::endl ;
  *out << "   <RUN_TYPE>test</RUN_TYPE>"							  << std::endl ;
  *out << "   <RUN_NUMBER>1</RUN_NUMBER>"							  << std::endl ;
  *out << "   <RUN_BEGIN_TIMESTAMP>" << PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << std::endl ;
  *out << "   <COMMENT_DESCRIPTION>Test of DAC Settings xml</COMMENT_DESCRIPTION>"		  << std::endl ;
  *out << "   <LOCATION>CERN TAC</LOCATION>"							  << std::endl ;
  *out << "   <INITIATED_BY_USER>Dario Menasce</INITIATED_BY_USER>"				  << std::endl ;
  *out << "  </RUN>"										  << std::endl ;
  *out << " </HEADER>"  									  << std::endl ;
  *out << ""											  << std::endl ;
  *out << " <DATA_SET>" 									  << std::endl ;
  *out << "  <VERSION>" << version << "</VERSION>"						  << std::endl ;
  *out << " "											  << std::endl ;
  *out << "  <PART>"										  << std::endl ;
  *out << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"  					  << std::endl ;
  *out << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"					  << std::endl ;
  *out << "  </PART>"										  << std::endl ;
  *out << " "                                                                       		  << std::endl ;

  std::cout << mthn << "Header written" << std::endl ;
}

//=============================================================================================
void PixelDACSettings::writeXML( std::ofstream *out) const {
  std::string mthn = "[PixelDACSettings::writeXML()]\t\t\t    " ;

  for(unsigned int i=0;i<dacsettings_.size();i++){
    dacsettings_[i].writeXML(out);
  }
}

//=============================================================================================
void PixelDACSettings::writeXMLTrailer(std::ofstream *out) const {
  std::string mthn = "[PixelDACSettings::writeXMLTrailer()]\t\t\t    " ;

  *out << " </DATA_SET>"              << std::endl ;
  *out << "</ROOT>"                   << std::endl ;

  std::cout << mthn << "Closing input stream" << std::endl ;
  out->close() ;
  std::cout << mthn << "Data written" << std::endl ;
}

//=============================================================================================
void PixelDACSettings::writeXML(pos::PixelConfigKey key, int version, std::string path) const {
  std::string mthn = "[PixelDACSettings::writeXML()]\t\t\t    " ;
  std::stringstream fullPath ;

  PixelModuleName module(dacsettings_[0].getROCName().rocname());
  fullPath << path << "/dacsettings_" << module.modulename() << ".xml" ;
  std::cout << mthn << "Writing to: |" << fullPath.str()  << "|" << std::endl ;
  

  std::ofstream out(fullPath.str().c_str()) ;
  
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
  out << "   <INITIATED_BY_USER>Dario Menasce</INITIATED_BY_USER>"		         	 << std::endl ;
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

//=============================================================================================
void PixelDACSettings::generateConfiguration(PixelFECConfigInterface* pixelFEC,
					     PixelNameTranslation* trans) const{

  bool bufferData=true; 

  std::vector<unsigned int> dacs;

  //pixelFEC->fecDebug(1);  //FIXME someday maybe don't want to take the time

  for(unsigned int i=0;i<dacsettings_.size();i++){

    dacsettings_[i].getDACs(dacs);

    PixelHdwAddress theROC=*(trans->getHdwAddress(dacsettings_[i].getROCName()));

    //Need to set readout speed (40MHz) and Vcal range (0-1800 mV) and enable the chip

    int controlreg=dacsettings_[i].getControlRegister();
    //std::cout << "ROC control reg to be set to: " <<  controlreg <<std::endl;

    pixelFEC->progdac(theROC.mfec(),
		      theROC.mfecchannel(),
		      theROC.hubaddress(),
		      theROC.portaddress(),
		      theROC.rocid(),
		      0xfd,
		      controlreg,
		      bufferData);

    pixelFEC->setAllDAC(theROC,dacs,bufferData);

    // start with no pixels on for calibration
    pixelFEC->clrcal(theROC.mfec(), 
		     theROC.mfecchannel(), 
		     theROC.hubaddress(), 
		     theROC.portaddress(),  
		     theROC.rocid(),
		     bufferData);

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
  }

  if (bufferData) {
    pixelFEC->qbufsend();
  }

} 


std::ostream& operator<<(std::ostream& s, const PixelDACSettings& dacs){

  s << dacs.getDACSettings(0) <<std::endl; 

  return s;

}

