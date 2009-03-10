//
// Implementation of the detector configuration
//
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelDetectorConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <ios>
#include <assert.h>
#include <stdio.h>

using namespace std;
using namespace pos;


PixelDetectorConfig::PixelDetectorConfig(std::vector< std::vector < std::string> > &tableMat):PixelConfigBase("","",""){

  std::vector< std::string > ins = tableMat[0];
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  /*
    CONFIG_KEY_ID                             NOT NULL NUMBER(38)
    CONFG_KEY                                 NOT NULL VARCHAR2(80)
    VERSION                                            VARCHAR2(40)
    KIND_OF_COND                              NOT NULL VARCHAR2(40)
    SERIAL_NUMBER                                      VARCHAR2(40)
    ROC_NAME                                  NOT NULL VARCHAR2(200)
    PANEL_NAME                                NOT NULL VARCHAR2(200)
    ROC_STATUS                                NOT NULL VARCHAR2(200)
  */
  colNames.push_back("CONFIG_KEY_ID");//0
  colNames.push_back("CONFIG_KEY");   //1
  colNames.push_back("VERSION");      //2
  colNames.push_back("KIND_OF_COND"); //3
  colNames.push_back("SERIAL_NUMBER");//4
  colNames.push_back("ROC_NAME");     //5
  colNames.push_back("PANEL_NAME");   //6
  colNames.push_back("ROC_STATUS");   //7

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
      std::cerr << "[PixelDetectorConfig::PixelDetectorConfig()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
      assert(0);
    }
  }
  

  modules_.clear();
  rocs_.clear() ;
  std::string module= "";
  for(unsigned int r = 1 ; r < tableMat.size() ; r++)
    {    //Goes to every row of the Matrix
      PixelROCName roc(tableMat[r][colM["ROC_NAME"]]) ; // see DACSettings!!!!
      PixelROCStatus rocstatus;
      std::string status = tableMat[r][colM["ROC_STATUS"]];
      // The following is due to the fact that enabled ROCs are 
      // labelled as ON in the DataBase, but have NO label in 
      // configuration files!!!
      if(status.find("ON") != string::npos)
	{
	  status = "" ;
	}
      if (status!=""){
	rocstatus.set(status);
      }
      rocs_[roc]=rocstatus;
      if (!rocstatus.get(PixelROCStatus::noInit)){

	PixelModuleName module(tableMat[r][colM["PANEL_NAME"]]);
	if (!containsModule(module)) {
	  modules_.push_back(module);
	}
      }
  }//end for r

  std::cout<<"Number of Modules in Detector Configuration Class:"<<getNModules()<<std::endl;

}//end constructor

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PixelDetectorConfig::PixelDetectorConfig(std::string filename):
  PixelConfigBase("","",""){

  if (filename[filename.size()-1]=='t'){

    std::ifstream in(filename.c_str());

    if (!in.good()){
      std::cout << "[PixelDetectorConfig::PixelDetectorConfig()]\t\t    Could not open: "<<filename<<std::endl;
      assert(0);
    }
    else {
      std::cout << "[PixelDetectorConfig::PixelDetectorConfig()]\t\t    Opened: "<<filename<<std::endl;
    }
        
    if (in.eof()){
      std::cout << "[PixelDetectorConfig::PixelDetectorConfig()]\t\t    EOF before reading anything!"<<std::endl;
      ::abort();
    }

        
    modules_.clear();
    rocs_.clear() ;
        
    std::string module;
        
    in >> module;

    if (module=="Rocs:") {
      std::cout << "[PixelDetectorConfig::PixelDetectorConfig()]\t\t    New format of detconfig"<<std::endl;
      //new format with list of ROCs.
      std::string rocname;
      in >> rocname;
      while (!in.eof()){
        //cout << "Read rocname:"<<rocname<<endl;
        PixelROCName roc(rocname);
        std::string line;
        getline(in,line);
        //cout << "Read line:'"<<line<<"'"<<endl;
        istringstream instring(line);
        PixelROCStatus rocstatus;
        std::string status;
        while (!instring.eof()) {
          instring >> status;
//        cout << "Read status:"<<status<<endl;
          if (status!=""){
            rocstatus.set(status);
          }
        }
        rocs_[roc]=rocstatus;
        if (!rocstatus.get(PixelROCStatus::noInit)){
          PixelModuleName module(rocname);
          if (!containsModule(module)) {
            modules_.push_back(module);
          }
        }
        in >> rocname;
      }
      return;
    }
        

    //std::cout << "Read module:"<<module<<std::endl;

    if (in.eof()) std::cout << "[PixelDetectorConfig::PixelDetectorConfig()]\t\t    EOF after reading first module name"
			    << std::endl;

    std::cout << "[PixelDetectorConfig::PixelDetectorConfig()]\t\t    Old format of detconfig"<<std::endl;
    while (!in.eof()){

      //std::cout << "Read module:"<<module<<std::endl;

      PixelModuleName moduleName(module);

      modules_.push_back(moduleName);
            
      in >> module;
            
      assert(modules_.size()<10000);
            
    }
        
    in.close();

  }
  else{

    assert(0);

    /*
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

      //std::cout << "PixelDetectorConfig::PixelDetectorConfig read s1:"<<s1<<std::endl;

      PixelROCName rocid(s1);

      //td::cout << "PixelDetectorConfig::PixelDetectorConfig read rocid:"<<rocid<<std::endl;
            
      PixelROCDetectorConfig tmp;
      
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

    */

  }


  //std::cout << "Read dac settings for "<<dacsettings_.size()<<" ROCs"<<std::endl;


}

unsigned int PixelDetectorConfig::getNModules() const {

  return modules_.size();

}
 
PixelModuleName PixelDetectorConfig::getModule(unsigned int i) const {

  return modules_[i];

}

std::set <unsigned int> PixelDetectorConfig::getFEDs(PixelNameTranslation* translation) const 
{

  std::set <unsigned int> feds;
  assert(modules_.size()!=0);
  std::vector<PixelModuleName>::const_iterator imodule=modules_.begin();
        
  for (;imodule!=modules_.end();++imodule) {
  
                std::set<PixelChannel> channelsOnThisModule = translation->getChannelsOnModule(*imodule);
                for ( std::set<PixelChannel>::const_iterator channelsOnThisModule_itr = channelsOnThisModule.begin(); channelsOnThisModule_itr != channelsOnThisModule.end(); channelsOnThisModule_itr++ )
                {
                        const PixelHdwAddress& channel_hdwaddress = translation->getHdwAddress(*channelsOnThisModule_itr);
                        unsigned int fednumber=channel_hdwaddress.fednumber();
                        feds.insert(fednumber);
                }

  }
        
  return feds;
}


// Returns the FED numbers and channels within each FED that are used
std::map <unsigned int, std::set<unsigned int> > PixelDetectorConfig::getFEDsAndChannels(PixelNameTranslation* translation) const
{
  //      FED Number                channels

  std::map <unsigned int, std::set<unsigned int> > fedsChannels;
  assert(modules_.size()!=0);
  std::vector<PixelModuleName>::const_iterator imodule=modules_.begin();

  for (;imodule!=modules_.end();++imodule) {
  
                std::set<PixelChannel> channelsOnThisModule = translation->getChannelsOnModule(*imodule);
                for ( std::set<PixelChannel>::const_iterator channelsOnThisModule_itr = channelsOnThisModule.begin(); channelsOnThisModule_itr != channelsOnThisModule.end(); channelsOnThisModule_itr++ )
                {
                        const PixelHdwAddress& channel_hdwaddress = translation->getHdwAddress(*channelsOnThisModule_itr);
                        unsigned int fednumber=channel_hdwaddress.fednumber();
                        unsigned int fedchannel=channel_hdwaddress.fedchannel();
                        fedsChannels[fednumber].insert(fedchannel);
                }
  
  }

  return fedsChannels;
}
 
bool PixelDetectorConfig::containsModule(const PixelModuleName& moduleToFind) const
{
  for ( std::vector<PixelModuleName>::const_iterator modules_itr = modules_.begin(); modules_itr != modules_.end(); modules_itr++ )
    {
      if ( *modules_itr == moduleToFind ) return true;
    }
  return false;
}

// modified by MR on 11-01-2008 15:06:51
void PixelDetectorConfig::writeASCII(std::string dir) const {

  if (dir!="") dir+="/";
  std::string filename=dir+"detectconfig.dat";

  std::ofstream out(filename.c_str(), std::ios_base::out) ;
  if(!out) {
    std::cout << "[PixelDetectorConfig::writeASCII()]\t\t    Could not open file " << filename << " for write" << std::endl ;
    exit(1);
  }


  if(rocs_.size() == 0) 
    {
      std::vector<PixelModuleName>::const_iterator imodule=modules_.begin();
      
      for (;imodule!=modules_.end();++imodule) 
        {
          out << *imodule << std::endl;
        }
    } 
  else 
    {
      out << "Rocs:" << endl ;
      std::map<PixelROCName, PixelROCStatus>::const_iterator irocs = rocs_.begin();
      for(; irocs != rocs_.end() ; irocs++)
        {
          out << (irocs->first).rocname() << " " << (irocs->second).statusName() << endl ;
        }
    }
  
  out.close();

}

//=============================================================================================
void PixelDetectorConfig::writeXML(pos::PixelConfigKey key, int version, std::string path) const {
  std::string mthn = "[PixelDetectorConfig::writeXML()]\t\t\t    " ;
  std::stringstream fullPath ;

  fullPath << path << "/detectorconfig.xml" ;
  cout << mthn << "Writing to: " << fullPath.str() << endl ;
  
  std::ofstream out(fullPath.str().c_str()) ;
  
  out << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"			 	      << endl ;
  out << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 		 	      << endl ;
  out << " <HEADER>"								         	      << endl ;
  out << "  <TYPE>"								         	      << endl ;
  out << "   <EXTENSION_TABLE_NAME>FPIX_DETECTOR_CONFIG</EXTENSION_TABLE_NAME>"          	      << endl ;
  out << "   <NAME>FPix Detector Configuration</NAME>"				         	      << endl ;
  out << "  </TYPE>"								         	      << endl ;
  out << "  <RUN>"								         	      << endl ;
  out << "   <RUN_TYPE>test</RUN_TYPE>" 		                                 	      << endl ;
  out << "   <RUN_NUMBER>1</RUN_NUMBER>"					         	      << endl ;
  out << "   <RUN_BEGIN_TIMESTAMP>" << pos::PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << endl ;
  out << "   <COMMENT_DESCRIPTION>Test of DetectorConfig xml</COMMENT_DESCRIPTION>"      	      << endl ;
  out << "   <LOCATION>CERN TAC</LOCATION>"					         	      << endl ;
  out << "   <INITIATED_BY_USER>Dario Menasce</INITIATED_BY_USER>"			 	      << endl ;
  out << "  </RUN>"								         	      << endl ;
  out << " </HEADER>"								         	      << endl ;
  out << ""										 	      << endl ;
  out << " <DATA_SET>"  						    	 	      	      << endl ;
  out << "  <VERSION>" << version << "</VERSION>"			    	 	      	      << endl ;
  out << "  <PART>"							    	 	      	      << endl ;
  out << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"   	 	                      	      << endl ;
  out << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"		    	 	      	      << endl ;
  out << "  </PART>"							    	 	      	      << endl ;
  out << ""							    	 	      	              << endl ;

  if(rocs_.size() == 0) 
    {
      std::vector<PixelModuleName>::const_iterator imodule=modules_.begin();
      
      // This needs to be fixed: given a module name, actually loop over ROCs to write the XML data
      for (;imodule!=modules_.end();++imodule) 
        {
          out << "  <DATA>"							    	 	      << endl ;
//---->          out << "   <ROC_NAME>" << (irocs->first).rocname() << "</ROC_NAME>"   	 	      << endl ;
          out << "   <ROC_STATUS>on</ROC_STATUS>"	                                 	      << endl ;
          out << "  </DATA>"							    	 	      << endl ;
          out << " "                                                                     	      << endl ;
        }
    } 
  else 
    {
      std::map<PixelROCName, PixelROCStatus>::const_iterator irocs = rocs_.begin();
      for(; irocs != rocs_.end() ; irocs++)
        {
          std::string sts = (irocs->second).statusName() ;
          if( sts == "" ) {sts = "on" ;}
          out << "  <DATA>"							    	 	      << endl ;
          out << "   <ROC_NAME>"   << (irocs->first).rocname() << "</ROC_NAME>"		     	      << endl ;
          out << "   <ROC_STATUS>" << sts << "</ROC_STATUS>"	                         	      << endl ;
          out << "  </DATA>"							    	 	      << endl ;
          out << " "                                                                     	      << endl ;
        }
    }
  out << " </DATA_SET>" 						    	 	              << endl ;
  out << "</ROOT> "								         	      << endl ;
  out.close() ;
  assert(0) ;
}

//=============================================================================================
void PixelDetectorConfig::addROC(   PixelROCName &theROC)  // Added by Dario (March 3, 2008)
{
 std::string mthn = "[PixelDetectorConfig::addROC()]\t\t\t\t" ;
 std::map<PixelROCName, PixelROCStatus>::iterator theROCIt = rocs_.find(theROC) ;
 if( theROCIt == rocs_.end() ) // if theROC was not there, add it and turn it on
 {
  PixelROCStatus  theStatus ;
  theStatus.reset() ;
  rocs_[theROC] = theStatus ; 
//  cout << mthn << "Non existing ROC (" << theROC.rocname() << "): adding it"  << endl ;  
 } else {
  theROCIt->second.reset() ;  // otherwise just turn it on by resetting it to zero
//  cout << mthn << "Already existing ROC (" << theROC.rocname() << "): switching it on"  << endl ;  
 }
}

//=============================================================================================
void PixelDetectorConfig::addROC(   PixelROCName &theROC, string statusLabel)  // modified by MR on 14-05-2008 11:29:51
{
 std::string mthn = "[PixelDetectorConfig::addROC()]\t\t\t\t" ;
 std::map<PixelROCName, PixelROCStatus>::iterator theROCIt = rocs_.find(theROC) ;
 if( theROCIt == rocs_.end() ) // if theROC was not there, add it and turn it on
 {
  PixelROCStatus  theStatus ;
  theStatus.set(statusLabel) ;
  theStatus.reset() ;
  rocs_[theROC] = theStatus ; 
//  cout << mthn << "Non existing ROC (" << theROC.rocname() << "): adding it"  << endl ;  
 } else {
  theROCIt->second.set(statusLabel) ;  // otherwise just turn it on by resetting it to zero
//  cout << mthn << "Already existing ROC (" << theROC.rocname() << "): switching it on"  << endl ;  
 }
}

//=============================================================================================
void PixelDetectorConfig::removeROC(PixelROCName &theROC)  // Added by Dario (March 3, 2008)
{
 std::string mthn = "[PixelDetectorConfig::removeROC()]\t\t\t\t" ;
 std::map<PixelROCName, PixelROCStatus>::iterator theROCIt = rocs_.find(theROC) ;
 if( theROCIt != rocs_.end() ) // if theROC was there remove it, otherwise ignore
 {
  theROCIt->second.set("noInit") ;  
//  cout << mthn << "Already existing ROC (" << theROC.rocname() << "): switching it off"  << endl ;  
 } else {
  PixelROCStatus  theStatus ;
  theStatus.set("noInit") ;
  rocs_[theROC] = theStatus ; 
//  cout << mthn << "ROC " << theROC.rocname() << " was not individually declared in the file: declare and switch off"  << endl ;  
 }
}

//std::ostream& operator<<(std::ostream& s, const PixelDetectorConfig& dacs){
//
//  s << dacs.getDetectorConfig(0) <<std::endl; 
//
//  return s;
//
//}

