#ifndef PixelConfigFile_h
#define PixelConfigFile_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelConfigFile.h
*   \brief This class implements..
*
*   OK, first this is not a DB; this class will try to
*   define an interface to accessing the configuration data.
*/

#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigAlias.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigList.h"
#include "CalibFormats/SiPixelObjects/interface/PixelAliasList.h"
#include "CalibFormats/SiPixelObjects/interface/PixelVersionAlias.h"
#include "CalibFormats/SiPixelObjects/interface/PixelCalibBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigKey.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTrimBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTrimAllPixels.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaskAllPixels.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDACSettings.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTBMSettings.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDetectorConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelLowVoltageMap.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaxVsf.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFEDCard.h"
#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"
#include "CalibFormats/SiPixelObjects/interface/PixelPortCardConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelPortcardMap.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDelay25Calib.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTKFECConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFEDConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTTCciConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelLTCConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFEDTestDAC.h"
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <sys/stat.h>
#include <sys/types.h>

namespace pos{
/*! \class PixelConfigFile PixelConfigFile.h "interface/PixelConfigFile.h"
*
*
*   OK, first this is not a DB; this class will try to
*   define an interface to accessing the configuration data.
*/
  class PixelConfigFile {

  public:

    static std::vector<std::pair<std::string, unsigned int> > getAliases(){
      PixelAliasList& aliases=getAlias();
      std::vector<std::pair<std::string, unsigned int> > tmp;
      for(unsigned int i=0;i<aliases.nAliases();i++){
	std::pair<std::string, unsigned int> apair(aliases.name(i),aliases.key(i));
	tmp.push_back(apair);  
      }
      return tmp;
    }

    static std::vector<std::string> getVersionAliases(std::string path){
      return getAlias().getVersionAliases(path);
    }

    static std::map<std::string, unsigned int> getAliases_map(){
      PixelAliasList& aliases=getAlias();
      std::map<std::string, unsigned int> tmp;
      for(unsigned int i=0;i<aliases.nAliases();i++){
	tmp.insert(make_pair(aliases.name(i), aliases.key(i)));
      }
      return tmp;
    }

    static unsigned int getVersion(std::string path,std::string alias){
      return getAlias().getVersion(path,alias);
    }

    static void addAlias(std::string alias, unsigned int key){
      PixelConfigAlias anAlias(alias,key);
      getAlias().insertAlias(anAlias);
      getAlias().writefile();
    }


    static void addAlias(std::string alias, unsigned int key,
			 std::vector<std::pair<std::string, std::string> > versionaliases){
      PixelConfigAlias anAlias(alias,key);
      for(unsigned int i=0;i<versionaliases.size();i++){
	anAlias.addVersionAlias(versionaliases[i].first,versionaliases[i].second);
      }
      getAlias().insertAlias(anAlias);
      getAlias().writefile();
    }

    static std::vector< std::pair< std::string, unsigned int> > getVersions(pos::PixelConfigKey key){

      PixelConfigList& configs=getConfig();
      PixelConfig& theConfig=configs[key.key()];
      return theConfig.versions();
      
    }

    static void addVersionAlias(std::string path, unsigned int version, std::string alias){

      PixelConfigList& configs=getConfig();

      PixelVersionAlias anAlias(path, version, alias);
      getAlias().insertVersionAlias(anAlias);
      getAlias().updateConfigAlias(path,version,alias,configs);
      getAlias().writefile();
      configs.writefile();
    }

    static unsigned int makeKey(std::vector<std::pair<std::string, unsigned int> > versions){

      PixelConfig config;

      for(unsigned int i=0;i<versions.size();i++){
	config.add(versions[i].first,versions[i].second);
      }

      PixelConfigList& configs=getConfig();

      unsigned int newkey=configs.add(config);

      configs.writefile();
    
      return newkey;

    }

    static PixelConfigList& getConfig(){

      static std::string directory;
      static int first=1;
    
      static PixelConfigList configs;
    
      directory=getenv("PIXELCONFIGURATIONBASE");
      std::string filename=directory+"/configurations.txt";
      if(!first)
	{
//	  std::cout << "[pos::PixelConfigFile::getConfig()] Reloading configurations.txt"<< std::endl ;
	  configs.reload(filename) ;
//	  std::cout << "[pos::PixelConfigFile::getConfig()] Size reloaded: " << configs.size() << std::endl ;
	}
      if (first) 
	{
	  first=0;
//	  std::cout << "[pos::PixelConfigFile::getConfig()] Reading configurations.txt"<< std::endl ;
	  configs.readfile(filename);
//	  std::cout << "[pos::PixelConfigFile::getConfig()] Size read: " << configs.size() << std::endl ;
	}

      return configs;

    }

    static PixelAliasList& getAlias(){

      static std::string directory;
      static int first=1;
    
      static PixelAliasList aliases;
    
      if (first) {
	first=0;
	directory=getenv("PIXELCONFIGURATIONBASE");
      
	std::string filename=directory+"/aliases.txt";

	aliases.readfile(filename);
                      
      }

      return aliases;

    }



    //Returns the path the the configuration data.
    static std::string getPath(std::string path, PixelConfigKey key){

      unsigned int theKey=key.key();
    
      assert(theKey<=getConfig().size());
    
      unsigned int last=path.find_last_of("/");
      assert(last!=std::string::npos);
    
      std::string base=path.substr(0,last);
      std::string ext=path.substr(last+1);
    
      unsigned int slashpos=base.find_last_of("/");
      if (slashpos==std::string::npos) {
	std::cout << "[pos::PixelConfigFile::getPath()]\t\t\tOn path:"                <<path               <<std::endl;
	std::cout << "[pos::PixelConfigFile::getPath()]\t\t\tRecall that you need a trailing /"            <<std::endl;
	::abort();
      }
    
      std::string dir=base.substr(slashpos+1);
    
//      std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted dir:" <<dir <<std::endl;
//      std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted base:"<<base<<std::endl;
//      std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted ext :"<<ext <<std::endl;
    
      unsigned int version;
      int err=getConfig()[theKey].find(dir,version);   
      // assert(err==0);
      if(0!=err) 
	{
	  return "";
	}
    
      std::ostringstream s1;
      s1 << version;
      std::string strversion=s1.str();

      static std::string directory;
      directory=getenv("PIXELCONFIGURATIONBASE");
    
      std::string fullpath=directory+"/"+dir+"/"+strversion+"/";
    
      return fullpath;
    }



    
    //Returns a pointer to the data found in the path with configuration key.
    template <class T>
      static void get(T* &data, std::string path, PixelConfigKey key){

      unsigned int theKey=key.key();
    
      assert(theKey<=getConfig().size());
    
      unsigned int last=path.find_last_of("/");
      assert(last!=std::string::npos);
    
      std::string base=path.substr(0,last);
      std::string ext=path.substr(last+1);
    
      unsigned int slashpos=base.find_last_of("/");
      if (slashpos==std::string::npos) {
	std::cout << "[pos::PixelConfigFile::get()]\t\t\tAsking for data of type:"<<typeid(data).name()<<std::endl;
	std::cout << "[pos::PixelConfigFile::get()]\t\t\tOn path:"                <<path               <<std::endl;
	std::cout << "[pos::PixelConfigFile::get()]\t\t\tRecall that you need a trailing /"            <<std::endl;
	::abort();
      }
    
      std::string dir=base.substr(slashpos+1);
    
//      std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted dir:" <<dir <<std::endl;
//      std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted base:"<<base<<std::endl;
//      std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted ext :"<<ext <<std::endl;
    
      unsigned int version;
      int err=getConfig()[theKey].find(dir,version);   
      // assert(err==0);
      if(0!=err) 
	{
	  data= 0; 
	  return;
	}
    
      std::ostringstream s1;
      s1 << version;
      std::string strversion=s1.str();

      static std::string directory;
      directory=getenv("PIXELCONFIGURATIONBASE");
    
      std::string fullpath=directory+"/"+dir+"/"+strversion+"/";
    
      //std::cout << "Directory for configuration data:"<<fullpath<<std::endl;
    
      if (typeid(data)==typeid(PixelTrimBase*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelTrimBase" << std::endl;
	assert(dir=="trim");
	data = (T*) new PixelTrimAllPixels(fullpath+"ROC_Trims_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelMaskBase*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelMaskBase" << std::endl;
	assert(dir=="mask");
	data = (T*) new PixelMaskAllPixels(fullpath+"ROC_Masks_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelDACSettings*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelDACSettings" << std::endl;
	assert(dir=="dac");
	data = (T*) new PixelDACSettings(fullpath+"ROC_DAC_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelTBMSettings*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelTBMSettings" << std::endl;
	assert(dir=="tbm");
	data = (T*) new PixelTBMSettings(fullpath+"TBM_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelDetectorConfig*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelDetectorConfig" << std::endl;
	assert(dir=="detconfig");
	data = (T*) new PixelDetectorConfig(fullpath+"detectconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelLowVoltageMap*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill fetch PixelLowVoltageMap" << std::endl;
	assert(dir=="lowvoltagemap");
	data = (T*) new PixelLowVoltageMap(fullpath+"lowvoltagemap.dat");
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return the PixelLowVoltageMap" << std::endl;
	return;
      }else if (typeid(data)==typeid(PixelMaxVsf*)){
	//std::cout << "Will fetch PixelMaxVsf" << std::endl;
	assert(dir=="maxvsf");
	data = (T*) new PixelMaxVsf(fullpath+"maxvsf.dat");
	//std::cout << "Will return the PixelMaxVsf" << std::endl;
	return;
      }else if (typeid(data)==typeid(PixelNameTranslation*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelDACSettings" << std::endl;
	assert(dir=="nametranslation");
	data = (T*) new PixelNameTranslation(fullpath+"translation.dat");
	return;
      }else if (typeid(data)==typeid(PixelFEDCard*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelFEDCard" << std::endl;
	assert(dir=="fedcard");
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill open:"<<fullpath+"params_fed_"+ext+".dat"<< std::endl;
	data = (T*) new PixelFEDCard(fullpath+"params_fed_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelCalibBase*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelCalibBase" << std::endl;
	assert(dir=="calib");
	std::string calibfile=fullpath+"calib.dat";
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tLooking for file " << calibfile << std::endl;
	std::ifstream calibin(calibfile.c_str());
	if(calibin.good()){
	  data = (T*) new PixelCalibConfiguration(calibfile);
	}else{
	  calibfile=fullpath+"delay25.dat";
	  //std::cout << "[pos::PixelConfigFile::get()]\t\t\tNow looking for file " << calibfile << std::endl;
	  std::ifstream delayin(calibfile.c_str());
	  if(delayin.good()){
	    data = (T*) new PixelDelay25Calib(calibfile);
	  }else{
	    calibfile=fullpath+"fedtestdac.dat";
	    //std::cout << "[pos::PixelConfigFile::get()]\t\t\tNow looking for file " << calibfile << std::endl;
	    std::ifstream delayin(calibfile.c_str());
	    if(delayin.good()){
	      data = (T*) new PixelFEDTestDAC(calibfile);
	    }else{
	      std::cout << "[pos::PixelConfigFile::get()]\t\t\tCan't find calibration file calib.dat or delay25.dat or fedtestdac.dat" << std::endl;
	      data=0;
	    }
	  }
	}
	return;
      }else if (typeid(data)==typeid(PixelTKFECConfig*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelTKFECConfig" << std::endl;
	assert(dir=="tkfecconfig");
	data = (T*) new PixelTKFECConfig(fullpath+"tkfecconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelFECConfig*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelFECConfig" << std::endl;
	assert(dir=="fecconfig");
	data = (T*) new PixelFECConfig(fullpath+"fecconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelFEDConfig*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelFEDConfig" << std::endl;
	assert(dir=="fedconfig");
	data = (T*) new PixelFEDConfig(fullpath+"fedconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelPortCardConfig*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelPortCardConfig" << std::endl;
	assert(dir=="portcard");
	data = (T*) new PixelPortCardConfig(fullpath+"portcard_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelPortcardMap*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelPortcardMap" << std::endl;
	assert(dir=="portcardmap");
	data = (T*) new PixelPortcardMap(fullpath+"portcardmap.dat");
	return;
      }else if (typeid(data)==typeid(PixelDelay25Calib*)){
	//cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelDelay25Calib" << std::endl;
	assert(dir=="portcard");
	data = (T*) new PixelDelay25Calib(fullpath+"delay25.dat");
	return;
      }else if (typeid(data)==typeid(PixelTTCciConfig*)){
	//cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelTTCciConfig" << std::endl;
	assert(dir=="ttcciconfig");
	data = (T*) new PixelTTCciConfig(fullpath+"TTCciConfiguration.txt");
	return;
      }else if (typeid(data)==typeid(PixelLTCConfig*)){
	//cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelLTCConfig" << std::endl;
	assert(dir=="ltcconfig");
	data = (T*) new PixelLTCConfig(fullpath+"LTCConfiguration.txt");
	return;
      }else{
	std::cout << "[pos::PixelConfigFile::get()]\t\t\tNo match" << std::endl;
	assert(0);
	data=0;
	return;
      }

    }

    //----- Method added by Dario (March 10, 2008)
    template <class T>
      static bool configurationDataExists(T* &data, std::string path, PixelConfigKey key){

      unsigned int theKey=key.key();
    
      assert(theKey<=getConfig().size());
    
      unsigned int last=path.find_last_of("/");
      assert(last!=std::string::npos);
    
      std::string base=path.substr(0,last);
      std::string ext=path.substr(last+1);
    
      unsigned int slashpos=base.find_last_of("/");
      if (slashpos==std::string::npos) {
	std::cout << "[pos::PixelConfigFile::configurationDataExists()]\t\t\tAsking for data of type:" << typeid(data).name() <<std::endl;
	std::cout << "[pos::PixelConfigFile::configurationDataExists()]\t\t\tOn path:" 	               << path		      <<std::endl;
	std::cout << "[pos::PixelConfigFile::configurationDataExists()]\t\t\tRecall that you need a trailing /"	              <<std::endl;
	::abort();
      }
    
      std::string dir=base.substr(slashpos+1);
    
//      std::cout << "[pos::PixelConfigFile::configurationDataExists()]\t\t\tExtracted dir:"  << dir  <<std::endl;
//      std::cout << "[pos::PixelConfigFile::configurationDataExists()]\t\t\tExtracted base:" << base <<std::endl;
//      std::cout << "[pos::PixelConfigFile::configurationDataExists()]\t\t\tExtracted ext :" << ext  <<std::endl;
    
      unsigned int version;
      int err=getConfig()[theKey].find(dir,version);   
      // assert(err==0);
      if(0!=err) 
	{
	  data= 0; 
	  return false ;
	}
    
      std::ostringstream s1;
      s1 << version;
      std::string strversion=s1.str();

      static std::string directory;
      directory=getenv("PIXELCONFIGURATIONBASE");
    
      std::string fullpath=directory+"/"+dir+"/"+strversion+"/";
    
      //std::cout << "[pos::PixelConfigFile::configurationDataExists()]\t\t\tDirectory for configuration data:"<<fullpath<<std::endl;
    
      std::string fileName ;
      if (typeid(data)==typeid(PixelTrimBase*)){
        fileName = fullpath+"ROC_Trims_module_"+ext+".dat" ;
      }else if (typeid(data)==typeid(PixelMaskBase*)){
	fileName = fullpath+"ROC_Masks_module_"+ext+".dat";
      }else if (typeid(data)==typeid(PixelDACSettings*)){
	fileName = fullpath+"ROC_DAC_module_"+ext+".dat";
      }else if (typeid(data)==typeid(PixelTBMSettings*)){
	fileName = fullpath+"TBM_module_"+ext+".dat";
      }else if (typeid(data)==typeid(PixelDetectorConfig*)){
	fileName = fullpath+"detectconfig.dat";
      }else if (typeid(data)==typeid(PixelLowVoltageMap*)){
	fileName = fullpath+"lowvoltagemap.dat";
      }else if (typeid(data)==typeid(PixelMaxVsf*)){
	fileName = fullpath+"maxvsf.dat";
      }else if (typeid(data)==typeid(PixelNameTranslation*)){
	fileName = fullpath+"translation.dat";
      }else if (typeid(data)==typeid(PixelFEDCard*)){
	fileName = fullpath+"params_fed_"+ext+".dat";
      }else if (typeid(data)==typeid(PixelTKFECConfig*)){
	fileName = fullpath+"tkfecconfig.dat";
      }else if (typeid(data)==typeid(PixelFECConfig*)){
	fileName = fullpath+"fecconfig.dat";
      }else if (typeid(data)==typeid(PixelFEDConfig*)){
	fileName = fullpath+"fedconfig.dat";
      }else if (typeid(data)==typeid(PixelPortCardConfig*)){
	fileName = fullpath+"portcard_"+ext+".dat";
      }else if (typeid(data)==typeid(PixelPortcardMap*)){
	fileName = fullpath+"portcardmap.dat";
      }else if (typeid(data)==typeid(PixelDelay25Calib*)){
	fileName = fullpath+"delay25.dat";
      }else if (typeid(data)==typeid(PixelTTCciConfig*)){
	fileName = fullpath+"TTCciConfiguration.txt";
      }else if (typeid(data)==typeid(PixelLTCConfig*)){
	fileName = fullpath+"LTCConfiguration.txt";
      }else if (typeid(data)==typeid(PixelCalibBase*)){
	assert(dir=="calib");
	std::string calibfile=fullpath+"calib.dat";
	std::ifstream calibin(calibfile.c_str());
	if(calibin.good())
	{
	 return true ;
	}else{
	 calibfile=fullpath+"delay25.dat";
	 std::ifstream delayin(calibfile.c_str());
	 if(delayin.good())
	 {
	  return true ;
	 }else{
	  calibfile=fullpath+"fedtestdac.dat";
	  std::ifstream delayin(calibfile.c_str());
	  if(delayin.good())
	  {
	   return true ;
	  }else{
	   std::cout << "[pos::PixelConfigFile::configurationDataExists()]\t\t\tCan't find calibration file calib.dat or delay25.dat or fedtestdac.dat" << std::endl;
           return false  ;
	  }
	 }
	}
      }else{
	std::cout << "[pos::PixelConfigFile::configurationDataExists()]\t\t\tNo match of class type" << std::endl;
	return false ;
      }

      std::ifstream in(fileName.c_str());
      if (!in.good()){return false ;}
      in.close() ;
      return true ;
    }
    //----- End of method added by Dario (March 10, 2008)


    //Returns a pointer to the data found in the path with configuration key.
    template <class T>
      static void get(T* &data, std::string path, unsigned int version){

      unsigned int last=path.find_last_of("/");
      assert(last!=std::string::npos);
    
      std::string base=path.substr(0,last);
      std::string ext=path.substr(last+1);
    
      unsigned int slashpos=base.find_last_of("/");
      //if (slashpos==std::string::npos) {
      //std::cout << "[pos::PixelConfigFile::get()]\t\t\tAsking for data of type:"<<typeid(data).name()<<std::endl;
      //std::cout << "[pos::PixelConfigFile::get()]\t\t\tOn path:"<<path<<std::endl;
      //std::cout << "[pos::PixelConfigFile::get()]\t\t\tRecall that you need a trailing /" << std::endl;
      //::abort();
      //}
    
      std::string dir=base.substr(slashpos+1);
    
      //std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted dir:"<<dir<<std::endl;
      //std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted base:"<<base<<std::endl;
      //std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted ext :"<<ext<<std::endl;
    
      ostringstream s1;
      s1 << version;
      std::string strversion=s1.str();

      static std::string directory;
      directory=getenv("PIXELCONFIGURATIONBASE");
    
      std::string fullpath=directory+"/"+dir+"/"+strversion+"/";
    
      //std::cout << "[pos::PixelConfigFile::get()]\t\t\tDirectory for configuration data:"<<fullpath<<std::endl;
    
      if (typeid(data)==typeid(PixelTrimBase*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelTrimBase" << std::endl;
	assert(dir=="trim");
	data = (T*) new PixelTrimAllPixels(fullpath+"ROC_Trims_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelMaskBase*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelMaskBase" << std::endl;
	assert(dir=="mask");
	data = (T*) new PixelMaskAllPixels(fullpath+"ROC_Masks_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelDACSettings*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelDACSettings" << std::endl;
	assert(dir=="dac");
	data = (T*) new PixelDACSettings(fullpath+"ROC_DAC_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelTBMSettings*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelTBMSettings" << std::endl;
	assert(dir=="tbm");
	data = (T*) new PixelTBMSettings(fullpath+"TBM_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelDetectorConfig*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelDACSettings" << std::endl;
	assert(dir=="detconfig");
	data = (T*) new PixelDetectorConfig(fullpath+"detectconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelLowVoltageMap*)){
	std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill fetch1 PixelLowVoltageMap" << std::endl;
	assert(dir=="lowvoltagemap");
	data = (T*) new PixelLowVoltageMap(fullpath+"detectconfig.dat");
	std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return1 the PixelLowVoltageMap" << std::endl;
	return;
      }else if (typeid(data)==typeid(PixelMaxVsf*)){
	std::cout << "Will fetch1 PixelMaxVsf" << std::endl;
	assert(dir=="maxvsf");
	data = (T*) new PixelMaxVsf(fullpath+"maxvsf.dat");
	std::cout << "Will return1 the PixelMaxVsf" << std::endl;
	return;
      }else if (typeid(data)==typeid(PixelNameTranslation*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelDACSettings" << std::endl;
	assert(dir=="nametranslation");
	data = (T*) new PixelNameTranslation(fullpath+"translation.dat");
	return;
      }else if (typeid(data)==typeid(PixelFEDCard*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelFEDCard" << std::endl;
	assert(dir=="fedcard");
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill open:"<<fullpath+"params_fed_"+ext+".dat"<< std::endl;
	data = (T*) new PixelFEDCard(fullpath+"params_fed_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelCalibBase*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelCalibBase" << std::endl;
	assert(base=="calib");
	std::string calibfile=fullpath+"calib.dat";
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tLooking for file " << calibfile << std::endl;
	std::ifstream calibin(calibfile.c_str());
	if(calibin.good()){
	  data = (T*) new PixelCalibConfiguration(calibfile);
	}else{
	  calibfile=fullpath+"delay25.dat";
	  //std::cout << "[pos::PixelConfigFile::get()]\t\t\tNow looking for file " << calibfile << std::endl;
	  std::ifstream delayin(calibfile.c_str());
	  if(delayin.good()){
	    data = (T*) new PixelDelay25Calib(calibfile);
	  }else{
	    calibfile=fullpath+"fedtestdac.dat";
	    //std::cout << "[pos::PixelConfigFile::get()]\t\t\tNow looking for file " << calibfile << std::endl;
	    std::ifstream delayin(calibfile.c_str());
	    if(delayin.good()){
	      data = (T*) new PixelFEDTestDAC(calibfile);
	    }else{
	      std::cout << "[pos::PixelConfigFile::get()]\t\t\tCan't find calibration file calib.dat or delay25.dat or fedtestdac.dat" << std::endl;
	      data=0;
	    }
	  }
	}
	return;
      }else if (typeid(data)==typeid(PixelTKFECConfig*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelTKFECConfig" << std::endl;
	assert(dir=="tkfecconfig");
	data = (T*) new PixelTKFECConfig(fullpath+"tkfecconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelFECConfig*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelFECConfig" << std::endl;
	assert(dir=="fecconfig");
	data = (T*) new PixelFECConfig(fullpath+"fecconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelFEDConfig*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelFEDConfig" << std::endl;
	assert(dir=="fedconfig");
	data = (T*) new PixelFEDConfig(fullpath+"fedconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelPortCardConfig*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelPortCardConfig" << std::endl;
	assert(dir=="portcard");
	data = (T*) new PixelPortCardConfig(fullpath+"portcard_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelPortcardMap*)){
	//std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelPortcardMap" << std::endl;
	assert(dir=="portcardmap");
	data = (T*) new PixelPortcardMap(fullpath+"portcardmap.dat");
	return;
      }else if (typeid(data)==typeid(PixelDelay25Calib*)){
	//cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelDelay25Calib" << std::endl;
	assert(dir=="portcard");
	data = (T*) new PixelDelay25Calib(fullpath+"delay25.dat");
	return;
      }else if (typeid(data)==typeid(PixelTTCciConfig*)){
	//cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelTTCciConfig" << std::endl;
	assert(dir=="ttcciconfig");
	data = (T*) new PixelTTCciConfig(fullpath+"TTCciConfiguration.txt");
	return;
      }else if (typeid(data)==typeid(PixelLTCConfig*)){
	//cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelLTCConfig" << std::endl;
	assert(dir=="ltcconfig");
	data = (T*) new PixelLTCConfig(fullpath+"LTCConfiguration.txt");
	return;
      }else{
	std::cout << "[pos::PixelConfigFile::get()]\t\t\tNo match" << std::endl;
	assert(0);
	data=0;
	return;
      }

    }

    template <class T>
      static void get(std::map<std::string, T*> &pixelObjects, PixelConfigKey key){

      typename std::map<std::string, T* >::iterator iObject=pixelObjects.begin();

      for(;iObject!=pixelObjects.end();++iObject){
	get(iObject->second,iObject->first,key);
      }

    }


    static int makeNewVersion(std::string path, std::string &dir){
      std::cout << "[pos::PixelConfigFile::makeNewVersion()]\t\tInserting data on path:"<<path<<std::endl;
      struct stat stbuf;
      std::string directory=getenv("PIXELCONFIGURATIONBASE");
      directory+="/";
      directory+=path;
      if (stat(directory.c_str(),&stbuf)!=0){
        
	std::cout << "[pos::PixelConfigFile::makeNewVersion()]\t\tThe path:"<<path<<" does not exist."<<std::endl;
	std::cout << "[pos::PixelConfigFile::makeNewVersion()]\t\tFull path:"<<directory<<endl;
	return -1;
      }
      directory+="/";
      int version=-1;
      do{
	version++;
	std::ostringstream s1;
	s1 << version  ;
	std::string strversion=s1.str();
	dir=directory+strversion;
	std::cout << "[pos::PixelConfigFile::makeNewVersion()]\t\tWill check for version:"<<dir<<std::endl;
      }while(stat(dir.c_str(),&stbuf)==0);
      std::cout << "[pos::PixelConfigFile::makeNewVersion()]\t\tThe new version is:"<<version<<std::endl;
      mkdir(dir.c_str(),0777);
      return version;
    }


    template <class T>
    static int put(const T* object, std::string path){
      std::string dir;
      int version=makeNewVersion(path,dir);
      object->writeASCII(dir);
      return version;
    }

    template <class T>
    static int put(std::vector<T*> objects, std::string path){
      //cout << "[pos::PixelConfigFile::put()]\t\t# of objects to write: "<< objects.size() << endl;
      std::string dir;
      int version=makeNewVersion(path,dir);
      for(unsigned int i=0;i<objects.size();i++){
	//cout << "[pos::PixelConfigFile::put()]\t\tWill write i="<<i<<endl;
	objects[i]->writeASCII(dir);
      }
      return version;
    }

  private:


  };

}
#endif
