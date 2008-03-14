#ifndef PixelConfigFile_h
#define PixelConfigFile_h
//
// OK, first of this is not a DB; this class will try to 
// define an interface to accessing the configuration data.
// 
// 
// 
//

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


    static void addVersionAlias(std::string path, unsigned int version, std::string alias){
      PixelVersionAlias anAlias(path, version, alias);
      getAlias().insertVersionAlias(anAlias);
      getAlias().updateConfigAlias(path,version,alias,getConfig());
      getAlias().writefile();
      getConfig().writefile();
    }

    static unsigned int makeKey(std::vector<std::pair<std::string, unsigned int> > versions){

      PixelConfig config;

      for(unsigned int i=0;i<versions.size();i++){
	config.add(versions[i].first,versions[i].second);
      }

      unsigned int newkey=getConfig().add(config);

      getConfig().writefile();
    
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
	  //std::cout << "[pos::PixelConfigFile::getConfig()] Reloading configurations.txt"<< std::endl ;
	  configs.reload(filename) ;
	  //std::cout << "[pos::PixelConfigFile::getConfig()] Size reloaded: " << configs.size() << std::endl ;
	}
      if (first) 
	{
	  first=0;
	  //std::cout << "[pos::PixelConfigFile::getConfig()] Reading configurations.txt"<< std::endl ;
	  configs.readfile(filename);
	  //std::cout << "[pos::PixelConfigFile::getConfig()] Size read: " << configs.size() << std::endl ;
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
	std::cout << "Asking for data of type:"<<typeid(data).name()<<std::endl;
	std::cout << "On path:"<<path<<std::endl;
	std::cout << "Recall that you need a trailing /" << std::endl;
	::abort();
      }
    
      std::string dir=base.substr(slashpos+1);
    
/*       std::cout << "Extracted dir:"<<dir<<std::endl; */
/*       std::cout << "Extracted base:"<<base<<std::endl; */
/*       std::cout << "Extracted ext :"<<ext<<std::endl; */
    
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
	//std::cout << "Will return PixelTrimBase" << std::endl;
	assert(dir=="trim");
	data = (T*) new PixelTrimAllPixels(fullpath+"ROC_Trims_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelMaskBase*)){
	//std::cout << "Will return PixelMaskBase" << std::endl;
	assert(dir=="mask");
	data = (T*) new PixelMaskAllPixels(fullpath+"ROC_Masks_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelDACSettings*)){
	//std::cout << "Will return PixelDACSettings" << std::endl;
	assert(dir=="dac");
	data = (T*) new PixelDACSettings(fullpath+"ROC_DAC_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelTBMSettings*)){
	//std::cout << "Will return PixelTBMSettings" << std::endl;
	assert(dir=="tbm");
	data = (T*) new PixelTBMSettings(fullpath+"TBM_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelDetectorConfig*)){
	//std::cout << "Will return PixelDetectorConfig" << std::endl;
	assert(dir=="detconfig");
	data = (T*) new PixelDetectorConfig(fullpath+"detectconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelNameTranslation*)){
	//std::cout << "Will return PixelDACSettings" << std::endl;
	assert(dir=="nametranslation");
	data = (T*) new PixelNameTranslation(fullpath+"translation.dat");
	return;
      }else if (typeid(data)==typeid(PixelFEDCard*)){
	//std::cout << "Will return PixelFEDCard" << std::endl;
	assert(dir=="fedcard");
	//std::cout << "Will open:"<<fullpath+"params_fed_"+ext+".dat"<< std::endl;
	data = (T*) new PixelFEDCard(fullpath+"params_fed_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelCalibBase*)){
	//std::cout << "Will return PixelCalibBase" << std::endl;
	assert(dir=="calib");
	std::string calibfile=fullpath+"calib.dat";
	//std::cout << "Looking for file " << calibfile << std::endl;
	std::ifstream calibin(calibfile.c_str());
	if(calibin.good()){
	  data = (T*) new PixelCalibConfiguration(calibfile);
	}else{
	  calibfile=fullpath+"delay25.dat";
	  //std::cout << "Now looking for file " << calibfile << std::endl;
	  std::ifstream delayin(calibfile.c_str());
	  if(delayin.good()){
	    data = (T*) new PixelDelay25Calib(calibfile);
	  }else{
	    calibfile=fullpath+"fedtestdac.dat";
	    //std::cout << "Now looking for file " << calibfile << std::endl;
	    std::ifstream delayin(calibfile.c_str());
	    if(delayin.good()){
	      data = (T*) new PixelFEDTestDAC(calibfile);
	    }else{
	      std::cout << "Can't find calibration file calib.dat or delay25.dat or fedtestdac.dat" << std::endl;
	      data=0;
	    }
	  }
	}
	return;
      }else if (typeid(data)==typeid(PixelTKFECConfig*)){
	//std::cout << "Will return PixelTKFECConfig" << std::endl;
	assert(dir=="tkfecconfig");
	data = (T*) new PixelTKFECConfig(fullpath+"tkfecconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelFECConfig*)){
	//std::cout << "Will return PixelFECConfig" << std::endl;
	assert(dir=="fecconfig");
	data = (T*) new PixelFECConfig(fullpath+"fecconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelFEDConfig*)){
	//std::cout << "Will return PixelFEDConfig" << std::endl;
	assert(dir=="fedconfig");
	data = (T*) new PixelFEDConfig(fullpath+"fedconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelPortCardConfig*)){
	//std::cout << "Will return PixelPortCardConfig" << std::endl;
	assert(dir=="portcard");
	data = (T*) new PixelPortCardConfig(fullpath+"portcard_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelPortcardMap*)){
	//std::cout << "Will return PixelPortcardMap" << std::endl;
	assert(dir=="portcardmap");
	data = (T*) new PixelPortcardMap(fullpath+"portcardmap.dat");
	return;
      }else if (typeid(data)==typeid(PixelDelay25Calib*)){
	//cout << "Will return PixelDelay25Calib" << std::endl;
	assert(dir=="portcard");
	data = (T*) new PixelDelay25Calib(fullpath+"delay25.dat");
	return;
      }else if (typeid(data)==typeid(PixelTTCciConfig*)){
	//cout << "Will return PixelTTCciConfig" << std::endl;
	assert(dir=="ttcciconfig");
	data = (T*) new PixelTTCciConfig(fullpath+"TTCciConfiguration.txt");
	return;
      }else if (typeid(data)==typeid(PixelLTCConfig*)){
	//cout << "Will return PixelLTCConfig" << std::endl;
	assert(dir=="ltcconfig");
	data = (T*) new PixelLTCConfig(fullpath+"LTCConfiguration.txt");
	return;
      }else{
	std::cout << "No match" << std::endl;
	assert(0);
	data=0;
	return;
      }

    }

    //Returns a pointer to the data found in the path with configuration key.
    template <class T>
      static void get(T* &data, std::string path, unsigned int version){

      unsigned int last=path.find_last_of("/");
      assert(last!=std::string::npos);
    
      std::string base=path.substr(0,last);
      std::string ext=path.substr(last+1);
    
      unsigned int slashpos=base.find_last_of("/");
      //if (slashpos==std::string::npos) {
      //std::cout << "Asking for data of type:"<<typeid(data).name()<<std::endl;
      //std::cout << "On path:"<<path<<std::endl;
      //std::cout << "Recall that you need a trailing /" << std::endl;
      //::abort();
      //}
    
      std::string dir=base.substr(slashpos+1);
    
      //std::cout << "Extracted dir:"<<dir<<std::endl;
      //std::cout << "Extracted base:"<<base<<std::endl;
      //std::cout << "Extracted ext :"<<ext<<std::endl;
    
      ostringstream s1;
      s1 << version<<(char)(0);
      std::string strversion=s1.str();

      static std::string directory;
      directory=getenv("PIXELCONFIGURATIONBASE");
    
      std::string fullpath=directory+"/"+dir+"/"+strversion+"/";
    
      //std::cout << "Directory for configuration data:"<<fullpath<<std::endl;
    
      if (typeid(data)==typeid(PixelTrimBase*)){
	//std::cout << "Will return PixelTrimBase" << std::endl;
	assert(dir=="trim");
	data = (T*) new PixelTrimAllPixels(fullpath+"ROC_Trims_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelMaskBase*)){
	//std::cout << "Will return PixelMaskBase" << std::endl;
	assert(dir=="mask");
	data = (T*) new PixelMaskAllPixels(fullpath+"ROC_Masks_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelDACSettings*)){
	//std::cout << "Will return PixelDACSettings" << std::endl;
	assert(dir=="dac");
	data = (T*) new PixelDACSettings(fullpath+"ROC_DAC_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelTBMSettings*)){
	//std::cout << "Will return PixelTBMSettings" << std::endl;
	assert(dir=="tbm");
	data = (T*) new PixelTBMSettings(fullpath+"TBM_module_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelDetectorConfig*)){
	//std::cout << "Will return PixelDACSettings" << std::endl;
	assert(dir=="detconfig");
	data = (T*) new PixelDetectorConfig(fullpath+"detectconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelNameTranslation*)){
	//std::cout << "Will return PixelDACSettings" << std::endl;
	assert(dir=="nametranslation");
	data = (T*) new PixelNameTranslation(fullpath+"translation.dat");
	return;
      }else if (typeid(data)==typeid(PixelFEDCard*)){
	//std::cout << "Will return PixelFEDCard" << std::endl;
	assert(dir=="fedcard");
	//std::cout << "Will open:"<<fullpath+"params_fed_"+ext+".dat"<< std::endl;
	data = (T*) new PixelFEDCard(fullpath+"params_fed_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelCalibBase*)){
	//std::cout << "Will return PixelCalibBase" << std::endl;
	assert(base=="calib");
	std::string calibfile=fullpath+"calib.dat";
	//std::cout << "Looking for file " << calibfile << std::endl;
	std::ifstream calibin(calibfile.c_str());
	if(calibin.good()){
	  data = (T*) new PixelCalibConfiguration(calibfile);
	}else{
	  calibfile=fullpath+"delay25.dat";
	  //std::cout << "Now looking for file " << calibfile << std::endl;
	  std::ifstream delayin(calibfile.c_str());
	  if(delayin.good()){
	    data = (T*) new PixelDelay25Calib(calibfile);
	  }else{
	    calibfile=fullpath+"fedtestdac.dat";
	    //std::cout << "Now looking for file " << calibfile << std::endl;
	    std::ifstream delayin(calibfile.c_str());
	    if(delayin.good()){
	      data = (T*) new PixelFEDTestDAC(calibfile);
	    }else{
	      std::cout << "Can't find calibration file calib.dat or delay25.dat or fedtestdac.dat" << std::endl;
	      data=0;
	    }
	  }
	}
	return;
      }else if (typeid(data)==typeid(PixelTKFECConfig*)){
	//std::cout << "Will return PixelTKFECConfig" << std::endl;
	assert(dir=="tkfecconfig");
	data = (T*) new PixelTKFECConfig(fullpath+"tkfecconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelFECConfig*)){
	//std::cout << "Will return PixelFECConfig" << std::endl;
	assert(dir=="fecconfig");
	data = (T*) new PixelFECConfig(fullpath+"fecconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelFEDConfig*)){
	//std::cout << "Will return PixelFEDConfig" << std::endl;
	assert(dir=="fedconfig");
	data = (T*) new PixelFEDConfig(fullpath+"fedconfig.dat");
	return;
      }else if (typeid(data)==typeid(PixelPortCardConfig*)){
	//std::cout << "Will return PixelPortCardConfig" << std::endl;
	assert(dir=="portcard");
	data = (T*) new PixelPortCardConfig(fullpath+"portcard_"+ext+".dat");
	return;
      }else if (typeid(data)==typeid(PixelPortcardMap*)){
	//std::cout << "Will return PixelPortcardMap" << std::endl;
	assert(dir=="portcardmap");
	data = (T*) new PixelPortcardMap(fullpath+"portcardmap.dat");
	return;
      }else if (typeid(data)==typeid(PixelDelay25Calib*)){
	//cout << "Will return PixelDelay25Calib" << std::endl;
	assert(dir=="portcard");
	data = (T*) new PixelDelay25Calib(fullpath+"delay25.dat");
	return;
      }else if (typeid(data)==typeid(PixelTTCciConfig*)){
	//cout << "Will return PixelTTCciConfig" << std::endl;
	assert(dir=="ttcciconfig");
	data = (T*) new PixelTTCciConfig(fullpath+"TTCciConfiguration.txt");
	return;
      }else if (typeid(data)==typeid(PixelLTCConfig*)){
	//cout << "Will return PixelLTCConfig" << std::endl;
	assert(dir=="ltcconfig");
	data = (T*) new PixelLTCConfig(fullpath+"LTCConfiguration.txt");
	return;
      }else{
	std::cout << "No match" << std::endl;
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
      std::cout << "Inserting data on path:"<<path<<std::endl;
      struct stat stbuf;
      std::string directory=getenv("PIXELCONFIGURATIONBASE");
      directory+="/";
      directory+=path;
      if (stat(directory.c_str(),&stbuf)!=0){
        
	std::cout << "The path:"<<path<<" does not exist."<<std::endl;
	cout << "Full path:"<<directory<<endl;
	return -1;
      }
      directory+="/";
      int version=-1;
      do{
	version++;
	std::ostringstream s1;
/* 	s1 << version <<(char)(0); */
	s1 << version  ;
	std::string strversion=s1.str();
	dir=directory+strversion;
	std::cout << "Will check for version:"<<dir<<std::endl;
      }while(stat(dir.c_str(),&stbuf)==0);
      std::cout << "The new version is:"<<version<<std::endl;
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
      cout << "In PixelConfigFile::put"<<endl;
      std::string dir;
      int version=makeNewVersion(path,dir);
      for(unsigned int i=0;i<objects.size();i++){
	//cout << "Will write i="<<i<<endl;
	objects[i]->writeASCII(dir);
      }
      return version;
    }

  private:


  };

}
#endif
