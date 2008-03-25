#ifndef PixelAliasList_h
#define PixelAliasList_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelAliasList.h
*   \brief The class to handle 'aliases.txt'
*
*   A longer explanation will be placed here later
*/

#include "CalibFormats/SiPixelObjects/interface/PixelConfigAlias.h"
#include "CalibFormats/SiPixelObjects/interface/PixelVersionAlias.h"


namespace pos{
/*! \class PixelAliasList PixelAliasList.h "interface/PixelAliasList.h"
*
*   A longer explanation will be placed here later
*/
  class PixelAliasList {

  public:

    void writefile(){

      std::string directory=getenv("PIXELCONFIGURATIONBASE");
    
      std::string filename=directory+"/aliases.txt";

      std::ofstream out(filename.c_str());

      out << "ConfigurationAliases" <<std::endl;
      for(unsigned int i=0;i<pixelConfigAliases_.size();i++){
	PixelConfigAlias& theAlias=pixelConfigAliases_[i];
	out << theAlias.name() << "     " 
	    << theAlias.key() << "   ";
      
	unsigned int n=theAlias.nVersionAliases();
	for (unsigned int j=0;j<n;j++){
	  out << theAlias.versionAliasesPath(j) << "  ";
	  out << theAlias.versionAliasesAlias(j) << "  ";
	}

	out << ";" << std::endl; 

      }

      out << "VersionAliases" <<std::endl;
      for(unsigned int i=0;i<pixelVersionAliases_.size();i++){
	PixelVersionAlias& theAlias=pixelVersionAliases_[i];

	out << theAlias.path() << "  "
	    << theAlias.version() << " "
	    << theAlias.alias() << " ;"<<std::endl;

      }
    }

    void readfile(std::string filename){

      std::ifstream in(filename.c_str());
      if (!in.good()) {
	std::cout << "Could not open file:"<<filename<<std::endl;
      }
      assert(in.good());

      std::string tag;
	    
      in >> tag;
      while(tag.substr(0,1) == "#") {
	in.ignore(4096, '\n'); //skips to endl;
	in >> tag;
      }

      assert(tag=="ConfigurationAliases");

      in >> tag;

      while(tag!="VersionAliases"){

	std::string alias=tag;

	unsigned int key;
	in >> key;

	//std::cout << "Alias, key:"<<alias<<" "<<key<<std::endl;

	PixelConfigAlias anAlias(alias,key);
      
	in >> tag;
      
	while(tag  != ";") {
	  std::string path;
	  std::string alias;

	  path=tag;
	  in >> alias;
	
	  //std::cout << "path, alias:"<<path<<" "<<alias<<std::endl;
	
	  anAlias.addVersionAlias(path,alias);
	  in >> tag;

	}

	pixelConfigAliases_.push_back(anAlias);

	in >> tag;

      }

      assert(tag=="VersionAliases");

      std::string path;
      std::string alias;
      unsigned int version;

      in >> path;
      in >> version;
      in >> alias;
      in >> tag;

      //std::cout << "path version alias tag:"<<path<<" "<<version
      //	      <<" "<<alias<<" "<<tag<<std::endl;

      while(!in.eof()){
	assert(tag==";");
	PixelVersionAlias aVersionAlias(path,version,alias);
	pixelVersionAliases_.push_back(aVersionAlias);

	in >> path;
	in >> version;
	in >> alias;
	in >> tag;

      }
    
      in.close();
    }


    void insertAlias(PixelConfigAlias& anAlias){
      for(unsigned int i=0;i<pixelConfigAliases_.size();i++){
	if (pixelConfigAliases_[i].name()==anAlias.name()){
	  std::cout << "Replacing existing alias:" << anAlias.name()<<std::endl;
	  pixelConfigAliases_[i]=anAlias;
	  return;
	}
      }
      pixelConfigAliases_.push_back(anAlias);
    }

    void insertVersionAlias(PixelVersionAlias& anAlias){
      for(unsigned int i=0;i<pixelVersionAliases_.size();i++){
	if (pixelVersionAliases_[i].alias()==anAlias.alias()&&
	    pixelVersionAliases_[i].path()==anAlias.path()){
	  std::cout << "Replacing existing version alias:" 
		    <<anAlias.path()<< " " << anAlias.alias() << std::endl;
	  pixelVersionAliases_[i]=anAlias;
	  return;
	}
      }
      pixelVersionAliases_.push_back(anAlias);
    }

    void updateConfigAlias(std::string path,unsigned int version,
			   std::string alias, PixelConfigList& config){
 
      //first loop over configuration aliases
      for(unsigned int i=0;i<pixelConfigAliases_.size();i++){
	//std::cout << "Looping over aliases:"<<i<<std::endl;
	for(unsigned int j=0;j<pixelConfigAliases_[i].nVersionAliases();j++){
	  //std::cout << "Looping over versionAliases:"<<j<<std::endl;
	  if(path==pixelConfigAliases_[i].versionAliasesPath(j)&&
	     alias==pixelConfigAliases_[i].versionAliasesAlias(j)){
	    //std::cout << "Making clone!"<<std::endl;
	    unsigned int newkey=config.clone(pixelConfigAliases_[i].key(),path,version);
	    pixelConfigAliases_[i].setKey(newkey);
	  }
	}
      }
    }

    std::vector<std::string> getVersionAliases(std::string path){
      std::vector<std::string> tmp;
      for(unsigned int i=0;i<pixelVersionAliases_.size();i++){
	std::cout << "path alias:"<<pixelVersionAliases_[i].path()
		  << pixelVersionAliases_[i].alias() << std::endl;
	if (pixelVersionAliases_[i].path()==path){
	  tmp.push_back(pixelVersionAliases_[i].alias());
	}
      }
      return tmp;
    }


    unsigned int getVersion(std::string path, std::string alias){
      for(unsigned int i=0;i<pixelVersionAliases_.size();i++){
	if (pixelVersionAliases_[i].alias()==alias&&
	    pixelVersionAliases_[i].path()==path){
	  return pixelVersionAliases_[i].version();
	}
      }
      assert(0);
      return 0;
    }

    unsigned int nAliases() { return pixelConfigAliases_.size(); }
    std::string name(unsigned int i) { return pixelConfigAliases_[i].name();}  
    unsigned int key(unsigned int i) { return pixelConfigAliases_[i].key();}

  private:

    std::vector<PixelConfigAlias> pixelConfigAliases_;
    std::vector<PixelVersionAlias> pixelVersionAliases_;

  };
}

#endif
