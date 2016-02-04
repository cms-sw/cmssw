#ifndef PixelConfig_h
#define PixelConfig_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelConfig.h
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/

namespace pos{
/*! \class PixelConfig PixelConfig.h "interface/PixelConfig.h"
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/
  class PixelConfig {
    
  public:

    void write(std::ofstream& out){
      for (unsigned int i=0;i<versions_.size();i++){
	out << versions_[i].first<<"   "<<versions_[i].second<<std::endl;
      }
    }

    void add(std::string dir, unsigned int version){
      std::pair<std::string,unsigned int> aPair(dir,version);
      versions_.push_back(aPair);    
    }
    //returns -1 if it can not find the dir.
    int find(std::string dir, unsigned int &version){
//      std::cout << "[pos::PixelConfig::find()] versions_.size() = " << versions_.size() << std::endl ;
      for(unsigned int i=0;i<versions_.size();i++){
//	std::cout << "Looking :"<<versions_[i].first
//	          <<" "<<versions_[i].second<<std::endl;
	if (versions_[i].first==dir) {
	  version=versions_[i].second;
	  return 0;
	}
      }
      return -1;
    }

    //returns -1 if it can not find the dir.
    int update(std::string dir, unsigned int &version, unsigned int newversion){
      for(unsigned int i=0;i<versions_.size();i++){
	//std::cout << "Looking :"<<versions_[i].first
	//	      <<" "<<versions_[i].second<<std::endl;
	if (versions_[i].first==dir) {
	  version=versions_[i].second;
	  versions_[i].second=newversion;
	  return 0;
	}
      }
      return -1;
    }

    std::vector<std::pair<std::string,unsigned int> > versions(){
      return versions_;
    }

  private:
   
    std::vector<std::pair<std::string,unsigned int> > versions_; 

  };
}
#endif
