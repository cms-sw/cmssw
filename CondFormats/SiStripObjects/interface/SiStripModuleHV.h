#ifndef SiStripModuleHV_h
#define SiStripModuleHV_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>


class SiStripModuleHV {

 public:

  struct DetRegistry{
    uint32_t detid;
     };

   

  typedef std::vector<int>::const_iterator        ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<DetRegistry>                         Registry;
  typedef Registry::const_iterator                         RegistryIterator;
 
  SiStripModuleHV(){};
  ~SiStripModuleHV(){};
  
  bool put(std::vector<uint32_t> & DetId);
  bool putalldetids(std::vector<uint32_t> & DetId);
  
 
  void getDetIds(std::vector<uint32_t>& DetIds_) const ;
  void GetAllDetIds(std::vector<uint32_t>& AllDetIds) const;
  

  bool  IsModuleHVOff(uint32_t DetID) const;
 


 private:
  std::vector<uint32_t> v_hvoff; 
  std::vector<uint32_t> v_det_ids; 
  
  

};

#endif
