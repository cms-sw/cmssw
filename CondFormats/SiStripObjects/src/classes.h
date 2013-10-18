#include "CondFormats/SiStripObjects/src/headers.h"


namespace {
  struct dictionary {
    std::vector< std::vector<FedChannelConnection> > tmp1;
  
#ifdef SISTRIPCABLING_USING_NEW_STRUCTURE
  
//    SiStripFedCabling::Registry            temp12;

#endif
   
    std::vector<SiStripThreshold::Container>  tmp22;
    std::vector< SiStripThreshold::DetRegistry >  tmp24;
 
  };
}  
  
