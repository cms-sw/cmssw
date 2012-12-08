#include <DataFormats/GEMDigi/interface/GEMDigi.h>
#include <DataFormats/GEMDigi/interface/GEMDigiCollection.h>

#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>
#include <map>

namespace{ 
  struct dictionary {
    
    GEMDigi g;
    std::vector<GEMDigi>  vg;
    std::vector<std::vector<GEMDigi> >  vvg; 
    GEMDigiCollection gc;
    edm::Wrapper<GEMDigiCollection> wg;
    
    edm::Wrapper<std::map< std::pair<int,int>, int > > a2;
  };
}
