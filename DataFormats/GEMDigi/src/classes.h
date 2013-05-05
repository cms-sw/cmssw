#include <DataFormats/GEMDigi/interface/GEMDigi.h>
#include <DataFormats/GEMDigi/interface/GEMDigiCollection.h>

#include <DataFormats/GEMDigi/interface/GEMCSCPadDigi.h>
#include <DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h>

#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>

namespace{ 
  struct dictionary {
    
    GEMDigi g;
    std::vector<GEMDigi>  vg;
    std::vector<std::vector<GEMDigi> >  vvg;
    GEMDigiCollection gcol;
    edm::Wrapper<GEMDigiCollection> wg;

    GEMCSCPadDigi gc;
    std::vector<GEMCSCPadDigi>  vgc;
    std::vector<std::vector<GEMCSCPadDigi> >  vvgc;
    GEMCSCPadDigiCollection gccol;
    edm::Wrapper<GEMCSCPadDigiCollection> wgc;
  };
}
