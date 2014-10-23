#include <DataFormats/GEMDigi/interface/GEMDigi.h>
#include <DataFormats/GEMDigi/interface/GEMDigiCollection.h>

#include <DataFormats/GEMDigi/interface/GEMPadDigi.h>
#include <DataFormats/GEMDigi/interface/GEMPadDigiCollection.h>

#include <DataFormats/GEMDigi/interface/GEMCoPadDigi.h>
#include <DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h>

#include <DataFormats/GEMDigi/interface/ME0DigiPreReco.h>
#include <DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h>

#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>

namespace DataFormats_GEMDigi {
  struct dictionary {
    
    GEMDigi g;
    std::vector<GEMDigi>  vg;
    std::vector<std::vector<GEMDigi> >  vvg;
    GEMDigiCollection gcol;
    edm::Wrapper<GEMDigiCollection> wg;

    GEMPadDigi gc;
    std::vector<GEMPadDigi>  vgc;
    std::vector<std::vector<GEMPadDigi> >  vvgc;
    GEMPadDigiCollection gccol;
    edm::Wrapper<GEMPadDigiCollection> wgc;

    GEMCoPadDigi gcp;
    std::vector<GEMCoPadDigi>  vgcp;
    std::vector<std::vector<GEMCoPadDigi> >  vvgcp;
    GEMCoPadDigiCollection gcpcol;
    edm::Wrapper<GEMCoPadDigiCollection> wgcp;

    ME0DigiPreReco m;
    std::vector<ME0DigiPreReco>  vm;
    std::vector<std::vector<ME0DigiPreReco> >  vvm;
    ME0DigiPreRecoCollection mcol;
    edm::Wrapper<ME0DigiPreRecoCollection> wm;
  };
}
