#include "DataFormats/GEMDigi/interface/GEMDigi.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include "DataFormats/GEMDigi/interface/GEMPadDigi.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"

#include "DataFormats/GEMDigi/interface/GEMPadDigiCluster.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"

#include "DataFormats/GEMDigi/interface/GEMCoPadDigi.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"

#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"

#include "DataFormats/GEMDigi/interface/ME0Digi.h"
#include "DataFormats/GEMDigi/interface/ME0DigiCollection.h"

#include "DataFormats/GEMDigi/interface/ME0PadDigi.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiCollection.h"

#include "DataFormats/GEMDigi/interface/ME0PadDigiCluster.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiClusterCollection.h"

#include "DataFormats/GEMDigi/interface/ME0TriggerDigi.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"

#include "DataFormats/Common/interface/Wrapper.h"
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

    GEMPadDigiCluster gcc;
    std::vector<GEMPadDigiCluster>  vgcc;
    std::vector<std::vector<GEMPadDigiCluster> >  vvgcc;
    GEMPadDigiClusterCollection gcccol;
    edm::Wrapper<GEMPadDigiClusterCollection> wgcc;

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
    ME0DigiPreRecoMap mmap;
    edm::Wrapper<ME0DigiPreRecoMap> wmmap;
    
    ME0Digi mm;
    std::vector<ME0Digi>  vmm;
    std::vector<std::vector<ME0Digi> >  vvmm;
    ME0DigiCollection mmcol;
    edm::Wrapper<ME0DigiCollection> wmm;

    ME0PadDigi mp;
    std::vector<ME0PadDigi>  vmp;
    std::vector<std::vector<ME0PadDigi> >  vvmp;
    ME0PadDigiCollection mpcol;
    edm::Wrapper<ME0PadDigiCollection> wmp;

    ME0PadDigiCluster mpc;
    std::vector<ME0PadDigiCluster>  vmpc;
    std::vector<std::vector<ME0PadDigiCluster> >  vvmpc;
    ME0PadDigiClusterCollection mpccol;
    edm::Wrapper<ME0PadDigiClusterCollection> wmpc;

    ME0TriggerDigi ml;
    std::vector<ME0TriggerDigi>  vml;
    std::vector<std::vector<ME0TriggerDigi> >  vvml;
    ME0TriggerDigiCollection mlcol;
    edm::Wrapper<ME0TriggerDigiCollection> wml;
  };
}
