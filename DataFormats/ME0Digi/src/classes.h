#include <DataFormats/ME0Digi/interface/ME0Digi.h>
#include <DataFormats/ME0Digi/interface/ME0DigiCollection.h>

#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>

namespace{ 
  struct dictionary {
    
    ME0Digi g;
    std::vector<ME0Digi>  vg;
    std::vector<std::vector<ME0Digi> >  vvg;
    ME0DigiCollection gcol;
    edm::Wrapper<ME0DigiCollection> wg;
  };
}
