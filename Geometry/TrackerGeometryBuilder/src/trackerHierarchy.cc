#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include<string>
#include<vector>

std::string trackerHierarchy(unsigned int rawid) {
  DetId id(rawid);
  int subdetid = id.subdetId();
  switch (subdetid) {
    
    // PXB
  case 1:
    {
      PXBDetId module(rawid);
      char theLayer  = module.layer();
      char theLadder = module.ladder();
      char theModule = module.module();
      char key[] = { 1, theLayer , theLadder, theModule};
      return std::string(key,4);
    }
    
    // P1XF
  case 2:
    {
      PXFDetId module(rawid);
      char thePanel  = module.panel();
      char theDisk   = module.disk();
      char theBlade  = module.blade();
      char theModule = module.module();
      char key[] = { 2,
		     char(module.side()),
		     thePanel , theDisk, 
		     theBlade, theModule};
      return std::string(key,6);
    }
  
  // TIB
  case 3:
    {
      TIBDetId module(rawid);
      char            theLayer  = module.layer();
      std::vector<unsigned int> theString = module.string();
      char             theModule = module.module();
      //side = (theString[0] == 1 ) ? "-" : "+";
      //part = (theString[1] == 1 ) ? "int" : "ext";
      char key[] = { 3, 
		     theLayer, 
		     char(theString[0]),
		     char(theString[1]), 
		     char(theString[2]), 
		     theModule,
		     char(module.glued() ? module.stereo()+1 : 0)
      };
      return std::string(key, module.glued() ? 7 : 6);
    }
    
    // TID
  case 4:
    {
      TIDDetId module(rawid);
      unsigned int         theDisk   = module.wheel();
      unsigned int         theRing   = module.ring();
      std::vector<unsigned int> theModule = module.module();
      // side = (module.side() == 1 ) ? "-" : "+";
      // part = (theModule[0] == 1 ) ? "back" : "front";
      char key[] = { 4, 
		     char(module.side()),
		     theDisk , 
		     theRing,
		     char(theModule[0]), 
		     char(theModule[1]),
		     char(module.glued() ? module.stereo()+1 : 0)
      };
      return std::string(key,module.glued() ? 7 : 6);
    }
    
    // TOB
  case 5:
    {
      TOBDetId module(rawid);
      unsigned int              theLayer  = module.layer();
      std::vector<unsigned int> theRod    = module.rod();
      unsigned int              theModule = module.module();
      //	side = (theRod[0] == 1 ) ? "-" : "+";
      char key[] = { 5, theLayer , 
		     char(theRod[0]), 
		     char(theRod[1]), 
		     theModule,
		     char(module.glued() ? module.stereo()+1 : 0)
      };
      return std::string(key, module.glued() ?  6 : 5);
    }
    
    // TEC
  case 6:
    {
      TECDetId module(rawid);
      unsigned int              theWheel  = module.wheel();
      unsigned int              theModule = module.module();
      std::vector<unsigned int> thePetal  = module.petal();
      unsigned int              theRing   = module.ring();
      //	side  = (module.side() == 1 ) ? "-" : "+";
      //	petal = (thePetal[0] == 1 ) ? "back" : "front";
      // int out_side  = (module.side() == 1 ) ? -1 : 1;
      
      char key[] = { 6, 
		     char(module.side()),
		     theWheel,
		     char(thePetal[0]), 
		     char(thePetal[1]),
		     theRing,
		     theModule,
		     char(module.glued() ? module.stereo()+1 : 0)
      };
      return std::string(key, module.glued() ? 8 : 7);
    }
  default:
    return std::string();
  }
}












}
