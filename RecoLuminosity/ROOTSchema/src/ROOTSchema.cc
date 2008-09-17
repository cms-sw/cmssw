#include "RecoLuminosity/ROOTSchema/interface/ROOTSchema.h"
/*

Author: Adam Hunt - Princeton University
Date: 2007-10-05

*/


//
// constructor and destructor
//

HCAL_HLX::ROOTSchema::ROOTSchema(){  
#ifdef DEBUG
  std::cout << "In " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

HCAL_HLX::ROOTSchema::~ROOTSchema(){   

#ifdef DEBUG
  std::cout << "In " << __PRETTY_FUNCTION__ << std::endl;
#endif

}

bool HCAL_HLX::ROOTSchema::ProcessSection(const HCAL_HLX::LUMI_SECTION &lumiSection){

  SetFileName(lumiSection);
  FillTree(lumiSection);
  CloseTree();

  return true;

}
