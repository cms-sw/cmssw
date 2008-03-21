#include "RecoLuminosity/ROOTSchema/interface/ROOTSchema.h"
/*

Author: Adam Hunt - Princeton University
Date: 2007-10-05

*/


//
// constructor and destructor
//

ROOTSchema::ROOTSchema(){  
  #ifdef DEBUG
  cout << "In " << __PRETTY_FUNCTION__ << endl;
  #endif
}

ROOTSchema::~ROOTSchema(){   

  #ifdef DEBUG
  cout << "In " << __PRETTY_FUNCTION__ << endl;
  #endif

}

void ROOTSchema::ProcessSection(const HCAL_HLX::LUMI_SECTION &lumiSection){

  CreateFileName(lumiSection);
  CreateTree(lumiSection);  
  FillTree(lumiSection);
  CloseTree();
  Concatenate(lumiSection);
}
