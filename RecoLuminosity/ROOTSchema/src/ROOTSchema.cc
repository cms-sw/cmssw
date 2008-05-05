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
  cout << "In " << __PRETTY_FUNCTION__ << endl;
  #endif
}

HCAL_HLX::ROOTSchema::~ROOTSchema(){   

  #ifdef DEBUG
  cout << "In " << __PRETTY_FUNCTION__ << endl;
  #endif

}

void HCAL_HLX::ROOTSchema::ProcessSection(const HCAL_HLX::LUMI_SECTION &lumiSection){

  SetFileName(CreateLSFileName(lumiSection.hdr.runNumber, lumiSection.hdr.sectionNumber, lumiSection.hdr.bCMSLive));
  CreateTree(lumiSection);  
  FillTree(lumiSection);
  CloseTree();

}
