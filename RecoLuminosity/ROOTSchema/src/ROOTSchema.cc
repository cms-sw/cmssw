#include "RecoLuminosity/ROOTSchema/interface/ROOTSchema.h"


/*

Author: Adam Hunt - Princeton University
Date: 2007-10-05

*/


//
// constructor and destructor
//


ROOTSchema::ROOTSchema(unsigned int runNumber = 0, unsigned int sectionNumber = 0){  
  
  #ifdef DEBUG
  cout << "In " << __PRETTY_FUNCTION__ << endl;
  #endif

  Header = NULL;
  Summary = NULL;
  BX = NULL;
  
  std::ostringstream runStream;
  std::ostringstream sectionStream;
  std::string filename;

  int i;
  
  runStream << std::dec << runNumber;
  sectionStream << std::dec << sectionNumber;

  filename = "LS_";
  
  for (i = 0; i < 5; i++){
    runNumber /= 10;
    if(runNumber == 0)
      filename = filename + "0";
  }
  
  filename = filename + runStream.str() + "_";

  for (i = 0; i < 5; i++){
    sectionNumber /= 10;
    if(sectionNumber == 0)
      filename = filename + "0";
  }

  filename = filename + sectionStream.str() + ".root";

  m_file   = new TFile(filename.c_str(), "RECREATE");  
  m_file->cd();
  
  m_tree  = new TTree("LumiTree","");

  Header  = new HCAL_HLX::LUMI_SECTION_HEADER;
  Summary = new HCAL_HLX::LUMI_SUMMARY;
  BX      = new HCAL_HLX::LUMI_BUNCH_CROSSING;

  EtSum     = new HCAL_HLX::ET_SUM_SECTION[HCAL_HLX_MAX_HLXS];
  Occupancy = new HCAL_HLX::OCCUPANCY_SECTION[HCAL_HLX_MAX_HLXS];
  LHC       = new HCAL_HLX::LHC_SECTION[HCAL_HLX_MAX_HLXS];

  Threshold       = new HCAL_HLX::LUMI_THRESHOLD;
  LumiSectionHist = new HCAL_HLX::LUMI_SECTION_HST;
  L1HLTrigger     = new HCAL_HLX::LEVEL1_HLT_TRIGGER;
  TriggerDeadtime = new HCAL_HLX::TRIGGER_DEADTIME;

}

ROOTSchema::~ROOTSchema(){   
  #ifdef DEBUG
  cout << "In " << __PRETTY_FUNCTION__ << endl;
  #endif

  m_file->Write();
  m_file->Close();

  // delete  m_tree;

  delete Header;
  delete Summary;
  delete BX;

  delete [] EtSum;
  delete [] Occupancy;
  delete [] LHC;

  delete Threshold;
  delete LumiSectionHist;
  delete L1HLTrigger;
  delete TriggerDeadtime; 
}

void ROOTSchema::FillTree(const HCAL_HLX::LUMI_SECTION& localSection){

  #ifdef DEBUG
  cout << "In " << __PRETTY_FUNCTION__ << endl;
  #endif

  unsigned int compress = 100;

  HCAL_HLX::ET_SUM_SECTION    *EtSumPtr[HCAL_HLX_MAX_HLXS];
  HCAL_HLX::OCCUPANCY_SECTION *OccupancyPtr[HCAL_HLX_MAX_HLXS];
  HCAL_HLX::LHC_SECTION       *LHCPtr[HCAL_HLX_MAX_HLXS];
  
  m_tree->Bronch("Header.",        "HCAL_HLX::LUMI_SECTION_HEADER", &Header, sizeof(Header)/compress);
  m_tree->Bronch("Summary.",       "HCAL_HLX::LUMI_SUMMARY",        &Summary, sizeof(Summary)/compress);
  m_tree->Bronch("BunchCrossing.", "HCAL_HLX::LUMI_BUNCH_CROSSING", &BX, sizeof(BX)/compress);

  for(int i = 0; i < HCAL_HLX_MAX_HLXS; i++){
    EtSumPtr[i] = &EtSum[i];
    MBCD(localSection.etSum[i], &EtSumPtr[i], i, compress);
    OccupancyPtr[i] = &Occupancy[i];
    MBCD(localSection.occupancy[i], &OccupancyPtr[i], i, compress);
    LHCPtr[i] = &LHC[i];
    MBCD(localSection.lhc[i], &LHCPtr[i], i, compress);
    // Yes, that is supposed to be the address of a pointer.  ROOT is strange.
  }

  m_tree->Bronch("Threshold.",        "HCAL_HLX::LUMI_THRESHOLD",      &Threshold, sizeof(Threshold)/compress);
  m_tree->Bronch("Lumi_Section_Hist.","HCAL_HLX::LUMI_SECTION_HST",    &LumiSectionHist, sizeof(LumiSectionHist)/compress);  
  m_tree->Bronch("Level1_HLTrigger.", "HCAL_HLX::LEVEL1_HLT_TRIGGER",  &L1HLTrigger, sizeof(L1HLTrigger)/compress);
  m_tree->Bronch("Trigger_Deadtime.", "HCAL_HLX::TRIGGER_DEADTIME",    &TriggerDeadtime, sizeof(TriggerDeadtime)/compress);

  Threshold->Threshold1Set1 = 51;
  Threshold->Threshold2Set1 = 52;
  Threshold->Threshold1Set2 = 53;
  Threshold->Threshold2Set2 = 54;
  Threshold->ET             = 55;
  
  LumiSectionHist->IsDataTaking      = true;
  LumiSectionHist->BeginOrbitNumber  = 61;
  LumiSectionHist->EndOrbitNumber    = 62;
  LumiSectionHist->RunNumber         = 63;
  LumiSectionHist->LumiSectionNumber = 64;
  LumiSectionHist->FillNumber        = 65;
  LumiSectionHist->SecStopTime       = 66;
  LumiSectionHist->SecStartTime      = 67;
  
  L1HLTrigger->TriggerValue          = 71;
  L1HLTrigger->TriggerBitNumber      = 72;
  
  TriggerDeadtime->TriggerDeadtime   = 81;

  memcpy(Header,  &localSection.hdr,               sizeof (localSection.hdr));
  memcpy(Summary, &localSection.lumiSummary,       sizeof(localSection.lumiSummary));
  memcpy(BX,      &localSection.lumiBunchCrossing, sizeof(localSection.lumiBunchCrossing));

  m_tree->Fill();
  //  m_tree->Print();
}

void ROOTSchema::MBCD(const HCAL_HLX::ET_SUM_SECTION &in, HCAL_HLX::ET_SUM_SECTION **out, int num, unsigned int compress = 100){
  #ifdef DEBUG
  cout << "In " << __PRETTY_FUNCTION__ << endl;
  #endif
  
  // TODO: Generalize the following line for ET_SUM, OCCUPANCY, and LHC. 
  std::string branchName = "ET_SUM";
  //  Size of class?  Sorry, sizeof(ET_SUM_SECTION) == sizeof(LHC_SECTION).

  std::string className = "HCAL_HLX::" +branchName+ "_SECTION";

  std::ostringstream numString;
  numString << std::dec << num;
  branchName = branchName + ((num / 10 == 0)? "0" : "") + numString.str() + "."; 
  m_tree->Bronch(branchName.c_str() , className.c_str(), out, sizeof(in)/compress); 
  memcpy(*out, &in, sizeof(in));    
}

void ROOTSchema::MBCD(const HCAL_HLX::OCCUPANCY_SECTION &in, HCAL_HLX::OCCUPANCY_SECTION **out, int num, unsigned int compress = 100){
  #ifdef DEBUG
  cout << "In " << __PRETTY_FUNCTION__ << endl;
  #endif
  
  // TODO: Generalize the following line.
  std::string branchName = "OCCUPANCY";
  //  Size of class?  Sorry, sizeof(ET_SUM_SECTION) == sizeof(LHC_SECTION).

  std::string className = "HCAL_HLX::" + branchName + "_SECTION";
  std::ostringstream numString;

  numString << std::dec << num;
  branchName = branchName + ((num / 10 == 0)? "0" : "") + numString.str() + "."; 
  m_tree->Bronch(branchName.c_str() , className.c_str(), out, sizeof(in)/compress); 
  memcpy(*out, &in, sizeof(in));  
}

void ROOTSchema::MBCD(const HCAL_HLX::LHC_SECTION &in, HCAL_HLX::LHC_SECTION **out, int num,unsigned int compress = 100){
  #ifdef DEBUG
  cout << "In " << __PRETTY_FUNCTION__ << endl;
  #endif
  
  // TODO: Generalize the following line.
  std::string branchName = "LHC";
  //  Size of class?  Sorry, sizeof(ET_SUM_SECTION) == sizeof(LHC_SECTION).

  std::string className = "HCAL_HLX::" + branchName+ "_SECTION";
  std::ostringstream numString;

  numString << std::dec << num;
  branchName = branchName + ((num / 10 == 0)? "0" : "") + numString.str() + "."; 
  m_tree->Bronch(branchName.c_str() , className.c_str(), out, sizeof(in)/compress); 
  memcpy(*out, &in, sizeof(in));  
}
