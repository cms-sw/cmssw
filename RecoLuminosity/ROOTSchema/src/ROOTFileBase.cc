#include "RecoLuminosity/ROOTSchema/interface/ROOTFileBase.h"

ROOTFileBase::ROOTFileBase(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  Header  = new HCAL_HLX::LUMI_SECTION_HEADER;
  Summary = new HCAL_HLX::LUMI_SUMMARY;
  Detail  = new HCAL_HLX::LUMI_DETAIL;

  EtSum     = new HCAL_HLX::ET_SUM_SECTION[HCAL_HLX_MAX_HLXS];
  Occupancy = new HCAL_HLX::OCCUPANCY_SECTION[HCAL_HLX_MAX_HLXS];
  LHC       = new HCAL_HLX::LHC_SECTION[HCAL_HLX_MAX_HLXS];

  Threshold       = new HCAL_HLX::LUMI_THRESHOLD;
  L1HLTrigger     = new HCAL_HLX::LEVEL1_HLT_TRIGGER;
  TriggerDeadtime = new HCAL_HLX::TRIGGER_DEADTIME;

  outputDir = "/cms/mon/dqm/lumi/root/ls";

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

ROOTFileBase::~ROOTFileBase(){

#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  delete Header;
  delete Summary;
  delete Detail;

  delete [] EtSum;
  delete [] Occupancy;
  delete [] LHC;

  delete Threshold;
  delete L1HLTrigger;
  delete TriggerDeadtime; 

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif

}

void ROOTFileBase::CreateFileName(const HCAL_HLX::LUMI_SECTION &localSection){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  std::ostringstream outputString;
  outputString << outputDir << "/LS_"
	       << std::setfill('0') << std::setw(9)  << localSection.hdr.runNumber
	       << "_" << std::setfill('0') << std::setw(6) << localSection.hdr.sectionNumber << ".root";
  
  fileName = outputString.str();
  
#ifdef DEBUG
  std::cout << "Output file is " << fileName << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void ROOTFileBase::CreateTree(const HCAL_HLX::LUMI_SECTION & localSection){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  m_file   = new TFile(fileName.c_str(), "RECREATE");  
  if(!m_file){
    std::cout << " *** Couldn't make or open file: " << fileName << " *** " << std::endl;
  }

  m_file->cd();
  
  m_tree  = new TTree("LumiTree","");

  m_tree->Bronch("Header.",  "HCAL_HLX::LUMI_SECTION_HEADER", &Header, 1);
  m_tree->Bronch("Summary.", "HCAL_HLX::LUMI_SUMMARY",        &Summary, 1);
  m_tree->Bronch("Detail.",  "HCAL_HLX::LUMI_DETAIL",         &Detail, 1);

  m_tree->Bronch("Threshold.",        "HCAL_HLX::LUMI_THRESHOLD",      &Threshold, 1);
  m_tree->Bronch("Level1_HLTrigger.", "HCAL_HLX::LEVEL1_HLT_TRIGGER",  &L1HLTrigger, 1);
  m_tree->Bronch("Trigger_Deadtime.", "HCAL_HLX::TRIGGER_DEADTIME",    &TriggerDeadtime, 1);

  for(int i = 0; i < HCAL_HLX_MAX_HLXS; i++){
    EtSumPtr[i] = &EtSum[i];
    MakeBranch(localSection.etSum[i], &EtSumPtr[i], i);

    OccupancyPtr[i] = &Occupancy[i];
    MakeBranch(localSection.occupancy[i], &OccupancyPtr[i], i);

    LHCPtr[i] = &LHC[i];
    MakeBranch(localSection.lhc[i], &LHCPtr[i], i);
    // Yes, that is supposed to be the address of a pointer.
  }

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void ROOTFileBase::FillTree(const HCAL_HLX::LUMI_SECTION& localSection){
  
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
  
  for(int i = 0; i < HCAL_HLX_MAX_HLXS; i++){
    memcpy(&EtSum[i],     &localSection.etSum[i],     sizeof(HCAL_HLX::ET_SUM_SECTION));
    memcpy(&Occupancy[i], &localSection.occupancy[i], sizeof(HCAL_HLX::OCCUPANCY_SECTION));
    memcpy(&LHC[i],       &localSection.lhc[i],       sizeof(HCAL_HLX::LHC_SECTION));
  }

  memcpy(Header,  &localSection.hdr,         sizeof (localSection.hdr));
  memcpy(Summary, &localSection.lumiSummary, sizeof(HCAL_HLX::LUMI_SUMMARY));
  memcpy(Detail,  &localSection.lumiDetail,  sizeof(HCAL_HLX::LUMI_DETAIL));

  // TODO: Fill this information somewhere else.
  Threshold->OccThreshold1Set1 = 51;
  Threshold->OccThreshold2Set1 = 52;
  Threshold->OccThreshold1Set2 = 53;
  Threshold->OccThreshold2Set2 = 54;
  Threshold->ETSum             = 55;

  L1HLTrigger->TriggerValue     = 71;
  L1HLTrigger->TriggerBitNumber = 72;
  
  TriggerDeadtime->TriggerDeadtime = 81;

  m_tree->Fill();

#ifdef DEBUG
  //  m_tree->Print();
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif

}

template< class T >
void ROOTFileBase::MakeBranch(const T &in, T **out, int HLXNum){

#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
  
  const std::string typeName = typeid(T).name();
  std::string className;
  std::string branchName;
  std::ostringstream numString;

  if(typeName == "N8HCAL_HLX11LHC_SECTIONE"){
    className = "HCAL_HLX::LHC_SECTION";
    branchName = "LHC";
  }else if(typeName == "N8HCAL_HLX17OCCUPANCY_SECTIONE"){
    className = "HCAL_HLX::OCCUPANCY_SECTION";
    branchName = "Occupancy";
  }else if(typeName == "N8HCAL_HLX14ET_SUM_SECTIONE"){
    className = "HCAL_HLX::ET_SUM_SECTION";
    branchName = "ETSum";
  }

#ifdef DEBUG
  std::cout << "Class: " << className << std::endl;
  std::cout << "Class: " << typeid(T).name() << std::endl;
#endif
  
  numString << std::setfill('0') << std::setw(2) << HLXNum;
  branchName = branchName + numString.str() + ".";
  m_tree->Bronch(branchName.c_str(), className.c_str(), out, 1);

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif

}

void ROOTFileBase::CloseTree(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  m_file->Write();
  m_file->Close();

  //delete m_tree; // NO!!! root does this when you delete m_file
  delete m_file;

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif

}
