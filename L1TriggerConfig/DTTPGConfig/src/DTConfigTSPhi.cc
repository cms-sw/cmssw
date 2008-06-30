//-------------------------------------------------
//
//   Class: DTConfigSectColl
//
//   Description: Configurable parameters and constants 
//   for Level1 Mu DT Trigger - TS Phi
//
//
//   Author List:
//   C. Battilana
//
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTSPhi.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//----------------
// Constructors --
//----------------
DTConfigTSPhi::DTConfigTSPhi(const edm::ParameterSet& ps) { 

  setDefaults(ps);
  if (debug()) print();

}

//--------------
// Destructor --
//--------------
DTConfigTSPhi::~DTConfigTSPhi() {}

//--------------
// Operations --
//--------------

void
DTConfigTSPhi::setDefaults(const edm::ParameterSet& ps) {

  // Debug flag 
  m_debug = ps.getUntrackedParameter<bool>("Debug");
  
  // Order of quaity bits in TSS for sort1
  int mymsk = ps.getParameter<int>("TSSMSK1");
  if (checkMask(mymsk)) 
    m_tssmsk[0] = mymsk;
  else {
    std::cout << "DTConfigTSPhi::setDefaults : TSSMSK1 not in correct form! Default Used" << std::endl;
    m_tssmsk[0] = default_tsmsk;
  }
  
  // Order of quaity bits in TSS for sort2
  mymsk= ps.getParameter<int>("TSSMSK2");  
  if (checkMask(mymsk)) 
    m_tssmsk[1] = mymsk;
  else {
    std::cout << "DTConfigTSPhi::setDefaults : TSSMSK2 not in correct form! Default Used" << std::endl;
    m_tssmsk[1] = default_tsmsk;
  }
  
  // Htrig checking in TSS for sort1
  m_tsshte[0] = ps.getParameter<bool>("TSSHTE1");
  
  // Htrig checking in TSS for sort2
  m_tsshte[1] = ps.getParameter<bool>("TSSHTE2");

  // Htrig checking in TSS for carry
  m_tsshte[2] = ps.getParameter<bool>("TSSHTEC");

  // Inner SL checking in TSS for sort1
  m_tssnoe[0] = ps.getParameter<bool>("TSSNOE1");
  
  // Inner SL checking in TSS for sort2
  m_tssnoe[1] = ps.getParameter<bool>("TSSNOE2");
 
  // Inner SL checking in TSS for carry
  m_tssnoe[2] = ps.getParameter<bool>("TSSNOEC");

  // Correlation checking in TSS for sort1
  m_tsscce[0] = ps.getParameter<bool>("TSSCCE1");

  // Correlation checking in TSS for sort2
  m_tsscce[1] = ps.getParameter<bool>("TSSCCE2");
 
  // Correlation checking in TSS for carry
  m_tsscce[2] = ps.getParameter<bool>("TSSCCEC");

  // Ghost 1 supperssion option in TSS
  int mygs = ps.getParameter<int>("TSSGS1");
  if (mygs>=0 && mygs<3)
    m_tssgs1 = mygs;
  else {
    std::cout << "DTConfigTSPhi::setDefaults : TSSGS1 value is not correct! Default Used" << std::endl;
    m_tssgs1 = default_gs;
  }
  
  // Ghost 2 supperssion option in TSS
  mygs= ps.getParameter<int>("TSSGS2");
  if (mygs>=0 && mygs<5)
    m_tssgs2 = mygs;
  else {
    std::cout << "DTConfigTSPhi::setDefaults : TSSGS2 value is not correct! Default Used" << std::endl;
    m_tssgs2 = default_gs;
  }

  // Correlated ghost 1 supperssion option in TSS
  m_tsscgs1 = ps.getParameter<bool>("TSSCGS1");
  
  // Correlated ghost 2 supperssion option in TSS
  m_tsscgs2 = ps.getParameter<bool>("TSSCGS2");

  // Order of quaity bits in TSM for sort1
  mymsk = ps.getParameter<int>("TSMMSK1");
  if (checkMask(mymsk)) 
    m_tsmmsk[0] = mymsk;
  else {
    std::cout << "DTConfigTSPhi::setDefaults : TSMMSK1 not in correct form! Default Used" << std::endl;
    m_tsmmsk[0] = default_tsmsk;
  }

  // Order of quaity bits in TSM for sort2
  mymsk= ps.getParameter<int>("TSMMSK2");
  if (checkMask(mymsk)) 
    m_tsmmsk[1] = mymsk;
  else {
    std::cout << "DTConfigTSPhi::setDefaults : TSMMSK2 not in correct form! Default Used" << std::endl;
    m_tsmmsk[1] = default_tsmsk;
  }
  
  // Htrig checking in TSM for sort1
  m_tsmhte[0] = ps.getParameter<bool>("TSMHTE1");
  
  // Htrig checking in TSM for sort2
  m_tsmhte[1] = ps.getParameter<bool>("TSMHTE2");
 
  // Htrig checking in TSM for carry
  m_tsmhte[2] = ps.getParameter<bool>("TSMHTEC");
  
  // Inner SL checking in TSM for sort1
  m_tsmnoe[0] = ps.getParameter<bool>("TSMNOE1");
  
  // Inner SL checking in TSM for sort2
  m_tsmnoe[1] = ps.getParameter<bool>("TSMNOE2");
 
  // Inner SL checking in TSM for carry
  m_tsmnoe[2] = ps.getParameter<bool>("TSMNOEC");

  // Correlation checking in TSM for sort1
  m_tsmcce[0] = ps.getParameter<bool>("TSMCCE1");
  
  // Correlation checking in TSM for sort2
  m_tsmcce[1] = ps.getParameter<bool>("TSMCCE2");
 
  // Correlation checking in TSM for carry
  m_tsmcce[2] = ps.getParameter<bool>("TSMCCEC");

  // Ghost 1 supperssion option in TSM
  mygs = ps.getParameter<int>("TSMGS1");
  if (mygs>=0 && mygs<3)
    m_tsmgs1 = mygs;
  else {
    std::cout << "DTConfigTSPhi::setDefaults : TSMGS1 value is not correct! Default Used" << std::endl;
    m_tsmgs1 = default_gs;
  }

  // Ghost 2 supperssion option in TSM
  mygs= ps.getParameter<int>("TSMGS2");
  if (mygs>=0 && mygs<5)
    m_tsmgs2 = mygs;
  else {
    std::cout << "DTConfigTSPhi::setDefaults : TSMGS2 value is not correct! Default Used" << std::endl;
    m_tsmgs2 = default_gs;
  }

  // Correlated ghost 1 supperssion option in TSM
  m_tsmcgs1 = ps.getParameter<bool>("TSMCGS1");
  
  // Correlated ghost 2 supperssion option in TSM
  m_tsmcgs2 = ps.getParameter<bool>("TSMCGS2");

  // Handling carry in case of pile-up
  int myhsp = ps.getParameter<int>("TSMHSP");
  if (myhsp>=0 && myhsp<3)
    m_tsmhsp = myhsp;
  else {
    std::cout << "DTConfigTSPhi::setDefaults : TSMHSP value is not correct! Default Used" << std::endl;
    m_tsmhsp = default_hsp;
  }

  // Handling TSMS masking parameters
  m_tsmword.one();
  int word = ps.getParameter<int>("TSMWORD");
  if (word<0 || word>255){
    std::cout << "DTConfigTSPhi::setDefaults : TSMWORD value is not correct! Default Used" << std::endl;
    word = default_tsmword;
  }
  for (int i=0;i<7;i++){
    short int bit = word%2;
    word /= 2;
    if (bit==0) m_tsmword.unset(i);
  }

  //! Enabled TRACOs in TS
  m_tstren.one();
  for (int i=0;i<24;i++){
    std::stringstream os;
    os << "TSTREN" << i;
    if (ps.getParameter<bool>(os.str())== 0)
      m_tstren.unset(i);
  }
		  
}

int 
DTConfigTSPhi::TSSinTSMD(int stat, int sect) { //CB it should set value when constructor is called (it should be done when we have station by station config)  

  // Number of TSS for each TSMD (it changes from station to station) The DT stations are indicated in parenthesis
  // in the DT column.
  //
  //      MB                    nb.TSS        nb.TTS per TSMD
  //      1                       3             2   
  //      2                       4             2  
  //      3                       5             3  
  //      4(1,2,3,5,6,7)          6             3   
  //      4(8,12)                 6             3   
  //      4(9,11)                 3             2     
  //      4(4L)                   5             3    
  //      4(4R)                   5             3    
  //      4(10L)                  4             2     
  //      4(10R)                  4             2     
  
  if( stat==1 ||
      stat==2 ||
      ( stat==4 && (sect==9 || sect==11 ||
		    sect==10))) {
    m_ntsstsmd = 2;
  } else {
    m_ntsstsmd = 3;
  }

  return (int)m_ntsstsmd;

}

void 
DTConfigTSPhi::print() const {

  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*              DTTrigger configuration : TSPhi chips                         *" << std::endl;
  std::cout << "******************************************************************************" << std::endl << std::endl;
  std::cout << "Debug flag : " <<  debug() << std::endl;
  std::cout << "               TSS Parameters:" << std::endl;
  std::cout <<  "TSSMSK 1/2:" << TssMasking(0) << " " <<  TssMasking(1) << std::endl;
  std::cout << "TSSHTE 1/2/carry :" << TssHtrigEna(0) << " " << TssHtrigEna(1) << " " << TssHtrigEnaCarry() << std::endl;
  std::cout << "TSSNOE 1/2/carry :" << TssInOutEna(0) << " " << TssInOutEna(1) << " " << TssInOutEnaCarry() << std::endl;
  std::cout << "TSSCCE 1/2/carry :" << TssCorrEna(0)  << " " << TssCorrEna(1)  << " " << TssCorrEnaCarry()  << std::endl;
  std::cout << "TSSGS 1/2:" << TssGhost1Flag() << " " << TssGhost2Flag() << std::endl;
  std::cout << "TSSCGS 1/2:" << TssGhost1Corr() << " " << TssGhost2Corr() << std::endl;
  std::cout << "               TSM Parameters:" << std::endl;
  std::cout << "TSMMSK 1/2:" << TsmMasking(0) << " " <<  TsmMasking(1) << std::endl;
  std::cout << "TSMHTE 1/2/carry :" << TsmHtrigEna(0) << " " << TsmHtrigEna(1) << " " << TsmHtrigEnaCarry() << std::endl;
  std::cout << "TSMNOE 1/2/carry :" << TsmInOutEna(0) << " " << TsmInOutEna(1) << " " << TsmInOutEnaCarry() << std::endl;
  std::cout << "TSMCCE 1/2/carry :" << TsmCorrEna(0)  << " " << TsmCorrEna(1)  << " " << TsmCorrEnaCarry()  << std::endl;
  std::cout << "TSMGS 1/2:" <<  TsmGhost1Flag() << " " << TsmGhost2Flag() << std::endl;
  std::cout << "TSMCGS 1/2:" <<  TsmGhost1Corr() << " " << TsmGhost2Corr() << std::endl;
  std::cout << "TSMHSP :" << TsmGetCarryFlag() << std::endl;
  std::cout << "TSTREN[i] :";
  for (int i=1;i<25;i++) std::cout << usedTraco(i) << " ";
  std::cout << std::endl;
  std::cout << "TSMWORD :";
  TsmStatus().print();
  std::cout << std::endl;
//   int stat=4, sect=5;
//   std::cout << "TSSTSMD(4,14 :" <<  TSSinTSMD(stat,sect) << std::endl;
  std::cout << "******************************************************************************" << std::endl;

}

bool
DTConfigTSPhi::checkMask(int msk){
  
  bool hasone = false;
  bool hastwo = false;
  bool hasthree = false;
  for(int i=0;i<3;i++){
    int mynum = msk%10;
    switch (mynum){
    case 1:
      hasone = true;
      break;
    case 2:
      hastwo = true;
      break;
    case 3:
      hasthree =true;
      break;
    }
    msk /= 10;
  }
  if (hasone==true && hastwo==true && hasthree==true) return true;
  return false;

}

    
