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
#include <cstring>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/Utilities/interface/Exception.h"

//----------------
// Constructors --
//----------------
DTConfigTSPhi::DTConfigTSPhi(const edm::ParameterSet& ps) { 

  setDefaults(ps);
  if (debug()) print();
}

DTConfigTSPhi::DTConfigTSPhi(bool debugTS, unsigned short int tss_buffer[7][31], int ntss, unsigned short int tsm_buffer[9]) { 

  if (ntss == 0) {
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::DTConfigTSPhi : ntss=" << ntss << std::endl
				  << "configuration from CCB strings not possible " << std::endl
				  << "if error occurs configuring from DB this is " << std::endl
                                  << "likely to be a DTCCBConfigRcd issue" << std::endl;
  }

  m_debug = debugTS;

  bool tstren[24];
  bool tsscgs2, tsscgs1, tsscce1, tsshte1, tssnoe1, carrytss; 
  bool tsscce2, tsshte2, tssnoe2, tssccec, tsshtec, tssnoec;
  unsigned short  tssgs1, tssgs2, tssmsk1, tssmsk2;

  tsscgs2 = tsscgs1 = tsscce1 = tsshte1 = tssnoe1 = carrytss = 0;
  tsscce2 = tsshte2 = tssnoe2 = tssccec = tsshtec = tssnoec  = 0;
  tssgs1 = tssgs2 = tssmsk1 = tssmsk2 = 0;

  memset(tstren,true,24*sizeof(bool));

  // TSS unpacking
  for (int itss=0; itss<ntss; itss++) {     
    unsigned short int memory_tss[27];

    for(int ts=0;ts<27;ts++){
      memory_tss[ts] = tss_buffer[itss][ts+4];
      //std::cout << std::hex << memory_tss[ts] << " ";
    }
      
    tstren[itss*4]   = !(memory_tss[1]&0x08);
    tstren[itss*4+1] = !(memory_tss[1]&0x80);
    tstren[itss*4+2] = !(memory_tss[2]&0x08);
    tstren[itss*4+3] = !(memory_tss[2]&0x80);

    if(!itss) {
      tsscgs2 = !(memory_tss[0]&0x08);
      tssgs2  = memory_tss[0]&0x04 ? 0 : 1;
      tsscgs1 = !(memory_tss[0]&0x02);
      tssgs1  = memory_tss[0]&0x01 ? 0 : 1;
      tsscce1 = !(memory_tss[4]&0x04); 
      tsshte1 = !(memory_tss[4]&0x02); 
      tssnoe1 = !(memory_tss[4]&0x01);
      tsscce2 = !(memory_tss[3]&0x40); 
      tsshte2 = !(memory_tss[3]&0x20); 
      tssnoe2 = !(memory_tss[3]&0x10); 
      tssccec = !(memory_tss[3]&0x04); 
      tsshtec = !(memory_tss[3]&0x02); 
      tssnoec = !(memory_tss[3]&0x01);
      carrytss= !(memory_tss[4]&0x40);
      tssmsk1  = memory_tss[4]&0x10 ? 132 : 312;
      tssmsk2  = memory_tss[4]&0x20 ? 132 : 312;
    }
  }

  // TSM unpacking
  unsigned short int memory_tsms[2], memory_tsmdu[2], memory_tsmdd[2];

  for(int ts=0;ts<2;ts++){
    memory_tsms[ts]  = tsm_buffer[ts+3];
    memory_tsmdu[ts] = tsm_buffer[ts+5];
    memory_tsmdd[ts] = tsm_buffer[ts+7];
    //std::cout << std::hex << memory_tsms[ts] << " " 
    //<< memory_tsmdu[ts] << " " << memory_tsmdd[ts] << " " << std::endl;
  }

  bool tsmcgs1 = true;
  unsigned short  tsmgs1  = memory_tsms[1]&0x02 ? 0 : 1;
  bool tsmcgs2 = true;
  unsigned short  tsmgs2  = 1;
  bool tsmcce1 = true; 
  bool tsmhte1 = true;  
  bool tsmnoe1 = true; 
  bool tsmcce2 = true; 
  bool tsmhte2 = true; 
  bool tsmnoe2 = true;
  bool tsmccec = true; 
  bool tsmhtec = true; 
  bool tsmnoec = true;
  bool carrytsms = !(memory_tsms[1]&0x01);
  unsigned short tsmmsk1  = memory_tsms[1]&0x08 ? 321 : 312;
  unsigned short tsmmsk2  = tsmmsk1;
  bool tsmword[8];
  tsmword[0] = !((memory_tsmdu[0]&0x80)&&(memory_tsmdd[0]&0x80)); 
  tsmword[1] = !(memory_tsms[0]&0x01); 
  tsmword[2] = !(memory_tsms[0]&0x02);
  tsmword[3] = !(memory_tsms[0]&0x04);
  tsmword[4] = !(memory_tsms[0]&0x08); 
  tsmword[5] = !(memory_tsms[0]&0x10);
  tsmword[6] = !(memory_tsms[0]&0x20); 
  tsmword[7] = !(memory_tsms[0]&0x40); 
  bool carrytsmd = !((memory_tsmdu[0]&0x10)&&(memory_tsmdd[0]&0x10));
    
  unsigned short tsmhsp = carrytss && carrytsms && carrytsmd;

  if (debug()) {
    std::cout << "TSS :" << std::dec << std::endl << "tstren= " ;
    for (int i=0; i<24 ;i++) std::cout << tstren[i] << " ";
    std::cout << " tsscgs1="  << tsscgs1
	      << " tssgs1="   << tssgs1
	      << " tsscgs2="  << tsscgs2
	      << " tssgs2="   << tssgs2
	      << " tsscce1="  << tsscce1
	      << " tsshte1="  << tsshte1
	      << " tssnoe1="  << tssnoe1
	      << " tsscce2="  << tsscce2
	      << " tsshte2="  << tsshte2
	      << " tssnoe2="  << tssnoe2
	      << " tssccec="  << tssccec
	      << " tsshtec="  << tsshtec
	      << " tssnoec="  << tssnoec
	      << " carrytss=" << carrytss
	      << " tssmsk1="  << tssmsk1
	      << " tssmsk2="  << tssmsk2 << std::endl;

    std::cout << "TSM : "<< std::endl
	      << "tsmcgs1="  << tsmcgs1
	      << " tsmgs1="   << tsmgs1
	      << " tsmcgs2="  << tsmcgs2
	      << " tsmgs2="   << tsmgs2
	      << " tsmcce1="  << tsmcce1
	      << " tsmhte1="  << tsmhte1
	      << " tsmnoe1="  << tsmnoe1
	      << " tsmcce2="  << tsmcce2
	      << " tsmhte2="  << tsmhte2
	      << " tsmnoe2="  << tsmnoe2
	      << " tsmccec="  << tsmccec
	      << " tsmhtec="  << tsmhtec
	      << " tsmnoec="  << tsmnoec
	      << " tsmhsp=" << tsmhsp
	      << " carrytsms=" << carrytsms
	      << " carrytsmd=" << carrytsmd
	      << " tsmword="; 
    for (int i=0;i<8;i++) std::cout << tsmword[i] << " ";
    std::cout << " tsmmsk1="  << tsmmsk1
	      << " tsmmsk2="  << tsmmsk2 << std::endl;
  }

  setTssMasking(tssmsk1,1);
  setTssMasking(tssmsk2,2);
  setTssHtrigEna(tsshte1,1);
  setTssHtrigEna(tsshte2,2);
  setTssHtrigEnaCarry(tsshtec);
  setTssInOutEna(tssnoe1,1);
  setTssInOutEna(tssnoe2,2);
  setTssInOutEnaCarry(tssnoec);
  setTssCorrEna(tsscce1,1);
  setTssCorrEna(tsscce2,2);
  setTssCorrEnaCarry(tssccec);
  setTssGhost1Flag(tssgs1);
  setTssGhost2Flag(tssgs2);
  setTssGhost1Corr(tsscgs1);
  setTssGhost2Corr(tsscgs2);

  setTsmMasking(tsmmsk1,1);
  setTsmMasking(tsmmsk2,2);
  setTsmHtrigEna(tsmhte1,1);
  setTsmHtrigEna(tsmhte2,2);
  setTsmHtrigEnaCarry(tsmhtec);
  setTsmInOutEna(tsmnoe1,1);
  setTsmInOutEna(tsmnoe2,2);
  setTsmInOutEnaCarry(tsmnoec);
  setTsmCorrEna(tsmcce1,1);
  setTsmCorrEna(tsmcce2,2);
  setTsmCorrEnaCarry(tsmccec);
  setTsmGhost1Flag(tsmgs1);
  setTsmGhost2Flag(tsmgs2);
  setTsmGhost1Corr(tsmcgs1);
  setTsmGhost2Corr(tsmcgs2);
  setTsmCarryFlag(tsmhsp);

  for (int i=0;i<24;i++) setUsedTraco(i,tstren[i]);
  for (int i=0;i<8;i++) setTsmStatus(i,tsmword[i]);
  
    
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
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::setDefaults : TSSMSK1 not in correct form: " << mymsk << std::endl;
  }
  
  // Order of quaity bits in TSS for sort2
  mymsk= ps.getParameter<int>("TSSMSK2");  
  if (checkMask(mymsk)) 
    m_tssmsk[1] = mymsk;
  else {
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::setDefaults : TSSMSK2 not in correct form: " << mymsk << std::endl;
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
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::setDefaults : TSSGS1 value is not correct: " << mygs << std::endl;
  }
  
  // Ghost 2 supperssion option in TSS
  mygs= ps.getParameter<int>("TSSGS2");
  if (mygs>=0 && mygs<5)
    m_tssgs2 = mygs;
  else {
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::setDefaults : TSSGS2 value is not correct: " << mygs << std::endl;
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
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::setDefaults : TSMMSK1 not in correct form: " << mymsk << std::endl;
  }

  // Order of quaity bits in TSM for sort2
  mymsk= ps.getParameter<int>("TSMMSK2");
  if (checkMask(mymsk)) 
    m_tsmmsk[1] = mymsk;
  else {
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::setDefaults : TSMMSK2 not in correct form: " << mymsk << std::endl;
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
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::setDefaults : TSMGS1 value is not correct: " << mygs << std::endl;
  }

  // Ghost 2 supperssion option in TSM
  mygs= ps.getParameter<int>("TSMGS2");
  if (mygs>=0 && mygs<5)
    m_tsmgs2 = mygs;
  else {
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::setDefaults : TSMGS2 value is not correct: " << mygs << std::endl;
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
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::setDefaults : TSMHSP value is not correct: " << myhsp << std::endl;
  }

  // Handling TSMS masking parameters
  m_tsmword.one();
  int word = ps.getParameter<int>("TSMWORD");
  if (word<0 || word>255){
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::setDefaults : TSMWORD value is not correct: " << word << std::endl;
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

void 
DTConfigTSPhi::setTssMasking(unsigned short tssmsk, int i) {
  if (checkMask(tssmsk)) 
    m_tssmsk[i-1] = tssmsk;
  else {
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::setTssMasking : TSSMSK" << i << " not in correct form: " << tssmsk << std::endl;
  }
}

void 
DTConfigTSPhi::setTsmMasking(unsigned short tsmmsk, int i) {
  if (checkMask(tsmmsk)) 
    m_tsmmsk[i-1] = tsmmsk;
  else {
    throw cms::Exception("DTTPG") << "DTConfigTSPhi::setTsmMasking : TSMMSK" << i << " not in correct form: " << tsmmsk << std::endl;
  }
}
  
bool
DTConfigTSPhi::checkMask(unsigned short msk){
  
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

    
