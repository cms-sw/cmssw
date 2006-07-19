//-------------------------------------------------
//
/**  \class DTConfig
 *
 *   Configurable parameters and constants 
 *   for Level-1 Muon DT Trigger 
 *
 *
 *   $Date: 2004/07/14 14:20:25 $
 *   $Revision: 1.20 $
 *
 *   \author  C. Grandi, S. Vanini, S. Marcellini, D. Bonacorsi
 *
 *   Modifications: 
 *   23/X/02  SV : AC1,AC2,ACH,ACL added
 *   12/XI/02 SV : 4ST3 and 4RE3 added
 *   15/XI/02 SV : setParamValue method added
 +   10/II/04 SM : Incluide TSM back up mode stuff
 *   22/VI/04 SV : ST43 and RE43 - digi offset moved to BtiCard
 *   27/V/05  SV : testbeam 2004 update
 *   17/VI/05 SV : bti mask in traco
 */
//
//--------------------------------------------------
#ifndef DT_CONFIG_H
#define DT_CONFIG_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "L1Trigger/DTUtilities/interface/DTParameter.h"
#include "L1Trigger/DTUtilities//interface/BitArray.h"
//----------------------
// Base Class Headers --
//----------------------

//---------------
// C++ Headers --
//---------------
#include <vector>
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------
typedef std::vector<DTParameter>                       ParamContainer;
typedef ParamContainer::iterator                       ParamIterator;
typedef ParamContainer::const_iterator                 ParamConstIterator;

class DTConfig {

  public:

  //! Constructor
  DTConfig();
  
  //! Destructor 
  ~DTConfig();

  //! Constants

  enum {
    //! number of steps
    NSTEPL=24,
    //NSTEPL=30 ,       
    //! first step to start trigger finding 
    NSTEPF=9 ,
    //! Width of a TRACO (number of BTI's in inner plane)
    NBTITC=4 ,       
    //! Number of TRACOs in input to a TSS (XMTSV(10))
    NTCTSS=4 ,   
    //! maximum number of TSS in input to the TSM
    NTSSTSM=7 ,      
    //! Number of cell (BTI) in theta view planes
    NCELLTH=57,      
    //! Resolution for psi and DeltaPsiR (phi_B)
    RESOLPSI=512,    
    //! Resulution for psiR (phi)
    RESOLPSIR=4096,
    //! number of TSMD
    NTSMD=2,
    //! maximum number of TSS in input to a single TSMD
    NTSSTSMD=3 ,      
    //! sector collector: maximum number of TSMD in input to SC
    NTSMSC = 4
  }; 

  //! Set a parameter to a given value of allowed values list
  void setParam(std::string, std::string); 
    
  //! Set a parameter to a given value
  void setParamValue(std::string, std::string, float); 
    
  //! Add a parameter
  void addParam(DTParameter);

//    //! Return the parameter value meaning. Search with index to speed up
//    std::string paramMeaning(unsigned) const;

//    //! Return the value of a parameter. Search with index to speed up
//    double paramValue(unsigned) const;

  //! Return the parameter value meaning 
  // Search with index to speed up
  inline std::string paramMeaning(unsigned index) const { return param_[index].currentMeaning(); }

  //! Return the value of a parameter
  // Search with index to speed up
  inline double paramValue(unsigned index) const { return param_[index].currentValue(); }

  //! debug flag
  inline int debug() const { return (int)paramValue(0); }

  //! max drift time in 25 ns steps - now obsolete!                           (-XDTBX(1))
  inline float TMAX() const { 
     return (float)(0.5*(0.75*paramValue(115) + 0.25*paramValue(116))); }

  //! max drift time in 12.5 ns steps - SV
  inline float ST() const {
     return (float)( 0.75*paramValue(115) + 0.25*paramValue(116) ); }

  //! max drift time in 12.5 ns steps
  inline int lstep() const { return (int)(2*TMAX()); }

  //! Time indep. K equation suppression                             (XMTFL(3))
  inline int  TimIndKEqSupp() const { return (int)paramValue(2); }
  
  //! superlayer LTS flag                                    (MOD(XMTFL(4),10))
  inline int  slLTS() const { return (int)paramValue(3); }
  
  //! side LTS flag                                         (INT(XMTFL(4)/10-2)
  inline int  sideLTS() const { return (int)paramValue(4); }
  
  //! number of bx suppressed at high side with LTS for each SL      (XMTFL(5))
  inline int  nbxLTS(int i) const { 
    if(i==1) return (int)paramValue(6);
    return (int)paramValue(5); 
  }
  
  //! suppr. of LTRIG in BTI adj. to HTRIG                         (! XMTFL(7))
  inline int  adjBtiLts() const { return (int)paramValue(7); }
  
  //! BTI setup time : in tdc units!!!                             (XMTCT(1))
  inline float  SetupTime() const {
    float time = (float)paramValue(8);
    return time; }

  //only for TB 2004 purpose - 127 for MB1 and 128 for MB3
  inline int  SetupTimeMB1() const {
    return (int)paramValue(127);
  }
  inline int  SetupTimeMB3() const {
    return (int)paramValue(128);
  }
  inline int  SetupTimeTHETA() const {
    return (int)paramValue(129);
  }

  
  //! large angle BTI corr                                           (XMTCT(2))
  inline float  LAngBTICorr() const { return (float)paramValue(9); }
  
  //! Max K param accepted                              (XMTCT(3) and XMTCT(7))
  inline int  KCut(int i) const {
    if(i==1) return (int)paramValue(11);
    return (int)paramValue(10); 
  }
  
  //! acceptance pattern A                                           (XMTCT(4))
  inline int  AccPattA() const { return (int)paramValue(12); }
  
  //! acceptance pattern B                                           (XMTCT(5))
  inline int  AccPattB() const { return (int)paramValue(13); }
  
  //! BTI angular acceptance in theta view                           (XMTCT(8))
  inline int  KAccTheta() const { return (int)paramValue(14); }
  
  //! bending angle cut in stations 3 and 4 (0=no cut, other=N-1)   (XMTCT(10))
  inline int BendingAngleCut() const { 
    if(paramMeaning(15)=="Numerical Value")
      return (int)paramValue(15)+1;
    return 0;
  }
 
  //! ascend. order for K sorting first/second tracks              (! XMTCR(1))
  inline int sortKascend(int i) const { 
    if(i==0)
      return (int)paramValue(16);
    else
      return (int)paramValue(17);
  }
     
  //! preference to HTRIG on first/second tracks                     (XMTCR(2))
  inline int prefHtrig(int i) const { 
    if(i==0)
      return (int)paramValue(18);
    else
      return (int)paramValue(19);
  }
  
  //! single HTRIG enabling on first/second tracks                   (XMTCR(3))
  inline int  singleHflag(int i) const { 
    if(i==0)
      return (int)paramValue(20);
    else
      return (int)paramValue(21);
  }

  //! single LTRIG enabling on first/second tracks                   (XMTCR(4))
  inline int  singleLflag(int i) const { 
    int l=(i==0)?(int)paramValue(22):(int)paramValue(23);
    if(l<=3)
      return l;
    else
      return 0;
  }

  //! preference to inner on first/second tracks                    (!XMTCR(5))
  inline int  prefInner(int i) const { 
    if(i==0)
      return (int)paramValue(24);
    else
      return (int)paramValue(25);
  }
 
  //! K tollerance for correlation in TRACO            ( XMTCR(6) and XMTCR(7))
  inline int  TcKToll(int i) const { 
    if(i==0)
      return (int)paramValue(26);
    else
      return (int)paramValue(27);
  }

  
  //! single LTRIG accept enabling on first/second tracks            (XMTCR(8))
  inline int  singleLenab(int i) const { 
    return (int)paramValue(123);
  }

  
  //! suppr. of LTRIG in 4 BX before HTRIG                           (XMTCR(9))
  inline int  TcBxLts() const { return (int)paramValue(28); }
  
  //! recycling of TRACO cand. in inner/outer SL                    ( XMTFL(8))
  inline int  TcReuse(int i) const { 
    if(i==0)
      return (int)paramValue(29);
    else
      return (int)paramValue(30);
  }
  
  //! maximum number of TRACO output candidates        (not present in FORTRAN)
  inline int nMaxOutCand() const { return (int)paramValue(31); }

  //! Order of quality bits in TSS for sort1/2         (not present in FORTRAN)
  inline int  TssMasking(int i) const { 
    if(i==0)
      return (int)paramValue(32);
    else
      return (int)paramValue(33);
  }
  
  //! enable Htrig checking in TSS for sort1/2         (not present in FORTRAN)
  inline int  TssHtrigEna(int i) const { 
    if(i==0)
      return (int)paramValue(34);
    else
      return (int)paramValue(35);
  }
  
  //! enable Htrig checking in TSS for carry         (not present in FORTRAN)
  inline int  TssHtrigEnaCarry() const { return (int)paramValue(36); }
  
  //! enable Inner SL checking in TSS for sort1/2      (not present in FORTRAN)
  inline int  TssInOutEna(int i) const { 
    if(i==0)
      return (int)paramValue(37);
    else
      return (int)paramValue(38);
  }
  
  //! enable Inner SL checking in TSS for carry      (not present in FORTRAN)
  inline int  TssInOutEnaCarry() const { return (int)paramValue(39); }
  
  //! enable Correlation checking in TSS for sort1/2   (not present in FORTRAN)
  inline int  TssCorrEna(int i) const { 
    if(i==0)
      return (int)paramValue(40);
    else
      return (int)paramValue(41);
  }
  
  //! enable Correlation checking in TSS for carry   (not present in FORTRAN)
  inline int  TssCorrEnaCarry() const { return (int)paramValue(42); }
  
  //! Order of quality bits in TSM for sort1/2         (not present in FORTRAN)
  inline int  TsmMasking(int i) const { 
    if(i==0)
      return (int)paramValue(43);
    else
      return (int)paramValue(44);
  }
  
  //! enable Htrig checking in TSM for sort1/2         (not present in FORTRAN)
  inline int  TsmHtrigEna(int i) const { 
    if(i==0)
      return (int)paramValue(45);
    else
      return (int)paramValue(46);
  }
  
  //! enable Htrig checking in TSM for carry         (not present in FORTRAN)
  inline int  TsmHtrigEnaCarry() const { return (int)paramValue(47); }
  
  //! enable Inner SL checking in TSM for sort1/2      (not present in FORTRAN)
  inline int  TsmInOutEna(int i) const { 
    if(i==0)
      return (int)paramValue(48);
    else
      return (int)paramValue(49);
  }
  
  //! enable Inner SL checking in TSM for carry      (not present in FORTRAN)
  inline int  TsmInOutEnaCarry() const { return (int)paramValue(50); }
  
  //! enable Correlation checking in TSM  for sort1/2  (not present in FORTRAN)
  inline int  TsmCorrEna(int i) const { 
    if(i==0)
      return (int)paramValue(51);
    else
      return (int)paramValue(52);
  }
    
  //! enable Correlation checking in TSM  for carry  (not present in FORTRAN)
  inline int  TsmCorrEnaCarry() const { return (int)paramValue(53); }
    
  //! ghost 1 suppression option in TSS                 (XMTSV(8) and XMTSV(9))
  inline int  TssGhost1Flag() const { return (int)paramValue(54); }
  
  //! ghost 2 suppression option in TSS                              (XMTSV(8))
  inline int  TssGhost2Flag() const { return (int)paramValue(55); }

  //! ghost 1 suppression option in TSM                 (XMTSV(8) and XMTSV(9))
  inline int  TsmGhost1Flag() const { return (int)paramValue(56); }
  
  //! ghost 2 suppression option in TSM                              (XMTSV(8))
  inline int  TsmGhost2Flag() const { return (int)paramValue(57); }
  
  //! correlated ghost 1 suppression option in TSS
  inline int  TssGhost1Corr() const { return (int)paramValue(58); }
  
  //! correlated ghost 2 suppression option in TSS
  inline int  TssGhost2Corr() const { return (int)paramValue(59); }

  //! correlated ghost 1 suppression option in TSM
  inline int  TsmGhost1Corr() const { return (int)paramValue(60); }
  
  //! correlated ghost 2 suppression option in TSM
  inline int  TsmGhost2Corr() const { return (int)paramValue(61); }

  //! Handling of second track (carry) in case of pile-up, in TSM 
  inline int  TsmGetCarryFlag() const { return (int)paramValue(62); }

  //! acceptance pattern AC1                                           
  inline int  AccPattAC1() const { return (int)paramValue(63); }
  
  //! acceptance pattern AC2                                           
  inline int  AccPattAC2() const { return (int)paramValue(64); }
  
  //! acceptance pattern ACH                                           
  inline int  AccPattACH() const { return (int)paramValue(65); }
  
  //! acceptance pattern ACL                                           
  inline int  AccPattACL() const { return (int)paramValue(66); }

  //! redundant patterns flag RON
  inline bool RONflag() const { return (int)paramValue(67); }

  //! pattern mask flag 
  inline int PTMSflag(int patt) { return (int)paramValue(68+patt); }

  //! wire mask flag 
  inline int WENflag(int wire) { return (int)paramValue(99+wire); }

  //! K left limit for left traco
  inline int LL() { return (int)paramValue(109); }

  //! K right limit for left traco
  inline int LH() { return (int)paramValue(110); }

  //! K left limit for center traco
  inline int CL() { return (int)paramValue(111); }

  //! K right limit for center traco
  inline int CH() { return (int)paramValue(112); }

  //! K left limit for right traco
  inline int RL() { return (int)paramValue(113); }

  //! K right limit for right traco
  inline int RH() { return (int)paramValue(114); }

  //! now one for each bti:
  //! K left limit for left traco
  inline int LL_bti(int n, int sl)  { return LLvec[n-1][sl-1]; }
  inline void setLL_bti(int n, int sl, int val) { LLvec[n-1][sl-1]=val; }

  //! K right limit for left traco
  inline int LH_bti(int n, int sl) { return LHvec[n-1][sl-1]; }
  inline void setLH_bti(int n, int sl, int val) { LHvec[n-1][sl-1]=val; }

  //! K left limit for center traco
  inline int CL_bti(int n, int sl) { return CLvec[n-1][sl-1]; }
  inline void setCL_bti(int n, int sl,int val) { CLvec[n-1][sl-1]=val; }

  //! K right limit for center traco
  inline int CH_bti(int n, int sl) { return CHvec[n-1][sl-1]; }
  inline void setCH_bti(int n, int sl,int val) { CHvec[n-1][sl-1]=val; }

  //! K left limit for right traco
  inline int RL_bti(int n, int sl) { return RLvec[n-1][sl-1]; }
  inline void setRL_bti(int n, int sl,int val) { RLvec[n-1][sl-1]=val; }

  //! K right limit for right traco
  inline int RH_bti(int n, int sl) { return RHvec[n-1][sl-1]; }
  inline void setRH_bti(int n, int sl,int val) { RHvec[n-1][sl-1]=val; }

  //! ST and RE parameters for drift velocity 
  inline int ST43() { return (int)paramValue(115); }
  inline int RE43() { return (int)paramValue(116); }

  //! wire DEAD time parameter
  inline int DEADpar() { return (int)paramValue(117);}

  //! flag for Low validation parameter
  inline int LVALIDIFHpar() { return (int)paramValue(118);}

  //! KRAD traco parameter
  inline int KRADpar() { return (int)paramValue(119);}

  //! BTIC traco parameter: must be equal to ST parameter!!!!
  inline int BTICpar() { return (int)paramValue(120);}
  //inline int BTICpar() { return STpar();}

  //! IBTIOFF traco parameter
  inline int IBTIOFFpar() { return (int)paramValue(121);}

  //! DD traco parameter
  inline int DDpar() { return (int)paramValue(122);}

  //! Setting only one traco in output
  inline int usedTraco(int traco) const {
    if(paramValue(124)==-1)
      return 1;
    else
      return (int(paramValue(124))>>(traco-1)) & 0x01; 
  } 

  //! Connected bti in traco
  inline int usedBti(int bti) const {
    if(paramValue(130)==-1)
      return 1;
    else
      return (int(paramValue(130))>>(bti-1)) & 0x01; 
  } 


  //! Geometry parameters flag
  inline int trigSetupGeom() const { return (int)paramValue(125);}

  //! Enabling Carry in Sector Collector (1 means enabled, 0 disabled)
  inline int  SCGetCarryFlag() const { return (int)paramValue(126); }

  //! print the setup
  void print() const ;

  //! bti configuration acceptance from file
  void inputBtiAccep(const char* setupfile); 

  //! TSM status
  BitArray<8> TsmStatus(int stat, int sect, int whee);

  // DBSM-doubleTSM
  //! Return the max nb. of TSSs in input to a single TSMD (called ONLY in back-up mode)
  int TSSinTSMD(int stat, int sect);
  
private:

  //! Create the general parameters with the list of available options
  void createParametersGeneral();

  //! Create the parameters for BTI with the list of available options
  void createParametersBTI();

  //! Create the parameters for TRACO with the list of available options
  void createParametersTRACO();

  //! Create the parameters for TS with the list of available options
  void createParametersTS();

  //! Create others parameters for BTI - integration by SV
  void createParametersBTInew();

  //! Create others parameters for TRACO - integration by SV
  void createParametersTRACOnew();

  //! Create parameters for Sector Collector - (S. Marcellini)
  void createParametersSC();

  //! Read the parameters from .orcarc and set the others to the defaults
  void readParameters();

public:

  int LLvec[72][3];
  int LHvec[72][3];
  int CLvec[72][3];
  int CHvec[72][3];
  int RLvec[72][3];
  int RHvec[72][3];

private:

  // integer parameters
  ParamContainer param_;

  BitArray<8> _TsmWord; // TSM backup mode word
  // DBSM-doubleTSM
  int _ntsstsmd;        // nb tss to one of the tsmd (only if back-up mode)
};

#endif
