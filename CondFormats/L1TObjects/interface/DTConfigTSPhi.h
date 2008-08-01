//-------------------------------------------------
//
/**  \class DTConfigTSPhi
 *
 *   Configurable parameters and constants 
 *   for Level-1 Muon DT Trigger - TS Phi
 *
 *
 *   \author C. Battilana
 *
 */
//
//--------------------------------------------------
#ifndef DT_CONFIG_TSPHI_H
#define DT_CONFIG_TSPHI_H

//---------------
// C++ Headers --
//---------------

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/L1TObjects/interface/BitArray.h"
#include "CondFormats/L1TObjects/interface/DTConfig.h"
#include "boost/cstdint.hpp"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTConfigTSPhi : public DTConfig {

  public:
  
  //! Constant: maximum number of TSS in input to the TSM
  static const int NTSSTSM = 7;
  
  //! Constant: number of TSMD
  static const int NTSMD = 2;
  
  //! Constant: maximum number of TSS in input to a single TSMD
  static const int NTSSTSMD = 3;

  //! Constant: default TS mask parameter;
  static const int default_tsmsk = 312;

  //! Constant: default Ghost Suppression Option (1/2) parameter;
  static const int default_gs = 1;

  //! Constant: default handling of 2nd track parameter;
  static const int default_hsp = 1;  

  //! Constant: default tsmword parameter;
  static const int default_tsmword = 255;
  
  //! Constructor
  DTConfigTSPhi(const edm:: ParameterSet& ps);

  //! Constructor
  DTConfigTSPhi() {};

  //! Destructor
  ~DTConfigTSPhi();

  //! Return the debug flag
  inline bool debug() const { return m_debug; }

  //! Order of quality bits in TSS for sort1/2
  inline int  TssMasking(int i) const { return (int)m_tssmsk[i]; }
 
  //! Enable Htrig checking in TSS for sort1/2
  inline bool  TssHtrigEna(int i) const { return m_tsshte[i]; }

  //! Enable Htrig checking in TSS for carry
  inline bool  TssHtrigEnaCarry() const { return m_tsshte[2]; }
  
  //! Enable Inner SL checking in TSS for sort1/2
  inline bool  TssInOutEna(int i) const { return m_tssnoe[i]; }
  
  //! Enable Inner SL checking in TSS for carry
  inline bool  TssInOutEnaCarry() const { return m_tssnoe[2]; }
  
  //! Enable Correlation checking in TSS for sort1/2
  inline bool  TssCorrEna(int i) const { return m_tsscce[i]; }
  
  //! Enable Correlation checking in TSS for carry
  inline bool  TssCorrEnaCarry() const { return m_tsscce[2]; }
  
  //! Order of quality bits in TSM for sort1/2
  inline int  TsmMasking(int i) const { return (int)m_tsmmsk[i]; }
  
  //! Enable Htrig checking in TSM for sort1/2
  inline bool  TsmHtrigEna(int i) const { return m_tsmhte[i]; }
  
  //! Enable Htrig checking in TSM for carry
  inline bool  TsmHtrigEnaCarry() const { return m_tsmhte[2]; }
  
  //! Enable Inner SL checking in TSM for sort1/2
  inline bool  TsmInOutEna(int i) const { return m_tsmnoe[i]; }
  
  //! Enable Inner SL checking in TSM for carry
  inline bool  TsmInOutEnaCarry() const { return m_tsmnoe[2]; }
  
  //! Enable Correlation checking in TSM  for sort1/2
  inline bool  TsmCorrEna(int i) const { return m_tsmcce[i]; }
    
  //! Enable Correlation checking in TSM  for carry
  inline bool  TsmCorrEnaCarry() const { return m_tsmcce[2]; }
    
  //! Ghost 1 suppression option in TSS
  inline int  TssGhost1Flag() const { return (int)m_tssgs1; }
  
  //! Ghost 2 suppression option in TSS
  inline int  TssGhost2Flag() const { return (int)m_tssgs2; }

  //! Ghost 1 suppression option in TSM
  inline int  TsmGhost1Flag() const { return (int)m_tsmgs1; }
  
  //! Ghost 2 suppression option in TSM
  inline int  TsmGhost2Flag() const { return (int)m_tsmgs2; }
  
  //! Correlated ghost 1 suppression option in TSS
  inline bool  TssGhost1Corr() const { return m_tsscgs1; }
  
  //! Correlated ghost 2 suppression option in TSS
  inline bool  TssGhost2Corr() const { return m_tsscgs2; }

  //! Correlated ghost 1 suppression option in TSM
  inline bool  TsmGhost1Corr() const { return m_tsmcgs1; }
  
  //! Correlated ghost 2 suppression option in TSM
  inline bool  TsmGhost2Corr() const { return m_tsmcgs2; }

  //! Handling of second track (carry) in case of pile-up, in TSM 
  inline int  TsmGetCarryFlag() const { return (int)m_tsmhsp; }

  //! Enabled TRACOs in TS
  inline bool usedTraco(int i) const { return (bool) m_tstren.element(i-1); }

  //! TSM status
  inline BitArray<8> TsmStatus() const { return m_tsmword; };

  // DBSM-doubleTSM
  //! Return the max nb. of TSSs in input to a single TSMD (called ONLY in back-up mode)
  int TSSinTSMD(int stat, int sect);

  //! Print the setup
  void print() const ;

  private:

  //! Check mask correctness
  bool checkMask(int);

  //! Load pset values into class variables
  void setDefaults(const edm:: ParameterSet& ps);

  bool m_debug;

  // TSS Parameters
  unsigned short int m_tssmsk[2];  // [0]=1st [1]=2nd
  bool m_tsshte[3]; // [0]=1st [1]=2nd [2]=carry
  bool m_tssnoe[3]; // [0]=1st [1]=2nd [2]=carry
  bool m_tsscce[3]; // [0]=1st [1]=2nd [2]=carry
  unsigned short int m_tssgs1;
  unsigned short int m_tssgs2;
  bool m_tsscgs1;
  bool m_tsscgs2;

  //TSM Parameters
  unsigned short int m_tsmmsk[2];  // [0]=1st [1]=2nd
  bool m_tsmhte[3]; // [0]=1st [1]=2nd [2]=carry
  bool m_tsmnoe[3]; // [0]=1st [1]=2nd [2]=carry
  bool m_tsmcce[3]; // [0]=1st [1]=2nd [2]=carry
  unsigned short int m_tsmgs1; 
  unsigned short int m_tsmgs2;
  bool m_tsmcgs1;
  bool m_tsmcgs2;
  unsigned short int m_tsmhsp;

  
  BitArray<24> m_tstren;     // Enabled TRACOs
  BitArray<8> m_tsmword; // TSM backup mode word
  unsigned short int m_ntsstsmd;        // nb tss to one of the tsmd (only if back-up mode)

};

#endif
