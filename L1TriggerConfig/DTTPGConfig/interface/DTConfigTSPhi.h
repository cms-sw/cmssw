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
#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfig.h"

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

  //! Constructor
  DTConfigTSPhi(const edm::ParameterSet& ps);

  //! Constructor
  DTConfigTSPhi(){};

  //! Constructor
  DTConfigTSPhi(bool debug, unsigned short int tss_buffer[7][31], int ntss, unsigned short int tsm_buffer[9]);

  //! Destructor
  ~DTConfigTSPhi() override;

  //! Return the debug flag
  inline bool debug() const { return m_debug; }

  //! Order of quality bits in TSS for sort1/2
  inline int TssMasking(int i) const { return (int)m_tssmsk[i]; }

  //! Enable Htrig checking in TSS for sort1/2
  inline bool TssHtrigEna(int i) const { return m_tsshte[i]; }

  //! Enable Htrig checking in TSS for carry
  inline bool TssHtrigEnaCarry() const { return m_tsshte[2]; }

  //! Enable Inner SL checking in TSS for sort1/2
  inline bool TssInOutEna(int i) const { return m_tssnoe[i]; }

  //! Enable Inner SL checking in TSS for carry
  inline bool TssInOutEnaCarry() const { return m_tssnoe[2]; }

  //! Enable Correlation checking in TSS for sort1/2
  inline bool TssCorrEna(int i) const { return m_tsscce[i]; }

  //! Enable Correlation checking in TSS for carry
  inline bool TssCorrEnaCarry() const { return m_tsscce[2]; }

  //! Order of quality bits in TSM for sort1/2
  inline int TsmMasking(int i) const { return (int)m_tsmmsk[i]; }

  //! Enable Htrig checking in TSM for sort1/2
  inline bool TsmHtrigEna(int i) const { return m_tsmhte[i]; }

  //! Enable Htrig checking in TSM for carry
  inline bool TsmHtrigEnaCarry() const { return m_tsmhte[2]; }

  //! Enable Inner SL checking in TSM for sort1/2
  inline bool TsmInOutEna(int i) const { return m_tsmnoe[i]; }

  //! Enable Inner SL checking in TSM for carry
  inline bool TsmInOutEnaCarry() const { return m_tsmnoe[2]; }

  //! Enable Correlation checking in TSM  for sort1/2
  inline bool TsmCorrEna(int i) const { return m_tsmcce[i]; }

  //! Enable Correlation checking in TSM  for carry
  inline bool TsmCorrEnaCarry() const { return m_tsmcce[2]; }

  //! Ghost 1 suppression option in TSS
  inline int TssGhost1Flag() const { return (int)m_tssgs1; }

  //! Ghost 2 suppression option in TSS
  inline int TssGhost2Flag() const { return (int)m_tssgs2; }

  //! Ghost 1 suppression option in TSM
  inline int TsmGhost1Flag() const { return (int)m_tsmgs1; }

  //! Ghost 2 suppression option in TSM
  inline int TsmGhost2Flag() const { return (int)m_tsmgs2; }

  //! Correlated ghost 1 suppression option in TSS
  inline bool TssGhost1Corr() const { return m_tsscgs1; }

  //! Correlated ghost 2 suppression option in TSS
  inline bool TssGhost2Corr() const { return m_tsscgs2; }

  //! Correlated ghost 1 suppression option in TSM
  inline bool TsmGhost1Corr() const { return m_tsmcgs1; }

  //! Correlated ghost 2 suppression option in TSM
  inline bool TsmGhost2Corr() const { return m_tsmcgs2; }

  //! Handling of second track (carry) in case of pile-up, in TSM
  inline int TsmGetCarryFlag() const { return (int)m_tsmhsp; }

  //! Enabled TRACOs in TS
  inline bool usedTraco(int i) const { return (bool)m_tstren.element(i - 1); }

  //! TSM status
  inline BitArray<8> TsmStatus() const { return m_tsmword; };

  // DBSM-doubleTSM
  //! Return the max nb. of TSSs in input to a single TSMD (called ONLY in back-up mode)
  int TSSinTSMD(int stat, int sect) const;

  // Set Methods
  //! Set debug flag
  inline void setDebug(bool debug) { m_debug = debug; }

  //! Order of quality bits in TSS for sort1/2
  void setTssMasking(unsigned short int tssmsk, int i) { m_tssmsk[i - 1] = tssmsk; }

  //! Enable Htrig checking in TSS for sort1/2
  inline void setTssHtrigEna(bool tsshte, int i) { m_tsshte[i - 1] = tsshte; }

  //! Enable Htrig checking in TSS for carry
  inline void setTssHtrigEnaCarry(bool tsshte) { m_tsshte[2] = tsshte; }

  //! Enable Inner SL checking in TSS for sort1/2
  inline void setTssInOutEna(bool tssnoe, int i) { m_tssnoe[i - 1] = tssnoe; }

  //! Enable Inner SL checking in TSS for carry
  inline void setTssInOutEnaCarry(bool tssnoe) { m_tssnoe[2] = tssnoe; }

  //! Enable Correlation checking in TSS for sort1/2
  inline void setTssCorrEna(bool tsscce, int i) { m_tsscce[i - 1] = tsscce; }

  //! Enable Correlation checking in TSS for
  inline void setTssCorrEnaCarry(bool tsscce) { m_tsscce[2] = tsscce; }

  //! Order of quality bits in TSM for sort1/2
  void setTsmMasking(unsigned short int tsmmsk, int i) { m_tsmmsk[i - 1] = tsmmsk; }

  //! Enable Htrig checking in TSM for sort1/2
  inline void setTsmHtrigEna(bool tsmhte, int i) { m_tsmhte[i - 1] = tsmhte; }

  //! Enable Htrig checking in TSM for carry
  inline void setTsmHtrigEnaCarry(bool tsmhte) { m_tsmhte[2] = tsmhte; }

  //! Enable Inner SL checking in TSM for sort1/2
  inline void setTsmInOutEna(bool tsmnoe, int i) { m_tsmnoe[i - 1] = tsmnoe; }

  //! Enable Inner SL checking in TSM for carry
  inline void setTsmInOutEnaCarry(bool tsmnoe) { m_tsmnoe[2] = tsmnoe; }

  //! Enable Correlation checking in TSM  for sort1/2
  inline void setTsmCorrEna(bool tsmcce, int i) { m_tsmcce[i - 1] = tsmcce; }

  //! Enable Correlation checking in TSM  for carry
  inline void setTsmCorrEnaCarry(bool tsmcce) { m_tsmcce[2] = tsmcce; }

  //! Ghost 1 suppression option in TSS
  inline void setTssGhost1Flag(unsigned short tssgs1) { m_tssgs1 = tssgs1; }

  //! Ghost 2 suppression option in TSS
  inline void setTssGhost2Flag(unsigned short tssgs2) { m_tssgs2 = tssgs2; }

  //! Ghost 1 suppression option in TSM
  inline void setTsmGhost1Flag(unsigned short tsmgs1) { m_tsmgs1 = tsmgs1; }

  //! Ghost 2 suppression option in TSM
  inline void setTsmGhost2Flag(unsigned short tsmgs2) { m_tsmgs2 = tsmgs2; }

  //! Correlated ghost 1 suppression option in TSS
  inline void setTssGhost1Corr(bool tsscgs1) { m_tsscgs1 = tsscgs1; }

  //! Correlated ghost 2 suppression option in TSS
  inline void setTssGhost2Corr(bool tsscgs2) { m_tsscgs2 = tsscgs2; }

  //! Correlated ghost 1 suppression option in TSM
  inline void setTsmGhost1Corr(bool tsmcgs1) { m_tsmcgs1 = tsmcgs1; }

  //! Correlated ghost 2 suppression option in TSM
  inline void setTsmGhost2Corr(bool tsmcgs2) { m_tsmcgs2 = tsmcgs2; }

  //! Handling of second track (carry) in case of pile-up, in TSM
  inline void setTsmCarryFlag(unsigned short tsmhsp) { m_tsmhsp = tsmhsp; }

  //! Enabled TRACOs in TS
  inline void setUsedTraco(int i, int val) { m_tstren.set(i, val); }

  //! TSM status
  inline void setTsmStatus(int i, int val) { m_tsmword.set(i, val); };

  //! Number of correctly configured TSS
  int nValidTSS() const;

  //! Number of correctly configured TSS
  int nValidTSM() const;

  //! Print the setup
  void print() const;

private:
  //! Check mask correctness
  bool checkMask(unsigned short) const;

  //! Load pset values into class variables
  void setDefaults(const edm::ParameterSet& ps);

  bool m_debug;

  // TSS Parameters
  unsigned short int m_tssmsk[2];  // [0]=1st [1]=2nd
  bool m_tsshte[3];                // [0]=1st [1]=2nd [2]=carry
  bool m_tssnoe[3];                // [0]=1st [1]=2nd [2]=carry
  bool m_tsscce[3];                // [0]=1st [1]=2nd [2]=carry
  unsigned short int m_tssgs1;
  unsigned short int m_tssgs2;
  bool m_tsscgs1;
  bool m_tsscgs2;

  //TSM Parameters
  unsigned short int m_tsmmsk[2];  // [0]=1st [1]=2nd
  bool m_tsmhte[3];                // [0]=1st [1]=2nd [2]=carry
  bool m_tsmnoe[3];                // [0]=1st [1]=2nd [2]=carry
  bool m_tsmcce[3];                // [0]=1st [1]=2nd [2]=carry
  unsigned short int m_tsmgs1;
  unsigned short int m_tsmgs2;
  bool m_tsmcgs1;
  bool m_tsmcgs2;
  unsigned short int m_tsmhsp;

  BitArray<24> m_tstren;  // Enabled TRACOs
  BitArray<8> m_tsmword;  // TSM backup mode word
  //unsigned short int m_ntsstsmd;        // nb tss to one of the tsmd (only if back-up mode)

  short int m_ntss;
  short int m_ntsm;
};

#endif
