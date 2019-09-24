//-------------------------------------------------
//
/**  \class DTConfigTraco
 *
 *   Configurable parameters and constants 
 *   for Level-1 Muon DT Trigger - Traco chip
 *
 *
 *   \author S. Vanini
 *
 */
//
//--------------------------------------------------
#ifndef DT_CONFIG_TRACO_H
#define DT_CONFIG_TRACO_H

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
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfig.h"
#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"
#include <cstdint>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTConfigTraco : public DTConfig {
public:
  //! Costants: esolution for psi and DeltaPsiR (phi_B)
  static const int RESOLPSI = 512;
  //! Costant: resulution for psiR (phi)
  static const int RESOLPSIR = 4096;
  //! Costant: maximum number of TRACO output candidates to TS
  static const int NMAXCAND;

  //! Constructor
  DTConfigTraco(const edm::ParameterSet& ps);

  //! Constructor
  DTConfigTraco(){};

  //! Constructor from string
  DTConfigTraco(int debug, unsigned short int* buffer);

  //! Destructor
  ~DTConfigTraco() override;

  //! Set default parameters
  void setDefaults(const edm::ParameterSet& ps);

  //! Debug flag
  inline int debug() const { return m_debug; }

  //! KRAD traco parameter
  inline int KRAD() const { return m_krad; }

  //! BTIC traco parameter: must be equal to Btis ST parameter
  inline int BTIC() const { return m_btic; }

  //! DD traco parameter: this is fixed
  inline int DD() const { return m_dd; }

  //! Recycling of TRACO cand. in inner/outer SL : REUSEI/REUSEO
  inline int TcReuse(int i) const {
    if (i == 0)
      return m_reusei;
    else
      return m_reuseo;
  }

  //! Single HTRIG enabling on first/second tracks F(S)HTMSK
  inline int singleHflag(int i) const {
    if (i == 0)
      return m_fhtmsk;
    else
      return m_shtmsk;
  }

  //! Single LTRIG enabling on first/second tracks: F(S)LTMSK
  inline int singleLflag(int i) const {
    if (i == 0)
      return m_fltmsk;
    else
      return m_sltmsk;
  }

  //! Preference to inner on first/second tracks: F(S)SLMSK
  inline int prefInner(int i) const {
    if (i == 0)
      return m_fslmsk;
    else
      return m_sslmsk;
  }

  //! Preference to HTRIG on first/second tracks: F(S)HTPRF
  inline int prefHtrig(int i) const {
    if (i == 0)
      return m_fhtprf;
    else
      return m_shtprf;
  }

  //! Ascend. order for K sorting first/second tracks: F(S)HISM
  inline int sortKascend(int i) const {
    if (i == 0)
      return m_fhism;
    else
      return m_shism;
  }

  //! K tollerance for correlation in TRACO: F(S)PRGCOMP
  inline int TcKToll(int i) const {
    if (i == 0)
      return m_fprgcomp;
    else
      return m_sprgcomp;
  }

  //! Suppr. of LTRIG in 4 BX before HTRIG: LTS
  inline int TcBxLts() const { return m_lts; }

  //! Single LTRIG accept enabling on first/second tracks LTF
  inline int singleLenab(int i) const { return m_ltf; }

  //! Connected bti in traco: bti mask
  inline int usedBti(int bti) const { return m_trgenb.element(bti - 1); }

  //! IBTIOFF traco parameter
  inline int IBTIOFF() const { return m_ibtioff; }

  //! Bending angle cut for all stations and triggers : KPRGCOM
  inline int BendingAngleCut() const { return m_kprgcom; }

  //! Flag for Low validation parameter
  inline int LVALIDIFH() const { return m_lvalidifh; }

  //! Set single parameter functions
  //! Set debug flag
  inline void setDebug(int debug) { m_debug = debug; }

  //! Set KRAD traco parameter
  inline void setKRAD(int KRAD) { m_krad = KRAD; }

  //! Set BTIC traco parameter: must be equal to Btis ST parameter
  inline void setBTIC(int BTIC) { m_btic = BTIC; }

  //! Set DD traco parameter: this is fixed
  inline void setDD(int DD) { m_dd = DD; }

  //! Set Recycling of TRACO cand. in inner/outer SL : REUSEI/REUSEO
  inline void setTcReuse(int i, int TcReuse) {
    if (i == 0)
      m_reusei = TcReuse;
    else
      m_reuseo = TcReuse;
  }

  //! Set Single HTRIG enabling on first/second tracks F(S)HTMSK
  inline void setSingleHflag(int i, int singleHflag) {
    if (i == 0)
      m_fhtmsk = singleHflag;
    else
      m_shtmsk = singleHflag;
  }

  //! Set Single LTRIG enabling on first/second tracks: F(S)LTMSK
  inline void setSingleLflag(int i, int singleLflag) {
    if (i == 0)
      m_fltmsk = singleLflag;
    else
      m_sltmsk = singleLflag;
  }

  //! Set Preference to inner on first/second tracks: F(S)SLMSK
  inline void setPrefInner(int i, int prefInner) {
    if (i == 0)
      m_fslmsk = prefInner;
    else
      m_sslmsk = prefInner;
  }

  //! Set Preference to HTRIG on first/second tracks: F(S)HTPRF
  inline void setPrefHtrig(int i, int prefHtrig) {
    if (i == 0)
      m_fhtprf = prefHtrig;
    else
      m_shtprf = prefHtrig;
  }

  //! Set Ascend. order for K sorting first/second tracks: F(S)HISM
  inline void setSortKascend(int i, int sortKascend) {
    if (i == 0)
      m_fhism = sortKascend;
    else
      m_shism = sortKascend;
  }

  //! Set K tollerance for correlation in TRACO: F(S)PRGCOMP
  inline void setTcKToll(int i, int TcKToll) {
    if (i == 0)
      m_fprgcomp = TcKToll;
    else
      m_sprgcomp = TcKToll;
  }

  //! Set Suppr. of LTRIG in 4 BX before HTRIG: LTS
  inline void setTcBxLts(int TcBxLts) { m_lts = TcBxLts; }

  //! Set Single LTRIG accept enabling on first/second tracks LTF
  inline void setSingleLenab(int i, int singleLenab) { m_ltf = singleLenab; }

  //! Set Connected bti in traco: bti mask
  inline void setUsedBti(int bti, int mask) { m_trgenb.set(bti - 1, mask); }

  //! Set IBTIOFF traco parameter
  inline void setIBTIOFF(int IBTIOFF) { m_ibtioff = IBTIOFF; }

  //! Set Bending angle cut for all stations and triggers : KPRGCOM
  inline void setBendingAngleCut(int BendingAngleCut) { m_kprgcom = BendingAngleCut; }

  //! Set Flag for Low validation parameter
  inline void setLVALIDIFH(int LVALIDIFH) { m_lvalidifh = LVALIDIFH; }

  //! Print the setup
  void print() const;

  /*   //! Return pointer to parameter set */
  /*   const edm::ParameterSet* getParameterSet() { return m_ps; } */

private:
  //  const edm::ParameterSet* m_ps;

  int8_t m_debug;
  int8_t m_krad;
  int8_t m_btic;
  int8_t m_dd;
  int8_t m_reusei;
  int8_t m_reuseo;
  int8_t m_fhtmsk;
  int8_t m_shtmsk;
  int8_t m_fltmsk;
  int8_t m_sltmsk;
  int8_t m_fslmsk;
  int8_t m_sslmsk;
  int8_t m_fhtprf;
  int8_t m_shtprf;
  int8_t m_fhism;
  int8_t m_shism;
  int8_t m_fprgcomp;
  int8_t m_sprgcomp;
  int8_t m_lts;
  int8_t m_ltf;
  BitArray<16> m_trgenb;
  int8_t m_ibtioff;
  int16_t m_kprgcom;
  int8_t m_lvalidifh;
};

#endif
