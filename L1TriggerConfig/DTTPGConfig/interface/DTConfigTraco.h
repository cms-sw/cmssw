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
#include "L1Trigger/DTUtilities/interface/DTConfig.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------


class DTConfigTraco : public DTConfig {

  public:

  //! Costants: esolution for psi and DeltaPsiR (phi_B)
  static const int RESOLPSI=512;    
  //! Costant: resulution for psiR (phi)
  static const int RESOLPSIR=4096;
  //! Costant: maximum number of TRACO output candidates to TS
  static const int NMAXCAND;


  //! Constructor
  DTConfigTraco(const edm::ParameterSet& ps);
  
  //! Destructor 
  ~DTConfigTraco();

  //! Set default parameters
  void setDefaults();

  //! Debug flag
  inline int debug() const { return m_debug; }

  //! KRAD traco parameter
  inline int KRAD() { return m_krad;}

  //! BTIC traco parameter: must be equal to Btis ST parameter
  inline int BTIC() { return m_btic;}

  //! DD traco parameter: this is fixed
  inline int DD() { return m_dd;}

  //! Recycling of TRACO cand. in inner/outer SL : REUSEI/REUSEO
  inline int  TcReuse(int i) const {
    if(i==0)
      return m_reusei;
    else
      return m_reuseo;
  }

  //! Single HTRIG enabling on first/second tracks F(S)HTMSK
  inline int singleHflag(int i) const {
    if(i==0)
      return m_fhtmsk;
    else
      return m_shtmsk;
  }

  //! Single LTRIG enabling on first/second tracks: F(S)LTMSK
  inline int  singleLflag(int i) const {
    if(i==0)
      return m_fltmsk;
    else
      return m_sltmsk;
  } 

  //! Preference to inner on first/second tracks: F(S)SLMSK
  inline int  prefInner(int i) const {
    if(i==0)
      return m_fslmsk;
    else
      return m_sslmsk;
  }

  //! Preference to HTRIG on first/second tracks: F(S)HTPRF
  inline int prefHtrig(int i) const {
    if(i==0)
      return m_fhtprf;
    else
      return m_shtprf;
  }

  //! Ascend. order for K sorting first/second tracks: F(S)HISM
  inline int sortKascend(int i) const {
    if(i==0)
      return m_fhism;
    else
      return m_shism;
  }


  //! K tollerance for correlation in TRACO: F(S)PRGCOMP
  inline int  TcKToll(int i) const {
    if(i==0)
      return m_fprgcomp;
    else
      return m_sprgcomp;
  }

  //! Suppr. of LTRIG in 4 BX before HTRIG: LTS
  inline int  TcBxLts() const { return m_lts; }

  //! Single LTRIG accept enabling on first/second tracks LTF
  inline int  singleLenab(int i) const { return m_ltf; }

  //! Connected bti in traco: bti mask
  inline int usedBti(int bti) const { return m_trgenb[bti-1]; }

  //! IBTIOFF traco parameter
  inline int IBTIOFF() { return m_ibtioff; }

  //! Bending angle cut for all stations and triggers : KPRGCOM
  inline int BendingAngleCut() const { return m_kprgcom; }

  //! Flag for Low validation parameter
  inline int LVALIDIFH() { return m_lvalidifh;}

  //! Print the setup
  void print() const ;

  //! Return pointer to parameter set
  const edm::ParameterSet* getParameterSet() { return m_ps; }


private:
  const edm::ParameterSet* m_ps;

  int m_debug;
  int m_krad;
  int m_btic;
  int m_dd;
  int m_reusei;
  int m_reuseo;
  int m_fhtmsk;
  int m_shtmsk;
  int m_fltmsk;
  int m_sltmsk;
  int m_fslmsk;
  int m_sslmsk;
  int m_fhtprf;
  int m_shtprf;
  int m_fhism;
  int m_shism;
  int m_fprgcomp;
  int m_sprgcomp;
  int m_lts;
  int m_ltf;
  int m_trgenb[16];
  int m_ibtioff;
  int m_kprgcom;
  int m_lvalidifh;

};

#endif
