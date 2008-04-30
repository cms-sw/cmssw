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
#include "CondFormats/L1TObjects/interface/DTConfig.h"
#include "CondFormats/L1TObjects/interface/BitArray.h"

#include <boost/cstdint.hpp>

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

  //! Constructor
  DTConfigTraco() {};

  
  //! Destructor 
  ~DTConfigTraco();

  //! Set default parameters
  void setDefaults(const edm::ParameterSet& ps);

  //! Debug flag
  inline int debug() const { return m_debug; }

  //! KRAD traco parameter
  inline int KRAD() const { return m_krad;}

  //! BTIC traco parameter: must be equal to Btis ST parameter
  inline int BTIC() const { return m_btic;}

  //! DD traco parameter: this is fixed
  inline int DD() const { return m_dd;}

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
  inline int usedBti(int bti) const { return m_trgenb.element(bti-1); }

  //! IBTIOFF traco parameter
  inline int IBTIOFF() const { return m_ibtioff; }

  //! Bending angle cut for all stations and triggers : KPRGCOM
  inline int BendingAngleCut() const { return m_kprgcom; }

  //! Flag for Low validation parameter
  inline int LVALIDIFH() const { return m_lvalidifh;}

  //! Print the setup
  void print() const ;

/*   //! Return pointer to parameter set */
/*   const edm::ParameterSet* getParameterSet() { return m_ps; } */


private:
  //  const edm::ParameterSet* m_ps;

  unsigned short int m_debug;
  unsigned short int m_krad;
  unsigned short int m_btic;
  unsigned short int m_dd;
  unsigned short int m_reusei;
  unsigned short int m_reuseo;
  unsigned short int m_fhtmsk;
  unsigned short int m_shtmsk;
  unsigned short int m_fltmsk;
  unsigned short int m_sltmsk;
  unsigned short int m_fslmsk;
  unsigned short int m_sslmsk;
  unsigned short int m_fhtprf;
  unsigned short int m_shtprf;
  unsigned short int m_fhism;
  unsigned short int m_shism;
  unsigned short int m_fprgcomp;
  unsigned short int m_sprgcomp;
  unsigned short int m_lts;
  unsigned short int m_ltf;
  BitArray<16> m_trgenb;
  unsigned short int m_ibtioff;
  unsigned short int m_kprgcom;
  unsigned short int m_lvalidifh;

};

#endif
