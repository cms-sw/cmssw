#ifndef METReco_CaloMET_h
#define METReco_CaloMET_h

/** \class CaloMET
 *
 * \short MET made from CaloTowers
 *
 * CaloMET represents MET made from CaloTowers
 * Provide energy contributions from different subdetectors
 * in addition to generic MET parameters
 *
 * \author    R. Cavanaugh, UFL (inspiration taken from F. Ratnikov)
 *
 ************************************************************/

#include "DataFormats/METReco/interface/SpecificCaloMETData.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

namespace reco
{
  class CaloMET : public MET
    {
    public:
      /* Constructors*/
      CaloMET() ;
      CaloMET( SpecificCaloMETData calo_data_, double sumet_, 
	       const LorentzVector& fP4, const Point& fVertex ) 
	: MET( sumet_, fP4, fVertex ), calo_data( calo_data_ ) {}
      CaloMET( SpecificCaloMETData calo_data_, double sumet_, 
	       const std::vector<CorrMETData>& corr_,
	       const LorentzVector& fP4, const Point& fVertex ) 
	: MET( sumet_, corr_, fP4, fVertex ), calo_data( calo_data_ ) {}
      /* Default destructor*/
      virtual ~CaloMET() {}
      
      /* Returns the maximum energy deposited in ECAL towers */
      double maxEtInEmTowers() const {return calo_data.MaxEtInEmTowers;};
      /* Returns the maximum energy deposited in HCAL towers */
      double maxEtInHadTowers() const {return calo_data.MaxEtInHadTowers;};
      /* Returns the event hadronic energy fraction          */
      double etFractionHadronic () const 
	{return calo_data.EtFractionHadronic;};
      /* Returns the event electromagnetic energy fraction   */
      double emEtFraction() const {return calo_data.EtFractionEm;};
      /* Returns the event hadronic energy in HB             */
      double hadEtInHB() const {return calo_data.HadEtInHB;};
      /* Returns the event hadronic energy in HO             */
      double hadEtInHO() const {return calo_data.HadEtInHO;};
      /* Returns the event hadronic energy in HE             */
      double hadEtInHE() const {return calo_data.HadEtInHE;};
      /* Returns the event hadronic energy in HF             */
      double hadEtInHF() const {return calo_data.HadEtInHF;};
      /* Returns the event electromagnetic energy in EB      */
      double emEtInEB() const {return calo_data.EmEtInEB;};
      /* Returns the event electromagnetic energy in EE      */
      double emEtInEE() const {return calo_data.EmEtInEE;};
      /* Returns the event electromagnetic energy extracted from HF */
      double emEtInHF() const {return calo_data.EmEtInHF;};
      /* Returns the event MET Significance */
      double metSignificance() const {return this->significance();};
      /* Returns the event SET in HF+ */
      double CaloSETInpHF() const {return calo_data.CaloSETInpHF;};
      /* Returns the event SET in HF- */
      double CaloSETInmHF() const {return calo_data.CaloSETInmHF;};
      /* Returns the event MET in HF+ */
      double CaloMETInpHF() const {return calo_data.CaloMETInpHF;};
      /* Returns the event MET in HF- */
      double CaloMETInmHF() const {return calo_data.CaloMETInmHF;};
      /* Returns the event MET-phi in HF+ */
      double CaloMETPhiInpHF() const {return calo_data.CaloMETPhiInpHF;};
      /* Returns the event MET-phi in HF- */
      double CaloMETPhiInmHF() const {return calo_data.CaloMETPhiInmHF;};

      //Set Met Significance   
      void SetMetSignificance(double metsig){ calo_data.METSignificance = metsig; }

      // block accessors
      SpecificCaloMETData getSpecific() const {return calo_data;}
      
    private:
      virtual bool overlap( const Candidate & ) const;
      // Data members
      //Variables specific to to the CaloMET class
      SpecificCaloMETData calo_data;
    };
}
#endif
