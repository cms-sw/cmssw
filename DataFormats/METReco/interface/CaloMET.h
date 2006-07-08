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

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/SpecificCaloMETData.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

namespace reco
{
  class CaloMET : public MET
    {
    public:
      /* Constructors*/
      CaloMET() {}
      CaloMET( SpecificCaloMETData calo_data_, double sumet_, const LorentzVector& fP4, const Point& fVertex ) 
	: MET( sumet_, fP4, fVertex ), calo_data( calo_data_ ) {}
      /* Default destructor*/
      virtual ~CaloMET() {}
      
      /* Returns the maximum energy deposited in ECAL towers */
      double maxEInEmTowers() const {return calo_data.mMaxEInEmTowers;};
      /* Returns the maximum energy deposited in HCAL towers */
      double maxEInHadTowers() const {return calo_data.mMaxEInHadTowers;};
      /* Returns the event hadronic energy fraction          */
      double energyFractionHadronic () const {return calo_data.mEnergyFractionHadronic;};
      /* Returns the event electromagnetic energy fraction   */
      double emEnergyFraction() const {return calo_data.mEnergyFractionEm;};
      /* Returns the event hadronic energy in HB             */
      double hadEnergyInHB() const {return calo_data.mHadEnergyInHB;};
      /* Returns the event hadronic energy in HO             */
      double hadEnergyInHO() const {return calo_data.mHadEnergyInHO;};
      /* Returns the event hadronic energy in HE             */
      double hadEnergyInHE() const {return calo_data.mHadEnergyInHE;};
      /* Returns the event hadronic energy in HF             */
      double hadEnergyInHF() const {return calo_data.mHadEnergyInHF;};
      /* Returns the event electromagnetic energy in EB      */
      double emEnergyInEB() const {return calo_data.mEmEnergyInEB;};
      /* Returns the event electromagnetic energy in EE      */
      double emEnergyInEE() const {return calo_data.mEmEnergyInEE;};
      /* Returns the event electromagnetic energy extracted from HF */
      double emEnergyInHF() const {return calo_data.mEmEnergyInHF;};
      
      // block accessors
      //const Specific& getSpecific () const {return calo_data;}
      
    private:
      virtual bool overlap( const Candidate & ) const;
      // Data members
      //Variables specific to to the CaloMET class
      SpecificCaloMETData calo_data;
    };
}
#endif
