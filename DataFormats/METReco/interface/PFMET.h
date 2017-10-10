#ifndef METReco_PFMET_h
#define METReco_PFMET_h

/*
class: PFMET
description:  MET made from Particle Flow candidates
authors: R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
date: 10/27/08
*/

// Note : Data members refer to transverse quantities as is indicated by the accessors .
//        Eventually, we will change them accordingly when backward compatibility is not required 
// Date : 10/14/09

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/SpecificPFMETData.h"
namespace reco
{
  class PFMET:  public MET {
  public:
    PFMET() ;
    PFMET( const SpecificPFMETData& pf_data_, double sumet_,
	   const LorentzVector& fP4, const Point& fVertex )
      : MET( sumet_, fP4, fVertex ), pf_data( pf_data_ ) {}

    ~PFMET() override {}
    
    //getters
    double photonEtFraction() const { return pf_data.NeutralEMFraction; }
    double photonEt() const { return pf_data.NeutralEMFraction * sumEt(); }

    double neutralHadronEtFraction() const { return pf_data.NeutralHadFraction; }
    double neutralHadronEt() const { return pf_data.NeutralHadFraction * sumEt(); }

    double electronEtFraction() const { return pf_data.ChargedEMFraction; }
    double electronEt() const { return pf_data.ChargedEMFraction * sumEt(); }

    double chargedHadronEtFraction() const { return pf_data.ChargedHadFraction; }
    double chargedHadronEt() const { return pf_data.ChargedHadFraction * sumEt(); }

    double muonEtFraction() const { return pf_data.MuonFraction; }
    double muonEt() const { return pf_data.MuonFraction * sumEt(); }
    
    double HFHadronEtFraction() const { return pf_data.Type6Fraction; }
    double HFHadronEt() const { return pf_data.Type6Fraction * sumEt(); }

    double HFEMEtFraction() const { return pf_data.Type7Fraction; }
    double HFEMEt() const { return pf_data.Type7Fraction * sumEt(); }

    // Old accessors (should be removed in future)
    double NeutralEMEtFraction() const { return pf_data.NeutralEMFraction; }
    double NeutralEMEt() const { return pf_data.NeutralEMFraction * sumEt(); }
    double NeutralHadEtFraction() const { return pf_data.NeutralHadFraction; }
    double NeutralHadEt() const { return pf_data.NeutralHadFraction * sumEt(); }
    double ChargedEMEtFraction() const { return pf_data.ChargedEMFraction; }
    double ChargedEMEt() const { return pf_data.ChargedEMFraction * sumEt(); }
    double ChargedHadEtFraction() const { return pf_data.ChargedHadFraction; }
    double ChargedHadEt() const { return pf_data.ChargedHadFraction * sumEt(); }
    double MuonEtFraction() const { return pf_data.MuonFraction; }
    double MuonEt() const { return pf_data.MuonFraction * sumEt(); }
    double Type6EtFraction() const { return pf_data.Type6Fraction; }
    double Type6Et() const { return pf_data.Type6Fraction * sumEt(); }
    double Type7EtFraction() const { return pf_data.Type7Fraction; }
    double Type7Et() const { return pf_data.Type7Fraction * sumEt(); }
    double NeutralEMFraction() const { return pf_data.NeutralEMFraction; }
    double NeutralHadFraction() const { return pf_data.NeutralHadFraction; }
    double ChargedEMFraction() const { return pf_data.ChargedEMFraction; }
    double ChargedHadFraction() const { return pf_data.ChargedHadFraction; }
    double MuonFraction() const { return pf_data.MuonFraction; }

    // block accessors
    SpecificPFMETData getSpecific() const {return pf_data;}
   

  private:
    SpecificPFMETData pf_data;
  };
}
#endif
