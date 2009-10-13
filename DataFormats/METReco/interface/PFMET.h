#ifndef METReco_PFMET_h
#define METReco_PFMET_h

/*
class: PFMET
description:  MET made from Particle Flow candidates
authors: R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
date: 10/27/08
*/

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/SpecificPFMETData.h"
namespace reco
{
  class PFMET:  public MET {
  public:
    PFMET() {}
    PFMET( SpecificPFMETData pf_data_, double sumet_,
	   const LorentzVector& fP4, const Point& fVertex )
      : MET( sumet_, fP4, fVertex ), pf_data( pf_data_ ) {}

    virtual ~PFMET() {}
    
    //getters
    double NeutralEMEtFraction() const { return pf_data.NeutralEMEtFraction; }
    double NeutralEMEt() const { return pf_data.NeutralEMEtFraction * sumEt(); }

    double NeutralHadEtFraction() const { return pf_data.NeutralHadEtFraction; }
    double NeutralHadEt() const { return pf_data.NeutralHadEtFraction * sumEt(); }

    double ChargedEMEtFraction() const { return pf_data.ChargedEMEtFraction; }
    double ChargedEMEt() const { return pf_data.ChargedEMEtFraction * sumEt(); }

    double ChargedHadEtFraction() const { return pf_data.ChargedHadEtFraction; }
    double ChargedHadEt() const { return pf_data.ChargedHadEtFraction * sumEt(); }

    double MuonEtFraction() const { return pf_data.MuonEtFraction; }
    double MuonEt() const { return pf_data.MuonEtFraction * sumEt(); }
    
    double Type6EtFraction() const { return pf_data.Type6EtFraction; }
    double Type6Et() const { return pf_data.Type6EtFraction * sumEt(); }

    double Type7EtFraction() const { return pf_data.Type7EtFraction; }
    double Type7Et() const { return pf_data.Type7EtFraction * sumEt(); }

    // block accessors
    SpecificPFMETData getSpecific() const {return pf_data;}
   

  private:
    SpecificPFMETData pf_data;
  };
}
#endif
