#ifndef DataFormats_L1Trigger_EGamma_h
#define DataFormats_L1Trigger_EGamma_h


#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"


namespace l1t {

  class EGamma;
  typedef BXVector<EGamma> EGammaBxCollection;
  typedef edm::Ref< EGammaBxCollection > EGammaRef ;
  typedef edm::RefVector< EGammaBxCollection > EGammaRefVector ;
  typedef std::vector< EGammaRef > EGammaVectorRef ;

  class EGamma : public L1Candidate {

  public:
    EGamma(){}

    // ctor from base allowed, but note that extended variables will be set to zero:
    EGamma(const L1Candidate& rhs):L1Candidate(rhs){ clear_extended(); } 

    EGamma( const LorentzVector& p4,
	    int pt=0,
	    int eta=0,
	    int phi=0,
	    int qual=0,
	    int iso=0);

    EGamma( const PolarLorentzVector& p4,
	    int pt=0,
	    int eta=0,
	    int phi=0,
	    int qual=0,
	    int iso=0);

    ~EGamma();

    void setTowerIEta(short int ieta);  // ieta of seed tower
    void setTowerIPhi(short int iphi);  // iphi of seed tower
    void setRawEt(short int pt);        // raw (uncalibrated) cluster sum
    void setIsoEt(short int iso);       // raw isolation sum - cluster sum
    void setFootprintEt(short int fp);  // raw footprint
    void setNTT(short int ntt);         // n towers above threshold
    void setShape(short int s);         // cluster shape variable
    void setTowerHoE(short int HoE);         // H/E as computed in Layer-1

    short int towerIEta() const;
    short int towerIPhi() const;
    short int rawEt() const;
    short int isoEt() const ;
    short int footprintEt() const;
    short int nTT() const;
    short int shape() const;
    short int towerHoE() const;

  private:

    // additional hardware quantities common to L1 global EG
    void clear_extended();
    short int towerIEta_;
    short int towerIPhi_;
    short int rawEt_;
    short int isoEt_;
    short int footprintEt_;
    short int nTT_;
    short int shape_;
    short int towerHoE_;

  };

}

#endif
