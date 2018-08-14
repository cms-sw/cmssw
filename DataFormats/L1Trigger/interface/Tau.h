#ifndef DataFormats_L1Trigger_Tau_h
#define DataFormats_L1Trigger_Tau_h


#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/L1TObjComparison.h"

namespace l1t {

  class Tau;
  typedef BXVector<Tau> TauBxCollection;
  typedef edm::Ref< TauBxCollection > TauRef ;
  typedef edm::RefVector< TauBxCollection > TauRefVector ;
  typedef std::vector< TauRef > TauVectorRef ;

  typedef ObjectRefBxCollection<Tau> TauRefBxCollection;
  typedef ObjectRefPair<Tau> TauRefPair;
  typedef ObjectRefPairBxCollection<Tau> TauRefPairBxCollection;

  class Tau : public L1Candidate {

  public:
    Tau(){ clear_extended(); }

    // ctor from base allowed, but note that extended variables will be set to zero:
    Tau(const L1Candidate& rhs):L1Candidate(rhs){ clear_extended(); } 
    
    Tau( const LorentzVector& p4,
	    int pt=0,
	    int eta=0,
	    int phi=0,
	    int qual=0,
	    int iso=0);
    Tau( const PolarLorentzVector& p4,
	    int pt=0,
	    int eta=0,
	    int phi=0,
	    int qual=0,
	    int iso=0);


    ~Tau() override;

    void setTowerIEta(short int ieta);  // ieta of seed tower
    void setTowerIPhi(short int iphi);  // iphi of seed tower
    void setRawEt(short int et);    // raw (uncalibrated) cluster sum
    void setIsoEt(short int et);    // raw isolation sum - cluster sum
    void setNTT(short int ntt);     // n towers above threshold
    void setHasEM(bool hasEM);
    void setIsMerged(bool isMerged);

    short int towerIEta() const;
    short int towerIPhi() const;
    short int rawEt() const;
    short int isoEt() const;
    short int nTT() const;
    bool hasEM() const;
    bool isMerged() const;

    virtual bool operator==(const l1t::Tau& rhs) const;
    virtual inline bool operator!=(const l1t::Tau& rhs) const { return !(operator==(rhs)); };

  private:

    // additional hardware quantities common to L1 global tau
    void clear_extended();
    short int towerIEta_;
    short int towerIPhi_;
    short int rawEt_;
    short int isoEt_;
    short int nTT_;
    bool hasEM_;
    bool isMerged_;

  };

}

#endif
