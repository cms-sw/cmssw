#ifndef DataFormats_L1Trigger_Jet_h
#define DataFormats_L1Trigger_Jet_h


#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  class Jet;
  typedef BXVector<Jet> JetBxCollection;
  typedef edm::Ref< JetBxCollection > JetRef ;
  typedef edm::RefVector< JetBxCollection > JetRefVector ;
  typedef std::vector< JetRef > JetVectorRef ;

  class Jet : public L1Candidate {

  public:
  Jet(){}
  Jet( const LorentzVector& p4,
       int pt=0,
       int eta=0,
       int phi=0,
       int qual=0);
  Jet( const PolarLorentzVector& p4,
       int pt=0,
       int eta=0,
       int phi=0,
       int qual=0);

  ~Jet();

                  
  void setTowerIEta(short int ieta);  // ieta of seed tower                   
  void setTowerIPhi(short int iphi);  // iphi of seed tower                   
  void setRawEt(short int et);    // raw (uncalibrated) cluster sum
  void setSeedEt(short int et);
  void setPUEt(short int et);
  void setPUDonutEt(uint i, short int et);

  short int towerIEta() const;
  short int towerIPhi() const;
  short int rawEt() const;
  short int seedEt() const;
  short int puEt() const ;
  short int puDonutEt(int i) const;

  private:

  // additional hardware quantities common to L1 global jet
  void clear_extended();
  short int towerIEta_;
  short int towerIPhi_;
  short int rawEt_;
  short int seedEt_;
  short int puEt_;
  short int puDonutEt_[4];

  };

}

#endif
