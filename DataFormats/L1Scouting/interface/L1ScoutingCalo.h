#ifndef DataFormats_L1Scouting_L1ScoutingCalo_h
#define DataFormats_L1Scouting_L1ScoutingCalo_h

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

namespace l1ScoutingRun3 {

  class ScJet;
  typedef OrbitCollection<ScJet>    ScJetOrbitCollection;
  class ScEGamma;
  typedef OrbitCollection<ScEGamma> ScEGammaOrbitCollection;
  class ScTau;
  typedef OrbitCollection<ScTau>    ScTauOrbitCollection;
  class ScEtSum;
  typedef OrbitCollection<ScEtSum>  ScEtSumOrbitCollection;

  class ScCaloObject {
  public:
    ScCaloObject()
    : hwEt_(0),
      hwEta_(0),
      hwPhi_(0),
      hwIso_(0){}

    ScCaloObject(
      int hwEt,
      int hwEta,
      int hwPhi,
      int iso)
    : hwEt_(hwEt),
      hwEta_(hwEta),
      hwPhi_(hwPhi),
      hwIso_(iso) {}

    ScCaloObject(const ScCaloObject& other) = default;
    ScCaloObject(ScCaloObject&& other) = default;
    ScCaloObject & operator=(const ScCaloObject& other) = default;
    ScCaloObject & operator=(ScCaloObject&& other) = default;

    void swap(ScCaloObject& other){
      using std::swap;
      swap(hwEt_, other.hwEt_);
      swap(hwEta_, other.hwEta_);
      swap(hwPhi_, other.hwPhi_);
      swap(hwIso_, other.hwIso_);
    }

    inline void setHwEt(int hwEt) { hwEt_= hwEt;}
    inline void setHwEta(int hwEta) { hwEta_= hwEta;}
    inline void setHwPhi(int hwPhi) { hwPhi_= hwPhi;}
    inline void setHwIso(int hwIso) { hwIso_= hwIso;}

    inline int hwEt() const {return hwEt_;}
    inline int hwEta() const {return hwEta_;}
    inline int hwPhi() const {return hwPhi_;}
    inline int hwIso() const {return hwIso_;}

  private:
    int hwEt_;
    int hwEta_;
    int hwPhi_;
    int hwIso_;

  };

  class ScJet: public ScCaloObject {
  public:
    ScJet(): ScCaloObject(0, 0 ,0 , 0){}

    ScJet(
      int hwEt,
      int hwEta,
      int hwPhi,
      int iso)
    : ScCaloObject(hwEt, hwEta ,hwPhi , iso) {}

  };

  class ScEGamma: public ScCaloObject {
  public:
    ScEGamma(): ScCaloObject(0, 0 ,0 , 0){}

    ScEGamma(
      int hwEt,
      int hwEta,
      int hwPhi,
      int iso)
    : ScCaloObject(hwEt, hwEta ,hwPhi , iso) {}
  };

  class ScTau: public ScCaloObject {
  public:
    ScTau(): ScCaloObject(0, 0 ,0 , 0){}

    ScTau(
      int hwEt,
      int hwEta,
      int hwPhi,
      int iso)
    : ScCaloObject(hwEt, hwEta ,hwPhi , iso) {}
  };


  class ScEtSum {
  public:
    ScEtSum()
    : hwEt_(0),
      hwPhi_(0),
      type_(l1t::EtSum::kUninitialized) {}

    ScEtSum(
      int hwEt,
      int hwPhi,
      l1t::EtSum::EtSumType type)
    : hwEt_(hwEt),
      hwPhi_(hwPhi),
      type_(type) {}

    ScEtSum(const ScEtSum& other) = default;
    ScEtSum(ScEtSum&& other) = default;
    ScEtSum & operator=(const ScEtSum& other) = default;
    ScEtSum & operator=(ScEtSum&& other) = default;

    void swap(ScEtSum& other){
      using std::swap;
      swap(hwEt_, other.hwEt_);
      swap(hwPhi_, other.hwPhi_);
      swap(type_, other.type_);
    }

    inline void setHwEt(int hwEt) { hwEt_= hwEt;}
    inline void setHwPhi(int hwPhi) { hwPhi_= hwPhi;}
    inline void setType(l1t::EtSum::EtSumType type) { type_= type;}

    inline int hwEt() const {return hwEt_;}
    inline int hwPhi() const {return hwPhi_;}
    inline l1t::EtSum::EtSumType type() const {return type_;}

    // inline float Et() const {
    //   return et_scale_* hwEt_;
    // }
    // inline float phi() const {
    //   float fPhi = phi_scale_*hwPhi_;
    //   fPhi = fPhi>=M_PI ? fPhi-2.*M_PI : fPhi;
    //   return fPhi;
    // }

  private:
    int hwEt_;
    int hwPhi_;
    l1t::EtSum::EtSumType type_;

    // static constexpr float phi_scale_ = 2.*M_PI/144.;
    // static constexpr float et_scale_  = 0.5; 
  };

} // namespace l1ScoutingRun3
#endif // DataFormats_L1Scouting_L1ScoutingCalo_h