#ifndef DataFormats_L1Scouting_L1ScoutingCalo_h
#define DataFormats_L1Scouting_L1ScoutingCalo_h

#include "DataFormats/L1Trigger/interface/EtSum.h"
#include <cmath>

namespace scoutingRun3 {

  class ScCaloObject {
  public:
    ScCaloObject()
    : hwEt_(0),
      hwEta_(0),
      hwPhi_(0),
      iso_(0){}

    ScCaloObject(
      int hwEt,
      int hwEta,
      int hwPhi,
      int iso)
    : hwEt_(hwEt),
      hwEta_(hwEta),
      hwPhi_(hwPhi),
      iso_(iso) {}

    ScCaloObject(const ScCaloObject& other) = default;
    ScCaloObject(ScCaloObject&& other) = default;
    ScCaloObject & operator=(const ScCaloObject& other) = default;
    ScCaloObject & operator=(ScCaloObject&& other) = default;

    inline void setHwEt(int hwEt) { hwEt_= hwEt;}
    inline void setHwEta(int hwEta) { hwEta_= hwEta;}
    inline void setHwPhi(int hwPhi) { hwPhi_= hwPhi;}
    inline void setIso(int iso) { iso_= iso;}

    inline int getHwEt() const {return hwEt_;}
    inline int getHwEta() const {return hwEta_;}
    inline int getHwPhi() const {return hwPhi_;}
    inline int getIso() const {return iso_;}

    inline float getEt() const {
      return et_scale_* hwEt_;
    }
    inline float getEta()const {
      return eta_scale_*hwEta_;
    }
    inline float getPhi() const {
      float fPhi = phi_scale_*hwPhi_;
      fPhi = fPhi>=2.*M_PI ? fPhi-2.*M_PI : fPhi;
      return fPhi;
    }

  private:
    int hwEt_;
    int hwEta_;
    int hwPhi_;
    int iso_;

    static constexpr float phi_scale_ = 2.*M_PI/144.;
    static constexpr float eta_scale_ = 0.0435;
    static constexpr float et_scale_  = 0.5;
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

    inline void setHwEt(int hwEt) { hwEt_= hwEt;}
    inline void setHwPhi(int hwPhi) { hwPhi_= hwPhi;}
    inline void setType(l1t::EtSum::EtSumType type) { type_= type;}

    inline int getHwEt() const {return hwEt_;}
    inline int getHwPhi() const {return hwPhi_;}
    inline l1t::EtSum::EtSumType getType() const {return type_;}

    inline float getEt() const {
      return et_scale_* hwEt_;
    }
    inline float getPhi() const {
      float fPhi = phi_scale_*hwPhi_;
      fPhi = fPhi>=2.*M_PI ? fPhi-2.*M_PI : fPhi;
      return fPhi;
    }

  private:
    int hwEt_;
    int hwPhi_;
    l1t::EtSum::EtSumType type_;

    static constexpr float phi_scale_ = 2.*M_PI/144.;
    static constexpr float et_scale_  = 0.5; 
  };

} // namespace scoutingRun3
#endif // DataFormats_L1Scouting_L1ScoutingCalo_h