#ifndef slimmedroi_h
#define slimmedroi_h

#include "TObject.h"

/**
   @short use to strip down the information of a Region Of Interest
 */
class SlimmedROI : public TObject
{
 public:
 SlimmedROI() : pt_(0), eta_(0) {}
 SlimmedROI(float pt, float eta, float phi,float mass,float area): pt_(pt), eta_(eta), phi_(phi), mass_(mass), area_(area) { }
  SlimmedROI(const SlimmedROI &other)
    {
      pt_       = other.pt_;            
      eta_      = other.eta_; 
      phi_      = other.phi_; 
      mass_     = other.mass_;
      area_     = other.area_;
      nhf_      = other.nhf_;           
      nhm_      = other.nhm_;      
      pf_       = other.pf_;            
      pm_       = other.pm_; 
      chf_      = other. chf_ ;         
      chm_      = other.chm_;
      genpt_    = other.genpt_;      
      geneta_   = other.geneta_;       
      genphi_   = other.genphi_;      
      genmass_  = other.genmass_;
      genarea_  = other.area_;
      partonpt_ = other.partonpt_;
      partoneta_= other.partoneta_; 
      partonphi_= other.partonphi_;
      stablept_ = other.stablept_;
      stableeta_= other.stableeta_; 
      stablephi_= other.stablephi_;
      stablex_  = other.stablex_;
      stabley_  = other.stabley_;
      stablez_  = other.stablez_;
      partonid_ = other.partonid_; 
      stableid_ = other.stableid_;
      stablemotherid_ = other.stablemotherid_;
      betaStar_ = other.betaStar_;
    }

  void setPFEnFractions(float nhf, float pf,float chf) { nhf_=nhf; pf_=pf; chf_=chf; }
  void setPFMultiplicities(float nhm, float pm, float chm) { nhm_=nhm; pm_=pm; chm_=chm; }
  void setGenJet(float pt, float eta, float phi, float mass, float area) { genpt_=pt; geneta_=eta; genphi_=phi; genmass_=mass; genarea_=area; }
  void setParton(float pt, float eta, float phi, int id) { partonpt_=pt; partoneta_=eta; partonphi_=phi; partonid_=id;}
  void setStable(int id, int motherid,   
		 float pt, float eta, float phi,
		 float hitx, float hity, float hitz) 
  { 
    stableid_=id;  stablemotherid_=motherid;
    stablept_=pt;  stableeta_=eta; stablephi_=phi; 
    stablex_=hitx; stabley_=hity;  stablez_=hitz;
  }
  void addBetaStar(float betaStar) { betaStar_.push_back(betaStar); }
  virtual ~SlimmedROI() { }

  float pt_,eta_,phi_,mass_,area_;
  float nhf_,nhm_,pf_,pm_,chf_,chm_;
  float genpt_, geneta_,genphi_,genmass_,genarea_;
  float partonpt_, partoneta_, partonphi_;
  float stablept_, stableeta_, stablephi_;
  float stablex_, stabley_, stablez_;
  int partonid_,stableid_, stablemotherid_;
  std::vector<float> betaStar_;

  ClassDef(SlimmedROI,1)
};

#endif
