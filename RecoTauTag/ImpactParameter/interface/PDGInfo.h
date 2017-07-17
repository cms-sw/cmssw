#ifndef RecoTauTag_ImpactParameter_PDGInfo_h
#define RecoTauTag_ImpactParameter_PDGInfo_h

/* From SimpleFits Package
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

namespace tauImpactParameter {

class PDGInfo {
 public:
  static double pi_mass(){return  0.13957018;}
  static double tau_mass(){return 1.77682;}
  static double nu_mass(){return  0.0;}

  static double pi_mass_MCGen(){return  0.139;}
  static double tau_mass_MCGen(){return 1.777;}
  static double nu_mass_MCGen(){return  0.0;}

  enum PDGMCNumbering {
    tau_minus = 15 ,
    tau_plus = -15 ,
    nu_tau = 16 ,
    anti_nu_tau = -16 ,
    a_1_plus = 20213 ,
    a_1_minus = -20213 ,
   };
};

}
#endif
