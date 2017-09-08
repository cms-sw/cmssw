#ifndef CALIBCALORIMETRY_HCALALGOS_HCALTIMESLEW_H
#define CALIBCALORIMETRY_HCALALGOS_HCALTIMESLEW_H 1

#include <vector>

/** \class HcalTimeSlew
  * 
  * Provides pulse delay as a function of amplitude for three choices
  * of QIE bias setting.  The "Medium" setting is used in HB and HE,
  * while the "Slow" (and lower noise) setting is used in HO.  All
  * data taken from bench measurements of the QIE and plotted in
  * Physics TDR Vol 1.
  *
  * Not to be used for HF at this time (unlikely to have much effect, however)
  *
  * \author J. Mans - Minnesota
  */
class HcalTimeSlew {
 public:
  class HcalTimeSlewM2Parameters{
  public:
    //M2 Parameters
    double tzero;                    //{23.960177, 13.307784, 9.109694};
    double slope;                    //{-3.178648,  -1.556668, -1.075824 };
    double  tmax;                    //{16.00, 10.00, 6.25 }
    
    HcalTimeSlewM2Parameters(double t0, double m, double tmaximum):tzero(t0), slope(m), tmax(tmaximum){}
  };

  class HcalTimeSlewM3Parameters{
  public:
    //M3 Parameters
    double cap;                      //6.0;
    double tspar0;                   //{15.5, 12.2999};
    double tspar1;                   //{-3.2,-2.19142};
    double tspar2;                   //{32, 0};
    double tspar0_siPM;              //{0., 0.}; // 0ns delay for MC and DATA, recheck later for data
    double tspar1_siPM;              //{0, 0};
    double tspar2_siPM;              //{0, 0};
  
  HcalTimeSlewM3Parameters(double capCon, double tspar0Con, double tspar1Con, double tspar2Con, double tspar0_siPMCon, double tspar1_siPMCon, double tspar2_siPMCon):cap(capCon), tspar0(tspar0Con), tspar1(tspar1Con), tspar2(tspar2Con), tspar0_siPM(tspar0_siPMCon), tspar1_siPM(tspar1_siPMCon), tspar2_siPM(tspar2_siPMCon){} 
  };

  void addM2ParameterSet(double tzero, double slope, double tmax);
  void addM3ParameterSet(double cap, double tspar0, double tspar1, double tspar2, double tspar0_siPM, double tspar1_siPM, double tspar2_siPM);

  enum ParaSource { TestStand=0, Data=1, MC=2, InputPars=3 };
  enum BiasSetting { Slow=0, Medium=1, Fast=2 };
  static constexpr double tspar[3] = {12.2999, -2.19142, 0};
  /** \brief Returns the amount (ns) by which a pulse of the given
   number of fC will be delayed by the timeslew effect, for the
   specified bias setting. */
  double delay(double fC, BiasSetting bias=Medium) const; 
  double delay(double fC, ParaSource source=InputPars, BiasSetting bias=Medium, double par0=tspar[0], double par1=tspar[1], double par2=tspar[2], bool isHPD=true) const;
  
 private:
  std::vector<HcalTimeSlewM2Parameters> parametersM2_;
  std::vector<HcalTimeSlewM3Parameters> parametersM3_;
};

#endif
