#ifndef GUARD_HCALHF_PETALGORITHM_H
#define GUARD_HCALHF_PETALGORITHM_H 1

#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

// Forward declarations
class HcalChannelQuality;
class HcalSeverityLevelComputer;

/**  \class HcalHF_PETalgorithm

   Class evaluates the ratio |(L-S)/(L+S)| for a given cell, and flags the 
   cell if the threshold exceeds a given maximum value R(Energy).
   Each cell must also pass ieta-dependent energy and ET cuts to be considered for flagging.

   \author J. Temple and D. Ferencek
*/


class HcalHF_PETalgorithm {
 public:
  /** Constructors **/
  HcalHF_PETalgorithm();
  
  HcalHF_PETalgorithm(const std::vector<double>& short_R, 
		      const std::vector<double>& short_Energy, 
		      const std::vector<double>& short_ET, 
		      const std::vector<double>& long_R, 
		      const std::vector<double>& long_Energy, 
		      const std::vector<double>& long_ET,
		      int HcalAcceptSeverityLevel,
		      // special case for ieta=29
		      const std::vector<double>& short_R_29,
		      const std::vector<double>& long_R_29);

  // Destructor
  ~HcalHF_PETalgorithm();

  void HFSetFlagFromPET(HFRecHit& hf,
			HFRecHitCollection& rec,
			const HcalChannelQuality* myqual,
			const HcalSeverityLevelComputer* mySeverity);
  double CalcThreshold(double abs_energy,const std::vector<double>& params);

  void SetShort_R(const std::vector<double>& x){short_R=x;}
  void SetShort_ET_Thresh(const std::vector<double>& x){short_ET_Thresh=x;}
  void SetShort_Energy_Thresh(const std::vector<double>& x){short_Energy_Thresh=x;}
  void SetLong_R(const std::vector<double>& x){long_R=x;}
  void SetLong_ET_Thresh(const std::vector<double>& x){long_ET_Thresh=x;}
  void SetLong_Energy_Thresh(const std::vector<double>& x){long_Energy_Thresh=x;}

  std::vector<double> GetShort_R(){return short_R;}
  std::vector<double> GetShort_ET_Thresh(){return short_ET_Thresh;}
  std::vector<double> GetShort_Energy_Thresh(){return short_Energy_Thresh;}
  std::vector<double> GetLong_R(){return long_R;}
  std::vector<double> GetLong_ET_Thresh(){return long_ET_Thresh;}
  std::vector<double> GetLong_Energy_Thresh(){return long_Energy_Thresh;}

  double bit(){return HcalCaloFlagLabels::HFLongShort;}

 private:
  std::vector<double> short_R;
  std::vector<double> short_ET_Thresh;
  std::vector<double> short_Energy_Thresh; 

  std::vector<double> long_R;
  std::vector<double> long_ET_Thresh;
  std::vector<double> long_Energy_Thresh;
  int HcalAcceptSeverityLevel_;
  std::vector<double> short_R_29;
  std::vector<double> long_R_29;
};


#endif
