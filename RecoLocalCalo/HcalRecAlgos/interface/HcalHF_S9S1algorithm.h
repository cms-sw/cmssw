#ifndef GUARD_HCALHF_S9S1ALGORITHM_H
#define GUARD_HCALHF_S9S1ALGORITHM_H 1

#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"

// Forward declarations
class HcalChannelQuality;
class HcalSeverityLevelComputer;

/**  \class HcalHF_S9S1algorithm

   Class evaluates the ratio |(L-S)/(L+S)| for a given cell, and flags the 
   cell if the threshold exceeds a given maximum value R(Energy).
   Each cell must also pass ieta-dependent energy and ET cuts to be considered for flagging.

   $Date: 2011/04/15 19:11:48 $
   $Revision: 1.4 $
   \author J. Temple and D. Ferencek
*/


class HcalHF_S9S1algorithm {
 public:
  /** Constructors **/
  HcalHF_S9S1algorithm();
  
  HcalHF_S9S1algorithm(std::vector<double> short_optimumSlope, 
		       std::vector<double> short_Energy, 
		       std::vector<double> short_ET, 
		       std::vector<double> long_optimumSlope, 
		       std::vector<double> long_Energy, 
		       std::vector<double> long_ET,
		       int HcalAcceptSeverityLevel,
		       bool isS8S1);

  // Destructor
  ~HcalHF_S9S1algorithm();

  void HFSetFlagFromS9S1(HFRecHit& hf,
			HFRecHitCollection& rec,
			const HcalChannelQuality* myqual,
			const HcalSeverityLevelComputer* mySeverity);
  double CalcSlope(int abs_ieta, std::vector<double> params);
  double CalcEnergyThreshold(double abs_energy,std::vector<double> params);

  double bit(){return HcalCaloFlagLabels::HFLongShort;}

 private:

  std::vector<double> short_ET_;
  std::vector<double> short_Energy_; 
  std::vector<double> long_ET_;
  std::vector<double> long_Energy_;

  std::vector<double> LongSlopes;
  std::vector<double> ShortSlopes;
  std::vector<double> LongEnergyThreshold;
  std::vector<double> ShortEnergyThreshold;
  std::vector<double> LongETThreshold;
  std::vector<double> ShortETThreshold;
  int HcalAcceptSeverityLevel_;
  bool isS8S1_;
};


#endif
