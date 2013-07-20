#ifndef HCALHBHETIMESTATUSFROMDIGIS_H
#define HCALHBHETIMESTATUSFROMDIGIS_H 1

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

/** \class HBHETimeProfileStatusBitSetter
    
   This class sets status bit in the status words for the revised CaloRecHit objets according to informatino from the digi associated to the hit.
    
   $Date: 2013/05/30 21:37:33 $
   $Revision: 1.2 $
   \author B Jones -- University of Bristol / University of Maryland
*/

class HBHETimeProfileStatusBitSetter {
public:
  /** Full featured constructor for HB/HE and HO (HPD-based detectors) */
  HBHETimeProfileStatusBitSetter();
  HBHETimeProfileStatusBitSetter(double R1Min, double R1Max, 
				 double R2Min, double R2Max, 
				 double FracLeaderMin, double FracLeaderMax, 
				 double SlopeMin, double SlopeMax, 
				 double OuterMin, double OuterMax, double EnergyThreshold); 
  
  // Destructor
  ~HBHETimeProfileStatusBitSetter();

  // Methods for setting the status flag values
  void hbheSetTimeFlagsFromDigi(HBHERecHitCollection *, const std::vector<HBHEDataFrame>&, const std::vector<int>&);




  // setter functions
  void SetExpLimits(double R1Min, double R1Max, double R2Min, double R2Max)
    { R1Min_ = R1Min; R1Max_ = R1Max;  R2Min_ = R2Max; R2Max_ = R2Max; }
  void SetFracLeaderLimits(double FracLeaderMin, double FracLeaderMax)
    { FracLeaderMin_ = FracLeaderMin; FracLeaderMax_ = FracLeaderMax;}
  void SetSlopeLimits(double SlopeMin, double SlopeMax)
    { SlopeMin_ = SlopeMin; SlopeMax_ = SlopeMax;}
  void SetOuterLimits(double OuterMin, double OuterMax)
    { OuterMin_ = OuterMin; OuterMax_ = OuterMax;}
  double EnergyThreshold(){return EnergyThreshold_;}

private:
  // variables for cfg files
  double R1Min_, R1Max_, R2Min_, R2Max_;
  double FracLeaderMin_,FracLeaderMax_;
  double SlopeMin_,SlopeMax_;
  double OuterMin_,OuterMax_;
  double EnergyThreshold_;
  struct compare_digi_energy : public std::binary_function<HBHEDataFrame, HBHEDataFrame, bool> {
    bool operator()(const HBHEDataFrame& x, const HBHEDataFrame& y) {
      double TotalX=0, TotalY=0;
      for(int i=0; i!=x.size(); TotalX += x.sample(i++).nominal_fC());
      for(int i=0; i!=y.size(); TotalY += y.sample(i++).nominal_fC());

      return (TotalX>TotalY) ;

    }
  };
 
  double TotalEnergyInDataFrame(const HBHEDataFrame& x) {
    double Total=0;
    for(int i=0; i!=x.size(); Total += x.sample(i++).nominal_fC());
    return Total;
  }
      

};

#endif
