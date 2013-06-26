#ifndef HcalRecAlgos_HcalTDCReco_h
#define HcalRecAlgos_HcalTDCReco_h

class HcalUpgradeDataFrame;
class HBHERecHit;

class HcalTDCReco
{
public:
  HcalTDCReco();
  void reconstruct(const HcalUpgradeDataFrame & digi, HBHERecHit & recHit) const;
};
#endif

