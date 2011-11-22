#ifndef HcalRecAlgos_HcalTDCReco_h
#define HcalRecAlgos_HcalTDCReco_h

class HcalUpgradeDataFrame;
class HcalUpgradeRecHit;

class HcalTDCReco
{
public:
  HcalTDCReco();
  void reconstruct(const HcalUpgradeDataFrame & digi, HcalUpgradeRecHit & recHit) const;
};
#endif

