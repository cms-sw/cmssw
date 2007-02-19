#ifndef MYHCALCLASSES_INCLUDED
#define MYHCALCLASSES_INCLUDED

  enum MyHcalSubdetector { HcalEmpty=0,
			   HcalBarrel=1,
			   HcalEndcap=2,
			   HcalOuter=3,
			   HcalForward=4,
			   HcalTriggerTower=5,
			   HcalOther=7 };
  typedef struct {
    MyHcalSubdetector subdet;
    int               ieta;
    int               iphi;
    int               depth;
  }
  MyHcalDetId;

#endif // MYHCALCLASSES_INCLUDED
