//Typedef

#ifndef CASTORBUNCH_H
#define CASTORBUNCH_H 1

      struct NewBunch
        {
          HcalCastorDetId detid;
          bool usedflag;
          double tsCapId[20];
          double tsAdc[20];
	  // pedestal??
          double tsfC[20];
        };

#endif


