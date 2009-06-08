#ifndef DQM_HCALMONITORTASKS_HCALETAPHIHISTS_H
#define DQM_HCALMONITORTASKS_HCALETAPHIHISTS_H

#include "TH1F.h"
#include "TH2F.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <iostream>

class EtaPhiHists{
  // Make a set of eta-phi histograms (one for each depth)
 public:
  EtaPhiHists();
  ~EtaPhiHists();

  void setBinLabels()
    {
      // Set labels for all depth histograms
      d1->setAxisTitle("i#eta",1);
      d1->setAxisTitle("i#phi",2);
      d2->setAxisTitle("i#eta",1);
      d2->setAxisTitle("i#phi",2);
      d3->setAxisTitle("i#eta",1);
      d3->setAxisTitle("i#phi",2);
      d4->setAxisTitle("i#eta",1);
      d4->setAxisTitle("i#phi",2);

      std::stringstream label;
      // set label on every other bin
      for (int i=-41;i<=-29;i=i+2)
	{
	  label<<i;
	  d1->setBinLabel(i+42,label.str().c_str());
	  d2->setBinLabel(i+42,label.str().c_str());
	  label.str("");
	}
      d1->setBinLabel(14,"-29HE");
      d2->setBinLabel(14,"-29HE");
      // offset by one for HE
      for (int i=-27;i<=27;i=i+2)
	{
	  label<<i;
	  d1->setBinLabel(i+43,label.str().c_str());
	  d2->setBinLabel(i+43,label.str().c_str());
	}
      d1->setBinLabel(72,"29HE");
      d2->setBinLabel(72,"29HE");
      for (int i=29;i<=41;i=i+2)
	{
	  label<<i;
	  d1->setBinLabel(i+44,label.str().c_str());
	  d2->setBinLabel(i+44,label.str().c_str());
	  label.str("");
	}

    };
  // special fill call for depth 3 -- eventually will need special treatment
  void Fill_d3(int ieta, int iphi, double val=0){};
  MonitorElement *d1;
  MonitorElement *d2;
  MonitorElement *d3;
  MonitorElement *d4;
  
  
};

#endif
