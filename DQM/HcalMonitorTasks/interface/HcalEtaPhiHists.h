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
      for (unsigned int i=0;i<depth.size();++i)
	{
	  depth[i]->setAxisTitle("i#eta",1);
	  depth[i]->setAxisTitle("i#phi",2);
	}

      std::stringstream label;
      // set label on every other bin
      for (int i=-41;i<=-29;i=i+2)
	{
	  label<<i;
	  depth[0]->setBinLabel(i+42,label.str().c_str());
	  depth[1]->setBinLabel(i+42,label.str().c_str());
	  label.str("");
	}
      depth[0]->setBinLabel(14,"-29HE");
      depth[1]->setBinLabel(14,"-29HE");
      // offset by one for HE
      for (int i=-27;i<=27;i=i+2)
	{
	  label<<i;
	  depth[0]->setBinLabel(i+43,label.str().c_str());
	  depth[1]->setBinLabel(i+43,label.str().c_str());
	}
      depth[0]->setBinLabel(72,"29HE");
      depth[1]->setBinLabel(72,"29HE");
      for (int i=29;i<=41;i=i+2)
	{
	  label<<i;
	  depth[0]->setBinLabel(i+44,label.str().c_str());
	  depth[1]->setBinLabel(i+44,label.str().c_str());
	  label.str("");
	}

    };
  // special fill call based on detid -- eventually will need special treatment
  void Fill_d3(HcalDetId& id, double val=0){};
  std::vector<MonitorElement*> depth;
  
  
};

#endif
