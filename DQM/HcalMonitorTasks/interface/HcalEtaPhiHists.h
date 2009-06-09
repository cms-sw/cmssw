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
  EtaPhiHists(){};
  ~EtaPhiHists(){};

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
	  label.str("");
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
  void Fill(HcalDetId& id, double val=1)
    { 
      // If in HF, need to shift by 1 bin (-1 bin lower in -HF, +1 bin higher in +HF)
      if (id.subdet()==HcalForward)
	depth[id.depth()-1]->Fill(id.ieta()<0 ? id.ieta()-1 : id.ieta()+1, id.iphi(), val);
      else 
	depth[id.depth()-1]->Fill(id.ieta(),id.iphi(),val);
    };

  int CalcIeta(HcalSubdetector subdet, int eta, int depth)
    {
      int ieta;
      ieta=eta-43; // default shift: bin 1 corresponds to a histogram ieta of -42 (which is offset by 1 from true HF value of -41)
      if (subdet==HcalBarrel)
	{
	  if (depth>2) 
	    ieta=-9999; // non-physical value
	}
      else if (subdet==HcalForward)
	{
	  if (depth>2)
	    ieta=-9999;
	  if (eta<14) ieta++;
	  else if (eta>72) ieta--;
	  else ieta=-9999; // if outside forward range, return dummy
	}
      // add in HE depth 3, HO later
      return ieta;
    };
  
  int CalcIeta(int eta, int depth)
    {
      int ieta;
      ieta=eta-43; // default shift: bin 1 corresponds to a histogram ieta of -42 (which is offset by 1 from true HF value of -41)
      if (depth<=2)
	{
	  if (eta<14) ieta++;
	  else if (eta>72) ieta--;
	}
      // add in HE depth 3, HO later
      return ieta;
    };

  // elements -- should we make a base Eta/Phi class that would contain a histo, nbinsx, nbinsy, etc.?
  std::vector<MonitorElement*> depth;
  


};

#endif
