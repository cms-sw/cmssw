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

  void setup(DQMStore* &m_dbe,std::string Name, std::string Units="")
    {
      std::stringstream name;
      name<<Name;

      std::stringstream unitname;
      std::stringstream unittitle;
      std::string s(Units);
      if (s.empty())
	{
	  unitname<<Units;
	  unittitle<<"No Units";
	}
      else
	{
	  unitname<<" "<<Units;
	  unittitle<<Units;
	}
      
      // Push back depth plots
      depth.push_back(m_dbe->book2D(("HB HE HF Depth 1 "+name.str()+unitname.str()).c_str(),
				    (name.str()+" Depth 1 -- HB HE HF ("+unittitle.str().c_str()+")"),
				    85,-42.5,42.5,
				    72,0.5,72.5));
      float ybins[73];
      for (int i=0;i<=72;i++) ybins[i]=(float)(i+0.5);
      float xbinsd2[]={-42.5,-41.5,-40.5,-39.5,-38.5,-37.5,-36.5,-35.5,-34.5,-33.5,-32.5,-31.5,-30.5,-29.5,
		       -28.5,-27.5,-26.5,-25.5,-24.5,-23.5,-22.5,-21.5,-20.5,-19.5,-18.5,-17.5,-16.5,
		       -15.5,-14.5,
		       14.5, 15.5,
		       16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5,30.5,
		       31.5,32.5,33.5,34.5,35.5,36.5,37.5,38.5,39.5,40.5,41.5,42.5};
      depth.push_back(m_dbe->book2D(("HB HE HF Depth 2 "+name.str()+unitname.str()).c_str(),
				    (name.str()+" Depth 2 -- HB HE HF ("+unittitle.str().c_str()+")"),
				    57, xbinsd2, 72, ybins));
      
      // Set up variable-sized bins for HE depth 3 (MonitorElement also requires phi bins to be entered in array format)
      float xbins[]={-28.5,-27.5,-26.5,-16.5,-15.5,
		     15.5,16.5,26.5,27.5,28.5};
      
      depth.push_back(m_dbe->book2D(("HE Depth 3 "+name.str()+unitname.str()).c_str(),
				    (name.str()+" Depth 3 -- HE ("+unittitle.str().c_str()+")"),
				    // Use variable-sized eta bins 
				    9, xbins, 72, ybins));
      // HO bins are fixed width, but cover a smaller eta range (-15 -> 15)
      depth.push_back(m_dbe->book2D(("HO Depth 4 "+name.str()+unitname.str()).c_str(),
				    (name.str()+" Depth 4 -- HO ("+unittitle.str().c_str()+")"),
				    31,-15.5,15.5,
				    72,0.5,72.5));
      setBinLabels(); // set axis titles, special bins
      
    } // void setup(...)
  
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
	  label.str("");
	}
      depth[0]->setBinLabel(72,"29HE");
      for (int i=29;i<=41;i=i+2)
	{
	  label<<i;
	  depth[0]->setBinLabel(i+44,label.str().c_str());
	  label.str("");
	}
      for (int i=16;i<=28;i=i+2)
	{
	  label<<i-43;
	  depth[1]->setBinLabel(i,label.str().c_str());
	  label.str("");
	}
      depth[1]->setBinLabel(29,"NULL");
      for (int i=15;i<=27;i=i+2)
	{
	  label<<i;
	  depth[1]->setBinLabel(i+15,label.str().c_str());
	  label.str("");
	}

      depth[1]->setBinLabel(44,"29HE");
      for (int i=29;i<=41;i=i+2)
	{
	  label<<i;
	  depth[1]->setBinLabel(i+16,label.str().c_str());
	  label.str("");
	}

      // HE depth 3 labels;
      depth[2]->setBinLabel(1,"-28");
      depth[2]->setBinLabel(2,"-27");
      depth[2]->setBinLabel(3,"Null");
      depth[2]->setBinLabel(4,"-16");
      depth[2]->setBinLabel(5,"Null");
      depth[2]->setBinLabel(6,"16");
      depth[2]->setBinLabel(7,"Null");
      depth[2]->setBinLabel(8,"27");
      depth[2]->setBinLabel(9,"28");
     
      for (int d=0;d<4;++d)
	{
	  depth[d]->setAxisTitle("i#eta",1);
	  depth[d]->setAxisTitle("i#phi",2); 
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

  //******************************************************************************************

  // elements -- should we make a base Eta/Phi class that would contain a histo, nbinsx, nbinsy, etc.?
  std::vector<MonitorElement*> depth;

};

#endif
