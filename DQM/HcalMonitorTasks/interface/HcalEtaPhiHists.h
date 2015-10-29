#ifndef DQM_HCALMONITORTASKS_HCALETAPHIHISTS_H
#define DQM_HCALMONITORTASKS_HCALETAPHIHISTS_H

#include "TH1F.h"
#include "TH2F.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include <iostream>


// Define EtaPhi histograms

class EtaPhiHists{
  // Make a set of eta-phi histograms (one for each depth)
 public:
  EtaPhiHists(){};
  ~EtaPhiHists(){};


  void setup(DQMStore::IBooker & m_dbe,std::string Name, std::string Units="")
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
      depth.push_back(m_dbe.book2D(("HB HE HF Depth 1 "+name.str()+unitname.str()).c_str(),
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
      depth.push_back(m_dbe.book2D(("HB HE HF Depth 2 "+name.str()+unitname.str()).c_str(),
				    (name.str()+" Depth 2 -- HB HE HF ("+unittitle.str().c_str()+")"),
				    57, xbinsd2, 72, ybins));
      
      // Set up variable-sized bins for HE depth 3 (MonitorElement also requires phi bins to be entered in array format)
      float xbins[]={-28.5,-27.5,-26.5,-16.5,-15.5,
		     15.5,16.5,26.5,27.5,28.5};
      
      depth.push_back(m_dbe.book2D(("HE Depth 3 "+name.str()+unitname.str()).c_str(),
				    (name.str()+" Depth 3 -- HE ("+unittitle.str().c_str()+")"),
				    // Use variable-sized eta bins 
				    9, xbins, 72, ybins));
      // HO bins are fixed width, but cover a smaller eta range (-15 -> 15)
      depth.push_back(m_dbe.book2D(("HO Depth 4 "+name.str()+unitname.str()).c_str(),
				    (name.str()+" Depth 4 -- HO ("+unittitle.str().c_str()+")"),
				    31,-15.5,15.5,
				    72,0.5,72.5));
      for (unsigned int i=0;i<depth.size();++i)
	(depth[i]->getTH2F())->SetOption("colz");
      setBinLabels(); // set axis titles, special bins
      
    } // void setup(...)
  
  int getEtaBins(int depth)
  {
    if (depth==1) return 85;
    if (depth==2) return 57;
    if (depth==3) return 9;
    if (depth==4) return 31;
    return 0;
  }
  
  int getPhiBins(int depth)
  {
    if (depth>0 && depth<5)
      return 72;
    return 0;
  }

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

  void Reset(void) 
    {
      for (unsigned int d=0;d<depth.size();d++)
	if(depth[d]) depth[d]->Reset();
    }; // void Reset(void)

  //******************************************************************************************
  std::vector<MonitorElement*> depth;

};

// Now define functions that can be used in conjunction with EtaPhi histograms

// This arrays the eta binning for depth 2 histograms (with a gap between -15 -> +15)
const int binmapd2[]={-42,-41,-40,-39,-38,-37,-36,-35,-34,-33,-32,-31,-30,
				       -29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,
				       -16,-15,-9999, 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
				       30,31,32,33,34,35,36,37,38,39,40,41,42};

// This stores eta binning in depth 3 (where HE is only present at a few ieta values)

const int binmapd3[]={-28,-27,-9999,-16,-9999,16,-9999,27,28};

inline int CalcEtaBin(int subdet, int ieta, int depth)
{
  // This takes the eta value from a subdetector and return an eta counter value as used by eta-phi array
  // (ieta=-41 corresponds to bin 0, +41 to bin 85 -- there are two offsets to deal with the overlap at |ieta|=29).
  // For HO, ieta = -15 corresponds to bin 0, and ieta=15 is bin 30
  // For HE depth 3, things are more complicated, but feeding the ieta value will give back the corresponding counter eta value
  
  // The CalcEtaBin value is the value as used within our array counters, and thus starts at 0.
  // If you are using it with getBinContent or setBinContent, you will need to add +1 to the result of this function
  
  int etabin=-9999; // default invalid value

  if (depth==1)
    {
      // Depth 1 is fairly straightforward -- just shift HF-, HF+ by -/+1
      etabin=ieta+42;
      if (subdet==HcalForward)
	{
	  ieta < 0 ? etabin-- : etabin++;
	}
    }

  else if (depth==2)
    {
      // Depth 2 is more complicated, given that there are no cells in the range |ieta|<15
      if (ieta<-14)
	{
	  etabin=ieta+42;
	  if (subdet==HcalForward) etabin--;
	}
      else if (ieta>14)
	{
	  etabin=ieta+14;
	  if (subdet==HcalForward) etabin++;
	}
      
    }
  // HO is also straightforward; a simple offset to the ieta value is applied
  else if (subdet==HcalOuter && abs(ieta)<16)
    etabin=ieta+15;
  else if (subdet==HcalEndcap)
    {
      // HE depth 3 has spotty coverage; hard-code the bin response
      if (depth==3)
	{
	  if (ieta==-28) etabin=0;
	  else if (ieta==-27) etabin=1;
	  else if (ieta==-16) etabin=3;
	  else if (ieta==16)  etabin=5;
	  else if (ieta==27)  etabin=7;
	  else if (ieta==28)  etabin=8;
	}
    }
  return etabin;
}

inline int CalcIeta(int subdet, int eta, int depth)
{
  // This function returns the 'true' ieta value given subdet, eta, and depth
  // Here 'eta' is the index from our arrays (it starts at 0);
  // remember that histogram bins start with bin 1, so there's an offset of 1
  // to consider if using getBinContent(eta,phi)

  // eta runs from 0...X  (X depends on depth)
  int ieta=-9999; // default value is nonsensical
  if (subdet==HcalBarrel)
    {
      if (depth==1) 
	{
	  ieta=eta-42;
	  if (ieta==0) return -9999;
	  return ieta;
	}
      else if (depth==2)
	{
	  ieta=binmapd2[eta];
	  if (ieta==0) return -9999;
	  if (ieta==17 || ieta == -17) 
	    return -9999; // no depth 2 cells at |ieta| = 17
	  return ieta;
	}
      else
	return -9999; // non-physical value
    }
  else if (subdet==HcalForward)
    {
      if (depth==1)
	{
	  ieta=eta-42;
	  if (eta<13) ieta++;
	  else if (eta>71) ieta--;
	  else return -9999; // if outside forward range, return dummy
	  return ieta;
	}
      else if (depth==2)
	{
	  ieta=binmapd2[eta]; // special map for depth 2
	  if (ieta<=-30) ieta++;
	  else if (ieta>=30) ieta--;
	  else return -9999;
	  return ieta;
	}
      else return -9999;
    }

  else if (subdet==HcalEndcap)
    {
      if (depth==1) 
	ieta=eta-42;
      else if (depth==2) 
	{
	  ieta=binmapd2[eta];
	  if (abs(ieta)>29 || abs(ieta)<18) return -9999; // outside HE
	  if (ieta==0) return -9999;
	  return ieta;
	}
      else if (depth==3)
	{
	  if (eta<0 || eta>8) return -9999;
	  else
	    ieta=binmapd3[eta]; // special map for depth 3
	  if (ieta==0) return -9999;
	  return ieta;
	}
      else return -9999;
    } // HcalEndcap
  else if ( subdet==HcalOuter)
    {
      if (depth!=4)
	return -9999;
      else
	{
	  ieta= eta-15;  // bin 0 is ieta=-15, all bins increment normally from there
	  if (abs(ieta)>15) return -9999;
	  if (ieta==0) return -9999;
	  return ieta;
	}
    } // HcalOuter
  if (ieta==0) return -9999;
  return ieta;
}

inline int CalcIeta(int eta, int depth)
{
  // This version of CalcIeta does the same as the function above,
  // but does not require that 'subdet' be specified.

  // returns ieta value give an eta counter.
  // eta runs from 0...X  (X depends on depth)
  int ieta=-9999;
  if (eta<0) return ieta;
  if (depth==1)
    {
      ieta=eta-42; // default shift: bin 0 corresponds to a histogram ieta of -42 (which is offset by 1 from true HF value of -41)
      if (eta<13) ieta++;
      else if (eta>71) ieta--;
      if (ieta==0) ieta=-9999;
      return ieta;
    }
  else if (depth==2)
    {
      if (eta>56) return -9999;
      else
	{
	  ieta=binmapd2[eta];
	  if (ieta==-9999) return ieta;
	  if (ieta==0) return -9999;
	  if (ieta==17 || ieta == -17) return -9999; // no depth 2 cells at |ieta| = 17
	  else if (ieta<=-30) ieta++;
	  else if (ieta>=30) ieta--;
	  return ieta;
	}
    }
  else if (depth==3)
    {
      if (eta>8) return -9999;
      else
	ieta=binmapd3[eta];
      if (ieta==0) return -9999;
      return ieta;
    }
  else if (depth==4)
    {
      ieta= eta-15;  // bin 0 is ieta=-15, all bins increment normally from there
      if (abs(ieta)>15) return -9999;
      if (ieta==0) return -9999;
      return ieta;
    }
  return ieta; // avoids compilation warning
}


// Functions to check whether a given (eta,depth) value is valid for a given subdetector

inline std::vector<std::string> HcalEtaPhiHistNames()
{
  std::vector<std::string> name;
  name.push_back("HB HE HF Depth 1 ");
  name.push_back("HB HE HF Depth 2 ");
  name.push_back("HE Depth 3 ");
  name.push_back("HO Depth 4 ");
  return name;
}


inline bool isHB(int etabin, int depth)
{
  if (depth>2) return false;
  else if (depth<1) return false;
  else
    {
      int ieta=CalcIeta(etabin,depth);
      if (ieta==-9999) return false;
      if (depth==1)
        {
          if (abs(ieta)<=16 ) return true;
          else return false;
        }
      else if (depth==2)
        {
          if (abs(ieta)==15 || abs(ieta)==16) return true;
          else return false;
        }
    }
  return false;
}

inline bool isHE(int etabin, int depth)
{
  if (depth>3) return false;
  else if (depth<1) return false;
  else
    {
      int ieta=CalcIeta(etabin,depth);
      if (ieta==-9999) return false;
      if (depth==1)
        {
          if (abs(ieta)>=17 && abs(ieta)<=28 ) return true;
          if (ieta==-29 && etabin==13) return true; // HE -29
          if (ieta==29 && etabin == 71) return true; // HE +29
        }
      else if (depth==2)
        {
          if (abs(ieta)>=17 && abs(ieta)<=28 ) return true;
          if (ieta==-29 && etabin==13) return true; // HE -29
          if (ieta==29 && etabin == 43) return true; // HE +29
        }
      else if (depth==3)
        return true;
    }
  return false;
}

inline bool isHF(int etabin, int depth)
{
  if (depth>2) return false;
  else if (depth<1) return false;
  else
    {
      int ieta=CalcIeta(etabin,depth);
      if (ieta==-9999) return false;
      if (depth==1)
        {
          if (ieta==-29 && etabin==13) return false; // HE -29
          else if (ieta==29 && etabin == 71) return false; // HE +29
          else if (abs(ieta)>=29 ) return true;
        }
      else if (depth==2)
        {
          if (ieta==-29 && etabin==13) return false; // HE -29
          else if (ieta==29 && etabin==43) return false; // HE +29
          else if (abs(ieta)>=29 ) return true;
        }
    }
  return false;
}

inline bool isHO(int etabin, int depth)
{
  if (depth!=4) return false;
  int ieta=CalcIeta(etabin,depth);
  if (ieta!=-9999) return true;
  return false;
}

// Checks whether HO region contains SiPM

inline bool isSiPM(int ieta, int iphi, int depth)
{
  if (depth!=4) return false;
  // HOP1
  if (ieta>=5 && ieta <=10 && iphi>=47 && iphi<=58) return true;  
  // HOP2
  if (ieta>=11 && ieta<=15 && iphi>=59 && iphi<=70) return true;
  return false;
}  // bool isSiPM

/*
// Checks whether (subdet, ieta, iphi, depth) value is a valid Hcal cell

inline bool validDetId(HcalSubdetector sd, int ies, int ip, int dp)
{
  // inputs are (subdetector, ieta, iphi, depth)
  // stolen from latest version of DataFormats/HcalDetId/src/HcalDetId.cc (not yet available in CMSSW_2_1_9)

  const int ie ( abs( ies ) ) ;

  return ( ( ip >=  1         ) &&
	   ( ip <= 72         ) &&
	   ( dp >=  1         ) &&
	   ( ie >=  1         ) &&
	   ( ( ( sd == HcalBarrel ) &&
	       ( ( ( ie <= 14         ) &&
		   ( dp ==  1         )    ) ||
		 ( ( ( ie == 15 ) || ( ie == 16 ) ) && 
		   ( dp <= 2          )                ) ) ) ||
	     (  ( sd == HcalEndcap ) &&
		( ( ( ie == 16 ) &&
		    ( dp ==  3 )          ) ||
		  ( ( ie == 17 ) &&
		    ( dp ==  1 )          ) ||
		  ( ( ie >= 18 ) &&
		    ( ie <= 20 ) &&
		    ( dp <=  2 )          ) ||
		  ( ( ie >= 21 ) &&
		    ( ie <= 26 ) &&
		    ( dp <=  2 ) &&
		    ( ip%2 == 1 )         ) ||
		  ( ( ie >= 27 ) &&
		    ( ie <= 28 ) &&
		    ( dp <=  3 ) &&
		    ( ip%2 == 1 )         ) ||
		  ( ( ie == 29 ) &&
		    ( dp <=  2 ) &&
		    ( ip%2 == 1 )         )          )      ) ||
	     (  ( sd == HcalOuter ) &&
		( ie <= 15 ) &&
		( dp ==  4 )           ) ||
	     (  ( sd == HcalForward ) &&
		( dp <=  2 )          &&
		( ( ( ie >= 29 ) &&
		    ( ie <= 39 ) &&
		    ( ip%2 == 1 )    ) ||
		  ( ( ie >= 40 ) &&
		    ( ie <= 41 ) &&
		    ( ip%4 == 3 )         )  ) ) ) ) ;



} // bool validDetId(HcalSubdetector sd, int ies, int ip, int dp)
*/

// Sets eta, phi labels for 'summary' eta-phi plots (identical to Depth 1 Eta-Phi labelling)

inline void SetEtaPhiLabels(MonitorElement* h)
{
  std::stringstream label;
  for (int i=-41;i<=-29;i=i+2)
    {
      label<<i;
      h->setBinLabel(i+42,label.str().c_str());
      label.str("");
    }
  h->setBinLabel(14,"-29HE");
    
  // offset by one for HE
  for (int i=-27;i<=27;i=i+2)
    {
      label<<i;
      h->setBinLabel(i+43,label.str().c_str());
      label.str("");
    }
  h->setBinLabel(72,"29HE");
  for (int i=29;i<=41;i=i+2)
    {
      label<<i;
      h->setBinLabel(i+44,label.str().c_str());
      label.str("");
    }
  return;
}


// Fill Unphysical bins in histograms

inline void FillUnphysicalHEHFBins(EtaPhiHists &hh)
{
  int ieta=0;
  int iphi=0;
  // First 2 depths have 5-10-20 degree corrections
  for (unsigned int d=0;d<3;++d)
    {
      if (!hh.depth[d]) continue;
      for (int eta=0;eta<hh.depth[d]->getNbinsX();++eta)
	{
	  ieta=CalcIeta(eta,d+1);
	  if (ieta==-9999 || abs(ieta)<21) continue;
	  for (int phi=0;phi<hh.depth[d]->getNbinsY();++phi)
	    {
	      iphi=phi+1;
	      if (iphi%2==1 && abs(ieta)<40 && iphi<73)
		{
		  hh.depth[d]->setBinContent(eta+1,iphi+1,hh.depth[d]->getBinContent(eta+1,iphi));
		}
	      // last two eta strips span 20 degrees in phi
	      // Fill the phi cell above iphi, and the 2 below it
	      else  if (abs(ieta)>39 && iphi%4==3 && iphi<73)
		{
		  //ieta=40, iphi=3 covers iphi 3,4,5,6
		  hh.depth[d]->setBinContent(eta+1,(iphi)%72+1, hh.depth[d]->getBinContent(eta+1,iphi));
		  hh.depth[d]->setBinContent(eta+1,(iphi+1)%72+1, hh.depth[d]->getBinContent(eta+1,iphi));
		  hh.depth[d]->setBinContent(eta+1,(iphi+2)%72+1, hh.depth[d]->getBinContent(eta+1,iphi));
		}
	    } // for (int phi...)
	} // for (int eta...)
    } // for (int d=0;...)
  // no corrections needed for HO (depth 4)
  return;
} // FillUnphysicalHEHFBins(MonitorElement* hh)


// Fill unphysical bins for single ME
inline void FillUnphysicalHEHFBins(MonitorElement* hh)
{
  // Fills unphysical HE/HF bins for Summary Histogram
  // Summary Histogram is binned with the same binning as the Depth 1 EtaPhiHists
  if (hh==0) return; 
  int ieta=0;
  int iphi=0;
  int etabins = hh->getNbinsX();
  int phibins = hh->getNbinsY();
  float binval=0;
  for (int eta=0;eta<etabins;++eta) // loop over eta bins
    {
      ieta=CalcIeta(eta,1);
      if (ieta==-9999 || abs(ieta)<21) continue;  // ignore etas that don't exist, or that have 5 degree phi binning

      for (int phi=0;phi<phibins;++phi)
        {
	  iphi=phi+1;
	  if (iphi%2==1 && abs(ieta)<40 && iphi<73) // 10 degree phi binning condition
	    {
	      binval=hh->getBinContent(eta+1,iphi);
	      hh->setBinContent(eta+1,iphi+1,binval);
	    } // if (iphi%2==1...) 
	  else if (abs(ieta)>39 && iphi%4==3 && iphi<73) // 20 degree phi binning condition
	    {
	      // Set last two eta strips where each cell spans 20 degrees in phi
	      // Set next phi cell above iphi, and 2 cells below the actual cell 
	      hh->setBinContent(eta+1, (iphi)%72+1, hh->getBinContent(eta+1,iphi));
	      hh->setBinContent(eta+1, (iphi+1)%72+1, hh->getBinContent(eta+1,iphi));
	      hh->setBinContent(eta+1, (iphi+2)%72+1, hh->getBinContent(eta+1,iphi));
	    } // else if (abs(ieta)>39 ...)
	} // for (int phi=0;phi<72;++phi)

    } // for (int eta=0; eta< (etaBins_-2);++eta)

  return;
} // FillUnphysicalHEHFBins(std::vector<MonitorElement*> &hh)




#endif
