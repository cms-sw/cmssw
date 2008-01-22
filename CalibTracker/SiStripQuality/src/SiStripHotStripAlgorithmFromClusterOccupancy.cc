#include "CalibTracker/SiStripQuality/interface/SiStripHotStripAlgorithmFromClusterOccupancy.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include <iostream>

SiStripHotStripAlgorithmFromClusterOccupancy::~SiStripHotStripAlgorithmFromClusterOccupancy(){
  LogTrace("SiStripHotStripAlgorithmFromClusterOccupancy")<<"[SiStripHotStripAlgorithmFromClusterOccupancy::~SiStripHotStripAlgorithmFromClusterOccupancy] "<<std::endl;
}

void SiStripHotStripAlgorithmFromClusterOccupancy::extractBadStrips(SiStripQuality* siStripQuality,HistoMap& DM){

  LogTrace("SiStripHotStripAlgorithmFromClusterOccupancy")<<"[SiStripHotStripAlgorithmFromClusterOccupancy::extractBadStrips] "<<std::endl;

  HistoMap::iterator it=DM.begin();
  HistoMap::iterator itEnd=DM.end();
  std::vector<unsigned int> badStripList;
  uint32_t detid;
  for (;it!=itEnd;++it){
    pHisto phisto;
    phisto._th1f=it->second.get();
    phisto._NEntries=phisto._th1f->GetEntries();
    detid=it->first;
    
    pQuality=siStripQuality;
    badStripList.clear();
    iterativeSearch(phisto,badStripList);
    
    if (badStripList.begin()==badStripList.end())
      continue;

    siStripQuality->compact(detid,badStripList);


    SiStripQuality::Range range(badStripList.begin(),badStripList.end());
    if ( ! siStripQuality->put(detid,range) )
      edm::LogError("SiStripHotStripAlgorithmFromClusterOccupancy")<<"[SiStripHotStripAlgorithmFromClusterOccupancy::extractBadStrips] detid already exists"<<std::endl;
  }
  siStripQuality->fillBadComponents();
  LogTrace("SiStripHotStripAlgorithmFromClusterOccupancy") << ss.str() << std::endl;
}

  
void SiStripHotStripAlgorithmFromClusterOccupancy::iterativeSearch(pHisto& histo,std::vector<unsigned int>& vect){
  if (!histo._NEntries || histo._NEntries <MinNumEntries_)
    return;
  
  size_t startingSize=vect.size();
  long double diff=1.-prob_; 
  
  int Nbins=histo._th1f->GetNbinsX();
  int ibinStart= 1; 
  int ibinStop= Nbins+1; 
  int MaxEntry=(int)histo._th1f->GetMaximum();

  std::vector<long double> vPoissonProbs(MaxEntry+1,0);
  float meanVal=histo._NEntries/(Nbins-histo._NEmptyBins); 
  evaluatePoissonian(vPoissonProbs,meanVal);

  for (Int_t i=ibinStart; i<ibinStop; ++i){
    unsigned int entries= (unsigned int)histo._th1f->GetBinContent(i);
    if (entries<MinNumEntriesPerStrip_)
      continue;

    if(diff<vPoissonProbs[entries]){
      histo._th1f->SetBinContent(i,0.);
      histo._NEntries-=entries;
      histo._NEmptyBins++;
      if (edm::isDebugEnabled())
	ss << " [SiStripHotStripAlgorithmFromClusterOccupancy::iterativeSearch] rejecting strip " << i-1 << std::endl;
      vect.push_back(pQuality->encode(i-1,1,0));
    }
  }
  if (edm::isDebugEnabled())
    ss << " [SiStripHotStripAlgorithmFromClusterOccupancy::iterativeSearch] Nbins="<< Nbins << " MaxEntry="<<MaxEntry << " meanVal=" << meanVal << " NEmptyBins="<<histo._NEmptyBins<< " NEntries=" << histo._NEntries << " " << histo._th1f->GetEntries()<< " startingSize " << startingSize << " vector.size " << vect.size() << std::endl;

  if (vect.size()!=startingSize)
    iterativeSearch(histo,vect);
}

void SiStripHotStripAlgorithmFromClusterOccupancy::evaluatePoissonian(std::vector<long double>& vPoissonProbs, float& meanVal){
  for(size_t i=0;i<vPoissonProbs.size();++i){
    vPoissonProbs[i]= (i==0)?TMath::Poisson(i,meanVal):vPoissonProbs[i-1]+TMath::Poisson(i,meanVal);
  }
}
