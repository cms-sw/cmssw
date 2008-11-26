#include "CalibTracker/SiStripQuality/interface/SiStripHotStripAlgorithmFromClusterOccupancy.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"


SiStripHotStripAlgorithmFromClusterOccupancy::~SiStripHotStripAlgorithmFromClusterOccupancy(){
  LogTrace("SiStripHotStripAlgorithmFromClusterOccupancy")<<"[SiStripHotStripAlgorithmFromClusterOccupancy::~SiStripHotStripAlgorithmFromClusterOccupancy] "<<std::endl;
}

void SiStripHotStripAlgorithmFromClusterOccupancy::extractBadStrips(SiStripQuality* siStripQuality,HistoMap& DM){

  _StripOccupancyHotStrips.clear();

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
    DetId detectorId=DetId(detid);
    phisto._SubdetId=detectorId.subdetId();
    
   if (edm::isDebugEnabled())
     LogTrace("SiStripHotStrip") << "Analyzing detid " << detid<< std::endl;
   
    pQuality=siStripQuality;
    badStripList.clear();
    iterativeSearch(phisto,badStripList);
    
    if (badStripList.begin()==badStripList.end())
      continue;

    siStripQuality->compact(detid,badStripList);


    SiStripQuality::Range range(badStripList.begin(),badStripList.end());
    if ( ! siStripQuality->put(detid,range) )
      edm::LogError("SiStripHotStrip")<<"[SiStripHotStripAlgorithmFromClusterOccupancy::extractBadStrips] detid already exists"<<std::endl;
  }
  siStripQuality->fillBadComponents();
  LogTrace("SiStripHotStrip") << ss.str() << std::endl;
}

  
void SiStripHotStripAlgorithmFromClusterOccupancy::iterativeSearch(pHisto& histo,std::vector<unsigned int>& vect){
  if (!histo._NEntries || histo._NEntries <=MinNumEntries_ || histo._NEntries <= minNevents_)
    return;
  
  size_t startingSize=vect.size();
  long double diff=1.-prob_; 
  
  int Nbins     = histo._th1f->GetNbinsX();
  int ibinStart = 1; 
  int ibinStop  = Nbins+1; 
  int MaxEntry  = (int)histo._th1f->GetMaximum();
  int subdetid  = histo._SubdetId;

  std::vector<long double> vPoissonProbs(MaxEntry+1,0);
  long double meanVal=1.*histo._NEntries/(1.*Nbins-histo._NEmptyBins); 
  evaluatePoissonian(vPoissonProbs,meanVal);

  for (Int_t i=ibinStart; i<ibinStop; ++i){
    unsigned int entries= (unsigned int)histo._th1f->GetBinContent(i);
    _StripOccupancyAllStrips.push_back(std::make_pair(entries/(double) Nevents_,subdetid));

    if (entries<=MinNumEntriesPerStrip_ || entries <= minNevents_)
      continue;

    if(diff<vPoissonProbs[entries]){
      _StripOccupancyHotStrips.push_back(std::make_pair(entries/(double) Nevents_,subdetid));
      histo._th1f->SetBinContent(i,0.);
      histo._NEntries-=entries;
      histo._NEmptyBins++;
      if (edm::isDebugEnabled())
	LogTrace("SiStripHotStrip")<< " rejecting strip " << i-1 << " value " << entries << " diff  " << diff << " prob " << vPoissonProbs[entries]<< std::endl;
      vect.push_back(pQuality->encode(i-1,1,0));
    }
  }
  if (edm::isDebugEnabled())
    LogTrace("SiStripHotStrip") << " [SiStripHotStripAlgorithmFromClusterOccupancy::iterativeSearch] Nbins="<< Nbins << " MaxEntry="<<MaxEntry << " meanVal=" << meanVal << " NEmptyBins="<<histo._NEmptyBins<< " NEntries=" << histo._NEntries << " thEntries " << histo._th1f->GetEntries()<< " startingSize " << startingSize << " vector.size " << vect.size() << std::endl;

  if (vect.size()!=startingSize)
    iterativeSearch(histo,vect);
}

void SiStripHotStripAlgorithmFromClusterOccupancy::evaluatePoissonian(std::vector<long double>& vPoissonProbs, long double& meanVal){
  for(size_t i=0;i<vPoissonProbs.size();++i){
    vPoissonProbs[i]= (i==0)?TMath::Poisson(i,meanVal):vPoissonProbs[i-1]+TMath::Poisson(i,meanVal);
  }
}

void SiStripHotStripAlgorithmFromClusterOccupancy::setNumberOfEvents(uint32_t Nevents){
  Nevents_=Nevents;
  minNevents_=occupancy_*Nevents_; 
  if (edm::isDebugEnabled())                                                                                                                                                                          
    LogTrace("SiStripHotStrip")<<" [SiStripHotStripAlgorithmFromClusterOccupancy::setNumberOfEvents] minNumber of Events per strip used to consider a strip bad" << minNevents_ << " for occupancy " << occupancy_ << std::endl;
}
