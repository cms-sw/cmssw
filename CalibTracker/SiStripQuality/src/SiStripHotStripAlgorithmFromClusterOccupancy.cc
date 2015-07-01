#include "CalibTracker/SiStripQuality/interface/SiStripHotStripAlgorithmFromClusterOccupancy.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"




SiStripHotStripAlgorithmFromClusterOccupancy::SiStripHotStripAlgorithmFromClusterOccupancy(const edm::ParameterSet& iConfig, const TrackerTopology* theTopo):
    prob_(1.E-7),
    ratio_(1.5),
    MinNumEntries_(0),
    MinNumEntriesPerStrip_(0),
    Nevents_(0),
    occupancy_(0),
    OutFileName_("Occupancy.root"),
    tTopo(theTopo),
    UseInputDB_(iConfig.getUntrackedParameter<bool>("UseInputDB",false))
  {  
    minNevents_=Nevents_*occupancy_;
  }


SiStripHotStripAlgorithmFromClusterOccupancy::~SiStripHotStripAlgorithmFromClusterOccupancy(){
  LogTrace("SiStripHotStripAlgorithmFromClusterOccupancy")<<"[SiStripHotStripAlgorithmFromClusterOccupancy::~SiStripHotStripAlgorithmFromClusterOccupancy] "<<std::endl;
}

void SiStripHotStripAlgorithmFromClusterOccupancy::extractBadStrips(SiStripQuality* OutSiStripQuality,HistoMap& DM,edm::ESHandle<SiStripQuality>& InSiStripQuality){

  LogTrace("SiStripHotStripAlgorithmFromClusterOccupancy")<<"[SiStripHotStripAlgorithmFromClusterOccupancy::extractBadStrips] "<<std::endl;


  
  if (WriteOutputFile_==true){
  f = new TFile(OutFileName_.c_str(),"RECREATE");
  f->cd();

  striptree = new TTree("stripOccupancy","tree");

  striptree->Branch("DetRawId",             &detrawid,          "DetRawId/I");
  striptree->Branch("SubDetId",             &subdetid,          "SubDetId/I");
  striptree->Branch("Layer_Ring",           &layer_ring,        "Layer_Ring/I");
  striptree->Branch("Disc",                 &disc,              "Disc/I");
  striptree->Branch("IsBack",               &isback,            "IsBack/I");
  striptree->Branch("IsExternalString",     &isexternalstring,  "IsExternalString/I");
  striptree->Branch("IsZMinusSide",         &iszminusside,      "IsZMinusSide/I");
  striptree->Branch("RodStringPetal",       &rodstringpetal,    "RodStringPetal/I");
  striptree->Branch("IsStereo",             &isstereo,          "IsStereo/I");
  striptree->Branch("ModulePosition",       &module_position,   "ModulePosition/I");
  striptree->Branch("NumberOfStrips",       &number_strips,     "NumberOfStrips/I");
  striptree->Branch("StripNumber",          &strip_number,      "StripNumber/I");
  striptree->Branch("APVChannel",           &apv_channel,       "APVChannel/I");
  striptree->Branch("StripGlobalPositionX", &global_position_x, "StripGlobalPositionX/F");
  striptree->Branch("StripGlobalPositionY", &global_position_y, "StripGlobalPositionY/F");
  striptree->Branch("StripGlobalPositionZ", &global_position_z, "StripGlobalPositionZ/F");
  striptree->Branch("IsHot",                &isHot,             "IsHot/I");
  striptree->Branch("HotStripsPerAPV",      &hotStripsPerAPV,   "HotStripsPerAPV/I");
  striptree->Branch("HotStripsPerModule",   &hotStripsPerModule,"HotStripsPerModule/I");
  striptree->Branch("StripOccupancy",       &stripOccupancy,    "StripOccupancy/D");
  striptree->Branch("StripHits",            &stripHits,         "StripHits/I");
  striptree->Branch("PoissonProb",          &poissonProb,       "PoissonProb/D");
  striptree->Branch("MedianAPVHits",        &medianAPVHits,     "MedianAPVHits/D");
  striptree->Branch("AvgAPVHits",           &avgAPVHits,        "AvgAPVHits/D");
}

  
  HistoMap::iterator it=DM.begin();
  HistoMap::iterator itEnd=DM.end();
  std::vector<unsigned int> badStripList;
  uint32_t detid;
  for (;it!=itEnd;++it){
    pHisto phisto;
    detid=it->first;

    DetId detectorId=DetId(detid);
    phisto._SubdetId=detectorId.subdetId();
    
    if (edm::isDebugEnabled())
      LogTrace("SiStripHotStrip") << "Analyzing detid " << detid<< std::endl;
    
    int numberAPVs = (int)(it->second.get())->GetNbinsX()/128;

    // Set the values for the tree:

    detrawid = detid;
    subdetid = detectorId.subdetId();
    number_strips = (int)(it->second.get())->GetNbinsX();
    switch (detectorId.subdetId())
      {
      case StripSubdetector::TIB :
	layer_ring = tTopo->tibLayer(detrawid);
	disc       = -1;
	isstereo = tTopo->tibIsStereo(detrawid);
	isback     = -1;
	if (tTopo->tibIsExternalString(detrawid)) isexternalstring = 1;
	else                                       isexternalstring = 0;
	if (tTopo->tibIsZMinusSide(detrawid)) iszminusside = 1;
	else                                   iszminusside = 0;
	rodstringpetal  = tTopo->tibString(detrawid);
	module_position = tTopo->tibModule(detrawid);
	break;

      case StripSubdetector::TID :
	layer_ring = tTopo->tidRing(detrawid);
	disc       = tTopo->tidWheel(detrawid);
	isstereo = tTopo->tidIsStereo(detrawid);
	if (tTopo->tidIsBackRing(detrawid)) isback = 1;
	else                                 isback = 0;
	if (tTopo->tidIsZMinusSide(detrawid)) iszminusside = 1;
	else                                   iszminusside = 0;
	isexternalstring = -1;
	rodstringpetal   = -1;
	module_position  = tTopo->tidModule(detrawid);
	break;

      case StripSubdetector::TOB :
	layer_ring = tTopo->tobLayer(detrawid);
	disc       = -1;
	isstereo = tTopo->tobIsStereo(detrawid);
	isback     = -1;
	if (tTopo->tobIsZMinusSide(detrawid)) iszminusside = 1;
	else                                   iszminusside = 0;
	isexternalstring = -1;
	rodstringpetal   = tTopo->tobRod(detrawid);
	module_position  = tTopo->tobModule(detrawid);
	break;

      case StripSubdetector::TEC :
	layer_ring = tTopo->tecRing(detrawid);
	disc       = tTopo->tecWheel(detrawid);
	isstereo = tTopo->tecIsStereo(detrawid);
	if (tTopo->tecIsBackPetal(detrawid)) isback = 1;
	else                                  isback = 0;
	if (tTopo->tecIsZMinusSide(detrawid)) iszminusside = 1;
	else                                   iszminusside = 0;
	isexternalstring = -1;
	rodstringpetal   = tTopo->tecPetalNumber(detrawid);
	module_position  = tTopo->tecModule(detrawid);
	break;

      default :
	std::cout << "### Detector does not belong to TIB, TID, TOB or TEC !? ###" << std::endl;
	std::cout << "### DetRawId: " << detrawid << " ###" << std::endl;
      }

    // End: Set the values for the tree.


    pQuality=OutSiStripQuality;
    badStripList.clear();

    for (int i=0; i<768; i++){
      ishot[i]               = 0;
      stripoccupancy[i]      = 0;
      striphits[i]           = 0;
      poissonprob[i]         = 0;
      hotstripsperapv[i/128] = 0;
	}

    hotstripspermodule = 0;

    for (int apv=0; apv<numberAPVs; apv++){
      if(UseInputDB_){
	if(InSiStripQuality->IsApvBad(detid,apv) ){
	  if(edm::isDebugEnabled())
	    LogTrace("SiStripHotStrip")<<"(Module and Apv number) "<<detid<<" , "<<apv<<" excluded by input ESetup."<<std::endl;
	  continue;//if the apv is already flagged as bad, continue.
	}
	else{
	  if(edm::isDebugEnabled())
	    LogTrace("SiStripHotStrip")<<"(Module and Apv number) "<<detid<<" , "<<apv<<" good by input ESetup."<<std::endl;
	}
      }

      phisto._th1f = new TH1F("tmp","tmp",128,0.5,128.5);
      int NumberEntriesPerAPV=0;
      
      for (int strip=0; strip<128; strip++){
	phisto._th1f->SetBinContent(strip+1,(it->second.get())->GetBinContent((apv*128)+strip+1));
	NumberEntriesPerAPV += (int)(it->second.get())->GetBinContent((apv*128)+strip+1);
      }

      phisto._th1f->SetEntries(NumberEntriesPerAPV);
      phisto._NEntries=(int)phisto._th1f->GetEntries();
      phisto._NEmptyBins=0;

      LogTrace("SiStripHotStrip") << "Number of clusters in APV " << apv << ": " << NumberEntriesPerAPV << std::endl;

      iterativeSearch(phisto,badStripList,apv);

      delete phisto._th1f;
    }

    const StripGeomDetUnit*  theStripDet = dynamic_cast<const StripGeomDetUnit*>( (TkGeom->idToDet(detectorId)) );
    const StripTopology* theStripTopol   = dynamic_cast<const StripTopology*>( &(theStripDet->specificTopology()) );  

    for (int strip=0; strip<number_strips; strip++)
      {
	strip_number   = strip+1;
	apv_channel    = (strip%128)+1;
	isHot          = ishot[strip];
	stripOccupancy = stripoccupancy[strip];
	stripHits      = striphits[strip];
	poissonProb    = poissonprob[strip];
	medianAPVHits  = medianapvhits[strip/128];
	avgAPVHits     = avgapvhits[strip/128];

	hotStripsPerModule = hotstripspermodule;
	hotStripsPerAPV    = hotstripsperapv[strip/128];

	LocalPoint  pos_strip_local  = theStripTopol->localPosition(strip);
	GlobalPoint pos_strip_global = (TkGeom->idToDet(detectorId))->surface().toGlobal(pos_strip_local);

	global_position_x = pos_strip_global.x();
	global_position_y = pos_strip_global.y();
	global_position_z = pos_strip_global.z();

	if (WriteOutputFile_==true) striptree->Fill();
      }

    if (badStripList.begin()==badStripList.end())
      continue;

    OutSiStripQuality->compact(detid,badStripList);

    SiStripQuality::Range range(badStripList.begin(),badStripList.end());
    if ( ! OutSiStripQuality->put(detid,range) )
      edm::LogError("SiStripHotStrip")<<"[SiStripHotStripAlgorithmFromClusterOccupancy::extractBadStrips] detid already exists"<<std::endl;
  }
  OutSiStripQuality->fillBadComponents();

  if (WriteOutputFile_==true){
  f->cd();
  striptree->Write();
  f->Close();
  }

  LogTrace("SiStripHotStrip") << ss.str() << std::endl;
}

  
void SiStripHotStripAlgorithmFromClusterOccupancy::iterativeSearch(pHisto& histo,std::vector<unsigned int>& vect, int apv){
  if (!histo._NEntries || histo._NEntries <=MinNumEntries_ || histo._NEntries <= minNevents_)
    return;
  
  size_t startingSize=vect.size();
  long double diff=1.-prob_; 
  
  size_t Nbins     = histo._th1f->GetNbinsX();
  size_t ibinStart = 1; 
  size_t ibinStop  = Nbins+1; 
  int MaxEntry  = (int)histo._th1f->GetMaximum();

  std::vector<long double> vPoissonProbs(MaxEntry+1,0);
  long double meanVal=1.*histo._NEntries/(1.*Nbins-histo._NEmptyBins); 
  evaluatePoissonian(vPoissonProbs,meanVal);

  // Find median occupancy, taking into account only good strips
  unsigned int goodstripentries[128];
  int nGoodStrips = 0;
  for (size_t i=ibinStart; i<ibinStop; ++i){
    if (ishot[(apv*128)+i-1]==0){
      goodstripentries[nGoodStrips] = (unsigned int)histo._th1f->GetBinContent(i);
      nGoodStrips++;
    }
  }
  double median = TMath::Median(nGoodStrips,goodstripentries);

  for (size_t i=ibinStart; i<ibinStop; ++i){
    unsigned int entries= (unsigned int)histo._th1f->GetBinContent(i);

    if (ishot[(apv*128)+i-1]==0){
      stripoccupancy[(apv*128)+i-1] = entries/(double) Nevents_;
      striphits[(apv*128)+i-1]      = entries;
      poissonprob[(apv*128)+i-1]    = 1-vPoissonProbs[entries];
      medianapvhits[apv]  = median;
      avgapvhits[apv] = meanVal;
    }
    if (entries<=MinNumEntriesPerStrip_ || entries <= minNevents_ || entries / median < ratio_) continue;

    if(diff<vPoissonProbs[entries]){
      ishot[(apv*128)+i-1] = 1;
      hotstripspermodule++;
      hotstripsperapv[apv]++;
      histo._th1f->SetBinContent(i,0.);
      histo._NEntries-=entries;
      histo._NEmptyBins++;
      if (edm::isDebugEnabled())
	LogTrace("SiStripHotStrip")<< " rejecting strip " << (apv*128)+i-1 << " value " << entries << " diff  " << diff << " prob " << vPoissonProbs[entries]<< std::endl;
      vect.push_back(pQuality->encode((apv*128)+i-1,1,0));
    }

  }
  if (edm::isDebugEnabled())
    LogTrace("SiStripHotStrip") << " [SiStripHotStripAlgorithmFromClusterOccupancy::iterativeSearch] Nbins="<< Nbins << " MaxEntry="<<MaxEntry << " meanVal=" << meanVal << " NEmptyBins="<<histo._NEmptyBins<< " NEntries=" << histo._NEntries << " thEntries " << histo._th1f->GetEntries()<< " startingSize " << startingSize << " vector.size " << vect.size() << std::endl;

  if (vect.size()!=startingSize)
    iterativeSearch(histo,vect,apv);
}

void SiStripHotStripAlgorithmFromClusterOccupancy::evaluatePoissonian(std::vector<long double>& vPoissonProbs, long double& meanVal){
  for(size_t i=0;i<vPoissonProbs.size();++i){
    vPoissonProbs[i]= (i==0)?TMath::Poisson(i,meanVal):vPoissonProbs[i-1]+TMath::Poisson(i,meanVal);
  }
}

void SiStripHotStripAlgorithmFromClusterOccupancy::setNumberOfEvents(double Nevents){
  Nevents_=Nevents;
  minNevents_=occupancy_*Nevents_; 
  if (edm::isDebugEnabled())                                                                                                                                                                          
    LogTrace("SiStripHotStrip")<<" [SiStripHotStripAlgorithmFromClusterOccupancy::setNumberOfEvents] minNumber of Events per strip used to consider a strip bad" << minNevents_ << " for occupancy " << occupancy_ << std::endl;
}
