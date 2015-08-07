#include "CalibTracker/SiStripQuality/interface/SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"


SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy(const edm::ParameterSet& iConfig, const TrackerTopology* theTopo):
  ratio_(1.5),
  lowoccupancy_(0),
  highoccupancy_(100),
  absolutelow_(0),
  numberiterations_(2),
  Nevents_(0),
  absolute_occupancy_(0),
  OutFileName_("Occupancy.root"),
  DQMOutfileName_("DQMOutput"),
  UseInputDB_(iConfig.getUntrackedParameter<bool>("UseInputDB",false)),
  tTopo(theTopo)
  {
    minNevents_=Nevents_*absolute_occupancy_;
  }

SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::~SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy(){
  LogTrace("SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy")<<"[SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::~SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy] "<<std::endl;
}

void SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::extractBadAPVSandStrips(SiStripQuality* siStripQuality,HistoMap& DM, edm::ESHandle<SiStripQuality>& inSiStripQuality){

  LogTrace("SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy")<<"[SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::extractBadAPVs] "<<std::endl;

  if (WriteOutputFile_== true)
    {
      f = new TFile(OutFileName_.c_str(),"RECREATE");
      f->cd();

      apvtree = new TTree("moduleOccupancy","tree");

      apvtree->Branch("DetRawId",                &detrawid,                "DetRawId/I");
      apvtree->Branch("SubDetId",                &subdetid,                "SubDetId/I");
      apvtree->Branch("Layer_Ring",              &layer_ring,              "Layer_Ring/I");
      apvtree->Branch("Disc",                    &disc,                    "Disc/I");
      apvtree->Branch("IsBack",                  &isback,                  "IsBack/I");
      apvtree->Branch("IsExternalString",        &isexternalstring,        "IsExternalString/I");
      apvtree->Branch("IsZMinusSide",            &iszminusside,            "IsZMinusSide/I");
      apvtree->Branch("RodStringPetal",          &rodstringpetal,          "RodStringPetal/I");
      apvtree->Branch("IsStereo",                &isstereo,                "IsStereo/I");
      apvtree->Branch("ModuleNumber",            &module_number,           "ModuleNumber/I");
      apvtree->Branch("NumberOfStrips",          &number_strips,           "NumberOfStrips/I");
      apvtree->Branch("APVGlobalPositionX",      &global_position_x,       "APVGlobalPositionX/F");
      apvtree->Branch("APVGlobalPositionY",      &global_position_y,       "APVGlobalPositionY/F");
      apvtree->Branch("APVGlobalPositionZ",      &global_position_z,       "APVGlobalPositionZ/F");
      apvtree->Branch("APVNumber",               &apv_number,              "APVNumber/I");
      apvtree->Branch("APVAbsoluteOccupancy",    &apvAbsoluteOccupancy,    "apvAbsoluteOccupancy/I");
      apvtree->Branch("APVMedianOccupancy",      &apvMedianOccupancy,      "apvMedianOccupancy/D");
      apvtree->Branch("IsBad",                   &isBad,                   "IsBad/I");

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
      striptree->Branch("ModulePosition",       &module_number,     "ModulePosition/I");
      striptree->Branch("NumberOfStrips",       &number_strips,     "NumberOfStrips/I");
      striptree->Branch("StripNumber",          &strip_number,      "StripNumber/I");
      striptree->Branch("APVChannel",           &apv_channel,       "APVChannel/I");
      striptree->Branch("StripGlobalPositionX", &strip_global_position_x, "StripGlobalPositionX/F");
      striptree->Branch("StripGlobalPositionY", &strip_global_position_y, "StripGlobalPositionY/F");
      striptree->Branch("StripGlobalPositionZ", &strip_global_position_z, "StripGlobalPositionZ/F");
      striptree->Branch("IsHot",                &isHot,             "IsHot/I");
      striptree->Branch("HotStripsPerAPV",      &hotStripsPerAPV,   "HotStripsPerAPV/I");
      striptree->Branch("HotStripsPerModule",   &hotStripsPerModule,"HotStripsPerModule/I");
      striptree->Branch("StripOccupancy",       &singleStripOccupancy, "StripOccupancy/D");
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

    Apv APV;

    for (int apv=0; apv<6; apv++)
      {
	APV.apvMedian[apv]            = 0;
	APV.apvabsoluteOccupancy[apv] = 0;
	APV.NEntries[apv]            = 0;
	APV.NEmptyBins[apv]          = 0;

	for (int strip=0; strip<128; strip++)
	  {
	    stripOccupancy[apv][strip] = 0;
	    stripWeight[apv][strip]    = 0;
	  }
      }

    number_strips  = (int)((it->second.get())->GetNbinsX());
    number_apvs    = number_strips/128;
    APV.numberApvs = number_apvs;

    for (int apv=0; apv<number_apvs; apv++)
      {
	APV.th1f[apv] = new TH1F("tmp","tmp",128,0.5,128.5);
	int NumberEntriesPerAPV=0;

	for (int strip=0; strip<128; strip++)
	  {
	    stripOccupancy[apv][strip] = (it->second.get())->GetBinContent((apv*128)+strip+1); // Remember: Bin=0 is underflow bin!
	    stripWeight[apv][strip]    = 1;
	    APV.apvabsoluteOccupancy[apv] += (it->second.get())->GetBinContent((apv*128)+strip+1); // Remember: Bin=0 is underflow bin!
	    APV.th1f[apv]->SetBinContent(strip+1,(it->second.get())->GetBinContent((apv*128)+strip+1));
	    NumberEntriesPerAPV += (int)(it->second.get())->GetBinContent((apv*128)+strip+1);
	  }

	APV.th1f[apv]->SetEntries(NumberEntriesPerAPV);
	APV.NEntries[apv]=(int)APV.th1f[apv]->GetEntries();
      }

    for (int apv=0; apv<number_apvs; apv++)
      {
	APV.apvMedian[apv] = TMath::Median(128,stripOccupancy[apv],stripWeight[apv]);
      }

    detid=it->first;
    DetId detectorId=DetId(detid);

    if (edm::isDebugEnabled())
      LogTrace("SiStripBadAPV") << "Analyzing detid " << detid<< std::endl;

    detrawid     = detid;
    APV.detrawId = detrawid;
    subdetid     = detectorId.subdetId();

    switch (detectorId.subdetId())
      {
      case StripSubdetector::TIB :
	layer_ring         = tTopo->tibLayer(detrawid);
	module_number      = tTopo->tibModule(detrawid);
	APV.modulePosition = module_number;

	if      (layer_ring == 1) medianValues_TIB_Layer1.push_back(APV);
	else if (layer_ring == 2) medianValues_TIB_Layer2.push_back(APV);
	else if (layer_ring == 3) medianValues_TIB_Layer3.push_back(APV);
	else if (layer_ring == 4) medianValues_TIB_Layer4.push_back(APV);
	break;

      case StripSubdetector::TID :
	layer_ring         = tTopo->tidRing(detrawid);
	disc               = tTopo->tidWheel(detrawid);
	APV.modulePosition = layer_ring;

	if (tTopo->tidIsZMinusSide(detrawid)) iszminusside = 1;
	else                                   iszminusside = 0;

	if (iszminusside==0)
	  {
	    if      (disc==1) medianValues_TIDPlus_Disc1.push_back(APV);
	    else if (disc==2) medianValues_TIDPlus_Disc2.push_back(APV);
	    else if (disc==3) medianValues_TIDPlus_Disc3.push_back(APV);
	  }
	else if (iszminusside==1)
	  {
	    if      (disc==1) medianValues_TIDMinus_Disc1.push_back(APV);
	    else if (disc==2) medianValues_TIDMinus_Disc2.push_back(APV);
	    else if (disc==3) medianValues_TIDMinus_Disc3.push_back(APV);
	  }
	break;

      case StripSubdetector::TOB :
	layer_ring         = tTopo->tobLayer(detrawid);
	module_number      = tTopo->tobModule(detrawid);
	APV.modulePosition = module_number;

	if      (layer_ring == 1) medianValues_TOB_Layer1.push_back(APV);
	else if (layer_ring == 2) medianValues_TOB_Layer2.push_back(APV);
	else if (layer_ring == 3) medianValues_TOB_Layer3.push_back(APV);
	else if (layer_ring == 4) medianValues_TOB_Layer4.push_back(APV);
	else if (layer_ring == 5) medianValues_TOB_Layer5.push_back(APV);
	else if (layer_ring == 6) medianValues_TOB_Layer6.push_back(APV);
	break;

      case StripSubdetector::TEC :
	layer_ring         = tTopo->tecRing(detrawid);
	disc               = tTopo->tecWheel(detrawid);
	APV.modulePosition = layer_ring;

	if (tTopo->tecIsZMinusSide(detrawid)) iszminusside = 1;
	else                                   iszminusside = 0;

	if (iszminusside==0)
	  {
	    if      (disc==1) medianValues_TECPlus_Disc1.push_back(APV);
	    else if (disc==2) medianValues_TECPlus_Disc2.push_back(APV);
	    else if (disc==3) medianValues_TECPlus_Disc3.push_back(APV);
	    else if (disc==4) medianValues_TECPlus_Disc4.push_back(APV);
	    else if (disc==5) medianValues_TECPlus_Disc5.push_back(APV);
	    else if (disc==6) medianValues_TECPlus_Disc6.push_back(APV);
	    else if (disc==7) medianValues_TECPlus_Disc7.push_back(APV);
	    else if (disc==8) medianValues_TECPlus_Disc8.push_back(APV);
	    else if (disc==9) medianValues_TECPlus_Disc9.push_back(APV);
	  }
	else if (iszminusside==1)
	  {
	    if      (disc==1) medianValues_TECMinus_Disc1.push_back(APV);
	    else if (disc==2) medianValues_TECMinus_Disc2.push_back(APV);
	    else if (disc==3) medianValues_TECMinus_Disc3.push_back(APV);
	    else if (disc==4) medianValues_TECMinus_Disc4.push_back(APV);
	    else if (disc==5) medianValues_TECMinus_Disc5.push_back(APV);
	    else if (disc==6) medianValues_TECMinus_Disc6.push_back(APV);
	    else if (disc==7) medianValues_TECMinus_Disc7.push_back(APV);
	    else if (disc==8) medianValues_TECMinus_Disc8.push_back(APV);
	    else if (disc==9) medianValues_TECMinus_Disc9.push_back(APV);
	  }
	break;

      default :
	std::cout << "### Detector does not belong to TIB, TID, TOB or TEC !? ###" << std::endl;
	std::cout << "### DetRawId: " << detrawid << " ###" << std::endl;
      }

  } // end loop on modules

  // Calculate Mean and RMS for each Layer
  CalculateMeanAndRMS(medianValues_TIB_Layer1,MeanAndRms_TIB_Layer1,numberiterations_);
  CalculateMeanAndRMS(medianValues_TIB_Layer2,MeanAndRms_TIB_Layer2,numberiterations_);
  CalculateMeanAndRMS(medianValues_TIB_Layer3,MeanAndRms_TIB_Layer3,numberiterations_);
  CalculateMeanAndRMS(medianValues_TIB_Layer4,MeanAndRms_TIB_Layer4,numberiterations_);

  CalculateMeanAndRMS(medianValues_TOB_Layer1,MeanAndRms_TOB_Layer1,numberiterations_);
  CalculateMeanAndRMS(medianValues_TOB_Layer2,MeanAndRms_TOB_Layer2,numberiterations_);
  CalculateMeanAndRMS(medianValues_TOB_Layer3,MeanAndRms_TOB_Layer3,numberiterations_);
  CalculateMeanAndRMS(medianValues_TOB_Layer4,MeanAndRms_TOB_Layer4,numberiterations_);
  CalculateMeanAndRMS(medianValues_TOB_Layer5,MeanAndRms_TOB_Layer5,numberiterations_);
  CalculateMeanAndRMS(medianValues_TOB_Layer6,MeanAndRms_TOB_Layer6,numberiterations_);

  CalculateMeanAndRMS(medianValues_TIDPlus_Disc1,MeanAndRms_TIDPlus_Disc1,numberiterations_);
  CalculateMeanAndRMS(medianValues_TIDPlus_Disc2,MeanAndRms_TIDPlus_Disc2,numberiterations_);
  CalculateMeanAndRMS(medianValues_TIDPlus_Disc3,MeanAndRms_TIDPlus_Disc3,numberiterations_);
  CalculateMeanAndRMS(medianValues_TIDMinus_Disc1,MeanAndRms_TIDMinus_Disc1,numberiterations_);
  CalculateMeanAndRMS(medianValues_TIDMinus_Disc2,MeanAndRms_TIDMinus_Disc2,numberiterations_);
  CalculateMeanAndRMS(medianValues_TIDMinus_Disc3,MeanAndRms_TIDMinus_Disc3,numberiterations_);

  CalculateMeanAndRMS(medianValues_TECPlus_Disc1,MeanAndRms_TECPlus_Disc1,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECPlus_Disc2,MeanAndRms_TECPlus_Disc2,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECPlus_Disc3,MeanAndRms_TECPlus_Disc3,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECPlus_Disc4,MeanAndRms_TECPlus_Disc4,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECPlus_Disc5,MeanAndRms_TECPlus_Disc5,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECPlus_Disc6,MeanAndRms_TECPlus_Disc6,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECPlus_Disc7,MeanAndRms_TECPlus_Disc7,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECPlus_Disc8,MeanAndRms_TECPlus_Disc8,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECPlus_Disc9,MeanAndRms_TECPlus_Disc9,numberiterations_);

  CalculateMeanAndRMS(medianValues_TECMinus_Disc1,MeanAndRms_TECMinus_Disc1,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECMinus_Disc2,MeanAndRms_TECMinus_Disc2,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECMinus_Disc3,MeanAndRms_TECMinus_Disc3,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECMinus_Disc4,MeanAndRms_TECMinus_Disc4,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECMinus_Disc5,MeanAndRms_TECMinus_Disc5,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECMinus_Disc6,MeanAndRms_TECMinus_Disc6,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECMinus_Disc7,MeanAndRms_TECMinus_Disc7,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECMinus_Disc8,MeanAndRms_TECMinus_Disc8,numberiterations_);
  CalculateMeanAndRMS(medianValues_TECMinus_Disc9,MeanAndRms_TECMinus_Disc9,numberiterations_);

  pQuality=siStripQuality;
  badStripList.clear();

  // Initialize the DQM output histograms
  initializeDQMHistograms();

  // Analyze the Occupancy for both APVs and Strips
  AnalyzeOccupancy(siStripQuality,medianValues_TIB_Layer1,MeanAndRms_TIB_Layer1,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TIB_Layer2,MeanAndRms_TIB_Layer2,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TIB_Layer3,MeanAndRms_TIB_Layer3,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TIB_Layer4,MeanAndRms_TIB_Layer4,badStripList,inSiStripQuality);

  AnalyzeOccupancy(siStripQuality,medianValues_TOB_Layer1,MeanAndRms_TOB_Layer1,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TOB_Layer2,MeanAndRms_TOB_Layer2,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TOB_Layer3,MeanAndRms_TOB_Layer3,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TOB_Layer4,MeanAndRms_TOB_Layer4,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TOB_Layer5,MeanAndRms_TOB_Layer5,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TOB_Layer6,MeanAndRms_TOB_Layer6,badStripList,inSiStripQuality);

  AnalyzeOccupancy(siStripQuality,medianValues_TIDPlus_Disc1,MeanAndRms_TIDPlus_Disc1,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TIDPlus_Disc2,MeanAndRms_TIDPlus_Disc2,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TIDPlus_Disc3,MeanAndRms_TIDPlus_Disc3,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TIDMinus_Disc1,MeanAndRms_TIDMinus_Disc1,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TIDMinus_Disc2,MeanAndRms_TIDMinus_Disc2,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TIDMinus_Disc3,MeanAndRms_TIDMinus_Disc3,badStripList,inSiStripQuality);

  AnalyzeOccupancy(siStripQuality,medianValues_TECPlus_Disc1,MeanAndRms_TECPlus_Disc1,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECPlus_Disc2,MeanAndRms_TECPlus_Disc2,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECPlus_Disc3,MeanAndRms_TECPlus_Disc3,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECPlus_Disc4,MeanAndRms_TECPlus_Disc4,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECPlus_Disc5,MeanAndRms_TECPlus_Disc5,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECPlus_Disc6,MeanAndRms_TECPlus_Disc6,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECPlus_Disc7,MeanAndRms_TECPlus_Disc7,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECPlus_Disc8,MeanAndRms_TECPlus_Disc8,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECPlus_Disc9,MeanAndRms_TECPlus_Disc9,badStripList,inSiStripQuality);

  AnalyzeOccupancy(siStripQuality,medianValues_TECMinus_Disc1,MeanAndRms_TECMinus_Disc1,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECMinus_Disc2,MeanAndRms_TECMinus_Disc2,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECMinus_Disc3,MeanAndRms_TECMinus_Disc3,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECMinus_Disc4,MeanAndRms_TECMinus_Disc4,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECMinus_Disc5,MeanAndRms_TECMinus_Disc5,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECMinus_Disc6,MeanAndRms_TECMinus_Disc6,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECMinus_Disc7,MeanAndRms_TECMinus_Disc7,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECMinus_Disc8,MeanAndRms_TECMinus_Disc8,badStripList,inSiStripQuality);
  AnalyzeOccupancy(siStripQuality,medianValues_TECMinus_Disc9,MeanAndRms_TECMinus_Disc9,badStripList,inSiStripQuality);

  siStripQuality->fillBadComponents();

  // Fill DQM histograms
  for(unsigned int i = 0; i < subDetName.size(); i++)
    {
      projYDistanceVsStripNumber[i]->Add((TH1F*)distanceVsStripNumber[i]->ProjectionY());
      pfxDistanceVsStripNumber[i]->Add(distanceVsStripNumber[i]->ProfileX(pfxDistanceVsStripNumber[i]->GetName(),1,998));
      projYNHitsVsStripNumber[i]->Add(nHitsVsStripNumber[i]->ProjectionY());
      projYNHitsGoodStripsVsStripNumber[i]->Add(nHitsGoodStripsVsStripNumber[i]->ProjectionY());
      projYNHitsHotStripsVsStripNumber[i]->Add(nHitsHotStripsVsStripNumber[i]->ProjectionY());
      projYOccupancyVsStripNumber[i]->Add(occupancyVsStripNumber[i]->ProjectionY());
      projYOccupancyGoodStripsVsStripNumber[i]->Add(occupancyGoodStripsVsStripNumber[i]->ProjectionY());
      projYOccupancyHotStripsVsStripNumber[i]->Add(occupancyHotStripsVsStripNumber[i]->ProjectionY());
      pfxOccupancyVsStripNumber[i]->Add(occupancyVsStripNumber[i]->ProfileX(pfxOccupancyVsStripNumber[i]->GetName(),-8.,0.));
      pfxOccupancyGoodStripsVsStripNumber[i]->Add(occupancyGoodStripsVsStripNumber[i]->ProfileX(pfxOccupancyGoodStripsVsStripNumber[i]->GetName(),-8.,0.));
      pfxOccupancyHotStripsVsStripNumber[i]->Add(occupancyHotStripsVsStripNumber[i]->ProfileX(pfxOccupancyHotStripsVsStripNumber[i]->GetName(),-8.,0.));
      projYPoissonProbVsStripNumber[i]->Add(poissonProbVsStripNumber[i]->ProjectionY());
      projYPoissonProbGoodStripsVsStripNumber[i]->Add(poissonProbGoodStripsVsStripNumber[i]->ProjectionY());
      projYPoissonProbHotStripsVsStripNumber[i]->Add(poissonProbHotStripsVsStripNumber[i]->ProjectionY());
      pfxPoissonProbVsStripNumber[i]->Add(poissonProbVsStripNumber[i]->ProfileX(pfxPoissonProbVsStripNumber[i]->GetName(),-18., 0.));
      pfxPoissonProbGoodStripsVsStripNumber[i]->Add(poissonProbGoodStripsVsStripNumber[i]->ProfileX(pfxPoissonProbGoodStripsVsStripNumber[i]->GetName(),-18., 0.));
      pfxPoissonProbHotStripsVsStripNumber[i]->Add(poissonProbHotStripsVsStripNumber[i]->ProfileX(pfxPoissonProbHotStripsVsStripNumber[i]->GetName(),-18., 0.));
      projXDistanceVsStripNumber[i]->Add(distanceVsStripNumber[i]->ProjectionX(projXDistanceVsStripNumber[i]->GetName(),1,998));

    }

  // Save output files

  if (WriteOutputFile_==true){
  f->cd();
  apvtree->Write();
  striptree->Write();
  f->Close();
  }

  if (WriteDQMHistograms_==true){
    dqmStore->cd();
    dqmStore->save(DQMOutfileName_.c_str());
  }

  LogTrace("SiStripBadAPV") << ss.str() << std::endl;
}


void SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::CalculateMeanAndRMS(const std::vector<Apv>& a, std::pair<double,double>* MeanRMS, int number_iterations)
{
  Double_t tot[7], tot2[7];
  Double_t n[7];

  Double_t Mean[7] = {0};
  Double_t Rms[7]  = {1000,1000,1000,1000,1000,1000,1000};

  int Moduleposition;

  for (int i=0; i<number_iterations; i++)
    {
      for (int j=0; j<7; j++)
	{
	  n[j]    = 0;
	  tot[j]  = 0;
	  tot2[j] = 0;
	}

      for (uint32_t it=0; it<a.size(); it++)
	{
	  Moduleposition = (a[it].modulePosition)-1;

	  for (int apv=0; apv<a[it].numberApvs; apv++)
	    {
	      if (i>0)
		{
		  if (a[it].apvMedian[apv]<(Mean[Moduleposition]-3*Rms[Moduleposition]) || (a[it].apvMedian[apv]>(Mean[Moduleposition]+5*Rms[Moduleposition])))
		    {
		      continue;
		    }
		}
	      tot[Moduleposition]  += a[it].apvMedian[apv];
	      tot2[Moduleposition] += (a[it].apvMedian[apv])*(a[it].apvMedian[apv]);
	      n[Moduleposition]++;
	    }
	}

      for (int j=0; j<7; j++)
	{
	  if (n[j]!=0)
	    {
	      Mean[j] = tot[j]/n[j];
	      Rms[j]  = TMath::Sqrt(TMath::Abs(tot2[j]/n[j] -Mean[j]*Mean[j]));
	    }
	}
    }

  for (int j=0; j<7; j++)
    {
      MeanRMS[j] = std::make_pair(Mean[j],Rms[j]);
    }

}

void SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::AnalyzeOccupancy(SiStripQuality* quality, std::vector<Apv>& medianValues, std::pair<double,double>* MeanAndRms, std::vector<unsigned int>& BadStripList, edm::ESHandle<SiStripQuality>& InSiStripQuality)
{
  int Moduleposition;
  uint32_t Detid;

  for (uint32_t it=0; it<medianValues.size(); it++)
    {
      Moduleposition = (medianValues[it].modulePosition)-1;
      Detid          = medianValues[it].detrawId;

      setBasicTreeParameters(Detid);

      DetId DetectorId=DetId(Detid);
      const StripGeomDetUnit*  TheStripDet   = dynamic_cast<const StripGeomDetUnit*>( (TkGeom->idToDet(DetectorId)) );
      const StripTopology*     TheStripTopol = dynamic_cast<const StripTopology*>( &(TheStripDet->specificTopology()) );

      //Analyze the occupancies
      hotstripspermodule = 0;
      vHotStripsInModule.clear();

      for (int apv=0; apv<medianValues[it].numberApvs; apv++)
	{
	  double logMedianOccupancy = -1;
	  double logAbsoluteOccupancy = -1;

	  for (int i=0; i<128; i++)
	    {
	      ishot[i]               = 0;
	      stripoccupancy[i]      = 0;
	      striphits[i]           = 0;
	      poissonprob[i]         = 0;
	    }

	  number_strips        = (medianValues[it].numberApvs)*128;
	  apv_number           = apv+1;
	  apvMedianOccupancy   = medianValues[it].apvMedian[apv];
	  apvAbsoluteOccupancy = medianValues[it].apvabsoluteOccupancy[apv];
	  isBad                = 0;
	  hotstripsperapv[apv] = 0;

	  LocalPoint  pos_apv_local  = TheStripTopol->localPosition((apv*128));
	  GlobalPoint pos_apv_global = (TkGeom->idToDet(DetectorId))->surface().toGlobal(pos_apv_local);

	  global_position_x = pos_apv_global.x();
	  global_position_y = pos_apv_global.y();
	  global_position_z = pos_apv_global.z();

	  if (apvMedianOccupancy>0) logMedianOccupancy = log10(apvMedianOccupancy);
	  if (apvAbsoluteOccupancy>0) logAbsoluteOccupancy = log10(apvAbsoluteOccupancy);

	  //Fill the DQM histograms
	  unsigned int layer = 0;
	  if(subdetid==3 || subdetid==5)
	    layer=layer_ring;
	  else
	    layer=disc;

	  // Fill histograms for all the tracker
	  medianVsAbsoluteOccupancy[0][0]->Fill(logAbsoluteOccupancy,logMedianOccupancy);
	  medianOccupancy[0][0]->Fill(logMedianOccupancy);
	  absoluteOccupancy[0][0]->Fill(logAbsoluteOccupancy);
	  // Fill summary histograms for each subdetector
	  medianVsAbsoluteOccupancy[subdetid-2][0]->Fill(logAbsoluteOccupancy,logMedianOccupancy);
	  medianOccupancy[subdetid-2][0]->Fill(logMedianOccupancy);
	  absoluteOccupancy[subdetid-2][0]->Fill(logAbsoluteOccupancy);
	  // Fill histograms for each layer/disk
	  medianVsAbsoluteOccupancy[subdetid-2][layer]->Fill(logAbsoluteOccupancy,logMedianOccupancy);
	  medianOccupancy[subdetid-2][layer]->Fill(logMedianOccupancy);
	  absoluteOccupancy[subdetid-2][layer]->Fill(logAbsoluteOccupancy);

	  if(UseInputDB_)
	    {
	      if(InSiStripQuality->IsApvBad(Detid,apv) )
		{
		  if (WriteOutputFile_==true)
		    {
		      apvtree->Fill();
		      for (int strip=0; strip<128; strip++)
			{
			  strip_number         = (apv*128)+strip+1;
			  apv_channel          = apv+1;
			  isHot                = ishot[strip];
			  singleStripOccupancy = stripoccupancy[strip];
			  stripHits            = striphits[strip];
			  poissonProb          = poissonprob[strip];

			  hotStripsPerModule = hotstripspermodule;
			  hotStripsPerAPV    = hotstripsperapv[apv];

			  LocalPoint  pos_strip_local  = TheStripTopol->localPosition(strip);
			  GlobalPoint pos_strip_global = (TkGeom->idToDet(DetectorId))->surface().toGlobal(pos_strip_local);

			  strip_global_position_x = pos_strip_global.x();
			  strip_global_position_y = pos_strip_global.y();
			  strip_global_position_z = pos_strip_global.z();
			  striptree->Fill();

			  // Fill the strip DQM Plots
			  fillStripDQMHistograms();
			}
     
		      if(vHotStripsInModule.size()==1)
			{
			  distance = 999;
			  distanceVsStripNumber[0]->Fill(vHotStripsInModule[0], distance);
			  distanceVsStripNumber[subdetid-2]->Fill(vHotStripsInModule[0], distance);
			}
		      else if(vHotStripsInModule.size()>1)
			{
			  for(unsigned int iVec = 0; iVec != vHotStripsInModule.size(); iVec++)
			    {
			      if(iVec==0)
				distance = vHotStripsInModule[1] - vHotStripsInModule[0];
			      else if(iVec==vHotStripsInModule.size()-1)
				{
				  distance = vHotStripsInModule[vHotStripsInModule.size()-1] - vHotStripsInModule[vHotStripsInModule.size() -2];
				}
			      else if(vHotStripsInModule.size()>2)
				{
				  distanceR = vHotStripsInModule[iVec + 1] -  vHotStripsInModule[iVec];
				  distanceL = vHotStripsInModule[iVec] - vHotStripsInModule[iVec - 1];
				  distance = distanceL>distanceR?distanceR:distanceL;
				}
			      else
				{
				  std::cout << "ERROR! distance is never computed!!!\n";
				}
			      distanceVsStripNumber[0]->Fill(vHotStripsInModule[iVec], distance);
			      distanceVsStripNumber[subdetid-2]->Fill(vHotStripsInModule[iVec], distance);
			    }
			}

		    }
		  continue;//if the apv is already flagged as bad, continue.
		}
	    }

	  if (medianValues[it].apvMedian[apv] > minNevents_)
	    {
	      if ((medianValues[it].apvMedian[apv]>(MeanAndRms[Moduleposition].first+highoccupancy_*MeanAndRms[Moduleposition].second)) && (medianValues[it].apvMedian[apv]>absolutelow_))
		{
		  BadStripList.push_back(pQuality->encode((apv*128),128,0));
		  isBad = 1;
		}
	    }
	  else if (medianValues[it].apvMedian[apv]<(MeanAndRms[Moduleposition].first-lowoccupancy_*MeanAndRms[Moduleposition].second) && (MeanAndRms[Moduleposition].first>2 || medianValues[it].apvabsoluteOccupancy[apv]==0))
	    {
	      BadStripList.push_back(pQuality->encode((apv*128),128,0));
	      isBad = 1;
	    }

	  if (isBad!=1)
	    {
	      iterativeSearch(medianValues[it],BadStripList,apv);
	    }

	  if (WriteOutputFile_==true)
	    {
	      apvtree->Fill();
	      for (int strip=0; strip<128; strip++)
		{
		  strip_number         = (apv*128)+strip+1;
		  apv_channel          = apv+1;
		  isHot                = ishot[strip];
		  singleStripOccupancy = stripoccupancy[strip];
		  stripHits            = striphits[strip];
		  poissonProb          = poissonprob[strip];

		  hotStripsPerModule = hotstripspermodule;
		  hotStripsPerAPV    = hotstripsperapv[apv];

		  LocalPoint  pos_strip_local  = TheStripTopol->localPosition(strip);
		  GlobalPoint pos_strip_global = (TkGeom->idToDet(DetectorId))->surface().toGlobal(pos_strip_local);

		  strip_global_position_x = pos_strip_global.x();
		  strip_global_position_y = pos_strip_global.y();
		  strip_global_position_z = pos_strip_global.z();
		  striptree->Fill();

		  // Fill the strip DQM Plots
		  fillStripDQMHistograms();
		}
	      if(vHotStripsInModule.size()==1)
		{
		  distance = 999;
		  distanceVsStripNumber[0]->Fill(vHotStripsInModule[0], distance);
		  distanceVsStripNumber[subdetid-2]->Fill(vHotStripsInModule[0], distance);
		}
	      else if(vHotStripsInModule.size()>1)
		{
		  for(unsigned int iVec = 0; iVec != vHotStripsInModule.size(); iVec++)
		    {
		      if(iVec==0)
			distance = vHotStripsInModule[1] - vHotStripsInModule[0];
		      else if(iVec==vHotStripsInModule.size()-1)
			{
			  distance = vHotStripsInModule[vHotStripsInModule.size()-1] - vHotStripsInModule[vHotStripsInModule.size() -2];
			}
		      else if(vHotStripsInModule.size()>2)
			{
			  distanceR = vHotStripsInModule[iVec + 1] -  vHotStripsInModule[iVec];
			  distanceL = vHotStripsInModule[iVec] - vHotStripsInModule[iVec - 1];
			  distance = distanceL>distanceR?distanceR:distanceL;
			}
		      else
			{
			  std::cout << "ERROR! distance is never computed!!!\n";
			}
		      distanceVsStripNumber[0]->Fill(vHotStripsInModule[iVec], distance);
		      distanceVsStripNumber[subdetid-2]->Fill(vHotStripsInModule[iVec], distance);
		    }
		}
	    }
	}

      if (BadStripList.begin()!=BadStripList.end())
	{
	  quality->compact(Detid,BadStripList);
	  SiStripQuality::Range range(BadStripList.begin(),BadStripList.end());
	  quality->put(Detid,range);
	}
      BadStripList.clear();
    }
}

void SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::iterativeSearch(Apv& histo,std::vector<unsigned int>& vect, int apv){
  if (!histo.NEntries[apv] || histo.NEntries[apv] <=MinNumEntries_ || histo.NEntries[apv] <= minNevents_)
    return;
  
  size_t startingSize=vect.size();
  long double diff=1.-prob_; 
  
  size_t Nbins     = histo.th1f[apv]->GetNbinsX();
  size_t ibinStart = 1; 
  size_t ibinStop  = Nbins+1; 
  int MaxEntry  = (int)histo.th1f[apv]->GetMaximum();

  std::vector<long double> vPoissonProbs(MaxEntry+1,0);
  long double meanVal=1.*histo.NEntries[apv]/(1.*Nbins-histo.NEmptyBins[apv]); 
  evaluatePoissonian(vPoissonProbs,meanVal);

  // Find median occupancy, taking into account only good strips
  unsigned int goodstripentries[128];
  int nGoodStrips = 0;
  for (size_t i=ibinStart; i<ibinStop; ++i){
    if (ishot[(apv*128)+i-1]==0){
      goodstripentries[nGoodStrips] = (unsigned int)histo.th1f[apv]->GetBinContent(i);
      nGoodStrips++;
    }
  }
  double median = TMath::Median(nGoodStrips,goodstripentries);

  for (size_t i=ibinStart; i<ibinStop; ++i){
    unsigned int entries= (unsigned int)histo.th1f[apv]->GetBinContent(i);

    if (ishot[i-1]==0){
      stripoccupancy[i-1] = entries/(double) Nevents_;
      striphits[i-1]      = entries;
      poissonprob[i-1]    = 1-vPoissonProbs[entries];
      medianapvhits[apv]  = median;
      avgapvhits[apv] = meanVal;
    }

    if (entries<=MinNumEntriesPerStrip_ || entries <= minNevents_ || entries / median < ratio_) continue;

    if(diff<vPoissonProbs[entries]){
      ishot[i-1] = 1;
      hotstripspermodule++;
      hotstripsperapv[apv]++;
      histo.th1f[apv]->SetBinContent(i,0.);
      histo.NEntries[apv]-=entries;
      histo.NEmptyBins[apv]++;
      if (edm::isDebugEnabled())
	LogTrace("SiStripHotStrip")<< " rejecting strip " << (apv*128)+i-1 << " value " << entries << " diff  " << diff << " prob " << vPoissonProbs[entries]<< std::endl;
      vect.push_back(pQuality->encode((apv*128)+i-1,1,0));
    }

  }
  if (edm::isDebugEnabled())
    LogTrace("SiStripHotStrip") << " [SiStripHotStripAlgorithmFromClusterOccupancy::iterativeSearch] Nbins="<< Nbins << " MaxEntry="<<MaxEntry << " meanVal=" << meanVal << " NEmptyBins="<<histo.NEmptyBins[apv]<< " NEntries=" << histo.NEntries[apv] << " thEntries " << histo.th1f[apv]->GetEntries()<< " startingSize " << startingSize << " vector.size " << vect.size() << std::endl;

  if (vect.size()!=startingSize)
    iterativeSearch(histo,vect,apv);
}

void SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::evaluatePoissonian(std::vector<long double>& vPoissonProbs, long double& meanVal){
  for(size_t i=0;i<vPoissonProbs.size();++i){
    vPoissonProbs[i]= (i==0)?TMath::Poisson(i,meanVal):vPoissonProbs[i-1]+TMath::Poisson(i,meanVal);
  }
}

void SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::setBasicTreeParameters(int detid){
  DetId DetectorID=DetId(detid);

  detrawid = detid;
  subdetid = DetectorID.subdetId();

  switch (DetectorID.subdetId())
    {
    case StripSubdetector::TIB :
      layer_ring = tTopo->tibLayer(detid);
      disc       = -1;
      isstereo   = tTopo->tibIsStereo(detid);
      isback     = -1;
      if (tTopo->tibIsExternalString(detid)) isexternalstring = 1;
      else                                    isexternalstring = 0;
      if (tTopo->tibIsZMinusSide(detid)) iszminusside = 1;
      else                                iszminusside = 0;
      rodstringpetal     = tTopo->tibString(detid);
      module_number      = tTopo->tibModule(detid);

      break;

    case StripSubdetector::TID :
      layer_ring = tTopo->tidRing(detid);
      disc       = tTopo->tidWheel(detid);
      isstereo   = tTopo->tidIsStereo(detid);
      if (tTopo->tidIsBackRing(detid)) isback = 1;
      else                              isback = 0;
      if (tTopo->tidIsZMinusSide(detid)) iszminusside = 1;
      else                                iszminusside = 0;
      isexternalstring   = -1;
      rodstringpetal     = -1;
      module_number      = tTopo->tidModule(detid);

      break;

    case StripSubdetector::TOB :
      layer_ring = tTopo->tobLayer(detid);
      disc       = -1;
      isstereo   = tTopo->tobIsStereo(detid);
      isback     = -1;
      if (tTopo->tobIsZMinusSide(detid)) iszminusside = 1;
      else                                iszminusside = 0;
      isexternalstring   = -1;
      rodstringpetal     = tTopo->tobRod(detid);
      module_number      = tTopo->tobModule(detid);

      break;

    case StripSubdetector::TEC :
      layer_ring = tTopo->tecRing(detid);
      disc       = tTopo->tecWheel(detid);
      isstereo   = tTopo->tecIsStereo(detid);
      if (tTopo->tecIsBackPetal(detid)) isback = 1;
      else                               isback = 0;
      if (tTopo->tecIsZMinusSide(detid)) iszminusside = 1;
      else                                iszminusside = 0;
      isexternalstring   = -1;
      rodstringpetal     = tTopo->tecPetalNumber(detid);
      module_number      = tTopo->tecModule(detid);

      break;

    default :
      std::cout << "### Detector does not belong to TIB, TID, TOB or TEC !? ###" << std::endl;
      std::cout << "### DetRawId: " << detid << " ###" << std::endl;
    }
}

void SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::setMinNumOfEvents()
{
  minNevents_=absolute_occupancy_*Nevents_;
}

void SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::initializeDQMHistograms()
{
  oss.str("");
  oss << 1; //runNumber

  dqmStore = edm::Service<DQMStore>().operator->();
  dqmStore->setCurrentFolder("ChannelStatusPlots");

  // Initialize histograms
  subDetName.push_back(""); subDetName.push_back("TIB"); subDetName.push_back("TID"); subDetName.push_back("TOB"); subDetName.push_back("TEC");
  nLayers.push_back(0); nLayers.push_back(4); nLayers.push_back(3); nLayers.push_back(6); nLayers.push_back(9);
  layerName.push_back(""); layerName.push_back("Layer"); layerName.push_back("Disk"); layerName.push_back("Layer"); layerName.push_back("Disk");
  
  std::string histoName;
  std::string histoTitle;

  for(unsigned int i = 0; i < subDetName.size(); i++)
  {
    histoName = "distanceVsStripNumber" + subDetName[i];
    histoTitle = "Distance between hot strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 999, 0.5, 999.5);
    distanceVsStripNumber.push_back(tmp->getTH2F());

    histoName = "pfxDistanceVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore->bookProfile(histoName.c_str(), tmp_prof);
    pfxDistanceVsStripNumber.push_back(tmp->getTProfile());
    pfxDistanceVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxDistanceVsStripNumber[i]->GetYaxis()->SetTitle("Distance");

    histoName = "projXDistanceVsStripNumber" + subDetName[i];
    histoTitle = "Number of hot strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    projXDistanceVsStripNumber.push_back(tmp->getTH1F());
    projXDistanceVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    projXDistanceVsStripNumber[i]->GetYaxis()->SetTitle("N_{hot}");
    
    histoName = "projYDistanceVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of distance between hot strips";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 999, 0.5, 999.5);
    projYDistanceVsStripNumber.push_back(tmp->getTH1F());
    projYDistanceVsStripNumber[i]->GetXaxis()->SetTitle("Distance");
    projYDistanceVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "occupancyVsStripNumber" + subDetName[i];
    histoTitle = "Occupancy of strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 1000, -8.,0.);
    occupancyVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxOccupancyVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore->bookProfile(histoName.c_str(), tmp_prof);
    pfxOccupancyVsStripNumber.push_back(tmp->getTProfile());
    pfxOccupancyVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxOccupancyVsStripNumber[i]->GetYaxis()->SetTitle("log_{10}(Occupancy)");
    
    histoName = "projYOccupancyVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of strip occupancy";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -8., 0.);
    projYOccupancyVsStripNumber.push_back(tmp->getTH1F());
    projYOccupancyVsStripNumber[i]->GetXaxis()->SetTitle("log_{10}(Occupancy)");
    projYOccupancyVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "occupancyHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "Occupancy of hot strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 1000, -8., 0.);
    occupancyHotStripsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxOccupancyHotStripsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore->bookProfile(histoName.c_str(), tmp_prof);
    pfxOccupancyHotStripsVsStripNumber.push_back(tmp->getTProfile());
    pfxOccupancyHotStripsVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxOccupancyHotStripsVsStripNumber[i]->GetYaxis()->SetTitle("log_{10}(Occupancy)");
    
    histoName = "projYOccupancyHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of hot strip occupancy";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -8., 0.);
    projYOccupancyHotStripsVsStripNumber.push_back(tmp->getTH1F());
    projYOccupancyHotStripsVsStripNumber[i]->GetXaxis()->SetTitle("log_{10}(Occupancy)");
    projYOccupancyHotStripsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "occupancyGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "Occupancy of good strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 1000, -8., 0.);
    occupancyGoodStripsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxOccupancyGoodStripsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore->bookProfile(histoName.c_str(), tmp_prof);
    pfxOccupancyGoodStripsVsStripNumber.push_back(tmp->getTProfile());
    pfxOccupancyGoodStripsVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxOccupancyGoodStripsVsStripNumber[i]->GetYaxis()->SetTitle("log_{10}(Occupancy)");
    
    histoName = "projYOccupancyGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of good strip occupancy";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -8., 0.);
    projYOccupancyGoodStripsVsStripNumber.push_back(tmp->getTH1F());
    projYOccupancyGoodStripsVsStripNumber[i]->GetXaxis()->SetTitle("log_{10}(Occupancy)");
    projYOccupancyGoodStripsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "poissonProbVsStripNumber" + subDetName[i];
    histoTitle = "Poisson probability of strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 1000, -18., 0.);
    poissonProbVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxPoissonProbVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore->bookProfile(histoName.c_str(), tmp_prof);
    pfxPoissonProbVsStripNumber.push_back(tmp->getTProfile());
    pfxPoissonProbVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxPoissonProbVsStripNumber[i]->GetYaxis()->SetTitle("log_{10}(Probability)");
    
    histoName = "projYPoissonProbVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of strip Poisson probability";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -18., 0.);
    projYPoissonProbVsStripNumber.push_back(tmp->getTH1F());
    projYPoissonProbVsStripNumber[i]->GetXaxis()->SetTitle("log_{10}(Probability)");
    projYPoissonProbVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "poissonProbHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "Poisson probability of hot strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 1000, -18., 0.);
    poissonProbHotStripsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxPoissonProbHotStripsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore->bookProfile(histoName.c_str(), tmp_prof);
    pfxPoissonProbHotStripsVsStripNumber.push_back(tmp->getTProfile());
    pfxPoissonProbHotStripsVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxPoissonProbHotStripsVsStripNumber[i]->GetYaxis()->SetTitle("log_{10}(Probability)");
    
    histoName = "projYPoissonProbHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of hot strip Poisson probability";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -18., 0.);
    projYPoissonProbHotStripsVsStripNumber.push_back(tmp->getTH1F());
    projYPoissonProbHotStripsVsStripNumber[i]->GetXaxis()->SetTitle("log_{10}(Probability)");
    projYPoissonProbHotStripsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "poissonProbGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "Poisson probability of good strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 1000, -18., 0.);
    poissonProbGoodStripsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxPoissonProbGoodStripsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore->bookProfile(histoName.c_str(), tmp_prof);
    pfxPoissonProbGoodStripsVsStripNumber.push_back(tmp->getTProfile());
    pfxPoissonProbGoodStripsVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxPoissonProbGoodStripsVsStripNumber[i]->GetYaxis()->SetTitle("log_{10}(Probability)");
    
    histoName = "projYPoissonProbGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of good strip Poisson probability";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -18., 0.);
    projYPoissonProbGoodStripsVsStripNumber.push_back(tmp->getTH1F());
    projYPoissonProbGoodStripsVsStripNumber[i]->GetXaxis()->SetTitle("log_{10}(Probability)");
    projYPoissonProbGoodStripsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");

    //
    histoName = "nHitsVsStripNumber" + subDetName[i];
    histoTitle = "NHits in strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 10000, -0.5, 9999.5);
    nHitsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxNHitsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore->bookProfile(histoName.c_str(), tmp_prof);
    pfxNHitsVsStripNumber.push_back(tmp->getTProfile());
    
    histoName = "projXNHitsVsStripNumber" + subDetName[i];
    histoTitle = "Cumulative nHits in strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    projXNHitsVsStripNumber.push_back(tmp->getTH1F());
    
    histoName = "projYNHitsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of nHits for all strips";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 10000, -0.5, 9999.5);
    projYNHitsVsStripNumber.push_back(tmp->getTH1F());
    projYNHitsVsStripNumber[i]->GetXaxis()->SetTitle("N_{hits}");
    projYNHitsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "nHitsHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "NHits in hot strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 10000, -0.5, 9999.5);
    nHitsHotStripsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxNHitsHotStripsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore->bookProfile(histoName.c_str(), tmp_prof);
    pfxNHitsHotStripsVsStripNumber.push_back(tmp->getTProfile());
    
    histoName = "projXNHitsHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "Cumulative nHits in hot strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    projXNHitsHotStripsVsStripNumber.push_back(tmp->getTH1F());
    
    histoName = "projYNHitsHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of nHits for hot strips";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 10000, -0.5, 9999.5);
    projYNHitsHotStripsVsStripNumber.push_back(tmp->getTH1F());
    projYNHitsHotStripsVsStripNumber[i]->GetXaxis()->SetTitle("N_{hits}");
    projYNHitsHotStripsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "nHitsGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "NHits in good strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 10000, -0.5, 9999.5);
    nHitsGoodStripsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxNHitsGoodStripsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore->bookProfile(histoName.c_str(), tmp_prof);
    pfxNHitsGoodStripsVsStripNumber.push_back(tmp->getTProfile());
    
    histoName = "projXNHitsGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "Cumulative nHits in good strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    projXNHitsGoodStripsVsStripNumber.push_back(tmp->getTH1F());
    
    histoName = "projYNHitsGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of nHits for good strips";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 10000, -0.5, 9999.5);
    projYNHitsGoodStripsVsStripNumber.push_back(tmp->getTH1F());
    projYNHitsGoodStripsVsStripNumber[i]->GetXaxis()->SetTitle("N_{hits}");
    projYNHitsGoodStripsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");

    for(unsigned int j = 0; j <= nLayers[i]; j++)
    {
      histoName = "medianVsAbsoluteOccupancy" + subDetName[i];
      if(j!=0)
      {
        oss.str("");
        oss << j;
        histoName += layerName[i] + oss.str();
      }
      histoTitle = "Median APV occupancy vs. absolute APV occupancy";
      if(i!=0)
        histoTitle += " in " + subDetName[i];
      if(j!=0)
      {
        histoTitle += " " + layerName[i] + " " + oss.str();
      }
      tmp = dqmStore->book2D(histoName.c_str(), histoTitle.c_str(), 1000, 0., 6., 1000, -1., 3.);
      medianVsAbsoluteOccupancy[i][j] = tmp->getTH2F();
      medianVsAbsoluteOccupancy[i][j]->Rebin2D(10,10);
      medianVsAbsoluteOccupancy[i][j]->GetXaxis()->SetTitle("log_{10}(Abs. Occupancy)");
      medianVsAbsoluteOccupancy[i][j]->GetYaxis()->SetTitle("log_{10}(Median Occupancy)");
      //
      histoName = "medianOccupancy" + subDetName[i];
      if(j!=0)
      {
        oss.str("");
        oss << j;
        histoName += layerName[i] + oss.str();
      }
      histoTitle = "Median APV occupancy";
      if(i!=0)
        histoTitle += " in " + subDetName[i];
      if(j!=0)
      {
        histoTitle += " " + layerName[i] + " " + oss.str();
      }
      tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -1., 3.);
      medianOccupancy[i][j] = tmp->getTH1F();
      medianOccupancy[i][j]->GetXaxis()->SetTitle("log_{10}(Occupancy)");
      medianOccupancy[i][j]->GetYaxis()->SetTitle("APVs");
      //
      histoName = "absoluteOccupancy" + subDetName[i];
      if(j!=0)
      {
        oss.str("");
        oss << j;
        histoName += layerName[i] + oss.str();
      }
      histoTitle = "Absolute APV occupancy";
      if(i!=0)
        histoTitle += " in " + subDetName[i];
      if(j!=0)
      {
        histoTitle += " " + layerName[i] + " " + oss.str();
      }
      tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 1000, 0., 6.);
      absoluteOccupancy[i][j] = tmp->getTH1F();
      absoluteOccupancy[i][j]->GetXaxis()->SetTitle("log_{10}(Occupancy)");
      absoluteOccupancy[i][j]->GetYaxis()->SetTitle("APVs");
    }
  }
}

void SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::fillStripDQMHistograms()
{
  double logStripOccupancy = -1;
  double logPoissonProb = -1;

  if (singleStripOccupancy>0) logStripOccupancy = log10(singleStripOccupancy);
  if (poissonProb>0) logPoissonProb = log10(fabs(poissonProb));

  occupancyVsStripNumber[0]->Fill(strip_number,logStripOccupancy);
  occupancyVsStripNumber[subdetid-2]->Fill(strip_number,logStripOccupancy);
  poissonProbVsStripNumber[0]->Fill(strip_number,logPoissonProb);
  poissonProbVsStripNumber[subdetid-2]->Fill(strip_number,logPoissonProb);
  nHitsVsStripNumber[0]->Fill(strip_number,stripHits);
  nHitsVsStripNumber[subdetid-2]->Fill(strip_number,stripHits);
       
  if(isHot)
    {
      vHotStripsInModule.push_back(strip_number);
      occupancyHotStripsVsStripNumber[0]->Fill(strip_number,logStripOccupancy);
      occupancyHotStripsVsStripNumber[subdetid-2]->Fill(strip_number,logStripOccupancy);
      poissonProbHotStripsVsStripNumber[0]->Fill(strip_number,logPoissonProb);
      poissonProbHotStripsVsStripNumber[subdetid-2]->Fill(strip_number,logPoissonProb);
      nHitsHotStripsVsStripNumber[0]->Fill(strip_number,stripHits);
      nHitsHotStripsVsStripNumber[subdetid-2]->Fill(strip_number,stripHits);
    }
  else
    {
      occupancyGoodStripsVsStripNumber[0]->Fill(strip_number,logStripOccupancy);
      occupancyGoodStripsVsStripNumber[subdetid-2]->Fill(strip_number,logStripOccupancy);
      poissonProbGoodStripsVsStripNumber[0]->Fill(strip_number,logPoissonProb);
      poissonProbGoodStripsVsStripNumber[subdetid-2]->Fill(strip_number,logPoissonProb);
      nHitsGoodStripsVsStripNumber[0]->Fill(strip_number,stripHits);
      nHitsGoodStripsVsStripNumber[subdetid-2]->Fill(strip_number,stripHits);
    }
}
