#include "CalibTracker/SiStripQuality/interface/SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"


SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy(const edm::ParameterSet& iConfig):
  lowoccupancy_(0),
  highoccupancy_(100),
  absolutelow_(0),
  numberiterations_(2),
  Nevents_(0),
  absolute_occupancy_(0),
  OutFileName_("Occupancy.root"),
  UseInputDB_(iConfig.getUntrackedParameter<bool>("UseInputDB",false))
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
	layer_ring         = TIBDetId(detrawid).layer();
	module_number      = TIBDetId(detrawid).moduleNumber();
	APV.modulePosition = module_number;

	if      (layer_ring == 1) medianValues_TIB_Layer1.push_back(APV);
	else if (layer_ring == 2) medianValues_TIB_Layer2.push_back(APV);
	else if (layer_ring == 3) medianValues_TIB_Layer3.push_back(APV);
	else if (layer_ring == 4) medianValues_TIB_Layer4.push_back(APV);
	break;

      case StripSubdetector::TID :
	layer_ring         = TIDDetId(detrawid).ring();
	disc               = TIDDetId(detrawid).wheel();
	APV.modulePosition = layer_ring;

	if (TIDDetId(detrawid).isZMinusSide()) iszminusside = 1;
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
	layer_ring         = TOBDetId(detrawid).layer();
	module_number      = TOBDetId(detrawid).moduleNumber();
	APV.modulePosition = module_number;

	if      (layer_ring == 1) medianValues_TOB_Layer1.push_back(APV);
	else if (layer_ring == 2) medianValues_TOB_Layer2.push_back(APV);
	else if (layer_ring == 3) medianValues_TOB_Layer3.push_back(APV);
	else if (layer_ring == 4) medianValues_TOB_Layer4.push_back(APV);
	else if (layer_ring == 5) medianValues_TOB_Layer5.push_back(APV);
	else if (layer_ring == 6) medianValues_TOB_Layer6.push_back(APV);
	break;

      case StripSubdetector::TEC :
	layer_ring         = TECDetId(detrawid).ring();
	disc               = TECDetId(detrawid).wheel();
	APV.modulePosition = layer_ring;

	if (TECDetId(detrawid).isZMinusSide()) iszminusside = 1;
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

  if (WriteOutputFile_==true){
  f->cd();
  apvtree->Write();
  striptree->Write();
  f->Close();
  }

  LogTrace("SiStripBadAPV") << ss.str() << std::endl;
}


void SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy::CalculateMeanAndRMS(std::vector<Apv> a, std::pair<double,double>* MeanRMS, int number_iterations)
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

      for (int apv=0; apv<medianValues[it].numberApvs; apv++)
	{

	  for (int i=0; i<128; i++)
	    {
	      ishot[i]               = 0;
	      stripoccupancy[i]      = 0;
	      striphits[i]           = 0;
	      poissonprob[i]         = 0;
	    }

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
  
  int Nbins     = histo.th1f[apv]->GetNbinsX();
  int ibinStart = 1; 
  int ibinStop  = Nbins+1; 
  int MaxEntry  = (int)histo.th1f[apv]->GetMaximum();

  std::vector<long double> vPoissonProbs(MaxEntry+1,0);
  long double meanVal=1.*histo.NEntries[apv]/(1.*Nbins-histo.NEmptyBins[apv]); 
  evaluatePoissonian(vPoissonProbs,meanVal);

  for (Int_t i=ibinStart; i<ibinStop; ++i){
    unsigned int entries= (unsigned int)histo.th1f[apv]->GetBinContent(i);

    if (ishot[i-1]==0){
      stripoccupancy[i-1] = entries/(double) Nevents_;
      striphits[i-1]      = entries;
      poissonprob[i-1]    = 1-vPoissonProbs[entries];
    }

    if (entries<=MinNumEntriesPerStrip_ || entries <= minNevents_)
      continue;

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

  if (SiStripDetId(detid).stereo() !=0 ) isstereo = 1; // It's a stereo module
  else                                   isstereo = 0; // It's an rphi module
  switch (DetectorID.subdetId())
    {
    case StripSubdetector::TIB :
      layer_ring = TIBDetId(detid).layer();
      disc       = -1;
      isback     = -1;
      if (TIBDetId(detid).isExternalString()) isexternalstring = 1;
      else                                    isexternalstring = 0;
      if (TIBDetId(detid).isZMinusSide()) iszminusside = 1;
      else                                iszminusside = 0;
      rodstringpetal     = TIBDetId(detid).stringNumber();
      module_number      = TIBDetId(detid).moduleNumber();

      break;

    case StripSubdetector::TID :
      layer_ring = TIDDetId(detid).ring();
      disc       = TIDDetId(detid).wheel();
      if (TIDDetId(detid).isBackRing()) isback = 1;
      else                              isback = 0;
      if (TIDDetId(detid).isZMinusSide()) iszminusside = 1;
      else                                iszminusside = 0;
      isexternalstring   = -1;
      rodstringpetal     = -1;
      module_number      = TIDDetId(detid).moduleNumber();

      break;

    case StripSubdetector::TOB :
      layer_ring = TOBDetId(detid).layer();
      disc       = -1;
      isback     = -1;
      if (TOBDetId(detid).isZMinusSide()) iszminusside = 1;
      else                                iszminusside = 0;
      isexternalstring   = -1;
      rodstringpetal     = TOBDetId(detid).rodNumber();
      module_number      = TOBDetId(detid).moduleNumber();

      break;

    case StripSubdetector::TEC :
      layer_ring = TECDetId(detid).ring();
      disc       = TECDetId(detid).wheel();
      if (TECDetId(detid).isBackPetal()) isback = 1;
      else                               isback = 0;
      if (TECDetId(detid).isZMinusSide()) iszminusside = 1;
      else                                iszminusside = 0;
      isexternalstring   = -1;
      rodstringpetal     = TECDetId(detid).petalNumber();
      module_number      = TECDetId(detid).moduleNumber();

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
