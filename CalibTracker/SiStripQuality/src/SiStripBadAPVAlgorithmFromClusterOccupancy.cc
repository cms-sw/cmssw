#include "CalibTracker/SiStripQuality/interface/SiStripBadAPVAlgorithmFromClusterOccupancy.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"


SiStripBadAPVAlgorithmFromClusterOccupancy::SiStripBadAPVAlgorithmFromClusterOccupancy(const edm::ParameterSet& iConfig, const TrackerTopology* theTopo):
  lowoccupancy_(0),
  highoccupancy_(100),
  absolutelow_(0),
  numberiterations_(2),
  Nevents_(0),
  occupancy_(0),
  OutFileName_("Occupancy.root"),
  UseInputDB_(iConfig.getUntrackedParameter<bool>("UseInputDB",false)),
  tTopo(theTopo)
  {
    minNevents_=Nevents_*occupancy_;
  }

SiStripBadAPVAlgorithmFromClusterOccupancy::~SiStripBadAPVAlgorithmFromClusterOccupancy(){
  LogTrace("SiStripBadAPVAlgorithmFromClusterOccupancy")<<"[SiStripBadAPVAlgorithmFromClusterOccupancy::~SiStripBadAPVAlgorithmFromClusterOccupancy] "<<std::endl;
}

void SiStripBadAPVAlgorithmFromClusterOccupancy::extractBadAPVs(SiStripQuality* siStripQuality,HistoMap& DM, edm::ESHandle<SiStripQuality>& inSiStripQuality){

  LogTrace("SiStripBadAPVAlgorithmFromClusterOccupancy")<<"[SiStripBadAPVAlgorithmFromClusterOccupancy::extractBadAPVs] "<<std::endl;

  if (WriteOutputFile_==true){
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
  apvtree->Branch("APVAbsoluteOccupancy",    &apvAbsoluteOccupancy,    "apvAbsoluteOccupancy/D");
  apvtree->Branch("APVMedianOccupancy",      &apvMedianOccupancy,      "apvMedianOccupancy/D");
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

	for (int strip=0; strip<128; strip++)
	  {
	    stripOccupancy[apv][strip] = 0;
	    stripWeight[apv][strip]    = 0;
	  }
      }

    pHisto phisto;
    phisto._th1f = it->second.get();
    phisto._NEntries = (int)phisto._th1f->GetEntries();
    phisto._NBins = phisto._th1f->GetNbinsX();

    number_strips  = (int)phisto._NBins;
    number_apvs    = number_strips/128;
    APV.numberApvs = number_apvs;

    for (int apv=0; apv<number_apvs; apv++)
      {
	for (int strip=0; strip<128; strip++)
	  {
	    stripOccupancy[apv][strip]     = phisto._th1f->GetBinContent((apv*128)+strip+1); // Remember: Bin=0 is underflow bin!
	    stripWeight[apv][strip]        = 1;
	    APV.apvabsoluteOccupancy[apv] += phisto._th1f->GetBinContent((apv*128)+strip+1); // Remember: Bin=0 is underflow bin!
	  }
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
	layer_ring = tTopo->tibLayer(detrawid);
	disc       = -1;
	isstereo   = tTopo->tibIsStereo(detrawid);
	isback     = -1;
	if (tTopo->tibIsExternalString(detrawid)) isexternalstring = 1;
	else                                       isexternalstring = 0;
	if (tTopo->tibIsZMinusSide(detrawid)) iszminusside = 1;
	else                                   iszminusside = 0;
	rodstringpetal     = tTopo->tibString(detrawid);
	module_number      = tTopo->tibModule(detrawid);
	APV.modulePosition = module_number;

	if      (layer_ring == 1) medianValues_TIB_Layer1.push_back(APV);
	else if (layer_ring == 2) medianValues_TIB_Layer2.push_back(APV);
	else if (layer_ring == 3) medianValues_TIB_Layer3.push_back(APV);
	else if (layer_ring == 4) medianValues_TIB_Layer4.push_back(APV);
	break;

      case StripSubdetector::TID :
	layer_ring = tTopo->tidRing(detrawid);
	disc       = tTopo->tidWheel(detrawid);
	isstereo   = tTopo->tidIsStereo(detrawid);
	if (tTopo->tidIsBackRing(detrawid)) isback = 1;
	else                                 isback = 0;
	if (tTopo->tidIsZMinusSide(detrawid)) iszminusside = 1;
	else                                   iszminusside = 0;
	isexternalstring   = -1;
	rodstringpetal     = -1;
	module_number      = tTopo->tidModule(detrawid);
	APV.modulePosition = layer_ring;

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
	layer_ring = tTopo->tobLayer(detrawid);
	disc       = -1;
	isstereo   = tTopo->tobIsStereo(detrawid);
	isback     = -1;
	if (tTopo->tobIsZMinusSide(detrawid)) iszminusside = 1;
	else                                   iszminusside = 0;
	isexternalstring   = -1;
	rodstringpetal     = tTopo->tobRod(detrawid);
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
	layer_ring = tTopo->tecRing(detrawid);
	disc       = tTopo->tecWheel(detrawid);
	isstereo   = tTopo->tecIsStereo(detrawid);
	if (tTopo->tecIsBackPetal(detrawid)) isback = 1;
	else                                  isback = 0;
	if (tTopo->tecIsZMinusSide(detrawid)) iszminusside = 1;
	else                                   iszminusside = 0;
	isexternalstring   = -1;
	rodstringpetal     = tTopo->tecPetalNumber(detrawid);
	module_number      = tTopo->tecModule(detrawid);
	APV.modulePosition = layer_ring;

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

    const StripGeomDetUnit*  theStripDet = dynamic_cast<const StripGeomDetUnit*>( (TkGeom->idToDet(detectorId)) );
    const StripTopology* theStripTopol   = dynamic_cast<const StripTopology*>( &(theStripDet->specificTopology()) );

    for (int apv=0; apv<number_apvs; apv++)
      {
	apv_number           = apv+1;
	apvMedianOccupancy   = APV.apvMedian[apv];
	apvAbsoluteOccupancy = APV.apvabsoluteOccupancy[apv];

	LocalPoint  pos_strip_local  = theStripTopol->localPosition((apv*128));
        GlobalPoint pos_strip_global = (TkGeom->idToDet(detectorId))->surface().toGlobal(pos_strip_local);

        global_position_x = pos_strip_global.x();
        global_position_y = pos_strip_global.y();
        global_position_z = pos_strip_global.z();

	if (WriteOutputFile_==true) apvtree->Fill();
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

  // Analyze the APV Occupancy for hot APVs
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
  f->Close();
  }

  LogTrace("SiStripBadAPV") << ss.str() << std::endl;
}


void SiStripBadAPVAlgorithmFromClusterOccupancy::CalculateMeanAndRMS(const std::vector<Apv>& a, std::pair<double,double>* MeanRMS, int number_iterations)
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

void SiStripBadAPVAlgorithmFromClusterOccupancy::AnalyzeOccupancy(SiStripQuality* quality, std::vector<Apv>& medianValues, std::pair<double,double>* MeanAndRms, std::vector<unsigned int>& BadStripList, edm::ESHandle<SiStripQuality>& InSiStripQuality)
{
  int Moduleposition;
  uint32_t Detid;

  for (uint32_t it=0; it<medianValues.size(); it++)
    {
      Moduleposition = (medianValues[it].modulePosition)-1;
      Detid          = medianValues[it].detrawId;

      for (int apv=0; apv<medianValues[it].numberApvs; apv++)
	{
	  if(UseInputDB_)
	    {
	      if(InSiStripQuality->IsApvBad(Detid,apv) )
		{
		  continue;//if the apv is already flagged as bad, continue.
		}
	    }
	  if (medianValues[it].apvMedian[apv] > minNevents_)
	    {
	      if ((medianValues[it].apvMedian[apv]>(MeanAndRms[Moduleposition].first+highoccupancy_*MeanAndRms[Moduleposition].second)) && (medianValues[it].apvMedian[apv]>absolutelow_))
		BadStripList.push_back(pQuality->encode((apv*128),128,0));
	    }
	  else if (medianValues[it].apvMedian[apv]<(MeanAndRms[Moduleposition].first-lowoccupancy_*MeanAndRms[Moduleposition].second) && (MeanAndRms[Moduleposition].first>2 || medianValues[it].apvabsoluteOccupancy[apv]==0))
	    {
	      BadStripList.push_back(pQuality->encode((apv*128),128,0));
	      std::cout << "Dead APV! DetId: " << medianValues[it].detrawId << ", APV number: " << apv+1 << ", APVMedian: " << medianValues[it].apvMedian[apv] << ", Mean: " << MeanAndRms[Moduleposition].first << ", RMS: " << MeanAndRms[Moduleposition].second << ", LowThreshold: " << lowoccupancy_ << ", Mean-Low*RMS: " << (MeanAndRms[Moduleposition].first-lowoccupancy_*MeanAndRms[Moduleposition].second) << std::endl;
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

void SiStripBadAPVAlgorithmFromClusterOccupancy::setMinNumOfEvents()
{
  minNevents_=occupancy_*Nevents_;
}
