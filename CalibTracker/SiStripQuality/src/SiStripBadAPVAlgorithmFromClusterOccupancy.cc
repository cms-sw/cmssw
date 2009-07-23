#include "CalibTracker/SiStripQuality/interface/SiStripBadAPVAlgorithmFromClusterOccupancy.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"


SiStripBadAPVAlgorithmFromClusterOccupancy::~SiStripBadAPVAlgorithmFromClusterOccupancy(){
  LogTrace("SiStripBadAPVAlgorithmFromClusterOccupancy")<<"[SiStripBadAPVAlgorithmFromClusterOccupancy::~SiStripBadAPVAlgorithmFromClusterOccupancy] "<<std::endl;
}

void SiStripBadAPVAlgorithmFromClusterOccupancy::extractBadAPVs(SiStripQuality* siStripQuality,HistoMap& DM){

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
  apvtree->Branch("ModulePosition",          &module_position,         "ModulePosition/I");
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
	APV.apvMedian[apv]        = 0;
	apvabsoluteOccupancy[apv] = 0;

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
	    stripOccupancy[apv][strip] = phisto._th1f->GetBinContent((apv*128)+strip+1); // Remember: Bin=0 is underflow bin!
	    stripWeight[apv][strip]    = 1;
	    apvabsoluteOccupancy[apv] += phisto._th1f->GetBinContent((apv*128)+strip+1); // Remember: Bin=0 is underflow bin!
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
    if (SiStripDetId(detrawid).stereo() !=0 ) isstereo = 1; // It's a stereo module
    else                                      isstereo = 0; // It's an rphi module
    switch (detectorId.subdetId())
      {
      case StripSubdetector::TIB :
	layer_ring = TIBDetId(detrawid).layer();
	disc       = -1;
	isback     = -1;
	if (TIBDetId(detrawid).isExternalString()) isexternalstring = 1;
	else                                       isexternalstring = 0;
	if (TIBDetId(detrawid).isZMinusSide()) iszminusside = 1;
	else                                   iszminusside = 0;
	rodstringpetal  = TIBDetId(detrawid).stringNumber();
	module_position = TIBDetId(detrawid).moduleNumber();

	if      (layer_ring == 1) medianValues_TIB_Layer1.push_back(APV);
	else if (layer_ring == 2) medianValues_TIB_Layer2.push_back(APV);
	else if (layer_ring == 3) medianValues_TIB_Layer3.push_back(APV);
	else if (layer_ring == 4) medianValues_TIB_Layer4.push_back(APV);
	break;

      case StripSubdetector::TID :
	layer_ring = TIDDetId(detrawid).ring();
	disc       = TIDDetId(detrawid).wheel();
	if (TIDDetId(detrawid).isBackRing()) isback = 1;
	else                                 isback = 0;
	if (TIDDetId(detrawid).isZMinusSide()) iszminusside = 1;
	else                                   iszminusside = 0;
	isexternalstring = -1;
	rodstringpetal   = -1;
	module_position  = TIDDetId(detrawid).moduleNumber();

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
	layer_ring = TOBDetId(detrawid).layer();
	disc       = -1;
	isback     = -1;
	if (TOBDetId(detrawid).isZMinusSide()) iszminusside = 1;
	else                                   iszminusside = 0;
	isexternalstring = -1;
	rodstringpetal   = TOBDetId(detrawid).rodNumber();
	module_position  = TOBDetId(detrawid).moduleNumber();

	if      (layer_ring == 1) medianValues_TOB_Layer1.push_back(APV);
	else if (layer_ring == 2) medianValues_TOB_Layer2.push_back(APV);
	else if (layer_ring == 3) medianValues_TOB_Layer3.push_back(APV);
	else if (layer_ring == 4) medianValues_TOB_Layer4.push_back(APV);
	else if (layer_ring == 5) medianValues_TOB_Layer5.push_back(APV);
	else if (layer_ring == 6) medianValues_TOB_Layer6.push_back(APV);
	break;

      case StripSubdetector::TEC :
	layer_ring = TECDetId(detrawid).ring();
	disc       = TECDetId(detrawid).wheel();
	if (TECDetId(detrawid).isBackPetal()) isback = 1;
	else                                  isback = 0;
	if (TECDetId(detrawid).isZMinusSide()) iszminusside = 1;
	else                                   iszminusside = 0;
	isexternalstring = -1;
	rodstringpetal   = TECDetId(detrawid).petalNumber();
	module_position  = TECDetId(detrawid).moduleNumber();

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
	apvAbsoluteOccupancy = apvabsoluteOccupancy[apv];

	LocalPoint  pos_strip_local  = theStripTopol->localPosition((apv*128));
        GlobalPoint pos_strip_global = (TkGeom->idToDet(detectorId))->surface().toGlobal(pos_strip_local);

        global_position_x = pos_strip_global.x();
        global_position_y = pos_strip_global.y();
        global_position_z = pos_strip_global.z();

	if (WriteOutputFile_==true) apvtree->Fill();
      }

  } // end loop on modules

  MeanAndRms_TIB_Layer1 = CalculateMeanAndRMS(medianValues_TIB_Layer1,numberiterations_);
  MeanAndRms_TIB_Layer2 = CalculateMeanAndRMS(medianValues_TIB_Layer2,numberiterations_);
  MeanAndRms_TIB_Layer3 = CalculateMeanAndRMS(medianValues_TIB_Layer3,numberiterations_);
  MeanAndRms_TIB_Layer4 = CalculateMeanAndRMS(medianValues_TIB_Layer4,numberiterations_);

  MeanAndRms_TOB_Layer1 = CalculateMeanAndRMS(medianValues_TOB_Layer1,numberiterations_);
  MeanAndRms_TOB_Layer2 = CalculateMeanAndRMS(medianValues_TOB_Layer2,numberiterations_);
  MeanAndRms_TOB_Layer3 = CalculateMeanAndRMS(medianValues_TOB_Layer3,numberiterations_);
  MeanAndRms_TOB_Layer4 = CalculateMeanAndRMS(medianValues_TOB_Layer4,numberiterations_);
  MeanAndRms_TOB_Layer5 = CalculateMeanAndRMS(medianValues_TOB_Layer5,numberiterations_);
  MeanAndRms_TOB_Layer6 = CalculateMeanAndRMS(medianValues_TOB_Layer6,numberiterations_);

  MeanAndRms_TIDPlus_Disc1  = CalculateMeanAndRMS(medianValues_TIDPlus_Disc1,numberiterations_);
  MeanAndRms_TIDPlus_Disc2  = CalculateMeanAndRMS(medianValues_TIDPlus_Disc2,numberiterations_);
  MeanAndRms_TIDPlus_Disc3  = CalculateMeanAndRMS(medianValues_TIDPlus_Disc3,numberiterations_);
  MeanAndRms_TIDMinus_Disc1 = CalculateMeanAndRMS(medianValues_TIDMinus_Disc1,numberiterations_);
  MeanAndRms_TIDMinus_Disc2 = CalculateMeanAndRMS(medianValues_TIDMinus_Disc2,numberiterations_);
  MeanAndRms_TIDMinus_Disc3 = CalculateMeanAndRMS(medianValues_TIDMinus_Disc3,numberiterations_);

  MeanAndRms_TECPlus_Disc1 = CalculateMeanAndRMS(medianValues_TECPlus_Disc1,numberiterations_);
  MeanAndRms_TECPlus_Disc2 = CalculateMeanAndRMS(medianValues_TECPlus_Disc2,numberiterations_);
  MeanAndRms_TECPlus_Disc3 = CalculateMeanAndRMS(medianValues_TECPlus_Disc3,numberiterations_);
  MeanAndRms_TECPlus_Disc4 = CalculateMeanAndRMS(medianValues_TECPlus_Disc4,numberiterations_);
  MeanAndRms_TECPlus_Disc5 = CalculateMeanAndRMS(medianValues_TECPlus_Disc5,numberiterations_);
  MeanAndRms_TECPlus_Disc6 = CalculateMeanAndRMS(medianValues_TECPlus_Disc6,numberiterations_);
  MeanAndRms_TECPlus_Disc7 = CalculateMeanAndRMS(medianValues_TECPlus_Disc7,numberiterations_);
  MeanAndRms_TECPlus_Disc8 = CalculateMeanAndRMS(medianValues_TECPlus_Disc8,numberiterations_);
  MeanAndRms_TECPlus_Disc9 = CalculateMeanAndRMS(medianValues_TECPlus_Disc9,numberiterations_);

  MeanAndRms_TECMinus_Disc1 = CalculateMeanAndRMS(medianValues_TECMinus_Disc1,numberiterations_);
  MeanAndRms_TECMinus_Disc2 = CalculateMeanAndRMS(medianValues_TECMinus_Disc2,numberiterations_);
  MeanAndRms_TECMinus_Disc3 = CalculateMeanAndRMS(medianValues_TECMinus_Disc3,numberiterations_);
  MeanAndRms_TECMinus_Disc4 = CalculateMeanAndRMS(medianValues_TECMinus_Disc4,numberiterations_);
  MeanAndRms_TECMinus_Disc5 = CalculateMeanAndRMS(medianValues_TECMinus_Disc5,numberiterations_);
  MeanAndRms_TECMinus_Disc6 = CalculateMeanAndRMS(medianValues_TECMinus_Disc6,numberiterations_);
  MeanAndRms_TECMinus_Disc7 = CalculateMeanAndRMS(medianValues_TECMinus_Disc7,numberiterations_);
  MeanAndRms_TECMinus_Disc8 = CalculateMeanAndRMS(medianValues_TECMinus_Disc8,numberiterations_);
  MeanAndRms_TECMinus_Disc9 = CalculateMeanAndRMS(medianValues_TECMinus_Disc9,numberiterations_);

  pQuality=siStripQuality;
  badStripList.clear();

  // ############# TIB Layer 1 #############
  for (uint32_t it=0; it<medianValues_TIB_Layer1.size(); it++)
    {
      for (int apv=0; apv<medianValues_TIB_Layer1[it].numberApvs; apv++)
	{
	  if ((medianValues_TIB_Layer1[it].apvMedian[apv]<(MeanAndRms_TIB_Layer1.first-lowoccupancy_*MeanAndRms_TIB_Layer1.second)) || ((medianValues_TIB_Layer1[it].apvMedian[apv]>(MeanAndRms_TIB_Layer1.first+highoccupancy_*MeanAndRms_TIB_Layer1.second)) && medianValues_TIB_Layer1[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TIB_Layer1[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TIB_Layer1[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TIB Layer 2 #############
  for (uint32_t it=0; it<medianValues_TIB_Layer2.size(); it++)
    {
      for (int apv=0; apv<medianValues_TIB_Layer2[it].numberApvs; apv++)
	{
	  if ((medianValues_TIB_Layer2[it].apvMedian[apv]<(MeanAndRms_TIB_Layer2.first-lowoccupancy_*MeanAndRms_TIB_Layer2.second)) || ((medianValues_TIB_Layer2[it].apvMedian[apv]>(MeanAndRms_TIB_Layer2.first+highoccupancy_*MeanAndRms_TIB_Layer2.second)) && medianValues_TIB_Layer2[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TIB_Layer2[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TIB_Layer2[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TIB Layer 3 #############
  for (uint32_t it=0; it<medianValues_TIB_Layer3.size(); it++)
    {
      for (int apv=0; apv<medianValues_TIB_Layer3[it].numberApvs; apv++)
	{
	  if ((medianValues_TIB_Layer3[it].apvMedian[apv]<(MeanAndRms_TIB_Layer3.first-lowoccupancy_*MeanAndRms_TIB_Layer3.second)) || ((medianValues_TIB_Layer3[it].apvMedian[apv]>(MeanAndRms_TIB_Layer3.first+highoccupancy_*MeanAndRms_TIB_Layer3.second)) && medianValues_TIB_Layer3[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TIB_Layer3[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TIB_Layer3[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TIB Layer 4 #############
  for (uint32_t it=0; it<medianValues_TIB_Layer4.size(); it++)
    {
      for (int apv=0; apv<medianValues_TIB_Layer4[it].numberApvs; apv++)
	{
	  if ((medianValues_TIB_Layer4[it].apvMedian[apv]<(MeanAndRms_TIB_Layer4.first-lowoccupancy_*MeanAndRms_TIB_Layer4.second)) || ((medianValues_TIB_Layer4[it].apvMedian[apv]>(MeanAndRms_TIB_Layer4.first+highoccupancy_*MeanAndRms_TIB_Layer4.second)) && medianValues_TIB_Layer4[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TIB_Layer4[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TIB_Layer4[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TOB Layer 1 #############
  for (uint32_t it=0; it<medianValues_TOB_Layer1.size(); it++)
    {
      for (int apv=0; apv<medianValues_TOB_Layer1[it].numberApvs; apv++)
	{
	  if ((medianValues_TOB_Layer1[it].apvMedian[apv]<(MeanAndRms_TOB_Layer1.first-lowoccupancy_*MeanAndRms_TOB_Layer1.second)) || ((medianValues_TOB_Layer1[it].apvMedian[apv]>(MeanAndRms_TOB_Layer1.first+highoccupancy_*MeanAndRms_TOB_Layer1.second)) && medianValues_TOB_Layer1[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TOB_Layer1[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TOB_Layer1[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TOB Layer 2 #############
  for (uint32_t it=0; it<medianValues_TOB_Layer2.size(); it++)
    {
      for (int apv=0; apv<medianValues_TOB_Layer2[it].numberApvs; apv++)
	{
	  if ((medianValues_TOB_Layer2[it].apvMedian[apv]<(MeanAndRms_TOB_Layer2.first-lowoccupancy_*MeanAndRms_TOB_Layer2.second)) || ((medianValues_TOB_Layer2[it].apvMedian[apv]>(MeanAndRms_TOB_Layer2.first+highoccupancy_*MeanAndRms_TOB_Layer2.second)) && medianValues_TOB_Layer2[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TOB_Layer2[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TOB_Layer2[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TOB Layer 3 #############
  for (uint32_t it=0; it<medianValues_TOB_Layer3.size(); it++)
    {
      for (int apv=0; apv<medianValues_TOB_Layer3[it].numberApvs; apv++)
	{
	  if ((medianValues_TOB_Layer3[it].apvMedian[apv]<(MeanAndRms_TOB_Layer3.first-lowoccupancy_*MeanAndRms_TOB_Layer3.second)) || ((medianValues_TOB_Layer3[it].apvMedian[apv]>(MeanAndRms_TOB_Layer3.first+highoccupancy_*MeanAndRms_TOB_Layer3.second)) && medianValues_TOB_Layer3[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TOB_Layer3[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TOB_Layer3[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TOB Layer 4 #############
  for (uint32_t it=0; it<medianValues_TOB_Layer4.size(); it++)
    {
      for (int apv=0; apv<medianValues_TOB_Layer4[it].numberApvs; apv++)
	{
	  if ((medianValues_TOB_Layer4[it].apvMedian[apv]<(MeanAndRms_TOB_Layer4.first-lowoccupancy_*MeanAndRms_TOB_Layer4.second)) || ((medianValues_TOB_Layer4[it].apvMedian[apv]>(MeanAndRms_TOB_Layer4.first+highoccupancy_*MeanAndRms_TOB_Layer4.second)) && medianValues_TOB_Layer4[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TOB_Layer4[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TOB_Layer4[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TOB Layer 5 #############
  for (uint32_t it=0; it<medianValues_TOB_Layer5.size(); it++)
    {
      for (int apv=0; apv<medianValues_TOB_Layer5[it].numberApvs; apv++)
	{
	  if ((medianValues_TOB_Layer5[it].apvMedian[apv]<(MeanAndRms_TOB_Layer5.first-lowoccupancy_*MeanAndRms_TOB_Layer5.second)) || ((medianValues_TOB_Layer5[it].apvMedian[apv]>(MeanAndRms_TOB_Layer5.first+highoccupancy_*MeanAndRms_TOB_Layer5.second)) && medianValues_TOB_Layer5[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TOB_Layer5[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TOB_Layer5[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TOB Layer 6 #############
  for (uint32_t it=0; it<medianValues_TOB_Layer6.size(); it++)
    {
      for (int apv=0; apv<medianValues_TOB_Layer6[it].numberApvs; apv++)
	{
	  if ((medianValues_TOB_Layer6[it].apvMedian[apv]<(MeanAndRms_TOB_Layer6.first-lowoccupancy_*MeanAndRms_TOB_Layer6.second)) || ((medianValues_TOB_Layer6[it].apvMedian[apv]>(MeanAndRms_TOB_Layer6.first+highoccupancy_*MeanAndRms_TOB_Layer6.second)) && medianValues_TOB_Layer6[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TOB_Layer6[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TOB_Layer6[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TIDPlus Disk 1 #############
  for (uint32_t it=0; it<medianValues_TIDPlus_Disc1.size(); it++)
    {
      for (int apv=0; apv<medianValues_TIDPlus_Disc1[it].numberApvs; apv++)
	{
	  if ((medianValues_TIDPlus_Disc1[it].apvMedian[apv]<(MeanAndRms_TIDPlus_Disc1.first-lowoccupancy_*MeanAndRms_TIDPlus_Disc1.second)) || ((medianValues_TIDPlus_Disc1[it].apvMedian[apv]>(MeanAndRms_TIDPlus_Disc1.first+highoccupancy_*MeanAndRms_TIDPlus_Disc1.second)) && medianValues_TIDPlus_Disc1[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TIDPlus_Disc1[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TIDPlus_Disc1[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TIDPlus Disk 2 #############
  for (uint32_t it=0; it<medianValues_TIDPlus_Disc2.size(); it++)
    {
      for (int apv=0; apv<medianValues_TIDPlus_Disc2[it].numberApvs; apv++)
	{
	  if ((medianValues_TIDPlus_Disc2[it].apvMedian[apv]<(MeanAndRms_TIDPlus_Disc2.first-lowoccupancy_*MeanAndRms_TIDPlus_Disc2.second)) || ((medianValues_TIDPlus_Disc2[it].apvMedian[apv]>(MeanAndRms_TIDPlus_Disc2.first+highoccupancy_*MeanAndRms_TIDPlus_Disc2.second)) && medianValues_TIDPlus_Disc2[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TIDPlus_Disc2[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TIDPlus_Disc2[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TIDPlus Disk 3 #############
  for (uint32_t it=0; it<medianValues_TIDPlus_Disc3.size(); it++)
    {
      for (int apv=0; apv<medianValues_TIDPlus_Disc3[it].numberApvs; apv++)
	{
	  if ((medianValues_TIDPlus_Disc3[it].apvMedian[apv]<(MeanAndRms_TIDPlus_Disc3.first-lowoccupancy_*MeanAndRms_TIDPlus_Disc3.second)) || ((medianValues_TIDPlus_Disc3[it].apvMedian[apv]>(MeanAndRms_TIDPlus_Disc3.first+highoccupancy_*MeanAndRms_TIDPlus_Disc3.second)) && medianValues_TIDPlus_Disc3[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TIDPlus_Disc3[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TIDPlus_Disc3[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TIDMinus Disk 1 #############
  for (uint32_t it=0; it<medianValues_TIDMinus_Disc1.size(); it++)
    {
      for (int apv=0; apv<medianValues_TIDMinus_Disc1[it].numberApvs; apv++)
	{
	  if ((medianValues_TIDMinus_Disc1[it].apvMedian[apv]<(MeanAndRms_TIDMinus_Disc1.first-lowoccupancy_*MeanAndRms_TIDMinus_Disc1.second)) || ((medianValues_TIDMinus_Disc1[it].apvMedian[apv]>(MeanAndRms_TIDMinus_Disc1.first+highoccupancy_*MeanAndRms_TIDMinus_Disc1.second)) && medianValues_TIDMinus_Disc1[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TIDMinus_Disc1[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TIDMinus_Disc1[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TIDMinus Disk 2 #############
  for (uint32_t it=0; it<medianValues_TIDMinus_Disc2.size(); it++)
    {
      for (int apv=0; apv<medianValues_TIDMinus_Disc2[it].numberApvs; apv++)
	{
	  if ((medianValues_TIDMinus_Disc2[it].apvMedian[apv]<(MeanAndRms_TIDMinus_Disc2.first-lowoccupancy_*MeanAndRms_TIDMinus_Disc2.second)) || ((medianValues_TIDMinus_Disc2[it].apvMedian[apv]>(MeanAndRms_TIDMinus_Disc2.first+highoccupancy_*MeanAndRms_TIDMinus_Disc2.second)) && medianValues_TIDMinus_Disc2[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TIDMinus_Disc2[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TIDMinus_Disc2[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TIDMinus Disk 3 #############
  for (uint32_t it=0; it<medianValues_TIDMinus_Disc3.size(); it++)
    {
      for (int apv=0; apv<medianValues_TIDMinus_Disc3[it].numberApvs; apv++)
	{
	  if ((medianValues_TIDMinus_Disc3[it].apvMedian[apv]<(MeanAndRms_TIDMinus_Disc3.first-lowoccupancy_*MeanAndRms_TIDMinus_Disc3.second)) || ((medianValues_TIDMinus_Disc3[it].apvMedian[apv]>(MeanAndRms_TIDMinus_Disc3.first+highoccupancy_*MeanAndRms_TIDMinus_Disc3.second)) && medianValues_TIDMinus_Disc3[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TIDMinus_Disc3[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TIDMinus_Disc3[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECPlus Disk 1 #############
  for (uint32_t it=0; it<medianValues_TECPlus_Disc1.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECPlus_Disc1[it].numberApvs; apv++)
	{
	  if ((medianValues_TECPlus_Disc1[it].apvMedian[apv]<(MeanAndRms_TECPlus_Disc1.first-lowoccupancy_*MeanAndRms_TECPlus_Disc1.second)) || ((medianValues_TECPlus_Disc1[it].apvMedian[apv]>(MeanAndRms_TECPlus_Disc1.first+highoccupancy_*MeanAndRms_TECPlus_Disc1.second)) && medianValues_TECPlus_Disc1[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECPlus_Disc1[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECPlus_Disc1[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECPlus Disk 2 #############
  for (uint32_t it=0; it<medianValues_TECPlus_Disc2.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECPlus_Disc2[it].numberApvs; apv++)
	{
	  if ((medianValues_TECPlus_Disc2[it].apvMedian[apv]<(MeanAndRms_TECPlus_Disc2.first-lowoccupancy_*MeanAndRms_TECPlus_Disc2.second)) || ((medianValues_TECPlus_Disc2[it].apvMedian[apv]>(MeanAndRms_TECPlus_Disc2.first+highoccupancy_*MeanAndRms_TECPlus_Disc2.second)) && medianValues_TECPlus_Disc2[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECPlus_Disc2[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECPlus_Disc2[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECPlus Disk 3 #############
  for (uint32_t it=0; it<medianValues_TECPlus_Disc3.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECPlus_Disc3[it].numberApvs; apv++)
	{
	  if ((medianValues_TECPlus_Disc3[it].apvMedian[apv]<(MeanAndRms_TECPlus_Disc3.first-lowoccupancy_*MeanAndRms_TECPlus_Disc3.second)) || ((medianValues_TECPlus_Disc3[it].apvMedian[apv]>(MeanAndRms_TECPlus_Disc3.first+highoccupancy_*MeanAndRms_TECPlus_Disc3.second)) && medianValues_TECPlus_Disc3[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECPlus_Disc3[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECPlus_Disc3[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECPlus Disk 4 #############
  for (uint32_t it=0; it<medianValues_TECPlus_Disc4.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECPlus_Disc4[it].numberApvs; apv++)
	{
	  if ((medianValues_TECPlus_Disc4[it].apvMedian[apv]<(MeanAndRms_TECPlus_Disc4.first-lowoccupancy_*MeanAndRms_TECPlus_Disc4.second)) || ((medianValues_TECPlus_Disc4[it].apvMedian[apv]>(MeanAndRms_TECPlus_Disc4.first+highoccupancy_*MeanAndRms_TECPlus_Disc4.second)) && medianValues_TECPlus_Disc4[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECPlus_Disc4[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECPlus_Disc4[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECPlus Disk 5 #############
  for (uint32_t it=0; it<medianValues_TECPlus_Disc5.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECPlus_Disc5[it].numberApvs; apv++)
	{
	  if ((medianValues_TECPlus_Disc5[it].apvMedian[apv]<(MeanAndRms_TECPlus_Disc5.first-lowoccupancy_*MeanAndRms_TECPlus_Disc5.second)) || ((medianValues_TECPlus_Disc5[it].apvMedian[apv]>(MeanAndRms_TECPlus_Disc5.first+highoccupancy_*MeanAndRms_TECPlus_Disc5.second)) && medianValues_TECPlus_Disc5[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECPlus_Disc5[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECPlus_Disc5[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECPlus Disk 6 #############
  for (uint32_t it=0; it<medianValues_TECPlus_Disc6.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECPlus_Disc6[it].numberApvs; apv++)
	{
	  if ((medianValues_TECPlus_Disc6[it].apvMedian[apv]<(MeanAndRms_TECPlus_Disc6.first-lowoccupancy_*MeanAndRms_TECPlus_Disc6.second)) || ((medianValues_TECPlus_Disc6[it].apvMedian[apv]>(MeanAndRms_TECPlus_Disc6.first+highoccupancy_*MeanAndRms_TECPlus_Disc6.second)) && medianValues_TECPlus_Disc6[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECPlus_Disc6[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECPlus_Disc6[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECPlus Disk 7 #############
  for (uint32_t it=0; it<medianValues_TECPlus_Disc7.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECPlus_Disc7[it].numberApvs; apv++)
	{
	  if ((medianValues_TECPlus_Disc7[it].apvMedian[apv]<(MeanAndRms_TECPlus_Disc7.first-lowoccupancy_*MeanAndRms_TECPlus_Disc7.second)) || ((medianValues_TECPlus_Disc7[it].apvMedian[apv]>(MeanAndRms_TECPlus_Disc7.first+highoccupancy_*MeanAndRms_TECPlus_Disc7.second)) && medianValues_TECPlus_Disc7[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECPlus_Disc7[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECPlus_Disc7[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECPlus Disk 8 #############
  for (uint32_t it=0; it<medianValues_TECPlus_Disc8.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECPlus_Disc8[it].numberApvs; apv++)
	{
	  if ((medianValues_TECPlus_Disc8[it].apvMedian[apv]<(MeanAndRms_TECPlus_Disc8.first-lowoccupancy_*MeanAndRms_TECPlus_Disc8.second)) || ((medianValues_TECPlus_Disc8[it].apvMedian[apv]>(MeanAndRms_TECPlus_Disc8.first+highoccupancy_*MeanAndRms_TECPlus_Disc8.second)) && medianValues_TECPlus_Disc8[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECPlus_Disc8[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECPlus_Disc8[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECPlus Disk 9 #############
  for (uint32_t it=0; it<medianValues_TECPlus_Disc9.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECPlus_Disc9[it].numberApvs; apv++)
	{
	  if ((medianValues_TECPlus_Disc9[it].apvMedian[apv]<(MeanAndRms_TECPlus_Disc9.first-lowoccupancy_*MeanAndRms_TECPlus_Disc9.second)) || ((medianValues_TECPlus_Disc9[it].apvMedian[apv]>(MeanAndRms_TECPlus_Disc9.first+highoccupancy_*MeanAndRms_TECPlus_Disc9.second)) && medianValues_TECPlus_Disc9[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECPlus_Disc9[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECPlus_Disc9[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECMinus Disk 1 #############
  for (uint32_t it=0; it<medianValues_TECMinus_Disc1.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECMinus_Disc1[it].numberApvs; apv++)
	{
	  if ((medianValues_TECMinus_Disc1[it].apvMedian[apv]<(MeanAndRms_TECMinus_Disc1.first-lowoccupancy_*MeanAndRms_TECMinus_Disc1.second)) || ((medianValues_TECMinus_Disc1[it].apvMedian[apv]>(MeanAndRms_TECMinus_Disc1.first+highoccupancy_*MeanAndRms_TECMinus_Disc1.second)) && medianValues_TECMinus_Disc1[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECMinus_Disc1[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECMinus_Disc1[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECMinus Disk 2 #############
  for (uint32_t it=0; it<medianValues_TECMinus_Disc2.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECMinus_Disc2[it].numberApvs; apv++)
	{
	  if ((medianValues_TECMinus_Disc2[it].apvMedian[apv]<(MeanAndRms_TECMinus_Disc2.first-lowoccupancy_*MeanAndRms_TECMinus_Disc2.second)) || ((medianValues_TECMinus_Disc2[it].apvMedian[apv]>(MeanAndRms_TECMinus_Disc2.first+highoccupancy_*MeanAndRms_TECMinus_Disc2.second)) && medianValues_TECMinus_Disc2[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECMinus_Disc2[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECMinus_Disc2[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECMinus Disk 3 #############
  for (uint32_t it=0; it<medianValues_TECMinus_Disc3.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECMinus_Disc3[it].numberApvs; apv++)
	{
	  if ((medianValues_TECMinus_Disc3[it].apvMedian[apv]<(MeanAndRms_TECMinus_Disc3.first-lowoccupancy_*MeanAndRms_TECMinus_Disc3.second)) || ((medianValues_TECMinus_Disc3[it].apvMedian[apv]>(MeanAndRms_TECMinus_Disc3.first+highoccupancy_*MeanAndRms_TECMinus_Disc3.second)) && medianValues_TECMinus_Disc3[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECMinus_Disc3[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECMinus_Disc3[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECMinus Disk 4 #############
  for (uint32_t it=0; it<medianValues_TECMinus_Disc4.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECMinus_Disc4[it].numberApvs; apv++)
	{
	  if ((medianValues_TECMinus_Disc4[it].apvMedian[apv]<(MeanAndRms_TECMinus_Disc4.first-lowoccupancy_*MeanAndRms_TECMinus_Disc4.second)) || ((medianValues_TECMinus_Disc4[it].apvMedian[apv]>(MeanAndRms_TECMinus_Disc4.first+highoccupancy_*MeanAndRms_TECMinus_Disc4.second)) && medianValues_TECMinus_Disc4[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECMinus_Disc4[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECMinus_Disc4[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECMinus Disk 5 #############
  for (uint32_t it=0; it<medianValues_TECMinus_Disc5.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECMinus_Disc5[it].numberApvs; apv++)
	{
	  if ((medianValues_TECMinus_Disc5[it].apvMedian[apv]<(MeanAndRms_TECMinus_Disc5.first-lowoccupancy_*MeanAndRms_TECMinus_Disc5.second)) || ((medianValues_TECMinus_Disc5[it].apvMedian[apv]>(MeanAndRms_TECMinus_Disc5.first+highoccupancy_*MeanAndRms_TECMinus_Disc5.second)) && medianValues_TECMinus_Disc5[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECMinus_Disc5[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECMinus_Disc5[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECMinus Disk 6 #############
  for (uint32_t it=0; it<medianValues_TECMinus_Disc6.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECMinus_Disc6[it].numberApvs; apv++)
	{
	  if ((medianValues_TECMinus_Disc6[it].apvMedian[apv]<(MeanAndRms_TECMinus_Disc6.first-lowoccupancy_*MeanAndRms_TECMinus_Disc6.second)) || ((medianValues_TECMinus_Disc6[it].apvMedian[apv]>(MeanAndRms_TECMinus_Disc6.first+highoccupancy_*MeanAndRms_TECMinus_Disc6.second)) && medianValues_TECMinus_Disc6[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECMinus_Disc6[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECMinus_Disc6[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECMinus Disk 7 #############
  for (uint32_t it=0; it<medianValues_TECMinus_Disc7.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECMinus_Disc7[it].numberApvs; apv++)
	{
	  if ((medianValues_TECMinus_Disc7[it].apvMedian[apv]<(MeanAndRms_TECMinus_Disc7.first-lowoccupancy_*MeanAndRms_TECMinus_Disc7.second)) || ((medianValues_TECMinus_Disc7[it].apvMedian[apv]>(MeanAndRms_TECMinus_Disc7.first+highoccupancy_*MeanAndRms_TECMinus_Disc7.second)) && medianValues_TECMinus_Disc7[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECMinus_Disc7[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECMinus_Disc7[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECMinus Disk 8 #############
  for (uint32_t it=0; it<medianValues_TECMinus_Disc8.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECMinus_Disc8[it].numberApvs; apv++)
	{
	  if ((medianValues_TECMinus_Disc8[it].apvMedian[apv]<(MeanAndRms_TECMinus_Disc8.first-lowoccupancy_*MeanAndRms_TECMinus_Disc8.second)) || ((medianValues_TECMinus_Disc8[it].apvMedian[apv]>(MeanAndRms_TECMinus_Disc8.first+highoccupancy_*MeanAndRms_TECMinus_Disc8.second)) && medianValues_TECMinus_Disc8[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECMinus_Disc8[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECMinus_Disc8[it].detrawId,range);
	}
      badStripList.clear();
    }
  // ############# TECMinus Disk 9 #############
  for (uint32_t it=0; it<medianValues_TECMinus_Disc9.size(); it++)
    {
      for (int apv=0; apv<medianValues_TECMinus_Disc9[it].numberApvs; apv++)
	{
	  if ((medianValues_TECMinus_Disc9[it].apvMedian[apv]<(MeanAndRms_TECMinus_Disc9.first-lowoccupancy_*MeanAndRms_TECMinus_Disc9.second)) || ((medianValues_TECMinus_Disc9[it].apvMedian[apv]>(MeanAndRms_TECMinus_Disc9.first+highoccupancy_*MeanAndRms_TECMinus_Disc9.second)) && medianValues_TECMinus_Disc9[it].apvMedian[apv]>absolutelow_))
	    badStripList.push_back(pQuality->encode((apv*128),128,0));

	}
      if (badStripList.begin()!=badStripList.end())
	{
	  siStripQuality->compact(medianValues_TECMinus_Disc9[it].detrawId,badStripList);
	  SiStripQuality::Range range(badStripList.begin(),badStripList.end());
	  siStripQuality->put(medianValues_TECMinus_Disc9[it].detrawId,range);
	}
      badStripList.clear();
    }

  siStripQuality->fillBadComponents();

  if (WriteOutputFile_==true){
  f->cd();
  apvtree->Write();
  f->Close();
  }

  LogTrace("SiStripBadAPV") << ss.str() << std::endl;
}


std::pair<double,double> SiStripBadAPVAlgorithmFromClusterOccupancy::CalculateMeanAndRMS(std::vector<Apv> a, int number_iterations)
{
  Double_t tot, tot2;
  Double_t n;

  Double_t Mean = 0;
  Double_t Rms = 1000;

  for (int i=0; i<number_iterations; i++)
    {
      n    = 0;
      tot  = 0;
      tot2 = 0;

      for (uint32_t it=0; it<a.size(); it++)
	{
	  for (int apv=0; apv<a[it].numberApvs; apv++)
	    {
	      if (i>0)
		{
		  if (a[it].apvMedian[apv]<(Mean-3*Rms) || (a[it].apvMedian[apv]>(Mean+5*Rms)))
		    {
		      continue;
		    }
		}
	      tot  += a[it].apvMedian[apv];
	      tot2 += (a[it].apvMedian[apv])*(a[it].apvMedian[apv]);
	      n++;
	    }
	}

      Mean = tot/n;
      Rms  = TMath::Sqrt(TMath::Abs(tot2/n -Mean*Mean));
    }

  return std::make_pair(Mean,Rms);
}
