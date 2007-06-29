#include "RecoTBCalo/HcalPlotter/src/HcalQLPlotHistoMgr.h"
#include "TProfile.h"

//Declare Histogram Bin parameters
static const int PED_BINS=50;
static const int LED_BINS=50;
static const int LASER_BINS=70;
static const int BEAM_BINS=70;
static const int OTHER_BINS=100;
static const int TIME_BINS=75;
static const int PULSE_BINS=10;

HcalQLPlotHistoMgr::HcalQLPlotHistoMgr(TDirectory* parent,
				       const edm::ParameterSet& histoParams) {
  pedHistDir=parent->mkdir("PEDESTAL");
  ledHistDir=parent->mkdir("LED");
  laserHistDir=parent->mkdir("LASER");
  beamHistDir=parent->mkdir("BEAM");
  otherHistDir=parent->mkdir("OTHER");
  histoParams_ = histoParams;
}

std::string HcalQLPlotHistoMgr::nameForFlavor(HistType ht) {
  switch (ht) {
  case(ENERGY): return "Energy"; break;
  case(TIME):  return "Time"; break;
  case(PULSE):  return "Pulse"; break;
  default: return ""; break;
  }
}

std::string HcalQLPlotHistoMgr::nameForEvent(EventType et) {
  switch(et) {
  case(PEDESTAL): return "Pedestal"; break;
  case(LED): return "LED"; break;
  case(LASER): return "Laser"; break;
  case(BEAM): return "Beam"; break;
  default: return "Other"; break;
  }	
}

TH1* HcalQLPlotHistoMgr::GetAHistogram(const HcalDetId& id,
				       const HcalElectronicsId& eid,
				       HistType ht, EventType et) {

  std::string flavor=nameForFlavor(ht);
  TDirectory* td;

  switch (et) {
  case(PEDESTAL): td=pedHistDir; break;
  case(LED): td=ledHistDir; break;
  case(LASER): td=laserHistDir; break;
  case(BEAM): td=beamHistDir; break;
  case(UNKNOWN): td=otherHistDir; break;
  default: td=0; break;
  }

  if (td==0) {
    printf("Unknown %d !\n", et);
    return 0;
  }

  char name[120];

  std::string subdetStr;
  switch (id.subdet()) {
  case (HcalBarrel) : subdetStr="HB"; break;
  case (HcalEndcap) : subdetStr="HE"; break;
  case (HcalOuter) : subdetStr="HO"; break;
  case (HcalForward) : subdetStr="HF"; break;
  default: subdetStr="Other"; break;
  }

  TH1* retval=0;
  sprintf(name,"%s_%s_%d_%d_%d_eid=%d_%d_%d_%d_HTR_%d:%d%c",
	  flavor.c_str(),subdetStr.c_str(),id.ieta(),id.iphi(),id.depth(),
	  eid.dccid(),eid.spigot(), eid.fiberIndex(), eid.fiberChanId(),
	  eid.readoutVMECrateId(), eid.htrSlot(),(eid.htrTopBottom()==1)?('t'):('b') );

  retval=(TH1*)td->Get(name);
  int bins=0; double lo=0, hi=0;

  // If the histogram doesn't exist and we are authorized,
  // create it!
  //
  if (retval==0) {
    td->cd();
    switch (ht) {
    case(ENERGY): {
      switch (et) {
      case(PEDESTAL):
	bins=PED_BINS;
	lo=histoParams_.getParameter<double>("pedGeVlo");
	hi=histoParams_.getParameter<double>("pedGeVhi");
	break;
      case(LED):
	bins=LED_BINS;
	lo=histoParams_.getParameter<double>("ledGeVlo");
	hi=histoParams_.getParameter<double>("ledGeVhi");
	break;
      case(LASER):
	bins=LASER_BINS;
	lo=histoParams_.getParameter<double>("laserGeVlo");
	hi=histoParams_.getParameter<double>("laserGeVhi");
	break;
      case(BEAM):
	bins=BEAM_BINS;
	lo=histoParams_.getParameter<double>("beamGeVlo");
	hi=histoParams_.getParameter<double>("beamGeVhi");
	break;
      case(UNKNOWN):
	bins=OTHER_BINS;
	lo=histoParams_.getParameter<double>("otherGeVlo");
	hi=histoParams_.getParameter<double>("otherGeVhi");
	break;
      default: break;
      };
    }
      break;
    case(TIME):
      bins=TIME_BINS;
      lo=histoParams_.getParameter<double>("timeNSlo");
      hi=histoParams_.getParameter<double>("timeNShi");
      break;
    case(PULSE):
      bins=PULSE_BINS;
      lo=-0.5;
      hi=PULSE_BINS-0.5;
      break;
    }
   
    if (bins>0){
      if (ht==PULSE){
        retval=new TProfile(name,name,bins,lo,hi);
        retval->GetXaxis()->SetTitle("TimeSlice(25ns)");
        retval->GetYaxis()->SetTitle("fC");
      }   
      else if (ht==TIME){
        retval=new TH1F(name,name,bins,lo,hi);
        retval->GetXaxis()->SetTitle("Timing(ns)");
      }
      else if (ht==ENERGY){
        retval=new TH1F(name,name,bins,lo,hi);
        retval->GetXaxis()->SetTitle("Energy(GeV)");
      }
    } 
  }

  
  return retval;
}


std::vector<HcalDetId> HcalQLPlotHistoMgr::getDetIdsForType(HistType ht, EventType et) {
  char keyflavor[100];
  char keysubDet[100];
  int ieta, iphi, depth;
  TDirectory* td;
  TList* keyList;
  std::vector<HcalDetId> retvals;

  std::string flavor=nameForFlavor(ht);

  switch (et) {
  case(PEDESTAL): td=pedHistDir; break;
  case(LED): td=ledHistDir; break;
  case(LASER): td=laserHistDir; break;
  case(BEAM): td=beamHistDir; break;
  case(UNKNOWN): td=otherHistDir; break;
  default: td=0; break;
  }

  keyList = td->GetListOfKeys();
  
  for(int keyindex = 0; keyindex<keyList->GetEntries(); ++keyindex) {
    int converted;
    std::string keyname = keyList->At(keyindex)->GetName();
    while (keyname.find("_")!=std::string::npos)
      keyname.replace(keyname.find("_"),1," ");
    converted = sscanf(keyname.c_str(),"%s %s %d %d %d",
		       keyflavor,keysubDet,&ieta,&iphi,&depth);
    if( (flavor==keyflavor) && (converted==5) ) {
      HcalSubdetector subDet;

      if (!strcmp(keysubDet,"HB")) subDet=HcalBarrel;
      else if (!strcmp(keysubDet,"HE")) subDet=HcalEndcap;
      else if (!strcmp(keysubDet,"HO")) subDet=HcalOuter;
      else if (!strcmp(keysubDet,"HF")) subDet=HcalForward;
      else continue; // and do not include this in the list!

      retvals.push_back(HcalDetId(subDet,ieta,iphi,depth));
    }
  }

  return retvals;
}

