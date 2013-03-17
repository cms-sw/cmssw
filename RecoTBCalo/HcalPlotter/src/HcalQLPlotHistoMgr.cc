#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
  case(ADC):  return "ADC"; break;
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
				       HistType ht, EventType et)
{
  std::string flavor=nameForFlavor(ht);

  char name[120];

  std::string subdetStr;
  switch (id.subdet()) {
  case (HcalBarrel)  : subdetStr="HB"; break;
  case (HcalEndcap)  : subdetStr="HE"; break;
  case (HcalOuter)   : subdetStr="HO"; break;
  case (HcalForward) : subdetStr="HF"; break;
  default: subdetStr="Other"; break;
  }

  sprintf(name,"%s_%s_%d_%d_%d_eid=%d_%d_%d_%d_HTR_%d:%d%c",
	  flavor.c_str(),subdetStr.c_str(),id.ieta(),id.iphi(),id.depth(),
	  eid.dccid(),eid.spigot(), eid.fiberIndex(), eid.fiberChanId(),
	  eid.readoutVMECrateId(), eid.htrSlot(),(eid.htrTopBottom()==1)?('t'):('b') );

  return GetAHistogramImpl(name,ht,et);
}

TH1* HcalQLPlotHistoMgr::GetAHistogram(const HcalCalibDetId& id,
				       const HcalElectronicsId& eid,
				       HistType ht, EventType et)
{
  std::string flavor=nameForFlavor(ht);

  char name[120];

  std::string subdetStr;
  switch (id.hcalSubdet()) {
  case (HcalBarrel)  : subdetStr="HB"; break;
  case (HcalEndcap)  : subdetStr="HE"; break;
  case (HcalOuter)   : subdetStr="HO"; break;
  case (HcalForward) : subdetStr="HF"; break;
  default: subdetStr="Other"; break;
  }

  std::string chanstring = id.cboxChannelString();
  if (!chanstring.size()) {
    chanstring = "Unknown";
    edm::LogInfo("HcalQLPlotHistoMgr::GetAHistogram") << "Unknown calibration channel " << id.cboxChannel();
  }

  sprintf(name,"%s_CALIB_%s_%d_%d_chan=%s_eid=%d_%d_%d_%d_HTR_%d:%d%c",
	  flavor.c_str(),subdetStr.c_str(),id.ieta(),id.iphi(),chanstring.c_str(),
	  eid.dccid(),eid.spigot(), eid.fiberIndex(), eid.fiberChanId(),
	  eid.readoutVMECrateId(), eid.htrSlot(),(eid.htrTopBottom()==1)?('t'):('b') );

  return GetAHistogramImpl(name,ht,et);
}

TH1* HcalQLPlotHistoMgr::GetAHistogramImpl(const char *name,
					   HistType ht, EventType et)
{
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

  TH1* retval=0;

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
	try {
	  lo=histoParams_.getParameter<double>("pedGeVlo");
	  hi=histoParams_.getParameter<double>("pedGeVhi");
        } catch (std::exception& e) { // can't find it!
	  edm::LogError("HcalQLPlotHistoMgr::GetAHistogram") << "Parameter(s) pedGeVlo/hi not found.";
	  throw e;
	}
	break;
      case(LED):
	bins=LED_BINS;
	try {
	  lo=histoParams_.getParameter<double>("ledGeVlo");
	  hi=histoParams_.getParameter<double>("ledGeVhi");
        } catch (std::exception& e) { // can't find it!
	  edm::LogError("HcalQLPlotHistoMgr::GetAHistogram") << "Parameter(s) ledGeVlo/hi not found.";
	  throw e;
	}
	break;
      case(LASER):
	bins=LASER_BINS;
	try {
	  lo=histoParams_.getParameter<double>("laserGeVlo");
	  hi=histoParams_.getParameter<double>("laserGeVhi");
        } catch (std::exception& e) { // can't find it!
	  edm::LogError("HcalQLPlotHistoMgr::GetAHistogram") << "Parameter(s) laserGeVlo/hi not found.";
	  throw e;
	}
	break;
      case(BEAM):
	bins=BEAM_BINS;
	try {
	  lo=histoParams_.getParameter<double>("beamGeVlo");
	  hi=histoParams_.getParameter<double>("beamGeVhi");
        } catch (std::exception& e) { // can't find it!
	  edm::LogError("HcalQLPlotHistoMgr::GetAHistogram") << "Parameter(s) beamGeVlo/hi not found.";
	  throw e;
	}
	break;
      case(UNKNOWN):
	bins=OTHER_BINS;
	try {
	  lo=histoParams_.getParameter<double>("otherGeVlo");
	  hi=histoParams_.getParameter<double>("otherGeVhi");
        } catch (std::exception& e) { // can't find it!
	  edm::LogError("HcalQLPlotHistoMgr::GetAHistogram") << "Parameter(s) otherGeVlo/hi not found.";
	  throw e;
	}
	break;
      default: break;
      };
    }
      break;
    case(TIME):
      bins=TIME_BINS;
      try {
	lo=histoParams_.getParameter<double>("timeNSlo");
	hi=histoParams_.getParameter<double>("timeNShi");
      } catch (std::exception& e) { // can't find it!
	edm::LogError("HcalQLPlotHistoMgr::GetAHistogram") << "Parameter(s) timeNSlo/hi not found.";
	throw e;
      }
      break;
    case(PULSE):
      bins=PULSE_BINS;
      lo=-0.5;
      hi=PULSE_BINS-0.5;
      break;
    case(ADC):
      bins=PED_BINS;
      try {
	lo=histoParams_.getParameter<double>("pedADClo");
	hi=histoParams_.getParameter<double>("pedADChi");
      } catch (std::exception& e) { // can't find it!
	edm::LogError("HcalQLPlotHistoMgr::GetAHistogram") << "Parameter(s) pedADClo/hi not found.";
	throw e;
      }
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
      else if (ht==ADC){
        retval=new TH1F(name,name,bins,lo,hi);
        retval->GetXaxis()->SetTitle("ADC Counts");
      }
    } 
  }

  return retval;
}
