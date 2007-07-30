#include "HistoManager.h"
#include "TProfile.h"
#include <iostream>

HistoManager::HistoManager(TDirectory* parent)
{
  pedHistDir=(TDirectory*)parent->Get("PEDESTAL");
  ledHistDir=(TDirectory*)parent->Get("LED");
  laserHistDir=(TDirectory*)parent->Get("LASER");
  beamHistDir=(TDirectory*)parent->Get("BEAM");
  otherHistDir=(TDirectory*)parent->Get("OTHER");
}

std::string HistoManager::nameForFlavor(HistType ht)
{
  switch (ht) {
  case(ENERGY): return "Energy"; break;
  case(TIME):  return "Time"; break;
  case(PULSE):  return "Pulse"; break;
  default: return ""; break;
  }
}

std::string HistoManager::nameForEvent(EventType et)
{
  switch(et) {
  case(PEDESTAL): return "Pedestal"; break;
  case(LED): return "LED"; break;
  case(LASER): return "Laser"; break;
  case(BEAM): return "Beam"; break;
  default: return "Other"; break;
  }	
}

TH1* HistoManager::GetAHistogram(const MyHcalDetId& id,
				 HistType ht,
				 EventType et)
{
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
  switch (id.subdet) {
  case (HcalBarrel) : subdetStr="HB"; break;
  case (HcalEndcap) : subdetStr="HE"; break;
  case (HcalOuter) : subdetStr="HO"; break;
  case (HcalForward) : subdetStr="HF"; break;
  default: subdetStr="Other"; break;
  }

  TH1* retval=0;
  sprintf(name,"%s_%s_%d_%d_%d",
	  flavor.c_str(),subdetStr.c_str(),id.ieta,id.iphi,id.depth);

  TList* keyList = td->GetListOfKeys();
  
  for(int keyindex = 0; keyindex<keyList->GetEntries(); ++keyindex) {
    std::string keyname = keyList->At(keyindex)->GetName();
    if (strstr(keyname.c_str(),name)) {
      retval=(TH1*)td->Get(keyname.c_str());
      break;
    }
  }

  return retval;
}


std::vector<MyHcalDetId> HistoManager::getDetIdsForType(HistType ht,
							EventType et)
{
  char keyflavor[100];
  char keysubDet[100];
  MyHcalDetId mydetid;
  TDirectory* td;
  TList* keyList;
  std::vector<MyHcalDetId> retvals;

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
		       keyflavor,keysubDet,
		       &mydetid.ieta,&mydetid.iphi,&mydetid.depth);
    if( (flavor==keyflavor) && (converted==5) ) {
      if (!strcmp(keysubDet,"HB")) mydetid.subdet=HcalBarrel;
      else if (!strcmp(keysubDet,"HE")) mydetid.subdet=HcalEndcap;
      else if (!strcmp(keysubDet,"HO")) mydetid.subdet=HcalOuter;
      else if (!strcmp(keysubDet,"HF")) mydetid.subdet=HcalForward;
      else continue; // and do not include this in the list!

      retvals.push_back(mydetid);
    }
  }


  return retvals;
}

