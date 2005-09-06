#include "CondFormats/HcalMapping/interface/HcalMappingTextFileReader.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <stdio.h>
#include <iostream> 
using namespace std;

namespace cms {
  namespace hcal {
HcalMapping HcalMappingTextFileReader::readFromFile(const char* filename, bool maintainL2E) {

  HcalMapping hrm(maintainL2E);
  char buffer[1024];
  int lineNo, crate, slot;
  char tb;
  int dccNo, spigot, fiber, fiberchan;
  char subdet[10];
  int ieta, iphi, depth;

  FILE* f=fopen(filename,"r");

  if (f==0) {
    cerr << "Unable to open '" << filename << "'" << endl;
    throw cms::Exception("FileNotFound") << "Unable to open '" << filename << "'" << endl;
  }

  while (f!=0 && !feof(f)) {
    buffer[0]=0;
    fgets(buffer,1024,f);
    if (strstr(buffer,"#")!=0) *(strstr(buffer,"#"))=0; // comments removed
    if (strlen(buffer)<10) continue;

    int ifound=sscanf(buffer," %d %d %d %c %d %d %d %d %s %d %d %d ",
		      &lineNo, &crate, &slot, &tb, &dccNo, &spigot, &fiber, &fiberchan,
		      subdet, &ieta, &iphi, &depth);
    if (ifound!=12) continue;

    HcalElectronicsId eid(fiberchan,fiber,spigot,dccNo);
    if (tb=='t' || tb=='T') eid.setHTR(crate,slot,1);
    else if (tb=='b' || tb=='B') eid.setHTR(crate,slot,0);
    else cerr << "Unknown tb = " << tb;
    
    if (!strcasecmp(subdet,"HB")) {
      hrm.setMap(eid,HcalDetId(HcalBarrel,ieta,iphi,depth));
    } else if (!strcasecmp(subdet,"HE")) {
      hrm.setMap(eid,HcalDetId(HcalEndcap,ieta,iphi,depth));
    } else if (!strcasecmp(subdet,"HF")) {
      hrm.setMap(eid,HcalDetId(HcalForward,ieta,iphi,depth));
    } else if (!strcasecmp(subdet,"HO")) {
      hrm.setMap(eid,HcalDetId(HcalOuter,ieta,iphi,depth));
    } else if (!strcasecmp(subdet,"HT")) {
      hrm.setTriggerMap(eid,HcalTrigTowerDetId(ieta,iphi));
    } else {
      cerr << lineNo << " Unknown subdet = " << subdet << endl;
      continue;
    }

  }
  return hrm;
}
  }
}
