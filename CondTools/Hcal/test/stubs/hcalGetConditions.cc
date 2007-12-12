// Example of accessing HCAL conditions from DB directly
// F.Ratnikov (UMd)  Oct 11, 2006

#include <stdlib.h>
#include <vector>
#include <string>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondTools/Hcal/interface/HcalDbTool.h"


int main () {
  std::string tag_pedestals = "pedestals_mtcc2_v1";
  unsigned long long run = 1;

  std::string dbcon = "oracle://orcon/CMS_COND_HCAL";
  ::setenv ("POOL_CATALOG", "relationalcatalog_oracle://orcon/CMS_COND_GENERAL", 1);
  // std::string dbcon = "oracle://cms_orcoff/CMS_COND_HCAL";
  //::setenv ("POOL_CATALOG", "relationalcatalog_oracle://cms_orcoff/CMS_COND_GENERAL", 1);

  ::setenv ("CORAL_AUTH_PATH", "/afs/cern.ch/cms/DB/conddb", 1);

  HcalDbTool db (dbcon, false, true);

  HcalPedestals peds;
  if (db.getObject (&peds, tag_pedestals, run)) {
    std::vector<DetId> ids = peds.getAllChannels ();
    std::cout << "Pedestal values" << std::endl;
    for (unsigned i = 0; i < ids.size(); i++) {
      const HcalPedestal* ped = peds.getValues (ids[i]);
      std:: cout << "channel: " << HcalDetId (ids[i]) << ",  values: " 
		 << ped->getValue (0) << " / "
		 << ped->getValue (1) << " / "
		 << ped->getValue (2) << " / "
		 << ped->getValue (3) << std::endl;
    }
  }
  else {
    std::cerr << "can not get pedestals for tag " <<  tag_pedestals << ", run " << run << std::endl;
  }
  return 0;
}
