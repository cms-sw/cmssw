// Example of accessing HCAL conditions from DB directly
// F.Ratnikov (UMd)  Oct 11, 2006

#include <stdlib.h>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <fstream>

#include "CondCore/IOVService/interface/IOV.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondTools/Hcal/interface/HcalDbTool.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

typedef HcalDbTool::IOVRun IOVRun;
typedef std::map<IOVRun,std::string> IOVCollection;

int main () {
  std::string tag_pedestals = "HCAL_PEDESTALS_TB06_V2";
  std::string tag_pedestalws = "HCAL_PEDESTAL_WIDTHS_TB06_V2";
  std::string dbcon = "oracle://cmsr/CMS_ECALHCAL_H2_COND";
  ::setenv ("POOL_CATALOG", "relationalcatalog_oracle://cmsr/CMS_ECALHCAL_H2_COND", 1);

  HcalDbTool db (dbcon);

  cond::IOV iov;
  if (db.getObject (&iov, tag_pedestals)) {
    for (IOVCollection::const_iterator iovi = iov.iov.begin (); iovi != iov.iov.end (); iovi++) {
      IOVRun iovMax = iovi->first;
      std::cout << "Extracting pedestals for run " << iovMax << "..." << std::endl;
      HcalPedestals peds;
      if (db.getObject (&peds, tag_pedestals, iovMax)) {
	std::ostringstream filename;
	filename << tag_pedestals << "_" << iovMax << ".txt";
	std::ofstream outStream (filename.str().c_str());
	HcalDbASCIIIO::dumpObject (outStream, peds);
      }
      else {
	std::cerr << "printRuns-> can not find pedestals for tag " << tag_pedestals << " run " << iovMax << std::endl;
      }
      HcalPedestalWidths pedws;
      if (db.getObject (&pedws, tag_pedestalws, iovMax)) {
	std::ostringstream filename;
	filename << tag_pedestalws << "_" << iovMax << ".txt";
	std::ofstream outStream (filename.str().c_str());
	HcalDbASCIIIO::dumpObject (outStream, pedws);
      }
      else {
	std::cerr << "printRuns-> can not find pedestals for tag " << tag_pedestals << " run " << iovMax << std::endl;
      }
    }
  }
  else {
    std::cerr << "printRuns-> can not find IOV for tag " << tag_pedestals << std::endl;
  }
  return 0;
}
