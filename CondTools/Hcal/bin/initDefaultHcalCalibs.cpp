#include "CondCore/DBCommon/interface/DBWriter.h"
#include "CondCore/IOVService/interface/IOV.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "SealKernel/Service.h"
#include "POOLCore/POOLContext.h"
#include "SealKernel/Context.h"
#include <string>
#include <iostream>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceHardcode.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"

bool validHcalCell (const HcalDetId& fCell) {
  if (fCell.iphi () <=0)  return false;
  int absEta = abs (fCell.ieta ());
  int phi = fCell.iphi ();
  int depth = fCell.depth ();
  HcalSubdetector det = fCell.subdet ();
  // phi ranges
  if ((absEta >= 40 && phi > 18) ||
      (absEta >= 21 && phi > 36) ||
      phi > 72)   return false;
  if (absEta <= 0)       return false;
  else if (absEta <= 14) return (depth == 1 || depth == 4) && det == HcalBarrel; 
  else if (absEta == 15) return (depth == 1 || depth == 2 || depth == 4) && det == HcalBarrel; 
  else if (absEta == 16) return depth >= 1 && depth <= 2 && det == HcalBarrel || depth == 3 && det == HcalEndcap; 
  else if (absEta == 17) return depth == 1 && det == HcalEndcap; 
  else if (absEta <= 26) return depth >= 1 && depth <= 2 && det == HcalEndcap; 
  else if (absEta <= 28) return depth >= 1 && depth <= 3 && det == HcalEndcap; 
  else if (absEta == 29) return depth >= 1 && depth <= 2 && (det == HcalEndcap || det == HcalForward); 
  else if (absEta <= 41) return depth >= 1 && depth <= 2 && det == HcalForward;
  else return false;
}



int main (int argn, char** argv){
  std::string contact ("sqlite_file:hcal_default_calib.db");
  if (argn > 1) {
    contact = std::string (argv [1]);
  }
  std::cout << " Using DB connection: " << contact << std::endl; 
  // std::string contact("oracle://cmscald/ratnikov");
  // std::string contact("sqlite_file:hcal_default_calib.db");
  pool::POOLContext::loadComponent( "SEAL/Services/MessageService" );
  pool::POOLContext::loadComponent( "POOL/Services/EnvironmentAuthenticationService" );

  cond::DBWriter w(contact);
  w.startTransaction();

  HcalPedestals* pedestals=new HcalPedestals;
  HcalPedestalWidths* pedestalWidths=new HcalPedestalWidths;
  HcalGains* gains=new HcalGains;
  HcalGainWidths* gainWidths=new HcalGainWidths;

  int counter = 0;
  HcalTopology topology;
  for (int eta = -50; eta < 50; eta++) {
    for (int phi = 0; phi < 100; phi++) {
      for (int depth = 1; depth < 5; depth++) {
	for (int det = 1; det < 5; det++) {
	  HcalDetId cell ((HcalSubdetector) det, eta, phi, depth);
	  if (topology.valid(cell)) {
	    uint32_t cellId = cell.rawId(); 
	    HcalDbServiceHardcode srv;
	    pedestals->addValue (cellId, srv.pedestals (cell));
	    pedestalWidths->addValue (cellId, srv.pedestalErrors (cell));
	    gains->addValue (cellId, srv.gains (cell));
	    gainWidths->addValue (cellId, srv.gainErrors (cell));

	    counter++;
	    std::cout << counter << "  Added channel ID " << cellId 
		      << " eta/phi/depth/det: " << eta << '/' << phi << '/' << depth << '/' << det << std::endl;
	  }
	}
      }
    }
  }
  pedestals->sort ();
  pedestalWidths->sort ();
  gains->sort ();
  gainWidths->sort ();

  std::string pedtok=w.write<HcalPedestals> (pedestals, "HcalPedestals");//pool::Ref takes the ownership of ped1
  std::string pedWtok=w.write<HcalPedestalWidths> (pedestalWidths, "HcalPedestalWidths");//pool::Ref takes the ownership of ped1
  std::string gaintok=w.write<HcalGains> (gains, "HcalGains");//pool::Ref takes the ownership of ped1
  std::string gainWtok=w.write<HcalGainWidths> (gainWidths, "HcalGainWidths");//pool::Ref takes the ownership of ped1

  cond::IOV* iov=new cond::IOV;
  // this is cludge until IOV parameters are defined unsigned consistently with IOVSyncValue
  edm::IOVSyncValue endtime = edm::IOVSyncValue (edm::EventID(0x7FFFFFFF, 0x7FFFFFFF), edm::Timestamp::endOfTime());
  iov->iov.insert (std::make_pair (endtime.eventID().run(), pedtok));
  std::string iovToken1 = w.write<cond::IOV> (iov,"IOV");

  iov=new cond::IOV;
  iov->iov.insert (std::make_pair (endtime.eventID().run(), pedWtok));
  std::string iovToken2 = w.write<cond::IOV> (iov,"IOV");

  iov=new cond::IOV;
  iov->iov.insert (std::make_pair (endtime.eventID().run(), gaintok));
  std::string iovToken3 = w.write<cond::IOV> (iov,"IOV");

  iov=new cond::IOV;
  iov->iov.insert (std::make_pair (endtime.eventID().run(), gainWtok));
  std::string iovToken4 = w.write<cond::IOV> (iov,"IOV");

  std::cout << "\n\n==============================================" << std::endl;
  std::cout << "Pedestals token      -> " << pedtok  << std::endl;
  std::cout << "Pedestal Widths token-> " << pedWtok  << std::endl;
  std::cout << "Gains token          -> " << gaintok  << std::endl;
  std::cout << "GainWidths token     -> " << gainWtok  << std::endl;
  std::cout << "IOV tokens           -> " << iovToken1  << std::endl
	    << "                        " << iovToken2  << std::endl
	    << "                        " << iovToken3  << std::endl
	    << "                        " << iovToken4  << std::endl;
  
  w.commitTransaction();

  //register the iovToken to the metadata service
  cond::MetaData metadata_svc(contact);
  metadata_svc.addMapping("HcalPedestals_default_v1", iovToken1);  
  metadata_svc.addMapping("HcalPedestalWidths_default_v1", iovToken2);  
  metadata_svc.addMapping("HcalGains_default_v1", iovToken3);  
  metadata_svc.addMapping("HcalGainWidths_default_v1", iovToken4);  
}
