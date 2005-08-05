#ifndef HCALPEDESTALSRETRIEVER_H
#define HCALPEDESTALSRETRIEVER_H
// system include files
#include <string>
#include "DataSvc/Ref.h"
#include "DataSvc/IDataSvc.h"
#include "FileCatalog/IFileCatalog.h"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondCore/IOVService/interface/IOV.h"

// forward declarations
class HCalPedestalsRcd;

namespace edm{
  class ParameterSet;
};

class HcalPedestalsRetriever : public edm::eventsetup::ESProducer, 
			      public edm::eventsetup::EventSetupRecordIntervalFinder
{
    
public:
  HcalPedestalsRetriever(const edm::ParameterSet&  pset);
  virtual ~HcalPedestalsRetriever();
    
  // ---------- member functions ---------------------------
  const Pedestals* produce( const HCalPedestalsRcd& );
protected:
  //overriding from ContextRecordIntervalFinder
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue& , 
			       edm::ValidityInterval& ) ;
private:
  HcalPedestalsRetriever( const HcalPedestalsRetriever& ); // stop default
  const  HcalPedestalsRetriever& operator=( const HcalPedestalsRetriever& ); // stop default
  // ---------- member data --------------------------------
  std::string iovAToken_;
  std::string pedCid_;
  pool::Ref<cond::IOV> iovped_;
  pool::Ref<Pedestals> peds_;
  std::auto_ptr<pool::IFileCatalog> cat_;
  std::auto_ptr<pool::IDataSvc> svc_;
};
#endif 
