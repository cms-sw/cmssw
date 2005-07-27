#ifndef ECALPEDESTALSRETRIEVER_H
#define ECALPEDESTALSRETRIEVER_H
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
class ECalPedestalsRcd;

namespace edm{
  class ParameterSet;
};

class EcalPedestalsRetriever : public edm::eventsetup::ESProducer, 
			      public edm::eventsetup::EventSetupRecordIntervalFinder
{
    
public:
  EcalPedestalsRetriever(const edm::ParameterSet&  pset);
  virtual ~EcalPedestalsRetriever();
    
  // ---------- member functions ---------------------------
  const Pedestals* produce( const ECalPedestalsRcd& );
protected:
  //overriding from ContextRecordIntervalFinder
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::Timestamp& , 
			       edm::ValidityInterval& ) ;
private:
  EcalPedestalsRetriever( const EcalPedestalsRetriever& ); // stop default
  const  EcalPedestalsRetriever& operator=( const EcalPedestalsRetriever& ); // stop default
  // ---------- member data --------------------------------
  std::string iovAToken_;
  std::string pedCid_;
  pool::Ref<cond::IOV> iovped_;
  pool::Ref<Pedestals> peds_;
  std::auto_ptr<pool::IFileCatalog> cat_;
  std::auto_ptr<pool::IDataSvc> svc_;
};
#endif 
