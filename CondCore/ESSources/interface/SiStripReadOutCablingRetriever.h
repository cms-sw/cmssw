#ifndef SISTRIPREADOUTCABLINGRETRIEVER_H
#define SISTRIPREADOUTCABLINGRETRIEVER_H
// system include files
#include <string>
#include "DataSvc/Ref.h"
#include "DataSvc/IDataSvc.h"
#include "FileCatalog/IFileCatalog.h"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/TrackerMapping/interface/SiStripReadOutCabling.h"
#include "CondCore/IOVService/interface/IOV.h"

// forward declarations
class SiStripReadOutCablingRcd;

namespace edm{
  class ParameterSet;
};

class SiStripReadOutCablingRetriever : public edm::eventsetup::ESProducer, 
				       public edm::eventsetup::EventSetupRecordIntervalFinder
{
    
public:
  SiStripReadOutCablingRetriever(const edm::ParameterSet&  pset);
  virtual ~SiStripReadOutCablingRetriever();
  
  // ---------- member functions ---------------------------
  const SiStripReadOutCabling* produce( const SiStripReadOutCablingRcd& );
protected:
  //overriding from ContextRecordIntervalFinder
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue& , 
			       edm::ValidityInterval& ) ;
private:
  SiStripReadOutCablingRetriever( const SiStripReadOutCablingRetriever& ); // stop default
  const  SiStripReadOutCablingRetriever& operator=( const SiStripReadOutCablingRetriever& ); // stop default
  // ---------- member data --------------------------------
  std::string iovAToken_;
  std::string pedCid_;
  pool::Ref<cond::IOV> iovped_;
  pool::Ref<SiStripReadOutCabling> peds_;
  std::auto_ptr<pool::IFileCatalog> cat_;
  std::auto_ptr<pool::IDataSvc> svc_;
};
#endif 
