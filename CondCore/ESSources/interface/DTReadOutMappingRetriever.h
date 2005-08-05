#ifndef DTREADOUTMAPPINGRETRIEVER_H
#define DTREADOUTMAPPINGRETRIEVER_H
// system include files
#include <string>
#include "DataSvc/Ref.h"
#include "DataSvc/IDataSvc.h"
#include "FileCatalog/IFileCatalog.h"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/DTMapping/interface/DTReadOutMapping.h"
#include "CondCore/IOVService/interface/IOV.h"

// forward declarations
class DTReadOutMappingRcd;

namespace edm{
  class ParameterSet;
};

class DTReadOutMappingRetriever : public edm::eventsetup::ESProducer, 
				  public edm::eventsetup::EventSetupRecordIntervalFinder
{
    
public:
  DTReadOutMappingRetriever(const edm::ParameterSet&  pset);
  virtual ~DTReadOutMappingRetriever();
  
  // ---------- member functions ---------------------------
  const DTReadOutMapping* produce( const DTReadOutMappingRcd& );
protected:
  //overriding from ContextRecordIntervalFinder
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue& , 
			       edm::ValidityInterval& ) ;
private:
  DTReadOutMappingRetriever( const DTReadOutMappingRetriever& ); // stop default
  const DTReadOutMappingRetriever& operator=( const DTReadOutMappingRetriever& ); // stop default
  // ---------- member data --------------------------------
  std::string iovAToken_;
  std::string pedCid_;
  pool::Ref<cond::IOV> iovped_;
  pool::Ref<DTReadOutMapping> peds_;
  std::auto_ptr<pool::IFileCatalog> cat_;
  std::auto_ptr<pool::IDataSvc> svc_;
};
#endif 
