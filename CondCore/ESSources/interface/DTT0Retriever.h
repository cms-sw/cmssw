#ifndef DTT0RETRIEVER_H
#define DTT0RETRIEVER_H
// system include files
#include <string>
#include "DataSvc/Ref.h"
#include "DataSvc/IDataSvc.h"
#include "FileCatalog/IFileCatalog.h"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/Calibration/interface/DTT0.h"
#include "CondCore/IOVService/interface/IOV.h"

// forward declarations
class DTT0Rcd;

namespace edm{
  class ParameterSet;
};

class DTT0Retriever : public edm::eventsetup::ESProducer, 
				  public edm::eventsetup::EventSetupRecordIntervalFinder
{
    
public:
  DTT0Retriever(const edm::ParameterSet&  pset);
  virtual ~DTT0Retriever();
    
  // ---------- member functions ---------------------------
  const DTT0* produce( const DTT0Rcd& );
protected:
  //overriding from ContextRecordIntervalFinder
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue& , 
			       edm::ValidityInterval& ) ;
private:
  DTT0Retriever( const DTT0Retriever& ); // stop default
  const  DTT0Retriever& operator=( const DTT0Retriever& ); // stop default
  // ---------- member data --------------------------------
  std::string iovAToken_;
  std::string pedCid_;
  pool::Ref<cond::IOV> iovped_;
  pool::Ref<DTT0> peds_;
  std::auto_ptr<pool::IFileCatalog> cat_;
  std::auto_ptr<pool::IDataSvc> svc_;
};
#endif 
