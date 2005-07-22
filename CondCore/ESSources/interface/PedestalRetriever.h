#ifndef PEDESTALRETRIEVER_H
#define PEDESTALRETRIEVER_H
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
class CalibrationRecord;

namespace edm{
  class ParameterSet;
};

namespace cond{
  
  class PedestalRetriever : public edm::eventsetup::ESProducer, 
			    public edm::eventsetup::EventSetupRecordIntervalFinder
  {
    
  public:
    PedestalRetriever(const edm::ParameterSet&  pset);
    virtual ~PedestalRetriever();
    
    // ---------- member functions ---------------------------
    const Pedestals* produce( const CalibrationRecord& );
  protected:
    //overriding from ContextRecordIntervalFinder
    virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
				 const edm::Timestamp& , 
				 edm::ValidityInterval& ) ;
  private:
    PedestalRetriever( const PedestalRetriever& ); // stop default
    const  PedestalRetriever& operator=( const PedestalRetriever& ); // stop default
    // ---------- member data --------------------------------
    std::string iovAToken_;
    std::string pedCid_;
    pool::Ref<cond::IOV> iovped_;
    pool::Ref<Pedestals> peds_;
    std::auto_ptr<pool::IFileCatalog> cat_;
    std::auto_ptr<pool::IDataSvc> svc_;
  };
}
#endif 
