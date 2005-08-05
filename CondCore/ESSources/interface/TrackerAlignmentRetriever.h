#ifndef TRACKERALIGNMENTRETRIEVER_H
#define TRACKERALIGNMENTRETRIEVER_H
// system include files
#include <string>
#include <memory>
#include "DataSvc/Ref.h"
#include "DataSvc/IDataSvc.h"
#include "FileCatalog/IFileCatalog.h"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondCore/IOVService/interface/IOV.h"

class TrackerAlignmentRcd;
namespace edm{
  class ParameterSet;
};

class TrackerAlignmentRetriever : public edm::eventsetup::ESProducer, 
			   public edm::eventsetup::EventSetupRecordIntervalFinder
{
public:
  TrackerAlignmentRetriever(const edm::ParameterSet&  pset);
  virtual ~ TrackerAlignmentRetriever();
  // ---------- member functions ---------------------------
  const Alignments* produce( const TrackerAlignmentRcd& );
  
protected:
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
				 const edm::IOVSyncValue& , 
			       edm::ValidityInterval& ) ;
private:
  TrackerAlignmentRetriever( const TrackerAlignmentRetriever& ); // stop default
    
  const  TrackerAlignmentRetriever& operator=( const TrackerAlignmentRetriever& ); // stop default
  // ---------- member data --------------------------------
  std::string iovAToken_;
  std::string alignCid_;
  pool::Ref<cond::IOV> iovAlign_;
  pool::Ref<Alignments> aligns_;
  
  std::auto_ptr<pool::IFileCatalog> cat_;
  std::auto_ptr<pool::IDataSvc> svc_;
};

#endif 
