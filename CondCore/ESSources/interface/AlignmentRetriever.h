#ifndef ALIGNMENTRETRIEVER_H
#define ALIGNMENTRETRIEVER_H
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

class AlignmentRecord;
namespace edm{
  class ParameterSet;
};

class AlignmentRetriever : public edm::eventsetup::ESProducer, 
			   public edm::eventsetup::EventSetupRecordIntervalFinder
{
public:
  AlignmentRetriever(const edm::ParameterSet&  pset);
  virtual ~ AlignmentRetriever();
  // ---------- member functions ---------------------------
  const Alignments* produce( const AlignmentRecord& );
  
protected:
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
				 const edm::Timestamp& , 
			       edm::ValidityInterval& ) ;
private:
  AlignmentRetriever( const AlignmentRetriever& ); // stop default
    
  const  AlignmentRetriever& operator=( const AlignmentRetriever& ); // stop default
  // ---------- member data --------------------------------
  std::string iovAToken_;
  std::string alignCid_;
  pool::Ref<cond::IOV> iovAlign_;
  pool::Ref<Alignments> aligns_;
  
  std::auto_ptr<pool::IFileCatalog> cat_;
  std::auto_ptr<pool::IDataSvc> svc_;
};

#endif 
