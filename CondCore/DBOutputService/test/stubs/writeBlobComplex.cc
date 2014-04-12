#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Calibration/interface/BlobComplex.h"
#include "writeBlobComplex.h"

#include <iostream>

writeBlobComplex::writeBlobComplex(const edm::ParameterSet& iConfig):
  m_RecordName("BlobComplexRcd")
{
}

writeBlobComplex::~writeBlobComplex()
{
  std::cout<<"writeBlobComplex::writeBlobComplex"<<std::endl;
}

void
writeBlobComplex::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup)
{
  std::cout<<"writeBlobComplex::analyze "<<std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    std::cout<<"db service unavailable"<<std::endl;
    return;
  }
  try{
    BlobComplex* me = new BlobComplex;
    unsigned int serial = 123;
    me->fill(serial);
    std::cout<<"writeBlobComplex::about to write "<<std::endl;
    mydbservice->writeOne(me,mydbservice->currentTime(),m_RecordName);
  }catch(const std::exception& er){
    std::cout<<"caught std::exception "<<er.what()<<std::endl;
    throw er;
  }
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(writeBlobComplex);
