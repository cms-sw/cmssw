#include "CondCore/ORA/interface/QueryableVectorData.h"
#include "CondCore/ORA/interface/Selection.h"
#include "CondCore/ORA/interface/Exception.h"

ora::LoaderClient::LoaderClient():m_loader(){
}

ora::LoaderClient::LoaderClient(boost::shared_ptr<IVectorLoader>& loader):m_loader(loader){
}

ora::LoaderClient::LoaderClient( const LoaderClient& rhs ):m_loader(rhs.m_loader){
}

ora::LoaderClient& ora::LoaderClient::operator=(const ora::LoaderClient& rhs ){
  if(&rhs != this){
    m_loader= rhs.m_loader;
  }
  return *this;
}

ora::LoaderClient::~LoaderClient(){
}

boost::shared_ptr<ora::IVectorLoader> ora::LoaderClient::loader() const{
  return m_loader;
}

bool ora::LoaderClient::hasLoader() const {
  return m_loader.get()!=0;
}

void ora::LoaderClient::install(boost::shared_ptr<IVectorLoader>& loader){
  if(m_loader != loader){
    m_loader = loader;
  }
}

