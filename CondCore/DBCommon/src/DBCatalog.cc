#include "CondCore/DBCommon/interface/DBCatalog.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FileCatalog/IFileCatalog.h"
//#include "FileCatalog/FCException.h"
#include "FileCatalog/IFCAction.h"
#include "FileCatalog/IFCContainer.h"
//#include "POOLCore/Exception.h"

cond::DBCatalog::DBCatalog():m_poolcatalog(0){ 
}
cond::DBCatalog::~DBCatalog(){
  if(m_poolcatalog) delete m_poolcatalog; 
}
std::string
cond::DBCatalog::logicalserviceName( const std::string& input )const{
  std::string serviceName("");
  if( input.at(0) == '/' ){
    serviceName=input.substr(1,input.find_first_of('/',1)-1);
  }
  return serviceName;
}
std::string 
cond::DBCatalog::defaultOnlineCatalogName(){
  edm::FileInPath fip("CondCore/DBCommon/data/onlineCondDBCatalog.xml");
  std::string result=std::string("xmlcatalog_file://")+fip.fullPath();
  return result;
}
std::string 
cond::DBCatalog::defaultOfflineCatalogName(){
  edm::FileInPath fip("CondCore/DBCommon/data/offlineCondDBCatalog.xml");
  std::string result=std::string("xmlcatalog_file://")+fip.fullPath();
  return result;
}
std::string 
cond::DBCatalog::defaultDevCatalogName(){
  edm::FileInPath fip("CondCore/DBCommon/data/devCondDBCatalog.xml");
  std::string result=std::string("xmlcatalog_file://")+fip.fullPath();
  return result;
}
std::string 
cond::DBCatalog::defaultLocalCatalogName(){
  edm::FileInPath fip("CondCore/DBCommon/data/localCondDBCatalog.xml");
  std::string result=std::string("xmlcatalog_file://")+fip.fullPath();
  return result;
}
pool::IFileCatalog&
cond::DBCatalog::poolCatalog(){
  if( !m_poolcatalog ){
    m_poolcatalog=new pool::IFileCatalog;
  }
  return *m_poolcatalog;
}
//bool 
//cond::DBCatalog::isLFN(const std::string& input) const{
//  if( input.at(0) == '/' ){
//    return true;
//  }
//  return false;
//}
std::string
cond::DBCatalog::getPFN(pool::IFileCatalog& poolCatalog,
			const std::string& lfn,
			bool useCache){
  std::string pf("");
  pool::FClookup l;
  poolCatalog.setAction(l);
  pool::PFNContainer mypfns( &poolCatalog, 10 );
  l.lookupPFNByLFN(lfn,mypfns);
  while(mypfns.hasNext()){
    pool::PFNEntry pentry;
    pentry=mypfns.Next();
    pf=pentry.pfname();
    if( useCache ){
      if( pf.find("frontier://") != std::string::npos ){
	return pf;
      }
    }else{
      return pf;
    }
  }
  return pf;
}
