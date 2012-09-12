#ifndef RecoLuminosity_LumiProducer_LumiCorrectionSource_h
#define RecoLuminosity_LumiProducer_LumiCorrectionSource_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "boost/shared_ptr.hpp"
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
namespace coral{
  class ISchema;
}
namespace edm{
  class IOVSyncValue;
}
class LumiCorrectionParamRcd;
class LumiCorrectionParam;
/**
   retrieve lumi corrections and perrun parameters needed by the correction funcs 
 **/
class LumiCorrectionSource: public edm::ESProducer , public edm::EventSetupRecordIntervalFinder{
 public:
  LumiCorrectionSource(const edm::ParameterSet&);
  typedef boost::shared_ptr<LumiCorrectionParam> ReturnParamType;
  ReturnParamType produceLumiCorrectionParam(const LumiCorrectionParamRcd&);
  ~LumiCorrectionSource();
 protected:
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                               const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
 private:
  std::string translateFrontierConnect(const std::string& connectStr);
  void reloadAuth();
  const std::string servletTranslation(const std::string& servlet) const;
  std::string x2s(const XMLCh* input)const;
  XMLCh* s2x(const std::string& input)const;
  std::string toParentString(const xercesc::DOMNode &nodeToConvert)const;
 private:
  std::string m_connectStr;
  std::string m_authfilename;
  std::string m_datatag;
  std::string m_globaltag;
  std::string m_normtag;
  std::string m_siteconfpath;
  std::map< unsigned int,boost::shared_ptr<LumiCorrectionParam> > m_paramcache;
  bool m_isNullRun; //if lumi data exist for this run
  unsigned int m_paramcachedrun;
  unsigned int m_cachesize;
  boost::shared_ptr<LumiCorrectionParam> m_paramresult;
  const edm::IOVSyncValue* m_pcurrentTime;
 private:
  void fillparamcache(unsigned int runnumber);
  void parseGlobaltagForLumi(coral::ISchema& schema,const std::string& globaltag);
  float fetchIntglumi(coral::ISchema& schema,unsigned int runnumber);
};
#endif
