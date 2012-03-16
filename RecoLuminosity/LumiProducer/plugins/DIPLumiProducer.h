#ifndef RecoLuminosity_LumiProducer_DIPLumiProducer_h
#define RecoLuminosity_LumiProducer_DIPLumiProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

//#include <memory>
#include "boost/shared_ptr.hpp"
class DIPLumiSummaryRcd;
class DIPLumiProducer: public edm::ESProducer , public edm::EventSetupRecordIntervalFinder{
 public:
  DIPLumiProducer(const edm::ParameterSet&);
  typedef boost::shared_ptr<DIPLumiSummary> ReturnType;
  ReturnType produce(const DIPLumiSummaryRcd&);
  ~DIPLumiProducer();

  protected:
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                               const edm::IOVSyncValue&,
                               edm::ValidityInterval& );  

 private:
  edm::ParameterSet m_pset;
  std::map< unsigned int,boost::shared_ptr<DIPLumiSummary> > m_lscache;
  bool m_isNullRun; //if lumi data exist for this run
  unsigned int m_cachedrun;
  boost::shared_ptr<DIPLumiSummary> m_result;
 private:
  void fillcache(unsigned int runnumber,unsigned int startlsnum);
  void clearcache();
};
#endif
