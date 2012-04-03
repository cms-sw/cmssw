#ifndef RecoLuminosity_LumiProducer_DIPLumiProducer_h
#define RecoLuminosity_LumiProducer_DIPLumiProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "boost/shared_ptr.hpp"
namespace edm{
  class IOVSyncValue;
}
class DIPLuminosityRcd;
class DIPLumiSummary;
class DIPLumiDetail;
/**
   HF Luminosity numbers from DIP. Only exist if stable beam. No data available for other beam status
 **/
class DIPLumiProducer: public edm::ESProducer , public edm::EventSetupRecordIntervalFinder{
 public:
  DIPLumiProducer(const edm::ParameterSet&);
  typedef boost::shared_ptr<DIPLumiSummary> ReturnSummaryType;
  ReturnSummaryType produceSummary(const DIPLuminosityRcd&);
  typedef boost::shared_ptr<DIPLumiDetail> ReturnDetailType;
  ReturnDetailType produceDetail(const DIPLuminosityRcd&);
  ~DIPLumiProducer();

  protected:
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                               const edm::IOVSyncValue&,
                               edm::ValidityInterval& );  

 private:
  std::string m_connectStr;
  std::map< unsigned int,boost::shared_ptr<DIPLumiSummary> > m_summarycache;
  std::map< unsigned int,boost::shared_ptr<DIPLumiDetail> > m_detailcache;
  bool m_isNullRun; //if lumi data exist for this run
  unsigned int m_summarycachedrun;
  unsigned int m_detailcachedrun;
  unsigned int m_cachesize;
  boost::shared_ptr<DIPLumiSummary> m_summaryresult;
  boost::shared_ptr<DIPLumiDetail> m_detailresult;
  const edm::IOVSyncValue* m_pcurrentTime;
 private:
  void fillsummarycache(unsigned int runnumber,unsigned int startlsnum);
  void filldetailcache(unsigned int runnumber,unsigned int startlsnum);
};
#endif
