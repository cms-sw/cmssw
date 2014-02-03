#include "FWCore/Sources/interface/ProducerSourceBase.h"
#include <string>
namespace cond {
  class EmptyIOVSource : public edm::ProducerSourceBase {
  public:
    typedef unsigned long long Time_t;
    EmptyIOVSource(edm::ParameterSet const&, edm::InputSourceDescription const&);
    ~EmptyIOVSource();
  private:
    virtual void produce(edm::Event & e) override;
    virtual bool setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& time) override;
    virtual void initialize(edm::EventID& id, edm::TimeValue_t& time, edm::TimeValue_t& interval) override;
  private:
    std::string m_timeType;
    Time_t m_firstValid;
    Time_t m_lastValid;
    Time_t m_interval;
    Time_t m_current;
  };
}

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
//#include "DataFormats/Provenance/interface/EventID.h"
//#include <iostream>
namespace cond{
  //allowed parameters: firstRun, firstTime, lastRun, lastTime, 
  //common paras: timetype,interval
  EmptyIOVSource::EmptyIOVSource(edm::ParameterSet const& pset,
				 edm::InputSourceDescription const& desc):
    edm::ProducerSourceBase(pset,desc,true),
    m_timeType(pset.getParameter<std::string>("timetype")),
    m_firstValid(pset.getParameter<unsigned long long>("firstValue")),
    m_lastValid(pset.getParameter<unsigned long long>("lastValue")),
    m_interval(pset.getParameter<unsigned long long>("interval")){
    m_current=m_firstValid;
  }
  EmptyIOVSource::~EmptyIOVSource() {
  }
  void EmptyIOVSource::produce( edm::Event & ) {
  }  
  bool EmptyIOVSource::setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& time){
    if(m_current<=m_lastValid){
      if( m_timeType=="runnumber" ){
	id = edm::EventID(m_current, id.luminosityBlock(), 1);
      }else if( m_timeType=="timestamp" ){
	time = m_current;
      }else if( m_timeType=="lumiid" ){
	edm::LuminosityBlockID l(m_current);
        id = edm::EventID(l.run(), l.luminosityBlock(), 1);
	//std::cout<<"run "<<l.run()<<std::endl;
	//std::cout<<"luminosityBlock "<<l.luminosityBlock()<<std::endl;
      }
    }
    bool ok = !(m_lastValid<m_current);
    m_current += m_interval;
    return ok;
  }
  void EmptyIOVSource::initialize(edm::EventID& id, edm::TimeValue_t& time, edm::TimeValue_t& interval){
    if( m_timeType=="runnumber" ){
      id = edm::EventID(m_firstValid, id.luminosityBlock(), 1);
      interval = 0LL; 
    }else if( m_timeType=="timestamp" ){
      time = m_firstValid;
      interval = m_interval; 
    }else if( m_timeType=="lumiid" ){
      edm::LuminosityBlockID l(m_firstValid);
      id = edm::EventID(l.run(), l.luminosityBlock(), 1);
      interval = 0LL; 
    }else{
      throw edm::Exception(edm::errors::Configuration, std::string("EmptyIOVSource::initialize: ")+m_timeType+std::string("is not one of the supported types: runnumber,timestamp,lumiid") );
    }
  }

}//ns cond

#include "FWCore/Framework/interface/InputSourceMacros.h"
using cond::EmptyIOVSource;

DEFINE_FWK_INPUT_SOURCE(EmptyIOVSource);
