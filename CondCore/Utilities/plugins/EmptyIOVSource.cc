#include "FWCore/Framework/interface/ConfigurableInputSource.h"
#include <string>
namespace cond {
  class EmptyIOVSource : public edm::ConfigurableInputSource {
  public:
    typedef unsigned long long Time_t;
    EmptyIOVSource(edm::ParameterSet const&, edm::InputSourceDescription const&);
    ~EmptyIOVSource();
  private:
    virtual bool produce(edm::Event & e);
    virtual void setRunAndEventInfo();
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
    edm::ConfigurableInputSource(pset,desc),
    m_timeType(pset.getParameter<std::string>("timetype")),
    m_firstValid(pset.getParameter<unsigned long long>("firstValue")),
    m_lastValid(pset.getParameter<unsigned long long>("lastValue")),
    m_interval(pset.getParameter<unsigned long long>("interval")){
    m_current=m_firstValid;
    setRunAndEventInfo(); 
  }
  EmptyIOVSource::~EmptyIOVSource() {
  }
  bool EmptyIOVSource::produce( edm::Event & e ) {
    bool ok = !(m_lastValid<m_current);
    m_current += m_interval;
    return ok;
  }  
  void EmptyIOVSource::setRunAndEventInfo(){
    if(m_current<=m_lastValid){
      if( m_timeType=="runnumber" ){
	setRunNumber(m_current);
      }else if( m_timeType=="timestamp" ){
	setTime(m_current);
      }else if( m_timeType=="lumiid" ){
	edm::LuminosityBlockID l(m_current);
	setRunNumber(l.run());
	//std::cout<<"run "<<l.run()<<std::endl;
	//std::cout<<"luminosityBlock "<<l.luminosityBlock()<<std::endl;
	setLuminosityBlockNumber_t(l.luminosityBlock());
      }else{
	throw edm::Exception(edm::errors::Configuration, std::string("EmptyIOVSource::setRunAndEventInfo: ")+m_timeType+std::string("is not one of the supported types: runnumber,timestamp,lumiid") );
      }
      setEventNumber(1);
    }
  }
}//ns cond

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
using cond::EmptyIOVSource;

DEFINE_FWK_INPUT_SOURCE(EmptyIOVSource);
