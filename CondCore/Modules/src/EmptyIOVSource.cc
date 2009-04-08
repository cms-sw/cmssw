#include "CondCore/Modules/src/EmptyIOVSource.h"
#include "CondCore/DBCommon/interface/Exception.h"
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
    m_firstValid(pset.getParameter<boost::uint64_t>("firstValue")),
    m_lastValid((cond::Time_t)pset.getParameter<boost::uint64_t>("lastValue")),
    m_interval((cond::Time_t)pset.getParameter<boost::uint64_t>("interval")){
    for(cond::Time_t i=(cond::Time_t)m_firstValid; i<=m_lastValid; i+=(cond::Time_t)m_interval){
      m_iovs.insert(i);
    }
    m_iovit=m_iovs.begin();
    m_current=m_iovit;
    setRunAndEventInfo(); 
  }
  EmptyIOVSource::~EmptyIOVSource() {
  }
  bool EmptyIOVSource::produce( edm::Event & e ) {
    ++m_iovit;
    if( m_current != m_iovs.end() ){
      m_current=m_iovit;
      return true;
    }
    return false;
  }  
  void EmptyIOVSource::setRunAndEventInfo(){
    if(*m_current<=m_lastValid){
      if( m_timeType=="runnumber" ){
	setRunNumber(*m_current);
      }else if( m_timeType=="timestamp" ){
	setTime(*m_current);
      }else if( m_timeType=="lumiid" ){
	edm::LuminosityBlockID l(*m_current);
	setRunNumber(l.run());
	//std::cout<<"run "<<l.run()<<std::endl;
	//std::cout<<"luminosityBlock "<<l.luminosityBlock()<<std::endl;
	setLuminosityBlockNumber_t(l.luminosityBlock());
      }else{
	throw cond::Exception(std::string("EmptyIOVSource::setRunAndEventInfo: ")+m_timeType+std::string("is not one of the supported types: runnumber,timestamp,lumiid") );
      }
      setEventNumber(1);
    }
  }
}//ns cond
