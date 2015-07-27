// -*- C++ -*-
//
// Package:    FastTimeNumberingInitialization
// Class:      FastTimeNumberingInitialization
// 
/**\class FastTimeNumberingInitialization FastTimeNumberingInitialization.h Geometry/HGCalCommonData/interface/FastTimeNumberingInitialization.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Mar 21 16:40:29 PDT 2013
// $Id: FastTimeNumberingInitialization.cc,v 1.0 2014/04/24 12:47:41 sunanda Exp $
//
//

// system include files
#include <memory>
#include <boost/shared_ptr.hpp>

// user include files
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/Framework/interface/ESTransientHandle.h>

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDsvalues.h>
#include <Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h>
#include <Geometry/Records/interface/IdealGeometryRecord.h>

//#define DebugLog

class FastTimeNumberingInitialization : public edm::ESProducer {

public:
  FastTimeNumberingInitialization(const edm::ParameterSet&);
  ~FastTimeNumberingInitialization();

  typedef std::auto_ptr<FastTimeDDDConstants> ReturnType;

  ReturnType produce(const IdealGeometryRecord&);

private:
  FastTimeDDDConstants* fastTimeDDDConst_;
};

FastTimeNumberingInitialization::FastTimeNumberingInitialization(const edm::ParameterSet& iConfig) : fastTimeDDDConst_(0) {
#ifdef DebugLog
  std::cout <<"constructing FastTimeNumberingInitialization" << std::endl;
#endif
  setWhatProduced(this);
}


FastTimeNumberingInitialization::~FastTimeNumberingInitialization() {}


// ------------ method called to produce the data  ------------
FastTimeNumberingInitialization::ReturnType
FastTimeNumberingInitialization::produce(const IdealGeometryRecord& iRecord) {
#ifdef DebugLog
  std::cout << "in FastTimeNumberingInitialization::produce" << std::endl;
#endif
  if (fastTimeDDDConst_ == 0) {
    edm::ESTransientHandle<DDCompactView> pDD;
    iRecord.get(pDD);
    fastTimeDDDConst_ = new FastTimeDDDConstants(*pDD);
  }
  return std::auto_ptr<FastTimeDDDConstants> (fastTimeDDDConst_) ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(FastTimeNumberingInitialization);

