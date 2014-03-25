// -*- C++ -*-
//
// Package:    ShashlikNumberingInitialization
// Class:      ShashlikNumberingInitialization
// 
/**\class ShashlikNumberingInitialization ShashlikNumberingInitialization.h Geometry/HGCalCommonData/interface/ShashlikNumberingInitialization.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Mar 21 16:40:29 PDT 2013
// $Id: ShashlikNumberingInitialization.cc,v 1.0 2013/12/24 12:47:41 sunanda Exp $
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
#include <Geometry/HGCalCommonData/interface/ShashlikDDDConstants.h>
#include <Geometry/Records/interface/ShashlikNumberingRecord.h>

//#define DebugLog

class ShashlikNumberingInitialization : public edm::ESProducer {

public:
  ShashlikNumberingInitialization(const edm::ParameterSet&);
  ~ShashlikNumberingInitialization();

  typedef std::auto_ptr<ShashlikDDDConstants> ReturnType;

  ReturnType produce(const ShashlikNumberingRecord&);

private:
  ShashlikDDDConstants* hcalDDDConst_;
};

ShashlikNumberingInitialization::ShashlikNumberingInitialization(const edm::ParameterSet& iConfig) : hcalDDDConst_(0) {
#ifdef DebugLog
  std::cout <<"constructing ShashlikNumberingInitialization" << std::endl;
#endif
  setWhatProduced(this);
}


ShashlikNumberingInitialization::~ShashlikNumberingInitialization() {}


// ------------ method called to produce the data  ------------
ShashlikNumberingInitialization::ReturnType
ShashlikNumberingInitialization::produce(const ShashlikNumberingRecord& iRecord) {
#ifdef DebugLog
  std::cout << "in ShashlikNumberingInitialization::produce" << std::endl;
#endif
  if (hcalDDDConst_ == 0) {
    edm::ESTransientHandle<DDCompactView> pDD;
    iRecord.getRecord<IdealGeometryRecord>().get(pDD);
    hcalDDDConst_ = new ShashlikDDDConstants(*pDD);
  }
  return std::auto_ptr<ShashlikDDDConstants> (hcalDDDConst_) ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(ShashlikNumberingInitialization);
