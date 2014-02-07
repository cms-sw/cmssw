// -*- C++ -*-
//
// Package:    HcalRecNumberingInitialization
// Class:      HcalRecNumberingInitialization
// 
/**\class HcalRecNumberingInitialization HcalRecNumberingInitialization.h Geometry/HcalCommonData/interface/HcalRecNumberingInitialization.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Dec 24 16:40:29 PDT 2013
// $Id: HcalRecNumberingInitialization.cc,v 1.0 2013/12/24 12:47:41 sunanda Exp $
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
#include <Geometry/HcalCommonData/interface/HcalDDDSimConstants.h>
#include <Geometry/HcalCommonData/interface/HcalDDDRecConstants.h>
#include <Geometry/Records/interface/HcalRecNumberingRecord.h>

//#define DebugLog

class HcalRecNumberingInitialization : public edm::ESProducer {

public:
  HcalRecNumberingInitialization(const edm::ParameterSet&);
  ~HcalRecNumberingInitialization();

  typedef std::auto_ptr<HcalDDDRecConstants> ReturnType;

  ReturnType produce(const HcalRecNumberingRecord&);

private:
  HcalDDDRecConstants* hcalDDDConst_;
};

HcalRecNumberingInitialization::HcalRecNumberingInitialization(const edm::ParameterSet& iConfig) : hcalDDDConst_(0) {
#ifdef DebugLog
  std::cout <<"constructing HcalRecNumberingInitialization" << std::endl;
#endif
  setWhatProduced(this);
}


HcalRecNumberingInitialization::~HcalRecNumberingInitialization() {}


// ------------ method called to produce the data  ------------
HcalRecNumberingInitialization::ReturnType
HcalRecNumberingInitialization::produce(const HcalRecNumberingRecord& iRecord) {
#ifdef DebugLog
  std::cout << "in HcalRecNumberingInitialization::produce" << std::endl;
#endif
  if (hcalDDDConst_ == 0) {
    edm::ESTransientHandle<DDCompactView> pDD;
    iRecord.getRecord<IdealGeometryRecord>().get(pDD);
    edm::ESHandle<HcalDDDSimConstants>    hdc;
    iRecord.getRecord<HcalSimNumberingRecord>().get(hdc);
    hcalDDDConst_ = new HcalDDDRecConstants(*pDD, *hdc);
  }
  return std::auto_ptr<HcalDDDRecConstants> (hcalDDDConst_) ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalRecNumberingInitialization);
