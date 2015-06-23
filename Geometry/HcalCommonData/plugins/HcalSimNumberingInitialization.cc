// -*- C++ -*-
//
// Package:    HcalSimNumberingInitialization
// Class:      HcalSimNumberingInitialization
// 
/**\class HcalSimNumberingInitialization Geometry/HcalCommonData/plugins/HcalSimNumberingInitialization.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Dec 24 16:40:29 PDT 2013
// $Id: HcalSimNumberingInitialization.cc,v 1.0 2013/12/24 12:47:41 sunanda Exp $
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
#include <Geometry/Records/interface/HcalSimNumberingRecord.h>

//#define DebugLog

class HcalSimNumberingInitialization : public edm::ESProducer {

public:
  HcalSimNumberingInitialization(const edm::ParameterSet&);
  ~HcalSimNumberingInitialization();

  typedef std::auto_ptr<HcalDDDSimConstants> ReturnType;

  ReturnType produce(const HcalSimNumberingRecord&);

  void initializeHcalDDDSimConstants( const IdealGeometryRecord& igr);

private:
  std::string label_;
  HcalDDDSimConstants* hcalDDDConst_;
};

HcalSimNumberingInitialization::HcalSimNumberingInitialization(const edm::ParameterSet& iConfig) : hcalDDDConst_(0) {
#ifdef DebugLog
  std::cout <<"constructing HcalSimNumberingInitialization" << std::endl;
#endif
  setWhatProduced(this, dependsOn(&HcalSimNumberingInitialization::initializeHcalDDDSimConstants));
}


HcalSimNumberingInitialization::~HcalSimNumberingInitialization() {}


// ------------ method called to produce the data  ------------
HcalSimNumberingInitialization::ReturnType
HcalSimNumberingInitialization::produce(const HcalSimNumberingRecord& iRecord) {
#ifdef DebugLog
  std::cout << "in HcalSimNumberingInitialization::produce" << std::endl;
#endif
  if (hcalDDDConst_ == 0) {
    edm::LogError("HCalGeom") << "HcalSimNumberingInitialization::produceHcalDDDSimConstants has NOT been initialized!";
    throw cms::Exception("DDException") << "HcalSimNumberingInitialization::Cannot produce HcalDDDSimConstnats";
  }
   return std::auto_ptr<HcalDDDSimConstants> (hcalDDDConst_) ;
}

void HcalSimNumberingInitialization::initializeHcalDDDSimConstants(const IdealGeometryRecord& igr) {

  edm::ESTransientHandle<DDCompactView> pDD;
  igr.get(label_, pDD );
#ifdef DebugLog
  std::cout << "in HcalSimNumberingInitialization::initializeHcalDDDSimConstants" << std::endl;
#endif
  if ( hcalDDDConst_ != 0 ) {
    delete hcalDDDConst_;
  }
#ifdef DebugLog
  std::cout << "about to make my new hcalDDDConst_" << std::endl;
#endif
  hcalDDDConst_ = new HcalDDDSimConstants( *pDD );
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalSimNumberingInitialization);
