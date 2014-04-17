// -*- C++ -*-
//
// Package:    HGCalNumberingInitialization
// Class:      HGCalNumberingInitialization
// 
/**\class HGCalNumberingInitialization HGCalNumberingInitialization.h Geometry/HGCalCommonData/interface/HGCalNumberingInitialization.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Mar 21 16:40:29 PDT 2013
// $Id: HGCalNumberingInitialization.cc,v 1.0 2013/12/24 12:47:41 sunanda Exp $
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
#include <Geometry/HGCalCommonData/interface/HGCalDDDConstants.h>
#include <Geometry/Records/interface/IdealGeometryRecord.h>

//#define DebugLog

class HGCalNumberingInitialization : public edm::ESProducer {

public:
  HGCalNumberingInitialization(const edm::ParameterSet&);
  ~HGCalNumberingInitialization();

  typedef std::auto_ptr<HGCalDDDConstants> ReturnType;

  ReturnType produce(const IdealGeometryRecord&);

private:
  HGCalDDDConstants* hgcalDDDConst_;
  std::string        name_;
};

HGCalNumberingInitialization::HGCalNumberingInitialization(const edm::ParameterSet& iConfig) : hgcalDDDConst_(0) {
  name_ = iConfig.getUntrackedParameter<std::string>("Name");
#ifdef DebugLog
  std::cout <<"constructing HGCalNumberingInitialization for " << name_ << std::endl;
#endif
  setWhatProduced(this, name_);
}


HGCalNumberingInitialization::~HGCalNumberingInitialization() {}


// ------------ method called to produce the data  ------------
HGCalNumberingInitialization::ReturnType
HGCalNumberingInitialization::produce(const IdealGeometryRecord& iRecord) {
#ifdef DebugLog
  std::cout << "in HGCalNumberingInitialization::produce" << std::endl;
#endif
  if (hgcalDDDConst_ == 0) {
    edm::ESTransientHandle<DDCompactView> pDD;
    iRecord.get(pDD);
    hgcalDDDConst_ = new HGCalDDDConstants(*pDD, name_);
  }
  return std::auto_ptr<HGCalDDDConstants> (hgcalDDDConst_) ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HGCalNumberingInitialization);

