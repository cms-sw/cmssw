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

// user include files
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/Framework/interface/ESTransientHandle.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDsvalues.h>
#include "Geometry/HGCalCommonData/interface/FastTimeParameters.h"
#include <Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h>
#include <Geometry/Records/interface/IdealGeometryRecord.h>

//#define EDM_ML_DEBUG

class FastTimeNumberingInitialization : public edm::ESProducer {

public:
  FastTimeNumberingInitialization(const edm::ParameterSet&);
  ~FastTimeNumberingInitialization() override;

  using ReturnType = std::unique_ptr<FastTimeDDDConstants>;

  ReturnType produce(const IdealGeometryRecord&);
};

FastTimeNumberingInitialization::FastTimeNumberingInitialization(const edm::ParameterSet&) {
#ifdef EDM_ML_DEBUG
  std::cout <<"constructing FastTimeNumberingInitialization" << std::endl;
#endif
  setWhatProduced(this);
}

FastTimeNumberingInitialization::~FastTimeNumberingInitialization() {}


// ------------ method called to produce the data  ------------
FastTimeNumberingInitialization::ReturnType
FastTimeNumberingInitialization::produce(const IdealGeometryRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  std::cout << "in FastTimeNumberingInitialization::produce" << std::endl;
#endif
  edm::ESHandle<FastTimeParameters>  pFTpar;
  iRecord.get(pFTpar);
  return std::make_unique<FastTimeDDDConstants>(&(*pFTpar));
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(FastTimeNumberingInitialization);

