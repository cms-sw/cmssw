// -*- C++ -*-
//
// Package:    FastTimeNumberingInitialization
// Class:      FastTimeNumberingInitialization
//
/**\class FastTimeNumberingInitialization FastTimeNumberingInitialization.h
 Geometry/HGCalCommonData/interface/FastTimeNumberingInitialization.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Mar 21 16:40:29 PDT 2013
// $Id: FastTimeNumberingInitialization.cc,v 1.0 2014/04/24 12:47:41 sunanda Exp
// $
//
//

// user include files
#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/Framework/interface/ModuleFactory.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDsvalues.h>
#include <Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h>
#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include "Geometry/HGCalCommonData/interface/FastTimeParameters.h"

//#define EDM_ML_DEBUG

class FastTimeNumberingInitialization : public edm::ESProducer {
 public:
  FastTimeNumberingInitialization(const edm::ParameterSet&);
  ~FastTimeNumberingInitialization() override;

  using ReturnType = std::unique_ptr<FastTimeDDDConstants>;

  ReturnType produce(const IdealGeometryRecord&);

private:
  edm::ESGetToken<FastTimeParameters, IdealGeometryRecord> ftParToken_;
};

FastTimeNumberingInitialization::FastTimeNumberingInitialization(const edm::ParameterSet&):
  ftParToken_{setWhatProduced(this).consumes<FastTimeParameters>(edm::ESInputTag{})}
{
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") 
    << "constructing FastTimeNumberingInitialization";
#endif
}

FastTimeNumberingInitialization::~FastTimeNumberingInitialization() {}

// ------------ method called to produce the data  ------------
FastTimeNumberingInitialization::ReturnType
FastTimeNumberingInitialization::produce(const IdealGeometryRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom")
    << "in FastTimeNumberingInitialization::produce";
#endif
  const auto& pFTpar = iRecord.get(ftParToken_);
  return std::make_unique<FastTimeDDDConstants>(&pFTpar);
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(FastTimeNumberingInitialization);
