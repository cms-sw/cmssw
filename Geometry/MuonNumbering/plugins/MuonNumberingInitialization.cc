// -*- C++ -*-
//
// Package:    MuonNumberingInitialization
// Class:      MuonNumberingInitialization
//
/**\class MuonNumberingInitialization MuonNumberingInitialization.h Geometry/MuonNumberingInitialization/interface/MuonNumberingInitialization.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Case
//         Created:  Thu Sep 28 16:40:29 PDT 2006
//
//

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"

class MuonNumberingInitialization : public edm::ESProducer {
public:
  MuonNumberingInitialization(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<MuonDDDConstants>;

  ReturnType produce(const MuonNumberingRecord&);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> geomToken_;
};

MuonNumberingInitialization::MuonNumberingInitialization(const edm::ParameterSet&) {
  setWhatProduced(this).setConsumes(geomToken_);
}

MuonNumberingInitialization::ReturnType MuonNumberingInitialization::produce(const MuonNumberingRecord& iRecord) {
  edm::ESTransientHandle<DDCompactView> pDD = iRecord.getTransientHandle(geomToken_);

  return std::make_unique<MuonDDDConstants>(*pDD);
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonNumberingInitialization);
