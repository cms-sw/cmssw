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

class MuonNumberingInitialization : public edm::ESProducer
{
public:
  
  MuonNumberingInitialization( const edm::ParameterSet& );

  using ReturnType = std::unique_ptr<MuonDDDConstants>;

  ReturnType produce( const MuonNumberingRecord& );
};

MuonNumberingInitialization::MuonNumberingInitialization( const edm::ParameterSet& )
{
  setWhatProduced(this);
}

MuonNumberingInitialization::ReturnType
MuonNumberingInitialization::produce(const MuonNumberingRecord& iRecord)
{
  const IdealGeometryRecord& idealGeometryRecord = iRecord.getRecord<IdealGeometryRecord>();
  edm::ESTransientHandle<DDCompactView> pDD;
  idealGeometryRecord.get(pDD);

  return std::make_unique<MuonDDDConstants>(*pDD);
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonNumberingInitialization);
