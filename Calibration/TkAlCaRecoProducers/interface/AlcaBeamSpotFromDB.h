#ifndef TkAlCaRecoProducers_AlcaBeamSpotFromDB_h
#define TkAlCaRecoProducers_AlcaBeamSpotFromDB_h

/**_________________________________________________________________
   class:   AlcaBeamSpotFromDB.h
   package: RecoVertex/TkAlCaRecoProducers



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)


________________________________________________________________**/

// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"

class AlcaBeamSpotFromDB : public edm::one::EDProducer<edm::EndLuminosityBlockProducer> {
public:
  explicit AlcaBeamSpotFromDB(const edm::ParameterSet &);
  ~AlcaBeamSpotFromDB() override;

private:
  const edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> beamSpotToken_;
  void beginJob() final;
  void endLuminosityBlockProduce(edm::LuminosityBlock &lumiSeg, const edm::EventSetup &iSetup) final;
  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) final;
  void endJob() final;
};

#endif
