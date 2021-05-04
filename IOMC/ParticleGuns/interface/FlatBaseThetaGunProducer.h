#ifndef FlatBaseThetaGunProducer_H
#define FlatBaseThetaGunProducer_H

#include <string>

#include "HepPDT/defs.h"
#include "HepPDT/TableBuilder.hh"
#include "HepPDT/ParticleDataTable.hh"

#include "HepMC/GenEvent.h"

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include <memory>

namespace edm {

  class FlatBaseThetaGunProducer : public one::EDProducer<one::WatchRuns, EndRunProducer> {
  public:
    FlatBaseThetaGunProducer(const ParameterSet&);
    ~FlatBaseThetaGunProducer() override;
    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void endRun(const edm::Run& r, const edm::EventSetup&) override;
    void endRunProduce(edm::Run& r, const edm::EventSetup&) override;

  private:
  protected:
    // non-virtuals ! this and only way !
    //
    // data members

    // gun particle(s) characteristics
    std::vector<int> fPartIDs;
    double fMinTheta;
    double fMaxTheta;
    double fMinPhi;
    double fMaxPhi;

    // the event format itself
    HepMC::GenEvent* fEvt;

    // HepMC/HepPDT related things
    // (for particle/event construction)
    ESHandle<HepPDT::ParticleDataTable> fPDGTable;

    int fVerbosity;

    bool fAddAntiParticle;
  };
}  // namespace edm

#endif
