#ifndef BaseRandomtXiGunProducer_H
#define BaseRandomtXiGunProducer_H

/** \class FlatRandomEGunProducer
 *
 * Generates single particle gun in HepMC format
 * Julia Yarba 10/2005 
 ***************************************/
#include <string>

#include "HepPDT/defs.h"
#include "HepPDT/TableBuilder.hh"
#include "HepPDT/ParticleDataTable.hh"

#include "HepMC/GenEvent.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include <memory>

namespace edm {

  class BaseRandomtXiGunProducer : public one::EDProducer<one::WatchRuns, EndRunProducer> {
  public:
    BaseRandomtXiGunProducer(const ParameterSet&);
    ~BaseRandomtXiGunProducer() override;
    void beginRun(const edm::Run& r, const edm::EventSetup&) override;
    void endRun(const edm::Run& r, const edm::EventSetup&) override;
    void endRunProduce(edm::Run& r, const edm::EventSetup&) override;

  private:
  protected:
    // non-virtuals ! this and only way !
    //
    // data members

    // gun particle(s) characteristics
    std::vector<int> fPartIDs;
    double fMinPhi;
    double fMaxPhi;
    double fpEnergy;
    double fECMS;

    // the event format itself
    HepMC::GenEvent* fEvt;

    ESHandle<HepPDT::ParticleDataTable> fPDGTable;

    int fVerbosity;

    const HepPDT::ParticleData* PData;

    bool fFireForward;
    bool fFireBackward;
    bool fLog_t;
  };
}  // namespace edm
#endif
