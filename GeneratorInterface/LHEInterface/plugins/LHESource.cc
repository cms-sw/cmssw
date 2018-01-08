#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <memory>

#include <boost/bind.hpp>

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"

#include "LHESource.h"

using namespace lhef;

LHESource::LHESource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc) :
  ProducerSourceFromFiles(params, desc, false),
  reader_(new LHEReader(fileNames(), params.getUntrackedParameter<unsigned int>("skipEvents", 0))),
  wasMerged_(false),
  lheProvenanceHelper_(edm::TypeID(typeid(LHEEventProduct)), edm::TypeID(typeid(LHERunInfoProduct)), productRegistryUpdate()),
  phid_(),
  runPrincipal_()
{
  nextEvent();
  lheProvenanceHelper_.lheAugment(runInfo_.get());
  // Initialize metadata, and save the process history ID for use every event.
  phid_ = lheProvenanceHelper_.lheInit(processHistoryRegistryForUpdate());

  // These calls are not wanted, because the principals are used for putting the products.
  //produces<LHEEventProduct>();
  //produces<LHERunInfoProduct, edm::InRun>();
}

LHESource::~LHESource()
{
}

void LHESource::endJob()
{
  reader_.reset();
}

void LHESource::nextEvent()
{
  if (partonLevel_) {
    return;
  }

  bool newFileOpened;
  do {
    newFileOpened = false;
    partonLevel_ = reader_->next(&newFileOpened);
  } while (newFileOpened && !partonLevel_);

  if (!partonLevel_) {
    return;
  }

  auto runInfoThis = partonLevel_->getRunInfo();
  if (runInfoThis != runInfoLast_) {
    runInfo_ = runInfoThis;
    runInfoLast_ = runInfoThis;
  }
  if (runInfo_) {
    std::unique_ptr<LHERunInfoProduct> product(
                                             new LHERunInfoProduct(*runInfo_->getHEPRUP()));
    std::for_each(runInfo_->getHeaders().begin(),
                  runInfo_->getHeaders().end(),
                  boost::bind(
                              &LHERunInfoProduct::addHeader,
                              product.get(), _1));
    std::for_each(runInfo_->getComments().begin(),
                  runInfo_->getComments().end(),
                  boost::bind(&LHERunInfoProduct::addComment,
                              product.get(), _1));

    if (!runInfoProducts_.empty()) {
      if (runInfoProducts_.front()->mergeProduct(*product)) {
        if (!wasMerged_) {
          auto temp = std::move(runInfoProducts_.front());
          runInfoProducts_.pop_front();
          runInfoProducts_.push_front(std::move(product));
          wasMerged_ = true;
        }
      } else {
        lheProvenanceHelper_.lheAugment(runInfo_.get());
        // Initialize metadata, and save the process history ID for use every event.
        phid_ = lheProvenanceHelper_.lheInit(processHistoryRegistryForUpdate());
        resetRunAuxiliary();
      }
    }

    runInfo_.reset();
  }
}

// This is the only way we can now access the run principal.
void
LHESource::readRun_(edm::RunPrincipal& runPrincipal) {
  runAuxiliary()->setProcessHistoryID(phid_);
  runPrincipal.fillRunPrincipal(processHistoryRegistryForUpdate());
  runPrincipal_ = &runPrincipal;
}

void
LHESource::readLuminosityBlock_(edm::LuminosityBlockPrincipal& lumiPrincipal) {
  luminosityBlockAuxiliary()->setProcessHistoryID(phid_);
  lumiPrincipal.fillLuminosityBlockPrincipal(processHistoryRegistryForUpdate());
}

void LHESource::beginRun(edm::Run&)
{
  if (runInfoLast_) {
    runInfo_ = runInfoLast_;

    std::unique_ptr<LHERunInfoProduct> product(
                                               new LHERunInfoProduct(*runInfo_->getHEPRUP()));
    std::for_each(runInfo_->getHeaders().begin(),
                  runInfo_->getHeaders().end(),
                  boost::bind(
                              &LHERunInfoProduct::addHeader,
                              product.get(), _1));
    std::for_each(runInfo_->getComments().begin(),
                  runInfo_->getComments().end(),
                  boost::bind(&LHERunInfoProduct::addComment,
                              product.get(), _1));

    // keep a copy around in case of merging
    runInfoProducts_.emplace_back(new LHERunInfoProduct(*product));
    wasMerged_ = false;

    std::unique_ptr<edm::WrapperBase> rdp(new edm::Wrapper<LHERunInfoProduct>(std::move(product)));
    runPrincipal_->put(lheProvenanceHelper_.runProductBranchDescription_, std::move(rdp));

    runInfo_.reset();
  }
}

void LHESource::endRun(edm::Run&)
{
  if (!runInfoProducts_.empty()) {
    auto product = std::move(runInfoProducts_.front());
    runInfoProducts_.pop_front();
    std::unique_ptr<edm::WrapperBase> rdp(new edm::Wrapper<LHERunInfoProduct>(std::move(product)));
    runPrincipal_->put(lheProvenanceHelper_.runProductBranchDescription_, std::move(rdp));
  }
  runPrincipal_ = nullptr;
}

bool LHESource::setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&, edm::EventAuxiliary::ExperimentType&)
{
  nextEvent();
  if (!partonLevel_) {
    // We just finished an input file. See if there is another.
    nextEvent();
    if (!partonLevel_) {
      // No more input files.
      return false;
    }
  }
  return true;
}

void
LHESource::readEvent_(edm::EventPrincipal& eventPrincipal) {
  assert(eventCached() || processingMode() != RunsLumisAndEvents);
  edm::EventAuxiliary aux(eventID(), processGUID(), edm::Timestamp(presentTime()), false);
  aux.setProcessHistoryID(phid_);
  eventPrincipal.fillEventPrincipal(aux, processHistoryRegistryForUpdate());

  std::unique_ptr<LHEEventProduct> product(
                                           new LHEEventProduct(*partonLevel_->getHEPEUP(),
                                                               partonLevel_->originalXWGTUP())
                                           );
  if (partonLevel_->getPDF()) {
    product->setPDF(*partonLevel_->getPDF());
  }
  std::for_each(partonLevel_->weights().begin(),
                partonLevel_->weights().end(),
                boost::bind(&LHEEventProduct::addWeight,
                            product.get(), _1));
  product->setScales(partonLevel_->scales());
  product->setNpLO(partonLevel_->npLO());
  product->setNpNLO(partonLevel_->npNLO());
  std::for_each(partonLevel_->getComments().begin(),
                partonLevel_->getComments().end(),
                boost::bind(&LHEEventProduct::addComment,
                            product.get(), _1));

  std::unique_ptr<edm::WrapperBase> edp(new edm::Wrapper<LHEEventProduct>(std::move(product)));
  eventPrincipal.put(lheProvenanceHelper_.eventProductBranchDescription_, std::move(edp), lheProvenanceHelper_.eventProductProvenance_);

  partonLevel_.reset();

  resetEventCached();
}

std::shared_ptr<edm::RunAuxiliary>
LHESource::readRunAuxiliary_() {
  edm::Timestamp ts = edm::Timestamp(presentTime());
  resetNewRun();
  auto aux = std::make_shared<edm::RunAuxiliary>(eventID().run(), ts, edm::Timestamp::invalidTimestamp());
  aux->setProcessHistoryID(phid_);
  return aux;
}

std::shared_ptr<edm::LuminosityBlockAuxiliary>
LHESource::readLuminosityBlockAuxiliary_() {
  if (processingMode() == Runs) return std::shared_ptr<edm::LuminosityBlockAuxiliary>();
  edm::Timestamp ts = edm::Timestamp(presentTime());
  resetNewLumi();
  auto aux = std::make_shared<edm::LuminosityBlockAuxiliary>(eventID().run(), eventID().luminosityBlock(),
                                                             ts, edm::Timestamp::invalidTimestamp());
  aux->setProcessHistoryID(phid_);
  return aux;
}

DEFINE_FWK_INPUT_SOURCE(LHESource);
