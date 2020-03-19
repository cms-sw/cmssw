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

LHESource::LHESource(const edm::ParameterSet& params, const edm::InputSourceDescription& desc)
    : ProducerSourceFromFiles(params, desc, false),
      reader_(new LHEReader(fileNames(), params.getUntrackedParameter<unsigned int>("skipEvents", 0))),
      lheProvenanceHelper_(
          edm::TypeID(typeid(LHEEventProduct)), edm::TypeID(typeid(LHERunInfoProduct)), productRegistryUpdate()),
      phid_() {
  nextEvent();
  lheProvenanceHelper_.lheAugment(nullptr);
  // Initialize metadata, and save the process history ID for use every event.
  phid_ = lheProvenanceHelper_.lheInit(processHistoryRegistryForUpdate());

  // These calls are not wanted, because the principals are used for putting the products.
  //produces<LHEEventProduct>();
  //produces<LHERunInfoProduct, edm::InRun>();
}

LHESource::~LHESource() {}

void LHESource::endJob() { reader_.reset(); }

void LHESource::nextEvent() {
  if (partonLevel_) {
    return;
  }

  bool newFileOpened;
  do {
    newFileOpened = false;
    partonLevel_ = reader_->next(&newFileOpened);
    if (newFileOpened) {
      incrementFileIndex();
    }
  } while (newFileOpened && !partonLevel_);

  if (!partonLevel_) {
    return;
  }

  auto runInfoThis = partonLevel_->getRunInfo();
  if (runInfoThis != runInfoLast_) {
    runInfoLast_ = runInfoThis;
    std::unique_ptr<LHERunInfoProduct> product(new LHERunInfoProduct(*runInfoThis->getHEPRUP()));
    fillRunInfoProduct(*runInfoThis, *product);

    if (runInfoProductLast_) {
      if (!runInfoProductLast_->mergeProduct(*product)) {
        //cannot be merged so must start new Run
        runInfoProductLast_ = std::move(product);
        lheProvenanceHelper_.lheAugment(runInfoThis.get());
        // Initialize metadata, and save the process history ID for use every event.
        phid_ = lheProvenanceHelper_.lheInit(processHistoryRegistryForUpdate());
        resetRunAuxiliary();
      }
    } else {
      runInfoProductLast_ = std::move(product);
    }
  }
}

void LHESource::fillRunInfoProduct(lhef::LHERunInfo const& iInfo, LHERunInfoProduct& oProduct) {
  for (auto const& h : iInfo.getHeaders()) {
    oProduct.addHeader(h);
  }
  for (auto const& c : iInfo.getComments()) {
    oProduct.addComment(c);
  }
}

void LHESource::readRun_(edm::RunPrincipal& runPrincipal) {
  runAuxiliary()->setProcessHistoryID(phid_);
  runPrincipal.fillRunPrincipal(processHistoryRegistryForUpdate());

  putRunInfoProduct(runPrincipal);
}

void LHESource::readLuminosityBlock_(edm::LuminosityBlockPrincipal& lumiPrincipal) {
  luminosityBlockAuxiliary()->setProcessHistoryID(phid_);
  lumiPrincipal.fillLuminosityBlockPrincipal(
      processHistoryRegistry().getMapped(lumiPrincipal.aux().processHistoryID()));
}

void LHESource::putRunInfoProduct(edm::RunPrincipal& iRunPrincipal) {
  if (runInfoProductLast_) {
    auto product = std::make_unique<LHERunInfoProduct>(*runInfoProductLast_);
    std::unique_ptr<edm::WrapperBase> rdp(new edm::Wrapper<LHERunInfoProduct>(std::move(product)));
    iRunPrincipal.put(lheProvenanceHelper_.runProductBranchDescription_, std::move(rdp));
  }
}

bool LHESource::setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&, edm::EventAuxiliary::ExperimentType&) {
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

void LHESource::readEvent_(edm::EventPrincipal& eventPrincipal) {
  assert(eventCached() || processingMode() != RunsLumisAndEvents);
  edm::EventAuxiliary aux(eventID(), processGUID(), edm::Timestamp(presentTime()), false);
  aux.setProcessHistoryID(phid_);
  eventPrincipal.fillEventPrincipal(aux, processHistoryRegistry().getMapped(aux.processHistoryID()));

  std::unique_ptr<LHEEventProduct> product(
      new LHEEventProduct(*partonLevel_->getHEPEUP(), partonLevel_->originalXWGTUP()));
  if (partonLevel_->getPDF()) {
    product->setPDF(*partonLevel_->getPDF());
  }
  std::for_each(partonLevel_->weights().begin(),
                partonLevel_->weights().end(),
                boost::bind(&LHEEventProduct::addWeight, product.get(), _1));
  product->setScales(partonLevel_->scales());
  product->setNpLO(partonLevel_->npLO());
  product->setNpNLO(partonLevel_->npNLO());
  std::for_each(partonLevel_->getComments().begin(),
                partonLevel_->getComments().end(),
                boost::bind(&LHEEventProduct::addComment, product.get(), _1));

  std::unique_ptr<edm::WrapperBase> edp(new edm::Wrapper<LHEEventProduct>(std::move(product)));
  eventPrincipal.put(lheProvenanceHelper_.eventProductBranchDescription_,
                     std::move(edp),
                     lheProvenanceHelper_.eventProductProvenance_);

  partonLevel_.reset();

  resetEventCached();
}

std::shared_ptr<edm::RunAuxiliary> LHESource::readRunAuxiliary_() {
  edm::Timestamp ts = edm::Timestamp(presentTime());
  resetNewRun();
  auto aux = std::make_shared<edm::RunAuxiliary>(eventID().run(), ts, edm::Timestamp::invalidTimestamp());
  aux->setProcessHistoryID(phid_);
  return aux;
}

std::shared_ptr<edm::LuminosityBlockAuxiliary> LHESource::readLuminosityBlockAuxiliary_() {
  if (processingMode() == Runs)
    return std::shared_ptr<edm::LuminosityBlockAuxiliary>();
  edm::Timestamp ts = edm::Timestamp(presentTime());
  resetNewLumi();
  auto aux = std::make_shared<edm::LuminosityBlockAuxiliary>(
      eventID().run(), eventID().luminosityBlock(), ts, edm::Timestamp::invalidTimestamp());
  aux->setProcessHistoryID(phid_);
  return aux;
}

void LHESource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("A source which reads LHE files.");
  edm::ProducerSourceFromFiles::fillDescription(desc);
  desc.addUntracked<unsigned int>("skipEvents", 0U)->setComment("Skip the first 'skipEvents' events.");
  descriptions.add("source", desc);
}

DEFINE_FWK_INPUT_SOURCE(LHESource);
