#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <memory>

#include <boost/bind.hpp>
#include <boost/ptr_container/ptr_deque.hpp>

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "DataFormats/Common/interface/OrphanHandle.h"

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
	reader(new LHEReader(fileNames(), params.getUntrackedParameter<unsigned int>("skipEvents", 0))),
	wasMerged(false),
	lheProvenanceHelper_(edm::TypeID(typeid(LHEEventProduct)), edm::TypeID(typeid(LHERunInfoProduct))),
	phid_(),
        runPrincipal_()
{
        nextEvent();
        lheProvenanceHelper_.lheAugment(runInfo.get());
	// Initialize metadata, and save the process history ID for use every event.
	phid_ = lheProvenanceHelper_.lheInit(productRegistryUpdate());

        // These calls are not wanted, because the principals are used for putting the products.
	//produces<LHEEventProduct>();
	//produces<LHERunInfoProduct, edm::InRun>();
}

LHESource::~LHESource()
{
}

void LHESource::endJob()
{
	reader.reset();
}

void LHESource::nextEvent()
{
	if (partonLevel) {
		return;
        }

	bool newFileOpened = false;
	partonLevel = reader->next(&newFileOpened);

	if(newFileOpened) incrementFileIndex();
	if (!partonLevel) {
		return;
        }

	boost::shared_ptr<LHERunInfo> runInfoThis = partonLevel->getRunInfo();
	if (runInfoThis != runInfoLast) {
		runInfo = runInfoThis;
		runInfoLast = runInfoThis;
	}
}

// This is the only way we can now access the run principal.
boost::shared_ptr<edm::RunPrincipal>
LHESource::readRun_(boost::shared_ptr<edm::RunPrincipal> runPrincipal) {
  runAuxiliary()->setProcessHistoryID(phid_);
  runPrincipal->fillRunPrincipal();
  runPrincipal_ = runPrincipal;
  return runPrincipal;
}

boost::shared_ptr<edm::LuminosityBlockPrincipal>
LHESource::readLuminosityBlock_(boost::shared_ptr<edm::LuminosityBlockPrincipal> lumiPrincipal) {
  luminosityBlockAuxiliary()->setProcessHistoryID(phid_);
  lumiPrincipal->fillLuminosityBlockPrincipal();
  return lumiPrincipal;
}

void LHESource::beginRun(edm::Run&)
{
	if (runInfoLast) {
		runInfo = runInfoLast;

		std::auto_ptr<LHERunInfoProduct> product(
				new LHERunInfoProduct(*runInfo->getHEPRUP()));
		std::for_each(runInfo->getHeaders().begin(),
		              runInfo->getHeaders().end(),
		              boost::bind(
		              	&LHERunInfoProduct::addHeader,
		              	product.get(), _1));
		std::for_each(runInfo->getComments().begin(),
		              runInfo->getComments().end(),
		              boost::bind(&LHERunInfoProduct::addComment,
		              	product.get(), _1));

		// keep a copy around in case of merging
		runInfoProducts.push_back(new LHERunInfoProduct(*product));
		wasMerged = false;

                edm::WrapperOwningHolder rdp(new edm::Wrapper<LHERunInfoProduct>(product), edm::Wrapper<LHERunInfoProduct>::getInterface());
		runPrincipal_->put(lheProvenanceHelper_.runProductBranchDescription_, rdp);

		runInfo.reset();
	}
}

void LHESource::endRun(edm::Run&)
{
	if (!runInfoProducts.empty()) {
		std::auto_ptr<LHERunInfoProduct> product(
					runInfoProducts.pop_front().release());
                edm::WrapperOwningHolder rdp(new edm::Wrapper<LHERunInfoProduct>(product), edm::Wrapper<LHERunInfoProduct>::getInterface());
		runPrincipal_->put(lheProvenanceHelper_.runProductBranchDescription_, rdp);
	}
	runPrincipal_.reset();
}

bool LHESource::setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&)
{
	nextEvent();
	if (!partonLevel) {
		return false;
        }
        return true;
}

edm::EventPrincipal*
LHESource::readEvent_(edm::EventPrincipal& eventPrincipal) {
	assert(eventCached() || processingMode() != RunsLumisAndEvents);
	EventSourceSentry sentry(*this);
	edm::EventAuxiliary aux(eventID(), processGUID(), edm::Timestamp(presentTime()), false);
	aux.setProcessHistoryID(phid_);
	eventPrincipal.fillEventPrincipal(aux);

	std::auto_ptr<LHEEventProduct> product(
		     new LHEEventProduct(*partonLevel->getHEPEUP(),
					 partonLevel->originalXWGTUP())
		     );
	if (partonLevel->getPDF()) {
		product->setPDF(*partonLevel->getPDF());
        }		
	std::for_each(partonLevel->weights().begin(),
		      partonLevel->weights().end(),
		      boost::bind(&LHEEventProduct::addWeight,
				  product.get(), _1));
	std::for_each(partonLevel->getComments().begin(),
	              partonLevel->getComments().end(),
	              boost::bind(&LHEEventProduct::addComment,
	                          product.get(), _1));

	edm::WrapperOwningHolder edp(new edm::Wrapper<LHEEventProduct>(product), edm::Wrapper<LHEEventProduct>::getInterface());
	eventPrincipal.put(lheProvenanceHelper_.eventProductBranchDescription_, edp, lheProvenanceHelper_.eventProductProvenance_);

	if (runInfo) {
		std::auto_ptr<LHERunInfoProduct> product(
				new LHERunInfoProduct(*runInfo->getHEPRUP()));
		std::for_each(runInfo->getHeaders().begin(),
		              runInfo->getHeaders().end(),
		              boost::bind(
		              	&LHERunInfoProduct::addHeader,
		              	product.get(), _1));
		std::for_each(runInfo->getComments().begin(),
		              runInfo->getComments().end(),
		              boost::bind(&LHERunInfoProduct::addComment,
		              	product.get(), _1));

		if (!runInfoProducts.empty()) {
			runInfoProducts.front().mergeProduct(*product);
			if (!wasMerged) {
				runInfoProducts.pop_front();
				runInfoProducts.push_front(product);
				wasMerged = true;
			}
		}

		runInfo.reset();
	}

	partonLevel.reset();

	resetEventCached();
	return &eventPrincipal;
}

DEFINE_FWK_INPUT_SOURCE(LHESource);
