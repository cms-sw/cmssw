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
	lheProvenanceHelper_(edm::TypeID(typeid(LHEEventProduct)), edm::TypeID(typeid(LHERunInfoProduct)), productRegistryUpdate()),
	phid_(),
        runPrincipal_()
{
        nextEvent();
        lheProvenanceHelper_.lheAugment(runInfo.get());
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
	reader.reset();
}

void LHESource::nextEvent()
{
	if (partonLevel) {
		return;
        }

	bool newFileOpened = false;
	partonLevel = reader->next(&newFileOpened);

	if (!partonLevel) {
		return;
        }

	boost::shared_ptr<LHERunInfo> runInfoThis = partonLevel->getRunInfo();
	if (runInfoThis != runInfoLast) {
		runInfo = runInfoThis;
		runInfoLast = runInfoThis;
	}
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
		  if (runInfoProducts.front().mergeProduct(*product)) {
			if (!wasMerged) {
				runInfoProducts.pop_front();
				runInfoProducts.push_front(product);
				wasMerged = true;
			}
		  } else {
                    lheProvenanceHelper_.lheAugment(runInfo.get());
                    // Initialize metadata, and save the process history ID for use every event.
                    phid_ = lheProvenanceHelper_.lheInit(processHistoryRegistryForUpdate());
		    resetRunAuxiliary();
		  }
		}

		runInfo.reset();
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
	runPrincipal_ = nullptr;
}

bool LHESource::setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&)
{
	nextEvent();
	if (!partonLevel) {
                // We just finished an input file. See if there is another.
                nextEvent();
	        if (!partonLevel) {
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

	partonLevel.reset();

	resetEventCached();
}

DEFINE_FWK_INPUT_SOURCE(LHESource);
