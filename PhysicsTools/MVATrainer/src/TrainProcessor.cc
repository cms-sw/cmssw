#include <limits>
#include <string>

#include <TH1.h>

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/TrainProcessor.h"
#include "PhysicsTools/MVAComputer/interface/ProcessRegistry.icc"

EDM_REGISTER_PLUGINFACTORY(PhysicsTools::TrainProcessor::PluginFactory,
                           "PhysicsToolsMVATrainer");

namespace PhysicsTools {

TrainProcessor::TrainProcessor(const char *name,
                               const AtomicId *id,
                               MVATrainer *trainer) :
	Source(*id), name(name), trainer(trainer), monitoring(0), monModule(0)
{
}

TrainProcessor::~TrainProcessor()
{
}

void TrainProcessor::doTrainBegin()
{
	bool booked = false;
	unsigned int nBins = 50;

	if (!monitoring) {
		const char *source = getName();
		if (source) {
			monitoring = trainer->bookMonitor(name + "_" + source);
			monModule = trainer->bookMonitor(std::string("input_") +
			                                 source);
		} else {
			monModule = trainer->bookMonitor("output");
			nBins = 400;
		}

		booked = monModule != 0;
	}

	if (booked) {
		std::vector<SourceVariable*> inputs = getInputs().get();
		for(std::vector<SourceVariable*>::const_iterator iter =
			inputs.begin(); iter != inputs.end(); ++iter) {

			SourceVariable *var = *iter;
			std::string name =
				(const char*)var->getSource()->getName()
				+ std::string("_")
				+ (const char*)var->getName();

			SigBkg pair;
			pair.entries[0] = pair.entries[1] = 0;
			pair.histo[0] = monModule->book<TH1F>(name + "_bkg",
				(name + "_bkg").c_str(),
				(name + " background").c_str(), nBins, 0, 0);
			pair.histo[1] = monModule->book<TH1F>(name + "_sig",
				(name + "_sig").c_str(),
				(name + " signal").c_str(), nBins, 0, 0);
			pair.underflow[0] = pair.underflow[1] = 0.0;
			pair.overflow[0] = pair.overflow[1] = 0.0;

			pair.sameBinning = true;	// use as default
			if (monitoring) {
				pair.min = -std::numeric_limits<double>::infinity();
				pair.max = +std::numeric_limits<double>::infinity();
			} else {
				pair.min = -99999.0;
				pair.max = +99999.0;
			}

			monHistos.push_back(pair);
		}
	}

	trainBegin();
}

void TrainProcessor::doTrainData(const std::vector<double> *values,
                                 bool target, double weight,
                                 bool train, bool test)
{
	if (monModule && test) {
		for(std::vector<SigBkg>::iterator iter = monHistos.begin();
		    iter != monHistos.end(); ++iter) {
			const std::vector<double> &vals =
					values[iter - monHistos.begin()];
			for(std::vector<double>::const_iterator value =
				vals.begin(); value != vals.end(); ++value) {

				iter->entries[target]++;

				if (*value <= iter->min) {
					iter->underflow[target] += weight;
					continue;
				} else if (*value >= iter->max) {
					iter->overflow[target] += weight;
					continue;
				}

				iter->histo[target]->Fill(*value, weight);

				if (iter->sameBinning)
					iter->histo[!target]->Fill(*value, 0);
			}
		}
	}

	if (train)
		trainData(values, target, weight);
	if (test)
		testData(values, target, weight, train);
}

void TrainProcessor::doTrainEnd()
{
	trainEnd();

	if (monModule) {
		for(std::vector<SigBkg>::const_iterator iter =
			monHistos.begin(); iter != monHistos.end(); ++iter) {

			for(unsigned int i = 0; i < 2; i++) {
				Int_t oBin = iter->histo[i]->GetNbinsX() + 1;
				iter->histo[i]->SetBinContent(0,
					iter->histo[i]->GetBinContent(0) +
					iter->underflow[i]);
				iter->histo[i]->SetBinContent(oBin,
					iter->histo[i]->GetBinContent(oBin) +
					iter->overflow[i]);
				iter->histo[i]->SetEntries(iter->entries[i]);
			}
		}

		monModule = 0;
	}
}

template<>
TrainProcessor *ProcessRegistry<TrainProcessor, AtomicId,
                                MVATrainer>::Factory::create(
                const char *name, const AtomicId *id, MVATrainer *trainer)
{
	TrainProcessor *result = ProcessRegistry::create(name, id, trainer);
	if (!result) {
		// try to load the shared library and retry
		try {
			delete TrainProcessor::PluginFactory::get()->create(
				std::string("TrainProcessor/") + name);
			result = ProcessRegistry::create(name, id, trainer);
		} catch(const cms::Exception &e) {
			// caller will have to deal with the null pointer
			// in principle this will just give a slightly more
			// descriptive error message (and will rethrow anyhow)
		}
	}
	return result;
}

} // namespace PhysicsTools
template void PhysicsTools::ProcessRegistry<PhysicsTools::TrainProcessor, PhysicsTools::AtomicId, PhysicsTools::MVATrainer>::unregisterProcess(char const*);
template void PhysicsTools::ProcessRegistry<PhysicsTools::TrainProcessor, PhysicsTools::AtomicId, PhysicsTools::MVATrainer>::registerProcess(char const*, PhysicsTools::ProcessRegistry<PhysicsTools::TrainProcessor, PhysicsTools::AtomicId, PhysicsTools::MVATrainer> const*);
