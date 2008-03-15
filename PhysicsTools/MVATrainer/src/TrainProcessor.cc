#include <string>

#include <TH1.h>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/TrainProcessor.h"

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
			pair.sameBinning = !monitoring;
			pair.entries[0] = pair.entries[1] = 0;
			pair.histo[0] = monModule->book<TH1F>(name + "_bkg",
				(name + "_bkg").c_str(),
				(name + " background").c_str(), nBins, 0, 0);
			pair.histo[1] = monModule->book<TH1F>(name + "_sig",
				(name + "_sig").c_str(),
				(name + " signal").c_str(), nBins, 0, 0);
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

				iter->histo[target]->Fill(*value, weight);
				iter->entries[target]++;

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

			if (iter->sameBinning) {
				iter->histo[0]->SetEntries(iter->entries[0]);
				iter->histo[1]->SetEntries(iter->entries[1]);
			}
		}

		monModule = 0;
	}
}

} // namespace PhysicsTools
