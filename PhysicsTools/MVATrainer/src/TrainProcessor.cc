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

	if (!monitoring) {
		monitoring = trainer->bookMonitor(name + "_" +
		                                  (const char*)getName());
		monModule = trainer->bookMonitor(std::string("input_") +
		                                 (const char*)getName());
		booked = monitoring != 0;
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
			pair.first = monModule->book<TH1F>(name + "_bkg",
				(name + "_bkg").c_str(),
				(name + " background").c_str(), 50, 0, 0);
			pair.second = monModule->book<TH1F>(name + "_sig",
				(name + "_sig").c_str(),
				(name + " signal").c_str(), 50, 0, 0);
			monHistos.push_back(pair);
		}
	}

	trainBegin();
}

void TrainProcessor::doTrainData(const std::vector<double> *values,
                                 bool target, double weight)
{
	if (monModule) {
		for(std::vector<SigBkg>::const_iterator iter =
			monHistos.begin(); iter != monHistos.end(); ++iter) {

			TH1F *histo = target ? iter->second : iter->first;
			const std::vector<double> &vals =
					values[iter - monHistos.begin()];
			for(std::vector<double>::const_iterator value =
				vals.begin(); value != vals.end(); ++value)

				histo->Fill(*value, weight);
		}
	}

	trainData(values, target, weight);
}

void TrainProcessor::doTrainEnd()
{
	trainEnd();

	if (monModule)
		monModule = 0;
}

} // namespace PhysicsTools
