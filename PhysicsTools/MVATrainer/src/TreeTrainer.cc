#include <functional>
#include <algorithm>
#include <string>
#include <vector>

#include <TString.h>
#include <TTree.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/TreeReader.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/TreeTrainer.h"

namespace PhysicsTools {

TreeTrainer::TreeTrainer()
{
}

TreeTrainer::TreeTrainer(TTree *tree, double weight)
{
	addTree(tree, -1, weight);
}

TreeTrainer::TreeTrainer(TTree *signal, TTree *background, double weight)
{
	addTree(signal, true, weight);
	addTree(background, false, weight);
}

TreeTrainer::~TreeTrainer()
{
	reset();
}

Calibration::MVAComputer *TreeTrainer::train(const std::string &trainFile,
                                             double crossValidation,
                                             bool useXSLT)
{
	MVATrainer trainer(trainFile, useXSLT);
	trainer.setMonitoring(true);
	trainer.setCrossValidation(crossValidation);
	train(&trainer);
	return trainer.getCalibration();
}

void TreeTrainer::reset()
{
	readers.clear();
	std::for_each(weights.begin(), weights.end(),
	              std::ptr_fun(&::operator delete));
	weights.clear();
}

void TreeTrainer::addTree(TTree *tree, int target, double weight)
{
	static const bool targets[2] = { true, false };

	TreeReader reader(tree, false, weight > 0.0);

	if (target >= 0) {
		if (tree->GetBranch("__TARGET__"))
			throw cms::Exception("TreeTrainer")
				<< "__TARGET__ branch already present in file."
				<< std::endl;

		reader.addSingle(MVATrainer::kTargetId, &targets[!target]);
	}

	if (weight > 0.0) {
		double *ptr = new double(weight);
		weights.push_back(ptr);
		reader.addSingle(MVATrainer::kWeightId, ptr);
	}

	addReader(reader);
}

void TreeTrainer::addReader(const TreeReader &reader)
{
	readers.push_back(reader);
}

bool TreeTrainer::iteration(MVATrainer *trainer)
{
	Calibration::MVAComputer *calib = trainer->getTrainCalibration();   
	if (!calib)
		return true;

	MVAComputer computer(calib, true);

	for( auto && p : readers ) p.loop(&computer);

	return false;
}

void TreeTrainer::train(MVATrainer *trainer)
{
	while(!iteration(trainer));
}

} // namespace PhysicsTools
