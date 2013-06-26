#ifndef PhysicsTools_MVATrainer_TreeTrainer_h
#define PhysicsTools_MVATrainer_TreeTrainer_h

#include <string>
#include <vector>
#include <map>

#include <TTree.h>

#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/TreeReader.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"

namespace PhysicsTools {

class TreeTrainer {
    public:
	TreeTrainer();
	TreeTrainer(TTree *tree, double weight = -1.0);
	TreeTrainer(TTree *signal, TTree *background, double weight = -1.0);
	~TreeTrainer();

	Calibration::MVAComputer *train(const std::string &trainDescription,
	                                double crossValidation = 0.0,
	                                bool useXSLT = false);

	// more precise control

	void reset();

	void addTree(TTree *tree, int target = -1, double weight = -1.0);
	void addReader(const TreeReader &reader);

	bool iteration(MVATrainer *trainer);
	void train(MVATrainer *trainer);

    private:
	std::vector<TreeReader>	readers;

	std::vector<double*>	weights;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_TreeTrainer_h
