#include <assert.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

using namespace PhysicsTools;

static double gauss()
{
	return std::sqrt(-2.0 * std::log((random() + 0.5) / RAND_MAX))
	       * std::cos(random() * (2 * M_PI / RAND_MAX));
}

static void train(MVAComputer *computer)
{
	static const AtomicId idX("x");
	static const AtomicId idY("y");

	srandom(0);

	for(unsigned int i = 0; i < 20000; i++) {
		double x, y;

		x = +2 + 2 * gauss();
		y = +1 + 2 * gauss();

		Variable::Value sig[] = {
			Variable::Value(MVATrainer::kTargetId, true),
			Variable::Value(idX, x),
			Variable::Value(idY, y)
		};

		x = -1 + 2 * gauss();
		y = -2 + 2 * gauss();

		Variable::Value bkg[] = {
			Variable::Value(MVATrainer::kTargetId, false),
			Variable::Value(idX, x),
			Variable::Value(idY, y)
		};

		computer->eval(sig, sig + 3);
		computer->eval(bkg, bkg + 3);
	}
}

void test()
{
	edm::FileInPath file("PhysicsTools/MVATrainer/test/testMVATrainer.xml");
	MVATrainer trainer(file.fullPath());

	for(;;) {
		std::auto_ptr<Calibration::MVAComputer> calib(
					trainer.getTrainCalibration());

		if (!calib.get())
			break;

		std::auto_ptr<MVAComputer> computer(
					new MVAComputer(calib.get()));

		train(computer.get());
	}

	std::vector<Variable::Value> test;
	test.push_back(Variable::Value("x", 0));
	test.push_back(Variable::Value("y", 0));

	Calibration::MVAComputer *calib = trainer.getCalibration();

	MVAComputer *computer = new MVAComputer(calib);

	test[0].setValue(2);
	test[1].setValue(2);
	std::cout << "at (+2.0, +2.0): " << computer->eval(test) << std::endl;

	test[0].setValue(0.1);
	test[1].setValue(0.1);
	std::cout << "at (+0.1, +0.1): " << computer->eval(test) << std::endl;

	test[0].setValue(0);
	test[1].setValue(0);
	std::cout << "at (+0.0, +0.0): " << computer->eval(test) << std::endl;

	test[0].setValue(-0.1);
	test[1].setValue(-0.1);
	std::cout << "at (-0.1, -0.1): " << computer->eval(test) << std::endl;

	test[0].setValue(-2);
	test[1].setValue(-2);
	std::cout << "at (-2.0, -2.0): " << computer->eval(test) << std::endl;

	delete computer;

	delete calib;
}

int main()
{
	try {
		test();
	} catch(cms::Exception e) {
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
