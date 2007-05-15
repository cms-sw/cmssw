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

using namespace PhysicsTools;

static void train(MVAComputer *computer)
{
	static const AtomicId idX("x");
	static const AtomicId idY("y");

	srandom(0);

	for(unsigned int i = 0; i < 100000; i++) {
		double x, y;

		x = +2.0 + 2.5 * std::log(random() * 1.0 / RAND_MAX) * cos(random() * M_2_PI);
		y = +1.0 + 2.5 * std::log(random() * 1.0 / RAND_MAX) * cos(random() * M_2_PI);

		Variable::Value sig[] = {
			Variable::Value(MVATrainer::kTargetId, true),
			Variable::Value(idX, x),
			Variable::Value(idY, y)
		};

		x = -1.0 + 2.5 * std::log(random() * 1.0 / RAND_MAX) * cos(random() * M_2_PI);
		y = -2.0 + 2.5 * std::log(random() * 1.0 / RAND_MAX) * cos(random() * M_2_PI);

		Variable::Value bkg[] = {
			Variable::Value(MVATrainer::kTargetId, false),
			Variable::Value(idX, x),
			Variable::Value(idY, y)
		};

		computer->eval(sig,
		               sig + (sizeof sig / sizeof sig[0]));
		computer->eval(bkg,
		               bkg + (sizeof bkg / sizeof bkg[0]));
	}
}

void test()
{
	MVATrainer trainer("testMVATrainer.xml");

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

	test[0].value = 2;
	test[1].value = 2;
	std::cout << computer->eval(test) << std::endl;

	test[0].value = 0.1;
	test[1].value = 0.1;
	std::cout << computer->eval(test) << std::endl;

	test[0].value = 0;
	test[1].value = 0;
	std::cout << computer->eval(test) << std::endl;

	test[0].value = -0.1;
	test[1].value = -0.1;
	std::cout << computer->eval(test) << std::endl;

	test[0].value = -2;
	test[1].value = -2;
	std::cout << computer->eval(test) << std::endl;

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
