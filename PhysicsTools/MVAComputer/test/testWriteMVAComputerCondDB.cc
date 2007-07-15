#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "PhysicsTools/MVAComputer/interface/BitSet.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

using namespace PhysicsTools::Calibration;

class testWriteMVAComputerCondDB : public edm::EDAnalyzer {
    public:
	explicit testWriteMVAComputerCondDB(const edm::ParameterSet &params);

	virtual void analyze(const edm::Event& iEvent,
	                     const edm::EventSetup& iSetup);

	virtual void endJob();

    private:
	std::string	record;
};

testWriteMVAComputerCondDB::testWriteMVAComputerCondDB(
					const edm::ParameterSet &params) :
	record(params.getUntrackedParameter<std::string>("record"))
{
}

void testWriteMVAComputerCondDB::analyze(const edm::Event& iEvent,
                                         const edm::EventSetup& iSetup)
{
}

void testWriteMVAComputerCondDB::endJob()
{
// set up some dummy calibration by hand for testing

	MVAComputerContainer *container = new MVAComputerContainer();
	MVAComputer *computer = &container->add("test");

// vars

	Variable var;
	var.name = "test";
	computer->inputSet.push_back(var);

	var.name = "normal";
	computer->inputSet.push_back(var);

	var.name = "toast";
	computer->inputSet.push_back(var);

// normalize

	ProcNormalize *norm = new ProcNormalize();

	PhysicsTools::BitSet testSet(3);
	testSet[0] = testSet[1] = true;
	norm->inputVars = convert(testSet);

	PDF pdf;
	pdf.distr.push_back(1.0);
	pdf.distr.push_back(1.5);
	pdf.distr.push_back(1.0);
	pdf.range.first = 4.0;
	pdf.range.second = 5.5;
	norm->distr.push_back(pdf);
	norm->distr.push_back(pdf);

	computer->addProcessor(norm);

// likelihood

	ProcLikelihood *lkh = new ProcLikelihood();

	testSet = PhysicsTools::BitSet(5);
	testSet[2] = true;
	lkh->inputVars = convert(testSet);

	pdf.distr.push_back(1.0);
	pdf.distr.push_back(1.5);
	pdf.distr.push_back(1.0);
	pdf.range.first = 0.0;
	pdf.range.second = 1.0;
	ProcLikelihood::SigBkg sigBkg;
	sigBkg.signal = pdf;
	pdf.distr.push_back(1.5);
	pdf.distr.push_back(1.0);
	pdf.distr.push_back(1.7);
	sigBkg.background = pdf;
	lkh->pdfs.push_back(sigBkg);

	computer->addProcessor(lkh);

// likelihood 2

	testSet = PhysicsTools::BitSet(6);
	testSet[2] = testSet[3] = true;
	lkh->inputVars = convert(testSet);
	lkh->pdfs.push_back(sigBkg);

	computer->addProcessor(lkh);

// optional

	ProcOptional *opt = new ProcOptional();

	testSet = PhysicsTools::BitSet(7);
	testSet[5] = testSet[6] = true;
	opt->inputVars = convert(testSet);

	opt->neutralPos.push_back(0.6);
	opt->neutralPos.push_back(0.7);

	computer->addProcessor(opt);

// PCA

	ProcMatrix *pca = new ProcMatrix();

	testSet = PhysicsTools::BitSet(9);
	testSet[4] = testSet[7] = testSet[8] = true;
	pca->inputVars = convert(testSet);

	pca->matrix.rows = 2;
	pca->matrix.columns = 3;
	double elements[] = { 0.2, 0.3, 0.4, 0.8, 0.7, 0.6 };
	std::copy(elements, elements + sizeof elements / sizeof elements[0],
	          std::back_inserter(pca->matrix.elements));

	computer->addProcessor(pca);

// linear

	ProcLinear *lin = new ProcLinear();

	testSet = PhysicsTools::BitSet(11);
	testSet[9] = testSet[10] = true;
	lin->inputVars = convert(testSet);

	lin->coeffs.push_back(0.3);
	lin->coeffs.push_back(0.7);
	lin->offset = 0.0;

	computer->addProcessor(lin);

// output

	computer->output = 11;

// test computer

	PhysicsTools::MVAComputer comp(computer);

	PhysicsTools::Variable::Value values[] = {
		PhysicsTools::Variable::Value("toast", 4.4),
		PhysicsTools::Variable::Value("toast", 4.5),
		PhysicsTools::Variable::Value("test", 4.6),
		PhysicsTools::Variable::Value("toast", 4.7),
		PhysicsTools::Variable::Value("test", 4.8),
		PhysicsTools::Variable::Value("normal", 4.9)
	};

	std::cout << comp.eval(values,
			       values + sizeof values / sizeof values[0])
		  << std::endl;

// write

	edm::Service<cond::service::PoolDBOutputService> dbService;
	if (!dbService.isAvailable())
		return;

	dbService->createNewIOV<MVAComputerContainer>(
		container, dbService->endOfTime(),
		"BTauGenericMVAJetTagComputerRcd");
}

// define this as a plug-in
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(testWriteMVAComputerCondDB);
