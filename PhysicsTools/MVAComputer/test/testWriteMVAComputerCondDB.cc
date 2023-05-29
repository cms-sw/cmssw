// ***************************************************************************
// *                                                                         *
// *        IMPORTANT NOTE: You would never want to do this by hand!         *
// *                                                                         *
// * This is for testing purposes only. Use PhysicsTools/MVATrainer instead. *
// *                                                                         *
// ***************************************************************************

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "PhysicsTools/MVAComputer/interface/BitSet.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

using namespace PhysicsTools::Calibration;

class testWriteMVAComputerCondDB : public edm::one::EDAnalyzer<> {
public:
  explicit testWriteMVAComputerCondDB(const edm::ParameterSet& params);

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  void endJob() override;

private:
  std::string record;
};

testWriteMVAComputerCondDB::testWriteMVAComputerCondDB(const edm::ParameterSet& params)
    : record(params.getUntrackedParameter<std::string>("record")) {}

void testWriteMVAComputerCondDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

void testWriteMVAComputerCondDB::endJob() {
  // set up some dummy calibration by hand for testing
  //
  // ***************************************************************************
  // *                                                                         *
  // *        IMPORTANT NOTE: You would never want to do this by hand!         *
  // *                                                                         *
  // * This is for testing purposes only. Use PhysicsTools/MVATrainer instead. *
  // *                                                                         *
  // ***************************************************************************

  MVAComputerContainer container;
  MVAComputer* computer = &container.add("test");

  // vars

  Variable var;
  var.name = "test";
  computer->inputSet.push_back(var);

  var.name = "normal";
  computer->inputSet.push_back(var);

  var.name = "toast";
  computer->inputSet.push_back(var);

  // normalize

  ProcNormalize norm;

  PhysicsTools::BitSet testSet(3);
  testSet[0] = testSet[1] = true;
  norm.inputVars = convert(testSet);

  HistogramF pdf(3, 4.0, 5.5);
  pdf.setBinContent(1, 1.0);
  pdf.setBinContent(2, 1.5);
  pdf.setBinContent(3, 1.0);
  norm.categoryIdx = -1;
  norm.distr.push_back(pdf);
  norm.distr.push_back(pdf);

  computer->addProcessor(&norm);

  // likelihood

  ProcLikelihood lkh;

  testSet = PhysicsTools::BitSet(5);
  testSet[2] = true;
  lkh.inputVars = convert(testSet);

  pdf = HistogramF(6, 0.0, 1.0);
  pdf.setBinContent(1, 1.0);
  pdf.setBinContent(2, 1.5);
  pdf.setBinContent(3, 1.0);
  pdf.setBinContent(4, 1.0);
  pdf.setBinContent(5, 1.5);
  pdf.setBinContent(6, 1.0);
  ProcLikelihood::SigBkg sigBkg;
  sigBkg.signal = pdf;
  pdf = HistogramF(9, 0.0, 1.0);
  pdf.setBinContent(1, 1.0);
  pdf.setBinContent(2, 1.5);
  pdf.setBinContent(3, 1.0);
  pdf.setBinContent(4, 1.0);
  pdf.setBinContent(5, 1.5);
  pdf.setBinContent(6, 1.0);
  pdf.setBinContent(7, 1.5);
  pdf.setBinContent(8, 1.0);
  pdf.setBinContent(9, 1.7);
  sigBkg.background = pdf;
  sigBkg.useSplines = true;
  lkh.categoryIdx = -1;
  lkh.neverUndefined = true;
  lkh.individual = false;
  lkh.logOutput = false;
  lkh.keepEmpty = true;
  lkh.pdfs.push_back(sigBkg);

  computer->addProcessor(&lkh);

  // likelihood 2

  testSet = PhysicsTools::BitSet(6);
  testSet[2] = testSet[3] = true;
  lkh.inputVars = convert(testSet);
  sigBkg.useSplines = true;
  lkh.pdfs.push_back(sigBkg);

  computer->addProcessor(&lkh);

  // optional

  ProcOptional opt;

  testSet = PhysicsTools::BitSet(7);
  testSet[5] = testSet[6] = true;
  opt.inputVars = convert(testSet);

  opt.neutralPos.push_back(0.6);
  opt.neutralPos.push_back(0.7);

  computer->addProcessor(&opt);

  // PCA

  ProcMatrix pca;

  testSet = PhysicsTools::BitSet(9);
  testSet[4] = testSet[7] = testSet[8] = true;
  pca.inputVars = convert(testSet);

  pca.matrix.rows = 2;
  pca.matrix.columns = 3;
  double elements[] = {0.2, 0.3, 0.4, 0.8, 0.7, 0.6};
  std::copy(elements, elements + sizeof elements / sizeof elements[0], std::back_inserter(pca.matrix.elements));

  computer->addProcessor(&pca);

  // linear

  ProcLinear lin;

  testSet = PhysicsTools::BitSet(11);
  testSet[9] = testSet[10] = true;
  lin.inputVars = convert(testSet);

  lin.coeffs.push_back(0.3);
  lin.coeffs.push_back(0.7);
  lin.offset = 0.0;

  computer->addProcessor(&lin);

  // output

  computer->output = 11;

  // test computer

  PhysicsTools::MVAComputer comp(computer);

  PhysicsTools::Variable::Value values[] = {PhysicsTools::Variable::Value("toast", 4.4),
                                            PhysicsTools::Variable::Value("toast", 4.5),
                                            PhysicsTools::Variable::Value("test", 4.6),
                                            PhysicsTools::Variable::Value("toast", 4.7),
                                            PhysicsTools::Variable::Value("test", 4.8),
                                            PhysicsTools::Variable::Value("normal", 4.9)};

  std::cout << comp.eval(values, values + sizeof values / sizeof values[0]) << std::endl;

  // write

  edm::Service<cond::service::PoolDBOutputService> dbService;
  if (!dbService.isAvailable())
    return;

  dbService->createOneIOV(container, dbService->beginOfTime(), "BTauGenericMVAJetTagComputerRcd");
}

// define this as a plug-in
DEFINE_FWK_MODULE(testWriteMVAComputerCondDB);
