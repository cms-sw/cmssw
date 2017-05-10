#include "DQMHistogramTest.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TF1.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

//
// -------------------------------------- Constructor --------------------------------------------
//
DQMHistogramTest::DQMHistogramTest(const edm::ParameterSet& ps)
:path_(ps.getUntrackedParameter<std::string>("path", std::string("DQMTest/DBDump")))
,histograms_(ps.getUntrackedParameter<std::vector<std::string>>("histograms", std::vector<std::string>() ))
{
  edm::LogInfo("DQMHistogramTest") <<  "Constructor  DQMHistogramTest::DQMHistogramTest " << std::endl;
}

//
// -- Destructor
//
DQMHistogramTest::~DQMHistogramTest()
{
  edm::LogInfo("DQMHistogramTest") <<  "Destructor DQMHistogramTest::~DQMHistogramTest " << std::endl;
}

//
// -------------------------------------- beginRun --------------------------------------------
//
void DQMHistogramTest::dqmBeginRun(edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("DQMHistogramTest") <<  "DQMHistogramTest::beginRun" << std::endl;
}
//
// -------------------------------------- bookHistos --------------------------------------------
//
void DQMHistogramTest::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("DQMHistogramTest") <<  "DQMHistogramTest::bookHistograms" << std::endl;

  //book at beginRun
  bookHistos(ibooker_);
}
//
// -------------------------------------- beginLuminosityBlock --------------------------------------------
//
void DQMHistogramTest::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
                                            edm::EventSetup const& context)
{
  edm::LogInfo("DQMHistogramTest") <<  "DQMHistogramTest::beginLuminosityBlock" << std::endl;
}


//
// -------------------------------------- Analyze --------------------------------------------
//
void DQMHistogramTest::analyze(edm::Event const& e, edm::EventSetup const& eSetup)
{
  edm::LogInfo("DQMHistogramTest") <<  "DQMHistogramTest::analyze" << std::endl;
  TF1 f1("f1", "gaus(2)", 0, 5);
  f1.SetParameters(5, 3, 0.2);
  for (auto const &histo : mHistograms_){
    histo->getTH1()->FillRandom("f1", 2000);
  }
}
//
// -------------------------------------- endLuminosityBlock --------------------------------------------
//
void DQMHistogramTest::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
  edm::LogInfo("DQMHistogramTest") <<  "DQMHistogramTest::endLuminosityBlock" << std::endl;
}


//
// -------------------------------------- endRun --------------------------------------------
//
void DQMHistogramTest::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  edm::LogInfo("DQMHistogramTest") <<  "DQMHistogramTest::endRun" << std::endl;
}


//
// -------------------------------------- book histograms --------------------------------------------
//
void DQMHistogramTest::bookHistos(DQMStore::IBooker & ibooker_)
{
  //ibooker_.LSbasedMode_ = true;
  ibooker_.cd();
  ibooker_.setCurrentFolder(path_);
  for (auto histo : histograms_){
    mHistograms_.push_back(ibooker_.book1D(histo, histo + "desc", 40,-0.5,   39.5 ));
  }

  ibooker_.cd();

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMHistogramTest);

