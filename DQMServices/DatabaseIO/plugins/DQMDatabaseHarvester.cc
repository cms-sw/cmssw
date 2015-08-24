#include "DQMServices/Examples/interface/DQMExample_Step2DB.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

//
// -------------------------------------- Constructor --------------------------------------------
//
DQMExample_Step2DB::DQMExample_Step2DB(const edm::ParameterSet& ps) : DQMDbHarvester(ps)
{
  edm::LogInfo("DQMExample_Step2DB") <<  "Constructor  DQMExample_Step2DB::DQMExample_Step2DB " << std::endl;
  s_histogramsPath      =  ps.getParameter<std::string>("histogramsPath");
  vs_histogramsPerLumi  =  ps.getParameter<std::vector <std::string> >("histogramsPerLumi");
  vs_histogramsPerRun   =  ps.getParameter<std::vector <std::string> >("histogramsPerRun");
  numMonitorName_      =  ps.getParameter<std::string>("numMonitorName");
  denMonitorName_      =  ps.getParameter<std::string>("denMonitorName");

  std::cout << "Histograms: " << std::endl;
  for (std::string histo : vs_histogramsPerRun)
  {
    std::cout << histo << std::endl;
  }
  for (std::string histo : vs_histogramsPerLumi)
  {
    std::cout << histo << std::endl;
  }
}

//
// -- Destructor
//
DQMExample_Step2DB::~DQMExample_Step2DB()
{
  edm::LogInfo("DQMExample_Step2DB") <<  "Destructor DQMExample_Step2DB::~DQMExample_Step2DB " << std::endl;
}

//
// -------------------------------------- beginJob --------------------------------------------
//
void DQMExample_Step2DB::beginJob()
{
  edm::LogInfo("DQMExample_Step2DB") <<  "DQMExample_Step2DB::beginJob " << std::endl;
  std::cout << "DB beginJob" << std::endl;
  DQMDbHarvester::beginJob();
}
//
// -------------------------------------- get and book in the endJob --------------------------------------------
//
void DQMExample_Step2DB::dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_)
{
  std::cout << "DB dqmEndJob" << std::endl;

  // create and cd into new folder
  ibooker_.setCurrentFolder("What_I_do_in_the_client/Ratio");

  //get available histograms
  MonitorElement* numerator = igetter_.get(numMonitorName_);
  MonitorElement* denominator = igetter_.get(denMonitorName_);

  if (!numerator || !denominator)
    {
      edm::LogError("DQMExample_Step2DB") <<  "MEs not found!" << std::endl;
      return;
    }


  //book new histogram
  h_ptRatio = ibooker_.book1D("ptRatio","pt ratio pf matched objects",50,0.,100.);
  h_ptRatio->setAxisTitle("pt [GeV]");

  for (int iBin=1; iBin<numerator->getNbinsX(); ++iBin)
    {
      if(denominator->getBinContent(iBin) == 0)
        h_ptRatio->setBinContent(iBin, 0.);
      else
        h_ptRatio->setBinContent(iBin, numerator->getBinContent(iBin) / denominator->getBinContent(iBin));
    }
}

//
// -------------------------------------- get in the endLumi if needed --------------------------------------------
//
void DQMExample_Step2DB::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker_, DQMStore::IGetter & igetter_, edm::LuminosityBlock const & iLumi, edm::EventSetup const& iSetup)
{
  std::cout << "\t###" << std::endl;
  std::cout << "DB dqmEndLuminosityBlock" << std::endl;
  std::cout << "\t###" << std::endl;
  edm::LogInfo("DQMExample_Step2DB") <<  "DQMExample_Step2DB::endLumi " << std::endl;

  if(histogramsPerLumi.empty())
  {
    for (std::string histogramName : vs_histogramsPerLumi)
    {
      MonitorElement* histogram = igetter_.get(s_histogramsPath+histogramName);
      if(histogram)
      {
        histogramsPerLumi.push_back(histogram);
        std::cout << "\tPer lum: " << histogram->getName() << "\t" << histogram->getLumiFlag() << std::endl;
      }
      /* discard repetitions */
      auto sortFunction = [](auto a, auto b)->bool{ return a->getName().compare(b->getName()); };
      sort( histogramsPerLumi.begin(), histogramsPerLumi.end(), sortFunction);
      auto uniqueFunction = [](auto a, auto b)->bool{ return a->getName() == b->getName(); };
      histogramsPerLumi.erase( unique( histogramsPerLumi.begin(), histogramsPerLumi.end(), uniqueFunction ), histogramsPerLumi.end() );
    }
  }

  if(histogramsPerRun.empty())
  {
    for (std::string histogramName : vs_histogramsPerRun)
    {
      valuesOfHistogram histogramValues;
      MonitorElement* histogram = igetter_.get(s_histogramsPath + histogramName);
      if(histogram)
        histogramsPerRun.push_back(std::pair<MonitorElement *, valuesOfHistogram>(histogram, histogramValues));

      /* discard repetitions */
      auto sortFunction = [](auto a, auto b)->bool{ return a.first->getName().compare(b.first->getName()); };
      sort( histogramsPerRun.begin(), histogramsPerRun.end(), sortFunction);
      auto uniqueFunction = [](auto a, auto b)->bool{ return a.first->getName() == b.first->getName(); };
      histogramsPerRun.erase( unique( histogramsPerRun.begin(), histogramsPerRun.end(), uniqueFunction ), histogramsPerRun.end() );

    }
    //Parse histograms that should be treated as run based
    //It is neccessary to gather data from every lumi, so it cannot be done in the endRun
    dqmDbRunInitialize(histogramsPerRun);
  }
  dqmDbLumiDrop(histogramsPerLumi, iLumi.luminosityBlock(), iLumi.run());
}

//
// -------------------------------------- endRun --------------------------------------------
//
void DQMExample_Step2DB::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  //edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::endRun" << std::endl;
  std::cout << "\t###" << std::endl;
  std::cout << "DB endRun" << std::endl;
  std::cout << "\t###" << std::endl;

  //DB harvester already knows which histograms should be dropped - dqmDbRunInitialize
  dqmDbRunDrop();
}

DEFINE_FWK_MODULE(DQMExample_Step2DB);
