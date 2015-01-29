#ifndef DTGlobalRecoTask_H
#define DTGlobalRecoTask_H

/*
 * \file DTGlobalRecoTask.h
 *
 * \author M. Zanetti - INFN Padova
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include <iostream>
#include <fstream>
#include <vector>

class DTGlobalRecoTask: public DQMEDAnalyzer{

friend class DTMonitorModule;

public:

/// Constructor
DTGlobalRecoTask(const edm::ParameterSet& ps, const edm::EventSetup& context);

/// Destructor
virtual ~DTGlobalRecoTask();

protected:

// Book the histograms
void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);


private:

  int nevents;

  // My monitor elements

  std::ofstream logFile;

};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
