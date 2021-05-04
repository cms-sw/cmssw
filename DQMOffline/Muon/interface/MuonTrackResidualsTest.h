#ifndef MuonTrackResidualsTest_H
#define MuonTrackResidualsTest_H

/** \class MuonTrackResidualsTest
 * *
 *  DQMOffline Test Client
 *       check the residuals of the track parameters comparing STA/tracker only/global muons
 *
 *  \author  G. Mila - INFN Torino
 *   
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <map>

class MuonTrackResidualsTest : public DQMEDHarvester {
public:
  /// Constructor
  MuonTrackResidualsTest(const edm::ParameterSet& ps);

  /// Destructor
  ~MuonTrackResidualsTest() override{};

protected:
  void dqmEndRun(DQMStore::IBooker&, DQMStore::IGetter&, edm::Run const&, edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override{};

private:
  // Switch for verbosity
  std::string metname;
  edm::ParameterSet parameters;

  // source residuals histograms
  int prescaleFactor;
  std::string GaussianCriterionName;
  std::string MeanCriterionName;
  std::string SigmaCriterionName;

  std::map<std::string, std::vector<std::string> > histoNames;

  // test histograms
  std::map<std::string, MonitorElement*> MeanHistos;
  std::map<std::string, MonitorElement*> SigmaHistos;
};

#endif
