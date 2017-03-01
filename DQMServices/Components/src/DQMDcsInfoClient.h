#ifndef DQMDCSINFOCLIENT_H
#define DQMDCSINFOCLIENT_H

/*
 * \class DQMDcsInfoClient
 * \author Andreas Meyer
 *
 *
*/

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Run.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/ServiceRegistry/interface/Service.h>

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>

//
// class declaration
//

class DQMDcsInfoClient : public edm::EDAnalyzer {
public:
  DQMDcsInfoClient( const edm::ParameterSet& ps);
  ~DQMDcsInfoClient();

protected:

  void beginRun(const edm::Run& r, const edm::EventSetup& c);
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c);
  void endRun(const edm::Run& r, const edm::EventSetup& c);

private:

  edm::ParameterSet parameters_;
  std::string subsystemname_;
  std::string dcsinfofolder_;

  DQMStore * dbe_;

  std::vector<int> DCS;
  std::set<unsigned int> processedLS_;
  
  // ---------- member data ----------

  MonitorElement * reportSummary_;
  MonitorElement * reportSummaryMap_;
  MonitorElement * meProcessedLS_;

};

#endif
