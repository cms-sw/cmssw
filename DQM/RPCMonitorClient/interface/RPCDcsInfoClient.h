#ifndef RPCDCSINFOCLIENT_H
#define RPCDCSINFOCLIENT_H

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Run.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/ServiceRegistry/interface/Service.h>

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>

class RPCDcsInfoClient : public edm::EDAnalyzer {
public:
  RPCDcsInfoClient( const edm::ParameterSet& ps);
  ~RPCDcsInfoClient();

protected:

  void beginRun(const edm::Run& r, const edm::EventSetup& c);
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c);
  void endRun(const edm::Run& r, const edm::EventSetup& c);

private:

  std::string dcsinfofolder_;

  DQMStore * dbe_;

  std::vector<int> DCS;
};

#endif
