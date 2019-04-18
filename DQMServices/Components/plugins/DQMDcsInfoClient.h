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

#include <DQMServices/Core/interface/DQMEDHarvester.h>
#include <DQMServices/Core/interface/MonitorElement.h>

//
// class declaration
//

class DQMDcsInfoClient : public DQMEDHarvester {
public:
  DQMDcsInfoClient( const edm::ParameterSet& ps);
  ~DQMDcsInfoClient() override;

protected:

  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;
  void dqmEndLuminosityBlock(DQMStore::IBooker&, DQMStore::IGetter&, const edm::LuminosityBlock& l, const edm::EventSetup& c) override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:

  edm::ParameterSet parameters_;
  std::string subsystemname_;
  std::string dcsinfofolder_;


  std::vector<int> DCS;
  std::set<unsigned int> processedLS_;
  
  // ---------- member data ----------

  MonitorElement * reportSummary_;
  MonitorElement * reportSummaryMap_;
  MonitorElement * meProcessedLS_;

};

#endif
