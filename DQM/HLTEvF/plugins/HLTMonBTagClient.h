#ifndef DQM_HLTEvF_HLTMonBTagClient_H
#define DQM_HLTEvF_HLTMonBTagClient_H

/** \class HLTMonBTagClient
 * *
 *  DQM source for BJet HLT paths
 *
 *  $Date: 2009/10/14 07:22:39 $
 *  $Revision: 1.4 $
 *  \author Andrea Bocci, Pisa
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DQMStore;
class MonitorElement;

//
// class declaration
//

class HLTMonBTagClient : public edm::EDAnalyzer {
public:
  HLTMonBTagClient(const edm::ParameterSet & config);
  ~HLTMonBTagClient();

protected:
  void beginJob(void);
  void endJob(void);

  void beginRun(const edm::Run & run, const edm::EventSetup & setup);
  void endRun(const edm::Run & run, const edm::EventSetup & setup);

  void beginLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup);
  void endLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup);

  void analyze(const edm::Event & event, const edm::EventSetup & setup);

  void update(void);

private:
  MonitorElement * book(const std::string & name, const std::string & title, int x_bins, double x_min, double x_max, const char * x_axis = 0);
  MonitorElement * book(const std::string & name, const std::string & title, int x_bins, double x_min, double x_max, int y_bins, double y_min, double y_max, const char * x_axis = 0, const char * y_axis = 0);
  void efficiency(MonitorElement * target, const MonitorElement * numerator, const MonitorElement * denominator);

  bool m_runAtEndLumi;
  bool m_runAtEndRun;
  bool m_runAtEndJob;

  std::string m_pathName;
  std::string m_monitorName;
  std::string m_outputFile;
  bool m_storeROOT;
  edm::Service<DQMStore> m_dbe;

  // MonitorElement's (plots) filled by the source
  MonitorElement * m_plotRates;

  MonitorElement * m_plotL2JetsEnergy;
  MonitorElement * m_plotL2JetsET;
  MonitorElement * m_plotL2JetsEta;
  MonitorElement * m_plotL2JetsPhi;
  MonitorElement * m_plotL2JetsEtaPhi;
  MonitorElement * m_plotL2JetsEtaET;
  MonitorElement * m_plotL25JetsEnergy;
  MonitorElement * m_plotL25JetsET;
  MonitorElement * m_plotL25JetsEta;
  MonitorElement * m_plotL25JetsPhi;
  MonitorElement * m_plotL25JetsEtaPhi;
  MonitorElement * m_plotL25JetsEtaET;
  MonitorElement * m_plotL3JetsEnergy;
  MonitorElement * m_plotL3JetsET;
  MonitorElement * m_plotL3JetsEta;
  MonitorElement * m_plotL3JetsPhi;
  MonitorElement * m_plotL3JetsEtaPhi;
  MonitorElement * m_plotL3JetsEtaET;

  // MonitorElement's (plots) filled by the client
  MonitorElement * m_plotAbsEfficiencies;
  MonitorElement * m_plotRelEfficiencies;

  MonitorElement * m_plotEffL25JetsEnergy;
  MonitorElement * m_plotEffL25JetsET;
  MonitorElement * m_plotEffL25JetsEta;
  MonitorElement * m_plotEffL25JetsPhi;
  MonitorElement * m_plotEffL25JetsEtaPhi;
  MonitorElement * m_plotEffL25JetsEtaET;
  MonitorElement * m_plotEffL3JetsEnergy;
  MonitorElement * m_plotEffL3JetsET;
  MonitorElement * m_plotEffL3JetsEta;
  MonitorElement * m_plotEffL3JetsPhi;
  MonitorElement * m_plotEffL3JetsEtaPhi;
  MonitorElement * m_plotEffL3JetsEtaET;
};

#endif // DQM_HLTEvF_HLTMonBTagClient_H
