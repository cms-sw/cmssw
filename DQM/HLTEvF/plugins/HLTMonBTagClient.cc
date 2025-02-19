/** \class HLTMonBTagClient
 *
 *  DQM source for BJet HLT paths
 *
 *  $Date: 2009/10/07 15:20:00 $
 *  $Revision: 1.3 $
 *  \author Andrea Bocci, Pisa
 *
 */

#include <vector>
#include <string>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "HLTMonBTagClient.h"

HLTMonBTagClient::HLTMonBTagClient(const edm::ParameterSet & config) :
  m_runAtEndLumi(           config.getUntrackedParameter<bool>("updateLuminosityBlock", false) ),
  m_runAtEndRun(            config.getUntrackedParameter<bool>("updateRun", false) ),
  m_runAtEndJob(            config.getUntrackedParameter<bool>("updateJob", false) ),
  m_pathName(               config.getParameter<std::string>("pathName") ),
  m_monitorName(            config.getParameter<std::string>("monitorName" ) ),
  m_outputFile(             config.getUntrackedParameter<std::string>("outputFile", "HLTBJetDQM.root") ),
  m_storeROOT(              config.getUntrackedParameter<bool>("storeROOT", false) ),
  m_dbe(),
  // MonitorElement's (plots) filled by the source
  m_plotRates(0),
  m_plotL2JetsEnergy(0),
  m_plotL2JetsET(0),
  m_plotL2JetsEta(0),
  m_plotL2JetsPhi(0),
  m_plotL2JetsEtaPhi(0),
  m_plotL2JetsEtaET(0),
  m_plotL25JetsEnergy(0),
  m_plotL25JetsET(0),
  m_plotL25JetsEta(0),
  m_plotL25JetsPhi(0),
  m_plotL25JetsEtaPhi(0),
  m_plotL25JetsEtaET(0),
  m_plotL3JetsEnergy(0),
  m_plotL3JetsET(0),
  m_plotL3JetsEta(0),
  m_plotL3JetsPhi(0),
  m_plotL3JetsEtaPhi(0),
  m_plotL3JetsEtaET(0),
  // MonitorElement's (plots) filled by the client
  m_plotAbsEfficiencies(0),
  m_plotRelEfficiencies(0),
  m_plotEffL25JetsEnergy(0),
  m_plotEffL25JetsET(0),
  m_plotEffL25JetsEta(0),
  m_plotEffL25JetsPhi(0),
  m_plotEffL25JetsEtaPhi(0),
  m_plotEffL25JetsEtaET(0),
  m_plotEffL3JetsEnergy(0),
  m_plotEffL3JetsET(0),
  m_plotEffL3JetsEta(0),
  m_plotEffL3JetsPhi(0),
  m_plotEffL3JetsEtaPhi(0),
  m_plotEffL3JetsEtaET(0)
{
}

HLTMonBTagClient::~HLTMonBTagClient(void) {
}

void HLTMonBTagClient::beginJob() {
  if (not m_dbe.isAvailable())
    return;

  m_dbe->setVerbose(0);
  m_dbe->setCurrentFolder(m_monitorName + "/" + m_pathName);
  // MonitorElement's (plots) filled by the source
  m_plotRates                       = book("Rates",                  "Rates",                              6,  0.,     6);
  m_plotL2JetsEnergy                = book("L2_jet_energy",          "L2 jet energy",                    300,   0.,  300.,  "GeV");
  m_plotL2JetsET                    = book("L2_jet_eT",              "L2 jet eT",                        300,   0.,  300.,  "GeV");
  m_plotL2JetsEta                   = book("L2_jet_eta",             "L2 jet eta",                        60,  -3.0,   3.0, "#eta");
  m_plotL2JetsPhi                   = book("L2_jet_phi",             "L2 jet phi",                        64,  -3.2,   3.2, "#phi");
  m_plotL2JetsEtaPhi                = book("L2_jet_eta_phi",         "L2 jet eta vs. phi",                60,  -3.0,   3.0,  64, -3.2,   3.2, "#eta", "#phi");
  m_plotL2JetsEtaET                 = book("L2_jet_eta_et",          "L2 jet eta vs. eT",                 60,  -3.0,   3.0, 300,  0.,  300.,  "#eta", "GeV");
  m_plotL25JetsEnergy               = book("L25_jet_energy",         "L2.5 jet Energy",                  300,   0.,  300.,  "GeV");
  m_plotL25JetsET                   = book("L25_jet_eT",             "L2.5 jet ET",                      300,   0.,  300.,  "GeV");
  m_plotL25JetsEta                  = book("L25_jet_eta",            "L2.5 jet eta",                      60,  -3.0,   3.0, "#eta");
  m_plotL25JetsPhi                  = book("L25_jet_phi",            "L2.5 jet phi",                      64,  -3.2,   3.2, "#phi");
  m_plotL25JetsEtaPhi               = book("L25_jet_eta_phi",        "L2.5 jet eta vs. phi",              60,  -3.0,   3.0,  64, -3.2,   3.2, "#eta", "#phi");
  m_plotL25JetsEtaET                = book("L25_jet_eta_et",         "L2.5 jet eta vs. eT",               60,  -3.0,   3.0, 300,  0.,  300.,  "#eta", "GeV");
  m_plotL3JetsEnergy                = book("L3_jet_energy",          "L3 jet Energy",                    300,   0.,  300.,  "GeV");
  m_plotL3JetsET                    = book("L3_jet_eT",              "L3 jet ET",                        300,   0.,  300.,  "GeV");
  m_plotL3JetsEta                   = book("L3_jet_eta",             "L3 jet eta",                        60,  -3.0,   3.0, "#eta");
  m_plotL3JetsPhi                   = book("L3_jet_phi",             "L3 jet phi",                        64,  -3.2,   3.2, "#phi");
  m_plotL3JetsEtaPhi                = book("L3_jet_eta_phi",         "L3 jet eta vs. phi",                60,  -3.0,   3.0,  64, -3.2,   3.2, "#eta", "#phi");
  m_plotL3JetsEtaET                 = book("L3_jet_eta_et",          "L3 jet eta vs. eT",                 60,  -3.0,   3.0, 300,  0.,  300.,  "#eta", "GeV");
  // MonitorElement's (plots) filled by the client
  m_plotAbsEfficiencies             = book("EfficienciesAbsolute",  "Absolute Efficicncies",               6,  0.,     6);
  m_plotRelEfficiencies             = book("EfficienciesRelative",  "Relative Efficicncies",               6,  0.,     6);
  m_plotEffL25JetsEnergy            = book("L25_jeteff_energy",      "L2.5 jet eff. vs. Energy",         300,   0.,  300.,  "GeV");
  m_plotEffL25JetsET                = book("L25_jeteff_eT",          "L2.5 jet eff. vs. ET",             300,   0.,  300.,  "GeV");
  m_plotEffL25JetsEta               = book("L25_jeteff_eta",         "L2.5 jet eff. vs. eta",             60,  -3.0,   3.0, "#eta");
  m_plotEffL25JetsPhi               = book("L25_jeteff_phi",         "L2.5 jet eff. vs. phi",             64,  -3.2,   3.2, "#phi");
  m_plotEffL25JetsEtaPhi            = book("L25_jeteff_eta_phi",     "L2.5 jet eff. vs. eta vs. phi",     60,  -3.0,   3.0,  64, -3.2,   3.2, "#eta", "#phi");
  m_plotEffL25JetsEtaET             = book("L25_jeteff_eta_et",      "L2.5 jet eff. vs. eta vs. eT",      60,  -3.0,   3.0, 300,  0.,  300.,  "#eta", "GeV");
  m_plotEffL3JetsEnergy             = book("L3_jeteff_energy",       "L3 jet eff. vs. Energy",           300,   0.,  300.,  "GeV");
  m_plotEffL3JetsET                 = book("L3_jeteff_eT",           "L3 jet eff. vs. ET",               300,   0.,  300.,  "GeV");
  m_plotEffL3JetsEta                = book("L3_jeteff_eta",          "L3 jet eff. vs. eta",               60,  -3.0,   3.0, "#eta");
  m_plotEffL3JetsPhi                = book("L3_jeteff_phi",          "L3 jet eff. vs. phi",               64,  -3.2,   3.2, "#phi");
  m_plotEffL3JetsEtaPhi             = book("L3_jeteff_eta_phi",      "L3 jet eff. vs. eta vs. phi",       60,  -3.0,   3.0,  64, -3.2,   3.2, "#eta", "#phi");
  m_plotEffL3JetsEtaET              = book("L3_jeteff_eta_et",       "L3 jet eff. vs. eta vs. eT",        60,  -3.0,   3.0, 300,  0.,  300.,  "#eta", "GeV");
}

void HLTMonBTagClient::endJob() {
  if (m_runAtEndJob)
    update();

  if (m_dbe.isAvailable() and m_storeROOT)
    m_dbe->save(m_outputFile);
}

void HLTMonBTagClient::beginRun(const edm::Run & run, const edm::EventSetup & setup) {
}

void HLTMonBTagClient::endRun(const edm::Run & run, const edm::EventSetup & setup) {
  if (m_runAtEndRun)
    update();
}

void HLTMonBTagClient::beginLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup) {
}

void HLTMonBTagClient::endLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup) {
  if (m_runAtEndLumi)
    update();
}

void HLTMonBTagClient::update(void) {
  if (not m_dbe.isAvailable())
    return;

  // compute efficiencies (bin 0 is underflow, bin 7 is overflow)
  TH1F * rates    = m_plotRates->getTH1F();
  TH1F * absolute = m_plotAbsEfficiencies->getTH1F(); 
  TH1F * relative = m_plotRelEfficiencies->getTH1F(); 
  float total = rates->GetBinContent(1);
  if (total) {
    relative->SetBinContent(1, 1.);
    absolute->SetBinContent(1, 1.);
    for (size_t i = 2; i < 7; i++) {
      float n = rates->GetBinContent(i);
      float d = rates->GetBinContent(i-1);
      relative->SetBinContent(i, (d > 0) ? (n / d) : 0.);
      absolute->SetBinContent(i, n / total);
    }
  }

  // update efficiency plots
  efficiency( m_plotEffL25JetsEnergy,    m_plotL25JetsEnergy,    m_plotL2JetsEnergy  );
  efficiency( m_plotEffL25JetsET,        m_plotL25JetsET,        m_plotL2JetsET      );
  efficiency( m_plotEffL25JetsEta,       m_plotL25JetsEta,       m_plotL2JetsEta     );
  efficiency( m_plotEffL25JetsPhi,       m_plotL25JetsPhi,       m_plotL2JetsPhi     );
  efficiency( m_plotEffL25JetsEtaPhi,    m_plotL25JetsEtaPhi,    m_plotL2JetsEtaPhi  );
  efficiency( m_plotEffL25JetsEtaET,     m_plotL25JetsEtaET,     m_plotL2JetsEtaET   );
  efficiency( m_plotEffL3JetsEnergy,     m_plotL3JetsEnergy,     m_plotL25JetsEnergy );
  efficiency( m_plotEffL3JetsET,         m_plotL3JetsET,         m_plotL25JetsET     );
  efficiency( m_plotEffL3JetsEta,        m_plotL3JetsEta,        m_plotL25JetsEta    );
  efficiency( m_plotEffL3JetsPhi,        m_plotL3JetsPhi,        m_plotL25JetsPhi    );
  efficiency( m_plotEffL3JetsEtaPhi,     m_plotL3JetsEtaPhi,     m_plotL25JetsEtaPhi );
  efficiency( m_plotEffL3JetsEtaET,      m_plotL3JetsEtaET,      m_plotL25JetsEtaET  );
}

void HLTMonBTagClient::analyze(const edm::Event & event, const edm::EventSetup & setup) {
}

MonitorElement * HLTMonBTagClient::book(const std::string & name, const std::string & title , int x_bins, double x_min, double x_max, const char * x_axis) {
  MonitorElement * element = m_dbe->book1D(name, title, x_bins, x_min, x_max);
  if (x_axis)
    element->setAxisTitle(x_axis, 1);
  return element;
}

MonitorElement * HLTMonBTagClient::book(const std::string & name, const std::string & title , int x_bins, double x_min, double x_max, int y_bins, double y_min, double y_max, const char * x_axis, const char * y_axis) {
  MonitorElement * element = m_dbe->book2D(name, title, x_bins, x_min, x_max, y_bins, y_min, y_max);
  if (x_axis)
    element->setAxisTitle(x_axis, 1);
  if (y_axis)
    element->setAxisTitle(y_axis, 2);
  return element;
}

void HLTMonBTagClient::efficiency(MonitorElement * target, const MonitorElement * numerator, const MonitorElement * denominator) {
  // FIXME: should check that all 3 parameters have the same type (TH1F vs. TH2F vs. TProfile, etc. ) and "shape" (number of bins, axis range, etc.)
  TH1 * t = target->getTH1();
  TH1 * n = numerator->getTH1();
  TH1 * d = denominator->getTH1();
  t->Divide(n, d, 1., 1., "B");
}

// register as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTMonBTagClient);
