/** \class HLTMonBTagIPClient
 *
 *  DQM source for BJet HLT paths
 *
 *  $Date: 2009/09/03 18:25:56 $
 *  $Revision: 1.5 $
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
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "HLTMonBTagIPClient.h"

HLTMonBTagIPClient::HLTMonBTagIPClient(const edm::ParameterSet & config) :
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
  m_plotL25TrackMultiplicity(0),
  m_plotL25TrackHits(0),
  m_plotL25TrackChi2(0),
  m_plotL25TrackEtaPhi(0),
  m_plotL25TrackEtaPT(0),
  m_plotL25IP2ndTrack2d(0),
  m_plotL25IP2ndTrack2dSig(0),
  m_plotL25IP2ndTrack3d(0),
  m_plotL25IP2ndTrack3dSig(0),
  m_plotL25IP3ndTrack2d(0),
  m_plotL25IP3ndTrack2dSig(0),
  m_plotL25IP3ndTrack3d(0),
  m_plotL25IP3ndTrack3dSig(0),
  m_plotL25Discriminator(0),
  m_plotL3JetsEnergy(0),
  m_plotL3JetsET(0),
  m_plotL3JetsEta(0),
  m_plotL3JetsPhi(0),
  m_plotL3JetsEtaPhi(0),
  m_plotL3JetsEtaET(0),
  m_plotL3TrackMultiplicity(0),
  m_plotL3TrackHits(0),
  m_plotL3TrackChi2(0),
  m_plotL3TrackEtaPhi(0),
  m_plotL3TrackEtaPT(0),
  m_plotL3IP2ndTrack2d(0),
  m_plotL3IP2ndTrack2dSig(0),
  m_plotL3IP2ndTrack3d(0),
  m_plotL3IP2ndTrack3dSig(0),
  m_plotL3IP3ndTrack2d(0),
  m_plotL3IP3ndTrack2dSig(0),
  m_plotL3IP3ndTrack3d(0),
  m_plotL3IP3ndTrack3dSig(0),
  m_plotL3Discriminator(0),
  // MonitorElement's (plots) filled by the client
  m_plotEfficiencies(0)
{
}

HLTMonBTagIPClient::~HLTMonBTagIPClient(void) {
}

void HLTMonBTagIPClient::beginJob() {
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
  m_plotL25TrackMultiplicity        = book("L25_track_multiplicity", "L2.5 pixel tracks multiplicity",    25,   0,    25);
  m_plotL25TrackHits                = book("L25_track_hits",         "L2.5 pixel tracks n. of hits",       5,   0,     5);
  m_plotL25TrackChi2                = book("L25_track_chi2",         "L2.5 pixel tracks chi2/DoF",        20,   0.,   20.,  "#chi^2/DoF");
  m_plotL25TrackEtaPhi              = book("L25_track_eta_phi",      "L2.5 pixel tracks eta vs. phi",     60,  -3.0,   3.0,  64, -3.2,   3.2, "#eta", "#phi");
  m_plotL25TrackEtaPT               = book("L25_track_eta_pt",       "L2.5 pixel tracks eta vs. pT",      60,  -3.0,   3.0,  50,  0.,   50.,  "#eta", "GeV");
  m_plotL25IP2ndTrack2d             = book("L25_IP_2ndTrack_2d",     "L2.5 2nd pixel track 2D IP",        25,  -0.05, 0.20, "cm");
  m_plotL25IP2ndTrack2dSig          = book("L25_IP_2ndTrack_2dSig",  "L2.5 2nd pixel track 2D SIP",       80, -30.,   50.);
  m_plotL25IP2ndTrack3d             = book("L25_IP_2ndTrack_3d",     "L2.5 2nd pixel track 3D IP",        60,  -1.0,  5.0,  "cm");
  m_plotL25IP2ndTrack3dSig          = book("L25_IP_2ndTrack_3dSig",  "L2.5 2nd pixel track 3D SIP",       80, -30.,   50.);
  m_plotL25IP3ndTrack2d             = book("L25_IP_3ndTrack_2d",     "L2.5 3rd pixel track 2D IP",        25,  -0.05, 0.20, "cm");
  m_plotL25IP3ndTrack2dSig          = book("L25_IP_3ndTrack_2dSig",  "L2.5 3rd pixel track 2D SIP",       80, -30.,   50.);
  m_plotL25IP3ndTrack3d             = book("L25_IP_3ndTrack_3d",     "L2.5 3rd pixel track 3D IP",        60,  -1.0,  5.0,  "cm");
  m_plotL25IP3ndTrack3dSig          = book("L25_IP_3ndTrack_3dSig",  "L2.5 3rd pixel track 3D SIP",       80, -30.,   50.);
  m_plotL25Discriminator            = book("L25_discriminator",      "L2.5 b-tag discriminator",          80, -30.,   50.);
  m_plotL3JetsEnergy                = book("L3_jet_energy",          "L3 jet Energy",                    300,   0.,  300.,  "GeV");
  m_plotL3JetsET                    = book("L3_jet_eT",              "L3 jet ET",                        300,   0.,  300.,  "GeV");
  m_plotL3JetsEta                   = book("L3_jet_eta",             "L3 jet eta",                        60,  -3.0,   3.0, "#eta");
  m_plotL3JetsPhi                   = book("L3_jet_phi",             "L3 jet phi",                        64,  -3.2,   3.2, "#phi");
  m_plotL3JetsEtaPhi                = book("L3_jet_eta_phi",         "L3 jet eta vs. phi",                60,  -3.0,   3.0,  64, -3.2,   3.2, "#eta", "#phi");
  m_plotL3JetsEtaET                 = book("L3_jet_eta_et",          "L3 jet eta vs. eT",                 60,  -3.0,   3.0, 300,  0.,  300.,  "#eta", "GeV");
  m_plotL3TrackMultiplicity         = book("L3_track_multiplicity",  "L3 tracks multiplicity",            25,   0,    25);
  m_plotL3TrackHits                 = book("L3_track_hits",          "L3 tracks n. of hits",              20,   0,    20);
  m_plotL3TrackChi2                 = book("L3_track_chi2",          "L3 tracks chi2/DoF",                20,   0.,   20.,  "#chi^2/DoF");
  m_plotL3TrackEtaPhi               = book("L3_track_eta_phi",       "L3 tracks eta vs. phi",             60,  -3.0,   3.0,  64, -3.2,   3.2, "#eta", "#phi");
  m_plotL3TrackEtaPT                = book("L3_track_eta_pt",        "L3 tracks eta vs. pT",              60,  -3.0,   3.0,  50,  0.,   50.,  "#eta", "GeV");
  m_plotL3IP2ndTrack2d              = book("L3_IP_2ndTrack_2d",      "L3 2nd track 2D IP",                25,  -0.05, 0.20, "cm");
  m_plotL3IP2ndTrack2dSig           = book("L3_IP_2ndTrack_2dSig",   "L3 2nd track 2D SIP",               80, -30.,   50.);
  m_plotL3IP2ndTrack3d              = book("L3_IP_2ndTrack_3d",      "L3 2nd track 3D IP",                60,  -1.0,  5.0,  "cm");
  m_plotL3IP2ndTrack3dSig           = book("L3_IP_2ndTrack_3dSig",   "L3 2nd track 3D SIP",               80, -30.,   50.);
  m_plotL3IP3ndTrack2d              = book("L3_IP_3ndTrack_2d",      "L3 3rd track 2D IP",                25,  -0.05, 0.20, "cm");
  m_plotL3IP3ndTrack2dSig           = book("L3_IP_3ndTrack_2dSig",   "L3 3rd track 2D SIP",               80, -30.,   50.);
  m_plotL3IP3ndTrack3d              = book("L3_IP_3ndTrack_3d",      "L3 3rd track 3D IP",                60,  -1.0,  5.0,  "cm");
  m_plotL3IP3ndTrack3dSig           = book("L3_IP_3ndTrack_3dSig",   "L3 3rd track 3D SIP",               80, -30.,   50.);
  m_plotL3Discriminator             = book("L3_discriminator",       "L3 b-tag discriminator",            80, -30.,   50.);
  // MonitorElement's (plots) filled by the client
}

void HLTMonBTagIPClient::endJob() {
  if (m_dbe.isAvailable() and m_storeROOT)
    m_dbe->save(m_outputFile);
}

void HLTMonBTagIPClient::beginRun(const edm::Run & run, const edm::EventSetup & setup) {
}

void HLTMonBTagIPClient::endRun(const edm::Run & run, const edm::EventSetup & setup) {
}

void HLTMonBTagIPClient::beginLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup) {
}

void HLTMonBTagIPClient::endLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup) {
}

void HLTMonBTagIPClient::analyze(const edm::Event & event, const edm::EventSetup & setup) {
  if (not m_dbe.isAvailable())
    return;

}

MonitorElement * HLTMonBTagIPClient::book(const std::string & name, const std::string & title , int x_bins, double x_min, double x_max, const char * x_axis) {
  MonitorElement * element = m_dbe->book1D(name, title, x_bins, x_min, x_max);
  if (x_axis)
    element->setAxisTitle(x_axis, 1);
  return element;
}

MonitorElement * HLTMonBTagIPClient::book(const std::string & name, const std::string & title , int x_bins, double x_min, double x_max, int y_bins, double y_min, double y_max, const char * x_axis, const char * y_axis) {
  MonitorElement * element = m_dbe->book2D(name, title, x_bins, x_min, x_max, y_bins, y_min, y_max);
  if (x_axis)
    element->setAxisTitle(x_axis, 1);
  if (y_axis)
    element->setAxisTitle(y_axis, 2);
  return element;
}

// register as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTMonBTagIPClient);
