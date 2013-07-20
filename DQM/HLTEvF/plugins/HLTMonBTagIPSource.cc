/** \class HLTMonBTagIPSource
 *
 *  DQM source for BJet HLT paths
 *
 *  $Date: 2010/03/15 15:31:36 $
 *  $Revision: 1.11 $
 *  \author Andrea Bocci, Pisa
 *
 */

#include <vector>
#include <string>
#include <algorithm>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTMonBTagIPSource.h"

HLTMonBTagIPSource::HLTMonBTagIPSource(const edm::ParameterSet & config) :
  m_L1Filter(       config.getParameter<edm::InputTag>("L1Filter") ),
  m_L2Filter(       config.getParameter<edm::InputTag>("L2Filter") ),
  m_L25Filter(      config.getParameter<edm::InputTag>("L25Filter") ),
  m_L3Filter(       config.getParameter<edm::InputTag>("L3Filter") ),
  m_L2Jets(         config.getParameter<edm::InputTag>("L2Jets") ),
  m_L25TagInfo(     config.getParameter<edm::InputTag>("L25TagInfo") ),
  m_L25JetTags(     config.getParameter<edm::InputTag>("L25JetTags") ),
  m_L3TagInfo(      config.getParameter<edm::InputTag>("L3TagInfo") ),
  m_L3JetTags(      config.getParameter<edm::InputTag>("L3JetTags") ),
  m_triggerResults( config.getParameter<edm::InputTag>("triggerResults") ),
  m_processName(    config.getParameter<std::string>("processName") ),
  m_pathName(       config.getParameter<std::string>("pathName") ),
  m_monitorName(    config.getParameter<std::string>("monitorName" ) ),
  m_outputFile(     config.getUntrackedParameter<std::string>("outputFile", "HLTBJetDQM.root") ),
  m_storeROOT(      config.getUntrackedParameter<bool>("storeROOT", false) ),
  m_size(           config.getParameter<unsigned int>("interestingJets") ),
  m_dbe(),
  m_init(           false ),
  m_pathIndex(      (unsigned int) -1 ),
  m_L1FilterIndex(  (unsigned int) -1 ),
  m_L2FilterIndex(  (unsigned int) -1 ),
  m_L25FilterIndex( (unsigned int) -1 ),
  m_L3FilterIndex(  (unsigned int) -1 ),

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
  m_plotL3Discriminator(0)
{
}

HLTMonBTagIPSource::~HLTMonBTagIPSource(void) {
}

void HLTMonBTagIPSource::beginJob() {
  if (not m_dbe.isAvailable())
    return;

  m_dbe->setVerbose(0);
  m_dbe->setCurrentFolder(m_monitorName + "/" + m_pathName);

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
  m_plotL25IP2ndTrack3d             = book("L25_IP_2ndTrack_3d",     "L2.5 2nd pixel track 3D IP",        60,  -0.20, 1.00, "cm");
  m_plotL25IP2ndTrack3dSig          = book("L25_IP_2ndTrack_3dSig",  "L2.5 2nd pixel track 3D SIP",       80, -30.,   50.);
  m_plotL25IP3ndTrack2d             = book("L25_IP_3ndTrack_2d",     "L2.5 3rd pixel track 2D IP",        25,  -0.05, 0.20, "cm");
  m_plotL25IP3ndTrack2dSig          = book("L25_IP_3ndTrack_2dSig",  "L2.5 3rd pixel track 2D SIP",       80, -30.,   50.);
  m_plotL25IP3ndTrack3d             = book("L25_IP_3ndTrack_3d",     "L2.5 3rd pixel track 3D IP",        60,  -0.20, 1.00, "cm");
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
  m_plotL3IP2ndTrack3d              = book("L3_IP_2ndTrack_3d",      "L3 2nd track 3D IP",                60,  -0.20, 1.00, "cm");
  m_plotL3IP2ndTrack3dSig           = book("L3_IP_2ndTrack_3dSig",   "L3 2nd track 3D SIP",               80, -30.,   50.);
  m_plotL3IP3ndTrack2d              = book("L3_IP_3ndTrack_2d",      "L3 3rd track 2D IP",                25,  -0.05, 0.20, "cm");
  m_plotL3IP3ndTrack2dSig           = book("L3_IP_3ndTrack_2dSig",   "L3 3rd track 2D SIP",               80, -30.,   50.);
  m_plotL3IP3ndTrack3d              = book("L3_IP_3ndTrack_3d",      "L3 3rd track 3D IP",                60,  -0.20, 1.00, "cm");
  m_plotL3IP3ndTrack3dSig           = book("L3_IP_3ndTrack_3dSig",   "L3 3rd track 3D SIP",               80, -30.,   50.);
  m_plotL3Discriminator             = book("L3_discriminator",       "L3 b-tag discriminator",            80, -30.,   50.);
}

void HLTMonBTagIPSource::endJob() { 
  if (m_dbe.isAvailable() and m_storeROOT)
    m_dbe->save(m_outputFile);
}

void HLTMonBTagIPSource::beginRun(const edm::Run & run, const edm::EventSetup & setup) {

  HLTConfigProvider configProvider;

  bool changed = false;
  if (not configProvider.init(run, setup, m_processName, changed))
  {
    edm::LogWarning("ConfigurationError") << "process name \"" << m_processName << "\" is not valid.";
    m_init = false;
    return;
  }

  m_pathIndex = configProvider.triggerIndex( m_pathName );
  if (m_pathIndex == configProvider.size())
  {
    edm::LogWarning("ConfigurationError") << "trigger name \"" << m_pathName << "\" is not valid.";
    m_init = false;
    return;
  }

  m_init = true;

  // if their call fails, these will be set to one after the last valid module for their path
  // so they will never be "passed"
  unsigned int size = configProvider.size( m_pathIndex );

  m_L1FilterIndex  = configProvider.moduleIndex( m_pathIndex, m_L1Filter.encode()  );
  if (m_L1FilterIndex == size)
    edm::LogWarning("ConfigurationError") << "L1 filter \"" << m_L1Filter << "\" is not valid.";

  m_L2FilterIndex  = configProvider.moduleIndex( m_pathIndex, m_L2Filter.encode()  );
  if (m_L2FilterIndex == size)
    edm::LogWarning("ConfigurationError") << "L2 filter \"" << m_L2Filter << "\" is not valid.";

  m_L25FilterIndex = configProvider.moduleIndex( m_pathIndex, m_L25Filter.encode() );
  if (m_L25FilterIndex == size)
    edm::LogWarning("ConfigurationError") << "L2.5 filter \"" << m_L25Filter << "\" is not valid.";

  m_L3FilterIndex  = configProvider.moduleIndex( m_pathIndex, m_L3Filter.encode()  );
  if (m_L3FilterIndex == size)
    edm::LogWarning("ConfigurationError") << "L3 filter \"" << m_L3Filter << "\" is not valid.";
}

void HLTMonBTagIPSource::endRun(const edm::Run & run, const edm::EventSetup & setup) {
}

void HLTMonBTagIPSource::beginLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup) {
}

void HLTMonBTagIPSource::endLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup) {
}

void HLTMonBTagIPSource::analyze(const edm::Event & event, const edm::EventSetup & setup) {
  if (not m_dbe.isAvailable())
    return;

  if (not m_init)
    return;

  edm::Handle<edm::TriggerResults> h_triggerResults;
  edm::Handle<edm::View<reco::Jet> >          h_L2Jets;
  edm::Handle<reco::TrackIPTagInfoCollection> h_L25TagInfo;
  edm::Handle<reco::JetTagCollection>         h_L25JetTags;
  edm::Handle<reco::TrackIPTagInfoCollection> h_L3TagInfo;
  edm::Handle<reco::JetTagCollection>         h_L3JetTags;
  
  event.getByLabel(m_triggerResults, h_triggerResults);
  event.getByLabel(m_L2Jets,     h_L2Jets);
  event.getByLabel(m_L25TagInfo, h_L25TagInfo);
  event.getByLabel(m_L25JetTags, h_L25JetTags);
  event.getByLabel(m_L3TagInfo,  h_L3TagInfo);
  event.getByLabel(m_L3JetTags,  h_L3JetTags);

  // check if this path passed the L1, L2, L2.5 and L3 filters
  bool         wasrun = false;
  unsigned int latest = 0;
  bool         accept = false;
  if (h_triggerResults.isValid()) {
    wasrun = h_triggerResults->wasrun( m_pathIndex );
    latest = h_triggerResults->index(  m_pathIndex );
    accept = h_triggerResults->accept( m_pathIndex );
  }
  if (wasrun)
    m_plotRates->Fill( 0. );    // path was run
  if (latest > m_L1FilterIndex)
    m_plotRates->Fill( 1. );    // L1 accepted
  if (latest > m_L2FilterIndex)
    m_plotRates->Fill( 2. );    // L2 accepted  
  if (latest > m_L25FilterIndex)
    m_plotRates->Fill( 3. );    // L2.5 accepted  
  if (latest > m_L3FilterIndex)
    m_plotRates->Fill( 4. );    // L3 accepted  
  if (accept)
    m_plotRates->Fill( 5. );    // HLT accepted

  if ((latest > m_L1FilterIndex) and h_L2Jets.isValid()) {
    unsigned int size = std::min((unsigned int) h_L2Jets->size(), m_size);
    for (unsigned int i = 0; i < size; ++i) {
      const reco::Jet & jet = (*h_L2Jets)[i];
      m_plotL2JetsEnergy->Fill( jet.energy() );
      m_plotL2JetsET->Fill(     jet.et() );
      m_plotL2JetsEta->Fill(    jet.eta() );
      m_plotL2JetsPhi->Fill(    jet.phi() );
      m_plotL2JetsEtaPhi->Fill( jet.eta(), jet.phi() );
      m_plotL2JetsEtaET->Fill(  jet.eta(), jet.et() );
    }
  }
  if ((latest > m_L2FilterIndex) and h_L25TagInfo.isValid() and h_L25JetTags.isValid()) {
    unsigned int size = std::min((unsigned int) h_L25TagInfo->size(), m_size);
    for (unsigned int i = 0; i < size; ++i) {
      const reco::TrackIPTagInfo & info   = (*h_L25TagInfo)[i];
      const reco::Jet & jet = * info.jet();
      const reco::TrackRefVector & tracks = info.selectedTracks();
      const std::vector<reco::TrackIPTagInfo::TrackIPData> & data = info.impactParameterData();
      const reco::JetTag & tag = (*h_L25JetTags)[info.jet().key()];
      m_plotL25JetsEnergy->Fill( jet.energy() );
      m_plotL25JetsET->Fill(     jet.et() );
      m_plotL25JetsEta->Fill(    jet.eta() );
      m_plotL25JetsPhi->Fill(    jet.phi() );
      m_plotL25JetsEtaPhi->Fill( jet.eta(), jet.phi() );
      m_plotL25JetsEtaET->Fill(  jet.eta(), jet.et() );
      m_plotL25TrackMultiplicity->Fill( tracks.size() );
      for (unsigned int t = 0; t < tracks.size(); ++t) {
        m_plotL25TrackHits->Fill(   tracks[t]->numberOfValidHits() );
        m_plotL25TrackChi2->Fill(   tracks[t]->normalizedChi2() );
        m_plotL25TrackEtaPhi->Fill( tracks[t]->eta(), tracks[t]->phi() );
        m_plotL25TrackEtaPT->Fill(  tracks[t]->eta(), tracks[t]->pt() );
      }
      std::vector<size_t> indicesBy2d = info.sortedIndexes(reco::TrackIPTagInfo::IP2DSig);
      if (indicesBy2d.size() >= 2) {
        m_plotL25IP2ndTrack2d->Fill(    data[indicesBy2d[1]].ip2d.value() );
        m_plotL25IP2ndTrack2dSig->Fill( data[indicesBy2d[1]].ip2d.significance() );
      }
      if (indicesBy2d.size() >= 3) {
        m_plotL25IP3ndTrack2d->Fill(    data[indicesBy2d[2]].ip2d.value() );
        m_plotL25IP3ndTrack2dSig->Fill( data[indicesBy2d[2]].ip2d.significance() );
      }
      std::vector<size_t> indicesBy3d = info.sortedIndexes(reco::TrackIPTagInfo::IP3DSig);
      if (indicesBy3d.size() >= 2) {
        m_plotL25IP2ndTrack3d->Fill(    data[indicesBy3d[1]].ip3d.value() );
        m_plotL25IP2ndTrack3dSig->Fill( data[indicesBy3d[1]].ip3d.significance() );
      }
      if (indicesBy3d.size() >= 3) {
        m_plotL25IP3ndTrack3d->Fill(    data[indicesBy3d[2]].ip3d.value() );
        m_plotL25IP3ndTrack3dSig->Fill( data[indicesBy3d[2]].ip3d.significance() );
      }
      m_plotL25Discriminator->Fill( tag.second );
    }
  }
  if ((latest > m_L25FilterIndex) and h_L3TagInfo.isValid() and h_L3JetTags.isValid()) {
    unsigned int size = std::min((unsigned int) h_L3TagInfo->size(), m_size);
    for (unsigned int i = 0; i < size; ++i) {
      const reco::TrackIPTagInfo & info   = (*h_L3TagInfo)[i];
      const reco::Jet & jet = * info.jet();
      const reco::TrackRefVector & tracks = info.selectedTracks();
      const std::vector<reco::TrackIPTagInfo::TrackIPData> & data = info.impactParameterData();
      const reco::JetTag & tag = (*h_L3JetTags)[info.jet().key()];
      m_plotL3JetsEnergy->Fill( jet.energy() );
      m_plotL3JetsET->Fill(     jet.et() );
      m_plotL3JetsEta->Fill(    jet.eta() );
      m_plotL3JetsPhi->Fill(    jet.phi() );
      m_plotL3JetsEtaPhi->Fill( jet.eta(), jet.phi() );
      m_plotL3JetsEtaET->Fill(  jet.eta(), jet.et() );
      m_plotL3TrackMultiplicity->Fill( tracks.size() );
      for (unsigned int t = 0; t < tracks.size(); ++t) {
        m_plotL3TrackHits->Fill(   tracks[t]->numberOfValidHits() );
        m_plotL3TrackChi2->Fill(   tracks[t]->normalizedChi2() );
        m_plotL3TrackEtaPhi->Fill( tracks[t]->eta(), tracks[t]->phi() );
        m_plotL3TrackEtaPT->Fill(  tracks[t]->eta(), tracks[t]->pt() );
      }
      std::vector<size_t> indicesBy2d = info.sortedIndexes(reco::TrackIPTagInfo::IP2DSig);
      if (indicesBy2d.size() >= 2) {
        m_plotL3IP2ndTrack2d->Fill(    data[indicesBy2d[1]].ip2d.value() );
        m_plotL3IP2ndTrack2dSig->Fill( data[indicesBy2d[1]].ip2d.significance() );
      }
      if (indicesBy2d.size() >= 3) {
        m_plotL3IP3ndTrack2d->Fill(    data[indicesBy2d[2]].ip2d.value() );
        m_plotL3IP3ndTrack2dSig->Fill( data[indicesBy2d[2]].ip2d.significance() );
      }
      std::vector<size_t> indicesBy3d = info.sortedIndexes(reco::TrackIPTagInfo::IP3DSig);
      if (indicesBy3d.size() >= 2) {
        m_plotL3IP2ndTrack3d->Fill(    data[indicesBy3d[1]].ip3d.value() );
        m_plotL3IP2ndTrack3dSig->Fill( data[indicesBy3d[1]].ip3d.significance() );
      }
      if (indicesBy3d.size() >= 3) {
        m_plotL3IP3ndTrack3d->Fill(    data[indicesBy3d[2]].ip3d.value() );
        m_plotL3IP3ndTrack3dSig->Fill( data[indicesBy3d[2]].ip3d.significance() );
      }
      m_plotL3Discriminator->Fill( tag.second );
    }
  }
}

MonitorElement * HLTMonBTagIPSource::book(const std::string & name, const std::string & title , int x_bins, double x_min, double x_max, const char * x_axis) {
  MonitorElement * element = m_dbe->book1D(name, title, x_bins, x_min, x_max);
  if (x_axis)
    element->setAxisTitle(x_axis, 1);
  return element;
}

MonitorElement * HLTMonBTagIPSource::book(const std::string & name, const std::string & title , int x_bins, double x_min, double x_max, int y_bins, double y_min, double y_max, const char * x_axis, const char * y_axis) {
  MonitorElement * element = m_dbe->book2D(name, title, x_bins, x_min, x_max, y_bins, y_min, y_max);
  if (x_axis)
    element->setAxisTitle(x_axis, 1);
  if (y_axis)
    element->setAxisTitle(y_axis, 2);
  return element;
}

// register as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTMonBTagIPSource); 
