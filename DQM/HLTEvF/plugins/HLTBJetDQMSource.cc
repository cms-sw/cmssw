/** \class HLTBJetDQMSource
 *
 *  DQM source for BJet HLT paths
 *
 *  $Date: 2008/05/13 16:48:08 $
 *  $Revision: 1.4 $
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
#include "HLTBJetDQMSource.h"

HLTBJetDQMSource::HLTBJetDQMSource(const edm::ParameterSet & config) :
  m_lifetimeL2Jets(         config.getParameter<edm::InputTag>("lifetimeL2Jets") ),
  m_lifetimeL25TagInfo(     config.getParameter<edm::InputTag>("lifetimeL25TagInfo") ),
  m_lifetimeL25JetTags(     config.getParameter<edm::InputTag>("lifetimeL25JetTags") ),
  m_lifetimeL3TagInfo(      config.getParameter<edm::InputTag>("lifetimeL3TagInfo") ),
  m_lifetimeL3JetTags(      config.getParameter<edm::InputTag>("lifetimeL3JetTags") ),
  m_softmuonL2Jets(         config.getParameter<edm::InputTag>("softmuonL2Jets") ),
  m_softmuonL25TagInfo(     config.getParameter<edm::InputTag>("softmuonL25TagInfo") ),
  m_softmuonL25JetTags(     config.getParameter<edm::InputTag>("softmuonL25JetTags") ),
  m_softmuonL3TagInfo(      config.getParameter<edm::InputTag>("softmuonL3TagInfo") ),
  m_softmuonL3JetTags(      config.getParameter<edm::InputTag>("softmuonL3JetTags") ),
  m_performanceL2Jets(      config.getParameter<edm::InputTag>("performanceL2Jets") ),
  m_performanceL25TagInfo(  config.getParameter<edm::InputTag>("performanceL25TagInfo") ),
  m_performanceL25JetTags(  config.getParameter<edm::InputTag>("performanceL25JetTags") ),
  m_performanceL3TagInfo(   config.getParameter<edm::InputTag>("performanceL3TagInfo") ),
  m_performanceL3JetTags(   config.getParameter<edm::InputTag>("performanceL3JetTags") ),
  m_monitorName(            config.getUntrackedParameter<std::string>("monitorName", "HLT/HLTMonBJet") ),
  m_outputFile(             config.getUntrackedParameter<std::string>("outputFile", "HLTBJetDQM.root") ),
  m_storeROOT(              config.getUntrackedParameter<bool>("storeROOT", false) ),
  m_dbe(),
  m_lifetimeL2JetsEnergy(0),
  m_lifetimeL2JetsET(0),
  m_lifetimeL2JetsEta(0),
  m_lifetimeL2JetsPhi(0),
  m_lifetimeL25JetsEnergy(0),
  m_lifetimeL25JetsET(0),
  m_lifetimeL25JetsEta(0),
  m_lifetimeL25JetsPhi(0),
  m_lifetimeL25TrackMultiplicity(0),
  m_lifetimeL25TrackHits(0),
  m_lifetimeL25TrackChi2(0),
  m_lifetimeL25IP2ndTrack2d(0),
  m_lifetimeL25IP2ndTrack2dSig(0),
  m_lifetimeL25IP2ndTrack3d(0),
  m_lifetimeL25IP2ndTrack3dSig(0),
  m_lifetimeL25IP3ndTrack2d(0),
  m_lifetimeL25IP3ndTrack2dSig(0),
  m_lifetimeL25IP3ndTrack3d(0),
  m_lifetimeL25IP3ndTrack3dSig(0),
  m_lifetimeL25Discriminator(0),
  m_lifetimeL3JetsEnergy(0),
  m_lifetimeL3JetsET(0),
  m_lifetimeL3JetsEta(0),
  m_lifetimeL3JetsPhi(0),
  m_lifetimeL3TrackMultiplicity(0),
  m_lifetimeL3TrackHits(0),
  m_lifetimeL3TrackChi2(0),
  m_lifetimeL3IP2ndTrack2d(0),
  m_lifetimeL3IP2ndTrack2dSig(0),
  m_lifetimeL3IP2ndTrack3d(0),
  m_lifetimeL3IP2ndTrack3dSig(0),
  m_lifetimeL3IP3ndTrack2d(0),
  m_lifetimeL3IP3ndTrack2dSig(0),
  m_lifetimeL3IP3ndTrack3d(0),
  m_lifetimeL3IP3ndTrack3dSig(0),
  m_lifetimeL3Discriminator(0),
  m_softmuonL2JetsEnergy(0),
  m_softmuonL2JetsET(0),
  m_softmuonL2JetsEta(0),
  m_softmuonL2JetsPhi(0),
  m_softmuonL25JetsEnergy(0),
  m_softmuonL25JetsET(0),
  m_softmuonL25JetsEta(0),
  m_softmuonL25JetsPhi(0),
  m_softmuonL25MuonMultiplicity(0),
  m_softmuonL25MuonHits(0),
  m_softmuonL25MuonChi2(0),
  m_softmuonL25MuonDeltaR(0),
  m_softmuonL25MuonIP2dSig(0),
  m_softmuonL25MuonIP3dSig(0),
  m_softmuonL25MuonPtrel(0),
  m_softmuonL25Discriminator(0),
  m_softmuonL3JetsEnergy(0),
  m_softmuonL3JetsET(0),
  m_softmuonL3JetsEta(0),
  m_softmuonL3JetsPhi(0),
  m_softmuonL3MuonMultiplicity(0),
  m_softmuonL3MuonHits(0),
  m_softmuonL3MuonChi2(0),
  m_softmuonL3MuonDeltaR(0),
  m_softmuonL3MuonIP2dSig(0),
  m_softmuonL3MuonIP3dSig(0),
  m_softmuonL3MuonPtrel(0),
  m_softmuonL3Discriminator(0),
  m_performanceL2JetsEnergy(0),
  m_performanceL2JetsET(0),
  m_performanceL2JetsEta(0),
  m_performanceL2JetsPhi(0),
  m_performanceL25JetsEnergy(0),
  m_performanceL25JetsET(0),
  m_performanceL25JetsEta(0),
  m_performanceL25JetsPhi(0),
  m_performanceL25MuonMultiplicity(0),
  m_performanceL25MuonHits(0),
  m_performanceL25MuonChi2(0),
  m_performanceL25MuonDeltaR(0),
  m_performanceL25MuonIP2dSig(0),
  m_performanceL25MuonIP3dSig(0),
  m_performanceL25MuonPtrel(0),
  m_performanceL25Discriminator(0),
  m_performanceL3JetsEnergy(0),
  m_performanceL3JetsET(0),
  m_performanceL3JetsEta(0),
  m_performanceL3JetsPhi(0),
  m_performanceL3MuonMultiplicity(0),
  m_performanceL3MuonHits(0),
  m_performanceL3MuonChi2(0),
  m_performanceL3MuonDeltaR(0),
  m_performanceL3MuonIP2dSig(0),
  m_performanceL3MuonIP3dSig(0),
  m_performanceL3MuonPtrel(0),
  m_performanceL3Discriminator(0),
  m_counterEvent(0),
  m_prescaleEvent( config.getUntrackedParameter<unsigned int>("prescale", 0) )
{
  if (m_dbe.isAvailable()) {
    m_dbe->setVerbose(0);
    m_dbe->setCurrentFolder(m_monitorName);
  }
}

HLTBJetDQMSource::~HLTBJetDQMSource() {
}

void HLTBJetDQMSource::beginJob(const edm::EventSetup & setup) {
  if (not m_dbe.isAvailable())
    return;

  m_dbe->setCurrentFolder(m_monitorName + "/Lifetime");
  m_lifetimeL2JetsEnergy                = book("L2JetEnergy",           "L2 jet energy",                    500,   0.,  500.,  "GeV");
  m_lifetimeL2JetsET                    = book("L2JetET",               "L2 jet eT",                        500,   0.,  500.,  "GeV");
  m_lifetimeL2JetsEta                   = book("L2JetEta",              "L2 jet eta",                       600,  -3.0,   3.0, "#eta");
  m_lifetimeL2JetsPhi                   = book("L2JetPhi",              "L2 jet phi",                       640,  -3.2,   3.2, "#phi");
  m_lifetimeL25JetsEnergy               = book("L25JetEnergy",          "L2.5 jet Energy",                  500,   0.,  500.,  "GeV");
  m_lifetimeL25JetsET                   = book("L25JetET",              "L2.5 jet ET",                      500,   0.,  500.,  "GeV");
  m_lifetimeL25JetsEta                  = book("L25JetEta",             "L2.5 jet eta",                     600,  -3.0,   3.0, "#eta");
  m_lifetimeL25JetsPhi                  = book("L25JetPhi",             "L2.5 jet phi",                     640,  -3.2,   3.2, "#phi");
  m_lifetimeL25TrackMultiplicity        = book("L25TrackMultiplicity",  "L2.5 pixel tracks multiplicity",    25,   0,    25);
  m_lifetimeL25TrackHits                = book("L25TrackHits",          "L2.5 pixel tracks n. of hits",       5,   0,     5);
  m_lifetimeL25TrackChi2                = book("L25TrackChi2",          "L2.5 pixel tracks Chi2/DoF",        20,   0.,   20.,  "#chi^2/DoF");
  m_lifetimeL25IP2ndTrack2d             = book("L25IP2ndTrack2d",       "L2.5 2nd pixel track 2D IP",       250,  -0.05, 0.20, "cm");
  m_lifetimeL25IP2ndTrack2dSig          = book("L25IP2ndTrack2dSig",    "L2.5 2nd pixel track 2D SIP",       40, -30.,   50.);
  m_lifetimeL25IP2ndTrack3d             = book("L25IP2ndTrack3d",       "L2.5 2nd pixel track 3D IP",       300,  -1.0,  5.0,  "cm");
  m_lifetimeL25IP2ndTrack3dSig          = book("L25IP2ndTrack3dSig",    "L2.5 2nd pixel track 3D SIP",       40, -30.,   50.);
  m_lifetimeL25IP3ndTrack2d             = book("L25IP3ndTrack2d",       "L2.5 3rd pixel track 2D IP",       250,  -0.05, 0.20, "cm");
  m_lifetimeL25IP3ndTrack2dSig          = book("L25IP3ndTrack2dSig",    "L2.5 3rd pixel track 2D SIP",       40, -30.,   50.);
  m_lifetimeL25IP3ndTrack3d             = book("L25IP3ndTrack3d",       "L2.5 3rd pixel track 3D IP",       300,  -1.0,  5.0,  "cm");
  m_lifetimeL25IP3ndTrack3dSig          = book("L25IP3ndTrack3dSig",    "L2.5 3rd pixel track 3D SIP",       40, -30.,   50.);
  m_lifetimeL25Discriminator            = book("L25Discriminator",      "L2.5 b-tag discriminator",          40, -30.,   50.);
  m_lifetimeL3JetsEnergy                = book("L3JetEnergy",           "L3 jet Energy",                    500,   0.,  500.,  "GeV");
  m_lifetimeL3JetsET                    = book("L3JetET",               "L3 jet ET",                        500,   0.,  500.,  "GeV");
  m_lifetimeL3JetsEta                   = book("L3JetEta",              "L3 jet eta",                       600,  -3.0,   3.0, "#eta");
  m_lifetimeL3JetsPhi                   = book("L3JetPhi",              "L3 jet phi",                       640,  -3.2,   3.2, "#phi");
  m_lifetimeL3TrackMultiplicity         = book("L3TrackMultiplicity",   "L3 pixel tracks multiplicity",      25,   0,    25);
  m_lifetimeL3TrackHits                 = book("L3TrackHits",           "L3 pixel tracks n. of hits",         5,   0,     5);
  m_lifetimeL3TrackChi2                 = book("L3TrackChi2",           "L3 pixel tracks Chi2/DoF",          20,   0.,   20.,  "#chi^2/DoF");
  m_lifetimeL3IP2ndTrack2d              = book("L3IP2ndTrack2d",        "L3 2nd pixel track 2D IP",         250,  -0.05, 0.20, "cm");
  m_lifetimeL3IP2ndTrack2dSig           = book("L3IP2ndTrack2dSig",     "L3 2nd pixel track 2D SIP",        400, -30.,   50.);
  m_lifetimeL3IP2ndTrack3d              = book("L3IP2ndTrack3d",        "L3 2nd pixel track 3D IP",         300,  -1.0,  5.0,  "cm");
  m_lifetimeL3IP2ndTrack3dSig           = book("L3IP2ndTrack3dSig",     "L3 2nd pixel track 3D SIP",        400, -30.,   50.);
  m_lifetimeL3IP3ndTrack2d              = book("L3IP3ndTrack2d",        "L3 3rd pixel track 2D IP",         250,  -0.05, 0.20, "cm");
  m_lifetimeL3IP3ndTrack2dSig           = book("L3IP3ndTrack2dSig",     "L3 3rd pixel track 2D SIP",        400, -30.,   50.);
  m_lifetimeL3IP3ndTrack3d              = book("L3IP3ndTrack3d",        "L3 3rd pixel track 3D IP",         300,  -1.0,  5.0,  "cm");
  m_lifetimeL3IP3ndTrack3dSig           = book("L3IP3ndTrack3dSig",     "L3 3rd pixel track 3D SIP",        400, -30.,   50.);
  m_lifetimeL3Discriminator             = book("L3Discriminator",       "L3 b-tag discriminator",           400, -30.,   50.);

  m_dbe->setCurrentFolder(m_monitorName + "/Softmuon");
  m_softmuonL2JetsEnergy                = book("L2JetEnergy",           "L2 jet energy",                    500,   0.,  500.,  "GeV");
  m_softmuonL2JetsET                    = book("L2JetET",               "L2 jet eT",                        500,   0.,  500.,  "GeV");
  m_softmuonL2JetsEta                   = book("L2JetEta",              "L2 jet eta",                       600,  -3.0,   3.0, "#eta");
  m_softmuonL2JetsPhi                   = book("L2JetPhi",              "L2 jet phi",                       640,  -3.2,   3.2, "#phi");
  m_softmuonL25JetsEnergy               = book("L25JetEnergy",          "L2.5 jet Energy",                  500,   0.,  500.,  "GeV");
  m_softmuonL25JetsET                   = book("L25JetET",              "L2.5 jet ET",                      500,   0.,  500.,  "GeV");
  m_softmuonL25JetsEta                  = book("L25JetEta",             "L2.5 jet eta",                     600,  -3.0,   3.0, "#eta");
  m_softmuonL25JetsPhi                  = book("L25JetPhi",             "L2.5 jet phi",                     640,  -3.2,   3.2, "#phi");
  m_softmuonL25MuonMultiplicity         = book("L25MuonMultiplicity",   "L2 muons multiplicity",             25,   0,    25);
  m_softmuonL25MuonHits                 = book("L25MuonHits",           "L2 muons n. of hits",               50,   0,    50);
  m_softmuonL25MuonChi2                 = book("L25MuonChi2",           "L2 muons Chi2/DoF",                 20,   0.,   20.,  "#chi^2/DoF");
  m_softmuonL25MuonDeltaR               = book("L25MuonDeltaR",         "L2 muons DeltaR",                   50,   0.,    5.);
  m_softmuonL25MuonIP2dSig              = book("L25MuonIP2dSig",        "L2 muons 2D SIP",                  400, -30.,   50.);
  m_softmuonL25MuonIP3dSig              = book("L25MuonIP3dSig",        "L2 muons 3D SIP",                  400, -30.,   50.);
  m_softmuonL25MuonPtrel                = book("L25MuonPtrel",          "L2 muons pT_rel",                  100,   0.,   10.);
  m_softmuonL25Discriminator            = book("L25Discriminator",      "L2.5 b-tag discriminator",           2,   0,     2);
  m_softmuonL3JetsEnergy                = book("L3JetEnergy",           "L3 jet Energy",                    500,   0.,  500.,  "GeV");
  m_softmuonL3JetsET                    = book("L3JetET",               "L3 jet ET",                        500,   0.,  500.,  "GeV");
  m_softmuonL3JetsEta                   = book("L3JetEta",              "L3 jet eta",                       600,  -3.0,   3.0, "#eta");
  m_softmuonL3JetsPhi                   = book("L3JetPhi",              "L3 jet phi",                       640,  -3.2,   3.2, "#phi");
  m_softmuonL3MuonMultiplicity          = book("L3MuonMultiplicity",    "L3 muons multiplicity",             25,   0,    25);
  m_softmuonL3MuonHits                  = book("L3MuonHits",            "L3 muons n. of hits",               50,   0,    50);
  m_softmuonL3MuonChi2                  = book("L3MuonChi2",            "L3 muons Chi2/DoF",                 20,   0.,   20.,  "#chi^2/DoF");
  m_softmuonL3MuonDeltaR                = book("L3MuonDeltaR",          "L3 muons DeltaR",                   50,   0.,    5.);
  m_softmuonL3MuonIP2dSig               = book("L3MuonIP2dSig",         "L3 muons 2D SIP",                  400, -30.,   50.);
  m_softmuonL3MuonIP3dSig               = book("L3MuonIP3dSig",         "L3 muons 3D SIP",                  400, -30.,   50.);
  m_softmuonL3MuonPtrel                 = book("L3MuonPtrel",           "L3 muons pT_rel",                  100,   0.,   10.);
  m_softmuonL3Discriminator             = book("L3Discriminator",       "L3 b-tag discriminator",           100,   0.,    1.);

  m_dbe->setCurrentFolder(m_monitorName + "/Performance");
  m_performanceL2JetsEnergy             = book("L2JetEnergy",           "L2 jet energy",                    500,   0.,  500.,  "GeV");
  m_performanceL2JetsET                 = book("L2JetET",               "L2 jet eT",                        500,   0.,  500.,  "GeV");
  m_performanceL2JetsEta                = book("L2JetEta",              "L2 jet eta",                       600,  -3.0,   3.0, "#eta");
  m_performanceL2JetsPhi                = book("L2JetPhi",              "L2 jet phi",                       640,  -3.2,   3.2, "#phi");
  m_performanceL25JetsEnergy            = book("L25JetEnergy",          "L2.5 jet Energy",                  500,   0.,  500.,  "GeV");
  m_performanceL25JetsET                = book("L25JetET",              "L2.5 jet ET",                      500,   0.,  500.,  "GeV");
  m_performanceL25JetsEta               = book("L25JetEta",             "L2.5 jet eta",                     600,  -3.0,   3.0, "#eta");
  m_performanceL25JetsPhi               = book("L25JetPhi",             "L2.5 jet phi",                     640,  -3.2,   3.2, "#phi");
  m_performanceL25MuonMultiplicity      = book("L25MuonMultiplicity",   "L2 muons multiplicity",             25,   0,    25);
  m_performanceL25MuonHits              = book("L25MuonHits",           "L2 muons n. of hits",               50,   0,    50);
  m_performanceL25MuonChi2              = book("L25MuonChi2",           "L2 muons Chi2/DoF",                 20,   0.,   20.,  "#chi^2/DoF");
  m_performanceL25MuonDeltaR            = book("L25MuonDeltaR",         "L2 muons DeltaR",                   50,   0.,    5.);
  m_performanceL25MuonIP2dSig           = book("L25MuonIP2dSig",        "L2 muons 2D SIP",                  400, -30.,   50.);
  m_performanceL25MuonIP3dSig           = book("L25MuonIP3dSig",        "L2 muons 3D SIP",                  400, -30.,   50.);
  m_performanceL25MuonPtrel             = book("L25MuonPtrel",          "L2 muons pT_rel",                  100,   0.,   10.);
  m_performanceL25Discriminator         = book("L25Discriminator",      "L2.5 b-tag discriminator",           2,   0,     2);
  m_performanceL3JetsEnergy             = book("L3JetEnergy",           "L3 jet Energy",                    500,   0.,  500.,  "GeV");
  m_performanceL3JetsET                 = book("L3JetET",               "L3 jet ET",                        500,   0.,  500.,  "GeV");
  m_performanceL3JetsEta                = book("L3JetEta",              "L3 jet eta",                       600,  -3.0,   3.0, "#eta");
  m_performanceL3JetsPhi                = book("L3JetPhi",              "L3 jet phi",                       640,  -3.2,   3.2, "#phi");
  m_performanceL3MuonMultiplicity       = book("L3MuonMultiplicity",    "L3 muons multiplicity",             25,   0,    25);
  m_performanceL3MuonHits               = book("L3MuonHits",            "L3 muons n. of hits",               50,   0,    50);
  m_performanceL3MuonChi2               = book("L3MuonChi2",            "L3 muons Chi2/DoF",                 20,   0.,   20.,  "#chi^2/DoF");
  m_performanceL3MuonDeltaR             = book("L3MuonDeltaR",          "L3 muons DeltaR",                   50,   0.,    5.);
  m_performanceL3MuonIP2dSig            = book("L3MuonIP2dSig",         "L3 muons 2D SIP",                  400, -30.,   50.);
  m_performanceL3MuonIP3dSig            = book("L3MuonIP3dSig",         "L3 muons 3D SIP",                  400, -30.,   50.);
  m_performanceL3MuonPtrel              = book("L3MuonPtrel",           "L3 muons pT_rel",                  100,   0.,   10.);
  m_performanceL3Discriminator          = book("L3Discriminator",       "L3 b-tag discriminator",             2,   0.,    2.);

}

void HLTBJetDQMSource::endJob() { 
  if (m_dbe.isAvailable() and m_storeROOT)
    m_dbe->save(m_outputFile);
}

void HLTBJetDQMSource::beginRun(const edm::Run & run, const edm::EventSetup & setup) {
}

void HLTBJetDQMSource::endRun(const edm::Run & run, const edm::EventSetup & setup) {
}

void HLTBJetDQMSource::beginLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup) {
}

void HLTBJetDQMSource::endLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup) {
}

void HLTBJetDQMSource::analyze(const edm::Event & event, const edm::EventSetup & setup) {
  if (not m_dbe.isAvailable())
    return;
  
  ++m_counterEvent;
  if (m_prescaleEvent and m_counterEvent % m_prescaleEvent)
    return;

  analyzeLifetime(event, setup);
  analyzeSoftmuon(event, setup);
  analyzePerformance(event, setup);
}

void HLTBJetDQMSource::analyzeLifetime(const edm::Event & event, const edm::EventSetup & setup) {
  edm::Handle<edm::View<reco::Jet> >                h_lifetimeL2Jets;
  edm::Handle<reco::TrackIPTagInfoCollection>       h_lifetimeL25TagInfo;
  edm::Handle<reco::JetTagCollection>               h_lifetimeL25JetTags;
  edm::Handle<reco::TrackIPTagInfoCollection>       h_lifetimeL3TagInfo;
  edm::Handle<reco::JetTagCollection>               h_lifetimeL3JetTags;
  
  event.getByLabel(m_lifetimeL2Jets,        h_lifetimeL2Jets);
  event.getByLabel(m_lifetimeL25TagInfo,    h_lifetimeL25TagInfo);
  event.getByLabel(m_lifetimeL25JetTags,    h_lifetimeL25JetTags);
  event.getByLabel(m_lifetimeL3TagInfo,     h_lifetimeL3TagInfo);
  event.getByLabel(m_lifetimeL3JetTags,     h_lifetimeL3JetTags);

  if (h_lifetimeL2Jets.isValid()) {
    for (unsigned int i = 0; i < h_lifetimeL2Jets->size(); ++i) {
      const reco::Jet & jet = (*h_lifetimeL2Jets)[i];
      m_lifetimeL2JetsEnergy->Fill( jet.energy() );
      m_lifetimeL2JetsET->Fill(     jet.et() );
      m_lifetimeL2JetsEta->Fill(    jet.eta() );
      m_lifetimeL2JetsPhi->Fill(    jet.phi() );
    }
  }
  if (h_lifetimeL25TagInfo.isValid() and h_lifetimeL25JetTags.isValid()) {
    for (unsigned int i = 0; i < h_lifetimeL25TagInfo->size(); ++i) {
      const reco::TrackIPTagInfo & info   = (*h_lifetimeL25TagInfo)[i];
      const reco::Jet & jet = * info.jet();
      const reco::TrackRefVector & tracks = info.selectedTracks();
      const std::vector<reco::TrackIPTagInfo::TrackIPData> & data = info.impactParameterData();
      const reco::JetTag & tag = (*h_lifetimeL25JetTags)[info.jet().key()];
      m_lifetimeL25JetsEnergy->Fill( jet.energy() );
      m_lifetimeL25JetsET->Fill(     jet.et() );
      m_lifetimeL25JetsEta->Fill(    jet.eta() );
      m_lifetimeL25JetsPhi->Fill(    jet.phi() );
      m_lifetimeL25TrackMultiplicity->Fill( tracks.size() );
      for (unsigned int t = 0; t < tracks.size(); ++t) {
        m_lifetimeL25TrackHits->Fill( tracks[t]->numberOfValidHits() );
        m_lifetimeL25TrackChi2->Fill( tracks[t]->normalizedChi2() );
      }
      std::vector<size_t> indicesBy2d = info.sortedIndexes(reco::TrackIPTagInfo::IP2DSig);
      if (indicesBy2d.size() >= 2) {
        m_lifetimeL25IP2ndTrack2d->Fill(    data[indicesBy2d[1]].ip2d.value() );
        m_lifetimeL25IP2ndTrack2dSig->Fill( data[indicesBy2d[1]].ip2d.significance() );
      }
      if (indicesBy2d.size() >= 3) {
        m_lifetimeL25IP3ndTrack2d->Fill(    data[indicesBy2d[2]].ip2d.value() );
        m_lifetimeL25IP3ndTrack2dSig->Fill( data[indicesBy2d[2]].ip2d.significance() );
      }
      std::vector<size_t> indicesBy3d = info.sortedIndexes(reco::TrackIPTagInfo::IP3DSig);
      if (indicesBy3d.size() >= 2) {
        m_lifetimeL25IP2ndTrack3d->Fill(    data[indicesBy3d[1]].ip3d.value() );
        m_lifetimeL25IP2ndTrack3dSig->Fill( data[indicesBy3d[1]].ip3d.significance() );
      }
      if (indicesBy3d.size() >= 3) {
        m_lifetimeL25IP3ndTrack3d->Fill(    data[indicesBy3d[2]].ip3d.value() );
        m_lifetimeL25IP3ndTrack3dSig->Fill( data[indicesBy3d[2]].ip3d.significance() );
      }
      m_lifetimeL25Discriminator->Fill( tag.second );
    }
  }
  if (h_lifetimeL3TagInfo.isValid() and h_lifetimeL3JetTags.isValid()) {
    for (unsigned int i = 0; i < h_lifetimeL3TagInfo->size(); ++i) {
      const reco::TrackIPTagInfo & info   = (*h_lifetimeL3TagInfo)[i];
      const reco::Jet & jet = * info.jet();
      const reco::TrackRefVector & tracks = info.selectedTracks();
      const std::vector<reco::TrackIPTagInfo::TrackIPData> & data = info.impactParameterData();
      const reco::JetTag & tag = (*h_lifetimeL3JetTags)[info.jet().key()];
      m_lifetimeL3JetsEnergy->Fill( jet.energy() );
      m_lifetimeL3JetsET->Fill(     jet.et() );
      m_lifetimeL3JetsEta->Fill(    jet.eta() );
      m_lifetimeL3JetsPhi->Fill(    jet.phi() );
      m_lifetimeL3TrackMultiplicity->Fill( tracks.size() );
      for (unsigned int t = 0; t < tracks.size(); ++t) {
        m_lifetimeL3TrackHits->Fill( tracks[t]->numberOfValidHits() );
        m_lifetimeL3TrackChi2->Fill( tracks[t]->normalizedChi2() );
      }
      std::vector<size_t> indicesBy2d = info.sortedIndexes(reco::TrackIPTagInfo::IP2DSig);
      if (indicesBy2d.size() >= 2) {
        m_lifetimeL3IP2ndTrack2d->Fill(    data[indicesBy2d[1]].ip2d.value() );
        m_lifetimeL3IP2ndTrack2dSig->Fill( data[indicesBy2d[1]].ip2d.significance() );
      }
      if (indicesBy2d.size() >= 3) {
        m_lifetimeL3IP3ndTrack2d->Fill(    data[indicesBy2d[2]].ip2d.value() );
        m_lifetimeL3IP3ndTrack2dSig->Fill( data[indicesBy2d[2]].ip2d.significance() );
      }
      std::vector<size_t> indicesBy3d = info.sortedIndexes(reco::TrackIPTagInfo::IP3DSig);
      if (indicesBy3d.size() >= 2) {
        m_lifetimeL3IP2ndTrack3d->Fill(    data[indicesBy3d[1]].ip3d.value() );
        m_lifetimeL3IP2ndTrack3dSig->Fill( data[indicesBy3d[1]].ip3d.significance() );
      }
      if (indicesBy3d.size() >= 3) {
        m_lifetimeL3IP3ndTrack3d->Fill(    data[indicesBy3d[2]].ip3d.value() );
        m_lifetimeL3IP3ndTrack3dSig->Fill( data[indicesBy3d[2]].ip3d.significance() );
      }
      m_lifetimeL3Discriminator->Fill( tag.second );
    }
  }
}

void HLTBJetDQMSource::analyzeSoftmuon(const edm::Event & event, const edm::EventSetup & setup) {
  edm::Handle<edm::View<reco::Jet> >                h_softmuonL2Jets;
  edm::Handle<reco::SoftLeptonTagInfoCollection>    h_softmuonL25TagInfo;
  edm::Handle<reco::JetTagCollection>               h_softmuonL25JetTags;
  edm::Handle<reco::SoftLeptonTagInfoCollection>    h_softmuonL3TagInfo;
  edm::Handle<reco::JetTagCollection>               h_softmuonL3JetTags;

  event.getByLabel(m_softmuonL2Jets,        h_softmuonL2Jets);
  event.getByLabel(m_softmuonL25TagInfo,    h_softmuonL25TagInfo);
  event.getByLabel(m_softmuonL25JetTags,    h_softmuonL25JetTags);
  event.getByLabel(m_softmuonL3TagInfo,     h_softmuonL3TagInfo);
  event.getByLabel(m_softmuonL3JetTags,     h_softmuonL3JetTags);

  if (h_softmuonL2Jets.isValid()) { 
    for (unsigned int i = 0; i < h_softmuonL2Jets->size(); ++i) {
      const reco::Jet & jet = (*h_softmuonL2Jets)[i];
      m_softmuonL2JetsEnergy->Fill( jet.energy() );
      m_softmuonL2JetsET->Fill(     jet.et() );
      m_softmuonL2JetsEta->Fill(    jet.eta() );
      m_softmuonL2JetsPhi->Fill(    jet.phi() );
    }
  }
  if (h_softmuonL25TagInfo.isValid()) { 
    for (unsigned int i = 0; i < h_softmuonL25TagInfo->size(); ++i) {
      const reco::SoftLeptonTagInfo & info = (*h_softmuonL25TagInfo)[i];
      const reco::Jet & jet = * info.jet();
      const reco::JetTag & tag = (*h_softmuonL25JetTags)[info.jet().key()];
      m_softmuonL25JetsEnergy->Fill( jet.energy() );
      m_softmuonL25JetsET->Fill(     jet.et() );
      m_softmuonL25JetsEta->Fill(    jet.eta() );
      m_softmuonL25JetsPhi->Fill(    jet.phi() );
      m_softmuonL25MuonMultiplicity->Fill( info.leptons() );
      for (unsigned int l = 0; l < info.leptons(); ++l) {
        m_softmuonL25MuonHits->Fill(    info.lepton(l)->numberOfValidHits() );
        m_softmuonL25MuonChi2->Fill(    info.lepton(l)->normalizedChi2() );
        m_softmuonL25MuonDeltaR->Fill(  info.properties(l).deltaR );
        m_softmuonL25MuonIP2dSig->Fill( info.properties(l).sip2d );
        m_softmuonL25MuonIP3dSig->Fill( info.properties(l).sip3d );
        m_softmuonL25MuonPtrel->Fill(   info.properties(l).ptRel );
      }
      m_softmuonL25Discriminator->Fill( tag.second );
    }
  }
  if (h_softmuonL3TagInfo.isValid()) { 
    for (unsigned int i = 0; i < h_softmuonL3TagInfo->size(); ++i) {
      const reco::SoftLeptonTagInfo & info = (*h_softmuonL3TagInfo)[i];
      const reco::Jet & jet = * info.jet();
      const reco::JetTag & tag = (*h_softmuonL3JetTags)[info.jet().key()];
      m_softmuonL3JetsEnergy->Fill( jet.energy() );
      m_softmuonL3JetsET->Fill(     jet.et() );
      m_softmuonL3JetsEta->Fill(    jet.eta() );
      m_softmuonL3JetsPhi->Fill(    jet.phi() );
      m_softmuonL3MuonMultiplicity->Fill( info.leptons() );
      for (unsigned int l = 0; l < info.leptons(); ++l) {
        m_softmuonL3MuonHits->Fill(    info.lepton(l)->numberOfValidHits() );
        m_softmuonL3MuonChi2->Fill(    info.lepton(l)->normalizedChi2() );
        m_softmuonL3MuonDeltaR->Fill(  info.properties(l).deltaR );
        m_softmuonL3MuonIP2dSig->Fill( info.properties(l).sip2d );
        m_softmuonL3MuonIP3dSig->Fill( info.properties(l).sip3d );
        m_softmuonL3MuonPtrel->Fill(   info.properties(l).ptRel );
      }
      m_softmuonL3Discriminator->Fill( tag.second );
    }
  }
}

void HLTBJetDQMSource::analyzePerformance(const edm::Event & event, const edm::EventSetup & setup) {
  edm::Handle<edm::View<reco::Jet> >                h_performanceL2Jets;
  edm::Handle<reco::SoftLeptonTagInfoCollection>    h_performanceL25TagInfo;
  edm::Handle<reco::JetTagCollection>               h_performanceL25JetTags;
  edm::Handle<reco::SoftLeptonTagInfoCollection>    h_performanceL3TagInfo;
  edm::Handle<reco::JetTagCollection>               h_performanceL3JetTags;

  event.getByLabel(m_performanceL2Jets,        h_performanceL2Jets);
  event.getByLabel(m_performanceL25TagInfo,    h_performanceL25TagInfo);
  event.getByLabel(m_performanceL25JetTags,    h_performanceL25JetTags);
  event.getByLabel(m_performanceL3TagInfo,     h_performanceL3TagInfo);
  event.getByLabel(m_performanceL3JetTags,     h_performanceL3JetTags);

  if (h_performanceL2Jets.isValid()) { 
    for (unsigned int i = 0; i < h_performanceL2Jets->size(); ++i) {
      const reco::Jet & jet = (*h_performanceL2Jets)[i];
      m_performanceL2JetsEnergy->Fill( jet.energy() );
      m_performanceL2JetsET->Fill(     jet.et() );
      m_performanceL2JetsEta->Fill(    jet.eta() );
      m_performanceL2JetsPhi->Fill(    jet.phi() );
    }
  }
  if (h_performanceL25TagInfo.isValid()) { 
    for (unsigned int i = 0; i < h_performanceL25TagInfo->size(); ++i) {
      const reco::SoftLeptonTagInfo & info = (*h_performanceL25TagInfo)[i];
      const reco::Jet & jet = * info.jet();
      const reco::JetTag & tag = (*h_performanceL25JetTags)[info.jet().key()];
      m_performanceL25JetsEnergy->Fill( jet.energy() );
      m_performanceL25JetsET->Fill(     jet.et() );
      m_performanceL25JetsEta->Fill(    jet.eta() );
      m_performanceL25JetsPhi->Fill(    jet.phi() );
      m_performanceL25MuonMultiplicity->Fill( info.leptons() );
      for (unsigned int l = 0; l < info.leptons(); ++l) {
        m_performanceL25MuonHits->Fill(    info.lepton(l)->numberOfValidHits() );
        m_performanceL25MuonChi2->Fill(    info.lepton(l)->normalizedChi2() );
        m_performanceL25MuonDeltaR->Fill(  info.properties(l).deltaR );
        m_performanceL25MuonIP2dSig->Fill( info.properties(l).sip2d );
        m_performanceL25MuonIP3dSig->Fill( info.properties(l).sip3d );
        m_performanceL25MuonPtrel->Fill(   info.properties(l).ptRel );
      }
      m_performanceL25Discriminator->Fill( tag.second );
    }
  }
  if (h_performanceL3TagInfo.isValid()) { 
    for (unsigned int i = 0; i < h_performanceL3TagInfo->size(); ++i) {
      const reco::SoftLeptonTagInfo & info = (*h_performanceL3TagInfo)[i];
      const reco::Jet & jet = * info.jet();
      const reco::JetTag & tag = (*h_performanceL3JetTags)[info.jet().key()];
      m_performanceL3JetsEnergy->Fill( jet.energy() );
      m_performanceL3JetsET->Fill(     jet.et() );
      m_performanceL3JetsEta->Fill(    jet.eta() );
      m_performanceL3JetsPhi->Fill(    jet.phi() );
      m_performanceL3MuonMultiplicity->Fill( info.leptons() );
      for (unsigned int l = 0; l < info.leptons(); ++l) {
        m_performanceL3MuonHits->Fill(    info.lepton(l)->numberOfValidHits() );
        m_performanceL3MuonChi2->Fill(    info.lepton(l)->normalizedChi2() );
        m_performanceL3MuonDeltaR->Fill(  info.properties(l).deltaR );
        m_performanceL3MuonIP2dSig->Fill( info.properties(l).sip2d );
        m_performanceL3MuonIP3dSig->Fill( info.properties(l).sip3d );
        m_performanceL3MuonPtrel->Fill(   info.properties(l).ptRel );
      }
      m_performanceL3Discriminator->Fill( tag.second );
    }
  }
}

MonitorElement * HLTBJetDQMSource::book(const char * name, const char * title , int x_bins, double x_min, double x_max, const char * x_axis) {
  MonitorElement * element = m_dbe->book1D(name, title, x_bins, x_min, x_max);
  if (x_axis)
    element->setAxisTitle(x_axis, 1);
  return element;
}

MonitorElement * HLTBJetDQMSource::book(const char * name, const char * title , int x_bins, double x_min, double x_max, int y_bins, double y_min, double y_max, const char * x_axis, const char * y_axis) {
  MonitorElement * element = m_dbe->book2D(name, title, x_bins, x_min, x_max, y_bins, y_min, y_max);
  if (x_axis)
    element->setAxisTitle(x_axis, 1);
  if (y_axis)
    element->setAxisTitle(y_axis, 2);
  return element;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTBJetDQMSource); 
