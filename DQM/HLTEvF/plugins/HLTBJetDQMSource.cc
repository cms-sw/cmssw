/** \class HLTBJetDQMSource
 *
 *  DQM source for BJet HLT paths
 *
 *  $Date: 2008/05/13 16:38:49 $
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
  m_storeDQM(               config.getUntrackedParameter<bool>("storeDQM", true) ),
  m_storeROOT(              config.getUntrackedParameter<bool>("storeROOT", false) ),
  m_dbe(0),
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
  m_softmuonL25MuonIP2d(0),
  m_softmuonL25MuonIP2dSig(0),
  m_softmuonL25MuonIP3d(0),
  m_softmuonL25MuonIP3dSig(0),
  m_softmuonL25MuonPtrel(0),
  m_softmuonL25MuonPtrelSig(0),
  m_softmuonL25Discriminator(0),
  m_softmuonL3JetsEnergy(0),
  m_softmuonL3JetsET(0),
  m_softmuonL3JetsEta(0),
  m_softmuonL3JetsPhi(0),
  m_softmuonL3MuonMultiplicity(0),
  m_softmuonL3MuonHits(0),
  m_softmuonL3MuonChi2(0),
  m_softmuonL3MuonDeltaR(0),
  m_softmuonL3MuonIP2d(0),
  m_softmuonL3MuonIP2dSig(0),
  m_softmuonL3MuonIP3d(0),
  m_softmuonL3MuonIP3dSig(0),
  m_softmuonL3MuonPtrel(0),
  m_softmuonL3MuonPtrelSig(0),
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
  m_performanceL25MuonIP2d(0),
  m_performanceL25MuonIP2dSig(0),
  m_performanceL25MuonIP3d(0),
  m_performanceL25MuonIP3dSig(0),
  m_performanceL25MuonPtrel(0),
  m_performanceL25MuonPtrelSig(0),
  m_performanceL25Discriminator(0),
  m_performanceL3JetsEnergy(0),
  m_performanceL3JetsET(0),
  m_performanceL3JetsEta(0),
  m_performanceL3JetsPhi(0),
  m_performanceL3MuonMultiplicity(0),
  m_performanceL3MuonHits(0),
  m_performanceL3MuonChi2(0),
  m_performanceL3MuonDeltaR(0),
  m_performanceL3MuonIP2d(0),
  m_performanceL3MuonIP2dSig(0),
  m_performanceL3MuonIP3d(0),
  m_performanceL3MuonIP3dSig(0),
  m_performanceL3MuonPtrel(0),
  m_performanceL3MuonPtrelSig(0),
  m_performanceL3Discriminator(0),
  m_counterEvent(0),
  m_prescaleEvent( config.getUntrackedParameter<unsigned int>("prescale", 0) )
{
}

HLTBJetDQMSource::~HLTBJetDQMSource() { }

void HLTBJetDQMSource::beginJob(const edm::EventSetup & setup) { }
void HLTBJetDQMSource::endJob() { }

void HLTBJetDQMSource::beginRun(const edm::Run & run, const edm::EventSetup & setup) { }
void HLTBJetDQMSource::endRun(const edm::Run & run, const edm::EventSetup & setup) { }

void HLTBJetDQMSource::beginLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup) { }
void HLTBJetDQMSource::endLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup) { }

void HLTBJetDQMSource::analyze(const edm::Event & event, const edm::EventSetup & setup) {
  ++m_counterEvent;
  if (m_prescaleEvent and m_counterEvent % m_prescaleEvent)
    return;

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
  if (h_softmuonL25TagInfo.isValid() and h_softmuonL25JetTags.isValid()) {
  }
  if (h_softmuonL3TagInfo.isValid() and h_softmuonL3JetTags.isValid()) {
  }

  event.getByLabel(m_performanceL2Jets,     h_performanceL2Jets);
  event.getByLabel(m_performanceL25TagInfo, h_performanceL25TagInfo);
  event.getByLabel(m_performanceL25JetTags, h_performanceL25JetTags);
  event.getByLabel(m_performanceL3TagInfo,  h_performanceL3TagInfo);
  event.getByLabel(m_performanceL3JetTags,  h_performanceL3JetTags);

  edm::Handle<edm::View<reco::Jet> >                h_performanceL2Jets;
  edm::Handle<reco::SoftLeptonTagInfoCollection>    h_performanceL25TagInfo;
  edm::Handle<reco::JetTagCollection>               h_performanceL25JetTags;
  edm::Handle<reco::SoftLeptonTagInfoCollection>    h_performanceL3TagInfo;
  edm::Handle<reco::JetTagCollection>               h_performanceL3JetTags;

  if (h_performanceL2Jets.isValid()) { 
    for (unsigned int i = 0; i < h_performanceL2Jets->size(); ++i) {
      const reco::Jet & jet = (*h_performanceL2Jets)[i];
      m_performanceL2JetsEnergy->Fill( jet.energy() );
      m_performanceL2JetsET->Fill(     jet.et() );
      m_performanceL2JetsEta->Fill(    jet.eta() );
      m_performanceL2JetsPhi->Fill(    jet.phi() );
    }
  }
  if (h_performanceL25TagInfo.isValid() and h_performanceL25JetTags.isValid()) {
  }
  if (h_performanceL3TagInfo.isValid() and h_performanceL3JetTags.isValid()) {
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTBJetDQMSource); 
