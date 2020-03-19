// \class JetExtender JetExtender.cc
//
// Combines different Jet associations into single compact object
// which extends basic Jet information
// Fedor Ratnikov Sep. 10, 2007
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"

#include "JetExtender.h"

JetExtender::JetExtender(const edm::ParameterSet& fConfig)
    : mJets(fConfig.getParameter<edm::InputTag>("jets")),
      mJet2TracksAtVX(fConfig.getParameter<edm::InputTag>("jet2TracksAtVX")),
      mJet2TracksAtCALO(fConfig.getParameter<edm::InputTag>("jet2TracksAtCALO")) {
  token_mJets = consumes<edm::View<reco::Jet> >(mJets);
  if (!(mJet2TracksAtVX.label().empty()))
    token_mJet2TracksAtVX = consumes<reco::JetTracksAssociation::Container>(mJet2TracksAtVX);
  if (!(mJet2TracksAtCALO.label().empty()))
    token_mJet2TracksAtCALO = consumes<reco::JetTracksAssociation::Container>(mJet2TracksAtCALO);

  produces<reco::JetExtendedAssociation::Container>();
}

JetExtender::~JetExtender() {}

void JetExtender::produce(edm::Event& fEvent, const edm::EventSetup& fSetup) {
  // get stuff from Event
  edm::Handle<edm::View<reco::Jet> > jets_h;
  fEvent.getByToken(token_mJets, jets_h);
  edm::Handle<reco::JetTracksAssociation::Container> j2tVX_h;
  if (!(mJet2TracksAtVX.label().empty()))
    fEvent.getByToken(token_mJet2TracksAtVX, j2tVX_h);
  edm::Handle<reco::JetTracksAssociation::Container> j2tCALO_h;
  if (!(mJet2TracksAtCALO.label().empty()))
    fEvent.getByToken(token_mJet2TracksAtCALO, j2tCALO_h);

  auto jetExtender = std::make_unique<reco::JetExtendedAssociation::Container>(reco::JetRefBaseProd(jets_h));

  // loop over jets (make sure jets in associations are the same as in collection

  for (unsigned j = 0; j < jets_h->size(); ++j) {
    edm::RefToBase<reco::Jet> jet = jets_h->refAt(j);
    reco::JetExtendedAssociation::JetExtendedData extendedData;
    if (j2tVX_h.isValid()) {  // fill tracks@VX  summary
      try {
        extendedData.mTracksAtVertexNumber = reco::JetTracksAssociation::tracksNumber(*j2tVX_h, jet);
        extendedData.mTracksAtVertexP4 = reco::JetTracksAssociation::tracksP4(*j2tVX_h, jet);
      } catch (cms::Exception const&) {
        edm::LogError("MismatchedJets") << "Jets in original collection " << mJets
                                        << " mismatch jets in j2t VX association " << mJet2TracksAtVX
                                        << ". Wrong collections?";
        throw;
      }
    }
    if (j2tCALO_h.isValid()) {  // fill tracks@CALO  summary
      try {
        extendedData.mTracksAtCaloNumber = reco::JetTracksAssociation::tracksNumber(*j2tCALO_h, jet);
        extendedData.mTracksAtCaloP4 = reco::JetTracksAssociation::tracksP4(*j2tCALO_h, jet);
      } catch (cms::Exception const&) {
        edm::LogError("MismatchedJets") << "Jets in original collection " << mJets
                                        << " mismatch jets in j2t CALO association " << mJet2TracksAtCALO
                                        << ". Wrong collections?";
        throw;
      }
    }
    reco::JetExtendedAssociation::setValue(&*jetExtender, jet, extendedData);
  }
  fEvent.put(std::move(jetExtender));
}
