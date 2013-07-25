// \class JetExtender JetExtender.cc 
//
// Combines different Jet associations into single compact object
// which extends basic Jet information
// Fedor Ratnikov Sep. 10, 2007
// $Id: JetExtender.cc,v 1.4 2007/10/01 19:24:45 fedor Exp $
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

#include "JetExtender.h"

JetExtender::JetExtender(const edm::ParameterSet& fConfig)
  : mJets (fConfig.getParameter<edm::InputTag> ("jets")),
    mJet2TracksAtVX (fConfig.getParameter<edm::InputTag> ("jet2TracksAtVX")),
    mJet2TracksAtCALO (fConfig.getParameter<edm::InputTag> ("jet2TracksAtCALO"))
{
  produces<reco::JetExtendedAssociation::Container> ();
}

JetExtender::~JetExtender() {}

void JetExtender::produce(edm::Event& fEvent, const edm::EventSetup& fSetup) {
  // get stuff from Event
  edm::Handle <edm::View <reco::Jet> > jets_h;
  fEvent.getByLabel (mJets, jets_h);
  edm::Handle <reco::JetTracksAssociation::Container> j2tVX_h;
  if (!(mJet2TracksAtVX.label().empty())) fEvent.getByLabel (mJet2TracksAtVX, j2tVX_h);
  edm::Handle <reco::JetTracksAssociation::Container> j2tCALO_h;
  if (!(mJet2TracksAtCALO.label().empty())) fEvent.getByLabel (mJet2TracksAtCALO, j2tCALO_h);
  
  std::auto_ptr<reco::JetExtendedAssociation::Container> 
    jetExtender (new reco::JetExtendedAssociation::Container (reco::JetRefBaseProd(jets_h)));
  
  // loop over jets (make sure jets in associations are the same as in collection

  for (unsigned j = 0; j < jets_h->size(); ++j) {
    edm::RefToBase<reco::Jet> jet = jets_h->refAt(j);
    reco::JetExtendedAssociation::JetExtendedData extendedData;
    if (j2tVX_h.isValid ()) { // fill tracks@VX  summary
      try {
	extendedData.mTracksAtVertexNumber = reco::JetTracksAssociation::tracksNumber (*j2tVX_h, jet);
	extendedData.mTracksAtVertexP4 = reco::JetTracksAssociation::tracksP4 (*j2tVX_h, jet);
      }
      catch (cms::Exception e) {
	edm::LogError ("MismatchedJets") << "Jets in original collection " << mJets 
				    << " mismatch jets in j2t VX association " << mJet2TracksAtVX
				    << ". Wrong collections?";
	throw e;
      }
    }
    if (j2tCALO_h.isValid ()) { // fill tracks@CALO  summary
      try {
	extendedData.mTracksAtCaloNumber = reco::JetTracksAssociation::tracksNumber (*j2tCALO_h, jet);
	extendedData.mTracksAtCaloP4 = reco::JetTracksAssociation::tracksP4 (*j2tCALO_h, jet);
      }
      catch (cms::Exception e) {
	edm::LogError ("MismatchedJets") << "Jets in original collection " << mJets 
				    << " mismatch jets in j2t CALO association " << mJet2TracksAtCALO
				    << ". Wrong collections?";
	throw e;
      }
    }
    reco::JetExtendedAssociation::setValue (&*jetExtender, jet, extendedData);
  }
  fEvent.put (jetExtender);
}
