// \class JetTracksAssociatorAtCaloFace JetTracksAssociatorAtCaloFace.cc 
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// Accommodated for Jet Package by: Fedor Ratnikov Jul. 30, 2007
// $Id: JetTracksAssociatorAtCaloFace.cc,v 1.6 2013/02/27 20:42:22 eulisse Exp $
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"



#include "JetTracksAssociatorAtCaloFace.h"

JetTracksAssociatorAtCaloFace::JetTracksAssociatorAtCaloFace(const edm::ParameterSet& fConfig)
  : mJets (fConfig.getParameter<edm::InputTag> ("jets")),
    mExtrapolations (fConfig.getParameter<edm::InputTag> ("extrapolations")),
    firstRun(true),
    dR_(fConfig.getParameter<double>("coneSize"))
{
  produces<reco::JetTracksAssociation::Container> ();
}

void JetTracksAssociatorAtCaloFace::produce(edm::Event& fEvent, const edm::EventSetup& fSetup) {


  // Get geometry
  if ( firstRun ) {
    fSetup.get<CaloGeometryRecord>().get(pGeo);
    firstRun = false;
  } 

  if ( !pGeo.isValid() ) {
    throw cms::Exception("InvalidInput") << "Did not get geometry" << std::endl;
  }

  // get stuff from Event
  edm::Handle <edm::View <reco::Jet> > jets_h;
  fEvent.getByLabel (mJets, jets_h);
  edm::Handle <std::vector<reco::TrackExtrapolation> > extrapolations_h;
  fEvent.getByLabel (mExtrapolations, extrapolations_h);

  // Check to make sure we have inputs
  if ( jets_h->size() == 0 ) return;
  // Check to make sure the inputs are calo jets
  reco::CaloJet const * caloJet0 = dynamic_cast<reco::CaloJet const *>( & (jets_h->at(0)) );
  // Disallowed non-CaloJet inputs
  if ( caloJet0 == 0 ) {
    throw cms::Exception("InvalidInput") << " Jet-track association is only defined for CaloJets.";
  }
  
  std::auto_ptr<reco::JetTracksAssociation::Container> jetTracks (new reco::JetTracksAssociation::Container (reco::JetRefBaseProd(jets_h)));


  // format inputs
  std::vector <edm::RefToBase<reco::Jet> > allJets;
  allJets.reserve (jets_h->size());
  for (unsigned i = 0; i < jets_h->size(); ++i) allJets.push_back (jets_h->refAt(i));
  mAssociator.produce (&*jetTracks, allJets, *extrapolations_h, *pGeo, dR_ );


  // store output
  fEvent.put (jetTracks);
}


