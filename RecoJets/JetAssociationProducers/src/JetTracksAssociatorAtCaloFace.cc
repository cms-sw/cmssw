// \class JetTracksAssociatorAtCaloFace JetTracksAssociatorAtCaloFace.cc 
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// Accommodated for Jet Package by: Fedor Ratnikov Jul. 30, 2007
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"



#include "JetTracksAssociatorAtCaloFace.h"

JetTracksAssociatorAtCaloFace::JetTracksAssociatorAtCaloFace(const edm::ParameterSet& fConfig)
  : firstRun(true),
    dR_(fConfig.getParameter<double>("coneSize"))
{
  mJets = consumes<edm::View <reco::Jet> >(fConfig.getParameter<edm::InputTag> ("jets"));
  mExtrapolations  = consumes<std::vector<reco::TrackExtrapolation> >(fConfig.getParameter<edm::InputTag> ("extrapolations")),

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
  fEvent.getByToken (mJets, jets_h);
  edm::Handle <std::vector<reco::TrackExtrapolation> > extrapolations_h;
  fEvent.getByToken (mExtrapolations, extrapolations_h);

  auto jetTracks = std::make_unique<reco::JetTracksAssociation::Container>(reco::JetRefBaseProd(jets_h));

  // Check to make sure we have inputs
  if ( jets_h->size() == 0 ){
    // store output regardless the size of the inputs
    fEvent.put(std::move(jetTracks));
    return;
  }
  // Check to make sure the inputs are calo jets
  reco::CaloJet const * caloJet0 = dynamic_cast<reco::CaloJet const *>( & (jets_h->at(0)) );
  // Disallowed non-CaloJet inputs
  if ( caloJet0 == 0 ) {
    throw cms::Exception("InvalidInput") << " Jet-track association is only defined for CaloJets.";
  }
  

  // format inputs
  std::vector <edm::RefToBase<reco::Jet> > allJets;
  allJets.reserve (jets_h->size());
  for (unsigned i = 0; i < jets_h->size(); ++i) allJets.push_back (jets_h->refAt(i));
  mAssociator.produce (&*jetTracks, allJets, *extrapolations_h, *pGeo, dR_ );


  // store output
  fEvent.put(std::move(jetTracks));
}


