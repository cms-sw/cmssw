#ifndef JetReco_JetTracksAssociation_h
#define JetReco_JetTracksAssociation_h

/** \class JetTracksAssociation
 *
 * \short Association between jets and float value
 *
 * \author Fedor Ratnikov, July 27, 2007
 *
 * \version   $Id: JetTracksAssociation.h,v 1.3 2008/05/21 00:27:45 fedor Exp $
 ************************************************************/

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Math/interface/LorentzVector.h"

namespace fwlite {
  class Event;
}

namespace reco {
  namespace JetTracksAssociation {
    typedef math::PtEtaPhiELorentzVectorF LorentzVector;
    typedef reco::TrackRefVector Value;
    typedef std::vector<Value> Values;
    typedef edm::AssociationVector<reco::JetRefBaseProd, Values> Container;
    typedef Container::value_type value_type;
    typedef Container::transient_vector_type transient_vector_type;
    typedef edm::Ref <Container> Ref;
    typedef edm::RefProd <Container> RefProd;
    typedef edm::RefVector <Container> RefVector;

    /// Get number of tracks associated with jet
    int tracksNumber (const Container&, const reco::JetBaseRef);
    /// Get number of tracks associated with jet
    int tracksNumber (const Container&, const reco::Jet&);
    /// Get LorentzVector as sum of all tracks associated with jet.
    LorentzVector tracksP4 (const Container&, const reco::JetBaseRef);
    /// Get LorentzVector as sum of all tracks associated with jet.
    LorentzVector tracksP4 (const Container&, const reco::Jet&);

    /// associate jet with value. Returns false and associate nothing if jet is already associated
    bool setValue (Container&, const reco::JetBaseRef&, reco::TrackRefVector);
    /// associate jet with value. Returns false and associate nothing if jet is already associated
    bool setValue (Container*, const reco::JetBaseRef&, reco::TrackRefVector);
    /// get value for the association. Throw exception if no association found
    const reco::TrackRefVector& getValue (const Container&, const reco::JetBaseRef&);
    /// get value for the association. Throw exception if no association found
    const reco::TrackRefVector& getValue (const Container&, const reco::Jet&);
    /// fill list of all jets associated with values. Return # of jets in the list
    std::vector<reco::JetBaseRef > allJets (const Container&);
    /// check if jet is associated
    bool hasJet (const Container&, const reco::JetBaseRef&);
    /// check if jet is associated
    bool hasJet (const Container&, const reco::Jet&);
  }
  /// typedefs for backward compatibility
  typedef JetTracksAssociation::Container JetTracksAssociationCollection;
  typedef JetTracksAssociation::Ref JetTracksAssociationRef;
  typedef JetTracksAssociation::RefProd JetTracksAssociationRefProd;
  typedef JetTracksAssociation::RefVector JetTracksAssociationRefVector; 
}

#endif
