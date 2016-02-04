#ifndef JetReco_JetExtendedAssociation_h
#define JetReco_JetExtendedAssociation_h

/** \class JetExtendedAssociation
 *
 * \short Association between jets and extended Jet information
 *
 * \author Fedor Ratnikov, Sept. 9, 2007
 *
 * \version   $Id: JetExtendedAssociation.h,v 1.5 2008/05/21 00:27:45 fedor Exp $
 ************************************************************/

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Math/interface/LorentzVector.h"

namespace fwlite {
  class Event;
}

namespace reco {
  namespace JetExtendedAssociation {
    class JetExtendedData;
    typedef math::PtEtaPhiELorentzVectorF LorentzVector;
    typedef reco::JetExtendedAssociation::JetExtendedData Value;
    typedef std::vector<Value> Values;
    typedef edm::AssociationVector<reco::JetRefBaseProd, Values> Container;
    typedef Container::value_type value_type;
    typedef Container::transient_vector_type transient_vector_type;
    typedef edm::Ref <Container> Ref;
    typedef edm::RefProd <Container> RefProd;
    typedef edm::RefVector <Container> RefVector;


    /// Number of tracks associated in the vertex
    int tracksAtVertexNumber (const Container&, const reco::JetBaseRef&);
    /// Number of tracks associated in the vertex
    int tracksAtVertexNumber (const Container&, const reco::Jet&);
    /// p4 of tracks associated in the vertex
    const LorentzVector& tracksAtVertexP4 (const Container&, const reco::JetBaseRef&);
    const LorentzVector& tracksAtVertexP4 (const Container&, const reco::Jet&);
    /// Number of tracks associated at calo face
    int tracksAtCaloNumber (const Container&, const reco::JetBaseRef&);
    /// Number of tracks associated at calo face
    int tracksAtCaloNumber (const Container&, const reco::Jet&);
    /// p4 of tracks associated at calo face
    const LorentzVector& tracksAtCaloP4 (const Container&, const reco::JetBaseRef&);
    /// p4 of tracks associated at calo face
    const LorentzVector& tracksAtCaloP4 (const Container&, const reco::Jet&);

    /// associate jet with value. Returns false and associate nothing if jet is already associated
    bool setValue (Container&, const reco::JetBaseRef&, const JetExtendedData&);
    /// associate jet with value. Returns false and associate nothing if jet is already associated
    bool setValue (Container*, const reco::JetBaseRef&, const JetExtendedData&);
    /// get value for the association. Throw exception if no association found
    const JetExtendedData& getValue (const Container&, const reco::JetBaseRef&);
    /// get value for the association. Throw exception if no association found
    const JetExtendedData& getValue (const Container&, const reco::Jet&);
    /// fill list of all jets associated with values. Return # of jets in the list
    std::vector<reco::JetBaseRef > allJets (const Container&);
    /// check if jet is associated
    bool hasJet (const Container&, const reco::JetBaseRef&);
    /// check if jet is associated
    bool hasJet (const Container&, const reco::Jet&);

    class JetExtendedData {
    public:
      JetExtendedData ();
      ~JetExtendedData () {}
      int mTracksAtVertexNumber;
      LorentzVector mTracksAtVertexP4;
      int mTracksAtCaloNumber;
      LorentzVector mTracksAtCaloP4;
    };
  }
}

#endif
