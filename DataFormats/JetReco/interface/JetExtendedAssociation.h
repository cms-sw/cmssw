#ifndef JetReco_JetExtendedAssociation_h
#define JetReco_JetExtendedAssociation_h

/** \class JetExtendedAssociation
 *
 * \short Association between jets and extended Jet information
 *
 * \author Fedor Ratnikov, Sept. 9, 2007
 *
 * \version   $Id: JetExtendedAssociation.h,v 1.5 2007/08/24 17:35:23 fedor Exp $
 ************************************************************/

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Math/interface/LorentzVector.h"

namespace reco {
  namespace JetExtendedAssociation {
    class JetExtendedData;
    typedef math::PtEtaPhiELorentzVectorF LorentzVector;
    typedef std::pair <edm::RefToBase<reco::Jet>, reco::JetExtendedAssociation::JetExtendedData> Object;
    typedef std::vector <Object> Container;
    typedef edm::Ref <Container> Ref;
    typedef edm::RefProd <Container> RefProd;
    typedef edm::RefVector <Container> RefVector;


    /// Number of tracks associated in the vertex
    int tracksInVertexNumber (const Container&, const edm::RefToBase<reco::Jet>&);
    int tracksInVertexNumber (const Container&, const reco::Jet&);
    /// p4 of tracks associated in the vertex
    const LorentzVector& tracksInVertexP4 (const Container&, const edm::RefToBase<reco::Jet>&);
    const LorentzVector& tracksInVertexP4 (const Container&, const reco::Jet&);
    /// Number of tracks associated at calo face
    int tracksAtCaloNumber (const Container&, const edm::RefToBase<reco::Jet>&);
    int tracksAtCaloNumber (const Container&, const reco::Jet&);
    /// p4 of tracks associated at calo face
    const LorentzVector& tracksAtCaloP4 (const Container&, const edm::RefToBase<reco::Jet>&);
    const LorentzVector& tracksAtCaloP4 (const Container&, const reco::Jet&);

    /// associate jet with value. Returns false and associate nothing if jet is already associated
    bool setValue (Container&, const edm::RefToBase<reco::Jet>&, const JetExtendedData&);
    bool setValue (Container*, const edm::RefToBase<reco::Jet>&, const JetExtendedData&);
    /// get value for the association. Throw exception if no association found
    const JetExtendedData& getValue (const Container&, const edm::RefToBase<reco::Jet>&);
    const JetExtendedData& getValue (const Container&, const reco::Jet&);
    /// get editable value for the association. Return 0 if no association found
    JetExtendedData* getValue (Container*, const edm::RefToBase<reco::Jet>&);
    /// fill list of all jets associated with values. Return # of jets in the list
    std::vector<edm::RefToBase<reco::Jet> > allJets (const Container&);
    /// check if jet is associated
    bool hasJet (const Container&, const edm::RefToBase<reco::Jet>&);

    class JetExtendedData {
    public:
      JetExtendedData ();
      ~JetExtendedData () {}
      int mTracksInVertexNumber;
      LorentzVector mTracksInVertexP4;
      int mTracksAtCaloNumber;
      LorentzVector mTracksAtCaloP4;
    };
  }
}

#endif
