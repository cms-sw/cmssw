#ifndef JetReco_JetFloatAssociation_h
#define JetReco_JetFloatAssociation_h

/** \class JetFloatAssociation
 *
 * \short Association between jets and float value
 *
 * \author Giovanni Petrucciani, Fedor Ratnikov, June 12, 2007
 *
 * \version   $Id: JetFloatAssociation.h,v 1.2 2008/05/21 00:27:45 fedor Exp $
 ************************************************************/

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace fwlite {
  class Event;
}

namespace reco {
  namespace JetFloatAssociation {
    typedef float Value;
    typedef std::vector<Value> Values;
    typedef edm::AssociationVector<reco::JetRefBaseProd, Values> Container;
    typedef Container::value_type value_type;
    typedef Container::transient_vector_type transient_vector_type;
    typedef edm::Ref <Container> Ref;
    typedef edm::RefProd <Container> RefProd;
    typedef edm::RefVector <Container> RefVector;

    /// get value for the association. Throw exception if no association found
    float getValue (const Container&, const reco::JetBaseRef&);
    /// get value for the association. Throw exception if no association found
    float getValue (const Container&, const reco::Jet&);
    /// associate jet with value
    bool setValue (Container&, const reco::JetBaseRef&, float);
    /// associate jet with value
    bool setValue (Container*, const reco::JetBaseRef&, float);
    /// fill list of all jets associated with values. Return # of jets in the list
    std::vector<reco::JetBaseRef > allJets (const Container&);
    /// check if jet is associated
    bool hasJet (const Container&, const reco::JetBaseRef&);
    /// check if jet is associated
    bool hasJet (const Container&, const reco::Jet&);
  }
}

#endif
