#ifndef JetReco_JetToFloatAssociation_h
#define JetReco_JetToFloatAssociation_h

/** \class JetToFloatAssociation
 *
 * \short Association between jets and float value
 *
 * \author Giovanni Petrucciani, Fedor Ratnikov, June 12, 2007
 *
 * \version   $Id: JetToFloatAssociation.h,v 1.1 2007/06/12 19:19:12 fedor Exp $
 ************************************************************/

#include "DataFormats/JetReco/interface/Jet.h"

namespace reco {
  namespace JetToFloatAssociation {
    typedef edm::RefToBase<reco::Jet> Object;
    typedef std::vector <std::pair <Object, float> > Container;
    typedef std::vector <Object> Objects;
    /// associate jet with value. Returns false and associate nothing if jet is already associated
    bool setValue (Container&, const edm::RefToBase<reco::Jet>&, float);
    bool setValue (Container*, const edm::RefToBase<reco::Jet>&, float);
    /// get value for the association. Throw exception if no association found
    float getValue (const Container&, const edm::RefToBase<reco::Jet>&);
    /// fill list of all jets associated with values. Return # of jets in the list
    Objects allJets (const Container&);
    /// check if jet is associated
    bool hasJet (const Container&, const edm::RefToBase<reco::Jet>&);
  }
}

#endif
