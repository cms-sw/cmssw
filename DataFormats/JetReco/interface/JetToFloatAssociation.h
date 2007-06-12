#ifndef JetReco_JetToFloatAssociation_h
#define JetReco_JetToFloatAssociation_h

/** \class JetToFloatAssociation
 *
 * \short Association between jets and float value
 *
 * \author Giovanni Petrucciani, Fedor Ratnikov, June 12, 2007
 *
 * \version   $Id: CaloJet.h,v 1.24 2007/05/08 21:36:51 fedor Exp $
 ************************************************************/

#include "DataFormats/JetReco/interface/Jet.h"

namespace reco {
  namespace JetToFloatAssociation {
    typedef std::vector <std::pair <edm::RefToBase<reco::Jet>, float> > Container;
    /// associate jet with value. Returns false and associate nothing if jet is already associated
    bool setValue (Container&, const edm::RefToBase<reco::Jet>&, float);
    /// get value for the association. Throw exception if no association found
    float getValue (const Container&, const edm::RefToBase<reco::Jet>&);
    /// get list of all jets associated with values
    std::vector<edm::RefToBase<reco::Jet> > allJets (const Container&);
    /// check if jet is associated
    bool hasJet (const Container&, const edm::RefToBase<reco::Jet>&);
  }
}

#endif
