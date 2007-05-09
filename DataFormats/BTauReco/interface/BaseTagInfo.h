#ifndef DataFormats_BTauReco_BaseTagInfo_h
#define DataFormats_BTauReco_BaseTagInfo_h

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/BaseTagInfoFwd.h"

namespace reco {
 
class BaseTagInfo {
public:
  BaseTagInfo(void) { }

  virtual ~BaseTagInfo(void) { }
  
  /// returns a polymorphic reference to the tagget Jet
//  virtual const edm::RefToBase<Jet> & jet(void) const = 0;
  virtual edm::RefToBase<Jet>  jet(void) const  {return edm::RefToBase<Jet>() ; }

  /// returns a description of the extended informations in a TaggingVariableList
  virtual TaggingVariableList taggingVariables(void) const {
    // if this is called often, we can cache the results un return a reference
    return TaggingVariableList();
  }
};

}

#endif // DataFormats_BTauReco_BaseTagInfo_h
