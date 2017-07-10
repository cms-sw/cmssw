#ifndef DataFormats_BTauReco_BaseTagInfo_h
#define DataFormats_BTauReco_BaseTagInfo_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace reco {
 
class BaseTagInfo {
public:
  BaseTagInfo() { }

  virtual ~BaseTagInfo() { }

  /// clone
  virtual BaseTagInfo * clone() const {
    return new BaseTagInfo(*this);
  }
  
  /// returns a polymorphic reference to the tagged jet
  virtual edm::RefToBase<Jet> jet() const { 
    return edm::RefToBase<Jet>() ; 
  }

  /// returns a list of tracks associated to the jet
  virtual TrackRefVector tracks() const {
    return TrackRefVector();
  }

  /// check if the algorithm is using the tracks or not
  virtual bool hasTracks() const {
    return false;
  }
  
  /// returns a description of the extended informations in a TaggingVariableList
  virtual TaggingVariableList taggingVariables() const {
    return TaggingVariableList();
  }
};

DECLARE_EDM_REFS( BaseTagInfo )

}

#endif // DataFormats_BTauReco_BaseTagInfo_h
