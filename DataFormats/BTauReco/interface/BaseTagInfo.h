#ifndef DataFormats_BTauReco_BaseTagInfo_h
#define DataFormats_BTauReco_BaseTagInfo_h

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
 
class BaseTagInfo {
public:
  BaseTagInfo(void) { }

  virtual ~BaseTagInfo(void) { }
  
  /// returns a polymorphic reference to the tagged jet
  virtual edm::RefToBase<Jet> jet(void) const { 
    return edm::RefToBase<Jet>() ; 
  }

  /// returns a list of tracks associated to the jet
  virtual TrackRefVector tracks(void) const {
    return TrackRefVector();
  }

  /// check if the algorithm is using the tracks or not
  virtual bool hasTracks(void) const {
    return false;
  }
  
  /// returns a description of the extended informations in a TaggingVariableList
  virtual TaggingVariableList taggingVariables(void) const {
    return TaggingVariableList();
  }
};

}

#endif // DataFormats_BTauReco_BaseTagInfo_h
