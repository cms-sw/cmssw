#ifndef DataFormats_BTauReco_ShallowTagInfo_h
#define DataFormats_BTauReco_ShallowTagInfo_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace reco {

class ShallowTagInfo : public BaseTagInfo {
public:
  ShallowTagInfo(void) { }

  ShallowTagInfo(
    const TaggingVariableList & list,
    const edm::RefToBase<Jet> & jetref) :
      list_(list),
      jetRef_(jetref) { }

  virtual ~ShallowTagInfo(void) { }

  virtual ShallowTagInfo* clone(void) const { return new ShallowTagInfo(*this); }

  virtual edm::RefToBase<Jet> jet(void) const { return jetRef_; }

  virtual TaggingVariableList taggingVariables(void) const { return list_; }

protected:
  /*const*/ TaggingVariableList  list_;
	/*const*/ edm::RefToBase<Jet>  jetRef_;
};

DECLARE_EDM_REFS( ShallowTagInfo )

}

#endif // DataFormats_BTauReco_ShallowTagInfo_h
