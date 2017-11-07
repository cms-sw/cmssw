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

  ~ShallowTagInfo(void) override { }

  ShallowTagInfo* clone(void) const override { return new ShallowTagInfo(*this); }

  edm::RefToBase<Jet> jet(void) const override { return jetRef_; }

  TaggingVariableList taggingVariables(void) const override { return list_; }

protected:
  /*const*/ TaggingVariableList  list_;
	/*const*/ edm::RefToBase<Jet>  jetRef_;
};

DECLARE_EDM_REFS( ShallowTagInfo )

}

#endif // DataFormats_BTauReco_ShallowTagInfo_h
