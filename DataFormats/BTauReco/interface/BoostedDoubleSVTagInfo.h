#ifndef DataFormats_BTauReco_BoostedDoubleSVTagInfo_h
#define DataFormats_BTauReco_BoostedDoubleSVTagInfo_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace reco {

class BoostedDoubleSVTagInfo : public BaseTagInfo {
public:
  BoostedDoubleSVTagInfo(void) { }

  BoostedDoubleSVTagInfo(
    const TaggingVariableList & list,
    const edm::Ref<std::vector<CandSecondaryVertexTagInfo> > & svTagInfoRef) :
      m_list(list),
      m_svTagInfoRef(svTagInfoRef) { }

  ~BoostedDoubleSVTagInfo(void) override { }

  BoostedDoubleSVTagInfo* clone(void) const override { return new BoostedDoubleSVTagInfo(*this); }

  edm::RefToBase<Jet> jet(void) const override { return m_svTagInfoRef->jet(); }

  TaggingVariableList taggingVariables(void) const override { return m_list; }

protected:
  TaggingVariableList                                 m_list;
  edm::Ref<std::vector<CandSecondaryVertexTagInfo> >  m_svTagInfoRef;
};

DECLARE_EDM_REFS( BoostedDoubleSVTagInfo )

}

#endif // DataFormats_BTauReco_BoostedDoubleSVTagInfo_h
