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
  BoostedDoubleSVTagInfo() { }

  BoostedDoubleSVTagInfo(
    const TaggingVariableList & list,
    const edm::Ref<std::vector<CandSecondaryVertexTagInfo> > & svTagInfoRef) :
      m_list(list),
      m_svTagInfoRef(svTagInfoRef) { }

  virtual ~BoostedDoubleSVTagInfo() { }

  virtual BoostedDoubleSVTagInfo* clone() const { return new BoostedDoubleSVTagInfo(*this); }

  virtual edm::RefToBase<Jet> jet() const { return m_svTagInfoRef->jet(); }

  virtual TaggingVariableList taggingVariables() const { return m_list; }

protected:
  TaggingVariableList                                 m_list;
  edm::Ref<std::vector<CandSecondaryVertexTagInfo> >  m_svTagInfoRef;
};

DECLARE_EDM_REFS( BoostedDoubleSVTagInfo )

}

#endif // DataFormats_BTauReco_BoostedDoubleSVTagInfo_h
