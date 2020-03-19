#ifndef DataFormats_BTauReco_JTATagInfo_h
#define DataFormats_BTauReco_JTATagInfo_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

namespace reco {

  class JTATagInfo : public BaseTagInfo {
  public:
    JTATagInfo(void) : m_jetTracksAssociation() {}
    JTATagInfo(const JetTracksAssociationRef& jtaRef) : m_jetTracksAssociation(jtaRef) {}

    ~JTATagInfo(void) override {}

    JTATagInfo* clone(void) const override { return new JTATagInfo(*this); }

    edm::RefToBase<Jet> jet(void) const override { return m_jetTracksAssociation->first; }
    TrackRefVector tracks(void) const override { return m_jetTracksAssociation->second; }
    const JetTracksAssociationRef& jtaRef(void) const { return m_jetTracksAssociation; }

    bool hasTracks(void) const override { return true; }

    void setJTARef(const JetTracksAssociationRef& jtaRef) { m_jetTracksAssociation = jtaRef; }

  protected:
    JetTracksAssociationRef m_jetTracksAssociation;
  };

  DECLARE_EDM_REFS(JTATagInfo)

}  // namespace reco

#endif  // DataFormats_BTauReco_JTATagInfo_h
