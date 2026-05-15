#ifndef DataFormats_BTauReco_JetTagInfo_h
#define DataFormats_BTauReco_JetTagInfo_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"

namespace reco {
  namespace io_v1 {

    class JetTagInfo : public BaseTagInfo {
    public:
      JetTagInfo(void) : m_jet() {}

      template <typename T>
      JetTagInfo(const edm::Ref<T>& jetRef) : m_jet(jetRef) {}

      JetTagInfo(const edm::RefToBase<Jet>& jetRef) : m_jet(jetRef) {}

      ~JetTagInfo(void) override {}

      JetTagInfo* clone(void) const override { return new JetTagInfo(*this); }

      edm::RefToBase<Jet> jet(void) const override { return m_jet; }

      template <typename T>
      void setJetRef(const edm::Ref<T>& jetRef) {
        m_jet = edm::RefToBase<Jet>(jetRef);
      }

      void setJetRef(const edm::RefToBase<Jet>& jetRef) { m_jet = edm::RefToBase<Jet>(jetRef); }

    protected:
      edm::RefToBase<Jet> m_jet;
    };

  }  // namespace io_v1
  using JetTagInfo = io_v1::JetTagInfo;
  DECLARE_EDM_REFS(JetTagInfo)
}  // namespace reco

#endif  // DataFormats_BTauReco_JetTagInfo_h
