#ifndef DataFormats_BTauReco_BaseTagInfo_h
#define DataFormats_BTauReco_BaseTagInfo_h

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/BaseTagInfoFwd.h"

namespace reco {
 
class BaseTagInfo {
public:
  BaseTagInfo() : m_jetTag() { }

  virtual ~BaseTagInfo(void) {}
  
  virtual BaseTagInfo* clone(void) const { return new BaseTagInfo(*this); }

  // set the JetTag reference
  void setJetTag(const JetTagRef & ref) { m_jetTag = ref; }

  // get the JetTag reference
  const JetTagRef & getJetTag(void) const { return m_jetTag; }

  // forward to JetTag methods
  float discriminator(void)                            const { return m_jetTag->discriminator(); }
  const Jet & jet(void)                                const { return m_jetTag->jet(); }
  const edm::RefVector<TrackCollection> & tracks(void) const { return m_jetTag->tracks(); }
  const JetTracksAssociationRef & jtaRef(void)         const { return m_jetTag->jtaRef(); }

protected:
  JetTagRef m_jetTag;
};

}

#endif // DataFormats_BTauReco_BaseTagInfo_h
