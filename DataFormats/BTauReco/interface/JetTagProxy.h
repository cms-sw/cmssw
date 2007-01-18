#ifndef DataFormats_BTauReco_JetTagProxy_h
#define DataFormats_BTauReco_JetTagProxy_h

#include "DataFormats/BTauReco/interface/JetTag.h"

namespace reco {
 
class JetTagProxy {
public:
  JetTagProxy() : m_jetTag() { }

  virtual ~JetTagProxy(void) {}
  
  virtual JetTagProxy* clone(void) const { return new JetTagProxy(*this); }

  // set the JetTag reference
  void setJetTag(const JetTagRef & ref) { m_jetTag = ref; }

  // get the JetTag reference
  const JetTagRef & getJetTag(void) const { return m_jetTag; }

  // forward to JetTag methods
  double discriminator(void)                           const { return m_jetTag->discriminator(); }
  const Jet & jet(void)                                const { return m_jetTag->jet(); }
  const edm::RefVector<TrackCollection> & tracks(void) const { return m_jetTag->tracks(); }
  const JetTracksAssociationRef & jtaRef(void)         const { return m_jetTag->jtaRef(); }
  
protected:
  JetTagRef m_jetTag;
};

}

#endif // DataFormats_BTauReco_JetTagProxy_h
