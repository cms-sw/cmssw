#ifndef DataFormats_BTauReco_JTATagInfo_h
#define DataFormats_BTauReco_JTATagInfo_h

#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JTATagInfoFwd.h"

namespace reco {
 
class JTATagInfo : public BaseTagInfo {
public:
  
  JTATagInfo(void) : m_jetTracksAssociation() { }
  JTATagInfo(const JetTracksAssociationRef & jtaRef) : m_jetTracksAssociation(jtaRef) { }

  virtual ~JTATagInfo(void) { }
  
  virtual JTATagInfo* clone(void) const { return new JTATagInfo(*this); }

  virtual edm::RefToBase<Jet>     jet(void)    const { return m_jetTracksAssociation->first ; }
  virtual TrackRefVector          tracks(void) const { return m_jetTracksAssociation->second; }
  const JetTracksAssociationRef & jtaRef(void) const { return m_jetTracksAssociation; }

  virtual bool hasTracks(void) const { return true; }
  
  void setJTARef(const JetTracksAssociationRef & jtaRef) { m_jetTracksAssociation = jtaRef; } 
  
protected:
  JetTracksAssociationRef m_jetTracksAssociation;
};

}

#endif // DataFormats_BTauReco_JTATagInfo_h
