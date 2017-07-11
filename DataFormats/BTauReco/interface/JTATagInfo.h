#ifndef DataFormats_BTauReco_JTATagInfo_h
#define DataFormats_BTauReco_JTATagInfo_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

namespace reco {
 
class JTATagInfo : public BaseTagInfo {
public:
  
  JTATagInfo() : m_jetTracksAssociation() { }
  JTATagInfo(const JetTracksAssociationRef & jtaRef) : m_jetTracksAssociation(jtaRef) { }

  virtual ~JTATagInfo() { }
  
  virtual JTATagInfo* clone() const { return new JTATagInfo(*this); }

  virtual edm::RefToBase<Jet>     jet()    const { return m_jetTracksAssociation->first ; }
  virtual TrackRefVector          tracks() const { return m_jetTracksAssociation->second; }
  const JetTracksAssociationRef & jtaRef() const { return m_jetTracksAssociation; }

  virtual bool hasTracks() const { return true; }
  
  void setJTARef(const JetTracksAssociationRef & jtaRef) { m_jetTracksAssociation = jtaRef; } 
  
protected:
  JetTracksAssociationRef m_jetTracksAssociation;
};

DECLARE_EDM_REFS( JTATagInfo )

}

#endif // DataFormats_BTauReco_JTATagInfo_h
