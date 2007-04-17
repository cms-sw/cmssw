#ifndef DataFormats_BTauReco_BaseTagInfo_h
#define DataFormats_BTauReco_BaseTagInfo_h

#include "DataFormats/BTauReco/interface/BaseTagInfoFwd.h"

#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
 
class BaseTagInfo {
public:
  
  BaseTagInfo() : m_jetTracksAssociation() { }
  BaseTagInfo(const JetTracksAssociationRef & jtaRef) : m_jetTracksAssociation(jtaRef) { }

  virtual ~BaseTagInfo(void) {}
  
  virtual BaseTagInfo* clone(void) const { return new BaseTagInfo(*this); }

  void  setJTARef(const JetTracksAssociationRef & jtaRef) { m_jetTracksAssociation=jtaRef;  } 

  const JetTracksAssociationRef & jtaRef(void)         const { return m_jetTracksAssociation; }
  const Jet & jet() const { return *m_jetTracksAssociation->key; }
  const edm::RefVector<TrackCollection> & tracks() const { return m_jetTracksAssociation->val; }
	
  
protected:
  JetTracksAssociationRef m_jetTracksAssociation;
};

}

#endif // DataFormats_BTauReco_BaseTagInfo_h
