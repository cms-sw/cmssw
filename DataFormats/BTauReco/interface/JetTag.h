#ifndef BTauReco_JetTag_h
#define BTauReco_JetTag_h
// \class JetTag
// 
// \short base class for persistent tagging result 
// JetTag is an pure virtual interface class. Base class for result of all b- and tau-tagging algorithms.
// 
//
// \author Marcel Vos, based on ORCA version by Christian Weiser, Andrea Rizzi
// \version first version on January 12, 2006

#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JetTagFwd.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/Common/interface/RefToBase.h"

namespace reco {
  class JetTag {
  public:
    JetTag() : m_discriminator(0) {}
    JetTag(double discriminator) : 
      m_discriminator(discriminator){ }
    virtual ~JetTag(){}
    virtual JetTag* clone() const { return new JetTag( * this ); }
    double discriminator () const { return m_discriminator; }  

    void setTagInfo(const edm::RefToBase<BaseTagInfo> & ref) { m_tagInfo = ref; }
    const edm::RefToBase<BaseTagInfo> & tagInfoRef(void) const { return m_tagInfo; }
    
    const JetTracksAssociationRef& jtaRef() const { return m_tagInfo->jtaRef(); }
    const Jet & jet(void)                                const { return m_tagInfo->jet(); }
    const edm::RefVector<TrackCollection> & tracks(void) const { return m_tagInfo->tracks(); }

   
  private:
    double m_discriminator;
    edm::RefToBase<BaseTagInfo> m_tagInfo;
  };
  
}
#endif
