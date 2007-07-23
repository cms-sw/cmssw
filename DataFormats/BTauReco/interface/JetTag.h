#ifndef BTauReco_JetTag_h
#define BTauReco_JetTag_h
// \class JetTag
// 
// \short base class for persistent tagging result 
// JetTag is a simple class with a reference to a jet, it's extended tagging niformations, and a tagging discriminant
// 
//
// \author Marcel Vos, Andrea Rizzi, Andrea Bocci based on ORCA version by Christian Weiser, Andrea Rizzi
// \version first version on January 12, 2006

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"

namespace reco {
  class JetTag {
  public:
    JetTag() : m_discriminator(0) { }
    JetTag(double discriminator) : m_discriminator(discriminator){ }
    virtual ~JetTag() { }
    virtual JetTag* clone() const { return new JetTag( * this ); }
    
    void setTagInfo(const edm::RefToBase<BaseTagInfo> & ref) { m_tagInfo = ref; }

    float                       discriminator(void) const { return m_discriminator; }  
    edm::RefToBase<BaseTagInfo> tagInfoRef(void)    const { return m_tagInfo; }
    edm::RefToBase<Jet>         jet(void)           const { return m_tagInfo->jet(); }
    TrackRefVector              tracks(void)        const { return m_tagInfo->tracks(); }
/*           const JTATagInfo * jta = dynamic_cast<const JTATagInfo *> (m_tagInfo.get());
        if(jta){
          return jta->tracks();
        } else {
          return TrackRefVector();
        }*/
    
  private:
    double m_discriminator;
    edm::RefToBase<BaseTagInfo> m_tagInfo;
  };
  
}
#endif
