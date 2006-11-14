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
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
  class JetTag {
  public:
    JetTag() : m_discriminator(0), m_jetTracksAssociation() {}
    JetTag(double discriminator,JetTracksAssociationRef jetTracks) : 
      m_discriminator(discriminator), m_jetTracksAssociation(jetTracks){ }
    virtual ~JetTag(){}
    virtual JetTag* clone() const { return new JetTag( * this ); }
    double discriminator () const { return m_discriminator; }  
    const Jet & jet() const { return *m_jetTracksAssociation->key; }
    const edm::RefVector<TrackCollection> & tracks() const { return m_jetTracksAssociation->val; } 
    const JetTracksAssociationRef& jtaRef() const { return m_jetTracksAssociation; }

  private:
    double m_discriminator;
    JetTracksAssociationRef m_jetTracksAssociation;
  };
  
}
#endif
