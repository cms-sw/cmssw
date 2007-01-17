#ifndef BTauReco_TauMassTagInfo_h
#define BTauReco_TauMassTagInfo_h

#include <vector>
#include <map>

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TauMassTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {
 
 
  class TauMassTagInfo {
  public:

    typedef edm::AssociationMap < edm::OneToValue<BasicClusterCollection,
      float, unsigned short> > ClusterTrackAssociationCollection;

    TauMassTagInfo() {}
    virtual ~TauMassTagInfo() {}
    
    virtual TauMassTagInfo* clone() const { return new TauMassTagInfo( * this ); }
    
    
    double discriminator(const double rm_cone,const double pt_cut,const double rs_cone,
                         const double track_cone,const double m_cut) const;
    //default discriminator: returns the discriminator of the jet tag, 

    double discriminator() const;
    
    void  setIsolatedTauTag(const IsolatedTauTagInfoRef);
    const IsolatedTauTagInfoRef& getIsolatedTauTag() const;

    void storeClusterTrackCollection(reco::BasicClusterRef clusterRef,float dr);

    void setJetTag(const JetTagRef jetRef);
    const JetTagRef & getJetTag() const;

    double getInvariantMass(const double rm_cone, const double pt_cut,
                          const double rs_cone,const double track_cone) const;    

  private:
    

    JetTagRef             jetTag;
    IsolatedTauTagInfoRef isolatedTau;
    ClusterTrackAssociationCollection clusterMap; // const?

  };
 
}
#endif


