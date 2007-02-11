#ifndef BTauReco_TauMassTagInfo_h
#define BTauReco_TauMassTagInfo_h

#include <vector>
#include <map>

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/TauMassTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {
 
  class TauMassTagInfo : public BaseTagInfo {
  public:

    typedef edm::AssociationMap < edm::OneToValue<BasicClusterCollection,
      float, unsigned short> > ClusterTrackAssociationCollection;

    TauMassTagInfo() {}
    virtual ~TauMassTagInfo() {}
    
    virtual TauMassTagInfo* clone() const { return new TauMassTagInfo( * this ); }
    
    //default discriminator: returns the discriminator of the jet tag
    using BaseTagInfo::discriminator;
    
    float discriminator(const double rm_cone,const double pt_cut,const double rs_cone,
                        const double track_cone,const double m_cut) const;
    
    void  setIsolatedTauTag(const IsolatedTauTagInfoRef);
    const IsolatedTauTagInfoRef& getIsolatedTauTag() const;

    void storeClusterTrackCollection(reco::BasicClusterRef clusterRef,float dr);

    double getInvariantMass(const double rm_cone, const double pt_cut,
                          const double rs_cone,const double track_cone) const;    

  private:
    IsolatedTauTagInfoRef             isolatedTau;
    ClusterTrackAssociationCollection clusterMap; // const?

  };
 
}
#endif


