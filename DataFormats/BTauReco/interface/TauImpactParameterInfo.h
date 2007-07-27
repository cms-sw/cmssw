#ifndef DataFormats_BTauReco_TauImpactParameterInfo_h
#define DataFormats_BTauReco_TauImpactParameterInfo_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/TauImpactParameterInfoFwd.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfoFwd.h" 
#include "DataFormats/BTauReco/interface/TauImpactParameterInfoFwd.h" 

namespace reco {
 
  struct TauImpactParameterTrackData {
    Measurement1D transverseIp;
    Measurement1D ip3D;
  };

  typedef edm::AssociationMap <
            edm::OneToValue<
              reco::TrackCollection,
              reco::TauImpactParameterTrackData
            >
          > TrackTauImpactParameterAssociationCollection;

  typedef TrackTauImpactParameterAssociationCollection::value_type TrackTauImpactParameterAssociation;

  class TauImpactParameterInfo {
  public:
    TauImpactParameterInfo() {}
    virtual ~TauImpactParameterInfo() {}
    
    virtual TauImpactParameterInfo* clone() const { return new TauImpactParameterInfo( * this ); }
    
    float discriminator(double,double,double,bool,bool) const;
    float discriminator() const;
    
    const TauImpactParameterTrackData * getTrackData(const reco::TrackRef &) const;
    void storeTrackData(const reco::TrackRef &, const TauImpactParameterTrackData &);
    
    const IsolatedTauTagInfoRef & getIsolatedTauTag() const;
    void setIsolatedTauTag(const IsolatedTauTagInfoRef &);
    
  private:
    TrackTauImpactParameterAssociationCollection trackDataMap;
    IsolatedTauTagInfoRef                        isolatedTaus;
  };
 
}

#endif // DataFormats_BTauReco_TauImpactParameterInfo_h
