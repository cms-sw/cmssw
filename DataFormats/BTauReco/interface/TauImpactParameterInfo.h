#ifndef BTauReco_TauTagImpactParameterInfo_h
#define BTauReco_TauTagImpactParameterInfo_h

#include <vector>
#include <map>

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TauImpactParameterInfoFwd.h"
#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackTauImpactParameterAssociation.h"

using namespace std;

namespace reco {
 
  struct TauImpactParameterTrackData {
    TauImpactParameterTrackData() { }
    Measurement1D  transverseIp;
    Measurement1D  ip3D;
  };
 
  class TauImpactParameterInfo {
  public:
    TauImpactParameterInfo() {}
    virtual ~TauImpactParameterInfo() {}
    
    virtual TauImpactParameterInfo* clone() const { return new TauImpactParameterInfo( * this ); }
    
    double discriminator(double,double,double,bool,bool) const;
    double discriminator() const;
    
    const TauImpactParameterTrackData* getTrackData(reco::TrackRef) const;
    void 	         storeTrackData(reco::TrackRef,const TauImpactParameterTrackData&);
    
    void		 setJetTag(const JetTagRef);
    const JetTagRef& getJetTag() const;
    
    void 		 setIsolatedTauTag(const IsolatedTauTagInfoRef);
    const IsolatedTauTagInfoRef& getIsolatedTauTag() const;
    
  private:
    
    TrackTauImpactParameterAssociationCollection trackDataMap;
    JetTagRef jetTag;
    IsolatedTauTagInfoRef isolatedTaus;
  };
 
}
#endif


