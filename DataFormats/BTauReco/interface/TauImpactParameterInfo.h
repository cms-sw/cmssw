#ifndef BTauReco_TauTagImpactParameterInfo_h
#define BTauReco_TauTagImpactParameterInfo_h

#include <vector>
#include <map>

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TauImpactParameterInfoFwd.h"
#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"

#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

using namespace std;

namespace reco {
 
  class TauImpactParameterInfo {

    public:
	struct TrackData {
	      Measurement1D  transverseIp;
	      Measurement1D  ip3D;
	};

        typedef edm::AssociationMap < edm::OneToValue<std::vector<reco::Track>,
		reco::TauImpactParameterInfo::TrackData, unsigned short> > TrackDataAssociation;

	TauImpactParameterInfo() {}
	virtual ~TauImpactParameterInfo() {}
  
	virtual TauImpactParameterInfo* clone() const { return new TauImpactParameterInfo( * this ); }
  
	double discriminator(double,double,double,bool,bool) const;
        double discriminator() const;

        const TrackData* getTrackData(reco::TrackRef) const;
	void 	         storeTrackData(reco::TrackRef,const TrackData&);

	void		 setJetTag(const JetTagRef);
	const JetTagRef& getJetTag() const;

	void 		 setIsolatedTauTag(const IsolatedTauTagInfoRef);
	const IsolatedTauTagInfoRef& getIsolatedTauTag() const;

    private:

        TrackDataAssociation trackDataMap;
//	map<reco::TrackRef,TrackData> trackDataMap;
	JetTagRef jetTag;
	IsolatedTauTagInfoRef isolatedTaus;
  };

}
#endif


