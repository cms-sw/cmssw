#include "RecoTauTag/ImpactParameter/interface/ImpactParameterAlgorithm.h"

#include "RecoBTag/BTagTools/interface/SignedTransverseImpactParameter.h"
#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ImpactParameterAlgorithm::ImpactParameterAlgorithm(){
        ip_min   = -9999;
        ip_max   = 9999;
        sip_min  = 0;
        use_sign = false;
	use3D    = false; 
}

ImpactParameterAlgorithm::ImpactParameterAlgorithm(const edm::ParameterSet & parameters){
	ip_min       = parameters.getParameter<double>("TauImpactParameterMin");
	ip_max       = parameters.getParameter<double>("TauImpactParameterMax");
	sip_min	     = parameters.getParameter<double>("TauImpactParameterSignificanceMin");
	use_sign     = parameters.getParameter<bool>("UseTauImpactParameterSign");
	use3D        = parameters.getParameter<bool>("UseTau3DImpactParameter");

}

void ImpactParameterAlgorithm::setTransientTrackBuilder(const TransientTrackBuilder * builder) { 
	transientTrackBuilder = builder; 
}


std::pair<float,reco::TauImpactParameterInfo> ImpactParameterAlgorithm::tag(const reco::IsolatedTauTagInfoRef & tauRef, const reco::Vertex & pv) {

	if(transientTrackBuilder == 0){
	  throw cms::Exception("NullTransientTrackBuilder") << "Transient track builder is 0. ";
	}

        reco::TauImpactParameterInfo resultExtended;
	resultExtended.setIsolatedTauTag(tauRef);

	const reco::Jet* jet = tauRef->jet().get();
	GlobalVector direction(jet->px(),jet->py(),jet->pz());

        const reco::TrackRefVector& tracks = tauRef->selectedTracks();

	edm::RefVector<reco::TrackCollection>::const_iterator iTrack;
	for(iTrack = tracks.begin(); iTrack!= tracks.end(); iTrack++){

          const reco::TransientTrack transientTrack = (transientTrackBuilder->build(&(**iTrack)));

          SignedTransverseImpactParameter stip;
	  Measurement1D ip = stip.apply(transientTrack,direction,pv).second;

	  SignedImpactParameter3D signed_ip3D;
	  Measurement1D ip3D = signed_ip3D.apply(transientTrack,direction,pv).second;
	  LogDebug("ImpactParameterAlgorithm::tag") << "check pv,ip3d " << pv.z() << " " << ip3D.value()  ;
	  if(!use_sign){
	    Measurement1D tmp2D(fabs(ip.value()),ip.error());
	    ip = tmp2D;

	    Measurement1D tmp3D(fabs(ip3D.value()),ip3D.error());
            ip3D = tmp3D;
	  }

          reco::TauImpactParameterTrackData theData;

	  theData.transverseIp = ip;
          theData.ip3D = ip3D;
	  resultExtended.storeTrackData(*iTrack,theData);

	}

	float discriminator = resultExtended.discriminator(ip_min,ip_max,sip_min,use_sign,use3D);

	return std::make_pair( discriminator, resultExtended );
}
