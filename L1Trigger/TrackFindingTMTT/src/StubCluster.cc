#include "L1Trigger/TrackFindingTMTT/interface/StubCluster.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "DataFormats/Math/interface/deltaPhi.h"

namespace TMTT {

StubCluster::StubCluster( std::vector<const Stub *> stubs, double SectorPhi, int lres_dr ) : layerKF_(999)
{
    r_=0; phi_=0; z_=0; eta_=0;
    sigmaX_=0; sigmaZ_=0;
    layerId_=0; endcapRing_=0; barrel_=false; tiltedBarrel_=false; psModule_=false;
    nstubs_ = stubs.size();
    stubs_ = stubs;
    dphi_dl_ = 0;
    dphi_dr_ = 0;
    dphi_ = 0;
    deltai_ = 0;
    alpha_ = 0;
    moduleTilt_ = 0.;

    layerIdReduced_ = stubs.at(0)->layerIdReduced();
    layerId_ = stubs.at(0)->layerId();
    endcapRing_ = stubs.at(0)->endcapRing();
    barrel_ = stubs.at(0)->barrel();
    tiltedBarrel_ = stubs.at(0)->tiltedBarrel();
    psModule_ = stubs.at(0)->psModule();
    moduleTilt_ += stubs.at(0)->moduleTilt();

    if(nstubs_ == 0 ) return;

    double sum_r2(0), sum_z2(0), sum_phi2(0), sum_sigmaZ2(0), sum_dphi2_dr(0), sum_dphi2_dl(0);
    int lsign(0);

    for( auto stub : stubs ){
        r_ += stub->r();
        z_ += stub->z();
	eta_ += stub->eta();
	phi_ += reco::deltaPhi( stub->phi(),  SectorPhi );
	std::set<const TP *> assocTPs = stub->assocTPs();
	if( assocTPs.size() ){
	    assocTPs_.insert( assocTPs.begin(), assocTPs.end() );
	}
	sum_z2    += ( stub->z() * stub->z() ); 
	sum_r2    += ( stub->r() * stub->r() );
	sum_phi2 += ( reco::deltaPhi( stub->phi(), SectorPhi ) * reco::deltaPhi( stub->phi(), SectorPhi ) );

	double dphi_dl = stub->sigmaX() / stub->r();
	sum_dphi2_dl += ( dphi_dl * dphi_dl );

	int delta_i(0);
	double delta_l(0);
	if( 0 < lres_dr && lres_dr < (int)stub->nStrips() / 3 ) {
	    delta_i =  ( stub->iphi() - 0.5 * stub->nStrips() ) / lres_dr; 
	    delta_l = ( delta_i + 0.5 ) * stub->stripPitch() * lres_dr;
	}
	else if( lres_dr == 0 ){
	    delta_i = 0;
	    delta_l = 0;
	}
	else{
	    delta_i =  ( stub->iphi() - 0.5 * stub->nStrips() ); 
	    delta_i = delta_i > 0 ? 0.5 * stub->nStrips() - 1 : -0.5 * stub->nStrips();
	    delta_l = ( delta_i + 0.5 ) * stub->stripPitch();   
	}

	deltai_ += delta_i;
	deltal_ += delta_l;
	lsign += ( delta_i > 0 ? +1 : -1 ); 

	if( !stub->barrel() ){ 
	    double dphiOverdr = delta_l / ( stub->r() * stub->r() ); 
	    double dphi_dr = dphiOverdr * stub->sigmaZ();
	    sum_dphi2_dr += ( dphi_dr * dphi_dr ); 
	}
	sum_sigmaZ2 += ( stub->sigmaZ() * stub->sigmaZ() );

        alpha_ += stub->alpha();
    }

    r_ /= nstubs_;
    z_ /= nstubs_;
    eta_ /= nstubs_;
    phi_ /= nstubs_;
    deltal_ /= nstubs_;
    deltai_ /= nstubs_;

    dphi_dr_ = sqrt( sum_dphi2_dr ) / nstubs_;
    dphi_dl_ = sqrt( sum_dphi2_dl ) / nstubs_;
    if( lsign < 0 ){//if this sign is opposite, 1% eff. reduction is observed.
	dphi_dr_ = -1. * dphi_dr_;
    }

    double vZ(0);
    if( nstubs_ == 1 ){
	sigmaX_ = stubs.front()->sigmaX();
	sigmaZ_ = stubs.front()->sigmaZ();
	dphi_ = sqrt( dphi_dr_ * dphi_dr_ + dphi_dl_ * dphi_dl_ );
    }
    else{
	double vphi = ( sum_phi2 - nstubs_ * phi_ * phi_ ) / ( nstubs_ - 1 );
	dphi_ = sqrt( dphi_dr_ * dphi_dr_ + dphi_dl_ * dphi_dl_ + vphi );

	sigmaX_ = r_ * dphi_; 

	if( stubs.front()->barrel() ){

	    vZ = ( sum_z2 - nstubs_ * z_ * z_ ) / ( nstubs_ - 1 );

	}
	else{
	    vZ = ( sum_r2 - nstubs_ * r_ * r_ ) / ( nstubs_ - 1 );
	}
        sigmaZ_ = sqrt( vZ * vZ + sum_sigmaZ2 / nstubs_ ); 
    }

    phi_ = reco::deltaPhi( phi_ + SectorPhi, 0. );

    alpha_ /= nstubs_;
}

}
