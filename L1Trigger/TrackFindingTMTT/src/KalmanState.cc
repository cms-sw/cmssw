#include "L1Trigger/TrackFindingTMTT/interface/KalmanState.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubCluster.h"
#include <TMatrixD.h>

namespace TMTT {

KalmanState::KalmanState(): kLayerNext_(0), layerId_(0), xa_(0), pxxa_(), K_(), dcov_(), stubCluster_(0), chi2rphi_(0), chi2rz_(0), fitter_(0), fXtoTrackParams_(0), barrel_(true), n_skipped_(0){
}

KalmanState::KalmanState( const L1track3D& candidate, unsigned n_skipped, unsigned kLayer_next, unsigned layerId, const KalmanState *last_state, 
	const std::vector<double> &x, const TMatrixD &pxx, const TMatrixD &K, const TMatrixD &dcov, 
	const StubCluster* stubCluster, double chi2rphi, double chi2rz,
	L1KalmanComb *fitter, GET_TRACK_PARAMS f ){

    l1track3D_ = candidate;
    n_skipped_ = n_skipped;
    kLayerNext_ = kLayer_next;
    layerId_ = layerId;
    last_state_ = last_state;
    xa_ = x;
    pxxa_.Clear();
    pxxa_.ResizeTo( pxx.GetNrows(), pxx.GetNcols() );
    pxxa_ = pxx;
    K_.ResizeTo( K.GetNrows(), K.GetNcols() );
    K_ = K;
    dcov_.ResizeTo( dcov.GetNrows(), dcov.GetNcols() );
    dcov_ = dcov;
    stubCluster_ = stubCluster;
    chi2rphi_ = chi2rphi;
    chi2rz_   = chi2rz;
    kalmanChi2RphiScale_ = fitter->getSettings()->kalmanChi2RphiScale();

    hitPattern_ = 0;
    if (last_state != nullptr) hitPattern_ = last_state->hitPattern(); // Bit encoded list of hit layers
    if (stubCluster != nullptr) hitPattern_ |= (1 << (stubCluster->layerKF()));

    // EJC CLANG complains about this line, 
    // const KalmanState *state = this;

    r_ = 0.1;
    z_ = 0;
    barrel_ = true;
    endcapRing_ = 0;

    if( stubCluster ) {
	r_ = stubCluster->r();
	z_ = stubCluster->z();
	barrel_ = stubCluster->barrel();
	endcapRing_ = stubCluster->endcapRing();
    }

    n_stubs_ = kLayerNext_ - n_skipped_;

    fitter_ = fitter;
    fXtoTrackParams_ = f;
}

KalmanState::KalmanState(const KalmanState &p){

    l1track3D_ = p.candidate();
    n_skipped_ = p.nSkippedLayers();
    kLayerNext_ = p.nextLayer();
    layerId_ = p.layerId();
    endcapRing_ = p.endcapRing();
    r_ = p.r();
    z_ = p.z();
    last_state_ = p.last_state();
    xa_ = p.xa();
    pxxa_ = p.pxxa();
    K_ = p.K();
    dcov_ = p.dcov();
    stubCluster_ = p.stubCluster();
    chi2rphi_ = p.chi2rphi();
    chi2rz_ = p.chi2rz();
    n_stubs_ = p.nStubLayers();
    fitter_ = p.fitter();
    fXtoTrackParams_ = p.fXtoTrackParams();
    barrel_ = p.barrel();
}

KalmanState & KalmanState::operator=( const KalmanState &other )
{
    if (&other == this)
	return *this;

    l1track3D_ = other.candidate();
    n_skipped_ = other.nSkippedLayers();
    kLayerNext_ = other.nextLayer();
    layerId_ = other.layerId();
    endcapRing_ = other.endcapRing();
    r_ = other.r();
    z_ = other.z();
    last_state_ = other.last_state();
    xa_ = other.xa();
    pxxa_ = other.pxxa();
    K_ = other.K();
    dcov_ = other.dcov();
    stubCluster_ = other.stubCluster();
    chi2rphi_ = other.chi2rphi();
    chi2rz_ = other.chi2rz();
    n_stubs_ = other.nStubLayers();
    fitter_ = other.fitter();
    fXtoTrackParams_ = other.fXtoTrackParams();
    barrel_ = other.barrel();
    return *this;
}

bool KalmanState::good( const TP *tp )const{

    const KalmanState *state = this;
    while( state ){
	const StubCluster *stubCluster = state->stubCluster();
	if( stubCluster ){
	    set<const TP*> tps = stubCluster->assocTPs();

	    if( tps.find(tp) == tps.end() ) return false; 
	}
	state = state->last_state();
    }
    return true;
}

double KalmanState::reducedChi2() const
{ 
    if( 2 * n_stubs_ - xa_.size() > 0 ) return (this->chi2())/ ( 2 * n_stubs_ - xa_.size() ); 
    else return 0; 
} 

const KalmanState *KalmanState::last_update_state()const
{
    const KalmanState *state = this;
    while( state ){
	if( state->stubCluster() ) return state;
	state = state->last_state();
    }
    return 0;
}

std::vector<const Stub *> KalmanState::stubs()const
{
    std::vector<const Stub *> all_stubs;

    const KalmanState *state = this;
    while( state ){
	const StubCluster *stbcl = state->stubCluster();
	if( stbcl ){
	    std::vector<const Stub *> stubs = stbcl->stubs();
	    for( unsigned i=0; i < stubs.size(); i++ ){
		all_stubs.push_back( stubs.at(i) ); 
	    }
	}
	state = state->last_state();
    }
    std::reverse(all_stubs.begin(), all_stubs.end()); // Put innermost stub first.
    return all_stubs;
}

bool KalmanState::order(const KalmanState *left, const KalmanState *right){ return (left->nStubLayers() > right->nStubLayers()); }

bool KalmanState::orderMinSkipChi2(const KalmanState *left, const KalmanState *right){ 
  return ( left->chi2scaled()*(left->nSkippedLayers()+1) < right->chi2scaled()*(right->nSkippedLayers()+1) );
}

bool KalmanState::orderChi2(const KalmanState *left, const KalmanState *right){ 
  return ( left->chi2scaled() < right->chi2scaled() );
}

void KalmanState::dump( ostream &os, const TP *tp, bool all )const
{
    std::map<std::string, double> tp_x;
    bool useForAlgEff(false);
    if( tp ){
	useForAlgEff = tp->useForAlgEff();
	tp_x["qOverPt"] = tp->qOverPt();
	tp_x["phi0"] = tp->phi0();
	tp_x["z0"] = tp->z0();
	tp_x["t"] = tp->tanLambda();
	tp_x["d0"] = tp->d0();
    }
    std::map<std::string, double> y = fXtoTrackParams_( fitter_, this );

    os << "KalmanState : ";
    os << "next Kalman layer = " << kLayerNext_ << ", ";
    os << "layerId = " << layerId_ << ", ";
    os << " n_skipped = " << n_skipped_ << ", ";
    os << "barrel = " << barrel_ << ", ";
    os << "endcapRing = " << endcapRing_ << ", ";
    os << "r = " << r_ << ", "; 
    os << "z = " << z_ << ", ";
    for( auto pair : y ){
	os << pair.first << ":" << y[pair.first] << " "; 
    }
    os << endl;
    os << "xa = ( ";
    for( unsigned i=0; i<xa_.size()-1; i++ ) os << xa_[i] << ", ";
    os << xa_.back() << " )" << endl;

    os << "xcov" << endl;
    pxxa_.Print(); 
    os << " chi2rphi = " << chi2rphi_ << ", "; 
    os << " chi2rz = " << chi2rz_ << ", "; 
    os << " # of stublayers = " << n_stubs_ << endl;
    std::vector<const Stub *> stub_list = stubs();
    for( auto &stub : stub_list ){
	os << "              stub ";
	//	os << "[" << stub << "] "; 
	os << "index : " << stub->index() << " ";
	os << "layerId : " << stub->layerId() << " ";
	os  << "[r,phi,z] = ";
	os << "[" << stub->r() << ", " << stub->phi() << ", " << stub->z() << "] ";
	os << " assoc TP indices = [ "; 
	std::set<const TP*> tps = stub->assocTPs();
	for( auto tp : tps ) os << tp->index() << " "; 
	os << "] ";
	os << endl;
    }
    if( tp ){
	os << "\tTP index = " << tp->index() << " useForAlgEff = " << useForAlgEff << " ";
	os << "rel. residual ";
	for( auto pair : tp_x ){
	    os << pair.first << ":" << ( y[pair.first] - pair.second ) / pair.second << " "; 
	}
    }
    else{
	os << "\tTP index = "; 
    }
    os << endl;

    if( stubCluster_ ){
	os << "\tstub [r,phi,z] = ";
	os << "[" << stubCluster_->r() << ", " << stubCluster_->phi() << ", " << stubCluster_->z() << "] ";
	os << " assoc TP indices = [ "; 
	std::set<const TP*> tps = stubCluster_->assocTPs();
	for( auto tp : tps ) os << tp->index() << " "; 
	os << "] ";
    }
    else{
	os << "\tvirtual stub";
    }
    os << endl;

    if( all ){
	const KalmanState *state = last_state();
	if( state ){
	    state->dump( os, tp, all );
	    // state = state->last_state();
	}
	else return;
    }
}

}

