///=== This is the base class for the Kalman Combinatorial Filter track fit algorithm.

///=== Written by: S. Summers, K. Uchida, M. Pesaresi

#include "L1Trigger/TrackFindingTMTT/interface/L1KalmanComb.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"

#include <TMatrixD.h> 
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/KalmanState.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubCluster.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <algorithm>
#include <functional>
#include <fstream>
#include <iomanip>
#include <TH2F.h>
//#define CKF_DEBUG
// Enable debug printout to pair with that in Histos.cc enabled by recalc_debug.
//#define RECALC_DEBUG

// Enable merging of nearby stubs.
//#define MERGE_STUBS

namespace TMTT {

unsigned LayerId[16] = { 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25 };

static bool orderStubsByLayer(const Stub* a, const Stub* b){
  return (a->layerId() < b->layerId());
}

#ifdef MERGE_STUBS
static bool orderStubsByZ(const Stub* a, const Stub* b){
  return (a->z() < b->z());
}

static bool orderStubsByR(const Stub* a, const Stub* b){
  return (a->r() < b->r());
}
#endif

void L1KalmanComb::printTPSummary( std::ostream &os, const TP *tp, bool addReturn) const {
	
  if( tp ){
		
    os << "TP ";
    //  os << "addr=" << tp << " ";
    os << "index=" << tp->index() << " ";
    os << "qOverPt=" << tp->qOverPt() << " ";
    os << "phi0=" << tp->phi0() << " ";
    os << "z0=" << tp->z0() << " ";
    os << "t=" << tp->tanLambda() << " ";
    os << "d0=" << tp->d0();
    if( addReturn ) os << endl;
    else os << " | ";
  }
}

void L1KalmanComb::printTP( std::ostream &os, const TP *tp ) const {
        
  std::map<std::string, double> tpParams;
  bool useForAlgEff(false);
  if( tp ){
    useForAlgEff = tp->useForAlgEff();
    tpParams["qOverPt"] = tp->qOverPt();
    tpParams["phi0"] = tp->phi0();
    tpParams["z0"] = tp->z0();
    tpParams["t"] = tp->tanLambda();
    tpParams["d0"] = tp->d0();
  }
  if( tp ){
    os << "  TP index = " << tp->index() << " useForAlgEff = " << useForAlgEff << " ";
    for( auto pair : tpParams ){
      os << pair.first << ":" << pair.second << ", "; 
    }
    os << "  inv2R = " << tp->qOverPt() * getSettings()->invPtToInvR() * 0.5; 
  }
  else{
    os << "  Fake"; 
  }
  os << endl;
}

void L1KalmanComb::printStubLayers( std::ostream &os, std::vector<const Stub *> &stubs ) const {

  if( stubs.size() == 0 ) os << "stub layers = []" << endl;
  else{
    os << "stub layers = [ ";
    for( unsigned i=0; i<stubs.size()-1; i++ ) os << stubs[i]->layerId() << ", ";
    os << stubs.back()->layerId() << " ]" << endl;
  }
}

void L1KalmanComb::printStubCluster( std::ostream &os, const StubCluster * stubCluster, bool addReturn ) const {
  os << "stub: ";
  //   os << "addr=" << stub << " "; 
  os << "index=" << stubCluster->stubs()[0]->index() << " ";
  os << "layer=" << stubCluster->layerId() << " ";
  os << "ring=" << stubCluster->endcapRing() << " ";
  os << "r=" << stubCluster->r() << " ";
  os << "phi=" << stubCluster->phi() << " ";
  os << "z=" << stubCluster->z() << " ";
  os << "sigmaX=" << stubCluster->sigmaX() << " ";
  os << "sigmaZ=" << stubCluster->sigmaZ() << " ";
  os << "dphi_dr=" << stubCluster->dphi_dr() << " ";
  os << "#stubs= " << stubCluster->nStubs() << " ";
  os << "TPids="; 
  std::set<const TP*> tps = stubCluster->assocTPs();
  for( auto tp : tps ) os << tp->index() << ","; 
  if( addReturn ) os << endl;
  else os << " | ";
}

void L1KalmanComb::printStubClusters( std::ostream &os, std::vector<const StubCluster *> &stubClusters ) const {

  for( auto &stubcl : stubClusters ){
    printStubCluster( os, stubcl );
  }
}

void L1KalmanComb::printStub( std::ostream &os, const Stub * stub, bool addReturn ) const {
  os << "stub ";
  //   os << "addr=" << stub << " "; 
  os << "index=" << stub->index() << " ";
  os << "layerId=" << stub->layerId() << " ";
  os << "endcapRing=" << stub->endcapRing() << " ";
  os << "r=" << stub->r() << " ";
  os << "phi=" << stub->phi() << " ";
  os << "z=" << stub->z() << " ";
  os << "sigmaX=" << stub->sigmaX() << " ";
  os << "sigmaZ=" << stub->sigmaZ() << " ";
  os << "TPids="; 
  std::set<const TP*> tps = stub->assocTPs();
  for( auto tp : tps ) os << tp->index() << ","; 
  if( addReturn ) os << endl;
  else os << " | ";

}

void L1KalmanComb::printStubs( std::ostream &os, std::vector<const Stub *> &stubs ) const {

  for( auto &stub : stubs ){
    printStub( os, stub );
  }
}


//=== Get Kalman layer mapping (i.e. layer order in which stubs should be processed) 

unsigned int L1KalmanComb::getKalmanLayer(unsigned int iEtaReg, unsigned int layerIDreduced, bool barrel) const {

  // index across is GP encoded layer ID (where barrel layers=1,2,7,5,4,3 & endcap wheels=3,4,5,6,7 & 0 never occurs)
  // index down is eta reg
  // element is kalman layer, where 7 is invalid

  // If stub with given GP encoded layer ID can have different KF layer ID depending on whether it
  // is barrel or endcap, then in layerMap, the the barrel case is assumed.
  // The endcap case is fixed by hand later in this function.


  const unsigned int nEta = 16;
  const unsigned int nGPlayID = 7;

  if (nEta != numEtaRegions_) throw cms::Exception("ERROR L1KalmanComb::getKalmanLayer hardwired value of nEta differs from NumEtaRegions cfg param");

  static const unsigned layerMap[nEta/2][nGPlayID+1] = 
    { 
      { 7,  0,  1,  5,  4,  3,  7,  2 },
      { 7,  0,  1,  5,  4,  3,  7,  2 },
      { 7,  0,  1,  5,  4,  3,  7,  2 },
      { 7,  0,  1,  5,  4,  3,  7,  2 },
      { 7,  0,  1,  5,  4,  3,  7,  2 },
      { 7,  0,  1,  3,  4,  2,  6,  2 },
      { 7,  0,  1,  1,  2,  3,  4,  5 },
      { 7,  0,  7,  1,  2,  3,  4,  5 },
    };

  unsigned int kfEtaReg;  // KF VHDL eta sector def: small in barrel & large in endcap.
  if (iEtaReg < numEtaRegions_/2) {
    kfEtaReg = numEtaRegions_/2 - 1 - iEtaReg;
  } else {
    kfEtaReg = iEtaReg - numEtaRegions_/2;
  }

  unsigned int kalmanLayer = layerMap[kfEtaReg][layerIDreduced];

  // Fixes to endcap stubs.

  if ( not barrel ) {
			
    switch ( kfEtaReg ) {
    case 4:
      if (layerIDreduced==3) kalmanLayer = 4;
      if (layerIDreduced==4) kalmanLayer = 5;
      if (layerIDreduced==5) kalmanLayer = 6;
      break;
    case 5:
      if (layerIDreduced==5) kalmanLayer = 5;
      if (layerIDreduced==7) kalmanLayer = 6;
      break;
    default:
      break;
    }
			
  }

  return kalmanLayer;

}



L1KalmanComb::L1KalmanComb(const Settings* settings, const uint nPar, const string &fitterName, const uint nMeas ) : TrackFitGeneric(settings, fitterName ){
  nPar_ = nPar;
  nMeas_ = nMeas;
  numEtaRegions_ = settings->numEtaRegions();

  hymin = vector<double>( nPar_, -1 );
  hymax = vector<double>( nPar_,  1 );
  hymin[0] = -0.05;
  hymax[0] = +0.05;
  hymin[1] = -3.2;
  hymax[1] = +3.2;
  hymin[2] = -20;
  hymax[2] = +20;
  hymin[3] = -6;
  hymax[3] = +6;
  if (nPar_ == 5) {
    hymin[4] = -5;
    hymax[4] = +5;
  }

  hxmin = vector<double>( nPar_, -1 );
  hxmax = vector<double>( nPar_,  1 );

  hddMeasmin = vector<double>( 2, -1e-3 );
  hddMeasmax = vector<double>( 2,  1e-3 );

  hresmin = vector<double>( 2, -1e-2 );
  hresmax = vector<double>( 2,  1e-2 );

  hxaxtmin = vector<double>( nPar_, -1 );
  hxaxtmax = vector<double>( nPar_,  1 );

  hdxmin = vector<double>( nPar_, -1 );
  hdxmax = vector<double>( nPar_,  1 );

  hchi2min = 0; 
  hchi2max = 50; 

  maxNfitForDump_ = 10; 
  dump_ = false; 

  iLastPhiSec_ = 999;
  iLastEtaReg_ = 999;
}


L1fittedTrack L1KalmanComb::fit(const L1track3D& l1track3D){

  iLastPhiSec_ = iCurrentPhiSec_;
  iLastEtaReg_ = iCurrentEtaReg_;
  iCurrentPhiSec_ = l1track3D.iPhiSec();
  iCurrentEtaReg_ = l1track3D.iEtaReg();
  resetStates();
  deleteStubClusters();
  numUpdateCalls_ = 0;

  // Get cut on number of layers including variation due to dead sectors, pt dependence etc.
  minStubLayersRed_ = Utility::numLayerCut("FIT", getSettings(), l1track3D.iPhiSec(), l1track3D.iEtaReg(), fabs(l1track3D.qOverPt()), l1track3D.eta());

  //TP
  const TP* tpa(0);
  if( l1track3D.getMatchedTP() ){
    tpa = l1track3D.getMatchedTP();
  }
  tpa_ = tpa;

  //dump flag
  static unsigned nthFit(0);
  nthFit++;
  if( getSettings()->kalmanDebugLevel() >= 3 && nthFit <= maxNfitForDump_ ){
    if( tpa ) dump_ = true; 
    else dump_ = false;
  }
  else dump_ = false;

  //stub list from L1track3D, sorted in layer order - necessary for clustering only
  std::vector<const Stub*> stubs = l1track3D.getStubs();
		
  sort(stubs.begin(), stubs.end(), orderStubsByLayer); // Unnecessary?

#ifdef MERGE_STUBS
  // Eliminate identical duplicate stubs.
  for(unsigned i=0; i < stubs.size(); i++ ){
    const Stub *stub_a = stubs.at(i);
    for(unsigned j=i+1; j < stubs.size(); j++ ){
      const Stub *stub_b = stubs.at(j);
      if( stub_a->r() == stub_b->r() && stub_a->phi() == stub_b->phi() && stub_a->z() == stub_b->z() ){
	stubs.erase( stubs.begin() + j ); 
	if( getSettings()->kalmanFillInternalHists() ) 
	  hndupStub_->Fill(1);
	j--;
      }
    }
  }
#endif

  std::vector<const StubCluster *> stubcls;

  for( unsigned j_layer=0; j_layer < 16; j_layer++ ){

    std::vector<const Stub *> layer_stubs;
    for(unsigned i=0; i < stubs.size(); i++ ){
      const Stub *stub = stubs.at(i);
      if( stub->layerId() == LayerId[j_layer] ){
	layer_stubs.push_back( stub );
      }
    }

#ifdef MERGE_STUBS
    if( LayerId[j_layer] < 10 ) 
      sort( layer_stubs.begin(), layer_stubs.end(), orderStubsByZ ); // barrel
    else
      sort( layer_stubs.begin(), layer_stubs.end(), orderStubsByR ); // endcap
#endif

    for(unsigned i=0; i < layer_stubs.size(); i++ ){ // Stubs in single layer, ordered by z or r.

      std::vector<const Stub *> stubs_for_cls;
      stubs_for_cls.push_back(layer_stubs.at(i));

#ifdef MERGE_STUBS
	while( layer_stubs.at(i) != layer_stubs.back() ){
	if( isOverlap( layer_stubs.at(i), layer_stubs.at(i+1), TYPE_NORMAL ) ){
	stubs_for_cls.push_back( layer_stubs.at(i+1) );
	if( getSettings()->kalmanFillInternalHists() ) 
	hnmergeStub_->Fill(0);
	i++;
	}
	else break;
	}
#endif

      if( getSettings()->kalmanFillInternalHists() ) {

	if( tpa && tpa->useForAlgEff() ){

	  if( stubs_for_cls.size() > 1 ){

	    std::set<const TP*> s_tps = stubs_for_cls.at(0)->assocTPs();
	    if( s_tps.find( tpa ) != s_tps.end() ){

	      const Stub *sa = stubs_for_cls.front();
	      const Stub *sb = stubs_for_cls.back();

	      double drphi = fabs( sa->r() * reco::deltaPhi( sa->phi(),  sectorPhi() ) - sb->r() * reco::deltaPhi( sb->phi(), sectorPhi() ) ); 
	      double dz    = fabs( sa->z() - sb->z() );
	      double dr    = fabs( sa->r() - sb->r() );
	      TString hname;
	      if( LayerId[j_layer] < 10 ){

		hname = Form( "hBarrelStubMaxDistanceLayer%02d", LayerId[j_layer] );

		if( hBarrelStubMaxDistanceMap.find( hname ) == hBarrelStubMaxDistanceMap.end() ){
		  cout << hname << " does not exist." << endl;
		}
		else{
		  hBarrelStubMaxDistanceMap[hname]->Fill( drphi, dz );
		}
	      }
	      else{
		hname = Form( "hEndcapStubMaxDistanceRing%02d", sa->endcapRing()  );

		if( hEndcapStubMaxDistanceMap.find( hname ) == hEndcapStubMaxDistanceMap.end() ){
		  cout << hname << " does not exist." << endl;
		}
		else{
		  hEndcapStubMaxDistanceMap[hname]->Fill( drphi, dr );
		}
	      }
	    }
	  }
	}
      }

      // dl error now disabled
      StubCluster *stbcl = new StubCluster( stubs_for_cls, sectorPhi(), 0 );
      stbcl_list_.push_back( stbcl );
      stubcls.push_back( stbcl );

      if( getSettings()->kalmanFillInternalHists() ) {
	if( !stbcl->barrel() ){
	  TString hname = Form( "hphiErrorRatioRing%d", stbcl->endcapRing() );
	  if( hphiErrorRatioMap.find(hname) == hphiErrorRatioMap.end() ){
	    cout << hname << " does not exist." << endl;
	  }
	  else{
	    hphiErrorRatioMap[hname]->Fill( fabs( stbcl->deltai() + 0.5 ), fabs( stbcl->dphi_dr() ) / stbcl->dphi_dl() );
	  }
	}
      }
    }
  }
  if( getSettings()->kalmanFillInternalHists() ){ 
    if( tpa && tpa->useForAlgEff() ){
      hTrackEta_->Fill( tpa->eta() ); 
      static set<const TP *> set_tp;
      if( iCurrentPhiSec_ < iLastPhiSec_ && iCurrentEtaReg_ < iLastEtaReg_ ) set_tp.clear();
      if( set_tp.find( tpa ) == set_tp.end() ){
	hUniqueTrackEta_->Fill( tpa->eta() );
      }
      set_tp.insert( tpa );
    }
  }


  //track information dump
  if( getSettings()->kalmanDebugLevel() >= 1 ){

    std::cout << "===============================================================================" << endl;
    std::cout << "Input track cand: [phiSec,etaReg]=[" << l1track3D.iPhiSec() << "," << l1track3D.iEtaReg() << "]";
    std::cout <<" HT(m,c)=("<<l1track3D.getCellLocationHT().first << "," 
	                        <<l1track3D.getCellLocationHT().second << ") q/pt="
	      <<l1track3D.qOverPt()<<" tanL="<<l1track3D.tanLambda()<< " z0="<<l1track3D.z0()<< " phi0="<<l1track3D.phi0()
                                <<" nStubs="<<l1track3D.getNumStubs()<<" d0="<<l1track3D.d0()<<std::endl;
    if (not getSettings()->hybrid()) printTP( cout, tpa );
    if( getSettings()->kalmanDebugLevel() >= 2 ){
      printStubLayers( cout, stubs );
      printStubClusters( cout, stubcls );
    }
  }

  //Kalman Filter
  std::vector<const KalmanState *> cands = doKF( l1track3D, stubcls, tpa );

 
  //return L1fittedTrk for the selected state (if KF produced one it was happy with).
  if( cands.size() ) {

    const KalmanState *cand = cands[0];

    //cout<<"Final KF candidate eta="<<cand->candidate().iEtaReg()<<" ns="<<cand->nSkippedLayers()<<" klid="<<cand->nextLayer()-1<<" n="<<cand->nStubLayers()<<endl;

    // Get track helix params.
    std::map<std::string, double> trackParams = getTrackParams(cand);

    L1fittedTrack returnTrk(getSettings(), l1track3D, cand->stubs(), cand->hitPattern(), trackParams["qOverPt"], trackParams["d0"], trackParams["phi0"], trackParams["z0"], trackParams["t"], cand->chi2rphi(), cand->chi2rz(), nPar_, true);

    bool consistentHLS = false;  // No longer used
    //    if (this->isHLS()) {
    //      unsigned int mBinHelixHLS, cBinHelixHLS;
    //      cand->getHLSselect(mBinHelixHLS, cBinHelixHLS, consistentHLS);
    //      if( getSettings()->kalmanDebugLevel() >= 3 ){
    //        // Check if (m,c) corresponding to helix params are correctly calculated by HLS code.
    //        bool HLS_OK = ((mBinHelixHLS == returnTrk.getCellLocationFit().first) && (cBinHelixHLS == returnTrk.getCellLocationFit().second));
    //        if (not HLS_OK) std::cout<<"WARNING HLS mBinHelix disagrees with C++:"
    //                                 <<" (HLS,C++) m=("<<mBinHelixHLS<<","<<returnTrk.getCellLocationFit().first <<")"
    //                                 <<" c=("<<cBinHelixHLS<<","<<returnTrk.getCellLocationFit().second<<")"<<endl;
    //      }
    //    }

    // Store supplementary info, specific to KF fitter.
    if(this->isHLS() && nPar_ == 4) {
      returnTrk.setInfoKF( cand->nSkippedLayers(), numUpdateCalls_, consistentHLS );
    } else {
      returnTrk.setInfoKF( cand->nSkippedLayers(), numUpdateCalls_ );
    }

    // If doing 5 parameter fit, optionally also calculate helix params & chi2 with beam-spot constraint applied,
    // and store inside L1fittedTrack object.
    if (getSettings()->kalmanAddBeamConstr()) {
      if (nPar_ == 5) {
	double chi2rphi_bcon = 0.;
	std::map<std::string, double> trackParams_bcon = getTrackParams_BeamConstr(cand, chi2rphi_bcon);
	returnTrk.setBeamConstr(trackParams_bcon["qOverPt"], trackParams_bcon["phi0"], chi2rphi_bcon);
      }
    }

    // Fitted track params must lie in same sector as HT originally found track in.
    if (! getSettings()->hybrid() ) { // consistentSector() function not yet working for Hybrid.

      // Bodge to take into account digitisation in sector consistency check.
      if (getSettings()->enableDigitize()) returnTrk.digitizeTrack("KF4ParamsComb");

      if (! returnTrk.consistentSector()) {
        L1fittedTrack failedTrk(getSettings(), l1track3D, cand->stubs(), cand->hitPattern(), trackParams["qOverPt"], trackParams["d0"], trackParams["phi0"], trackParams["z0"], trackParams["t"], cand->chi2rphi(), cand->chi2rz(), nPar_, false);
        if(this->isHLS() && nPar_ == 4) {
          failedTrk.setInfoKF( cand->nSkippedLayers(), numUpdateCalls_, consistentHLS );
        } else {
          failedTrk.setInfoKF( cand->nSkippedLayers(), numUpdateCalls_ );
        }
        if ( getSettings()->kalmanDebugLevel() >= 1 ) cout<<"Track rejected by sector consistency test"<<endl;
        return failedTrk;
      }
    }

    //candidate dump
    if( getSettings()->kalmanDebugLevel() >= 3 ){
      cout << "------------------------------------" << endl;
      if( tpa && tpa->useForAlgEff() ){
	cout << "TP for eff. : index " << tpa->index() << endl;
      }
      cout << "Candidate : " << endl; 
      if( tpa && tpa->useForAlgEff() && returnTrk.getPurity() != 1 ){
	cout << "The candidate is not pure" << endl;
      }
      cand->dump( cout, tpa, true );
      cout << "------------------------------------" << endl;
    }
			
    //fill histograms for the selected state with TP for algEff
    if( getSettings()->kalmanFillInternalHists() ) fillCandHists( *cand, tpa );

    return returnTrk;

  } else {

    if (getSettings()->kalmanDebugLevel() >= 1) {
      bool goodTrack =  ( tpa && tpa->useForAlgEff() ); // Matches truth particle.
      if(goodTrack) {
	// Debug printout for Mark to understand why tracks are lost.

	int tpin=tpa->index();				
	cout<<"TRACK LOST: eta="<<l1track3D.iEtaReg()<<" pt="<<l1track3D.pt()<<" tp="<<tpin<<endl;
				
	for( auto stubCluster : stubcls ){
	  cout<<"    Stub: lay_red="<<stubCluster->layerIdReduced()<<" r="<<stubCluster->r()<<" z="<<stubCluster->z()<<"   assoc TPs =";
	  std::vector<const Stub *> stubs = stubCluster->stubs();
	  for( auto stub : stubs ){
	    for (const TP* tp_i : stub->assocTPs())  cout<<" "<<tp_i->index();
	    cout<<endl;
	    if(stub->assocTPs().size()==0) cout<<" none"<<endl;
	  }
	}
	cout<<"---------------------"<<endl;
	/*				
					for( it_last = last_states.begin(); it_last != last_states.end(); it_last++ ){
					const KalmanState *state = *it_last;
				
					//std::map<std::string, double> trackParams = getTrackParams(state);
					//L1fittedTrack returnTrk(getSettings(), l1track3D, state->stubs(), state->hitPattern(), trackParams["qOverPt"], trackParams["d0"], trackParams["phi0"], trackParams["z0"], trackParams["t"], state->chi2rphi(), state->chi2rz(), nPar_, true);
				
				
					std::vector<const Stub *> sstubs = state->stubs();
					for( auto stub : sstubs ){
				
					for (const TP* tp_i : stub->assocTPs()) {
					cout<<tp_i->index()<<endl;
					}
				
					cout<<stub->r()<<" "<<stub->z()<<" "<<state->nStubLayers()<<endl;
					}
				
					cout<<"---------------------"<<endl;
				
					}
	*/
	cout<<"====================="<<endl;
      }
    }
			
    //dump on the missed TP for efficiency calculation.
    if( getSettings()->kalmanDebugLevel() >= 3 ){
      if( tpa && tpa->useForAlgEff() ){
	cout << "TP for eff. missed addr. index : " << tpa << " " << tpa->index() << endl;
	printStubClusters( cout, stubcls );
	printStubs( cout, stubs );
      }
    }

    L1fittedTrack returnTrk(getSettings(), l1track3D, l1track3D.getStubs(), 0, l1track3D.qOverPt(), 0, l1track3D.phi0(), l1track3D.z0(), l1track3D.tanLambda(), 9999, 9999, nPar_, false);
    returnTrk.setInfoKF( 0, numUpdateCalls_ );
    return returnTrk;
  }

}


std::vector<const KalmanState *> L1KalmanComb::doKF( const L1track3D& l1track3D, const std::vector<const StubCluster *> &stubClusters, const TP *tpa ){

#ifdef RECALC_DEBUG
  cout<<"FITTER new track: HT cell=("<<l1track3D.getCellLocationHT().first<<","<<l1track3D.getCellLocationHT().second<<")"<<endl;
#endif

  // output container (contains 0 or 1 states).
  std::vector<const KalmanState *> finished_states;

  std::map<unsigned int, const KalmanState *, std::greater<unsigned int> > best_state_by_nstubs; // Best state (if any) for each viable no. of stubs on track value. 
	
  // seed helix params & their covariance.
  std::vector<double> x0 = seedx(l1track3D);
  TMatrixD pxx0 = seedP(l1track3D);
  TMatrixD K( nPar_, 2 );
  TMatrixD dcov( 2, 2 );
	
  const KalmanState *state0 = mkState( l1track3D, 0, 0, 0, nullptr, x0, pxx0, K, dcov, nullptr, 0, 0 );
	
  if( getSettings()->kalmanFillInternalHists() ) fillSeedHists( state0, tpa );
	
	
  // internal containers - i.e. the state FIFO. Contains estimate of helix params in last/next layer, with multiple entries if there were multiple stubs, yielding multiple states.
  std::vector<const KalmanState *> new_states;
  std::vector<const KalmanState *> prev_states;
  prev_states.push_back( state0 );
	  
  // arrange stubs into Kalman layers according to eta region
  int etaReg = l1track3D.iEtaReg();
  std::map<int, std::vector<const StubCluster *> > layerStubs;

  // Get dead layers, if any.
  // They are assumed to be idetnical to those defined in StubKiller.cc
  bool remove2PSCut = getSettings()->kalmanRemove2PScut();
  set<unsigned> kalmanDeadLayers = getKalmanDeadLayers( remove2PSCut );

  for( auto stubCluster : stubClusters ){
	
    // Get Kalman encoded layer ID for this stub.
    int kalmanLayer = this->getKalmanLayer(etaReg, stubCluster->layerIdReduced(), stubCluster->barrel());
		
    if (kalmanLayer != 7) {
      const_cast<StubCluster*>(stubCluster)->setLayerKF(kalmanLayer); // Ugly trick to store KF layer inside stub cluster.
      if (layerStubs[kalmanLayer].size() < getSettings()->kalmanMaxStubsPerLayer()) {
	layerStubs[kalmanLayer].push_back( stubCluster );
      } else {
	// If too many stubs, FW keeps the last stub.
	layerStubs[kalmanLayer].back() = stubCluster;
      }
    }
  }

  // iterate using state->nextLayer() to determine next Kalman layer(s) to add stubs from
  const unsigned int maxIterations = 6;       // Increase if you want to allow 7 stubs per fitted track.
  for( unsigned iteration = 0; iteration < maxIterations; iteration++ ){   

    int combinations_per_iteration = 0;
		
    bool easy = (l1track3D.getNumStubs() < getSettings()->kalmanMaxStubsEasy());
    unsigned int kalmanMaxSkipLayers = easy ? getSettings()->kalmanMaxSkipLayersEasy() : getSettings()->kalmanMaxSkipLayersHard();
		
    // update each state from previous iteration (or seed) using stubs in next Kalman layer
    std::vector<const KalmanState *>::const_iterator i_state = prev_states.begin();
    for(; i_state != prev_states.end(); i_state++ ){ 
		
      const KalmanState *the_state = *i_state;
			

      unsigned int layer = the_state->nextLayer();
      unsigned skipped = the_state->nSkippedLayers();

      // If this layer is known to be dead, skip to the next layer (layer+1)
      // The next_states_skipped will then look at layer+2
      // However, if there are stubs in this layer, then don't skip (e.g. our phi/eta boundaries might not line up exactly with a dead region)
      // Continue to skip until you reach a functioning layer (or a layer with stubs)
      unsigned nSkippedDeadLayers = 0;
      while ( kalmanDeadLayers.find(layer) != kalmanDeadLayers.end() && layerStubs[layer].size() == 0 ) {
	layer += 1;
	++nSkippedDeadLayers;
      }

      // containers for updated state+stub combinations
      std::vector<const KalmanState *> next_states;
      std::vector<const KalmanState *> next_states_skipped;
			
      // find stubs for this layer
      std::vector<const StubCluster *> stubs = layerStubs[layer]; // If layer > 6, this will return empty vector, so safe.

      // find stubs for next layer if we skip a layer, except when we are on the penultimate layer,
      // or we have exceeded the max skipped layers
      std::vector<const StubCluster *> next_stubs ;

      // If the next layer (layer+1) is a dead layer, then proceed to the layer after next (layer+2), if possible
      // Also note if we need to increase "skipped" by one more for these states
      unsigned nSkippedDeadLayers_nextStubs = 0;
      if ( skipped < kalmanMaxSkipLayers ) {
        if ( kalmanDeadLayers.find(layer+1) != kalmanDeadLayers.end()  && layerStubs[layer+1].size() == 0 ) {
	  next_stubs = layerStubs[layer+2];
	  nSkippedDeadLayers_nextStubs += 1;
        } else {
	  next_stubs = layerStubs[layer+1];
	}
      }

      // If track was not rejected by isGoodState() is previous iteration, failure here usually means the tracker ran out of layers to explore.
      // (Due to "kalmanLayer" not having unique ID for each layer within a given eta sector).
      if ( getSettings()->kalmanDebugLevel() >= 2 && best_state_by_nstubs.size() == 0 && stubs.size() == 0 && next_stubs.size() == 0) cout<<"State is lost by start of iteration "<<iteration<<" : #stubs="<<stubs.size()<<" #next_stubs="<<next_stubs.size()<<" layer="<<layer<<" eta="<<l1track3D.iEtaReg()<<endl;

      // If we skipped over a dead layer, only increment "skipped" after the stubs in next+1 layer have been obtained
      skipped += nSkippedDeadLayers;
		
      // check to guarantee no fewer than 2PS hits per state at iteration 1 (r<60cm)
      // iteration 0 will always include a PS hit, but iteration 1 could use 2S hits unless we include this
      if (iteration==1 && !remove2PSCut) {
	std::vector<const StubCluster *> temp_stubs;
	std::vector<const StubCluster *> temp_nextstubs;
	for (auto stub : stubs) {
	  if (stub->r()<60.0) temp_stubs.push_back(stub);
	}
	for (auto stub : next_stubs) {
	  if (stub->r()<60.0) temp_nextstubs.push_back(stub);
	}
	stubs = temp_stubs;
	next_stubs = temp_nextstubs;
      }

			
      combinations_per_iteration += stubs.size() + next_stubs.size();
			
			
      // loop over each stub in this layer and check for compatibility with this state
      for( unsigned i=0; i < stubs.size()  ; i++ ){
	
	const StubCluster * next_stubCluster = stubs[i];
				
	// Update helix params by adding this stub.
	const KalmanState * new_state = kalmanUpdate( skipped, layer+1, next_stubCluster, *the_state, tpa );
				
	if( getSettings()->kalmanFillInternalHists() ) fillStepHists( tpa, iteration, new_state );
				
	// Cut on track chi2, pt etc.
	if(isGoodState( *new_state ) ) next_states.push_back( new_state );
      }

      // loop over each stub in next layer if we skip, and check for compatibility with this state
      for( unsigned i=0; i < next_stubs.size()  ; i++ ){
	
	const StubCluster * next_stubCluster = next_stubs[i];
				
	const KalmanState * new_state = kalmanUpdate( skipped+1+nSkippedDeadLayers_nextStubs, layer+2+nSkippedDeadLayers_nextStubs, next_stubCluster, *the_state, tpa );
				
	if( getSettings()->kalmanFillInternalHists() ) fillStepHists( tpa, iteration, new_state );
				
	if(isGoodState( *new_state ) ) next_states_skipped.push_back( new_state );
      }		
			
      // post Kalman filter local sorting per state
      sort( next_states.begin(), next_states.end(), KalmanState::orderChi2);
      sort( next_states_skipped.begin(), next_states_skipped.end(), KalmanState::orderChi2);
			
			
      int i, max_states, max_states_skip;
			
      // If layer contained several stubs, so several states now exist, select only the best ones.
      // -- Disable this by setting to large values, as not used in latest KF firmware.
      // (But not too big as this wastes CPU).

      switch ( iteration ) {
      case 0:
	max_states = 15;
	max_states_skip = 15;
	break;
      case 1:
	max_states = 15;
	max_states_skip = 15;
	break;
      case 2:
	max_states = 15;
	max_states_skip = 15;
	break;
      case 3:
	max_states = 15;
	max_states_skip = 15;
	break;
      case 4:
	max_states = 15;
	max_states_skip = 15;
	break;
      case 5:
	max_states = 15;
	max_states_skip = 15;
	break;
      default:
	max_states = 15;
	max_states_skip = 15;
	break;
      }
			
			
      i = 0;
      for( auto state : next_states ){
					
	if( i < max_states ){
	  new_states.push_back( state );
	} else {
	  break;
	}
	i++;
	
      }
			
      i = 0; 
      for( auto state : next_states_skipped ){
	
	if( i < max_states_skip ){
	  new_states.push_back( state );
	} else {
	  break;
	}
	i++;
	
      }
			
    } //end of state loop


    if( getSettings()->kalmanFillInternalHists() ) {
      TString hname = Form( "hstubComb_itr%d", iteration );
      if( hstubCombMap.find(hname) == hstubCombMap.end() ){
	cout << hname << " does not exist." << endl;
      }
      else{
	hstubCombMap[hname]->Fill( combinations_per_iteration );
      }
    }
		
			 
    // copy new_states into prev_states for next iteration or end if we are on 
    // last iteration by clearing all states and making final state selection
		
    sort( new_states.begin(), new_states.end(), KalmanState::orderMinSkipChi2); // Sort by chi2*(skippedLayers+1)

    unsigned int nStubs = iteration + 1;
    // Success. We have at least one state that passes all cuts. Save best state found with this number of stubs.
    if (nStubs >= getSettings()->kalmanMinNumStubs() && new_states.size() > 0) best_state_by_nstubs[nStubs] = new_states[0]; 

    //if ( getSettings()->kalmanDebugLevel() >= 1 && best_state_by_nstubs.size() == 0 && new_states.size() == 0) cout<<"Track is lost by end iteration "<<iteration<<" : eta="<<l1track3D.iEtaReg()<<endl;

    if( nStubs == getSettings()->kalmanMaxNumStubs() ){ 
      // We're done.
      prev_states.clear();
      new_states.clear();
			
    } else {
			
      // Continue iterating.
      prev_states = new_states;
      new_states.clear(); 
			
    }
				
    /*
      int i = 0;
      bool found = false;
      for( auto best_state : best_states4 ){
			
      if( tpa && tpa->useForAlgEff() ) {
      std::map<std::string, double> trackParams = getTrackParams(best_state);
      L1fittedTrack returnTrk(getSettings(), l1track3D, best_state->stubs(), best_state->hitPattern(), trackParams["qOverPt"], trackParams["d0"], trackParams["phi0"], trackParams["z0"], trackParams["t"], best_state->chi2rphi(), best_state->chi2rz(), nPar_, true);
      if (returnTrk.getNumMatchedLayers()>=4) {
      //temp_states.push_back(best_state);
      if(i==0) found = true;
      if (!found) cout<<"Lost this cand "<<i<<" "<<best_state->chi2()<<" "<<best_state->reducedChi2()<<" "<<best_state->path()<<" chose instead "<<best_states4[0]->chi2()<<" "<<best_states4[0]->reducedChi2()<<" "<<best_statesn4[0]->path()<<endl;
      }
      }*/
		
  }

  if (best_state_by_nstubs.size()) {
    // Select state with largest number of stubs.
    const KalmanState* stateFinal = best_state_by_nstubs.begin()->second; // First element has largest number of stubs.
    finished_states.push_back(stateFinal);
    if ( getSettings()->kalmanDebugLevel() >= 1 ) {
      cout<<"Track found! final state selection: nLay="<<stateFinal->nStubLayers()<<" hitPattern="<<std::hex<<stateFinal->hitPattern()<<std::dec<<" phiSec="<<l1track3D.iPhiSec()<<" etaReg="<<l1track3D.iEtaReg()<<" HT(m,c)=("<<l1track3D.getCellLocationHT().first<<","<<l1track3D.getCellLocationHT().second<<")";
      std::map<std::string, double> y = getTrackParams( stateFinal );
      cout<<" q/pt="<<y["qOverPt"]<<" tanL="<<y["t"]<<" z0="<<y["z0"]<<" phi0="<<y["phi0"];
      if (nPar_==5) cout<<" d0="<<y["d0"];
      cout<<" chosen from states:";
      for (const auto& p : best_state_by_nstubs) cout<<" "<<p.second->chi2()<<"/"<<p.second->nStubLayers();
      cout<<endl;
    }
  } else {
    if ( getSettings()->kalmanDebugLevel() >= 1 ) {
      cout<<"Track lost"<<endl;
    }
  }

  return finished_states;
}


//--- Update a helix state by adding a stub. 
//--- ("layer" is not the layer of the stub being added now, but rather the next layer that will be searched after this stub has been added).

const KalmanState *L1KalmanComb::kalmanUpdate( unsigned skipped, unsigned layer, const StubCluster *stubCluster, const KalmanState &state, const TP *tpa ){

  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "---------------" << endl;
    cout << "kalmanUpdate" << endl;
    cout << "---------------" << endl;
    printStubCluster( cout, stubCluster );
  }

  numUpdateCalls_++; // For monitoring, count calls to updator per track.

  // Helix params & their covariance.
  std::vector<double> xa     = state.xa();
  TMatrixD            cov_xa = state.pxxa(); 
  if( state.barrel() && !stubCluster->barrel() ){ 
    if( getSettings()->kalmanDebugLevel() >= 4 ) {
      cout << "STATE BARREL TO ENDCAP BEFORE " << endl;
      cout << "state : " << xa.at(0) << " " << xa.at(1) << " " << xa.at(2) << " " << xa.at(3) << endl;
      cout << "cov(x): " << endl; 
      cov_xa.Print();
    }
    barrelToEndcap( state.r(), stubCluster, xa, cov_xa );
    if( getSettings()->kalmanDebugLevel() >= 4 ){
      cout << "STATE BARREL TO ENDCAP AFTER " << endl;
      cout << "state : " << xa.at(0) << " " << xa.at(1) << " " << xa.at(2) << " " << xa.at(3) << endl;
      cout << "cov(x): " << endl; 
      cov_xa.Print();
    }
  }
  // Matrix to propagate helix params from one layer to next (=identity matrix).
  TMatrixD f = F(stubCluster, &state );
  TMatrixD ft(TMatrixD::kTransposed, f );
  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "f" << endl;
    f.Print();
    cout << "ft" << endl;
    ft.Print();
  }

  std::vector<double> fx = Fx( f, xa ); // Multiply matrices to get helix params at next layer.
  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "fx = ["; 
    for( unsigned i = 0; i < nPar_; i++ ) cout << fx.at(i) << ", ";
    cout << "]" << endl;
  }

  std::vector<double> delta = residual(stubCluster, fx, state.candidate().qOverPt() );
  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "delta = " << delta[0] << ", " << delta[1] << endl;
  }

  // Derivative of predicted (phi,z) intercept with layer w.r.t. helix params.
  TMatrixD h = H(stubCluster);
  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "h" << endl;
    h.Print();
  }


  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "previous state covariance" << endl;
    cov_xa.Print();
  }
  // Get contribution to helix parameter covariance from scattering (NOT USED).
  TMatrixD pxxm = PxxModel( &state, stubCluster );
  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "model xcov" << endl;
    pxxm.Print();
  }
  // Get covariance on helix parameters.
  TMatrixD pxcov = f * cov_xa * ft + pxxm;
  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "forcast xcov + model xcov" << endl;
    pxcov.Print();
  }
  // Get hit position covariance matrix.
  TMatrixD dcov = PddMeas( stubCluster, &state );
  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "dcov" << endl;
    dcov.Print();
  }
  // Calculate Kalman Gain matrix.
  TMatrixD k = GetKalmanMatrix( h, pxcov, dcov );  
  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "k" << endl;
    k.Print();
  }
	 
  std::vector<double> new_xa(nPar_);
  TMatrixD new_pxxa;
  GetAdjustedState( k, pxcov, fx, stubCluster, delta, new_xa, new_pxxa );
  if( getSettings()->kalmanDebugLevel() >= 4 ){
    if( nPar_ == 4 )
      cout << "adjusted x = " << new_xa[0] << ", " << new_xa[1] << ", " << new_xa[2] << ", " << new_xa[3] << endl;
    else if( nPar_ == 5 )
      cout << "adjusted x = " << new_xa[0] << ", " << new_xa[1] << ", " << new_xa[2] << ", " << new_xa[3] << ", " << new_xa[4] << endl;
    cout << "adjusted covx " << endl;
    new_pxxa.Print();
  }

  const KalmanState *new_state = mkState( state.candidate(), skipped, layer, stubCluster->layerId(), &state, new_xa, new_pxxa, k, dcov, stubCluster, 0, 0 );
  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "new state" << endl;
    new_state->dump( cout, tpa  );
  }


  return new_state;
}


void L1KalmanComb::calcChi2( const KalmanState &state, double& chi2rphi, double& chi2rz )const{

  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "calcChi2 " << endl;
  }
  double deltaChi2rphi(0), deltaChi2rz;

  if( state.last_state() ) {
			
    const StubCluster *stubCluster = state.stubCluster();
			
#ifdef RECALC_DEBUG
    unsigned int ID = (stubCluster != nullptr)  ?  stubCluster->stubs()[0]->index()  :  99999;
#endif

    if( stubCluster ){
	
      std::vector<double> delta = residual( stubCluster, state.last_state()->xa(), state.last_state()->candidate().qOverPt() );
      TMatrixD dcov = PddMeas( stubCluster, &state );
#ifdef RECALC_DEBUG
      cout<<"    FITTER SIGMA:      rphi="<<1000*sqrt(dcov(0,0))<<" rz="<<sqrt(dcov(1,1))<<" ID="<<ID<<endl;
#endif

      if( getSettings()->kalmanDebugLevel() >= 4 ){
	cout << "dcov" << endl;
	dcov.Print();
	cout << "xcov" << endl;
	state.last_state()->pxxa().Print();
      }
      TMatrixD h = H(stubCluster);
      TMatrixD hxxh = HxxH( h, state.last_state()->pxxa() );
      if( getSettings()->kalmanDebugLevel() >= 4 ){
	cout << "h" << endl;
	h.Print();
	cout << "hxcovh" << endl;
	hxxh.Print();
      }
      TMatrixD covR = dcov + hxxh;
      if( getSettings()->kalmanDebugLevel() >= 4 ){
	cout << "covR" << endl;
	covR.Print();
	cout << "---" << endl;
	cout << scientific << "delta = " << delta[0] << ", " << delta[1] << endl;
      }
      this->getDeltaChi2( covR, delta, false, deltaChi2rphi, deltaChi2rz );  
	
    }
    chi2rphi = state.last_state()->chi2rphi() + deltaChi2rphi;
    chi2rz   = state.last_state()->chi2rz()   + deltaChi2rz;
#ifdef RECALC_DEBUG
    cout<<"  FITTER CHI2 UPDATE = "<<(chi2rphi+chi2rz)<<" delta chi2="<<(deltaChi2rphi+deltaChi2rz)<<" ID="<<ID<<endl;
#endif
  }
  return;
}


void L1KalmanComb::getDeltaChi2( const TMatrixD &dcov, const std::vector<double> &delta, bool debug,  
                                 double& deltaChi2rphi, double& deltaChi2rz) const
{
  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "dcov" << endl;
    dcov.Print();
  }

  if( dcov.Determinant() == 0 ) {
    deltaChi2rphi = 999;
    deltaChi2rz   = 999;
    return;
  };


  TMatrixD dcovi( dcov );
  dcovi.Invert();

  // Change in chi2 (with r-phi/r-z correlation term included in r-phi component)
  deltaChi2rphi = delta.at(0) * delta.at(0) * dcovi(0,0) + 2 * delta.at(0) * delta.at(1) * dcovi(0,1); 
  deltaChi2rz   = delta.at(1) * delta.at(1) * dcovi(1,1); 

#ifdef RECALC_DEBUG
  cout<<"    FITTER DELTA CHI2: rphi="<<deltaCii2rphi<<" rz="<<deltaChi2rz<<endl;
#endif

  if( debug ){
    cout << "CHI SQUARE OUTPUT" << endl;
    cout << "cov" << endl;
    dcov.Print();
    cout << "cov inv" << endl;
    dcovi.Print();
    for( unsigned i=0; i < delta.size(); i++ ) cout << delta.at(i) << " ";
    cout << endl;
  }
  return;
}


std::map<std::string, double> L1KalmanComb::getTrackParams( const L1KalmanComb *p, const KalmanState *state )
{
  return p->getTrackParams( state );
}


std::vector<double> L1KalmanComb::Hx( const TMatrixD &pH, const std::vector<double> &x )const
{
  std::vector<double> m( (unsigned) pH.GetNrows(), 0 );
  if( pH.GetNcols() != (int) x.size() ) { cerr << "Hx() : H and x have different dimensions" << endl; }
  else{

    for( int i=0; i < pH.GetNcols(); i++ ){ 
      for( int j=0; j < pH.GetNrows(); j++ ){ 
	m.at(j) += pH(j,i) * x.at(i);
      }
    }
  }
  return m;
}


std::vector<double> L1KalmanComb::Fx( const TMatrixD &pF, const std::vector<double> &x )const
{
  return Hx( pF, x );
}


TMatrixD L1KalmanComb::HxxH( const TMatrixD &pH, const TMatrixD &xx )const
{
  int nd = (unsigned) pH.GetNrows(); 
  TMatrixD tmp(nd,nPar_);
  TMatrixD mHxxH(nd,nd);
  if( pH.GetNcols() != xx.GetNcols() || pH.GetNcols() != xx.GetNrows() ) { cerr << "HxxH() : H and xx have different dimensions" << endl; }
  else{

    for( int i=0; i < pH.GetNrows(); i++ ){ 
      for( int j=0; j < xx.GetNrows(); j++ ){ 
	for( int k=0; k < xx.GetNcols(); k++ ){ 
	  tmp(i,k) += pH(i,j) * xx(j,k);
	}
      }
    }
    for( int i=0; i < tmp.GetNrows(); i++ ){ 
      for( int j=0; j < pH.GetNcols(); j++ ){ 
	for( int k=0; k < pH.GetNrows(); k++ ){ 
	  mHxxH(i,k) += tmp(i,j) * pH(k,j); 
	}
      }
    }
  }
  return mHxxH;

}


TMatrixD L1KalmanComb::GetKalmanMatrix( const TMatrixD &h, const TMatrixD &pxcov, const TMatrixD &dcov )const
{

  TMatrixD pxcovht(pxcov.GetNrows(),2);
  for( int i=0; i<pxcov.GetNrows(); i++ ){
    for( int j=0; j<pxcov.GetNcols(); j++ ){
      for( int k=0; k<h.GetNrows(); k++ ){
	pxcovht(i,k) += pxcov(i,j) * h(k,j);
      }
    }
  }
  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "pxcovht" << endl;
    pxcovht.Print();
  }

  TMatrixD tmp(dcov.GetNrows(), dcov.GetNcols() );
  TMatrixD hxxh = HxxH( h, pxcov );
  tmp = dcov + hxxh; 

  if( getSettings()->kalmanDebugLevel() >= 4 ){
    cout << "hxxh" << endl;
    hxxh.Print();
    cout << "dcov + hxxh " << endl;
    tmp.Print();
  }

  TMatrixD K( pxcovht.GetNrows(), tmp.GetNcols() );

  if(tmp.Determinant() == 0 ) return K; 
  tmp.Invert();

  for( int i=0; i<pxcovht.GetNrows(); i++ ){
    for( int j=0; j<pxcovht.GetNcols(); j++ ){
      for( int k=0; k<tmp.GetNcols(); k++ ){
	K(i,k)+=pxcovht(i,j)*tmp(j,k);
      }
    }
  }
  return K;
}


void L1KalmanComb::GetAdjustedState( const TMatrixD &K, const TMatrixD &pxcov, 
				     const std::vector<double> &x, const StubCluster *stubCluster, 
				     const std::vector<double>& delta,
				     std::vector<double> &new_x, TMatrixD &new_xcov )const
{
  TMatrixD h = H(stubCluster);

  for( int i=0; i < K.GetNrows(); i++ ){
    new_x.at(i) = x.at(i);  
    for( int j=0; j < K.GetNcols(); j++ ){
      new_x.at(i) += K(i,j) * delta.at(j);
    }
  }

  TMatrixD tmp(K.GetNrows(), h.GetNcols() );
  for( int i=0; i< K.GetNrows(); i++ ){
    tmp(i,i) = 1;
  }
  for( int i=0; i< K.GetNrows(); i++ ){
    for( int j=0; j< K.GetNcols(); j++ ){
      for( int k=0; k< h.GetNcols(); k++ ){
	tmp(i,k) += -1 * K(i,j) * h(j,k);
      }
    }
  }
  new_xcov.Clear();
  new_xcov.ResizeTo(pxcov.GetNrows(), pxcov.GetNcols());
  for( int i=0; i< tmp.GetNrows(); i++ ){
    for( int j=0; j< tmp.GetNcols(); j++ ){
      for( int k=0; k< pxcov.GetNcols(); k++ ){
	new_xcov(i,k) += tmp(i,j) * pxcov(j,k);
      }
    }
  }
}


void L1KalmanComb::resetStates()
{
  for( unsigned int i=0; i < state_list_.size(); i++ ){

    delete state_list_.at(i);
  }
  state_list_.clear();
}


const KalmanState *L1KalmanComb::mkState( const L1track3D &candidate, unsigned skipped, unsigned layer, unsigned layerId, const KalmanState *last_state, 
					  const std::vector<double> &x, const TMatrixD &pxx, const TMatrixD &K, const TMatrixD &dcov, const StubCluster* stubCluster, double chi2rphi, double chi2rz )
{

  KalmanState *new_state = new KalmanState( candidate, skipped, layer, layerId, last_state, x, pxx, K, dcov, stubCluster, chi2rphi, chi2rz, this, &getTrackParams );

  if( chi2rphi + chi2rz == 0 ){
    double new_state_chi2rphi = 0., new_state_chi2rz = 0.;
    this->calcChi2( *new_state, new_state_chi2rphi, new_state_chi2rz ); 
    new_state->setChi2( new_state_chi2rphi, new_state_chi2rz );
  }

  state_list_.push_back( new_state );
  return new_state;
}


std::vector<double> L1KalmanComb::residual(const StubCluster* stubCluster, const std::vector<double> &x, double candQoverPt )const{

  std::vector<double> vd = d(stubCluster); // Get (phi relative to sector, z) of hit.
  std::vector<double> hx = Hx( H(stubCluster), x ); // Ditto for intercept of helix with layer, in linear approximation.
  std::vector<double> delta(2);
  for( unsigned i=0; i<2; i++ ) delta.at(i) = vd.at(i) - hx.at(i);

  // Calculate higher order corrections to residuals.

  if (not getSettings()->kalmanHOdodgy()) {

    std::vector<double> correction = {0.,0.};

    float inv2R = (getSettings()->invPtToInvR()) * 0.5 * candQoverPt; // alternatively use x().at(0)
    float tanL = x.at(2);
    float z0 = x.at(3);

    float deltaS = 0.;
    if (getSettings()->kalmanHOhelixExp()) {
      // Higher order correction correction to circle expansion for improved accuracy at low Pt.
      double corr = stubCluster->r() * inv2R; 

      // N.B. In endcap 2S, this correction to correction[0] is exactly cancelled by the deltaS-dependent correction to it below.
      correction[0] += (1./6.)*pow(corr, 3); 

      deltaS = (1./6.)*(stubCluster->r())*pow(corr, 2);
      correction[1] -= deltaS * tanL;
    }

    if ( (not stubCluster->barrel()) && not (stubCluster->psModule())) {
      // These corrections rely on inside --> outside tracking, so r-z track params in 2S modules known.
      float rShift = (stubCluster->z() - z0)/tanL - stubCluster->r();

      // The above calc of rShift is approximate, so optionally check it with MC truth.
      // if (tpa_ != nullptr) rShift = (stubCluster->z() - tpa_->z0())/tpa_->tanLambda() - stubCluster->r();

      if (getSettings()->kalmanHOhelixExp()) rShift -= deltaS;

      if (getSettings()->kalmanHOprojZcorr() == 1) {
	// Add correlation term related to conversion of stub residuals from (r,phi) to (z,phi).
	correction[0] += inv2R * rShift; 
      }

      if (getSettings()->kalmanHOalpha()     == 1) {
	// Add alpha correction for non-radial 2S endcap strips..
	correction[0] += stubCluster->alpha() * rShift;
      }

      //cout<<"ENDCAP 2S STUB: (r,z)=("<<stubCluster->r()<<","<<stubCluster->z()<<") r*delta="<<stubCluster->r() * correction[0]<<" r*alphaCorr="<<stubCluster->r() * stubCluster->alpha() * rShift<<" rShift="<<rShift<<endl;
    }

    // Apply correction to residuals.
    delta[0] += correction[0];
    delta[1] += correction[1];
  }

  delta.at(0) = reco::deltaPhi(delta.at(0), 0.);

  return delta;
}


void L1KalmanComb::bookHists(){

  if ( getSettings()->kalmanFillInternalHists() ) {

    edm::Service<TFileService> fs_;
    string dirName;
    if( fitterName_.compare("") == 0 ) dirName = "L1KalmanCombInternal";
    else dirName = fitterName_ + "Internal";

    TFileDirectory inputDir = fs_->mkdir(dirName.c_str());


    TString hname;
    hTrackEta_ = inputDir.make<TH1F>( "hTrackEta", "Track #eta; #eta", 50, -2.5, 2.5 );
    hUniqueTrackEta_ = inputDir.make<TH1F>( "hUniqueTrackEta", "Unique Track #eta; #eta", 50, -2.5, 2.5 );
    hndupStub_ = inputDir.make<TH1F>( "hndupStub", "# of duplicated stubs", 1, 0, 1 );
    hnmergeStub_ = inputDir.make<TH1F>( "hnmergeStub", "# of merged stubs", 1, 0, 1 );


    for( unsigned j_layer=0; j_layer < 6; j_layer++ ){
      hname = Form( "hBarrelStubMaxDistanceLayer%02d", LayerId[j_layer] );
      hBarrelStubMaxDistanceMap[hname] = inputDir.make<TH2F>( hname, Form( "max distance of stubs in barrel Layer %02d; dr#phi; dz", LayerId[j_layer] ), 
							      100, 0, 1., 100, 0, 10 );
    }

    for( unsigned j_ecring=1; j_ecring < 16; j_ecring++ ){
      hname = Form( "hEndcapStubMaxDistanceRing%02d", j_ecring );
      hEndcapStubMaxDistanceMap[hname] = inputDir.make<TH2F>( hname, Form( "max distance of stubs in endcap Ring %02d; dr#phi; dr", j_ecring ), 
							      100, 0, 1., 100, 0, 10 );
      hname = Form( "hphiErrorRatioRing%d", j_ecring );
      hphiErrorRatioMap[hname] = inputDir.make<TH2F>( hname, Form( "; fabs( strip id - 0.5 x nStrips + 0.5 ); #delta #phi_{r} / #delta #phi_{l}" ), 508, 0.0, 508.0 , 50, -0.5, 49.5 );
    }



    float nbins(2002);
    for( unsigned i=0; i < nPar_; i++ ){
      hname = Form( "hyt_%d", i );
      hytMap[hname] = inputDir.make<TH1F>( hname, Form( "; true track parameter values %d", i ), nbins, hymin[i], hymax[i] );
      hname = Form( "hy0_%d", i );
      hy0Map[hname] = inputDir.make<TH1F>( hname, Form( "; after HT track parameter values %d", i ), nbins, hymin[i], hymax[i] );
      hname = Form( "hyf_%d", i );
      hyfMap[hname] = inputDir.make<TH1F>( hname, Form( "; after KF track parameter values %d", i ), nbins, hymin[i], hymax[i] );
      hname = Form( "hx_%d", i );
      hxMap[hname] = inputDir.make<TH1F>( hname, Form( "; x values %d", i ), nbins, hxmin[i], hxmax[i] );
    }

 
    for( unsigned itr=0; itr<=5; itr++ ){
		
      hname = Form( "hstubComb_itr%d", itr );
      hstubCombMap[hname] = inputDir.make<TH1F>( hname, Form( "; #state+stub combinations, iteration %d ", itr ), 100, 0., 100.);

      for( unsigned i=0; i < nPar_; i++ ){
	for( unsigned j=0; j <= i; j++ ){
	
	  hname = Form( "hxcov_itr%d_%d_%d", itr, i, j );
	  hxcovMap[hname] = inputDir.make<TH1F>( hname, Form( "; state covariance adjusted values, iteration %d (%d,%d)", itr, i, j ), 
						 nbins, -1 * hdxmin[i]*hdxmin[j], hdxmax[i]*hdxmax[j] );
	}
      }
      for( unsigned i=0; i < nPar_; i++ ){
	for( unsigned j=0; j < nMeas_; j++ ){
	  hname = Form( "hk_itr%d_%d_%d", itr, i, j );
	  hkMap[hname] = inputDir.make<TH1F>( hname, Form( "; K(%d,%d), Iteration %d", i, j, itr ), 200, -1., 1. );
	}
      }
      for( unsigned i=0; i < nMeas_; i++ ){
	hname = Form( "hres_itr%d_%d", itr, i );
	hresMap[hname] = inputDir.make<TH1F>( hname, Form( "; residual values, iteration %d (%d)", itr, i ), 
					      nbins, hresmin[i], hresmax[i] );
	for( unsigned j=0; j <= i; j++ ){
	  hname = Form( "hmcov_itr%d_%d_%d", itr, i, j );
	  hmcovMap[hname] = inputDir.make<TH1F>( hname, Form( "; measurement covariance values, iteration %d (%d,%d)", itr, i, j ), 
						 nbins, -1 * hddMeasmin[i]*hddMeasmin[i], hddMeasmax[i]*hddMeasmax[j] );
	}
      }
    }
  }  
}


void L1KalmanComb::fillCandHists( const KalmanState &state, const TP *tpa )
{
  if( tpa && tpa->useForAlgEff() ){

    const KalmanState *the_state = &state;
    while( the_state ){
      if( the_state->stubCluster() ){
	std::vector<double> x = the_state->xa();
	for( unsigned i=0; i < nPar_; i++ ){
	  TString hname = Form( "hx_%d", i );
	  if( hxMap.find(hname) == hxMap.end() ){
	    cout << hname << " does not exist." << endl;
	  }
	  else hxMap[hname]->Fill(x.at(i));
	}
      }
      the_state = the_state->last_state();
    }


    std::map<std::string, double> mx = getTrackParams( &state );
    std::vector<double> vx(nPar_);
    vx[0] = mx["qOverPt"];
    vx[1] = mx["phi0"];
    vx[2] = mx["z0"];
    vx[3] = mx["t"];
    if( nPar_ == 5 ) vx[4] = mx["d0"];
    for( unsigned i=0; i < nPar_; i++ ){
      TString hname = Form( "hyf_%d", i );
      if( hyfMap.find(hname) == hyfMap.end() ){
	cout << hname << " does not exist." << endl;
      }
      else hyfMap[hname]->Fill(vx[i]);
    }
  }
}


void L1KalmanComb::fillSeedHists( const KalmanState *state, const TP *tpa ){

  std::vector<double> x0   = state->xa();
  TMatrixD            pxx0 = state->pxxa();
  //Histogram Fill : seed pxxa 
  for( unsigned i=0; i < nPar_; i++ ){
    for( unsigned j=0; j <= i; j++ ){
      TString hname = Form( "hxcov_itr%d_%d_%d", 0, i, j );
      if( hxcovMap.find( hname ) == hxcovMap.end() ){
	cout << hname << " does not exist." << endl;
      }
      else hxcovMap[hname]->Fill( pxx0(i,j) );
    }
  }

  if( tpa && tpa->useForAlgEff() ){
    std::vector<double> tpParams(nPar_);
    tpParams[0] = tpa->qOverPt();
    tpParams[1] = tpa->phi0();
    tpParams[2] = tpa->z0();
    tpParams[3] = tpa->tanLambda();
    if( nPar_ == 5 ) tpParams[4] = tpa->d0();
    for( unsigned i=0; i < nPar_; i++ ){
      TString hname = Form( "hyt_%d", i );
      if( hytMap.find(hname) == hytMap.end() ){
	cout << hname << " does not exist." << endl;
      }
      else hytMap[hname]->Fill(tpParams[i]);
    }
    //Histogram Fill : Seed state 
    std::map<std::string, double> trackParams = getTrackParams( state );
    std::vector<double> trackParVec(nPar_);
    trackParVec[0] = trackParams["qOverPt"];
    trackParVec[1] = trackParams["phi0"];
    trackParVec[2] = trackParams["z0"];
    trackParVec[3] = trackParams["t"];
    if( nPar_ == 5 ) trackParVec[4] = trackParams["d0"];
    for( unsigned i=0; i < nPar_; i++ ){
      TString hname = Form( "hy0_%d", i );
      if( hy0Map.find(hname) == hy0Map.end() ){
	cout << hname << " does not exist." << endl;
      }
      else hy0Map[hname]->Fill(trackParVec[i]);
    }
  }
}


void L1KalmanComb::fillStepHists( const TP *tpa, unsigned nItr, const KalmanState *new_state )
{
  unsigned path = 0;

  const std::vector<double> &xa = new_state->xa();
  const StubCluster *stubCluster = new_state->stubCluster();
  const TMatrixD &pxxa = new_state->pxxa();

  TString hname;

  for( unsigned i=0; i < nPar_; i++ ){

    for( unsigned j=0; j <= i; j++ ){
      hname = Form( "hxcov_itr%d_%d_%d", nItr, i, j );
      if( hxcovMap.find( hname ) == hxcovMap.end() ){
	cout << hname << " does not exist." << endl;
      }
      else hxcovMap[hname]->Fill( pxxa(i,j) );
    }
  }
  for( unsigned i=0; i < nPar_; i++ ){
    for( int j=0; j < 2; j++ ){
      TString hname = Form( "hk_itr%d_%d_%d", nItr, i, j );
      if( hkMap.find( hname ) == hkMap.end() ){
	cout << hname << " does not exist." << endl;
      }
      else hkMap[hname]->Fill( new_state->K()(i,j) );
    }
  }
  std::vector<double> delta_new = residual(stubCluster, xa, new_state->candidate().qOverPt() );
  for( unsigned int i=0; i < delta_new.size(); i++ ){
    TString hname = Form( "hres_itr%d_%d", nItr, i );
    if( hresMap.find(hname) == hresMap.end() ){
      cout << hname << " does not exist." << endl;
    }
    else hresMap[hname]->Fill( delta_new[i] );  
  }
  for( int i=0; i < 2; i++ ){
    for( int j=0; j < i; j++ ){
      TString hname = Form( "hmcov_itr%d_%d_%d", nItr, i, j );
      if( hmcovMap.find( hname ) == hmcovMap.end() ){
	cout << hname << " does not exist." << endl;
      }
      else hmcovMap[hname]->Fill( new_state->dcov()(i,j) );
    }
  }
}


void L1KalmanComb::deleteStubClusters()
{
  for( unsigned int i=0; i < stbcl_list_.size(); i++ ){
    delete stbcl_list_.at(i);
  }
  stbcl_list_.clear();
}


double L1KalmanComb::DeltaRphiForClustering( unsigned layerId, unsigned endcapRing )
{
  static double barrel_drphi[6] = { 0.05, 0.04, 0.05, 0.12, 0.13, 0.19 }; 
  if( layerId < 10 ) return barrel_drphi[layerId - 1];

  static double ec_drphi[16] =  
    { 0.04, 0.05, 0.04, 0.06, 0.06, 0.04, 0.06, 0.07, 0.15, 0.08, 0.27, 0.08, 0.27, 0.12, 0.09 };
  return ec_drphi[endcapRing - 1];
};


double L1KalmanComb::DeltaRForClustering( unsigned endcapRing )
{
  static double ec_dr[16] =  
    { 0.52, 0.56, 0.59, 0.86, 0.66, 0.47, 0.55, 0.72, 1.53, 1.10, 2.72, 0.91, 2.69, 0.67, 0.09 };
  return ec_dr[endcapRing - 1];

}


bool L1KalmanComb::isOverlap( const Stub* a, const Stub*b, OVERLAP_TYPE type ){

  std::set<const TP*> a_tps = a->assocTPs();
  std::set<const TP*> b_tps = b->assocTPs();
  double drphi = DeltaRphiForClustering( a->layerId(), a->endcapRing() );
  double dr(0);
  switch ( type ){

  case TYPE_NORMAL:
    if( a->layerId() != b->layerId() ) return false;

    if( a->layerId() < 7 ){
      if( fabs( b->z() - a->z() ) > 0.5 * b->stripLength() || fabs( reco::deltaPhi( b->phi(), sectorPhi() ) * b->r() - reco::deltaPhi( a->phi(), sectorPhi() ) * a->r() ) > 0.5 * b->stripPitch() ) return false;
    }
    else{
      if( fabs( b->r() - a->r() ) > 0.5 * b->stripLength() || fabs( reco::deltaPhi( b->phi(), sectorPhi() ) * b->r() - reco::deltaPhi( a->phi(), sectorPhi() ) * a->r() ) > 0.5 * b->stripPitch() ) return false;
    }
    return true;
  case TYPE_V2:
    if( a->layerId() != b->layerId() ) return false;

    if( a->layerId() < 7 ){
      if( fabs( b->z() - a->z() ) > 0.5 * b->stripLength() || fabs( reco::deltaPhi( b->phi(), sectorPhi() ) * b->r() - reco::deltaPhi( a->phi(), sectorPhi() ) * a->r() ) > drphi ) return false;
    }
    else{
      dr = DeltaRForClustering( a->endcapRing() ); 
      if( fabs( b->r() - a->r() ) > dr || fabs( reco::deltaPhi( b->phi(), sectorPhi() ) * b->r() - reco::deltaPhi( a->phi(), sectorPhi() ) * a->r() ) > drphi ) return false;
    }
    return true;

  case TYPE_NOCLUSTERING:
    return false;

  case TYPE_TP:
    for( auto a_tp : a_tps ) 
      if( b_tps.find( a_tp ) != b_tps.end() ) return true;
    return false;
  default:
    return false;
  }
}

set<unsigned> L1KalmanComb::getKalmanDeadLayers( bool& remove2PSCut ) const {

  // Kill scenarios described in https://github.com/EmyrClement/StubKiller/blob/master/README.md

  // By which Stress Test scenario (if any) are dead modules being emulated?
  const unsigned int killScenario = getSettings()->killScenario(); 
  // Should TMTT tracking be modified to reduce efficiency loss due to dead modules?
  const bool killRecover = getSettings()->killRecover();

  set<pair<unsigned,bool>> deadLayers; // GP layer ID & boolean indicating if in barrel.

  if (killRecover) {
    if ( killScenario == 1 ) { // barrel layer 5
      deadLayers.insert(pair<unsigned,bool>(4,true));
      if ( iCurrentEtaReg_ < 5 || iCurrentEtaReg_ > 8 || iCurrentPhiSec_ < 8 || iCurrentPhiSec_ > 11 ) {
	deadLayers.clear();
      }

    }
    else if ( killScenario == 2 ) { // barrel layer 1
      deadLayers.insert(pair<unsigned,bool>(1,true));
      if ( iCurrentEtaReg_ > 8 || iCurrentPhiSec_ < 8 || iCurrentPhiSec_ > 11 ) {
	deadLayers.clear();
      }
      remove2PSCut = true;
    }
    else if ( killScenario == 3 ) { // barrel layers 1 & 2
      deadLayers.insert(pair<unsigned,bool>(1,true));
      deadLayers.insert(pair<unsigned,bool>(2,true));
      if ( iCurrentEtaReg_ > 8 || iCurrentPhiSec_ < 8 || iCurrentPhiSec_ > 11 ) {
	deadLayers.clear();
      }
      else if ( iCurrentEtaReg_ < 1 ) {
        deadLayers.insert(pair<unsigned,bool>(0,true));  // What is this doing?
      }
      remove2PSCut = true;
    }
    else if ( killScenario == 4 ) { // barrel layer 1 & disk 1
      deadLayers.insert(pair<unsigned,bool>(1,true));
      deadLayers.insert(pair<unsigned,bool>(3,false));
      if ( iCurrentEtaReg_ > 8 || iCurrentPhiSec_ < 8 || iCurrentPhiSec_ > 11 ) {
	deadLayers.clear();
      }
      else if ( iCurrentEtaReg_ > 3 ) {
        deadLayers.insert(pair<unsigned,bool>(0,true));
      }
      remove2PSCut = true;
    }
  }

  set<unsigned> kalmanDeadLayers;
  for ( const auto& p : deadLayers ) {
    unsigned int layer = p.first;
    bool barrel = p.second;
    unsigned int kalmanLayer = this->getKalmanLayer(iCurrentEtaReg_, layer, barrel);
    kalmanDeadLayers.insert( kalmanLayer );
  }

  return kalmanDeadLayers;
}

//=== Function to calculate approximation for tilted barrel modules (aka B) copied from Stub class.

float L1KalmanComb::getApproxB(float z, float r) const {
  return getSettings()->bApprox_gradient() * fabs(z)/r + getSettings()->bApprox_intercept();
}

}
