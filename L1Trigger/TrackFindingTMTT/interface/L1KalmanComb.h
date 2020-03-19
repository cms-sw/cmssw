///=== This is the base class for the Kalman Combinatorial Filter track fit algorithm.
 
#ifndef __L1_KALMAN_COMB__
#define __L1_KALMAN_COMB__
 
#include <TMatrixD.h>
#include "L1Trigger/TrackFindingTMTT/interface/TrackFitGeneric.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/KalmanState.h"
#include <map>
#include <vector>
#include <fstream>
#include <TString.h>

class TH1F;
class TH2F;

namespace TMTT {

class TP; 
class KalmanState;
class StubCluster;

class L1KalmanComb : public TrackFitGeneric{
 
    public:
	enum OVERLAP_TYPE { TYPE_NORMAL, TYPE_V2, TYPE_NOCLUSTERING, TYPE_TP };
    public:
	L1KalmanComb(const Settings* settings, const uint nPar, const string &fitterName="", const uint nMeas=2 );

        virtual ~L1KalmanComb() { this->resetStates(); this->deleteStubClusters();}

	L1fittedTrack fit(const L1track3D& l1track3D);
	void bookHists();

    protected:
	static  std::map<std::string, double> getTrackParams( const L1KalmanComb *p, const KalmanState *state );
	virtual std::map<std::string, double> getTrackParams( const KalmanState *state ) const=0;

        // Get track params with beam-spot constraint & chi2 (r-phi) after applying it..
  virtual std::map<std::string, double> getTrackParams_BeamConstr( const KalmanState *state, double& chi2rphi_bcon) const {
          chi2rphi_bcon = 0.0;
          return (this->getTrackParams(state)); // Returns unconstrained result, unless derived class overrides it.
        }

	double sectorPhi()const
	{
          float phiCentreSec0 = -M_PI/float(getSettings()->numPhiNonants()) + M_PI/float(getSettings()->numPhiSectors());
          return 2.*M_PI * float(iCurrentPhiSec_) / float(getSettings()->numPhiSectors()) + phiCentreSec0; 
	}
        //bool kalmanUpdate( const StubCluster *stubCluster, KalmanState &state, KalmanState &new_state, const TP *tpa );
	virtual const KalmanState *kalmanUpdate( unsigned skipped, unsigned layer, const StubCluster* stubCluster, const KalmanState &state, const TP *);
	void resetStates();
	void deleteStubClusters();
	const KalmanState *mkState( const L1track3D &candidate, unsigned skipped, unsigned layer, unsigned layerId, const KalmanState *last_state, 
				    const std::vector<double> &x, const TMatrixD &pxx, const TMatrixD &K, const TMatrixD &dcov, const StubCluster* stubCluster, double chi2rphi, double chi2rz );

    protected:
	/* Methods */
	std::vector<double> Hx( const TMatrixD &pH, const std::vector<double> &x )const;
	std::vector<double> Fx( const TMatrixD &pF, const std::vector<double> &x )const;
	TMatrixD HxxH( const TMatrixD &pH, const TMatrixD &xx )const;
        void getDeltaChi2( const TMatrixD &dcov, const std::vector<double> &delta, bool debug, 
   	                   double& deltaChi2rphi, double& deltaChi2rz )const;
	TMatrixD GetKalmanMatrix( const TMatrixD &h, const TMatrixD &pxcov, const TMatrixD &dcov )const;
	void GetAdjustedState( const TMatrixD &K, const TMatrixD &pxcov, 
			       const std::vector<double> &x, const StubCluster *stubCluster, const std::vector<double>& delta,  
			       std::vector<double> &new_x, TMatrixD &new_xcov )const;


	virtual std::vector<double> seedx(const L1track3D& l1track3D)const=0;
	virtual TMatrixD seedP(const L1track3D& l1track3D)const=0;
	virtual void barrelToEndcap( double r, const StubCluster *stubCluster, std::vector<double> &x, TMatrixD &cov_x )const{}
	virtual std::vector<double> d(const StubCluster* stubCluster )const=0;
	virtual TMatrixD H(const StubCluster* stubCluster)const=0;
	virtual TMatrixD F(const StubCluster* stubCluster=0, const KalmanState *state=0 )const=0;
	virtual TMatrixD PxxModel( const KalmanState *state, const StubCluster *stubCluster )const=0; 
  	virtual TMatrixD PddMeas(const StubCluster* stubCluster, const KalmanState *state )const=0;

        virtual std::vector<double> residual(const StubCluster* stubCluster, const std::vector<double> &x, double candQoverPt )const;
	virtual const KalmanState *updateSeedWithStub( const KalmanState &state, const StubCluster *stubCluster ){ return 0; }
	virtual bool isGoodState( const KalmanState &state )const{ return true; }

        virtual void calcChi2( const KalmanState &state, double& chi2rphi, double& chi2rz ) const;

	virtual double getRofState( unsigned layerId, const vector<double> &xa )const{ return 0;}
        virtual unsigned int getKalmanLayer(unsigned int iEtaReg, unsigned int layerIDreduced, bool barrel)const;

	std::vector<const KalmanState *> doKF( const L1track3D &l1track3D, const std::vector<const StubCluster *> &stubClusters, const TP *tpa );

        void printTPSummary( std::ostream &os, const TP *tp, bool addReturn=true ) const;
	void printTP( std::ostream &os, const TP *tp ) const;
        void printStubLayers( std::ostream &os, std::vector<const Stub *> &stubs ) const;
        void printStubCluster( std::ostream &os, const StubCluster * stubCluster, bool addReturn=true ) const;
        void printStubClusters( std::ostream &os, std::vector<const StubCluster *> &stubClusters ) const;
        void printStub( std::ostream &os, const Stub * stub, bool addReturn=true ) const;
        void printStubs( std::ostream &os, std::vector<const Stub *> &stubs ) const;

	void fillSeedHists( const KalmanState *state, const TP *tpa );
	void fillCandHists( const KalmanState &state, const TP *tpa=0 );
	void fillStepHists( const TP *tpa, unsigned nItr, const KalmanState *new_state );

	double DeltaRphiForClustering( unsigned layerId, unsigned endcapRing );
	double DeltaRForClustering( unsigned endcapRing );
	bool isOverlap( const Stub* a, const Stub*b, OVERLAP_TYPE type );

	set<unsigned> getKalmanDeadLayers( bool& remove2PSCut ) const;

        // Function to calculate approximation for tilted barrel modules (aka B) copied from Stub class.
        float getApproxB(float z, float r) const;

        // Is this HLS code?
        virtual bool isHLS() {return false;};

    protected:
	unsigned nPar_;
	unsigned nMeas_;
        unsigned numEtaRegions_;

	std::vector<KalmanState *> state_list_;
	std::vector<StubCluster *> stbcl_list_;

	std::vector<double> hxaxtmin;
	std::vector<double> hxaxtmax;
	std::vector<double> hxmin;
	std::vector<double> hxmax;
	std::vector<double> hymin;
	std::vector<double> hymax;
	std::vector<double> hdxmin;
	std::vector<double> hdxmax;
	std::vector<double> hresmin;
	std::vector<double> hresmax;
	std::vector<double> hddMeasmin;
	std::vector<double> hddMeasmax;

	TH1F * hTrackEta_;
	TH1F * hUniqueTrackEta_;
	std::map<TString, TH2F*> hBarrelStubMaxDistanceMap;
	std::map<TString, TH2F*> hEndcapStubMaxDistanceMap;
	std::map<TString, TH2F*> hphiErrorRatioMap;
	std::map<TString, TH1F*> hstubCombMap;

	TH1F*           hndupStub_;
	TH1F*           hnmergeStub_;
	std::map<TString, TH1F*> hytMap;
	std::map<TString, TH1F*> hy0Map;
	std::map<TString, TH1F*> hyfMap;
	std::map<TString, TH1F*> hxMap;
	std::map<TString, TH1F*> hxcovMap;
	std::map<TString, TH1F*> hkMap;
	std::map<TString, TH1F*> hresMap;
	std::map<TString, TH1F*> hmcovMap;

      	double hchi2min;
	double hchi2max;

	unsigned maxNfitForDump_;
	bool     dump_;
	unsigned int      iCurrentPhiSec_;
	unsigned int      iCurrentEtaReg_;
	unsigned int      iLastPhiSec_;
	unsigned int      iLastEtaReg_;

        unsigned int      minStubLayersRed_;

        unsigned int      numUpdateCalls_;

       const TP* tpa_;
};

}

#endif




