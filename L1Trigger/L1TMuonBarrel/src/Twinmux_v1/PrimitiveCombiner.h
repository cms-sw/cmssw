#ifndef L1Trigger_L1IntegratedMuonTrigger_PrimitiveCombiner_h_
#define L1Trigger_L1IntegratedMuonTrigger_PrimitiveCombiner_h_
//
// Class: L1TwinMux:: PrimitiveCombiner
//
// Info: This class combine information from DT and/or RPC primitives
//       in order to calculate better phi/phiBending
//
// Author: Giuseppe Codispoti
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "L1Trigger/L1TMuon/interface/deprecate/MuonTriggerPrimitive.h"

#include <cmath>

class DTGeometry;

using namespace L1TMuon;

namespace L1TwinMux {

  //class L1TwinMux::TriggerPrimitive;

  class PrimitiveCombiner {

  public :
    /// a struct useful for resulution info sharing
    struct resolutions {
      double xDt;
      double xRpc;
      double phibDtCorr;
      double phibDtUnCorr;
    resolutions( const double & xDtResol, const double & xRpcResol,
		 const double & phibDtCorrResol, const double & phibDtUnCorrResol )
    :  xDt( xDtResol ), xRpc( xRpcResol ),
	phibDtCorr( phibDtCorrResol ), phibDtUnCorr( phibDtUnCorrResol ) {};
    };

  public :
    explicit PrimitiveCombiner( const resolutions & res, edm::ESHandle<DTGeometry> & muonGeom );

    /// feed the combiner with the available primitives
    void addDt( const TriggerPrimitive & prim );
    void addDtHI( const TriggerPrimitive & prim );
    void addDtHO( const TriggerPrimitive & prim );
    void addRpcIn( const TriggerPrimitive & prim );
    void addRpcOut( const TriggerPrimitive & prim );

    /// do combine the primitives
    void combine();

    /// output result variables
    inline int bx() const { return _bx;};
    inline int radialAngle() const { return _radialAngle;};
    inline int bendingAngle() const { return _bendingAngle;};
    inline int bendingResol() const { return _bendingResol;};

    /// valid if we have at least: 1 rpc; 1 dt + 1 any
    bool isValid() const {
      int ret = _dtHI ? 1 : 0;
      ret += _dtHO ? 1 : 0;
      ret += _rpcIn ? 2 : 0;
      ret += _rpcOut ? 2 : 0;
      return ret > 1 ;
    };

    /// FIXME : Calculates new phiBending, check how to use
    inline float phiBCombined( const float & xDt, const float & zDt,
			       const float & xRpc, const float & zRpc )
    {
      return (xRpc - xDt) / (zRpc - zDt);
    };
    /// FIXME END

    /// FIXME : Calculates new phiBending resolution
    inline float phiBCombinedResol( const float & resol_xDt,
				    const float & resol_xRpc,
				    const float & zDt,
				    const float & zRpc
				    )
    {
      return std::sqrt( resol_xRpc*resol_xRpc + resol_xDt*resol_xDt )/std::fabs(zDt-zRpc);
    };
    /// FIXME END

    int getUncorrelatedQuality7() const {

      int qualityCode = 0;
      if ( _dtHI && _dtHO ) {
	if ( _rpcIn && _rpcOut ) qualityCode = 5;
	else qualityCode = 5;
      } else if ( _dtHO ) {// HO quality == 3
	if ( _rpcIn && _rpcOut ) qualityCode = 4;
	else if ( _rpcOut ) qualityCode = 4;
	else if ( _rpcIn ) qualityCode = 4;
	else  qualityCode = 2;
      } else if ( _dtHI ) {// HI, quality == 2
	if ( _rpcIn && _rpcOut ) qualityCode = 3;
	else if ( _rpcOut ) qualityCode = 3;
	else if ( _rpcIn ) qualityCode = 3;
	else  qualityCode = 1;
      } else {
	if ( _rpcIn && _rpcOut ) qualityCode = 0;
	else if ( _rpcOut ) qualityCode = -1;
	else if ( _rpcIn ) qualityCode = -1;
      }
      return qualityCode;
    }

    int getUncorrelatedQuality16() const {

      int qualityCode = 0;
      if ( _dtHI && _dtHO ) {
	if ( _rpcIn && _rpcOut ) qualityCode = 12;
	else qualityCode = 11;
      } else if ( _dtHO ) {// HO quality == 3
	if ( _rpcIn && _rpcOut ) qualityCode = 10;
	else if ( _rpcOut ) qualityCode = 8;
	else if ( _rpcIn ) qualityCode = 6;
	else  qualityCode = 4;
      } else if ( _dtHI ) {// HI, quality == 2
	if ( _rpcIn && _rpcOut ) qualityCode = 9;
	else if ( _rpcOut ) qualityCode = 7;
	else if ( _rpcIn ) qualityCode = 5;
	else  qualityCode = 3;
      } else {
	if ( _rpcIn && _rpcOut ) qualityCode = 2;
	else if ( _rpcOut ) qualityCode = 1;
	else if ( _rpcIn ) qualityCode = 0;
      }
      return qualityCode;
    }

  private :

    /// a struct for internal usage: store results
    struct results {
      double radialAngle;
      double bendingAngle;
      double bendingResol;
    results() : radialAngle(0), bendingAngle(0), bendingResol(0) {};
    };


    /// Calculates new phiBending, check how to use weights
    results combineDt( const TriggerPrimitive * dt,
		       const TriggerPrimitive * rpc );

    results dummyCombineDt( const TriggerPrimitive * dt);

    /// Calculates new phiBending, check how to use weights
    results combineDtRpc( const TriggerPrimitive * dt,
			  const TriggerPrimitive * rpc );

    /// Calculates new phiBending, check how to use weights
    results combineRpcRpc( const TriggerPrimitive * rpc1,
			   const TriggerPrimitive * rpc2 );


    int radialAngleFromGlobalPhi( const TriggerPrimitive * rpc );

  private :
    resolutions _resol;
    edm::ESHandle<DTGeometry> _muonGeom;

    int _bx;
    int _radialAngle;
    int _bendingAngle;
    int _bendingResol;

    const TriggerPrimitive * _dtHI;
    const TriggerPrimitive * _dtHO;
    const TriggerPrimitive * _rpcIn;
    const TriggerPrimitive * _rpcOut;

  };
}

#endif
