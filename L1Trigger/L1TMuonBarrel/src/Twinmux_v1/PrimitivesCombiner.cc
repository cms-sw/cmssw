//Author: Giuseppe Codispoti

#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/PrimitiveCombiner.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace L1TwinMux;

PrimitiveCombiner::PrimitiveCombiner( const PrimitiveCombiner::resolutions & res,
					      edm::ESHandle<DTGeometry> & muonGeom )
  : _resol(res), _muonGeom(muonGeom),
    _bx(-3), _radialAngle(4096), _bendingAngle(-520), _bendingResol(520),
    _dtHI(0), _dtHO(0), _rpcIn(0), _rpcOut(0)
{}



void PrimitiveCombiner::addDt( const TriggerPrimitive & dt )
{
  int qualityCode = dt.getDTData().qualityCode;
  switch ( qualityCode ) {
  case 2 : addDtHO( dt ); break;
  case 3 : addDtHI( dt ); break;
  default :
    throw cms::Exception("Invalid DT Quality")
      << "This module can combine only HI to HO (quality2/3), provided : "
      << qualityCode << std::endl;
  }

}


void PrimitiveCombiner::addDtHI( const TriggerPrimitive & prim )
{
  if ( _dtHI )
    throw cms::Exception("Invalid DT Quality")
      << "DT primitive with quality HI already provided"
      << std::endl;

  _dtHI = &prim;
}


void PrimitiveCombiner::addDtHO( const TriggerPrimitive & prim )
{
  if ( _dtHO )
    throw cms::Exception("Invalid DT Quality")
      << "DT primitive with quality HO already provided"
      << std::endl;

  _dtHO = &prim;
}


void PrimitiveCombiner::addRpcIn( const TriggerPrimitive & prim )
{
  _rpcIn = &prim;
}


void PrimitiveCombiner::addRpcOut( const TriggerPrimitive & prim )
{
  _rpcOut = &prim;
}



void PrimitiveCombiner::combine()
{
  if ( !isValid() ) return;

  typedef PrimitiveCombiner::results localResult;
  std::vector<localResult> localResults;

  _radialAngle = 0;
  // inner and outer DT
  if ( _dtHI && _dtHO ) {
    localResults.push_back( combineDt( _dtHI, _dtHO ) );
    _radialAngle = localResults.back().radialAngle;
  }
  // inner DT
  else if ( _dtHI ) {
    // if (!_dtHO) localResults.push_back(dummyCombineDt(_dtHI));
    if ( _rpcIn ) {
      localResults.push_back( combineDtRpc( _dtHI, _rpcIn ) );
      _radialAngle = _radialAngle ? _radialAngle :_dtHI->getDTData().radialAngle;
    }
    if ( _rpcOut ) {
      localResults.push_back( combineDtRpc( _dtHI, _rpcOut ) );
      _radialAngle = _radialAngle ? _radialAngle :_dtHI->getDTData().radialAngle;
    }
  }
  //outer DT
  else if ( _dtHO ) {
     // if (!_dtHI) localResults.push_back(dummyCombineDt(_dtHO));
    if ( _rpcIn ) {
      localResults.push_back( combineDtRpc( _dtHO, _rpcIn ) );
      _radialAngle = _radialAngle ? _radialAngle :_dtHO->getDTData().radialAngle;
    }
    if ( _rpcOut ) {
      localResults.push_back( combineDtRpc( _dtHO, _rpcOut ) );
      _radialAngle = _radialAngle ? _radialAngle :_dtHO->getDTData().radialAngle;
    }
  }
  // no DT
  else if ( !_dtHI && !_dtHO ) {
    if ( _rpcIn && _rpcOut ) {
      results local = combineRpcRpc( _rpcIn, _rpcOut );
      localResults.push_back( local );
      _radialAngle = local.radialAngle;
      //std::cout<<"Entered RPC-IN-OUT thing..."<<std::endl;
    } else if ( _rpcIn ) {
      //std::cout<<"Entered RPC-IN-ONLY thing..."<<std::endl;
      _radialAngle = radialAngleFromGlobalPhi( _rpcIn );
      _bendingAngle = -666;
      return;
    } else if ( _rpcOut ) {
      //std::cout<<"Entered RPC-OUT-only thing..."<<std::endl;
      _radialAngle = radialAngleFromGlobalPhi( _rpcOut );
      _bendingAngle = -666;
      return;
    }
   }
  double weightSum = 0;
  _bendingResol = 0;
  _bendingAngle = 0;

  std::vector<localResult>::const_iterator it = localResults.begin();
  std::vector<localResult>::const_iterator itend = localResults.end();
  int kcount=0;
  for ( ; it != itend; ++it ) {
    //weightSum += it->bendingResol;
    //_bendingAngle += it->bendingResol * it->bendingAngle;
    kcount++;
    //std::cout<<"combining result "<<kcount<<" with resolution "<<it->bendingResol<<std::endl;
    weightSum += 1.0/(( it->bendingResol)*(it->bendingResol));
    _bendingAngle += double(it->bendingAngle)/((it->bendingResol) * (it->bendingResol));
    _bendingResol += it->bendingResol * it->bendingResol;
  }

  _bendingAngle /= weightSum;
  _bendingResol = sqrt( _bendingResol );

}

PrimitiveCombiner::results
PrimitiveCombiner::dummyCombineDt( const TriggerPrimitive * dt)
{

  // i want to combine also the DT data alone
  results localResult;
  localResult.radialAngle = dt->getDTData().radialAngle;
  //std::cout<<" HEY!!! I am adding a DT benfing as a combination result! "<<std::endl;
  localResult.bendingAngle=dt->getDTData().bendingAngle;
  localResult.bendingResol=_resol.phibDtUnCorr;
  return localResult;
}




PrimitiveCombiner::results
PrimitiveCombiner::combineDt( const TriggerPrimitive * dt1,
				      const TriggerPrimitive * dt2 )
{

  const DTChamber* chamb1 = _muonGeom->chamber( dt1->detId<DTChamberId>() );
  LocalPoint point1 = chamb1->toLocal( dt1->getCMSGlobalPoint() );

  const DTChamber* chamb2 = _muonGeom->chamber( dt2->detId<DTChamberId>() );
  LocalPoint point2 = chamb2->toLocal( dt2->getCMSGlobalPoint() );

  results localResult;
  localResult.radialAngle = 0.5 * ( dt1->getDTData().radialAngle + dt2->getDTData().radialAngle );

  /// PhiB calculation :
  /// atan( (x2-x1)/(z2-z1) is the bending angle
  /// needs to be corrected w.r.t.the direction phi (4096 scale)
  /// and ported to 512 scale
  if ( ( dt1->getDTData().wheel > 0 ) ||
       ( ( dt1->getDTData().wheel == 0 ) &&
	 !( dt1->getDTData().sector == 0 || dt1->getDTData().sector == 3
	    || dt1->getDTData().sector == 4 || dt1->getDTData().sector == 7
	    || dt1->getDTData().sector == 8 || dt1->getDTData().sector == 11 ) )
       ) {
    /// positive chambers
    localResult.bendingAngle = ( atan ( phiBCombined( point1.x(), point1.z(),
						      point2.x(), point2.z() )
					)
				 - ( localResult.radialAngle/4096.0 )
				 ) * 512;
  } else {
    // negative chambers
    localResult.bendingAngle = ( atan ( -phiBCombined( point1.x(), point1.z(),
						       point2.x(), point2.z() )
					)
				 - ( localResult.radialAngle/4096.0)
				 ) * 512;
  }
  localResult.bendingResol = phiBCombinedResol( _resol.xDt, _resol.xDt,
						point1.z(), point2.z() );

  //std::cout<<" == === COMBINING DT-DT === == "<<std::endl;
  //std::cout << "dt-dt radial : " << dt1->getDTData().radialAngle << " * " << dt2->getDTData().radialAngle << " = " << localResult.radialAngle << '\n';
  //std::cout << " " << point1.x() << " " << point1.z() << " " << dt1->getDTData().qualityCode << '\n';
  //std::cout << " " << point2.x() << " " << point2.z() << " " << dt2->getDTData().qualityCode << '\n';
  //std::cout << "dt-dt bending : " << dt1->getDTData().bendingAngle << " * " << dt2->getDTData().bendingAngle << " = "
  //	    << localResult.bendingAngle << '\n';
  //std::cout<<" --- this was sector "<<dt1->getDTData().sector<<" and wheel "<<dt1->getDTData().wheel<<std::endl;

  return localResult;

}

PrimitiveCombiner::results
PrimitiveCombiner::combineDtRpc( const TriggerPrimitive * dt,
					 const TriggerPrimitive * rpc )
{

  results localResult;
  localResult.radialAngle = dt->getDTData().radialAngle;

  const DTChamber* chamb1 = _muonGeom->chamber( dt->detId<DTChamberId>() );
  LocalPoint point1 = chamb1->toLocal( dt->getCMSGlobalPoint() );
  int station = rpc->detId<RPCDetId>().station();
  int sector  = rpc->detId<RPCDetId>().sector();
  int wheel = rpc->detId<RPCDetId>().ring();
  const DTChamber* chamb2 = _muonGeom->chamber( DTChamberId( wheel, station, sector ) );
  LocalPoint point2 = chamb2->toLocal( rpc->getCMSGlobalPoint() );

  if ( ( dt->getDTData().wheel > 0 ) ||
      ( ( dt->getDTData().wheel == 0 ) &&
	!( dt->getDTData().sector == 0 || dt->getDTData().sector==3
	   || dt->getDTData().sector==4 || dt->getDTData().sector==7
	   || dt->getDTData().sector==8 || dt->getDTData().sector==11 ) )
       ) {
    // positive wheels
    localResult.bendingAngle = ( atan( phiBCombined( point1.x(), point1.z(),
						     point2.x(), point2.z() )
				       )
				 - ( localResult.radialAngle/4096.0 )
				 ) * 512;
  } else {
    // negative wheels
    localResult.bendingAngle = ( atan( -phiBCombined( point1.x(), point1.z(),
						      point2.x(), point2.z() )
				       )
				 - ( localResult.radialAngle/4096.0 )
				 ) * 512;
  }
  localResult.bendingResol = phiBCombinedResol( _resol.xDt, _resol.xRpc,
						point1.z(), point2.z() );

  return localResult;

}


// dt+rpc solo bending, phi dt
// dt+dt bending, media della posizione, direzione in base alla differenza dei due punti
// cancellare seconda traccia (bx diverso)

int
PrimitiveCombiner::radialAngleFromGlobalPhi( const TriggerPrimitive * rpc )
{
  int radialAngle = 0;
  int radialAngle2 = 0;
  int sector = rpc->detId<RPCDetId>().sector();
  float phiGlobal = rpc->getCMSGlobalPhi();
  // int wheel = rpc->detId<RPCDetId>().ring();
  // from phiGlobal to radialAngle of the primitive in sector sec in [1..12]
  if ( sector == 1) radialAngle = int( phiGlobal*4096 );
  else {
    if ( phiGlobal >= 0) radialAngle = int( (phiGlobal-(sector-1)*Geom::pi()/6)*4096 );
    else radialAngle = int( (phiGlobal+(13-sector)*Geom::pi()/6)*4096 );
  }
  //if ( ( wheel>0 ) ||
  //     ( ( wheel==0 ) &&
  //      ( sector==0 || sector==3 || sector==4 || sector==7 || sector==8 || sector==11 ) ) )
  radialAngle2 = radialAngle;
  //else
  // radialAngle2 = -radialAngle;
  return radialAngle2;
}

PrimitiveCombiner::results
PrimitiveCombiner::combineRpcRpc( const TriggerPrimitive * rpc1,
                                          const TriggerPrimitive * rpc2 )
{
  int station = rpc1->detId<RPCDetId>().station();
  int sector  = rpc1->detId<RPCDetId>().sector();
  int wheel = rpc1->detId<RPCDetId>().ring();
  const DTChamber* chamb1 = _muonGeom->chamber( DTChamberId( wheel, station, sector ) );
  LocalPoint point1 = chamb1->toLocal( rpc1->getCMSGlobalPoint() );


  station = rpc2->detId<RPCDetId>().station();
  sector  = rpc2->detId<RPCDetId>().sector();
  wheel = rpc2->detId<RPCDetId>().ring();
  const DTChamber* chamb2 = _muonGeom->chamber( DTChamberId( wheel, station, sector ) );
  LocalPoint point2 = chamb2->toLocal( rpc2->getCMSGlobalPoint() );


  results localResult;
  localResult.radialAngle = 0.5*(radialAngleFromGlobalPhi( rpc1 )+radialAngleFromGlobalPhi( rpc2 )) ;

  if ( ( wheel>0 ) ||
       ( ( wheel==0 ) &&
         !( sector==0 || sector==3 || sector==4 || sector==7 || sector==8 || sector==11 ) ) )
    localResult.bendingAngle = ( atan ( phiBCombined( point1.x(),
                                                      point1.z(),
                                                      point2.x(),
                                                      point2.z() )
                                        - ( localResult.radialAngle / 4096.0 )
                                        ) ) * 512;
    else
    localResult.bendingAngle = ( atan (-phiBCombined( point1.x(),
                                                      point1.z(),
                                                      point2.x(),
                                                      point2.z() )
                                       - ( localResult.radialAngle/4096.0 )
                                       ) ) * 512;
  localResult.bendingResol = phiBCombinedResol( _resol.xRpc, _resol.xRpc, point1.z(), point2.z());


  return localResult;

}
