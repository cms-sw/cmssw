//Authors:
// Carlo Battilana - Giuseppe Codispoti

#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/MBLTCollection.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/Math/interface/deltaPhi.h"

L1TwinMux::MBLTCollection::MBLTCollection( const DTChamberId & dtId )
{
  _wheel = dtId.wheel();
  _sector = dtId.sector();
  _station = dtId.station();
}

void L1TwinMux::MBLTCollection::addStub(const TriggerPrimitiveRef & stub)
{

  TriggerPrimitive::subsystem_type type = stub->subsystem();

  switch ( type ) {
  case TriggerPrimitive::kDT :
    _dtAssociatedStubs.push_back( stub );
    break;
  case TriggerPrimitive::kRPC : {
    const RPCDetId & rpcId = stub->detId<RPCDetId>();

    if ( rpcId.region() ) { // endcap
      throw cms::Exception("Invalid Subsytem")
	<< "The RPC stub is not in a barrel layer" << std::endl;
    }

    if ( rpcId.layer() == 1 ) _rpcInAssociatedStubs.push_back( stub );
    else if ( rpcId.layer() == 2 ) _rpcOutAssociatedStubs.push_back( stub );
    else throw cms::Exception("Invalid Subsytem")
      << "The RPC layer is not a barrel layer" << std::endl;
    break;
  }
  default :
    throw cms::Exception("Invalid Subsytem")
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }
}



void L1TwinMux::MBLTCollection::associate( double minRpcPhi )
{

  size_t dtSize = _dtAssociatedStubs.size();
  size_t rpcInSize = _rpcInAssociatedStubs.size();
  size_t rpcOutSize = _rpcOutAssociatedStubs.size();
  _dtMapAss.resize( dtSize );


//   std::vector< std::map<double, size_t> > dtIdxIn;
//   dtIdxIn.resize(rpcInSize);
//   std::vector< std::map<double, size_t> > dtIdxOut;
//   dtIdxOut.resize(rpcOutSize);

  std::vector< size_t > rpcInAss( rpcInSize, 0 );
  std::vector< size_t > rpcOutAss( rpcOutSize, 0 );

  for ( size_t iDt = 0; iDt < dtSize; ++iDt ) {

    double phi = _dtAssociatedStubs.at(iDt)->getCMSGlobalPhi();
    std::map< double, size_t > rpcInIdx;
    std::map< double, size_t > rpcOutIdx;

    for ( size_t iIn = 0; iIn < rpcInSize; ++iIn ) {
      double phiIn = _rpcInAssociatedStubs.at( iIn )->getCMSGlobalPhi();
      double deltaPhiIn = fabs( reco::deltaPhi( phi, phiIn ) );
      if ( deltaPhiIn < minRpcPhi ) {
	rpcInIdx[ deltaPhiIn ] = iIn;
	++rpcInAss[iIn];
// 	dtIdxIn[iIn][ deltaPhiIn ] = iDt;
      }
    }

    for ( size_t iOut = 0; iOut < rpcOutSize; ++iOut ) {
      double phiOut = _rpcOutAssociatedStubs.at( iOut )->getCMSGlobalPhi();
      double deltaPhiOut = fabs( reco::deltaPhi( phi, phiOut ) );
      if ( deltaPhiOut < minRpcPhi ) {
	rpcOutIdx[ deltaPhiOut ] = iOut;
	++rpcOutAss[iOut];
// 	dtIdxOut[iOut][ deltaPhiOut ] = iDt;
      }
    }

    L1TwinMux::MBLTCollection::primitiveAssociation & dtAss = _dtMapAss.at(iDt);

    /// fill up index for In associations
    std::map< double, size_t >::const_iterator it = rpcInIdx.begin();
    std::map< double, size_t >::const_iterator itend = rpcInIdx.end();
    dtAss.rpcIn.reserve( rpcInIdx.size() );
    for ( ; it != itend; ++it ) dtAss.rpcIn.push_back( it->second );

    /// fill up index for Out associations
    it = rpcOutIdx.begin();
    itend = rpcOutIdx.end();
    dtAss.rpcOut.reserve( rpcOutIdx.size() );
    for ( ; it != itend; ++it ) dtAss.rpcOut.push_back( it->second );

  }

  /// fill unassociated rpcIn
  for ( size_t iIn = 0; iIn < rpcInSize; ++iIn ) {
    if ( !rpcInAss.at(iIn) ) _rpcMapUnass.rpcIn.push_back(iIn);
  }
  if ( _rpcInAssociatedStubs.size() < _rpcMapUnass.rpcIn.size() )
    throw cms::Exception("More unassociated IN hits than the total rpc IN hits") << std::endl;

  /// fill unassociated rpcOut
  for ( size_t iOut = 0; iOut < rpcOutSize; ++iOut ) {
    if ( !rpcOutAss.at(iOut) ) _rpcMapUnass.rpcOut.push_back(iOut);
  }
  if ( _rpcOutAssociatedStubs.size() < _rpcMapUnass.rpcOut.size() )
    throw cms::Exception("More unassociated OUT hits than the total OUT rpc hits") << std::endl;

}


L1TwinMux::TriggerPrimitiveList L1TwinMux::MBLTCollection::getRpcInAssociatedStubs( size_t dtIndex ) const
{

  L1TwinMux::TriggerPrimitiveList returnList;

  try {
    const primitiveAssociation & prim = _dtMapAss.at(dtIndex);
    std::vector<size_t>::const_iterator it = prim.rpcIn.begin();
    std::vector<size_t>::const_iterator itend = prim.rpcIn.end();

    for ( ; it != itend; ++it ) returnList.push_back( _rpcInAssociatedStubs.at( *it ) );

  } catch ( const std::out_of_range & e ) {
    throw cms::Exception("DT Chamber Out of Range")
      << "Requested DT primitive in position " << dtIndex << " out of " << _dtMapAss.size() << " total primitives"
      << std::endl;
  }

  return returnList;

}


L1TwinMux::TriggerPrimitiveList L1TwinMux::MBLTCollection::getRpcOutAssociatedStubs( size_t dtIndex ) const
{

  L1TwinMux::TriggerPrimitiveList returnList;

  try {
    const primitiveAssociation & prim = _dtMapAss.at(dtIndex);

    std::vector<size_t>::const_iterator it = prim.rpcOut.begin();
    std::vector<size_t>::const_iterator itend = prim.rpcOut.end();

    for ( ; it != itend; ++it ) returnList.push_back( _rpcOutAssociatedStubs.at( *it ) );
  } catch ( const std::out_of_range & e ) {
    throw cms::Exception("DT Chamber Out of Range")
      << "The number of dt primitives in sector are " << _dtMapAss.size()
      << std::endl;
  }

  return returnList;

}




L1TwinMux::TriggerPrimitiveList
L1TwinMux::MBLTCollection::getRpcInUnassociatedStubs() const
{

  L1TwinMux::TriggerPrimitiveList returnList;
  std::vector<size_t>::const_iterator it = _rpcMapUnass.rpcIn.begin();
  std::vector<size_t>::const_iterator itend = _rpcMapUnass.rpcIn.end();

  for ( ; it != itend; ++it )
    returnList.push_back( _rpcInAssociatedStubs.at( *it ) );

  return returnList;

}


L1TwinMux::TriggerPrimitiveList
L1TwinMux::MBLTCollection::getRpcOutUnassociatedStubs() const
{

  L1TwinMux::TriggerPrimitiveList returnList;
  std::vector<size_t>::const_iterator it = _rpcMapUnass.rpcOut.begin();
  std::vector<size_t>::const_iterator itend = _rpcMapUnass.rpcOut.end();

  for ( ; it != itend; ++it )
    returnList.push_back( _rpcOutAssociatedStubs.at( *it ) );

  return returnList;

}




L1TwinMux::MBLTCollection::bxMatch L1TwinMux::MBLTCollection::haveCommonRpc( size_t dt1, size_t dt2 ) const
{

  L1TwinMux::MBLTCollection::bxMatch ret_val = NOMATCH;

  if ( dt1 == dt2 ) {
    throw cms::Exception("DT primitive compared to itself")
      << "The two id passed refer to the same primitive"
      << std::endl;
  }

  try {
    const primitiveAssociation & prim1 = _dtMapAss.at(dt1);
    const primitiveAssociation & prim2 = _dtMapAss.at(dt2);

//     bool in_match = false;
//     bool out_match = false;

//     size_t rpcInSize1 = prim1.rpcIn.size();
//     size_t rpcInSize2 = prim2.rpcIn.size();
//     for ( size_t i = 0; i < rpcInSize1; ++i )
//       for ( size_t j = 0; j < rpcInSize2; ++j )
// 	if ( prim1.rpcIn[i] == prim1.rpcIn[j] ) {
// 	  in_match = true;
// 	  i = rpcInSize1;
// 	  break;
// 	}

//     size_t rpcOutSize1 = prim1.rpcOut.size();
//     size_t rpcOutSize2 = prim2.rpcOut.size();
//     for ( size_t i = 0; i < rpcOutSize1; ++i )
//       for ( size_t j = 0; j < rpcOutSize2; ++j )
// 	if ( prim1.rpcOut[i] == prim1.rpcOut[j] ) {
// 	  out_match = true;
// 	  i = rpcOutSize1;
// 	  break;
// 	}
//     if ( in_match && out_match ) return FULLMATCH;
//     else if ( in_match ) return INMATCH;
//     else if ( out_match ) return OUTMATCH;
//     return NOMATCH;


    if ( !prim1.rpcIn.empty() && !prim2.rpcIn.empty() ) {
      if ( prim1.rpcIn.front() == prim2.rpcIn.front() ) {
    	ret_val = INMATCH;
      }
    }

    if ( !prim1.rpcOut.empty() && !prim2.rpcOut.empty() ) {
      if ( prim1.rpcOut.front() == prim2.rpcOut.front() ) {
    	ret_val = ( ret_val == INMATCH ) ? FULLMATCH : OUTMATCH;
      }
    }
    return ret_val;

  } catch ( const std::out_of_range & e ) {
    throw cms::Exception("DT Chamber Out of Range")
      << "The number of dt primitives in sector are " << _dtMapAss.size()
      << std::endl;
  }

  return ret_val;

}



/////////// RPC UTIL
bool
L1TwinMux::MBLTCollection::areCloseClusters( std::vector< size_t > & cluster1,
					  std::vector< size_t > & cluster2,
					  const L1TwinMux::TriggerPrimitiveList & rpcList1,
					  const L1TwinMux::TriggerPrimitiveList & rpcList2,
					  double minRpcPhi ) const
{

  size_t clSize1 = cluster1.size();
  size_t clSize2 = cluster2.size();

  for ( size_t idx1 = 0; idx1 < clSize1; ++idx1 ) {

    size_t uidx1 = cluster1.at(idx1);
    double phi1 = rpcList1.at( uidx1 )->getCMSGlobalPhi();

    for ( size_t idx2 = 0; idx2 < clSize2; ++idx2 ) {

      size_t uidx2 = cluster2.at(idx2);
      double phi2 = rpcList2.at( uidx2 )->getCMSGlobalPhi();
      double deltaPhiIn = fabs( reco::deltaPhi( phi1, phi2 ) );
      if ( deltaPhiIn < minRpcPhi ) {
	return true;
      }
    }
  }

  return false;

}


/////////// RPC UTILS
size_t
L1TwinMux::MBLTCollection::reduceRpcClusters( std::vector< std::vector <size_t> > & clusters,
					   const L1TwinMux::TriggerPrimitiveList & rpcList,
					   double minRpcPhi ) const
{

  size_t clusterSize = clusters.size();
  if ( clusterSize < 2 ) return 0;

  std::vector<bool> pickUpClusterMap( clusterSize, true );

  size_t reduced = 0;
  for ( size_t cidx1 = 0; cidx1 < clusterSize; ++cidx1 ) {

    if ( pickUpClusterMap.at(cidx1) ) {
      for ( size_t cidx2 = cidx1+1; cidx2 < clusterSize; ++cidx2 ) {
	if ( pickUpClusterMap.at(cidx2) &&
	     areCloseClusters( clusters.at(cidx1), clusters.at(cidx2),
			       rpcList, rpcList, minRpcPhi ) ) {

	  clusters.at(cidx1).insert( clusters.at(cidx1).end(),
				     clusters.at(cidx2).begin(),
				     clusters.at(cidx2).end() );
	  pickUpClusterMap[cidx2] = false;
	  ++reduced;
	}
      }
    }
  }

  if ( reduced ) {
    // std::cout << "### Reduce..." << std::endl; CB commented out, is it really needed?
    std::vector< std::vector <size_t> > tmpClusters;
    for ( size_t cidx = 0; cidx < clusterSize; ++cidx ) {
      if ( pickUpClusterMap.at(cidx) ) {
	tmpClusters.push_back( clusters.at(cidx) );
      }
    }
    clusters = tmpClusters;
  }

  return reduced;
}


/////////// RPC UTIL
void
L1TwinMux::MBLTCollection::getUnassociatedRpcClusters( const std::vector< size_t > & rpcUnass,
						    const L1TwinMux::TriggerPrimitiveList & rpcList,
						    double minRpcPhi,
						    std::vector< std::vector <size_t> > & clusters ) const
{

  if ( rpcUnass.empty() ) return;

  size_t rpcSizeU = rpcUnass.size();
  std::vector<bool> pickUpMap( rpcSizeU, true );

  for ( size_t idx1 = 0; idx1 < rpcSizeU; ++idx1 ) {

    if ( pickUpMap.at(idx1) ) {

      /// remember: we are running over an array of indices
      size_t uidx1 = rpcUnass.at(idx1);

      std::vector<size_t> subCluster = { uidx1 };
      double phi1 = rpcList.at( uidx1 )->getCMSGlobalPhi();

      for ( size_t idx2 = idx1+1; idx2 < rpcSizeU; ++idx2 ) {

	if ( pickUpMap.at(idx2) ) {

	  /// remember: we are running over an array of indices
	  size_t uidx2 = rpcUnass.at(idx2);

	  double phi2 = rpcList.at( uidx2 )->getCMSGlobalPhi();
	  double deltaPhiIn = fabs( reco::deltaPhi( phi1, phi2 ) );
	  if ( deltaPhiIn < minRpcPhi ) {
	    subCluster.push_back( uidx2 );
	    pickUpMap[idx2] = false;
	  }
	}
      }
      clusters.push_back( subCluster );
    }
  }

  size_t reduced = 0;
  do {
    reduced = reduceRpcClusters( clusters, rpcList, minRpcPhi );
  } while ( reduced );

}




/////////// RPC UTIL
std::vector< std::pair< L1TwinMux::TriggerPrimitiveList, L1TwinMux::TriggerPrimitiveList > >
L1TwinMux::MBLTCollection::getUnassociatedRpcClusters( double minRpcPhi ) const
{

  using L1TwinMux::TriggerPrimitiveList;

  std::vector< std::pair< L1TwinMux::TriggerPrimitiveList, L1TwinMux::TriggerPrimitiveList > > associated;

  if ( _rpcMapUnass.rpcIn.empty() && _rpcMapUnass.rpcOut.empty())
    return associated;
  ////////////////////////////////////////////////////
  /// loop over unassociated inner and outer RPC hits

  std::vector< std::vector <size_t> > inClusters;
  try {
    getUnassociatedRpcClusters( _rpcMapUnass.rpcIn, _rpcInAssociatedStubs,
				minRpcPhi, inClusters );
  } catch ( const std::out_of_range & e ) {
    std::cout << " getUnassociatedRpcClusters " << e.what() << std::endl;
    throw cms::Exception("RPC HIT Out of Range")<< std::endl;
  }

  size_t rpcInClusterSize = inClusters.size();
  std::vector< std::vector <size_t> > outClusters;
  try {
    getUnassociatedRpcClusters( _rpcMapUnass.rpcOut, _rpcOutAssociatedStubs,
				minRpcPhi, outClusters );
  } catch ( const std::out_of_range & e ) {
    std::cout << " getUnassociatedRpcClusters " << e.what() << std::endl;
    throw cms::Exception("RPC HIT Out of Range")<< std::endl;
  }

  size_t rpcOutClusterSize = outClusters.size();
  //////////

  std::vector<bool> spareInMap( rpcInClusterSize, true );
  std::vector<bool> spareOutMap( rpcOutClusterSize, true );

  for ( size_t in = 0; in < rpcInClusterSize; ++in ) {
    for ( size_t out = 0; out < rpcOutClusterSize; ++out ) {
      if ( areCloseClusters( inClusters.at(in),
			     outClusters.at(out),
			     _rpcInAssociatedStubs,
			     _rpcOutAssociatedStubs,
			     minRpcPhi ) ) {

	L1TwinMux::TriggerPrimitiveList primIn;
	std::vector<size_t>::const_iterator itIn = inClusters.at(in).begin();
	std::vector<size_t>::const_iterator itInEnd = inClusters.at(in).end();
	for ( ; itIn != itInEnd; ++itIn )
	  primIn.push_back( _rpcInAssociatedStubs.at(*itIn) );

	L1TwinMux::TriggerPrimitiveList primOut;
	std::vector<size_t>::const_iterator itOut = outClusters.at(out).begin();
	std::vector<size_t>::const_iterator itOutEnd = outClusters.at(out).end();
	for ( ; itOut != itOutEnd; ++itOut )
	  primOut.push_back( _rpcOutAssociatedStubs.at(*itOut) );

	associated.push_back( std::pair< L1TwinMux::TriggerPrimitiveList, L1TwinMux::TriggerPrimitiveList >(primIn, primOut) );
	spareInMap[in] = false;
	spareOutMap[out] = false;
      }
    }
  }

  for ( size_t in = 0; in < rpcInClusterSize; ++in ) {
    if ( spareInMap[in] ) {

      L1TwinMux::TriggerPrimitiveList primIn;
      std::vector<size_t>::const_iterator itIn = inClusters.at(in).begin();
      std::vector<size_t>::const_iterator itInEnd = inClusters.at(in).end();
      for ( ; itIn != itInEnd; ++itIn )
	primIn.push_back( _rpcInAssociatedStubs.at(*itIn) );

      L1TwinMux::TriggerPrimitiveList primOut;
      associated.push_back( std::pair< L1TwinMux::TriggerPrimitiveList, L1TwinMux::TriggerPrimitiveList >(primIn, primOut) );
    }
  }

  for ( size_t out = 0; out < rpcOutClusterSize; ++out ) {
    if ( spareOutMap[out] ) {


      L1TwinMux::TriggerPrimitiveList primIn;
      L1TwinMux::TriggerPrimitiveList primOut;
      std::vector<size_t>::const_iterator itOut = outClusters.at(out).begin();
      std::vector<size_t>::const_iterator itOutEnd = outClusters.at(out).end();
      for ( ; itOut != itOutEnd; ++itOut )
	primOut.push_back( _rpcOutAssociatedStubs.at(*itOut) );

      associated.push_back( std::pair< L1TwinMux::TriggerPrimitiveList, L1TwinMux::TriggerPrimitiveList >(primIn, primOut) );
    }
  }


  return associated;

}
