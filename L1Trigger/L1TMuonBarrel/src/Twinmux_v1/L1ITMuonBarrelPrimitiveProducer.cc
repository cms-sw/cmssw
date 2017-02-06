//Authors:
//Luigi Guiducci - Giuseppe Codispoti
// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/ConsumesCollector.h>

// L1IT include files
#include "L1Trigger/L1TMuon/interface/deprecate/MuonTriggerPrimitive.h"

#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/MBLTCollection.h"
#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/MBLTCollectionFwd.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/PrimitiveCombiner.h"

// user include files
#include "DataFormats/Math/interface/deltaPhi.h"

using namespace L1TwinMux;
using namespace L1TMuon;

class L1ITMuonBarrelPrimitiveProducer  {

public:
  inline L1ITMuonBarrelPrimitiveProducer(std::shared_ptr<MBLTContainer> _mbltContainer);
  inline ~L1ITMuonBarrelPrimitiveProducer();
  inline virtual std::auto_ptr<L1MuDTChambPhContainer> produce( const edm::EventSetup&);

private:
  edm::InputTag _mbltCollectionInput = edm::InputTag("MBLTProducer");
  edm::ESHandle<DTGeometry> _muonGeom;
  //std::auto_ptr<MBLTContainer> mbltContainer;
  std::shared_ptr<MBLTContainer> mbltContainer;
};

inline std::ostream & operator<< (std::ostream & out, const TriggerPrimitiveList & rpc )
{
  std::vector<TriggerPrimitiveRef>::const_iterator it = rpc.begin();
  std::vector<TriggerPrimitiveRef>::const_iterator end = rpc.end();
  for ( ; it != end ; ++it ) out << (*it)->getCMSGlobalPhi() << '\t';
  out << std::endl;
  return out;
}

inline L1ITMuonBarrelPrimitiveProducer::~L1ITMuonBarrelPrimitiveProducer()
{
}

inline L1ITMuonBarrelPrimitiveProducer::L1ITMuonBarrelPrimitiveProducer( std::shared_ptr<MBLTContainer> _mbltContainer )
: mbltContainer(_mbltContainer)
{
  //produces<L1MuDTChambPhContainer>("L1ITMuonBarrelPrimitiveProducer");

 //consumes<MBLTContainer>(iConfig.getParameter<edm::InputTag>("MBLTCollection"));
 //mbltContainer = _mbltContainer;
}


inline std::auto_ptr<L1MuDTChambPhContainer> L1ITMuonBarrelPrimitiveProducer::produce(const edm::EventSetup& iSetup )
{

  const PrimitiveCombiner::resolutions _resol(0.1, 2., 0.005, 0.04);
  const int _qualityRemappingMode = 2;
  const int _useRpcBxForDtBelowQuality = 5;
  const bool _is7QualityCodes = true;


  iSetup.get<MuonGeometryRecord>().get(_muonGeom);

  std::auto_ptr<L1MuDTChambPhContainer> out(new L1MuDTChambPhContainer);
  std::vector<L1MuDTChambPhDigi> phiChambVector;

 // edm::Handle<MBLTContainer> mbltContainer;
  //iEvent.getByLabel( _mbltCollectionInput, mbltContainer );

  MBLTContainer::const_iterator st = mbltContainer->begin();
  MBLTContainer::const_iterator stend = mbltContainer->end();

  L1MuDTChambPhContainer phiContainer;
  std::vector<L1MuDTChambPhDigi> phiVector;

  for ( ; st != stend; ++st ) {

    const  MBLTCollection & mbltStation = st->second;

    /// useful index
    int station = mbltStation.station();
    int wheel = mbltStation.wheel();
    int sector = mbltStation.sector();
    ///

    /// get dt to rpc associations
    size_t dtListSize = mbltStation.getDtSegments().size();
    std::vector<size_t> uncorrelated;
    std::vector<size_t> correlated;
    for ( size_t iDt = 0; iDt < dtListSize; ++iDt ) {
      const TriggerPrimitiveRef & dt = mbltStation.getDtSegments().at(iDt);
      int dtquality = dt->getDTData().qualityCode;
      //if ( dtquality == 2 || dtquality == 3 ) std::cout << "[o]" << dtquality << '\t' ; /// GC
      // if ( dtquality > 3 ) std::cout << "[o]" << dtquality << '\t' ; /// GC
      /// define new set of qualities
      /// skip for the moment uncorrelated
      // int qualityCode = -2;
      // switch ( dtquality ) {
      // case -1 : continue;/// -1 are theta
      // case 0 : /* qualityCode = -2;*/ break;
      // case 1 : /* qualityCode = -2;*/ break;
      // case 2 : uncorrelated.push_back( iDt ); continue;
      // case 3 : uncorrelated.push_back( iDt ); continue;
      // case 4 : correlated.push_back( iDt ); continue; //qualityCode = 5; break;
      // case 5 : correlated.push_back( iDt ); continue; //qualityCode = 5; break;
      // case 6 : correlated.push_back( iDt ); continue; //qualityCode = 5; break;
      // default : /* qualityCode = dtquality; */ break;
      // }

      switch ( dtquality ) {
      case -1 : continue;/// -1 are theta
      case 0 : /* qualityCode = -2;*/ break;
      case 1 : /* qualityCode = -2;*/ break;
      case 2 : uncorrelated.push_back( iDt ); continue;  // HI
      case 3 : uncorrelated.push_back( iDt ); continue;  // HO
      case 4 : correlated.push_back( iDt ); continue;    // LL
      case 5 : correlated.push_back( iDt ); continue;    // HL
      case 6 : correlated.push_back( iDt ); continue;    // HH
      default : /* qualityCode = dtquality; */ break;
      }

      //L1MuDTChambPhDigi chamb( dt->getBX(), wheel, sector-1, station, dt->getDTData().radialAngle,
      //		       dt->getDTData().bendingAngle, qualityCode,
      //			       dt->getDTData().Ts2TagCode, dt->getDTData().BxCntCode );
      //phiChambVector.push_back( chamb );
    }

    // START OF BX ANALYSIS FOR CORRELATED TRIGGER
    size_t cSize = correlated.size();
    for ( size_t idxDt = 0; idxDt < cSize; ++idxDt ) {
      int bx=-999;
      int iDt = correlated.at(idxDt);
      if ( iDt < 0 ) continue;
      const TriggerPrimitive & dt = *mbltStation.getDtSegments().at(iDt);
      TriggerPrimitiveList rpcInMatch = mbltStation.getRpcInAssociatedStubs( iDt );
      TriggerPrimitiveList rpcOutMatch = mbltStation.getRpcOutAssociatedStubs( iDt );
      size_t rpcInMatchSize = rpcInMatch.size();
      size_t rpcOutMatchSize = rpcOutMatch.size();
      if ( rpcInMatchSize && rpcOutMatchSize ) {
	const TriggerPrimitive & rpcIn = *rpcInMatch.front();
	const TriggerPrimitive & rpcOut = *rpcOutMatch.front();
	/// only the first is real...
	// LG try also to reassign BX to single H using RPC BX, e.g. do not ask for DT and RPC to have the same BX
	if ( ( dt.getBX() == rpcIn.getBX() && dt.getBX() == rpcOut.getBX() )
	    || (_qualityRemappingMode>1 && rpcIn.getBX()==rpcOut.getBX() && abs(dt.getBX()-rpcIn.getBX())<=1) ) {
	  bx = rpcIn.getBX();
	}
      } else if (rpcInMatchSize){
	const TriggerPrimitive & rpcIn = *rpcInMatch.front();
	if ( dt.getBX() == rpcIn.getBX() || (_qualityRemappingMode>1 && abs(dt.getBX()-rpcIn.getBX())<=1)) {
	  bx = rpcIn.getBX();
	}
      }
      else if (rpcOutMatchSize){
	const TriggerPrimitive & rpcOut = *rpcOutMatch.front();
	if ( dt.getBX() == rpcOut.getBX() || (_qualityRemappingMode>1 && abs(dt.getBX()-rpcOut.getBX())<=1)) {
	  bx = rpcOut.getBX();
	}
      }
      // add primitive here
      int newBx=dt.getBX();
      if (bx>-999 && dt.getDTData().qualityCode<_useRpcBxForDtBelowQuality){
	newBx=bx;
      }

      int qualityCode = 6;
      if ( ! _is7QualityCodes ) {
	qualityCode = 13;
	switch ( dt.getDTData().qualityCode ) {
	case 4 : qualityCode = 14; break;    // LL // TODO: LL+rpc=13
	case 5 : qualityCode = 15; break;    // HL
	case 6 : qualityCode = 15; break;    // HH
	default : break;
	}
      }

      //std::cout << "[n]" << dt.getDTData().qualityCode  << std::endl; /// GC
      L1MuDTChambPhDigi chamb( newBx, wheel, sector-1, station, dt.getDTData().radialAngle,
			       dt.getDTData().bendingAngle, qualityCode,
			       dt.getDTData().Ts2TagCode, dt.getDTData().BxCntCode );
      phiChambVector.push_back( chamb );
    }
    // END OF BX ANALYSIS FOR CORRELATED TRIGGER

    // BEGIN OF BX ANALYSIS FOR UNCORRELATED TRIGGER
    size_t uncSize = uncorrelated.size();
    for ( size_t idxDt = 0; idxDt < uncSize; ++idxDt ) {

      int iDt = uncorrelated.at(idxDt);
      if ( iDt < 0 ) continue;
      const TriggerPrimitive & dt = *mbltStation.getDtSegments().at(iDt);

      /// check if there is a pair of HI+HO at different bx
      int closest = -1;
      int closestIdx = -1;
      double minDeltaPhiDt = 9999999999;
      for ( size_t jdxDt = idxDt+1; jdxDt < uncSize; ++jdxDt ) {

	int jDt = uncorrelated.at(jdxDt);
	if ( jDt < 0 ) continue;

	const TriggerPrimitiveRef & dtM = mbltStation.getDtSegments().at(jDt);
	if ( dt.getBX() == dtM->getBX() || dt.getDTData().qualityCode == dtM->getDTData().qualityCode )
	  continue;

	double deltaPhiDt = fabs( reco::deltaPhi( dt.getCMSGlobalPhi(), dtM->getCMSGlobalPhi() ) );
	if ( deltaPhiDt < minDeltaPhiDt ) {
	  closest = jDt;
	  closestIdx = jdxDt;
	  minDeltaPhiDt=deltaPhiDt;
	}
      }

      /// check if the pair shares the closest rpc hit
      MBLTCollection::bxMatch match = MBLTCollection::NOMATCH;
      if ( closest > 0 && minDeltaPhiDt < 0.05 ) {
      //if ( closest > 0 ) {
	match = mbltStation.haveCommonRpc( iDt, closest );
      }

      /// this is just a set of output variables for building L1ITMuDTChambPhDigi
      // int qualityCode = dt.getDTData().qualityCode;
      int bx = -2;
      int radialAngle = 0;
      int bendingAngle = 0;
      PrimitiveCombiner combiner( _resol, _muonGeom );
      /// association HI/HO provided by the tool
      combiner.addDt( dt );

      /// there is a pair HI+HO with a shared inner RPC hit
      if ( match != MBLTCollection::NOMATCH ) {
	uncorrelated[closestIdx] = -1;

	/// association HI/HO provided by the tool
	combiner.addDt( *mbltStation.getDtSegments().at(closest) );

	/// redefine quality
	/// qualityCode = 4;
	TriggerPrimitiveList rpcInMatch = mbltStation.getRpcInAssociatedStubs( iDt );
	TriggerPrimitiveList rpcOutMatch = mbltStation.getRpcOutAssociatedStubs( iDt );

	/// there is a pair HI+HO with a shared inner RPC hit
	if ( match == MBLTCollection::INMATCH ) {

	  const TriggerPrimitive & rpcIn = *rpcInMatch.front();
	  combiner.addRpcIn( rpcIn );
	  bx = rpcIn.getBX();

	  /// there is a pair HI+HO with a shared outer RPC hit
	} else if ( match == MBLTCollection::OUTMATCH ) {

	  const TriggerPrimitive & rpcOut = *rpcOutMatch.front();
	  combiner.addRpcOut( rpcOut );
	  bx = rpcOut.getBX();

	  /// there is a pair HI+HO with both shared inner and outer RPC hit
	} else if ( match == MBLTCollection::FULLMATCH ) {

	  const TriggerPrimitive & rpcIn = *rpcInMatch.front();
	  const TriggerPrimitive & rpcOut = *rpcOutMatch.front();
	  combiner.addRpcIn( rpcIn );
	  combiner.addRpcOut( rpcOut );
	  bx = rpcIn.getBX();
	}


      } else { /// there is no match

	TriggerPrimitiveList rpcInMatch = mbltStation.getRpcInAssociatedStubs( iDt );
	TriggerPrimitiveList rpcOutMatch = mbltStation.getRpcOutAssociatedStubs( iDt );
	size_t rpcInMatchSize = rpcInMatch.size();
	size_t rpcOutMatchSize = rpcOutMatch.size();

	/// the uncorrelated has possibly inner and outer confirmation
	if ( rpcInMatchSize && rpcOutMatchSize ) {
	  const TriggerPrimitive & rpcIn = *rpcInMatch.front();
	  const TriggerPrimitive & rpcOut = *rpcOutMatch.front();
	  /// only the first is real...
	  // LG try also to reassign BX to single H using RPC BX, e.g. do not ask for DT and RPC to have the same BX
	  if (( dt.getBX() == rpcIn.getBX() && dt.getBX() == rpcOut.getBX() )
	      || (_qualityRemappingMode>1 && rpcIn.getBX()==rpcOut.getBX() && abs(dt.getBX()-rpcIn.getBX())<=1)) {
	    bx = rpcIn.getBX();
	    combiner.addRpcIn( rpcIn );
	    combiner.addRpcOut( rpcOut );
	  } else if ( dt.getBX() == rpcIn.getBX() ) {
	    bx = rpcIn.getBX();
	    combiner.addRpcIn( rpcIn );
	  } else if ( dt.getBX() == rpcOut.getBX() ) {
	    bx = rpcOut.getBX();
	    combiner.addRpcOut( rpcOut );
	  }

	/// the uncorrelated has a possible inner confirmation
	} else if ( rpcInMatchSize ) {
	  const TriggerPrimitive & rpcIn = *rpcInMatch.front();
	  if ( dt.getBX() == rpcIn.getBX() || (_qualityRemappingMode>1 && abs(dt.getBX()-rpcIn.getBX())<=1)) {
	    bx = rpcIn.getBX();
	    combiner.addRpcIn( rpcIn );
	  }

	/// the uncorrelated has a possible outer confirmation
	} else if ( rpcOutMatchSize ) {
	  const TriggerPrimitive & rpcOut = *rpcOutMatch.front();
	  if ( dt.getBX() == rpcOut.getBX()|| (_qualityRemappingMode>1  && abs(dt.getBX()-rpcOut.getBX())<=1)) {
	    bx = rpcOut.getBX();
	    combiner.addRpcOut( rpcOut );
	  }

	}
      }

      // match found, PrimitiveCombiner has the needed variables already calculated
      // 2016: the DT spatial parameters are not updated
      if ( combiner.isValid() ) {
	//std::cout<<"=== I am making a combination ==="<<std::endl;
	combiner.combine();
	radialAngle = dt.getDTData().radialAngle;
	bendingAngle = dt.getDTData().bendingAngle;
	
	//radialAngle = combiner.radialAngle();
	//bendingAngle = (combiner.bendingAngle() < -511 || combiner.bendingAngle() > 511) ? dt.getDTData().bendingAngle : combiner.bendingAngle( );
	
      } else {
	// no match found, keep the primitive as it is
	bx = dt.getBX();
	radialAngle = dt.getDTData().radialAngle;
	bendingAngle = dt.getDTData().bendingAngle;
	//if (_qualityRemappingMode==0)
	// qualityCode = ( qualityCode == 2 ) ? 0 : 1;
      }

      int qualityCode = ( _is7QualityCodes ?
			  combiner.getUncorrelatedQuality7() :
			  combiner.getUncorrelatedQuality16() );

      // std::cout << "[n]" << qualityCode << std::endl; /// GC
      L1MuDTChambPhDigi chamb( bx, wheel, sector-1, station, radialAngle,
			       bendingAngle, qualityCode,
			       dt.getDTData().Ts2TagCode, dt.getDTData().BxCntCode );
      phiChambVector.push_back( chamb );
      //if (abs(bendingAngle)>511||1==1){
	//	std::cout<<"Got bending angle: "<<bendingAngle<<std::endl;
	//std::cout<<"Original DT primitive had bending angle: "<<dt.getDTData().bendingAngle<<std::endl;
	//std::cout<<"Original radial angle: "<<radialAngle<<std::endl;
	//std::cout<<"Quality: "<<qualityCode<<std::endl;
	//std::cout<<"Station: "<<station<<std::endl;
      //}

    } /// end of the Uncorrelated loop
//     ////////////////////////////////////////////////////
//     /// loop over unassociated inner and outer RPC hits
//     const TriggerPrimitiveList & rpcInUnass = mbltStation.getRpcInUnassociatedStubs();
//     const TriggerPrimitiveList & rpcOutUnass = mbltStation.getRpcOutUnassociatedStubs();

//     size_t rpcInUSize = rpcInUnass.size();
//     size_t rpcOutUsize = rpcOutUnass.size();

//     for ( size_t in = 0; in < rpcInUSize; ++in ) {
//     for ( size_t out = 0; out < rpcOutUSize; ++out ) {

    const std::vector< std::pair< TriggerPrimitiveList, TriggerPrimitiveList > >
      rpcPairList = mbltStation.getUnassociatedRpcClusters( 0.05 );
    auto rpcPair = rpcPairList.cbegin();
    auto rpcPairEnd = rpcPairList.cend();
    for ( ; rpcPair != rpcPairEnd; ++ rpcPair ) {
      const TriggerPrimitiveList & inRpc = rpcPair->first;
      const TriggerPrimitiveList & outRpc = rpcPair->second;

      if ( inRpc.empty() && outRpc.empty() ) continue;

      PrimitiveCombiner combiner( _resol, _muonGeom );
      size_t inSize = inRpc.size();
      size_t outSize = outRpc.size();
      int station = -1;
      int sector  = -1;
      int wheel = -5;
      // double qualityCode = 0;

      if ( inSize ) {
	//std::cout<<"Producer working on IN&&!OUT"<<std::endl;
        size_t inPos = 0;
        // double avPhiIn = 0;
        double avPhiSin = 0;
        double avPhiCos = 0;
        for ( size_t i = 0; i < inSize; ++i ) {
          double locPhi = inRpc.at(i)->getCMSGlobalPhi();
          // avPhiIn += ( locPhi > 0 ? locPhi : 2*M_PI + locPhi );
	  avPhiSin += sin( locPhi );
	  avPhiCos += cos( locPhi );
        }
        // avPhiIn /= inSize;
	avPhiSin /= inSize;
	avPhiCos /= inSize;
	double avPhiIn = atan2( avPhiSin, avPhiCos );

        double minDist = fabs( inRpc.at(0)->getCMSGlobalPhi() - avPhiIn );
        for ( size_t i = 1; i < inSize; ++i ) {
          double dist = fabs( inRpc.at(i)->getCMSGlobalPhi() - avPhiIn );
          if ( dist < minDist ) {
            inPos = i;
            minDist = dist;
          }
        }

	// const TriggerPrimitive & rpc = (*inRpc.at(inPos));
	TriggerPrimitive rpc = (*inRpc.at(inPos));
	rpc.setCMSGlobalPhi( avPhiIn );
        station = rpc.detId<RPCDetId>().station();
        sector  = rpc.detId<RPCDetId>().sector();
        wheel = rpc.detId<RPCDetId>().ring();
        combiner.addRpcIn( rpc );


      }
      if ( outSize ) {
	//std::cout<<"Producer working on OUT&&!IN"<<std::endl;
        size_t outPos = 0;
        //double avPhiOut = 0;
        double avPhiSin = 0;
        double avPhiCos = 0;
        for ( size_t i = 0; i < outSize; ++i ) {
          double locPhi = outRpc.at(i)->getCMSGlobalPhi();
          // avPhiOut += ( locPhi > 0 ? locPhi : 2*M_PI + locPhi );
	  avPhiSin += sin( locPhi );
	  avPhiCos += cos( locPhi );
        }

        //avPhiOut /= outSize;
	avPhiSin /= outSize;
	avPhiCos /= outSize;
	double avPhiOut = atan2( avPhiSin, avPhiCos );
        double minDist = fabs( outRpc.at(0)->getCMSGlobalPhi() - avPhiOut );
        for ( size_t i = 1; i < outSize; ++i ) {
          double dist = fabs( outRpc.at(i)->getCMSGlobalPhi() - avPhiOut );
          if ( dist < minDist ) {
            outPos = i;
            minDist = dist;
          }
        }
        // const TriggerPrimitive & rpc = (*outRpc.at(outPos));
	TriggerPrimitive rpc = (*outRpc.at(outPos));
	rpc.setCMSGlobalPhi( avPhiOut );
        station = rpc.detId<RPCDetId>().station();
        sector  = rpc.detId<RPCDetId>().sector();
        wheel = rpc.detId<RPCDetId>().ring();
        combiner.addRpcOut( rpc );
      }
	//else // {
//         //	std::cout<<"Producer working on IN&&OUT"<<std::endl;
//         size_t inPos = 0;
//         size_t outPos = 0;
//         double minDist = 9999;

//       for ( size_t i = 0; i < inSize; ++i ) {
//           for ( size_t j = 0; j < outSize; ++j ) {

//             double dist = fabs( inRpc.at(0)->getCMSGlobalPhi()
//                                 - outRpc.at(0)->getCMSGlobalPhi() );
//             if ( dist < minDist ) {
//               inPos = i;
//               outPos = j;
//               minDist = dist;
//             }
//           }
//         }
//         const TriggerPrimitive & rpc_in = (*inRpc.at(inPos));

//         const TriggerPrimitive & rpc_out = (*outRpc.at(outPos));
//         station = rpc_in.detId<RPCDetId>().station();
//         sector  = rpc_in.detId<RPCDetId>().sector();
//         wheel = rpc_in.detId<RPCDetId>().ring();
//         combiner.addRpcIn( rpc_in );
//         combiner.addRpcOut( rpc_out );
//         qualityCode = 1;
//       }

      // if (inSize && outSize) qualityCode=1;
      combiner.combine();
      double radialAngle = combiner.radialAngle();
      double bendingAngle = combiner.bendingAngle();
      double bx = combiner.bx();
      double Ts2TagCode = 0;
      double BxCntCode = 0;


      int qualityCode = ( _is7QualityCodes ?
			  combiner.getUncorrelatedQuality7() :
			  combiner.getUncorrelatedQuality16() );
      if ( qualityCode >= 0 ) {
	// std::cout << "[r]" << qualityCode << std::endl ; /// GC
	L1MuDTChambPhDigi chamb( bx, wheel, sector-1, station, radialAngle,
				 bendingAngle, qualityCode,
				 Ts2TagCode, BxCntCode );
	phiChambVector.push_back( chamb );
      }

      //std::cout << "IN: \n" << inRpc;
      //std::cout << "OUT: \n" << outRpc;
      //std::cout << "\n";

    }

  }




  out->setContainer( phiChambVector );
  /// fill event
  //iEvent.put(out);
  return out;

}

//#include "FWCore/Framework/interface/MakerMacros.h"
//DEFINE_FWK_MODULE(L1ITMuonBarrelPrimitiveProducer);

