//// Trigger Primitive Converter for RPC hits
////
//// Takes in raw information from the TriggerPrimitive class
//// (part of L1TMuon software package) and outputs vector of 'ConvertedHits'
////

#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveConverterRPC.h"

PrimitiveConverterRPC::PrimitiveConverterRPC() {
}

l1t::EMTFHit2016ExtraCollection 
PrimitiveConverterRPC::convert( std::vector<L1TMuon::TriggerPrimitive> TrigPrim, 
				int SectIndex, edm::ESHandle<RPCGeometry> rpc_geom ) {

  // bool verbose = true;
  bool verbose = false;


  if (verbose) std::cout << "\n========== RPC Primitive Converter ==========" << std::endl;

  l1t::EMTFHit2016ExtraCollection tmpHits;
  for (std::vector<L1TMuon::TriggerPrimitive>::iterator iter = TrigPrim.begin(); iter != TrigPrim.end(); iter++) {

    /// Get all the input variables
    L1TMuon::TriggerPrimitive prim = *iter; // Eventually would like to deprecate TriggerPrimitive entirely - AWB 03.06.16
    RPCDetId detID = prim.detId<RPCDetId>();
    RPCDigi digi = RPCDigi( prim.getRPCData().strip, prim.getRPCData().bx );

    // Only include RPC hits from correct sector in endcap
    if ( abs(detID.region()) != 1 ) continue;
    if ( SectIndex != (detID.sector() - 1) + (detID.region() == -1)*6 ) continue;

    l1t::EMTFHit2016Extra thisHit;
    thisHit.ImportRPCDetId( detID );
    thisHit.ImportRPCDigi( digi );
    thisHit.set_sector_index( SectIndex );
    thisHit.set_layer( prim.getRPCData().layer ); // In RE1/2 there are two layers of chambers: 1 is inner (front) and 2 is outer (rear)
    
    tmpHits.push_back( thisHit );
  }

  l1t::EMTFHit2016ExtraCollection clustHits;
  for (unsigned int iHit = 0; iHit < tmpHits.size(); iHit++) {
    l1t::EMTFHit2016Extra hit1 = tmpHits.at(iHit);

    // Skip hit if it is already in a cluster
    bool hit_in_cluster = false;
    for (unsigned int jHit = 0; jHit < clustHits.size(); jHit++) {
      l1t::EMTFHit2016Extra clustHit = clustHits.at(jHit);
      if ( sameRpcChamber(hit1, clustHit) && hit1.Strip_hi() <= clustHit.Strip_hi() && 
	   hit1.Strip_low() >= clustHit.Strip_low() ) hit_in_cluster = true;
    }
    if (hit_in_cluster) continue;

    // Cluster adjascent strip hits.  Phi of cluster corresponds to central strip.
    // Does this clustering catch all the hits in large clusters? - AWB 03.06.16
    int prevHi = -999, prevLow = -999;
    while (hit1.Strip_hi() != prevHi || hit1.Strip_low() != prevLow) {
      prevHi = hit1.Strip_hi();
      prevLow = hit1.Strip_low();

      for (unsigned int jHit = 0; jHit < tmpHits.size(); jHit++) {
	if (iHit == jHit) continue;
	l1t::EMTFHit2016Extra hit2 = tmpHits.at(jHit);

	if (not sameRpcChamber(hit1, hit2)) continue;
	if (hit2.Strip_hi()  == hit1.Strip_hi()  + 1) hit1.set_strip_hi ( hit2.Strip_hi()  );
	if (hit2.Strip_low() == hit1.Strip_low() - 1) hit1.set_strip_low( hit2.Strip_low() );
      }
    }
    
    // Get phi and eta
    std::unique_ptr<const RPCRoll> roll(rpc_geom->roll( hit1.RPC_DetId() ));
    const LocalPoint lpHi = roll->centreOfStrip(hit1.Strip_hi());
    const GlobalPoint gpHi = roll->toGlobal(lpHi);
    const LocalPoint lpLow = roll->centreOfStrip(hit1.Strip_low());
    const GlobalPoint gpLow = roll->toGlobal(lpLow);
    roll.release();

    float glob_phi_hi  = gpHi.phi();  // Averaging global point phi's directly does weird things,
    float glob_phi_low = gpLow.phi(); // e.g. 2*pi/3 + 2*pi/3 = 4*pi/3 = -2*pi/3, so avg. = -pi/3, not 2*pi/3
    float glob_phi = (glob_phi_hi + glob_phi_low) / 2.0;
    float glob_eta = (gpHi.eta() + gpLow.eta()) / 2.0;

    if (verbose) std::cout << "RPC cluster phi = " << glob_phi << " (" << glob_phi_hi << ", " << glob_phi_low 
			   << "), eta = " << glob_eta << " (" << gpHi.eta() << ", " << gpLow.eta() << ")" << std::endl;

    hit1.set_phi_glob_rad( glob_phi );
    hit1.set_phi_glob_deg( glob_phi*180/Geom::pi() );
    hit1.set_eta( glob_eta );
    clustHits.push_back( hit1 );

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // Need to set phi_loc_int, theta_int, phi_hit, zone, csc_ID, quality,
    // pattern, wire, strip, phi_zone, and zone_contribution
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
  } // End loop over iHit

  return clustHits;

} // End PrimitiveConverterRPC::convert

std::vector<ConvertedHit>
PrimitiveConverterRPC::fillConvHits(l1t::EMTFHit2016ExtraCollection exHits) {

  std::vector<ConvertedHit> ConvHits;
  for (unsigned int iHit = 0; iHit < exHits.size(); iHit++) {
    l1t::EMTFHit2016Extra exHit = exHits.at(iHit);

    // // Replace with SetZoneWord - AWB 04.09.16
    // std::vector<int> zone_contribution;

    ConvertedHit ConvHit;
    ConvHit.SetValues(exHit.Phi_loc_int(), exHit.Theta_int(), exHit.Phi_hit(), exHit.Zone(), 
		      exHit.Station(), exHit.Subsector(), exHit.CSC_ID(), exHit.Quality(), 
		      exHit.Pattern(), exHit.Wire(), exHit.Strip(), exHit.BX() + 6);
    ConvHit.SetTP( L1TMuon::TriggerPrimitive( exHit.RPC_DetId(), exHit.Strip(), exHit.Layer(), exHit.BX() + 6 ) );
    ConvHit.SetZhit( exHit.Phi_zone() );
    // // Replace with SetZoneWord - AWB 04.09.16
    // ConvHit.SetZoneContribution(zone_contribution);
    ConvHit.SetSectorIndex( exHit.Sector_index() );
    ConvHit.SetNeighbor(0);
    ConvHits.push_back(ConvHit);
  }
  return ConvHits;
}

bool PrimitiveConverterRPC::sameRpcChamber( l1t::EMTFHit2016Extra hitA, l1t::EMTFHit2016Extra hitB ) {

  if ( hitA.Endcap() == hitB.Endcap() && hitA.Station() == hitB.Station() && hitA.Ring() == hitB.Ring() &&
       hitA.Roll() == hitB.Roll() && hitA.Sector() == hitB.Sector() && hitA.Subsector() == hitB.Subsector() &&
       hitA.Layer() == hitB.Layer() ) return true;
  else return false;
}
