///////////////////////////////////////////////////////////////
// Upgraded Encdap Muon Track Finding Algorithm		     //
//							     //
// Info: A human-readable version of the firmware based      //
//       track finding algorithm which will be implemented   //
//       in the upgraded endcaps of CMS. DT and RPC inputs   //
//	     are not considered in this algorithm.           //
//							     //
// Author: M. Carver (UF)				     //
///////////////////////////////////////////////////////////////

#define NUM_SECTORS 12

#include "L1Trigger/L1TMuonEndCap/plugins/L1TMuonEndCapTrackProducer.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCPatternLUT.h"
#include "L1Trigger/CSCTrackFinder/test/src/RefTrack.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "L1Trigger/L1TMuonEndCap/interface/BXAnalyzer.h"
#include "L1Trigger/L1TMuonEndCap/interface/ZoneCreation.h"
#include "L1Trigger/L1TMuonEndCap/interface/PatternRecognition.h"
#include "L1Trigger/L1TMuonEndCap/interface/SortSector.h"
#include "L1Trigger/L1TMuonEndCap/interface/Matching.h"
#include "L1Trigger/L1TMuonEndCap/interface/Deltas.h"
#include "L1Trigger/L1TMuonEndCap/interface/BestTracks.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignment.h"
#include "L1Trigger/L1TMuonEndCap/interface/ChargeAssignment.h"
#include "L1Trigger/L1TMuonEndCap/interface/MakeRegionalCand.h"

// New EDM output for detailed track and hit information - AWB 01.04.16
#include "L1Trigger/L1TMuonEndCap/interface/EMTFTrackTools.h"
#include "L1Trigger/L1TMuonEndCap/interface/EMTFHitTools.h"

using namespace L1TMuon;
class RPCGeometry;
// Pointers to the current geometry records
unsigned long long _geom_cache_id;
edm::ESHandle<RPCGeometry> _geom_rpc;


L1TMuonEndCapTrackProducer::L1TMuonEndCapTrackProducer(const PSet& p) {

  inputTokenCSC = consumes<CSCCorrelatedLCTDigiCollection>(p.getParameter<edm::InputTag>("CSCInput"));
  bxShiftCSC = p.getUntrackedParameter<int>("CSCInputBxShift", 0);
  inputTokenRPC = consumes<RPCDigiCollection>(p.getParameter<edm::InputTag>("RPCInput"));
  
  produces<l1t::RegionalMuonCandBxCollection >("EMTF");
  produces< l1t::EMTFTrackCollection >("");
  produces< l1t::EMTFHitCollection >("");  
  produces< l1t::EMTFTrackExtraCollection >("");
  produces< l1t::EMTFHitExtraCollection >("CSC");  
  produces< l1t::EMTFHitExtraCollection >("RPC");  

}


void L1TMuonEndCapTrackProducer::produce(edm::Event& ev,
			       const edm::EventSetup& es) {

  //bool verbose = false;

  //fprintf (write,"12345\n"); //<-- part of printing text file to send verilog code, not needed if George's package is included

  //std::auto_ptr<L1TMuon::InternalTrackCollection> FoundTracks (new L1TMuon::InternalTrackCollection);
  auto FoundTracks = std::make_unique<L1TMuon::InternalTrackCollection>();
  auto OutputCands = std::make_unique<l1t::RegionalMuonCandBxCollection>();
  auto OutTracks = std::make_unique<l1t::EMTFTrackCollection>();
  auto OutHits = std::make_unique<l1t::EMTFHitCollection>();
  auto OutputTracks = std::make_unique<l1t::EMTFTrackExtraCollection>();
  auto OutputHits = std::make_unique<l1t::EMTFHitExtraCollection>();
  auto OutputHitsRPC = std::make_unique<l1t::EMTFHitExtraCollection>();

  std::vector<BTrack> PTracks[NUM_SECTORS];
  std::vector<BTrack> PTracks_BX[NUM_SECTORS][3];

  std::vector<TriggerPrimitive> tester;
  std::vector<TriggerPrimitive> tester_rpc;
  //std::vector<InternalTrack> FoundTracks;

  // Get the RPC geometry
  const MuonGeometryRecord& geom = es.get<MuonGeometryRecord>();
  unsigned long long geomid = geom.cacheIdentifier();
  if( _geom_cache_id != geomid )
    geom.get(_geom_rpc);

  
  //////////////////////////////////////////////
  ///////// Make Trigger Primitives ////////////
  //////////////////////////////////////////////
  
  edm::Handle<CSCCorrelatedLCTDigiCollection> MDC;
  ev.getByToken(inputTokenCSC, MDC);

  edm::Handle<RPCDigiCollection> RDC;
  ev.getByToken(inputTokenRPC, RDC);
  
  std::vector<TriggerPrimitive> out;
  std::vector<TriggerPrimitive> out_rpc;

  auto chamber = MDC->begin();
  auto chend  = MDC->end();
  for( ; chamber != chend; ++chamber ) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for( ; digi != dend; ++digi ) {
      CSCCorrelatedLCTDigi tmp_digi = *digi;
      tmp_digi.setBX( tmp_digi.getBX() + std::max(bxShiftCSC, -1*tmp_digi.getBX()) );
      out.push_back(TriggerPrimitive((*chamber).first,tmp_digi));
      l1t::EMTFHitExtra thisHit;
      thisHit.ImportCSCDetId( (*chamber).first );
      thisHit.ImportCSCCorrelatedLCTDigi( tmp_digi );
      if (thisHit.Station() == 1 && thisHit.Ring() == 1 && thisHit.Strip() > 127) thisHit.set_ring(4);
      thisHit.set_neighbor(0);
      OutputHits->push_back( thisHit );
      if ( ((thisHit.Ring() != 1 || thisHit.Station() == 1) && (thisHit.Chamber() % 6 == 2)) ||
	   ((thisHit.Ring() == 1 && thisHit.Station()  > 1) && (thisHit.Chamber() % 3 == 1)) ) {
	l1t::EMTFHitExtra neighborHit = thisHit;
	neighborHit.set_neighbor(1);
	OutputHits->push_back( neighborHit );
      }
    }
  }

  auto rchamber = RDC->begin();
  auto rchend  = RDC->end();
  for( ; rchamber != rchend; ++rchamber) {
    auto rdigi = (*rchamber).second.first;
    auto rdend = (*rchamber).second.second;
    for( ; rdigi != rdend; ++rdigi) {
      out_rpc.push_back(TriggerPrimitive( (*rchamber).first, rdigi->strip(), 0, rdigi->bx())); // Layer unset.  How to access? - AWB 03.06.16 
    }
  }
  
  
  //////////////////////////////////////////////
  ///////// Get Trigger Primitives /////////////  Retrieve TriggerPrimitives from the event record: Currently does nothing because we don't take RPC's
  //////////////////////////////////////////////
  
  // auto tpsrc = _tpinputs.cbegin();
  //auto tpend = _tpinputs.cend();
  // for( ; tpsrc != tpend; ++tpsrc ) {
  // edm::Handle<TriggerPrimitiveCollection> tps;
  // ev.getByLabel(*tpsrc,tps);
  auto tp = out_rpc.cbegin();
  auto tpend = out_rpc.cend();
  
  for( ; tp != tpend; ++tp ) {
    if (tp->subsystem() == 2) tester_rpc.push_back(*tp);
  }
  
  // Create extra converted hits for chambers with two LCTs (ambiguous strip/wire pairing)
  for(unsigned int i1=0;i1<out.size();i1++){
    tester.push_back(out[i1]);

    for(unsigned int i2=i1+1;i2<out.size();i2++){
      if ( out[i1].detId<CSCDetId>().station() == out[i2].detId<CSCDetId>().station() &&
	   out[i1].detId<CSCDetId>().endcap() == out[i2].detId<CSCDetId>().endcap() &&
	   out[i1].detId<CSCDetId>().triggerSector() == out[i2].detId<CSCDetId>().triggerSector() &&
	   out[i1].detId<CSCDetId>().ring() == out[i2].detId<CSCDetId>().ring() &&
	   out[i1].detId<CSCDetId>().chamber() == out[i2].detId<CSCDetId>().chamber() &&
	   out[i1].getBX() == out[i2].getBX() && out[i1].Id() == out[i2].Id() &&
	   out[i1].getStrip() != out[i2].getStrip() && out[i1].getWire() != out[i2].getWire() ) {
	
	TriggerPrimitive NewWire1(out[i1],out[i2]);
	TriggerPrimitive NewWire2(out[i2],out[i1]);
	tester.push_back(NewWire1);
	tester.push_back(NewWire2);
      } 
    } // End loop: for(unsigned int i2=i1+1;i2<out.size();i2++)
  } // End loop: for(unsigned int i1=0;i1<out.size();i1++)

  uint nHits = OutputHits->size();
  for (uint iHit = 0; iHit < nHits; iHit++) {
    for (uint jHit = iHit+1; jHit < nHits; jHit++) {
      if ( OutputHits->at(iHit).Chamber() != OutputHits->at(jHit).Chamber() ) continue;
      if ( (OutputHits->at(iHit).Ring() % 3) != (OutputHits->at(jHit).Ring() % 3) ) continue;
      if ( OutputHits->at(iHit).Sector() != OutputHits->at(jHit).Sector() ) continue;
      if ( OutputHits->at(iHit).Station() != OutputHits->at(jHit).Station() ) continue;
      if ( OutputHits->at(iHit).Endcap() != OutputHits->at(jHit).Endcap() ) continue;
      if ( OutputHits->at(iHit).BX() != OutputHits->at(jHit).BX() ) continue;
      if ( OutputHits->at(iHit).Neighbor() != OutputHits->at(jHit).Neighbor() ) continue;
      if ( OutputHits->at(iHit).Strip() == OutputHits->at(jHit).Strip() ) continue;
      if ( OutputHits->at(iHit).Wire() == OutputHits->at(jHit).Wire() ) continue;
      
      l1t::EMTFHitExtra new_hit_1 = OutputHits->at(iHit).Clone();
      l1t::EMTFHitExtra new_hit_2 = OutputHits->at(jHit).Clone();
      new_hit_1.set_wire( OutputHits->at(jHit).Wire() );
      new_hit_2.set_wire( OutputHits->at(iHit).Wire() );
      new_hit_1.SetCSCLCTDigi( new_hit_1.CreateCSCCorrelatedLCTDigi() );
      new_hit_2.SetCSCLCTDigi( new_hit_2.CreateCSCCorrelatedLCTDigi() );
      OutputHits->push_back( new_hit_1 );
      OutputHits->push_back( new_hit_2 );
    }
  }

  std::vector<ConvertedHit> CHits[NUM_SECTORS];
  // MatchingOutput MO[NUM_SECTORS];
  
  for(int SectIndex=0;SectIndex<NUM_SECTORS;SectIndex++){//perform TF on all 12 sectors

    //////////////////////////////////////////////////////  Input is raw hit information from
    ///////////////// TP Conversion //////////////////////  Output is vector of Converted Hits
    //////////////////////////////////////////////////////
    
    std::vector<ConvertedHit> ConvHits = primConv_.convert(tester,SectIndex);
    CHits[SectIndex] = ConvHits;

    l1t::EMTFHitExtraCollection tmp_hits_rpc = primConvRPC_.convert(tester_rpc, SectIndex, _geom_rpc); 
    for (uint iHit = 0; iHit < tmp_hits_rpc.size(); iHit++) 
      OutputHitsRPC->push_back( tmp_hits_rpc.at(iHit) );
    std::vector<ConvertedHit> ConvHitsRPC = primConvRPC_.fillConvHits(tmp_hits_rpc);
    
    // Fill OutputHits with ConvertedHit information
    for (uint iCHit = 0; iCHit < ConvHits.size(); iCHit++) {
      // bool isMatched = false;
      
      for (uint iHit = 0; iHit < OutputHits->size(); iHit++) {
	if ( ConvHits.at(iCHit).Station()   == OutputHits->at(iHit).Station() &&
	     ( ConvHits.at(iCHit).Id()      == OutputHits->at(iHit).CSC_ID()  ||
	       ConvHits.at(iCHit).Id()      == ( (OutputHits->at(iHit).Ring() != 4) // Account for either ME1/1a 
						 ? OutputHits->at(iHit).CSC_ID()    // CSC ID numbering convention
						 : OutputHits->at(iHit).CSC_ID() + 9 ) ) &&
	     ConvHits.at(iCHit).Wire()       == OutputHits->at(iHit).Wire()    &&
	     ConvHits.at(iCHit).Strip()      == OutputHits->at(iHit).Strip()   &&
	     ConvHits.at(iCHit).BX() - 6     == OutputHits->at(iHit).BX()      &&
	     ConvHits.at(iCHit).IsNeighbor() == OutputHits->at(iHit).Neighbor() ) {
	  // isMatched = true;
	  OutputHits->at(iHit).set_neighbor    ( ConvHits.at(iCHit).IsNeighbor());
	  OutputHits->at(iHit).set_sector_index( ConvHits.at(iCHit).SectorIndex() );
	  OutputHits->at(iHit).set_phi_zone    ( ConvHits.at(iCHit).Zhit()   );
	  OutputHits->at(iHit).set_phi_hit     ( ConvHits.at(iCHit).Ph_hit() );
	  OutputHits->at(iHit).set_zone        ( ConvHits.at(iCHit).Phzvl()  );
	  OutputHits->at(iHit).set_phi_loc_int ( ConvHits.at(iCHit).Phi()    );
	  OutputHits->at(iHit).set_theta_int   ( ConvHits.at(iCHit).Theta()  );

	  // // Replace with ZoneWord - AWB 04.09.16
	  // OutputHits->at(iHit).SetZoneContribution ( ConvHits.at(iCHit).ZoneContribution() );
	  OutputHits->at(iHit).set_phi_loc_deg  ( l1t::calc_phi_loc_deg( OutputHits->at(iHit).Phi_loc_int() ) );
	  OutputHits->at(iHit).set_phi_loc_rad  ( l1t::calc_phi_loc_rad( OutputHits->at(iHit).Phi_loc_int() ) );
	  OutputHits->at(iHit).set_phi_glob_deg ( l1t::calc_phi_glob_deg_hit( OutputHits->at(iHit).Phi_loc_deg(), OutputHits->at(iHit).Sector_index() ) );
	  OutputHits->at(iHit).set_phi_glob_rad ( l1t::calc_phi_glob_rad_hit( OutputHits->at(iHit).Phi_loc_rad(), OutputHits->at(iHit).Sector_index() ) );
	  OutputHits->at(iHit).set_theta_deg    ( l1t::calc_theta_deg_from_int( OutputHits->at(iHit).Theta_int() ) );
	  OutputHits->at(iHit).set_theta_rad    ( l1t::calc_theta_rad_from_int( OutputHits->at(iHit).Theta_int() ) );
	  OutputHits->at(iHit).set_eta( l1t::calc_eta_from_theta_rad( OutputHits->at(iHit).Theta_rad() ) * OutputHits->at(iHit).Endcap() );
	  
	  OutHits->push_back( OutputHits->at(iHit).CreateEMTFHit() );
	}
      } // End loop: for (uint iHit = 0; iHit < OutputHits->size(); iHit++)

      // if (isMatched == false) {
      //   std::cout << "***********************************************" << std::endl;
      //   std::cout << "Unmatched ConvHit in event " << ev.id().event() << ", SectIndex " << SectIndex << std::endl;
      //   std::cout << "ConvHit: station = " << ConvHits.at(iCHit).Station() << ", CSC ID = " << ConvHits.at(iCHit).Id()
      // 	      << ", sector index = " << ConvHits.at(iCHit).SectorIndex() << ", subsector = " << ConvHits.at(iCHit).Sub()
      // 	      << ", wire = " << ConvHits.at(iCHit).Wire() << ", strip = " << ConvHits.at(iCHit).Strip()
      // 	      << ", BX = " << ConvHits.at(iCHit).BX() << ", neighbor = " << ConvHits.at(iCHit).IsNeighbor() << std::endl;
      
      //   for (uint iHit = 0; iHit < OutputHits->size(); iHit++) {
      //     std::cout << "EMTFHitExtra: station = " << OutputHits->at(iHit).Station() << ", CSC ID = " << OutputHits->at(iHit).CSC_ID()
      // 		<< ", sector index = " << OutputHits->at(iHit).Sector_index() << ", subsector = " << OutputHits->at(iHit).Subsector() 
      // 		<< ", wire = " << OutputHits->at(iHit).Wire() << ", strip = " << OutputHits->at(iHit).Strip()
      // 		<< ", BX = " << OutputHits->at(iHit).BX() << ", neighbor = " << OutputHits->at(iHit).Neighbor()
      // 		<< ", chamber = " << OutputHits->at(iHit).Chamber() << ", ring = " << OutputHits->at(iHit).Ring() 
      // 		<< ", endcap = " << OutputHits->at(iHit).Endcap() << ", sector = " << OutputHits->at(iHit).Sector() << std::endl;
      //   }
      // }
      
    } // End loop: for (uint iCHit = 0; iCHit < ConvHits.size(); iCHit++)
    
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////print values for input into Alex's emulator code/////////////////////////////////////////////////////
    //for(std::vector<ConvertedHit>::iterator h = ConvHits.begin();h != ConvHits.end();h++){
    
    //if((h->Id()) > 9){h->SetId(h->Id() - 9);h->SetStrip(h->Strip() + 128);}
    //fprintf (write,"0	1	1 	%d	%d\n",h->Sub(),h->Station());
    //fprintf (write,"1	%d	%d 	%d\n",h->Quality(),h->Pattern(),h->Wire());
    //fprintf (write,"%d	0	%d\n",h->Id(),h->Strip());
    //}
    ////////////////////////////////print values for input into Alex's emulator code/////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////  Takes the vector of converted hits and groups into 3 groups of hits
    ////////////////////// BX Grouper ////////////////////  which are 3 BX's wide. Effectively looking 2 BX's into the future and
    //////////////////////////////////////////////////////  past from the central BX, this analyzes a total of 5 BX's.
    //////////////////////////////////////////////////////
    
    std::vector<std::vector<ConvertedHit>> GroupedHits = GroupBX(ConvHits);
    
    
    ////////////////////////////////////////////////////////  Creates a zone for each of the three groups created in the BX Grouper module.
    ////////// Creat Zones for pattern Recognition /////////  The output of this module not only contains the zones but also the
    ////////////////////////////////////////////////////////  reference back to the TriggerPrimitives that went into making them.
    
    std::vector<ZonesOutput> Zout = Zones(GroupedHits);
    
    
    ///////////////////////////////
    ///// Pattern Recognition /////  Applies pattern recognition logic on each of the 3 BX groups and assigns a quality to each keystrip in the zone.
    ///// & quality assinment /////  The delete duplicate patterns function looks at the 3 BX groups and deletes duplicate patterns found from the
    ///////////////////////////////  same hits. This is where the BX analysis ends; Only 1 list of found patterns is given to the next module.
    
    std::vector<PatternOutput> Pout = Patterns(Zout);
    std::vector<PatternOutput> Pout_Hold = Pout;
    
    // PatternOutput Test = DeleteDuplicatePatterns(Pout);
    
    //PrintQuality(Test.detected);
    

    ///////////////////////////////
    //////Sector Sorting/////////// Sorts through the patterns found in each zone and selects the best three per zone to send to the next module.
    ///////Finding 3 Best Pattern//
    ///////////////////////////////
    
    // SortingOutput Sout = SortSect(Test);
    std::vector<SortingOutput> Sout_Hold = SortSect_Hold(Pout_Hold);
    
    
    //////////////////////////////////
    ///////// Match ph patterns ////// Loops over each sorted pattern and then loops over all possible triggerprimitives which could have made the pattern
    ////// to segment inputs ///////// and matches the associated full precision triggerprimitives to the detected pattern.
    //////////////////////////////////
    
    // MatchingOutput Mout = PhiMatching(Sout);
    std::vector<MatchingOutput> Mout_Hold = PhiMatching_Hold(Sout_Hold);
    
    /////////////////////////////////
    //////// Calculate delta //////// Once we have matched the hits we calculate the delta phi and theta between all
    ////////    ph and th    //////// stations present.
    /////////////////////////////////
    
    // std::vector<std::vector<DeltaOutput>> Dout = CalcDeltas(Mout);////
    DeltaOutArr3 Dout_Hold = CalcDeltas_Hold(Mout_Hold);
    
    
    /////////////////////////////////
    /////// Sorts and gives /////////  Loops over all of the found tracks(looking across zones) and selects the best three per sector.
    ////// Best 3 tracks/sector /////  Here ghost busting is done to delete tracks which are comprised of the same associated stubs.
    /////////////////////////////////
    
    // std::vector<BTrack> Bout = BestTracks(Dout);
    // PTracks[SectIndex] = Bout;

    std::vector<std::vector<BTrack>> Bout_Hold = BestTracks_Hold(Dout_Hold);
    for(int bx=0;bx<3;bx++)
      PTracks_BX[SectIndex][bx] = Bout_Hold[bx];

    
  } // End loop: for(int SectIndex=0;SectIndex<NUM_SECTORS;SectIndex++)
  
  
  /////////////////////////////////////// and BX windows and cancel across BXs even though not yet in FW
  /// Collect Muons from all sectors //// to see if sub percent kind of effect. If more then I can try to
  /////////////////////////////////////// FW exactly but it will take a lot more thinking. - mrcarver 24.06.16
  
  std::vector<BTrack> PTemp[NUM_SECTORS];
  std::vector<BTrack> AllTracks, AllTracks_PreCancel;
  for (int i=0; i<NUM_SECTORS; i++) PTemp[i] = PTracks[i];
  
  for(int bx=0;bx<3;bx++){
    for(int j=0;j<36;j++){
      
      // if(PTemp[j/3][j%3].phi)//no track
	// AllTracks.push_back(PTemp[j/3][j%3]);

      if(PTracks_BX[j/3][bx][j%3].phi) {//no track
	AllTracks_PreCancel.push_back(PTracks_BX[j/3][bx][j%3]);
	if (PTracks_BX[j/3][bx][j%3].theta == 0)
	  LogTrace("L1TMuonEndCapTrackProducer") << "PTrack_BX theta = 0" << std::endl;
      }
      
    }
  }

  // Cancel out tracks with identical hits
  for (unsigned int i1 = 0; i1 < AllTracks_PreCancel.size(); i1++) {
    bool dup = false;
    int rank1 = AllTracks_PreCancel[i1].winner.Rank();
    int rank2 = -99;
    int i1_dup = i1;
    int i2_dup = -99;
    for (unsigned int i2 = 0; i2 < AllTracks_PreCancel.size(); i2++) {
      if (i1 == i2) continue;
      for (std::vector<ConvertedHit>::iterator A1 = AllTracks_PreCancel[i1].AHits.begin(); A1 != AllTracks_PreCancel[i1].AHits.end(); A1++) {
	CSCDetId D1 = A1->TP().detId<CSCDetId>();
	TriggerPrimitive::CSCData C1 = A1->TP().getCSCData();
	for (std::vector<ConvertedHit>::iterator A2 = AllTracks_PreCancel[i2].AHits.begin(); A2 != AllTracks_PreCancel[i2].AHits.end(); A2++) {
	  CSCDetId D2 = A2->TP().detId<CSCDetId>();
	  TriggerPrimitive::CSCData C2 = A2->TP().getCSCData();
	  if ( A1->SectorIndex() == A2->SectorIndex() &&  D1.endcap() == D2.endcap() && D1.station() == D2.station() && 
	       D1.ring() == D2.ring() && D1.triggerSector() == D2.triggerSector() && D1.chamber() == D2.chamber() &&
	       C1.bx == C2.bx && C1.strip == C2.strip && C1.keywire == C2.keywire ) {
	    dup = true;
	    if (AllTracks_PreCancel[i2].winner.Rank() > rank2) {
	      i2_dup = i2;
	      rank2 = AllTracks_PreCancel[i2].winner.Rank();
	    } // Find the highest-ranked duplicate track
	  } // End if hits are identical
	} // End loop over hits in track 2
      } // End loop over hits in track 1
    } // End second loop over tracks

    // Only use ordering when ranks are equal.  Track order is not necessarily the same as FW. - AWB 21.09.16
    if ( (!dup) || (rank1 > rank2) || (rank1 == rank2 && i1_dup < i2_dup) ) AllTracks.push_back(AllTracks_PreCancel[i1]);
  } // End first loop over tracks

  
  ///////////////////////////////////
  /// Make Internal track if ////////
  /////// tracks are found //////////
  /////////////////////////////////// 
  
  std::vector<l1t::RegionalMuonCand> tester1;
  std::vector<std::pair<int,l1t::RegionalMuonCand>> holder, holder2;
  
  for(unsigned int fbest=0;fbest<AllTracks.size();fbest++){
    
    if(AllTracks[fbest].phi){
      
      InternalTrack tempTrack;
      tempTrack.setType(2);
      tempTrack.phi = AllTracks[fbest].phi;
      tempTrack.theta = AllTracks[fbest].theta;
      tempTrack.rank = AllTracks[fbest].winner.Rank();
      tempTrack.deltas = AllTracks[fbest].deltas;
      std::vector<int> ps, ts;

      if (tempTrack.theta == 0) LogTrace("L1TMuonEndCapTrackProducer") << "Track has theta 0" << std::endl;
      
      l1t::EMTFTrackExtra thisTrack;
      thisTrack.set_phi_loc_int ( AllTracks[fbest].phi           );
      thisTrack.set_theta_int   ( AllTracks[fbest].theta         );
      thisTrack.set_rank        ( AllTracks[fbest].winner.Rank() );
      // thisTrack.set_deltas        ( AllTracks[fbest].deltas        );
      int tempStraightness = 0;
      int tempRank = thisTrack.Rank();
      if (tempRank & 64)
	tempStraightness |= 4;
      if (tempRank & 16)
	tempStraightness |= 2;
      if (tempRank & 4)
	tempStraightness |= 1;
      thisTrack.set_straightness ( tempStraightness );

      int mode = 0;
      if(tempTrack.rank & 32)
	mode |= 8;
      if(tempTrack.rank & 8)
	mode |= 4;
      if(tempTrack.rank & 2)
	mode |= 2;
      if(tempTrack.rank & 1)
	mode |= 1;
      
      int sector = -1;
      // bool ME13 = false;
      int me1address = 0, me2address = 0, CombAddress = 0, mode_uncorr = 0;
      int ebx = 20, sebx = 20;
      int phis[4] = {-99,-99,-99,-99};
      
      int cHits_in_station[4] = {0,0,0,0};
      int eHits_in_station[4] = {0,0,0,0};
      for(std::vector<ConvertedHit>::iterator A = AllTracks[fbest].AHits.begin();A != AllTracks[fbest].AHits.end();A++){
	
	if(A->Phi() != -999){
	  cHits_in_station[A->TP().detId<CSCDetId>().station() - 1] += 1;
	  
	  l1t::EMTFHitExtra thisHit;
	  // thisHit.ImportCSCDetId( A->TP().detId<CSCDetId>() );

	  for (uint iHit = 0; iHit < OutputHits->size(); iHit++) {
	    if ( (A->TP().detId<CSCDetId>().endcap() == 1) == (OutputHits->at(iHit).Endcap() == 1) &&
		 A->TP().detId<CSCDetId>().station()       == OutputHits->at(iHit).Station() &&
		 A->TP().detId<CSCDetId>().triggerSector() == OutputHits->at(iHit).Sector()  &&
		 A->TP().getCSCData().cscID                == OutputHits->at(iHit).CSC_ID()  &&
		 A->Wire()                                 == OutputHits->at(iHit).Wire()    &&
		 A->Strip()                                == OutputHits->at(iHit).Strip()   &&
		 A->TP().getCSCData().bx - 6               == OutputHits->at(iHit).BX()      &&
		 A->IsNeighbor()                           == OutputHits->at(iHit).Neighbor() ) {
	      thisHit = OutputHits->at(iHit);
	      // Hacky fix because ConvertedHits are not removed when theta windows are applied - AWB 30.06.16
	      if ( (A->TP().detId<CSCDetId>().station() == 1 && (mode & 8)) ||
		   (A->TP().detId<CSCDetId>().station() == 2 && (mode & 4)) || 
		   (A->TP().detId<CSCDetId>().station() == 3 && (mode & 2)) || 
		   (A->TP().detId<CSCDetId>().station() == 4 && (mode & 1)) ) {
		eHits_in_station[OutputHits->at(iHit).Station() - 1] += 1;
		thisTrack.push_HitExtraIndex(iHit);
		thisTrack.push_HitExtra(thisHit); 
	      }
	      break;
	    }
	  }
	  
	  if ( thisHit.Station() < 0 ) {
	    LogTrace("L1TMuonEndCapTrackProducer") << "!@#$ Converted hit with station " << A->TP().detId<CSCDetId>().station() << ", CSC_ID " << A->TP().getCSCData().cscID
		      << ", sector index " << A->SectorIndex() << ", subsector " << A->Sub()
		      << ", wire " << A->Wire() << ", strip " << A->Strip() << ", BX " << A->TP().getCSCData().bx - 6 
		      << ", neighbor " << A->IsNeighbor() << " has no match" << std::endl;
	    for (uint iHit = 0; iHit < OutputHits->size(); iHit++) 
	      LogTrace("L1TMuonEndCapTrackProducer") << "!@#$ Option " << iHit+1 << " with endcap " << OutputHits->at(iHit).Endcap() 
			<< ", station " << OutputHits->at(iHit).Station() << ", CSC_ID " << OutputHits->at(iHit).CSC_ID() 
			<< ", ring " << OutputHits->at(iHit).Ring() << ", chamber " << OutputHits->at(iHit).Chamber()
			<< ", wire " << OutputHits->at(iHit).Wire() << ", strip " << OutputHits->at(iHit).Strip()
			<< ", BX " << OutputHits->at(iHit).BX() << ", neighbor " << OutputHits->at(iHit).Neighbor() << std::endl;
	  }
	  
	  thisTrack.set_endcap       ( thisHit.Endcap() );
	  thisTrack.set_sector_index ( thisHit.Sector_index() );
	  thisTrack.set_sector       ( l1t::calc_sector_from_index( thisHit.Sector_index() ) );
	  thisTrack.set_sector_GMT   ( l1t::calc_sector_GMT( thisHit.Sector() ) );
	  if ( thisHit.Neighbor() == 0 ) thisTrack.set_all_neighbor(0);
	  if ( thisHit.Neighbor() == 1 ) thisTrack.set_has_neighbor(1);
	  if ( thisHit.Neighbor() == 0 && thisTrack.Has_neighbor() == -999 ) thisTrack.set_has_neighbor(0);
	  if ( thisHit.Neighbor() == 1 && thisTrack.All_neighbor() == -999 ) thisTrack.set_all_neighbor(0);
	  
	  int station = A->TP().detId<CSCDetId>().station();
	  int id = A->TP().getCSCData().cscID;
	  int trknm = A->TP().getCSCData().trknmb;
	  
	  phis[station-1] = A->Phi();
	  
	  
	  if(A->TP().getCSCData().bx < ebx){
	    sebx = ebx;
	    ebx = A->TP().getCSCData().bx;
	  }
	  else if(A->TP().getCSCData().bx < sebx){
	    sebx = A->TP().getCSCData().bx;
	  }
	  
	  tempTrack.addStub(A->TP());
	  ps.push_back(A->Phi());
	  ts.push_back(A->Theta());
	  
	  sector = A->SectorIndex();
	  
	  switch(station){
	  case 1: mode_uncorr |= 8;break;
	  case 2: mode_uncorr |= 4;break;
	  case 3: mode_uncorr |= 2;break;
	  case 4: mode_uncorr |= 1;break;
	  default: mode_uncorr |= 0;
	  }
	  
	  
	  // if(A->TP().detId<CSCDetId>().station() == 1 && A->TP().detId<CSCDetId>().ring() == 3)
	  //   ME13 = true;
	  
	  if(station == 1 && id > 3 && id < 7){
	    
	    int sub = 2;
	    if(A->TP().detId<CSCDetId>().chamber()%6 > 2)
	      sub = 1;
	    
	    me1address = id;
	    me1address -= 3;
	    me1address += 3*(sub - 1);
	    me1address = id<<1;//
	    me1address |= trknm-1;
	    
	  }
	  
	  if(station == 2 && id > 3){
	    
	    me2address = id;
	    me2address -= 3;
	    me2address = me2address<<1;
	    me2address |= trknm-1;
	    
	  }
	  
	}
	
      }
      
      // if ( ( mode == 15 && (cHits_in_station[0] != 1 || cHits_in_station[1] != 1 || cHits_in_station[2] != 1 || cHits_in_station[3] != 1) ) ||
      // 	   ( mode == 14 && (cHits_in_station[0] != 1 || cHits_in_station[1] != 1 || cHits_in_station[2] != 1 || cHits_in_station[3] != 0) ) ||
      // 	   ( mode == 13 && (cHits_in_station[0] != 1 || cHits_in_station[1] != 1 || cHits_in_station[2] != 0 || cHits_in_station[3] != 1) ) ||
      // 	   ( mode == 11 && (cHits_in_station[0] != 1 || cHits_in_station[1] != 0 || cHits_in_station[2] != 1 || cHits_in_station[3] != 1) ) ||
      // 	   ( mode ==  7 && (cHits_in_station[0] != 0 || cHits_in_station[1] != 1 || cHits_in_station[2] != 1 || cHits_in_station[3] != 1) ) )
      // 	std::cout << "Mode " << mode << " track has " << cHits_in_station[0] << " / " << cHits_in_station[1] << " / "
      // 		  << cHits_in_station[2] << " / " << cHits_in_station[3] << " converted hits in stations 1 / 2 / 3 / 4" << std::endl;

      if ( ( mode == 15 && (eHits_in_station[0] != 1 || eHits_in_station[1] != 1 || eHits_in_station[2] != 1 || eHits_in_station[3] != 1) ) ||
	   ( mode == 14 && (eHits_in_station[0] != 1 || eHits_in_station[1] != 1 || eHits_in_station[2] != 1 || eHits_in_station[3] != 0) ) ||
	   ( mode == 13 && (eHits_in_station[0] != 1 || eHits_in_station[1] != 1 || eHits_in_station[2] != 0 || eHits_in_station[3] != 1) ) ||
	   ( mode == 11 && (eHits_in_station[0] != 1 || eHits_in_station[1] != 0 || eHits_in_station[2] != 1 || eHits_in_station[3] != 1) ) ||
	   ( mode ==  7 && (eHits_in_station[0] != 0 || eHits_in_station[1] != 1 || eHits_in_station[2] != 1 || eHits_in_station[3] != 1) ) )
	LogTrace("L1TMuonEndCapTrackProducer") << "Mode " << mode << " track has " << eHits_in_station[0] << " / " << eHits_in_station[1] << " / "
		  << eHits_in_station[2] << " / " << eHits_in_station[3] << " EMTF hits in stations 1 / 2 / 3 / 4" << std::endl;
      
      tempTrack.phis = ps;
      tempTrack.thetas = ts;
      tempTrack.deltas = AllTracks[fbest].deltas;
      
      // // Before Mulhearn cleanup, May 11
      // unsigned long xmlpt_address = 0;
      // float xmlpt = CalculatePt(tempTrack, es, mode, &xmlpt_address);
      
      // After Mulhearn cleanup, May 11
      unsigned long xmlpt_address = ptAssignment_.calculateAddress(tempTrack, es, mode);
      float xmlpt = ptAssignment_.calculatePt(xmlpt_address);
      
      tempTrack.pt = xmlpt*1.4;
      //FoundTracks->push_back(tempTrack);
      
      CombAddress = (me2address<<4) | me1address;
      
      int charge = getCharge(phis[0],phis[1],phis[2],phis[3],mode);
      
      l1t::RegionalMuonCand outCand = MakeRegionalCand(xmlpt*1.4,AllTracks[fbest].phi,AllTracks[fbest].theta,
						       charge,mode,CombAddress,sector);
      
      float theta_angle = l1t::calc_theta_rad_from_int( AllTracks[fbest].theta ); 
      float eta = l1t::calc_eta_from_theta_rad( theta_angle );
      
      thisTrack.set_phi_loc_deg  ( l1t::calc_phi_loc_deg( thisTrack.Phi_loc_int() ) );
      thisTrack.set_phi_loc_rad  ( l1t::calc_phi_loc_rad( thisTrack.Phi_loc_int() ) );
      thisTrack.set_phi_glob_deg ( l1t::calc_phi_glob_deg( thisTrack.Phi_loc_deg(), thisTrack.Sector() ) );
      thisTrack.set_phi_glob_rad ( l1t::calc_phi_glob_rad( thisTrack.Phi_loc_rad(), thisTrack.Sector() ) );
      thisTrack.set_quality    ( outCand.hwQual());
      thisTrack.set_mode       ( mode            ); 
      thisTrack.set_first_bx   ( ebx - 6         ); 
      thisTrack.set_second_bx  ( sebx - 6        ); 
      thisTrack.set_bx         ( thisTrack.Second_BX() );
      thisTrack.set_phis       ( ps              );
      thisTrack.set_thetas     ( ts              );
      thisTrack.set_pt         ( xmlpt*1.4       );
      thisTrack.set_pt_XML     ( xmlpt           );
      thisTrack.set_pt_LUT_addr( xmlpt_address   );
      thisTrack.set_charge     ( (charge == 1) ? -1 : 1 ); // uGMT uses opposite of physical charge (to match pdgID)
      thisTrack.set_charge_GMT ( charge          );
      thisTrack.set_theta_rad  ( theta_angle     );
      thisTrack.set_theta_deg  ( theta_angle * 180/Geom::pi() );
      thisTrack.set_eta        ( eta  * thisTrack.Endcap() );
      thisTrack.set_pt_GMT     ( outCand.hwPt()  ); 
      thisTrack.set_phi_GMT    ( outCand.hwPhi() );
      thisTrack.set_eta_GMT    ( outCand.hwEta() );
      
      thisTrack.ImportPtLUT    ( thisTrack.Mode(), thisTrack.Pt_LUT_addr() );
      
      // thisTrack.phi_loc_rad(); // Need to implement - AWB 04.04.16
      // thisTrack.phi_glob_rad(); // Need to implement - AWB 04.04.16
      
      // Use "ebx" for earliest LCT (used through August 24, 2016) - AWB 26.08.16
      std::pair<int,l1t::RegionalMuonCand> outPair(sebx,outCand);
      
      // // Extra debugging output - AWB 29.03.16
      // std::cout << "Input: eBX = " << ebx << ", seBX = " << sebx << ", pt = " << xmlpt*1.4 
      // 	    << ", phi = " << AllTracks[fbest].phi << ", eta = " << eta 
      // 	    << ", theta = " << AllTracks[fbest].theta << ", sign = " << 1 
      // 	    << ", quality = " << mode << ", trackaddress = " << 1 
      // 	    << ", sector = " << sector << std::endl;
      // std::cout << "Output: BX = " << sebx << ", hwPt = " << outCand.hwPt() << ", hwPhi = " << outCand.hwPhi() 
      // 	    << ", hwEta = " << outCand.hwEta() << ", hwSign = " << outCand.hwSign() 
      // 	    << ", hwQual = " << outCand.hwQual() << ", link = " << outCand.link()
      // 	    << ", processor = " << outCand.processor() 
      // 	    << ", trackFinderType = " << outCand.trackFinderType() << std::endl;
      holder.push_back(outPair);
      thisTrack.set_isGMT( 1 );

      OutputTracks->push_back( thisTrack );
      OutTracks->push_back( thisTrack.CreateEMTFTrack() );
    }
  }

  OutputCands->setBXRange(-2,2);
  
  for(int sect=0;sect<12;sect++){
    
    for(unsigned int h=0;h<holder.size();h++){
      
      int bx = holder[h].first - 6;
      int sector = holder[h].second.processor();
      if(holder[h].second.trackFinderType() == 3)
	sector += 6;
      
      if(sector == sect){
	OutputCands->push_back(bx,holder[h].second);
      }
      
    }
  }
  
  //ev.put(std::move(FoundTracks), "DataITC");
  ev.put(std::move(OutputCands), "EMTF");
  ev.put(std::move(OutHits), "");      // EMTFHitCollection
  ev.put(std::move(OutTracks), "");    // EMTFTrackCollection
  ev.put(std::move(OutputHits), "CSC");   // EMTFHitExtraCollection
  ev.put(std::move(OutputHitsRPC), "RPC");   // EMTFHitExtraCollection
  ev.put(std::move(OutputTracks), ""); // EMTFTrackExtraCollection
  
}//analyzer

void L1TMuonEndCapTrackProducer::beginJob()
{
}
void L1TMuonEndCapTrackProducer::endJob()
{
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TMuonEndCapTrackProducer);
