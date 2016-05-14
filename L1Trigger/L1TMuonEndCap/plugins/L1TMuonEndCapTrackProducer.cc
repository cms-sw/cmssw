///////////////////////////////////////////////////////////////
// Upgraded Encdap Muon Track Finding Algorithm		    	//
//							   								//
// Info: A human-readable version of the firmware based     //
//       track finding algorithm which will be implemented  //
//       in the upgraded endcaps of CMS. DT and RPC inputs  //
//	     are not considered in this algorithm.      		//
//								   							//
// Author: M. Carver (UF)				    				//
//////////////////////////////////////////////////////////////

#define NUM_SECTORS 12

#include "L1Trigger/L1TMuonEndCap/plugins/L1TMuonEndCapTrackProducer.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCPatternLUT.h"
#include "L1Trigger/CSCTrackFinder/test/src/RefTrack.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveConverter_Neighbor.h"
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
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"

using namespace L1TMuon;


L1TMuonEndCapTrackProducer::L1TMuonEndCapTrackProducer(const PSet& p) {

  inputTokenCSC = consumes<CSCCorrelatedLCTDigiCollection>(p.getParameter<edm::InputTag>("CSCInput"));
  
  produces<l1t::RegionalMuonCandBxCollection >("EMTF");
  produces< l1t::EMTFTrackCollection >("EMTF");
  produces< l1t::EMTFHitCollection >("EMTF");  
}


void L1TMuonEndCapTrackProducer::produce(edm::Event& ev,
			       const edm::EventSetup& es) {
				   

  //bool verbose = false;

  //fprintf (write,"12345\n"); //<-- part of printing text file to send verilog code, not needed if George's package is included


  //std::auto_ptr<L1TMuon::InternalTrackCollection> FoundTracks (new L1TMuon::InternalTrackCollection);
  std::auto_ptr<l1t::RegionalMuonCandBxCollection > OutputCands (new l1t::RegionalMuonCandBxCollection);
  std::auto_ptr<l1t::EMTFTrackCollection> OutputTracks (new l1t::EMTFTrackCollection);
  std::auto_ptr<l1t::EMTFHitCollection> OutputHits (new l1t::EMTFHitCollection);

  std::vector<BTrack> PTracks[NUM_SECTORS];

  std::vector<TriggerPrimitive> tester;
  //std::vector<InternalTrack> FoundTracks;
  
  //////////////////////////////////////////////
  ///////// Make Trigger Primitives ////////////
  //////////////////////////////////////////////
  edm::Handle<CSCCorrelatedLCTDigiCollection> MDC;
  ev.getByToken(inputTokenCSC, MDC);
  std::vector<TriggerPrimitive> out;
  
  auto chamber = MDC->begin();
  auto chend  = MDC->end();
  for( ; chamber != chend; ++chamber ) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for( ; digi != dend; ++digi ) {
      out.push_back(TriggerPrimitive((*chamber).first,*digi));
      l1t::EMTFHit thisHit;
      thisHit.ImportCSCDetId( (*chamber).first );
      thisHit.ImportCSCCorrelatedLCTDigi( *digi );
      if (thisHit.Station() == 1 && thisHit.Ring() == 1 && thisHit.Strip() > 127) thisHit.set_ring(4);
      OutputHits->push_back( thisHit );
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
    auto tp = out.cbegin();
    auto tpend = out.cend();

    for( ; tp != tpend; ++tp ) {
      if(tp->subsystem() == 1)
      {
		//TriggerPrimitiveRef tpref(out,tp - out.cbegin());

		tester.push_back(*tp);

		//std::cout<<"\ntrigger prim found station:"<<tp->detId<CSCDetId>().station()<<std::endl;
      }

     }
   //}
  std::vector<ConvertedHit> CHits[NUM_SECTORS];
  MatchingOutput MO[NUM_SECTORS];

for(int SectIndex=0;SectIndex<NUM_SECTORS;SectIndex++){//perform TF on all 12 sectors



  //////////////////////////////////////////////////////  Input is raw hit information from
  ///////////////// TP Conversion //////////////////////  Output is vector of Converted Hits
  //////////////////////////////////////////////////////


 	std::vector<ConvertedHit> ConvHits = PrimConv(tester,SectIndex);
	CHits[SectIndex] = ConvHits;

	// Fill OutputHits with ConvertedHit information
	for (uint iCHit = 0; iCHit < ConvHits.size(); iCHit++) {
	  // bool isMatched = false;
	  for (uint iHit = 0; iHit < OutputHits->size(); iHit++) {
	    if ( ConvHits.at(iCHit).Station() == OutputHits->at(iHit).Station() &&
		 ( ConvHits.at(iCHit).Id()    == OutputHits->at(iHit).CSC_ID()  ||
		   ConvHits.at(iCHit).Id()    == ( (OutputHits->at(iHit).Ring() != 4) // Account for either ME1/1a 
						   ? OutputHits->at(iHit).CSC_ID()    // CSC ID numbering convention
						   : OutputHits->at(iHit).CSC_ID() + 9 ) ) &&
		 ConvHits.at(iCHit).Wire()    == OutputHits->at(iHit).Wire()    &&
		 ConvHits.at(iCHit).Strip()   == OutputHits->at(iHit).Strip()   &&
		 ConvHits.at(iCHit).BX() - 6  == OutputHits->at(iHit).BX() ) {
	      // isMatched = true;
	      OutputHits->at(iHit).set_zone_hit    ( ConvHits.at(iCHit).Zhit()   );
	      OutputHits->at(iHit).set_phi_hit     ( ConvHits.at(iCHit).Ph_hit() );
	      OutputHits->at(iHit).set_phi_z_val   ( ConvHits.at(iCHit).Phzvl()  );
	      OutputHits->at(iHit).set_phi_loc_int ( ConvHits.at(iCHit).Phi()    );
	      OutputHits->at(iHit).set_theta_int   ( ConvHits.at(iCHit).Theta()  );

	      OutputHits->at(iHit).SetZoneContribution ( ConvHits.at(iCHit).ZoneContribution() );
	      OutputHits->at(iHit).set_phi_loc_deg  ( GetPackedPhi( OutputHits->at(iHit).Phi_loc_int() ) );
	      OutputHits->at(iHit).set_phi_loc_rad  ( OutputHits->at(iHit).Phi_loc_deg() * 3.14159/180 );
	      OutputHits->at(iHit).set_phi_glob_deg ( OutputHits->at(iHit).Phi_loc_deg() + 15 + (OutputHits->at(iHit).Sector() - 1)*60 );
	      OutputHits->at(iHit).set_phi_glob_rad ( OutputHits->at(iHit).Phi_glob_deg() * 3.14159/180 );
	      OutputHits->at(iHit).set_theta_deg    ( OutputHits->at(iHit).calc_theta_deg( OutputHits->at(iHit).Theta_int() ) );
	      OutputHits->at(iHit).set_theta_rad    ( OutputHits->at(iHit).calc_theta_rad( OutputHits->at(iHit).Theta_int() ) );
	      OutputHits->at(iHit).set_eta( OutputHits->at(iHit).calc_eta( OutputHits->at(iHit).Theta_rad() ) * OutputHits->at(iHit).Endcap() );
	    }
	  } // End loop: for (uint iHit = 0; iHit < OutputHits->size(); iHit++)

	  // if (isMatched == false) {
	  //   std::cout << "***********************************************" << std::endl;
	  //   std::cout << "Unmatched ConvHit in event " << ev.id().event() << ", SectIndex " << SectIndex << std::endl;
	  //   std::cout << "ConvHit: station = " << ConvHits.at(iCHit).Station() << ", CSC ID = " << ConvHits.at(iCHit).Id()
	  // 	      << ", wire = " << ConvHits.at(iCHit).Wire() << ", strip = " << ConvHits.at(iCHit).Strip()
	  // 	      << ", BX = " << ConvHits.at(iCHit).BX() << std::endl;
	    
	  //   for (uint iHit = 0; iHit < OutputHits->size(); iHit++) {
	  //     std::cout << "EMTFHit: station = " << OutputHits->at(iHit).Station() << ", CSC ID = " << OutputHits->at(iHit).CSC_ID()
	  // 		<< ", wire = " << OutputHits->at(iHit).Wire() << ", strip = " << OutputHits->at(iHit).Strip()
	  // 		<< ", BX = " << OutputHits->at(iHit).BX() << ", ring = " << OutputHits->at(iHit).Ring() 
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

  PatternOutput Test = DeleteDuplicatePatterns(Pout);

  //PrintQuality(Test.detected);


  ///////////////////////////////
  //////Sector Sorting/////////// Sorts through the patterns found in each zone and selects the best three per zone to send to the next module.
  ///////Finding 3 Best Pattern//
  ///////////////////////////////

  SortingOutput Sout = SortSect(Test);


  //////////////////////////////////
  ///////// Match ph patterns ////// Loops over each sorted pattern and then loops over all possible triggerprimitives which could have made the pattern
  ////// to segment inputs ///////// and matches the associated full precision triggerprimitives to the detected pattern.
  //////////////////////////////////

  MatchingOutput Mout = PhiMatching(Sout);
  MO[SectIndex] = Mout;

  /////////////////////////////////
  //////// Calculate delta //////// Once we have matched the hits we calculate the delta phi and theta between all
  ////////    ph and th    //////// stations present.
  /////////////////////////////////

 std::vector<std::vector<DeltaOutput>> Dout = CalcDeltas(Mout);////


  /////////////////////////////////
  /////// Sorts and gives /////////  Loops over all of the found tracks(looking across zones) and selects the best three per sector.
  ////// Best 3 tracks/sector /////  Here ghost busting is done to delete tracks which are comprised of the same associated stubs.
  /////////////////////////////////

  std::vector<BTrack> Bout = BestTracks(Dout);
   PTracks[SectIndex] = Bout;

   
 } // End loop: for(int SectIndex=0;SectIndex<NUM_SECTORS;SectIndex++)


 ///////////////////////////////////////
 /// Collect Muons from all sectors ////
 ///////////////////////////////////////

 std::vector<BTrack> PTemp[NUM_SECTORS];
 std::vector<BTrack> AllTracks;
 for (int i=0; i<NUM_SECTORS; i++) PTemp[i] = PTracks[i];


 	for(int j=0;j<36;j++){


			if(PTemp[j/3][j%3].phi)//no track
				AllTracks.push_back(PTemp[j/3][j%3]);

		

 	}

  ///////////////////////////////////
  /// Make Internal track if ////////
  /////// tracks are found //////////
  /////////////////////////////////// 
  
  std::vector<l1t::RegionalMuonCand> tester1;
  std::vector<std::pair<int,l1t::RegionalMuonCand>> holder;

  for(unsigned int fbest=0;fbest<AllTracks.size();fbest++){
  

  	if(AllTracks[fbest].phi){


		InternalTrack tempTrack;
  		tempTrack.setType(2);
		tempTrack.phi = AllTracks[fbest].phi;
		tempTrack.theta = AllTracks[fbest].theta;
		tempTrack.rank = AllTracks[fbest].winner.Rank();
		tempTrack.deltas = AllTracks[fbest].deltas;
		std::vector<int> ps, ts;

		l1t::EMTFTrack thisTrack;
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


		int sector = -1;
		bool ME13 = false;
		int me1address = 0, me2address = 0, CombAddress = 0, mode_uncorr = 0;
		int ebx = 20, sebx = 20;
		int phis[4] = {-99,-99,-99,-99};

		for(std::vector<ConvertedHit>::iterator A = AllTracks[fbest].AHits.begin();A != AllTracks[fbest].AHits.end();A++){

			if(A->Phi() != -999){
			  
			        l1t::EMTFHit thisHit;
			        // thisHit.ImportCSCDetId( A->TP().detId<CSCDetId>() );

				for (uint iHit = 0; iHit < OutputHits->size(); iHit++) {
				  if ( A->TP().detId<CSCDetId>().station() == OutputHits->at(iHit).Station() and
				       A->TP().getCSCData().cscID          == OutputHits->at(iHit).CSC_ID()  and
				       A->Wire()                           == OutputHits->at(iHit).Wire()    and
				       A->Strip()                          == OutputHits->at(iHit).Strip()   and
				       A->TP().getCSCData().bx - 6         == OutputHits->at(iHit).BX() ) {
				    thisHit = OutputHits->at(iHit);
				    thisTrack.push_HitIndex(iHit);
				  }
				}
				thisTrack.set_endcap       ( thisHit.Endcap()     );
				thisTrack.set_sector       ( thisHit.Sector()     );
				thisTrack.set_sector_GMT   ( thisHit.Sector_GMT() );

				int station = A->TP().detId<CSCDetId>().station();
				int id = A->TP().getCSCData().cscID;
				int trknm = A->TP().getCSCData().trknmb;//A->TP().getCSCData().bx
				
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
				
				sector = A->SectorIndex();//(A->TP().detId<CSCDetId>().endcap() -1)*6 + A->TP().detId<CSCDetId>().triggerSector() - 1;
				//std::cout<<"Q: "<<A->Quality()<<", keywire: "<<A->Wire()<<", strip: "<<A->Strip()<<std::endl;

				switch(station){
					case 1: mode_uncorr |= 8;break;
					case 2: mode_uncorr |= 4;break;
					case 3: mode_uncorr |= 2;break;
					case 4: mode_uncorr |= 1;break;
					default: mode_uncorr |= 0;
				}


				if(A->TP().detId<CSCDetId>().station() == 1 && A->TP().detId<CSCDetId>().ring() == 3)
					ME13 = true;

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
		
		
		int mode = 0;
		if(tempTrack.rank & 32)
			mode |= 8;
		if(tempTrack.rank & 8)
			mode |= 4;
		if(tempTrack.rank & 2)
			mode |= 2;
		if(tempTrack.rank & 1)
			mode |= 1;

		tempTrack.phis = ps;
		tempTrack.thetas = ts;

		unsigned long xmlpt_address = 0;
		
		float xmlpt = CalculatePt(tempTrack, es, mode, &xmlpt_address);
		tempTrack.pt = xmlpt*1.4;
		//FoundTracks->push_back(tempTrack);

		CombAddress = (me2address<<4) | me1address;

		int charge = getCharge(phis[0],phis[1],phis[2],phis[3],mode);
		
		

		l1t::RegionalMuonCand outCand = MakeRegionalCand(xmlpt*1.4,AllTracks[fbest].phi,AllTracks[fbest].theta,
								 charge,mode,CombAddress,sector);
        // NOTE: assuming that all candidates come from the central BX:
        //int bx = 0;
		float theta_angle = (AllTracks[fbest].theta*0.2851562 + 8.5)*(3.14159265359/180);
		float eta = (-1)*log(tan(theta_angle/2));


		thisTrack.set_phi_loc_deg  ( (thisTrack.Phi_loc_int() / 60.0) - 2.0 );
		thisTrack.set_phi_loc_rad  ( thisTrack.Phi_loc_deg() * 3.14159/180 );
		thisTrack.set_phi_glob_deg ( thisTrack.Phi_loc_deg() + 15 + (thisTrack.Sector() - 1)*60 );
		thisTrack.set_phi_glob_rad ( thisTrack.Phi_glob_deg() * 3.14159/180 );
		thisTrack.set_quality    ( outCand.hwQual());
		thisTrack.set_mode       ( mode            ); 
		thisTrack.set_first_bx   ( ebx             ); 
		thisTrack.set_second_bx  ( sebx            ); 
		thisTrack.set_phis       ( ps              );
		thisTrack.set_thetas     ( ts              );
		thisTrack.set_pt         ( xmlpt*1.4       );
		thisTrack.set_pt_XML     ( xmlpt           );
		thisTrack.set_pt_LUT     ( xmlpt_address   );
		thisTrack.set_charge     ( (charge == 1) ? 1 : -1 );
		thisTrack.set_charge_GMT ( charge          );
		thisTrack.set_theta_rad  ( theta_angle     );
		thisTrack.set_theta_deg  ( theta_angle * 180/3.14159265359 );
		thisTrack.set_eta        ( eta  * thisTrack.Endcap() );
		thisTrack.set_pt_GMT     ( outCand.hwPt()  ); 
		thisTrack.set_phi_GMT    ( outCand.hwPhi() );
		thisTrack.set_eta_GMT    ( outCand.hwEta() );

		thisTrack.Import_pT_LUT  ( thisTrack.Mode(), thisTrack.Pt_LUT() );

		// thisTrack.phi_loc_rad(); // Need to implement - AWB 04.04.16
		// thisTrack.phi_glob_rad(); // Need to implement - AWB 04.04.16

 		// // Optimal emulator configuration - AWB 29.03.16
		// std::pair<int,l1t::RegionalMuonCand> outPair(sebx,outCand);

		// Actual setting in firmware - AWB 12.04.16
		std::pair<int,l1t::RegionalMuonCand> outPair(ebx,outCand);

		if(!ME13 && fabs(eta) > 1.1) {
		  // // Extra debugging output - AWB 29.03.16
		  // std::cout << "\n\nInput: eBX = " << ebx << ", seBX = " << sebx << ", pt = " << xmlpt*1.4 
		  //  	<< ", phi = " << AllTracks[fbest].phi << ", eta = " << eta 
		 //   	<< ", theta = " << AllTracks[fbest].theta << ", sign = " << 1 
		  //  	<< ", quality = " << mode << ", trackaddress = " << 1 
		  //  	<< ", sector = " << sector << std::endl;
		 //  std::cout << "Output: BX = " << ebx << ", hwPt = " << outCand.hwPt() << ", hwPhi = " << outCand.hwPhi() 
		 //   	<< ", hwEta = " << outCand.hwEta() << ", hwSign = " << outCand.hwSign() 
		 //   	<< ", hwQual = " << outCand.hwQual() << ", link = " << outCand.link()
		 //   	<< ", processor = " << outCand.processor() 
		  //  	<< ", trackFinderType = " << outCand.trackFinderType() << std::endl;
			holder.push_back(outPair);
			thisTrack.set_isGMT( 1 );
			
		}
		OutputTracks->push_back( thisTrack );
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


//ev.put( FoundTracks, "DataITC");
ev.put( OutputCands, "EMTF");
 ev.put( OutputHits, "EMTF"); 
 ev.put( OutputTracks, "EMTF");
  //std::cout<<"End Upgraded Track Finder Prducer:::::::::::::::::::::::::::\n:::::::::::::::::::::::::::::::::::::::::::::::::\n\n";

}//analyzer

void L1TMuonEndCapTrackProducer::beginJob()
{

	

}
void L1TMuonEndCapTrackProducer::endJob()
{

}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TMuonEndCapTrackProducer);
