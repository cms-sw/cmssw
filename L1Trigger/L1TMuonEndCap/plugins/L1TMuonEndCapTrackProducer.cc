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

#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveConverter.h"
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


using namespace L1TMuon;


L1TMuonEndCapTrackProducer::L1TMuonEndCapTrackProducer(const PSet& p) {

  inputTokenCSC = consumes<CSCCorrelatedLCTDigiCollection>(p.getParameter<edm::InputTag>("CSCInput"));
  
  produces<l1t::RegionalMuonCandBxCollection >("EMTF");
}


void L1TMuonEndCapTrackProducer::produce(edm::Event& ev,
			       const edm::EventSetup& es) {

  //bool verbose = false;


  //std::cout<<"Start Upgraded Track Finder Producer::::: event = "<<ev.id().event()<<"\n\n";

  //fprintf (write,"12345\n"); //<-- part of printing text file to send verilog code, not needed if George's package is included


  //std::auto_ptr<L1TMuon::InternalTrackCollection> FoundTracks (new L1TMuon::InternalTrackCollection);
  std::auto_ptr<l1t::RegionalMuonCandBxCollection > OutputCands (new l1t::RegionalMuonCandBxCollection);

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



  }

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


		int sector = -1;
		bool ME13 = false;
		int me1address = 0, me2address = 0, CombAddress = 0, mode = 0;
		int ebx = 20, sebx = 20;
		int phis[4] = {-99,-99,-99,-99};

		for(std::vector<ConvertedHit>::iterator A = AllTracks[fbest].AHits.begin();A != AllTracks[fbest].AHits.end();A++){

			if(A->Phi() != -999){

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
				sector = (A->TP().detId<CSCDetId>().endcap() -1)*6 + A->TP().detId<CSCDetId>().triggerSector() - 1;
				//std::cout<<"Q: "<<A->Quality()<<", keywire: "<<A->Wire()<<", strip: "<<A->Strip()<<std::endl;

				switch(station){
					case 1: mode |= 8;break;
					case 2: mode |= 4;break;
					case 3: mode |= 2;break;
					case 4: mode |= 1;break;
					default: mode |= 0;
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
		tempTrack.phis = ps;
		tempTrack.thetas = ts;

		float xmlpt = CalculatePt(tempTrack,es);
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
		std::pair<int,l1t::RegionalMuonCand> outPair(sebx,outCand);
		
		if(!ME13 && fabs(eta) > 1.1)
			holder.push_back(outPair);
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
