/*
 *  See header file for a description of this class.
 *
 *
 *  $Date: 2007/07/06 11:59:46 $
 *  $Revision: 1.1 $
 *  \author M. Maggi, C. Viviani, D. Pagano - University of Pavia & INFN Pavia
 *
 */

#include "L1Trigger/RPCTechnicalTrigger/interface/RBCEmulator.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCOutputSignalContainer.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCOutputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCPolicy.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCPatternLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCChamberORLogic.h"

using namespace std;

RBCEmulator::RBCEmulator(const Event & event, const EventSetup& eventSetup): l(0){
  
  // Access to simhits and rechits
  cout << endl <<"--- [RPCTechnicalTrigger] Analysing Event: #Run: " << event.id().run()	    << " #Event: " << event.id().event() << std::endl;
  
  
  // Get the RPC Geometry
  ESHandle<RPCGeometry> rpcGeom;
  eventSetup.get<MuonGeometryRecord>().get(rpcGeom);
  
  // Get the digi collection from the event
  Handle<RPCDigiCollection> digi;
  event.getByLabel("muonRPCDigis", digi);
  
  int trig[5][12][3];
  int neitrig[5][12][3];
  int wtrig;
  
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 12; j++) {
      for (int k = 0; k < 3; k++) {
	trig[i][j][k] = 0;
	neitrig[i][j][k] = 0;
      }
    }
  }
  wtrig = 0;
  
  RPCDigiCollection::DigiRangeIterator detUnitIt;
  RPCDigiCollection::const_iterator digiItr; 
  
  for (detUnitIt=digi->begin(); detUnitIt!=digi->end(); ++detUnitIt) {
    
    
    digiItr = (*detUnitIt ).second.first;
    
    
    int bx = (*digiItr).bx();
    
    const RPCDetId& id = (*detUnitIt).first;
    const RPCRoll* roll = dynamic_cast<const RPCRoll* >( rpcGeom->roll(id));
    if((roll->isForward())) return;
    //const RPCDigiCollection::Range& range = (*detUnitIt).second;
    //counter = counter + 1;
    int wheel = roll->id().ring() + 2;
    int sector = roll->id().sector() - 1;
    
    trig[wheel][sector][bx] = trig[wheel][sector][bx] + 1;
    
  }
  //loop over wheel, sector and bx to look for events in neighbour wheels

  int b,c;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 12; j++) {
      for (int k = 0; k < 3; k++) {
	b = j + 1;
	c = j + 2;
	if (b == 12) b = 0;
	if (c == 12) c = 0;
	if (c == 13) c = 1;
	//      cout << "j = " << j +1 << endl;
	if (trig[i][j][k] > 0) {
	  
	  if (i == 0) wtrig = trig[i+1][j][k];
	  if (i == 1) wtrig = trig[i-1][j][k] + trig[i+1][j][k];
	  if (i == 2) wtrig = trig[i-1][j][k] + trig[i+1][j][k];
	  if (i == 3) wtrig = trig[i-1][j][k] + trig[i+1][j][k];
	  if (i == 4) wtrig = trig[i-1][j][k];
	  
	  neitrig[i][b][k] = trig[i][j][k] + trig[i][j+1][k] + trig [i][j+2][k] + wtrig;
	  j = j + 2;
	}	
      }
    }
    
  }
  
  //checking policy: poly = 1 one sector trigger
  //                 poly = 0 adjacent sectors trigger
  //check also bx

   if (poly == 1) {
     cout << "ChamberOR" <<endl;
     for (int i = 0; i < 5; i++) {
       for (int j = 0; j < 12; j++) {
	 if (BX == 1) trig[i][j][0] = trig[i][j][0] + trig[i][j][1];
	 if (BX == 2) trig[i][j][0] = trig[i][j][0] + trig[i][j][1] + trig[i][j][2];
	 if (BX == 3) trig[i][j][0] = trig[i][j][0] + trig[i][j][1] + trig[i][j][2] + trig[i][j][3];
	 //if (trig[i][j] > 0) cout << "(W, S): " << i - 2 << " " << j + 1 << " = " << trig[i][j] << endl;

	 //majority condition 
	 if (trig[i][j][0] >= majority) cout << "  ---> Trigger event." << endl;
       }      
     }
   }
  
   
   if (poly == 0) {
    cout << endl << "Neighbours" << endl;
    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 12; j++) {
	if (BX == 1) neitrig[i][j][0] = neitrig[i][j][0] + neitrig[i][j][1];
	if (BX == 2) neitrig[i][j][0] = neitrig[i][j][0] + neitrig[i][j][1] + neitrig[i][j][2];
	if (BX == 3) neitrig[i][j][0] = neitrig[i][j][0] + neitrig[i][j][1] + neitrig[i][j][2] + neitrig[i][j][3];
	// if (neitrig[i][j] > 0) cout << "(W, S): " << i - 2 << " " << j << " = " << neitrig[i][j] << endl;
	//majority condition
	if (neitrig[i][j][0] >= majority) cout << "  ---> Trigger event." << endl;
      } 
    }
    
  }  
   
}

RBCEmulator::
~RBCEmulator()
{
}

void RBCEmulator::
emulate(RBCPolicy* policy)
{
  l=policy->instance();
  std::cout <<"Setting the logic "<<policy->message()<<std::endl;
}

RBCOutputSignalContainer
RBCEmulator::triggers()
{
  RBCOutputSignalContainer c;
  l->action();
  return c;
}


