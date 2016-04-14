#include <iostream>
#include <iomanip>
#include <vector>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

using namespace std;

#include "L1Trigger/L1TCaloLayer1/src/UCTLayer1.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTCrate.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTCard.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTRegion.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTTower.hh"

#include "L1Trigger/L1TCaloLayer1/src/UCTGeometry.hh"

double flatRandom(double min, double max) {
  static double rMax = (double) 0x7FFFFFFF;
  uint32_t r = random();
  double d = (double) r;
  double f = min + ((max - min) * d / rMax);
  if(f < min) f = min;
  if(f > max) f = max;
  return f;
}

double gaussianRandom(double mean, double standardDeviation) {
  return 0;
}

double poissonRandom(double mean) {
  static double oldMean = -1;
  static double g;
  if(mean != oldMean) {
    oldMean = mean;
    if(mean == 0) {
      g = 0;
    }
    else {
      g = exp(-mean);
    }
  }    
  double em = -1;
  double t = 1;
  do {
    em++;
    t *= flatRandom(0., 1.);
  } while(t > g);
  return em;
}

void print(UCTLayer1& uct) {
  vector<UCTCrate*> crates = uct.getCrates();
  for(uint32_t crt = 0; crt < crates.size(); crt++) {
    vector<UCTCard*> cards = crates[crt]->getCards();
    for(uint32_t crd = 0; crd < cards.size(); crd++) {
      vector<UCTRegion*> regions = cards[crd]->getRegions();
      for(uint32_t rgn = 0; rgn < regions.size(); rgn++) {
	if(regions[rgn]->et() > 0) {
	  int hitEta = regions[rgn]->hitCaloEta();
	  int hitPhi = regions[rgn]->hitCaloPhi();
	  vector<UCTTower*> towers = regions[rgn]->getTowers();
	  for(uint32_t twr = 0; twr < towers.size(); twr++) {
	    if(towers[twr]->caloPhi() == hitPhi && towers[twr]->caloEta() == hitEta) {
	      std::cout << "*";
	    }
	    std::cout << towers[twr];
	  }
	  std::cout << regions[rgn];
	}
      }
      std::cout << cards[crd];
    }
    std::cout << crates[crt];
  }
  std::cout << uct;
}

int main(int argc, char** argv) {

  int nEvents = 10000;
  if(argc == 1) std::cout << "Running on " << nEvents << std::endl;
  else if(argc == 2) nEvents = atoi(argv[1]);
  else {std::cout << "Command syntax: testUCTLayer1 [nEvents]" << std::endl; return 1;}

  UCTLayer1 uctLayer1;

  // Event loop for test
  for(int event = 0; event < nEvents; event++) {

    if(!uctLayer1.clearEvent()) {
      std::cerr << "UCT: Failed to clear event" << std::endl;
      exit(1);
    }
    
    // Put a random number of towers in the UCT 

    uint32_t expectedTotalET = 0;
    
    // ECAL TPGs - set a mean of 100 random ECAL towers!
    uint32_t nHitTowers = poissonRandom(100.);
    for(uint32_t i = 0; i < nHitTowers; i++) {
      uint32_t et = (random() & 0xFF); // Random energy up to the maximum allowed
      bool fg = ((random() % 100) < 95); // 5% of the time eleFG Veto should "kill" electron
      int caloEta = ((random()+1) % 28); // Distribute uniformly in +/- eta within acceptance
      while(caloEta < 1 || caloEta > 28) caloEta = ((random()+1) % 28);
      if((random() & 0x1) != 0) caloEta = -caloEta;
      int caloPhi = ((random()+1) % 72); // Distribute uniformly in all phi
      while(caloPhi < 1 || caloPhi > 72) caloPhi = ((random()+1) % 72);
      UCTTowerIndex t = UCTTowerIndex(caloEta, caloPhi);
      if(!uctLayer1.setECALData(t, fg, et)) {
	std::cerr << "UCT: Failed loading an ECAL tower" << std::endl;
	exit(1);
      }
      expectedTotalET += et;
    }

    // HCAL TPGs - set a mean of 100 random HCAL towers!
    nHitTowers = poissonRandom(100.);
    for(uint32_t i = 0; i < nHitTowers; i++) {
      uint32_t et = (random() & 0xFF); // Random energy up to the maximum allowed
      uint32_t fb = (random() & 0x1F); // Set random five bits - this is true emulation!
      int caloEta = ((random()+1) % 28); // Distribute uniformly in +/- eta within acceptance
      while(caloEta < 1 || caloEta > 28) caloEta = ((random()+1) % 28);
      if((random() & 0x1) != 0) caloEta = -caloEta;
      int caloPhi = ((random()+1) % 72); // Distribute uniformly in all phi
      while(caloPhi < 1 || caloPhi > 72) caloPhi = ((random()+1) % 72);
      UCTTowerIndex t = UCTTowerIndex(caloEta, caloPhi);
      if(!uctLayer1.setHCALData(t, fb, et)) {
	std::cerr << "UCT: Failed loading an HCAL tower" << std::endl;
	exit(1);
      }
      expectedTotalET += et;
    }
      
    // Process
    if(!uctLayer1.process()) {
      std::cerr << "UCT: Failed to process layer 1" << std::endl;
      exit(1);
    }

    // Crude check if total ET is approximately OK!
    // We can't expect exact match as there is region level saturation to 10-bits
    // 10% is good enough
    int diff = uctLayer1.et() - expectedTotalET;
    if(diff < 0) diff = diff * -1;
    if(diff > (0.10 * expectedTotalET) ) {
      print(uctLayer1);
      std::cout << "Expected " 
		<< std::showbase << std::internal << std::setfill('0') << std::setw(10) << std::hex
		<< expectedTotalET << std::endl;
    }

  }

  return 0;

}
