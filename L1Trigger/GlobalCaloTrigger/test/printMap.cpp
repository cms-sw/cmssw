
#include <iostream>
#include <fstream>
#include <iomanip>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"


using std::ofstream;
using std::setw;
using std::endl;
using std::cout;

// print the region map and region-associated data (rct/gct card/input numbers)

void makePlot(ofstream& of, int plot);

int main() {


  /* === obsolete ===
  // GCT card number
  ofstream gctCardPlot("gctCardPlot.txt");
  makePlot(gctCardPlot, 0);
     ================= */

  /* === obsolete ===
  // GCT region number
  ofstream gctRgnPlot("gctRgnPlot.txt");
  makePlot(gctRgnPlot, 1);
     ================= */

  // RCT crate number
  ofstream rctCratePlot("rctCratePlot.txt");
  makePlot(rctCratePlot, 2);

  // RCT card number
  ofstream rctCardPlot("rctCardPlot.txt");
  makePlot(rctCardPlot, 3);

  // RCT region number
  ofstream rctRgnPlot("rctRgnPlot.txt");
  makePlot(rctRgnPlot, 4);

  // RCT phi
  ofstream rctPhiPlot("rctPhiPlot.txt");
  makePlot(rctPhiPlot, 5);

  // RCT eta
  ofstream rctEtaPlot("rctEtaPlot.txt");
  makePlot(rctEtaPlot, 6);
  
  // forward
  ofstream fwdRgnPlot("fwdRgnPlot.txt");
  makePlot(fwdRgnPlot, 7);


  // test RCT constructors
  for (unsigned crate=0; crate<18; crate++) {
    for (unsigned card=0; card<7; card++) {
      for (unsigned rgn=0; rgn<2; rgn++) {
	L1CaloRegionDetId r(crate, card, rgn);
	if ( (r.rctCrate()!=crate) || (r.rctCard()!=card) || (r.rctRegion()!=rgn) ) {
	  cout << "Error! : RCT crate " << crate << " card " << card << " region " << rgn << endl;
	}
      }
    }
  }

  // test HF constructors
  for (unsigned crate=0; crate<18; crate++) {
    for (unsigned rgn=0; rgn<8; rgn++) {
      L1CaloRegionDetId r(crate, 999, rgn);
      if ( (r.rctCrate()!=crate) || (r.rctRegion()!=rgn) ) {
	cout << "Error! : RCT crate " << crate << " HF region " << rgn << endl;
      }
    }
  }


  return 0;

}


void makePlot(ofstream &of, int plot) {

  // print header line
  of << "     ieta->" << endl;
  of << "    : ";
  for (int ieta=0; ieta<22; ieta++) {
    of << setw(3) << ieta << " ";
  }
  of << endl;

  // main loop
  for (int iphi=0; iphi<18; iphi++) {
  
    of << setw(3) << iphi << "  : ";
    
    for (int ieta=0; ieta<22; ieta++) {    

      L1CaloRegionDetId rgn(ieta, iphi);

      switch (plot) {
      /* === obsolete ===
      case 0 : 
	of << setw(3) << rgn.gctCard() << " "; break;
         ================ */
      /* === obsolete ===
      case 1 : 
	of << setw(3) << rgn.gctRegion() << " "; break;
         ================ */
      case 2 : 
	of << setw(3) << rgn.rctCrate() << " "; break;
      case 3 : 
	of << setw(3) << rgn.rctCard() << " "; break;
      case 4 : 
	of << setw(3) << rgn.rctRegion() << " "; break;
      case 5 : 
	of << setw(3) << rgn.rctPhi() << " "; break;
      case 6 : 
	of << setw(3) << rgn.rctEta() << " "; break;
      case 7 : 
	of << setw(3) << (rgn.isHf()?1:0) << " "; break;


      default :
	break;
      }

    }

    of << endl;
  }

}

