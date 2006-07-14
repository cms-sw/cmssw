
#include <iostream.h>
#include <fstream.h>
#include <iomanip.h>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

// print the region map and regioin-associated data (rct/gct card/input numbers)

void makePlot(ofstream& of, int plot);

int main() {


  // GCT card number
  ofstream gctCardPlot("gctCardPlot.txt", ios::out);
  makePlot(gctCardPlot, 0);

  // GCT region number
  ofstream gctRgnPlot("gctRgnPlot.txt", ios::out);
  makePlot(gctRgnPlot, 0);

  // RCT crate number
  ofstream rctCratePlot("rctCratePlot.txt", ios::out);
  makePlot(rctCratePlot, 2);

  // RCT card number
  ofstream rctCardPlot("rctCardPlot.txt", ios::out);
  makePlot(rctCardPlot, 3);

  // RCT region number
  ofstream rctRgnPlot("rctRgnPlot.txt", ios::out);
  makePlot(rctRgnPlot, 4);

  // RCT phi
  ofstream rctPhiPlot("rctPhiPlot.txt", ios::out);
  makePlot(rctPhiPlot, 5);

  // RCT eta
  ofstream rctEtaPlot("rctEtaPlot.txt", ios::out);
  makePlot(rctEtaPlot, 6);
  
  // forward
  ofstream fwdRgnPlot("fwdRgnPlot.txt", ios::out);
  makePlot(fwdRgnPlot, 7);



  return 0;

};


void makePlot(ofstream &of, int plot) {

  // print header line
  of << "     iphi->" << endl;
  of << "    : ";
  for (int ieta=0; ieta<22; ieta++) {
    of << setw(2) << ieta << " ";
  }
  of << endl;

  // main loop
  for (int iphi=0; iphi<18; iphi++) {
  
    of << setw(2) << iphi << "  : ";
    
    for (int ieta=0; ieta<22; ieta++) {    

      L1CaloRegionDetId rgn(ieta, iphi);

      switch (plot) {
      case 0 : 
	of << setw(2) << rgn.gctCard() << " "; break;
      case 1 : 
	of << setw(2) << rgn.gctRegion() << " "; break;
      case 2 : 
	of << setw(2) << rgn.rctCrate() << " "; break;
      case 3 : 
	of << setw(2) << rgn.rctCard() << " "; break;
      case 4 : 
	of << setw(2) << rgn.rctRegion() << " "; break;
      case 5 : 
	of << setw(2) << rgn.rctPhi() << " "; break;
      case 6 : 
	of << setw(2) << rgn.rctEta() << " "; break;
      case 7 : 
	of << setw(2) << (rgn.isForward()?1:0) << " "; break;


      deafult :
	break;
      }

    }

    of << endl;
  }

}
