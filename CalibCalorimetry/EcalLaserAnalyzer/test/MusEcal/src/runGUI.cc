#include "TMath.h"
#include "TRint.h"
#include "TGClient.h"
#include "MusEcalGUI.hh"
#include "../../interface/MEGeom.h"

#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
  int type_   = ME::iLaser;
  int color_  = ME::iBlue; 
  bool useEN_ = false;
  int first_  = MusEcal::firstRun;
  int last_   = MusEcal::lastRun;
  bool debug_ = false;

  int c;
  while ( (c = getopt( argc, argv, "t:c:f:l:ed" ) ) != EOF ) 
    {
      switch (c) 
	{
	case 't': type_     = atoi( optarg );    break;
	case 'c': color_    = atoi(optarg);      break;
	case 'f': first_    = atoi(optarg);      break;
	case 'l': last_     = atoi(optarg);      break;
	case 'e': useEN_    = true;              break;
	case 'd': debug_    = true;              break;
	}
    }

  cout << endl;
  cout << "  ------------------------------------------" << endl; 
  cout << "  ---------  Welcome to MusEcal ------------" << endl; 
  cout << "  ------------------------------------------" << endl; 
  cout << "  -- Monitoring and Useful Survey of Ecal --" << endl;
  cout << "  -------- Clever Analysis of Laser --------" << endl;
  cout << "  ------------------------------------------" << endl; 
  cout << endl;

  ME::useElectronicNumbering = useEN_;
  MusEcal::verbose=debug_;
  MusEcal::firstRun=first_;
  MusEcal::lastRun =last_;

  UInt_t w, h;
  Double_t h1, h2;

  MusEcalGUI* me;
  TRint *theApp = new TRint("App", &argc, argv, NULL, 0);

  w  = gClient->GetDisplayWidth();
  h  = gClient->GetDisplayHeight();

  w -= 4;

  h1 = h;
  h1 -= 60;
  h2 = ((TMath::Sqrt(5.0) - 1.0)/2.0)*w;
  h  = (UInt_t)TMath::Min(h1,h2);

  me = new MusEcalGUI(  gClient->GetRoot(), w, h,
			type_, color_ );

  // test
  //  int ix_=0; int iy_=0;
  //  me->setChannel( MEGeom::iCrystal, ix_, iy_, false );
  //  me->curMgr()->fillMaps();

  theApp->Run();

  return(0);
}
