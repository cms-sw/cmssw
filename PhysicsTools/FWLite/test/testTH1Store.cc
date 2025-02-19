// -*- C++ -*-

#include <iostream>

// CMS includes
#include "PhysicsTools/FWLite/interface/TH1Store.h"

// Root includes
#include "TROOT.h"

using namespace std;

///////////////////////////
// ///////////////////// //
// // Main Subroutine // //
// ///////////////////// //
///////////////////////////

int main (int argc, char* argv[]) 
{

   cout << "creating store" << endl;
   TH1Store store;

   cout << "adding histograms" << endl;
   store.add( new TH1F ("aaa", "aaa", 10, 0.5, 10.5) );
   store.add( new TH1F ("bbb", "bbb", 10, 0.5, 10.5), "one/two");
   store.add( new TH1F ("ccc", "ccc", 10, 0.5, 10.5), "three");
   store.add( new TH1F ("ddd", "ddd", 10, 0.5, 10.5), "one");
   store.add( new TH1F ("eee", "eee", 10, 0.5, 10.5), "three/four");

   cout << "filling" << endl;
   store.hist("aaa")->Fill(1);
   store.hist("bbb")->Fill(2);
   store.hist("ccc")->Fill(3);
   store.hist("ddd")->Fill(4);
   store.hist("eee")->Fill(5);

   cout << "saving" << endl;
   store.write ("test.root");
   cout << "done!" << endl;
   // All done!  Bye bye.
   return 0;
}
