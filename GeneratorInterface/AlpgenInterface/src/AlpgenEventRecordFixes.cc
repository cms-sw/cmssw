#include "GeneratorInterface/AlpgenInterface/interface/AlpgenEventRecordFixes.h"
#include <cstdlib>

// An important peculiarity of this code is that, inside the 
// HEPEUP, particles are numbered from 1 (a la FORTRAN), but
// the array that holds HEPEUP is acessed starting from 0
// (a la C/C++). This has to be given due consideration.
// For instance: to set a particle w/ index iup status code:
// 
//     hepeup.ISTUP[iup] = 2
//
// but to set that particle as mother of a particle idau:
// 
//     hepeup.MOTHUP[idau].first = iup+1
//                                ^^^^^^^ See the difference?
// Be careful with this.

math::XYZTLorentzVector alpgen::vectorFromHepeup(const lhef::HEPEUP& hepeup,
					 int index) {
  return math::XYZTLorentzVector (hepeup.PUP[index][0],
				  hepeup.PUP[index][1],
				  hepeup.PUP[index][2],
				  hepeup.PUP[index][3]);
}


void alpgen::fixEventWZ(lhef::HEPEUP &hepeup)
{
  // Nomenclature...
  int nup = hepeup.NUP;

  // Open up space for the vector boson.
  hepeup.resize(nup+1);

  // Assignments specific to individual hard processes.
  // This one fixes the Event Record for W and Z.
  int iwch = 0;

  // The last two particles in the record make the boson.
  double bosonPx = 0.; 
  double bosonPy = 0.; 
  double bosonPz = 0.; 
  double bosonE = 0.; 
  for(int i = nup - 2; i != nup; ++i) {
    bosonPx += hepeup.PUP[i][0];
    bosonPy += hepeup.PUP[i][1];
    bosonPz += hepeup.PUP[i][2];
    bosonE  += hepeup.PUP[i][3];
    hepeup.MOTHUP[i].first = nup + 1;
    hepeup.MOTHUP[i].second = 0;
    iwch = (iwch - hepeup.IDUP[i] % 2);
  }
  // electron+nubar -> 11 + (-12) => -(1)+0 = -1  => W-
  // positron+nu    -> -11+ 12    => -(-1)+0 = +1 => W+
  // u dbar -> 2 -1  => 0 -(-1) = 1 => W+
  // c dbar -> 4 -1  => W+
  // etc.

  // Boson ID.
  int bosonIndex = nup;
  int bosonId = 23;
  double bosonMass = std::sqrt(bosonE * bosonE -
			       (bosonPx * bosonPx +
				bosonPy * bosonPy +
				bosonPz * bosonPz));
  if (iwch > 0) bosonId = 24;
  if (iwch < 0) bosonId = -24;

  // Boson in the Event Record.
  hepeup.IDUP[bosonIndex] = bosonId;
  hepeup.ISTUP[bosonIndex] = 2;
  hepeup.MOTHUP[bosonIndex].first = 1;
  hepeup.MOTHUP[bosonIndex].second = 2;
  hepeup.PUP[bosonIndex][0] = bosonPx;
  hepeup.PUP[bosonIndex][1] = bosonPy;
  hepeup.PUP[bosonIndex][2] = bosonPz;
  hepeup.PUP[bosonIndex][3] = bosonE;
  hepeup.PUP[bosonIndex][4] = bosonMass;
  hepeup.ICOLUP[bosonIndex].first = 0;
  hepeup.ICOLUP[bosonIndex].second = 0;

  hepeup.AQEDUP = hepeup.AQCDUP = -1.0; // alphas are not saved by Alpgen
  for(int i = 0; i < hepeup.NUP; i++)
    hepeup.SPINUP[i] = -9;	// Alpgen does not store spin information
}


void alpgen::fixEventMultiBoson(lhef::HEPEUP &hepeup)
{
  int nup = hepeup.NUP;

  // find first gauge bosons
  int ivstart=0;
  int ivend=0;
  for(int i = 0; i != nup; ++i) {
    if(std::abs(hepeup.IDUP[i]) == 24 || hepeup.IDUP[i] == 23) {
      hepeup.ISTUP[i] = 2;
      if(ivstart == 0) ivstart = i;
      ivend = i+1;
    }
  }
  int nvb = ivend-ivstart;
  
  // decay products pointers, starting from the end
  for(int i = 0; i != nvb; ++i) {
    hepeup.MOTHUP[nup - 2*i -1].first = ivend-i;
    hepeup.MOTHUP[nup - 2*i -2].first = ivend-i;
    hepeup.MOTHUP[nup - 2*i -1].second = 0;
    hepeup.MOTHUP[nup - 2*i -2].second = 0;
  }

  hepeup.AQEDUP = hepeup.AQCDUP = -1.0; // alphas are not saved by Alpgen
  for(int i = 0; i < hepeup.NUP; i++)
    hepeup.SPINUP[i] = -9;	// Alpgen does not store spin information

}


void alpgen::fixEventTTbar(lhef::HEPEUP &hepeup)
{
  using namespace math;
  
  int nup = hepeup.NUP;
  
  // Assert top is in the third position.
  int thirdID = hepeup.IDUP[2];
  if(std::abs(thirdID) != 6) {
    std::cout << "Top is NOT in the third position - no need to fix." << std::endl;
    return;
  } 

  // Open up space for 2 W bosons and two b quarks. 
  hepeup.resize(nup+4);

  // reset top status codes
  hepeup.ISTUP[2] = 2;
  hepeup.ISTUP[3] = 2;
  int it;
  int itbar;
  if(thirdID == 6) {
    it = 2;
    itbar = 3;
  }
  else {
    it = 3;
    itbar = 2;
  }

  // Reconstruct W's from decay product, fix mother-daughter relations.
  // Glossary: iwdec is the location of the first W decay product;
  //           iwup is the location where the W will be put.
  //           ibup is the location where the b will be put.
  for (int iw = 0; iw != 2; ++iw) {
    int iwdec = nup - 4 + 2*iw;
    int iwup = nup + iw;
    int ibup = iwup + 2;
    int iwch = 0;
    for(int iup = iwdec; iup != iwdec+2; ++iup){
      hepeup.MOTHUP[iup].first = iwup+1;
      hepeup.MOTHUP[iup].second = 0;
      iwch = (iwch - hepeup.IDUP[iup]%2); 
    }
    if(iwch > 0) {
      hepeup.IDUP[iwup] = 24;
      hepeup.IDUP[ibup] = 5;
      hepeup.MOTHUP[iwup].first = it+1;
      hepeup.MOTHUP[iwup].second = 0;
      hepeup.MOTHUP[ibup].first = it+1;
      hepeup.MOTHUP[ibup].second = 0;
    }
    if(iwch < 0) {
      hepeup.IDUP[iwup] = -24;
      hepeup.IDUP[ibup] = -5;
      hepeup.MOTHUP[iwup].first = itbar+1;
      hepeup.MOTHUP[iwup].second = 0;
      hepeup.MOTHUP[ibup].first = itbar+1;
      hepeup.MOTHUP[ibup].second = 0;
    }
    hepeup.ISTUP[iwup] = 2;
    hepeup.ISTUP[ibup] = 1;

    // Reconstruct W momentum from its children.
    XYZTLorentzVector child1; 
    child1 = vectorFromHepeup(hepeup, iwdec); 
    XYZTLorentzVector child2;
    child2 = vectorFromHepeup(hepeup,iwdec+1);
    
    XYZTLorentzVector bosonW; bosonW = (child1+child2);
    hepeup.PUP[iwup][0] = bosonW.Px();
    hepeup.PUP[iwup][1] = bosonW.Py();
    hepeup.PUP[iwup][2] = bosonW.Pz();
    hepeup.PUP[iwup][3] = bosonW.E();
    hepeup.PUP[iwup][4] = bosonW.M();

    // Reconstruct b momentum from W and t.
    int topIndex = (hepeup.MOTHUP[iwup].first)-1; 
    XYZTLorentzVector topQuark;
    topQuark = vectorFromHepeup(hepeup, topIndex);

    XYZTLorentzVector bottomQuark; bottomQuark = (topQuark-bosonW);
    hepeup.PUP[ibup][0] = bottomQuark.Px();
    hepeup.PUP[ibup][1] = bottomQuark.Py();
    hepeup.PUP[ibup][2] = bottomQuark.Pz();
    hepeup.PUP[ibup][3] = bottomQuark.E();
    hepeup.PUP[ibup][4] = bottomQuark.M();

    // Set color labels.
    hepeup.ICOLUP[iwup].first = 0;
    hepeup.ICOLUP[iwup].second = 0;
    hepeup.ICOLUP[ibup].first = hepeup.ICOLUP[(hepeup.MOTHUP[iwup].first)-1].first;
    hepeup.ICOLUP[ibup].second = hepeup.ICOLUP[(hepeup.MOTHUP[iwup].first)-1].second;
  }

  hepeup.AQEDUP = hepeup.AQCDUP = -1.0; // alphas are not saved by Alpgen
  for(int i = 0; i < hepeup.NUP; i++)
    hepeup.SPINUP[i] = -9;	// Alpgen does not store spin information

}


void alpgen::fixEventHiggsTTbar(lhef::HEPEUP &hepeup)
{
  using namespace math;
  
  int nup = hepeup.NUP;
  
  // Assert top is in the fourth position.
  int fourthID = hepeup.IDUP[3];
  if(std::abs(fourthID) != 6) {
    std::cout << "Top is NOT in the fourth position - no need to fix."  << std::endl;
    return;
  } 

  // Open up space for 2 W bosons and two b quarks. 
  hepeup.resize(nup+4);

  // reset top status codes
  hepeup.ISTUP[3] = 2;
  hepeup.ISTUP[4] = 2;
  int it;
  int itbar;
  if(fourthID == 6) {
    it = 3;
    itbar = 4;
  }
  else {
    it = 4;
    itbar = 3;
  }

  // Reconstruct W's from decay product, fix mother-daughter relations.
  // Glossary: iwdec is the location of the first W decay product;
  //           iwup is the location where the W will be put.
  //           ibup is the location where the b will be put.
  for (int iw = 0; iw != 2; ++iw) {
    int iwdec = nup - 4 + 2*iw;
    int iwup = nup + iw;
    int ibup = iwup + 2;
    int iwch = 0;
    for(int iup = iwdec; iup != iwdec+2; ++iup){
      hepeup.MOTHUP[iup].first = iwup+1;
      hepeup.MOTHUP[iup].second = 0;
      iwch = (iwch - hepeup.IDUP[iup]%2); 
    }
    if(iwch > 0) {
      hepeup.IDUP[iwup] = 24;
      hepeup.IDUP[ibup] = 5;
      hepeup.MOTHUP[iwup].first = it+1;
      hepeup.MOTHUP[iwup].second = 0;
      hepeup.MOTHUP[ibup].first = it+1;
      hepeup.MOTHUP[ibup].second = 0;
    }
    if(iwch < 0) {
      hepeup.IDUP[iwup] = -24;
      hepeup.IDUP[ibup] = -5;
      hepeup.MOTHUP[iwup].first = itbar+1;
      hepeup.MOTHUP[iwup].second = 0;
      hepeup.MOTHUP[ibup].first = itbar+1;
      hepeup.MOTHUP[ibup].second = 0;
    }
    hepeup.ISTUP[iwup] = 2;
    hepeup.ISTUP[ibup] = 1;

    // Reconstruct W momentum from its children.
    XYZTLorentzVector child1; 
    child1 = vectorFromHepeup(hepeup, iwdec); 
    XYZTLorentzVector child2;
    child2 = vectorFromHepeup(hepeup,iwdec+1);
    
    XYZTLorentzVector bosonW; bosonW = (child1+child2);
    hepeup.PUP[iwup][0] = bosonW.Px();
    hepeup.PUP[iwup][1] = bosonW.Py();
    hepeup.PUP[iwup][2] = bosonW.Pz();
    hepeup.PUP[iwup][3] = bosonW.E();
    hepeup.PUP[iwup][4] = bosonW.M();

    // Reconstruct b momentum from W and t.
    int topIndex = (hepeup.MOTHUP[iwup].first)-1; 
    XYZTLorentzVector topQuark;
    topQuark = vectorFromHepeup(hepeup, topIndex);

    XYZTLorentzVector bottomQuark; bottomQuark = (topQuark-bosonW);
    hepeup.PUP[ibup][0] = bottomQuark.Px();
    hepeup.PUP[ibup][1] = bottomQuark.Py();
    hepeup.PUP[ibup][2] = bottomQuark.Pz();
    hepeup.PUP[ibup][3] = bottomQuark.E();
    hepeup.PUP[ibup][4] = bottomQuark.M();

    // Set color labels.
    hepeup.ICOLUP[iwup].first = 0;
    hepeup.ICOLUP[iwup].second = 0;
    hepeup.ICOLUP[ibup].first = hepeup.ICOLUP[(hepeup.MOTHUP[iwup].first)-1].first;
    hepeup.ICOLUP[ibup].second = hepeup.ICOLUP[(hepeup.MOTHUP[iwup].first)-1].second;
  }

  hepeup.AQEDUP = hepeup.AQCDUP = -1.0; // alphas are not saved by Alpgen
  for(int i = 0; i < hepeup.NUP; i++)
    hepeup.SPINUP[i] = -9;	// Alpgen does not store spin information

}


void alpgen::fixEventSingleTop(lhef::HEPEUP &hepeup, double mb, int itopprc) {
  using namespace math;
  
  int nup = hepeup.NUP;
  
  // Number of W bosons.
  int nw = 1;
  // Fix number of Ws in final state.
  if(itopprc >= 3) nw = 2;

  // Open up space for W bosons and b quarks. 
  hepeup.resize(nup+2);
  
  // Assign mass to the incoming bottom quark, if required.
  for(int i =0; i!= 2; ++i) {
    if(std::abs(hepeup.IDUP[i]) == 5) {
      hepeup.PUP[i][4] = mb;
      double energyb = hepeup.PUP[i][3];
      hepeup.PUP[i][3] = std::sqrt(energyb*energyb + mb*mb);
    }
  }

  // Identify the top.
  hepeup.ISTUP[2] = 2;
  int it = 0;
  int itbar = 0;
  if(hepeup.IDUP[2] == 6)
    it = 2;
  else if(hepeup.IDUP[2] == -6)
    itbar = 2;
  else {
    std::cout << "Wrong assumption about top position, stop." << std::endl;
    // FIXME: Should throw an exception here.
    return;
  }

  // TOP DECAY
  // Reconstruct W's from decay products.
  // iwdec: 1st W decay product.
  int iwdec = 0;
  if(nw == 1)
    iwdec = nup-2;
  else if(nw == 2)
    iwdec = nup-4;

  // Put W and b at the end.
  int iwup = nup;
  int ibup = iwup+1;
  int iwch = 0;
  for(int iup = iwdec; iup != iwdec+2; ++iup) {
      hepeup.MOTHUP[iup].first = iwup+1;
      hepeup.MOTHUP[iup].second = 0;
      iwch = (iwch - hepeup.IDUP[iup]%2); 
  }

  if(iwch > 0) {
    hepeup.IDUP[iwup] = 24;
    hepeup.IDUP[ibup] = 5;
    hepeup.MOTHUP[iwup].first = it+1;
    hepeup.MOTHUP[iwup].second = 0;
    hepeup.MOTHUP[ibup].first = it+1;
    hepeup.MOTHUP[ibup].second = 0;
    }
  if(iwch < 0) {
    hepeup.IDUP[iwup] = -24;
    hepeup.IDUP[ibup] = -5;
    hepeup.MOTHUP[iwup].first = itbar+1;
    hepeup.MOTHUP[iwup].second = 0;
    hepeup.MOTHUP[ibup].first = itbar+1;
    hepeup.MOTHUP[ibup].second = 0;
    }
  hepeup.ISTUP[iwup] = 2;
  hepeup.ISTUP[ibup] = 1;

  // Reconstruct W momentum from its children.
  XYZTLorentzVector child1; 
  child1 = vectorFromHepeup(hepeup, iwdec); 
  XYZTLorentzVector child2;
  child2 = vectorFromHepeup(hepeup, iwdec+1);
  
  XYZTLorentzVector bosonW; bosonW = (child1+child2);
  hepeup.PUP[iwup][0] = bosonW.Px();
  hepeup.PUP[iwup][1] = bosonW.Py();
  hepeup.PUP[iwup][2] = bosonW.Pz();
  hepeup.PUP[iwup][3] = bosonW.E();
  hepeup.PUP[iwup][4] = bosonW.M();
  
  // Reconstruct b momentum from W and t.
  int topIndex = (hepeup.MOTHUP[iwup].first)-1; 
  XYZTLorentzVector topQuark;
  topQuark = vectorFromHepeup(hepeup, topIndex);
  
  XYZTLorentzVector bottomQuark; bottomQuark = (topQuark-bosonW);
  hepeup.PUP[ibup][0] = bottomQuark.Px();
  hepeup.PUP[ibup][1] = bottomQuark.Py();
  hepeup.PUP[ibup][2] = bottomQuark.Pz();
  hepeup.PUP[ibup][3] = bottomQuark.E();
  hepeup.PUP[ibup][4] = bottomQuark.M();
  
  // Set color labels.
  hepeup.ICOLUP[iwup].first = 0;
  hepeup.ICOLUP[iwup].second = 0;
  hepeup.ICOLUP[ibup].first = hepeup.ICOLUP[(hepeup.MOTHUP[iwup].first)-1].first;
  hepeup.ICOLUP[ibup].second = hepeup.ICOLUP[(hepeup.MOTHUP[iwup].first)-1].second;

  nup = nup+2;
  
  // If there is a second W, we deal with it now.
  // Unlike the first W, this one appears EXPLICITLY in the Event Record.
  if(nw == 2){
    // W DECAY
    // iwdec: 1st W decay product.
    iwdec = nup-4;
    // iwup: location of the W in the event record.
    iwup = nup-7;
    iwch = 0;
    for(int iup = iwdec; iup != iwdec+2; ++iup) {
      hepeup.MOTHUP[iup].first = iwup+1;
      hepeup.MOTHUP[iup].second = 0;
      iwch = (iwch - hepeup.IDUP[iup]%2); 
    }
    hepeup.ISTUP[iwup] = 2;
    hepeup.ICOLUP[iwup].first = 0;
    hepeup.ICOLUP[iwup].second = 0;
  }

}
