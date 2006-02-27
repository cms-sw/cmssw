// File: Type1METAlgo.cc
// Description:  see Type1METAlgo.h
// Author: M. Schmitt, R. Cavanaugh, The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------
#include <math.h>
#include "JetMETCorrections/Type1MET/interface/Type1METAlgo.h"

using namespace std;

//----------------------------------------------------------------------------
Type1METAlgo::Type1METAlgo() {}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
Type1METAlgo::~Type1METAlgo() {}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
void Type1METAlgo::run(const TowerMETCollection *uncorrMET, TowerMETCollection &corrMET) 
{
  //----------------- Create holders for MET delta and corrected MET objects
  TowerMET u, delta, c; 
  //----------------- Initialise
  corrMET.clear();
  u.clearMET();
  //corrMET.clearMET();      
  delta.clearMET(); // Delta MET holder
  c.clearMET();
  //----------------- Set Corrected MET to uncorrected MET
  TowerMETCollection::const_iterator met_iter = uncorrMET->begin();
  u = *met_iter;
  //corrMET = *uncorrMET;
  //----------------- Calculate and set deltas for new MET correction
  double delta_met_x   = 10.0; // dummy correction for now
  double delta_met_y   = 10.0; // dummy correction for now
  double delta_met_z   = 10.0; // dummy correction for now
  double delta_sum_et  = 10.0; // dummy correction for now
  double delta_met     = sqrt( delta_met_x*delta_met_x + delta_met_y*delta_met_y );
  /*double delta_met_phi = atan2( uncorrMET->getMEy() + delta_met_y, 
				uncorrMET->getMEx() + delta_met_x ) 
				- uncorrMET->getPhi();*/
  double delta_met_phi = atan2( u.getMEy() + delta_met_y, 
				u.getMEx() + delta_met_x ) 
                         - u.getPhi();
  //----------------- Fill holder with delta MET values
  delta.setMET(   delta_met     );
  delta.setMEx(   delta_met_x   );
  delta.setMEy(   delta_met_y   );
  delta.setMEz(   delta_met_z   );
  delta.setPhi(   delta_met_phi );
  delta.setSumET( delta_sum_et  );
  //----------------- get previous corrections and push into new corrections (preserve ordering)
  std::vector<CommonMETData> prevCorr = u.getAllCorr();
  std::vector<CommonMETData>::const_iterator i = prevCorr.end();
  cout << " CORRECTION SIZE = " << prevCorr.size() << endl;
  for(int count = prevCorr.size(); count > 0; count-- ) 
    {
      i--;
      c.setLabel( i->label );
      c.setMET(   i->met );
      c.setMEx(   i->mex );
      c.setMEy(   i->mey );
      c.setMEz(   i->mez );
      c.setSumET( i->sumet );
      c.setPhi(   i->phi );
      c.pushDelta();
    }
  //----------------- set and push new MET deltas to new Corrected MET 
  c.setLabel("Type1");
  c.setMET(   delta.getMET() );
  c.setMEx(   delta.getMEx() );
  c.setMEy(   delta.getMEy() );
  c.setMEz(   delta.getMEz() );
  c.setSumET( delta.getSumET() );
  c.setPhi(   delta.getPhi() );
  c.pushDelta();
  //----------------- Fill holder with corrected MET (= uncorrected + delta) values
  c.setLabel("");  //Sets *internal* MET label (not EDM label)
  c.setMET(   u.getMET()   + delta.getMET()   );
  c.setMEx(   u.getMEx()   + delta.getMEx()   );
  c.setMEy(   u.getMEy()   + delta.getMEy()   );
  c.setMEz(   u.getMEz()   + delta.getMEz()   );
  c.setPhi(   u.getPhi()   + delta.getPhi()   );
  c.setSumET( u.getSumET() + delta.getSumET() );
  //----------------- Push onto MET Collection
  corrMET.push_back(c);
}
//----------------------------------------------------------------------------
