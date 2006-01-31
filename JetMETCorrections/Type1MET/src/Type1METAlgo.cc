// File: Type1METAlgo.cc
// Description:  see Type1METAlgo.h
// Author: R. Cavanaugh, The University of Florida
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
void Type1METAlgo::run(const METCollection *uncorrMET, METCollection &corrMET) 
{
  //----------------- Create holders for MET delta and corrected MET objects
  MET u, d, c; 
  //----------------- Initialise
  corrMET.clear();      
  u.clearMET(); // Uncorrected MET holder
  d.clearMET(); // Delta MET holder
  c.clearMET(); // Corrected MET holder
  double delta_met     = 0.0; 
  double delta_met_x   = 0.0;
  double delta_met_y   = 0.0;
  double delta_met_z   = 0.0;
  double delta_met_phi = 0.0;
  double delta_sum_et  = 0.0;
  //----------------- Get uncorrected MET from MET collection
  METCollection::const_iterator met_iter = uncorrMET->begin();  
  u = *met_iter;        // Fill uncorrected MET from MET at zero position
  //----------------- Calculate and set deltas for new MET correction
  delta_met     = 10.0; // dummy correction for now
  delta_met_x   = 10.0; // dummy correction for now
  delta_met_y   = 10.0; // dummy correction for now
  delta_met_z   = 10.0; // dummy correction for now
  delta_met_phi = atan2( u.getMETy() + delta_met_y, u.getMETx() + delta_met_x ) - u.getPhi();
  delta_sum_et  = 10.0; // dummy correction for now
  //----------------- Fill holder with delta MET values
  d.setLabel("Type1Corr");  //Sets *internal* MET label (not EDM label)
  d.setMET(   delta_met     );
  d.setMETx(  delta_met_x   );
  d.setMETy(  delta_met_y   );
  d.setMETz(  delta_met_z   );
  d.setPhi(   delta_met_phi );
  d.setSumEt( delta_sum_et  );
  //----------------- Fill holder with corrected MET (= uncorrected + delta) values
  c.setLabel("Type1");  //Sets *internal* MET label (not EDM label)
  c.setMET(   u.getMET()   + d.getMET()   );
  c.setMETx(  u.getMETx()  + d.getMETx()  );
  c.setMETy(  u.getMETy()  + d.getMETy()  );
  c.setMETz(  u.getMETz()  + d.getMETz()  );
  c.setPhi(   u.getPhi()   + d.getPhi()   );
  c.setSumEt( u.getSumEt() + d.getSumEt() );
  //----------------- push old MET deltas (if any) to new Corrected MET collection (preserve ordering)
  met_iter = uncorrMET->end(); 
  //for( met_iter = uncorrMET->end(); met_iter != uncorrMET->begin(); met_iter-- )
  for( int count = uncorrMET->size(); count > 1; count-- ) // above line does not work, for some reason
    {
      met_iter--;
      if( met_iter != uncorrMET->begin() ) corrMET.push_back( *met_iter );
    }
  //----------------- Now add New MET deltas to Corrected MET collection 
  corrMET.push_back(d);
  //----------------- Finally, corrected MET is pushed into the zero position
  corrMET.push_back(c);
}
//----------------------------------------------------------------------------
