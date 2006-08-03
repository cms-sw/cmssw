// File: Type1METAlgo.cc
// Description:  see Type1METAlgo.h
// Author: M. Schmitt, R. Cavanaugh, The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------
#include <math.h>
#include "JetMETCorrections/Type1MET/interface/Type1METAlgo.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CorrMETData.h"


using namespace std;
using namespace reco;

//----------------------------------------------------------------------------
Type1METAlgo::Type1METAlgo() {}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
Type1METAlgo::~Type1METAlgo() {}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
void Type1METAlgo::run(const CaloMETCollection *uncorrMET, 
		       JetInputColl uncorrJet,
		       JetInputColl corrJet, 
		       METCollection &corrMET) 
{
  double DeltaPx = 0.0;
  double DeltaPy = 0.0;
  double DeltaSumET = 0.0;
  JetInputColl::const_iterator jet = corrJet.begin();
  for( ; jet != corrJet.end(); jet++)
    {
      DeltaPx += (*jet)->px();
      DeltaPy += (*jet)->py(); 
      DeltaSumET += (*jet)->et();
    }
  std::cout << " ---------------------- " << DeltaPx << " , " << DeltaPy << std::endl;
  jet = uncorrJet.begin();
  for( ; jet != uncorrJet.end(); jet++)
    {
      DeltaPx -= (*jet)->px();
      DeltaPy -= (*jet)->py();
      DeltaSumET -= (*jet)->et();
    }
  std::cout << " ---------------------- " << DeltaPx << " , " << DeltaPy << std::endl;
  //----------------- Initialise
  corrMET.clear();
  //----------------- Set Corrected MET to uncorrected MET
  MET u = uncorrMET->front();
  std::vector<CorrMETData> corrections = u.mEtCorr();
  //----------------- Calculate and set deltas for new MET correction
  CorrMETData delta;
  if( corrections.size() > 0 ) 
    {
      CorrMETData old = corrections.back(); //cummulate corrections just to test...
      delta.mex   = old.mex + DeltaPx;   // dummy correction for now
      delta.mey   = old.mey + DeltaPy;   // dummy correction for now
      delta.sumet = old.sumet + DeltaSumET; // dummy correction for now
    }
  else
    {
      delta.mex   =  DeltaPx; // dummy correction for now
      delta.mey   =  DeltaPy; // dummy correction for now
      delta.sumet =  DeltaSumET; // dummy correction for now
    }
  //----------------- Fill holder with corrected MET (= uncorrected + delta) values
  LorentzVector correctedMET4vector( u.px()+delta.mex, 
				     u.py()+delta.mey, 
				     u.pz(), 
				     sqrt( (u.px()+delta.mex)*(u.px()+delta.mex) + 
					   (u.py()+delta.mey)*(u.py()+delta.mey) ));
  //----------------- Determine deltas to derived quantities
  //delta.met   = correctedMET.met - u.mEt();
  //delta.phi   = correctedMET.phi - u.phi();
  //----------------- get previous corrections and push into new corrections (preserve ordering)
  corrections.push_back( delta );
  //----------------- Push onto MET Collection
  Point vtx(0,0,0);
  MET result( u.sumEt()+delta.sumet, corrections, correctedMET4vector, vtx);
  corrMET.push_back(result);
}
//----------------------------------------------------------------------------
