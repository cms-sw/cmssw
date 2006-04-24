// File: Type1METAlgo.cc
// Description:  see Type1METAlgo.h
// Author: M. Schmitt, R. Cavanaugh, The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------
#include <math.h>
#include "JetMETCorrections/Type1MET/interface/Type1METAlgo.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

using namespace std;
using namespace reco;

//----------------------------------------------------------------------------
Type1METAlgo::Type1METAlgo() {}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
Type1METAlgo::~Type1METAlgo() {}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
void Type1METAlgo::run(const METCollection *uncorrMET, METCollection &corrMET) 
{
  //----------------- Initialise
  corrMET.clear();
  //----------------- Set Corrected MET to uncorrected MET
  MET u = uncorrMET->front();
  CommonMETData correctedMET = u.mEtData();
  std::vector<CommonMETData> corrections = u.mEtCorr();
  //----------------- Calculate and set deltas for new MET correction
      CommonMETData delta;
  if( corrections.size() > 0 ) 
    {
      CommonMETData old = corrections.back();
      delta.mex   = old.mex + 10.0; // dummy correction for now
      delta.mey   = old.mey + 10.0; // dummy correction for now
      delta.mez   = old.mez + 10.0; // dummy correction for now
      delta.sumet = old.sumet + 10.0; // dummy correction for now
    }
  else
    {
      delta.mex   =  + 10.0; // dummy correction for now
      delta.mey   =  + 10.0; // dummy correction for now
      delta.mez   =  + 10.0; // dummy correction for now
      delta.sumet =  + 10.0; // dummy correction for now
    }
  //----------------- Fill holder with corrected MET (= uncorrected + delta) values
  correctedMET.mex   += delta.mex;
  correctedMET.mey   += delta.mey;
  correctedMET.mez   += delta.mez;
  correctedMET.sumet += delta.sumet;
  correctedMET.met    = sqrt( correctedMET.mex*correctedMET.mex + 
			      correctedMET.mey*correctedMET.mey );
  correctedMET.phi    = atan2( correctedMET.mey, correctedMET.mex );
  //----------------- Determine deltas to derived quantities
  delta.met   = correctedMET.met - u.mEt();
  delta.phi   = correctedMET.phi - u.phi();
  //----------------- get previous corrections and push into new corrections (preserve ordering)
  corrections.push_back( delta );
  //----------------- Push onto MET Collection
  MET result( correctedMET, corrections );
  corrMET.push_back(result);
}
//----------------------------------------------------------------------------
