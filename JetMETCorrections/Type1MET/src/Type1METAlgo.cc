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
#include "DataFormats/METReco/interface/SpecificCaloMETData.h"
#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/JetReco/interface/CaloJet.h"


using namespace std;
using namespace reco;

//----------------------------------------------------------------------------
Type1METAlgo::Type1METAlgo() {}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
Type1METAlgo::~Type1METAlgo() {}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
//The "run" method is overloaded to allow multiple types of MET to be corrected
//This current implementation duplicates some code and is only a simple 
//cludge to provide functionality and will be re-implemented
//to avoid code duplication.  R.Cavanaugh  
//----------------------------------------------------------------------------
void Type1METAlgo::run(const CaloMETCollection *uncorMET, 
		       const CaloJetCollection *uncorJet,
		       const CaloJetCollection *corJet,
		       CaloMETCollection &corMET) 
{
  //Jet j = uncorJet->front(); std::cout << j.px() << std::endl;
  double DeltaPx = 0.0;
  double DeltaPy = 0.0;
  double DeltaSumET = 0.0;
  std::vector<CaloJet>::const_iterator jet = corJet->begin();
  //----------------- Calculate jet corrections : start with vector sum over
  //----------------- corrected jets
  for( ; jet != corJet->end(); jet++)
    {
      DeltaPx    += jet->px();
      DeltaPy    += jet->py(); 
      DeltaSumET += jet->et();
    }
  // ---------------- Now subtract off the vector sum over uncorrected jets
  for( jet = uncorJet->begin(); jet != uncorJet->end(); jet++)
    {
      DeltaPx    -= jet->px();
      DeltaPy    -= jet->py();
      DeltaSumET -= jet->et();
    }
  //----------------- Initialise corrected MET container
  corMET.clear();
  CaloMET u = uncorMET->front();
  std::vector<CorrMETData> corrections = u.mEtCorr();
  //----------------- Calculate and set deltas for new MET correction
  CorrMETData delta;
  delta.mex   =  DeltaPx;    
  delta.mey   =  DeltaPy;    
  delta.sumet =  DeltaSumET; 
  //----------------- Fill holder with corrected MET (= uncorrected + delta) values
  LorentzVector correctedMET4vector( u.px()+delta.mex, 
				     u.py()+delta.mey, 
				     0.0, 
				     sqrt( (u.px()+delta.mex)*(u.px()+delta.mex) + 
					   (u.py()+delta.mey)*(u.py()+delta.mey) ));
  //----------------- get previous corrections and push into new corrections 
  corrections.push_back( delta );
  //std::vector<CorrMETData>::const_iterator position = corrections.begin(); 
  //corrections.insert(position, delta);
  //----------------- Push onto MET Collection
  Point vtx(0,0,0);
  SpecificCaloMETData specificCalo = u.getSpecific();
  CaloMET result( specificCalo, u.sumEt()+delta.sumet, corrections, correctedMET4vector, vtx);
  corMET.push_back(result);
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
//The "run" method is overloaded to allow multiple types of MET to be corrected
//This current implementation duplicates some code and is only a simple 
//cludge to provide functionality and will be re-implemented
//to avoid code duplication.  R.Cavanaugh  
//----------------------------------------------------------------------------
void Type1METAlgo::run(const METCollection *uncorMET, 
		       const CaloJetCollection *uncorJet,
		       const CaloJetCollection *corJet,
		       METCollection &corMET) 
{
  //Jet j = uncorJet->front(); std::cout << j.px() << std::endl;
  double DeltaPx = 0.0;
  double DeltaPy = 0.0;
  double DeltaSumET = 0.0;
  std::vector<CaloJet>::const_iterator jet = corJet->begin();
  //----------------- Calculate jet corrections : start with vector sum over
  //----------------- corrected jets
  for( ; jet != corJet->end(); jet++)
    {
      DeltaPx    += jet->px();
      DeltaPy    += jet->py(); 
      DeltaSumET += jet->et();
    }
  // ---------------- Now subtract off the vector sum over uncorrected jets
  for( jet = uncorJet->begin(); jet != uncorJet->end(); jet++)
    {
      DeltaPx    -= jet->px();
      DeltaPy    -= jet->py();
      DeltaSumET -= jet->et();
    }
  //----------------- Initialise corrected MET container
  corMET.clear();
  MET u = uncorMET->front();
  std::vector<CorrMETData> corrections = u.mEtCorr();
  //----------------- Calculate and set deltas for new MET correction
  CorrMETData delta;
  delta.mex   =  DeltaPx;    
  delta.mey   =  DeltaPy;    
  delta.sumet =  DeltaSumET; 
  //----------------- Fill holder with corrected MET (= uncorrected + delta) values
  LorentzVector correctedMET4vector( u.px()+delta.mex, 
				     u.py()+delta.mey, 
				     0.0, 
				     sqrt( (u.px()+delta.mex)*(u.px()+delta.mex) + 
					   (u.py()+delta.mey)*(u.py()+delta.mey) ));
  //----------------- get previous corrections and push into new corrections 
  corrections.push_back( delta );
  //std::vector<CorrMETData>::const_iterator position = corrections.begin(); 
  //corrections.insert(position, delta);
  //----------------- Push onto MET Collection
  Point vtx(0,0,0);
  MET result( u.sumEt()+delta.sumet, corrections, correctedMET4vector, vtx);
  corMET.push_back(result);
}
//----------------------------------------------------------------------------
