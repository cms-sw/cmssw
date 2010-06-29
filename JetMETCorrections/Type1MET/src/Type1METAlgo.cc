// File: Type1METAlgo.cc
// Description:  see Type1METAlgo.h
// Author: M. Schmitt, R. Cavanaugh, The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------
#include <math.h>
#include <vector>
#include "JetMETCorrections/Type1MET/interface/Type1METAlgo.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/SpecificCaloMETData.h"
#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"


using namespace std;
using namespace reco;

namespace {
  CaloMET makeMet (const CaloMET& fMet, 
		   double fSumEt, 
		   const vector<CorrMETData>& fCorrections, 
		   const MET::LorentzVector& fP4) {
    return CaloMET (fMet.getSpecific (), fSumEt, fCorrections, fP4, fMet.vertex ());
  }
  
  MET makeMet (const MET& fMet, 
	       double fSumEt, 
	       const vector<CorrMETData>& fCorrections, 
	       const MET::LorentzVector& fP4) {
    return MET (fSumEt, fCorrections, fP4, fMet.vertex ());
  }
  
  template <class T>
  void Type1METAlgo_run(const vector<T>& uncorMET, 
			const JetCorrector& corrector,
			const CaloJetCollection& uncorJet,
			double jetPTthreshold,
			double jetEMfracLimit, 
			vector<T>* corMET) 
  {
    if (!corMET) {
      std::cerr << "Type1METAlgo_run-> undefined output MET collection. Stop. " << std::endl;
      return;
    }
    //Jet j = uncorJet->front(); std::cout << j.px() << std::endl;
    double DeltaPx = 0.0;
    double DeltaPy = 0.0;
    double DeltaSumET = 0.0;
    // ---------------- Calculate jet corrections, but only for those uncorrected jets
    // ---------------- which are above the given threshold.  This requires that the
    // ---------------- uncorrected jets be matched with the corrected jets.
    for( CaloJetCollection::const_iterator jet = uncorJet.begin(); jet != uncorJet.end(); ++jet) {
      if( jet->pt()*corrector.correction(*jet) > jetPTthreshold && jet->emEnergyFraction() < jetEMfracLimit ) {
	double corr = corrector.correction (*jet) - 1.; // correction itself
	DeltaPx +=  jet->px() * corr;
	DeltaPy +=  jet->py() * corr;
	DeltaSumET += jet->et() * corr;
      }
    }
    //----------------- Calculate and set deltas for new MET correction
    CorrMETData delta;
    delta.mex   =  - DeltaPx;    //correction to MET (from Jets) is negative,    
    delta.mey   =  - DeltaPy;    //since MET points in direction opposite of jets
    delta.sumet =  DeltaSumET; 
    //----------------- Fill holder with corrected MET (= uncorrected + delta) values
    const T* u = &(uncorMET.front());
    double corrMetPx = u->px()+delta.mex;
    double corrMetPy = u->py()+delta.mey;
    MET::LorentzVector correctedMET4vector( corrMetPx, corrMetPy, 0., 
				       sqrt (corrMetPx*corrMetPx + corrMetPy*corrMetPy)
				       );
    //----------------- get previous corrections and push into new corrections 
    std::vector<CorrMETData> corrections = u->mEtCorr();
    corrections.push_back( delta );
    //----------------- Push onto MET Collection
    T result = makeMet (*u, u->sumEt()+delta.sumet, corrections,correctedMET4vector); 
    corMET->push_back(result);
  }
}


//----------------------------------------------------------------------------
Type1METAlgo::Type1METAlgo() {}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
Type1METAlgo::~Type1METAlgo() {}

void Type1METAlgo::run(const CaloMETCollection& uncorMET, 
		       const JetCorrector& corrector,
		       const CaloJetCollection& uncorJet,
		       double jetPTthreshold,
		       double jetEMfracLimit, 
		       CaloMETCollection* corMET) 
{
  return Type1METAlgo_run (uncorMET, corrector, uncorJet, jetPTthreshold, jetEMfracLimit, corMET);
}

void Type1METAlgo::run(const METCollection& uncorMET, 
		       const JetCorrector& corrector,
		       const CaloJetCollection& uncorJet,
		       double jetPTthreshold,
		       double jetEMfracLimit, 
		       METCollection* corMET) 
{
  return Type1METAlgo_run (uncorMET, corrector, uncorJet, jetPTthreshold, jetEMfracLimit, corMET);
}  
