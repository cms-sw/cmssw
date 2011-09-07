// File: Type1METAlgo.cc
// Description:  see Type1METAlgo.h
// Author: M. Schmitt, R. Cavanaugh, The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------
// $Id$

#include <math.h>
#include <vector>
#include "JetMETCorrections/Type1MET/interface/Type1METAlgo.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/SpecificCaloMETData.h"
#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

using namespace std;
using namespace reco;

typedef math::XYZTLorentzVector LorentzVector;

namespace {
  CaloMET makeMet (const CaloMET& fMet, 
		   double fSumEt, 
		   const std::vector<CorrMETData>& fCorrections, 
		   const MET::LorentzVector& fP4) {
    return CaloMET (fMet.getSpecific (), fSumEt, fCorrections, fP4, fMet.vertex ());
  }
  
  MET makeMet (const PFMET& fMet, 
	       double fSumEt, 
	       const std::vector<CorrMETData>& fCorrections, 
	       const MET::LorentzVector& fP4) {
    return MET (fSumEt, fCorrections, fP4, fMet.vertex ());
  }
  

  ///////////////
  //  CALO MET
  ///////////////

  template <class T>
  void Type1METAlgo_run(const std::vector<T>& uncorMET, 
			const JetCorrector& corrector,
			const JetCorrector& corrector2,
			const CaloJetCollection& uncorJet,
			double jetPTthreshold,
			double jetEMfracLimit, 
 		        double UscaleA,
 		        double UscaleB,
 		        double UscaleC,
                        bool useTypeII,
                        bool hasMuonsCorr, 
			const edm::View<reco::Muon>& inputMuons,
                        const edm::ValueMap<reco::MuonMETCorrectionData>& vm_muCorrData,
			vector<T>* corMET, edm::Event& iEvent, const edm::EventSetup& iSetup, const bool subtractL1Fast) 
  {
    if (!corMET) {
      std::cerr << "Type1METAlgo_run-> undefined output MET collection. Stop. " << std::endl;
      return;
    }
  
    const T* u = &(uncorMET.front());

    double DeltaPx = 0.0;
    double DeltaPy = 0.0;
    double UDeltaP = 0.0;
    double UDeltaPx = 0.0;
    double UDeltaPy = 0.0;
    double DeltaSumET = 0.0;
    double USumET = u->sumEt();
    // ---------------- Calculate jet corrections, but only for those uncorrected jets
    // ---------------- which are above the given threshold.  This requires that the
    // ---------------- uncorrected jets be matched with the corrected jets.
    for( CaloJetCollection::const_iterator jet = uncorJet.begin(); jet != uncorJet.end(); ++jet) {
      if( jet->pt()*corrector.correction (*jet,iEvent,iSetup) > jetPTthreshold && jet->emEnergyFraction() < jetEMfracLimit ) {
	if (!subtractL1Fast)
	{
	  double corr = corrector.correction (*jet,iEvent,iSetup) - 1.; // correction itself
	  DeltaPx +=  jet->px() * corr;
	  DeltaPy +=  jet->py() * corr;
	  DeltaSumET += jet->et() * corr;
	} 
	else
	{
	  double corr = corrector.correction (*jet,iEvent,iSetup) - 1.; // correction itself
	  double corr2 = corrector2.correction (*jet,iEvent,iSetup) - 1.;
	  DeltaPx +=  jet->px() * corr - (jet->px() * corr2);
	  DeltaPy +=  jet->py() * corr - (jet->py() * corr2);
	  DeltaSumET += jet->et() * corr - (jet->et() * corr2);
	}
	UDeltaPx +=  jet->px() ;
	UDeltaPy +=  jet->py() ;
	USumET -=jet->et();
      }
      if( jet->pt() *corrector.correction (*jet,iEvent,iSetup)> jetPTthreshold && jet->emEnergyFraction() > jetEMfracLimit ) {
        UDeltaPx +=  jet->px() ;
        UDeltaPy +=  jet->py() ;
        USumET -=jet->et(); 
      }
    }
    if( hasMuonsCorr) {
       unsigned int nMuons = inputMuons.size();
       for(unsigned int iMu = 0; iMu<nMuons; iMu++) {
        const reco::Muon *mu = &inputMuons[iMu]; //new
        reco::MuonMETCorrectionData muCorrData = (vm_muCorrData)[inputMuons.refAt(iMu)];
        int flag   = muCorrData.type();

        LorentzVector mup4;
        if (flag == 0) 		//this muon is not used to correct the MET
          continue;
        mup4 = mu->p4();

        UDeltaPx +=  mup4.px();
        UDeltaPy +=  mup4.py();
        USumET -=mup4.pt(); 
       }

    } // end hasMuonsCorr

  
    //----------------- Calculate and set deltas for new MET correction
    CorrMETData delta;
    delta.mex   =  - DeltaPx;    //correction to MET (from Jets) is negative,    
    delta.mey   =  - DeltaPy;    //since MET points in direction opposite of jets
    delta.sumet =  DeltaSumET; 
    //----------------- Fill holder with corrected MET (= uncorrected + delta) values
    UDeltaPx += u->px();
    UDeltaPy += u->py();


     UDeltaP = sqrt(UDeltaPx*UDeltaPx+UDeltaPy*UDeltaPy);
    double Uscale = UscaleA+UscaleB*exp(UscaleC*UDeltaP);

     if(useTypeII){
       delta.mex += (Uscale-1.)*UDeltaPx;
       delta.mey += (Uscale-1.)*UDeltaPy;
       delta.sumet += (Uscale-1.)*USumET;
     }

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


  ///////////////
  //  PF MET (PFJetCollection)
  ///////////////


  void Type1METAlgo_run(const reco::PFMETCollection& uncorMET, 
			const JetCorrector& corrector,
			const JetCorrector& corrector2,
			const PFJetCollection& uncorJet,
			const PFCandidateCollection& uncorUnclustered,
			double jetPTthreshold,
			double jetEMfracLimit, 
 		        double UscaleA,
 		        double UscaleB,
 		        double UscaleC,
                        bool useTypeII,
			vector<reco::PFMET>* corMET, edm::Event& iEvent, const edm::EventSetup& iSetup, const bool subtractL1Fast) 
  {
    if (!corMET) {
      std::cerr << "Type1METAlgo_run-> undefined output MET collection. Stop. " << std::endl;
      return;
    }

    const reco::PFMET* u = &(uncorMET.front());

    //Jet j = uncorJet->front(); std::cout << j.px() << std::endl;
   double DeltaPx = 0.0;
   double DeltaPy = 0.0;
   double DeltaSumET = 0.0;

   double UDeltaP = 0.0;
   double UDeltaPx = 0.0;
   double UDeltaPy = 0.0;
   double USumET = 0;
    // ---------------- Calculate jet corrections, but only for those uncorrected jets
    // ---------------- which are above the given threshold.  This requires that the
    // ---------------- uncorrected jets be matched with the corrected jets.
    for( PFJetCollection::const_iterator jet = uncorJet.begin(); jet != uncorJet.end(); ++jet) {
      if( jet->pt()*corrector.correction (*jet,iEvent,iSetup) > jetPTthreshold && jet->photonEnergyFraction() < jetEMfracLimit ) {

	if (!subtractL1Fast)
	{
	  double corr = corrector.correction (*jet,iEvent,iSetup) - 1.; // correction itself
	  DeltaPx +=  jet->px() * corr;
	  DeltaPy +=  jet->py() * corr;
	  DeltaSumET += jet->et() * corr;
        }
	else
	{
	  double corr = corrector.correction (*jet,iEvent,iSetup) - 1.; // correction itself
	  double corr2 = corrector2.correction (*jet,iEvent,iSetup) - 1.;
	  DeltaPx +=  jet->px() * corr - (jet->px() * corr2);
	  DeltaPy +=  jet->py() * corr - (jet->py() * corr2);
	  DeltaSumET += jet->et() * corr - (jet->et() * corr2);
	}
      }

      if (jet->pt() * corrector.correction (*jet,iEvent,iSetup) < jetPTthreshold && jet->photonEnergyFraction() < jetEMfracLimit) {
	UDeltaPx -= jet->px();
	UDeltaPy -= jet->py();
	USumET += jet->et();
      }
      
    }
    //if typeII correction should be added, calculate U using collections of unclustered PFCandidates handed in by user
    if (useTypeII) {
	for (PFCandidateCollection::const_iterator cand =
				uncorUnclustered.begin(); cand != uncorUnclustered.end(); ++cand) {

	UDeltaPx -= cand->px();
	UDeltaPy -= cand->py();
	USumET += cand->et();

	}
    }

    //----------------- Calculate and set deltas for new MET correction
    CorrMETData delta;
    delta.mex   =  - DeltaPx;    //correction to MET (from Jets) is negative,    
    delta.mey   =  - DeltaPy;    //since MET points in direction opposite of jets
    delta.sumet =  DeltaSumET; 
    
    //std::cout << "delta.mex = " << delta.mex << std::endl;

     UDeltaP = sqrt(UDeltaPx*UDeltaPx+UDeltaPy*UDeltaPy);
     
    double Uscale = UscaleA+UscaleB*exp(UscaleC*UDeltaP);
    if (UDeltaP == 0){
	Uscale = 1;
    }

     if(useTypeII){
       delta.mex += (Uscale-1.)*UDeltaPx;
       delta.mey += (Uscale-1.)*UDeltaPy;
       delta.sumet += (Uscale-1.)*USumET;
     }

     double corrMetPx = u->px()+delta.mex;
     double corrMetPy = u->py()+delta.mey;

  
    MET::LorentzVector correctedMET4vector( corrMetPx, corrMetPy, 0., 
				       sqrt (corrMetPx*corrMetPx + corrMetPy*corrMetPy)
				       );
    //----------------- get previous corrections and push into new corrections 
    std::vector<CorrMETData> corrections = u->mEtCorr();
    corrections.push_back( delta );
    //----------------- Push onto MET Collection

  reco::PFMET specificPFMET( u->getSpecific(), u->sumEt()+delta.sumet, correctedMET4vector, u->vertex() );
  corMET->push_back(specificPFMET);

  }

}

//----------------------------------------------------------------------------
Type1METAlgo::Type1METAlgo() {}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
Type1METAlgo::~Type1METAlgo() {}

void Type1METAlgo::run(const CaloMETCollection& uncorMET, 
		       const JetCorrector& corrector,
		       const JetCorrector& corrector2,
		       const CaloJetCollection& uncorJet,
		       double jetPTthreshold,
		       double jetEMfracLimit, 
		       double UscaleA, 
		       double UscaleB, 
		       double UscaleC, 
		       bool useTypeII, 
                       bool hasMuonsCorr,
                       const edm::View<reco::Muon>& inputMuons,
                       const edm::ValueMap<reco::MuonMETCorrectionData>& vm_muCorrData,
		       CaloMETCollection* corMET, edm::Event& iEvent, const edm::EventSetup& iSetup, const bool subtractL1Fast) 
{
  return Type1METAlgo_run(uncorMET, corrector, corrector2, uncorJet, jetPTthreshold, jetEMfracLimit, UscaleA, UscaleB, UscaleC, useTypeII, hasMuonsCorr,inputMuons, vm_muCorrData, corMET, iEvent, iSetup, subtractL1Fast);
}

void Type1METAlgo::run(const PFMETCollection& uncorMET, 
		       const JetCorrector& corrector,
		       const JetCorrector& corrector2,
		       const PFJetCollection& uncorJet,
		       const PFCandidateCollection& uncorUnclustered,
		       double jetPTthreshold,
		       double jetEMfracLimit, 
		       double UscaleA,
                       double UscaleB,
                       double UscaleC,
                       bool useTypeII, 
		       PFMETCollection* corMET, edm::Event& iEvent, const edm::EventSetup& iSetup, const bool subtractL1Fast) 
{
  return Type1METAlgo_run(uncorMET, corrector, corrector2, uncorJet, uncorUnclustered, jetPTthreshold, jetEMfracLimit, UscaleA, UscaleB, UscaleC, useTypeII, corMET, iEvent, iSetup, subtractL1Fast);
}  

