#ifndef HeavyChHiggsToTauNuSkim_h
#define HeavyChHiggsToTauNuSkim_h

/** \class HeavyChHiggsToTauNuSkim
 *
 *  
 *  Filter to select events passing 
 *  L1 single tau
 *  HLT tau+MET
 *  3 offline jets
 *
 *  \author Sami Lehti  -  HIP Helsinki
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <HiggsAnalysis/Skimming/interface/HiggsAnalysisSkimType.h>

using namespace edm;
using namespace std;


class HeavyChHiggsToTauNuSkim : public HiggsAnalysisSkimType {

    public:
        explicit HeavyChHiggsToTauNuSkim(const edm::ParameterSet&);
        virtual ~HeavyChHiggsToTauNuSkim();
        virtual void endJob() ;

  	virtual bool skim(edm::Event&, const edm::EventSetup&, int& trigger);


   private:
	bool 		debug;

	InputTag	jetLabel;
        int 		minNumberOfjets;
        double 		jetEtMin;
        double 		jetEtaMin;
        double 		jetEtaMax;

        unsigned int  	nEvents;
        unsigned int 	nAccepted;
};
#endif


   
