#ifndef HeavyChHiggsToTauNuSkim_h
#define HeavyChHiggsToTauNuSkim_h

/** \class HeavyChHiggsToTauNuSkim
 *
 *  
 *  Filter to select events passing 
 *  HLT tau+MET
 *  4 offline jets (one of them is the tau)
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
#include "FWCore/Utilities/interface/InputTag.h"

class HeavyChHiggsToTauNuSkim : public edm::EDFilter {

    public:
        explicit HeavyChHiggsToTauNuSkim(const edm::ParameterSet&);
        ~HeavyChHiggsToTauNuSkim();

  	virtual bool filter(edm::Event&, const edm::EventSetup& );

   private:

	bool 		debug;

        edm::InputTag	tauLabel;
        edm::InputTag	jetLabel;
        int 		minNumberOfjets;
        double 		jetEtMin;
        double 		jetEtaMin;
        double 		jetEtaMax;
	double		minDRFromTau;

        int nEvents, nSelectedEvents;
};
#endif


   
