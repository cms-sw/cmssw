#ifndef EgammaProbeSelector_h
#define EgammaProbeSelector_h

/** \class EgammaProbeSelector
 *
 *  
 *  Filter to select events passing 
 *  offline jets and superclusters
 *
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




#include <math.h>

class EgammaProbeSelector : public edm::EDFilter {

    public:
        explicit EgammaProbeSelector(const edm::ParameterSet&);
        ~EgammaProbeSelector();

  	virtual bool filter(edm::Event&, const edm::EventSetup& );

   private:
	bool 		debug;

	edm::InputTag	jetLabel;
        int 		minNumberOfjets;
        double 		jetEtMin;
        double 		jetEtaMin;
        double 		jetEtaMax;

	edm::InputTag	scLabel;
        edm::InputTag        scEELabel;
        int 		minNumberOfSuperClusters;
        double 		scEtMin;
        double 		scEtaMin;
        double 		scEtaMax;

        int nEvents, nSelectedEvents;
};
#endif


   
