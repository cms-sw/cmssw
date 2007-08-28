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
#include "FWCore/ParameterSet/interface/InputTag.h"

using namespace edm;
using namespace std;

#include <math.h>

class EgammaProbeSelector : public edm::EDFilter {

    public:
        explicit EgammaProbeSelector(const edm::ParameterSet&);
        ~EgammaProbeSelector();

  	virtual bool filter(edm::Event&, const edm::EventSetup& );

   private:
	bool 		debug;

	InputTag	jetLabel;
        int 		minNumberOfjets;
        double 		jetEtMin;
        double 		jetEtaMin;
        double 		jetEtaMax;

	InputTag	scLabel;
        InputTag        scEELabel;
        int 		minNumberOfSuperClusters;
        double 		scEtMin;
        double 		scEtaMin;
        double 		scEtaMax;

        int nEvents, nSelectedEvents;
};
#endif


   
