#ifndef EgammaProbeSelector_h
#define EgammaProbeSelector_h

/** \class EgammaProbeSelector
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

	std::string	jetLabel;
        int 		minNumberOfjets;
        double 		jetEtMin;
        double 		jetEtaMin;
        double 		jetEtaMax;

	std::string	scLabel;
        std::string     scEELabel;
        int 		minNumberOfSuperClusters;
        double 		scEtMin;
        double 		scEtaMin;
        double 		scEtaMax;

        int nEvents, nSelectedEvents;
};
#endif


   
