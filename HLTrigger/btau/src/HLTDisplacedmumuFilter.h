#ifndef HLTDisplacedmumuFilter_h
#define HLTDisplacedmumuFilter_h
//
// Package:    HLTstaging
// Class:      HLTDisplacedmumuFilter
// 
/**\class HLTDisplacedmumuFilter 

 HLT Filter for b to (mumu) + X

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Nicolo Magini
//         Created:  Thu Nov  9 17:55:31 CET 2006
// Modified by Lotte Wilke
// Last Modification: 13.02.2007
//


// system include files
#include <memory>

// user include files
// #include "FWCore/Framework/interface/Frameworkfwd.h"

// #include "FWCore/Framework/interface/Event.h"

// #include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
//
// class declaration
//
	
class HLTDisplacedmumuFilter : public HLTFilter {
	public:
		explicit HLTDisplacedmumuFilter(const edm::ParameterSet&);
		~HLTDisplacedmumuFilter();
	
	private:
		virtual void beginJob(const edm::EventSetup&) ;
		virtual bool filter(edm::Event&, const edm::EventSetup&);
		virtual void endJob() ;
		
		// ----------member data ---------------------------
// 		edm::ParameterSet conf_;
		int nevent_;
		int ntrigger_;
		double maxEta_;
		double minPt_;
		double minPtPair_;
		double minInvMass_;
		double maxInvMass_;
		int chargeOpt_;
		bool fastAccept_;
		double minLxySignificance_;
		double maxNormalisedChi2_;
		double minCosinePointingAngle_;
		edm::InputTag src_;
};
#endif
