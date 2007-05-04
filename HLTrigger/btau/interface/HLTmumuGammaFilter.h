#ifndef HLTmumuGammaFilter_h
#define HLTmumuGammaFilter_h
//
// Package:    HLTstaging
// Class:     HLTmumuGammaFilter
// 
/*
HLT Filter for Bs to mumuGamma
 Implementation:
     <Notes on implementation>
*/
//
// modified by Arun

// system include files
#include <memory>
#include <string>
#include <iostream>
// user include files
// #include "FWCore/Framework/interface/Frameworkfwd.h"

// #include "FWCore/Framework/interface/Event.h"

// #include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//
using namespace edm;
using namespace std;

	
class HLTmumuGammaFilter : public HLTFilter {
	public:
		explicit HLTmumuGammaFilter(const edm::ParameterSet&);
		~HLTmumuGammaFilter();
	
	private:
		virtual void beginJob(const edm::EventSetup&) ;
		virtual bool filter(edm::Event&, const edm::EventSetup&);
		virtual void endJob() ;
		
		// ----------member data ---------------------------
// 		edm::ParameterSet conf_;
		int nevent_;
		int ntrigger_;
		edm::InputTag m_vertexSrc;
		edm::InputTag CandSrc_;
	        double deltaRCut, ClusPtMin, minInvMass, maxInvMass;
		
};
#endif
