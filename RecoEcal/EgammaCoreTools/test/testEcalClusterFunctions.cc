// -*- C++ -*-
//
// Package:    testEcalClusterFunctions
// Class:      testEcalClusterFunctions
// 
/**\class testEcalClusterFunctions testEcalClusterFunctions.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  "Federico Ferri federi
//         Created:  Mon Apr  7 14:11:00 CEST 2008
// $Id: testEcalClusterFunctions.cc,v 1.8 2008/10/30 14:17:16 sharper Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"

#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"

class testEcalClusterFunctions : public edm::EDAnalyzer {
        public:
                explicit testEcalClusterFunctions(const edm::ParameterSet&);
                ~testEcalClusterFunctions();

        private:
                virtual void beginJob(const edm::EventSetup&) ;
                virtual void analyze(const edm::Event&, const edm::EventSetup&);
                virtual void endJob() ;
                EcalClusterFunctionBaseClass *ff_;

};



testEcalClusterFunctions::testEcalClusterFunctions(const edm::ParameterSet& ps) :
        ff_(0)
{
        std::string functionName = ps.getParameter<std::string>("functionName");
        ff_ = EcalClusterFunctionFactory::get()->create( functionName, ps );
        std::cout << "got " << functionName << " function at: " << ff_ << "\n";
}



testEcalClusterFunctions::~testEcalClusterFunctions()
{
}



void testEcalClusterFunctions::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
        // init function parameters
        ff_->init( es );
        // basic test with empty collections
        reco::BasicCluster bc;
        EcalRecHitCollection rhColl;
        float corr = ff_->getValue( bc, rhColl );
        std::cout << "correction: " << corr << "\n";
}



void testEcalClusterFunctions::beginJob(const edm::EventSetup&)
{
}



void testEcalClusterFunctions::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(testEcalClusterFunctions);
