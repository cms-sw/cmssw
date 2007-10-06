// system include files
#include <memory>
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "DataFormats/Candidate/interface/CandMatchMap.h"

#include "TFile.h"
#include "TH1.h"

#include <Math/VectorUtil.h>

using namespace std;
using namespace reco;
using namespace edm;
using namespace ROOT::Math::VectorUtil;

class jetTest : public edm::EDAnalyzer {
  public:
    explicit jetTest(const edm::ParameterSet&);
    ~jetTest() {}
     virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
     virtual void beginJob(const edm::EventSetup& iSetup) ;
     virtual void endJob() ;

  private:

    InputTag source_;
    InputTag matched_;
    InputTag matchedjetsOne_;   
    string   fOutputFileName_;            

    Handle<CandidateCollection> source;
    Handle<CandidateCollection> matched;
    Handle<CandMatchMap>        matchedjetsOne;

    TFile*      hOutputFile ;
    TH1D*       hTotalLenght;
};

jetTest::jetTest(const edm::ParameterSet& iConfig)
{
  source_          = iConfig.getParameter<InputTag> ("src");
  matched_         = iConfig.getParameter<InputTag> ("matched");
  matchedjetsOne_  = iConfig.getParameter<InputTag> ("matchMapOne");
  fOutputFileName_ = iConfig.getUntrackedParameter<string>("HistOutFile",std::string("testMatch.root"));
}

void jetTest::beginJob( const edm::EventSetup& iSetup)
{
 
   hOutputFile   = new TFile( fOutputFileName_.c_str(), "RECREATE" ) ;
   hTotalLenght  = new TH1D( "hTotalLenght", "Total Lenght", 100,  0., 5. );
   return ;
}

void jetTest::endJob()
{      
   hOutputFile->Write() ;
   hOutputFile->Close() ;  
   return ;
}

void jetTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "[GenJetTest] analysing event " << iEvent.id() << endl;
  
  try {
    iEvent.getByLabel (source_,source);
    iEvent.getByLabel (matched_,matched);
    iEvent.getByLabel (matchedjetsOne_ , matchedjetsOne );
  } catch(std::exception& ce) {
    cerr << "[GenJetTest] caught std::exception " << ce.what() << endl;
    return;
  }
  
  //
  // Printout for OneToOne matcher
  //
  double dR=-1.;
  cout << "**********************" << endl;
  cout << "* OneToOne Printout  *" << endl;
  cout << "**********************" << endl;
  for( CandMatchMap::const_iterator f  = matchedjetsOne->begin();
                                    f != matchedjetsOne->end();
                                    f++) {

      const Candidate *sourceRef = &*(f->key);
      const Candidate *matchRef  = &*(f->val);
      dR= DeltaR( sourceRef->p4() , matchRef->p4() );

      printf("[GenJetTest] (pt,eta,phi) source = %6.2f %5.2f %5.2f matched = %6.2f %5.2f %5.2f dR=%5.3f\n",
	     sourceRef->et(),
	     sourceRef->eta(),
	     sourceRef->phi(), 
	     matchRef->et(), 
	     matchRef->eta(),
	     matchRef->phi(), 
	     dR);
      
  }

  hTotalLenght->Fill( dR );
  
}

DEFINE_FWK_MODULE( matchOneToOne );
