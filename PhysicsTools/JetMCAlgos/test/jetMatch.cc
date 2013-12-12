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
#include "DataFormats/Candidate/interface/CandMatchMapMany.h"

#include "TFile.h"
#include "TH1.h"

#include <Math/VectorUtil.h>

using namespace std;
using namespace reco;
using namespace edm;
using namespace ROOT::Math::VectorUtil;

class jetMatch : public edm::EDAnalyzer {
  public:
    explicit jetMatch(const edm::ParameterSet&);
    ~jetMatch() {}
     virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
     virtual void beginJob() ;
     virtual void endJob() ;

  private:

    EDGetTokenT<CandidateCollection> sourceToken_;
    EDGetTokenT<CandidateCollection> matchedToken_;
    EDGetTokenT<CandViewMatchMap> matchedjetsOneToken_;
    EDGetTokenT<CandMatchMapMany> matchedjetsManyToken_;
    string   fOutputFileName_;

    Handle<CandidateCollection> source;
    Handle<CandidateCollection> matched;
    Handle<CandViewMatchMap>        matchedjetsOne;
    Handle<CandMatchMapMany>    matchedjetsMany;

    TFile*      hOutputFile ;
    TH1D*       hTotalLenght;
};

jetMatch::jetMatch(const edm::ParameterSet& iConfig)
{
  sourceToken_          = consumes<CandidateCollection>(iConfig.getParameter<InputTag> ("src"));
  matchedToken_         = consumes<CandidateCollection>(iConfig.getParameter<InputTag> ("matched"));
  matchedjetsOneToken_  = consumes<CandViewMatchMap>(iConfig.getParameter<InputTag> ("matchMapOne"));
  matchedjetsManyToken_ = consumes<CandMatchMapMany>(iConfig.getParameter<InputTag> ("matchMapMany"));
  fOutputFileName_ = iConfig.getUntrackedParameter<string>("HistOutFile",std::string("testMatch.root"));
}

void jetMatch::beginJob()
{

   hOutputFile   = new TFile( fOutputFileName_.c_str(), "RECREATE" ) ;
   hTotalLenght  = new TH1D( "hTotalLenght", "Total Lenght", 100,  0., 5. );
   return ;
}

void jetMatch::endJob()
{
   hOutputFile->Write() ;
   hOutputFile->Close() ;
   return ;
}

void jetMatch::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "[GenJetTest] analysing event " << iEvent.id() << endl;

  try {
    iEvent.getByToken (sourceToken_,source);
    iEvent.getByToken (matchedToken_,matched);
    iEvent.getByToken (matchedjetsOneToken_ , matchedjetsOne );
    iEvent.getByToken (matchedjetsManyToken_, matchedjetsMany);
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
  for( CandViewMatchMap::const_iterator f  = matchedjetsOne->begin();
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

  //
  // Printout for OneToMany matcher
  //
  cout << "**********************" << endl;
  cout << "* OneToMany Printout *" << endl;
  cout << "**********************" << endl;
  for( CandMatchMapMany::const_iterator f  = matchedjetsMany->begin();
                                        f != matchedjetsMany->end();
                                        f++) {
    const Candidate *sourceRef = &*(f->key);

    printf("[GenJetTest] (pt,eta,phi) source = %6.2f %5.2f %5.2f\n",
	   sourceRef->et(),
	   sourceRef->eta(),
	   sourceRef->phi()  );

    const vector< pair<CandidateRef, double> > vectMatched = f->val;
    vector< pair<CandidateRef, double> >::const_iterator matchIT;
    for ( matchIT = vectMatched.begin(); matchIT != vectMatched.end(); matchIT++) {
      const Candidate *matchedRef = &*( (*matchIT).first );
      double deltaR = (*matchIT).second;
      printf("             (pt,eta,phi) matched = %6.2f %5.2f %5.2f - dR=%5.2f\n",
	     matchedRef->et(),
	     matchedRef->eta(),
	     matchedRef->phi(),
	     deltaR);

    }
  }

}

DEFINE_FWK_MODULE( jetMatch );
