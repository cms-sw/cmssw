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

class matchOneToOne : public edm::EDAnalyzer {
  public:
    explicit matchOneToOne(const edm::ParameterSet&);
    ~matchOneToOne() {}
     virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
     virtual void beginJob() ;
     virtual void endJob() ;

  private:

    EDGetTokenT<CandidateCollection> sourceToken_;
    EDGetTokenT<CandidateCollection> matchedToken_;
    EDGetTokenT<CandViewMatchMap> matchedjetsOne1Token_;
    EDGetTokenT<CandViewMatchMap> matchedjetsOne2Token_;
    string   fOutputFileName_;

    Handle<CandidateCollection> source;
    Handle<CandidateCollection> matched;
    Handle<CandViewMatchMap>        matchedjetsOne1;
    Handle<CandViewMatchMap>        matchedjetsOne2;

    TFile*      hOutputFile ;
    TH1D*       hTotalLenght;
    TH1F*       hDR;
};

matchOneToOne::matchOneToOne(const edm::ParameterSet& iConfig)
{
  sourceToken_          = consumes<CandidateCollection>(iConfig.getParameter<InputTag> ("src"));
  matchedToken_         = consumes<CandidateCollection>(iConfig.getParameter<InputTag> ("matched"));
  matchedjetsOne1Token_ = consumes<CandViewMatchMap>(iConfig.getParameter<InputTag> ("matchMapOne1"));
  matchedjetsOne2Token_ = consumes<CandViewMatchMap>(iConfig.getParameter<InputTag> ("matchMapOne2"));
  fOutputFileName_ = iConfig.getUntrackedParameter<string>("HistOutFile",std::string("testMatch.root"));
}

void matchOneToOne::beginJob()
{

   hOutputFile   = new TFile( fOutputFileName_.c_str(), "RECREATE" ) ;
   hDR           = new TH1F( "hDR","",1000,0.,10.);
   hTotalLenght  = new TH1D( "hTotalLenght", "Total Lenght", 1000,  0., 5. );
   return ;
}

void matchOneToOne::endJob()
{
   hOutputFile->Write() ;
   hOutputFile->Close() ;
   return ;
}

void matchOneToOne::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "[matchOneToOne] analysing event " << iEvent.id() << endl;

  try {
    iEvent.getByToken (sourceToken_,source);
    iEvent.getByToken (matchedToken_,matched);
    iEvent.getByToken (matchedjetsOne1Token_ , matchedjetsOne1 );
    iEvent.getByToken (matchedjetsOne2Token_ , matchedjetsOne2 );   } catch(std::exception& ce) {
    cerr << "[matchOneToOne] caught std::exception " << ce.what() << endl;
    return;
  }

  //
  // Printout for OneToOne matcher
  //
  double dR=-1.;
  float totalLenght=0;
  cout << "**********************" << endl;
  cout << "* OneToOne Printout  *" << endl;
  cout << "**********************" << endl;
  for( CandViewMatchMap::const_iterator f  = matchedjetsOne1->begin();
                                        f != matchedjetsOne1->end();
                                        f++) {

      const Candidate *sourceRef = &*(f->key);
      const Candidate *matchRef  = &*(f->val);
      dR= DeltaR( sourceRef->p4() , matchRef->p4() );
      totalLenght+=dR;

      printf("[matchOneToOne src2mtc] (pt,eta,phi) source = %6.2f %5.2f %5.2f matched = %6.2f %5.2f %5.2f dR=%5.3f\n",
	     sourceRef->et(),
	     sourceRef->eta(),
	     sourceRef->phi(),
	     matchRef->et(),
	     matchRef->eta(),
	     matchRef->phi(),
	     dR);

      hDR->Fill(dR);

  }

  cout << "-----------------" << endl;

  for( CandViewMatchMap::const_iterator f  = matchedjetsOne2->begin();
                                        f != matchedjetsOne2->end();
                                        f++) {

      const Candidate *sourceRef = &*(f->key);
      const Candidate *matchRef  = &*(f->val);
      dR= DeltaR( sourceRef->p4() , matchRef->p4() );
      printf("[matchOneToOne mtc2src] (pt,eta,phi) source = %6.2f %5.2f %5.2f matched = %6.2f %5.2f %5.2f dR=%5.3f\n",
             sourceRef->et(),
             sourceRef->eta(),
             sourceRef->phi(),
             matchRef->et(),
             matchRef->eta(),
             matchRef->phi(),
             dR);
  }


  hTotalLenght->Fill( totalLenght );
}

DEFINE_FWK_MODULE( matchOneToOne );
