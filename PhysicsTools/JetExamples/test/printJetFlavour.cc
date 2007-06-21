// system include files
#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <Math/VectorUtil.h>
#include <TMath.h>
#include <TFile.h>
#include <TH1.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"

#include "RecoBTag/MCTools/interface/JetFlavourIdentifier.h"

using namespace std;
using namespace reco;
using namespace edm;
using namespace JetMCTagUtils;
using namespace CandMCTagUtils;
using namespace ROOT::Math::VectorUtil;

class printJetFlavour : public edm::EDAnalyzer {
  public:
    explicit printJetFlavour(const edm::ParameterSet & );
    ~printJetFlavour() {};
    void beginJob(const edm::EventSetup& iSetup) ;
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
    void endJob() ;
     
  private:

    edm::InputTag JetPartonMap_;
    edm::InputTag theJetTagHandle_;
    string fOutputFileName_;   
    double ptMinHepMCjet;

    edm::Handle<reco::CandMatchMap>        theJetPartonMap;
    edm::Handle<reco::JetTagCollection>    theJetTagHandle;
    edm::Handle<reco::CandidateCollection> particles;
    edm::ESHandle<ParticleDataTable> pdt_;

    JetFlavourIdentifier theHepMCJetId;

    TFile*      hOutputFile ;
    TH1F*       hDeltaR;
    TH1F*       hEtCand;
    TH1F*       hEtHepMC;
    TH1F*       hDeltaFla;
    TH1F*       hDeltaFlaHepMCFlavNoZero;
};

printJetFlavour::printJetFlavour(const edm::ParameterSet& iConfig)
{
  JetPartonMap_       = iConfig.getParameter<InputTag> ("JetPartonMap"    );
  theJetTagHandle_    = iConfig.getParameter<InputTag> ("theJetTagHandle" );
  ptMinHepMCjet       = iConfig.getParameter<double>   ("ptMin");
  theHepMCJetId       = JetFlavourIdentifier(iConfig.getParameter<edm::ParameterSet>("theHepMCJetId"));
  fOutputFileName_    = iConfig.getUntrackedParameter<string>("HistOutFile",std::string("myPlots.root"));
}

void printJetFlavour::beginJob( const edm::EventSetup& iSetup)
{
 
   hOutputFile              = new TFile( fOutputFileName_.c_str(), "RECREATE" ) ;
   hDeltaR                  = new TH1F( "hDeltaR", "DeltaR", 100, 0., 0.5 );
   hEtCand                  = new TH1F( "hEtCand", "EtCand", 100, 0., 500. );  
   hEtHepMC                 = new TH1F( "hEtHepMC", "EtHepMC", 100, 0., 500. ); 
   hDeltaFla                = new TH1F( "hDeltaFla","Delta Flavour", 100, -30, 30 );
   hDeltaFlaHepMCFlavNoZero = new TH1F( "hDeltaFlaHepMCFlavNoZero","Delta Flavour HepMC Flaf != 0", 100, -30, 30 );
   return ;
}

void printJetFlavour::endJob()
{      
   hOutputFile->Write() ;
   hOutputFile->Close() ;  
   return ;
}

void printJetFlavour::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "[printJetFlavour] analysing event " << iEvent.id() << endl;
  
  try {
    iEvent.getByLabel (JetPartonMap_,theJetPartonMap);
    iEvent.getByLabel (theJetTagHandle_,theJetTagHandle);
    iEvent.getByLabel ("genParticleCandidates", particles );

    theHepMCJetId.readEvent(iEvent);

    iSetup.getData( pdt_ );

  } catch(std::exception& ce) {
    cerr << "[printFlavJet] caught std::exception " << ce.what() << endl;
    return;
  }

  cout << "*******************************" << endl;
  cout << "* the Jet Flavour (Candidate) *" << endl;
  cout << "*******************************" << endl;

  for( CandMatchMap::const_iterator f  = theJetPartonMap->begin();
                                    f != theJetPartonMap->end();
                                    f++) {

      const Candidate *theJet     = &*(f->key);
      const Candidate *theParton  = &*(f->val);

      hEtCand->Fill( theJet->et() );

      printf("[GenJetTest] (pt,eta,phi) jet = %7.3f %6.3f %6.3f | parton = %7.3f %6.3f %6.3f | %2d\n",
	     theJet->et(),
	     theJet->eta(),
	     theJet->phi(), 
	     theParton->et(), 
	     theParton->eta(),
	     theParton->phi(), 
             theParton->pdgId()
	     );
      
  }

  cout << endl;
  cout << "*******************************" << endl;
  cout << "* the Jet Flavour (HepMC)      *" << endl;
  cout << "*******************************" << endl;

  const reco::JetTagCollection & tagColl = *(theJetTagHandle.product());

  bool notMatch = false;
  for (unsigned int i = 0; i != tagColl.size(); ++i) {
    JetFlavour jetFlavour = theHepMCJetId.identifyBasedOnPartons(* tagColl[i].jet());
    if(tagColl[i].jet()->pt() < ptMinHepMCjet) continue;

    hEtHepMC->Fill( tagColl[i].jet()->et() );
    printf("[HepMC   ] (pt,eta,phi) jet = %7.3f %6.3f %6.3f | parton = %7.3f %6.3f %6.3f | %2d\n",
             tagColl[i].jet()->et(),
             tagColl[i].jet()->eta(),
             tagColl[i].jet()->phi(),
             jetFlavour.underlyingParton4Vec().Et(),
             jetFlavour.underlyingParton4Vec().eta(),
             jetFlavour.underlyingParton4Vec().phi(),
             jetFlavour.flavour() 
          );
    
    bool matched = false;
    for( CandMatchMap::const_iterator f  = theJetPartonMap->begin();
                                      f != theJetPartonMap->end();
                                      f++) {

      const Candidate *theJetInTheMatchMap = &*(f->key);    
      const Candidate *theMatchedParton    = &*(f->val);

      if( theJetInTheMatchMap->et() == tagColl[i].jet()->et() ) {
        matched = true;
        double myDist = DeltaR( theMatchedParton->p4(), jetFlavour.underlyingParton4Vec() );
        float myDeltaFla = abs( jetFlavour.flavour() ) - abs( theMatchedParton->pdgId() );
        hDeltaR->Fill(myDist);
        hDeltaFla->Fill(myDeltaFla);
        if( myDeltaFla != 0 ) {
          notMatch=true;
          if( abs(jetFlavour.flavour()) == 4 || abs(jetFlavour.flavour()) == 5 ) cout << "-------> Heavy HepMT not matched" << endl; 
          if( abs(theMatchedParton->pdgId()) == 4 || abs(theMatchedParton->pdgId()) == 5 ) cout << "------> Heavy Cand not matched" << endl;
        }
        if( jetFlavour.flavour() !=0 ) hDeltaFlaHepMCFlavNoZero->Fill(myDeltaFla);
      } 
    }
  }
}

DEFINE_FWK_MODULE( printJetFlavour );
