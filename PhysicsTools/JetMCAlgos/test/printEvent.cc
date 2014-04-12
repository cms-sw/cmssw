// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"

class printEvent : public edm::EDAnalyzer {
  public:
    explicit printEvent(const edm::ParameterSet & );
    ~printEvent() {};
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  private:

    edm::EDGetTokenT< edm::View<reco::Jet> > sourceToken_;
    edm::Handle< edm::View<reco::Jet> > genJets;
    edm::ESHandle<ParticleDataTable> pdt_;

};

// system include files
#include <memory>
#include <string>
#include <iostream>
#include <vector>

using namespace std;
using namespace reco;
using namespace edm;
using namespace JetMCTagUtils;
using namespace CandMCTagUtils;

printEvent::printEvent(const edm::ParameterSet& iConfig)
{
  sourceToken_ = consumes< edm::View<reco::Jet> >(iConfig.getParameter<InputTag> ("src"));
}

void printEvent::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  typedef edm::RefToBase<reco::Jet> JetRef;

  cout << "[printGenJet] analysing event " << iEvent.id() << endl;

  try {
    iEvent.getByToken (sourceToken_,genJets);
    iSetup.getData( pdt_ );
  } catch(std::exception& ce) {
    cerr << "[printGenJet] caught std::exception " << ce.what() << endl;
    return;
  }

  cout << endl;
  cout << "**********************" << endl;
  cout << "* GenJetCollection   *" << endl;
  cout << "**********************" << endl;

  for( View<Jet>::const_iterator f  = genJets->begin();
                                 f != genJets->end();
                                 f++) {

    double bRatio = EnergyRatioFromBHadrons( *f );
    double cRatio = EnergyRatioFromCHadrons( *f );

    printf("[GenJetTest] (pt,eta,phi | bRatio cRatio) = %6.2f %5.2f %5.2f | %5.3f %5.3f |\n",
	     f->pt(),
	     f->eta(),
	     f->phi(),
             bRatio,
             cRatio  );

    for( Candidate::const_iterator c  = f->begin();
                                   c != f->end();
                                   c ++) {
      bool isB = false;
      bool isC = false;
      isB = decayFromBHadron(*c);
      isC = decayFromCHadron(*c);
      printf("        [Constituents] (pt,eta,phi | isB,isC) = %6.2f %5.2f %5.2f | %1d %1d |\n",
                c->et(),
                c->eta(),
                c->phi(),
                isB,isC  );
    }
  }
}

DEFINE_FWK_MODULE( printEvent );
