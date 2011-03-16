
/* \class HiggsTo4LeptonsSkimDiLeptonProducer 
 *
 * Consult header file for description
 *
 * author:  N. De Filippis - INFN and Politecnico of Bari
 *
 */


// system include files
#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsSkimDiLeptonProducer.h>
#include "DataFormats/Common/interface/Handle.h"

// User include files
#include <FWCore/ParameterSet/interface/ParameterSet.h>

// Electrons
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

// Muons
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

// C++
#include <iostream>
#include <vector>

using namespace std;
using namespace edm;
using namespace reco;


// Constructor
HiggsToZZ4LeptonsSkimDiLeptonProducer::HiggsToZZ4LeptonsSkimDiLeptonProducer(const edm::ParameterSet& pset) {

  // Collections
  RECOcollOS       = pset.getParameter<edm::InputTag>("diLeptonsOScoll");
  RECOcollSS       = pset.getParameter<edm::InputTag>("diLeptonsSScoll");
  RECOcollZMM      = pset.getParameter<edm::InputTag>("diMuonsZcoll");
  RECOcollZEE      = pset.getParameter<edm::InputTag>("diElectronsZcoll");

  // Cuts on Pt and Eta for leptons for skimming
  cutPt         = pset.getParameter<double>("cutPt");
  cutEta        = pset.getParameter<double>("cutEta");

  std::string alias;
  produces<bool> (alias = "SkimDiLeptonA" ).setBranchAlias(alias);
  produces<bool> (alias = "SkimDiLeptonB" ).setBranchAlias(alias);

}


// Destructor
HiggsToZZ4LeptonsSkimDiLeptonProducer::~HiggsToZZ4LeptonsSkimDiLeptonProducer() {

}

// Produce flags for event
void HiggsToZZ4LeptonsSkimDiLeptonProducer::produce(edm::Event& iEvent, const edm::EventSetup& setup ) {

  bool keepSkimA   = false;
  bool keepSkimB   = false;

  // DiLeptons OS
  int countOScut40=0,countOScut12=0;
  Handle<edm::View<Candidate> > CandidatesOS;
  iEvent.getByLabel(RECOcollOS.label(), CandidatesOS);
  for ( edm::View<Candidate>::const_iterator hIter=CandidatesOS->begin(); hIter!= CandidatesOS->end(); ++hIter ){
    if (hIter->mass()>40.) countOScut40++;
    if (hIter->mass()>12.) countOScut12++;
  }
  
  // DiLeptons SS
  int countSScut40=0,countSScut12=0;
  Handle<edm::View<Candidate> > CandidatesSS;
  iEvent.getByLabel(RECOcollSS.label(), CandidatesSS);
  for ( edm::View<Candidate>::const_iterator hIter=CandidatesSS->begin(); hIter!= CandidatesSS->end(); ++hIter ){
    if (hIter->mass()>40.) countSScut40++;
    if (hIter->mass()>12.) countSScut12++;
  }
  
  // Dimuons Z
  int countZMMcut40=0,countZMMcut12=0;
  Handle<edm::View<Candidate> > CandidatesZMM;
  iEvent.getByLabel(RECOcollZMM.label(), CandidatesZMM);
  for ( edm::View<Candidate>::const_iterator hIter=CandidatesZMM->begin(); hIter!= CandidatesZMM->end(); ++hIter ){
    if (hIter->mass()>40.) countZMMcut40++;
    if (hIter->mass()>12. && 
	(hIter->daughter(0)->p4().pt()>cutPt && fabs(hIter->daughter(0)->eta()<=cutEta)) &&
	(hIter->daughter(1)->p4().pt()>cutPt && fabs(hIter->daughter(1)->eta()<=cutEta))
	)
      countZMMcut12++;
  }
  
  // DiElectrons Z
  int countZEEcut40=0,countZEEcut12=0;
  Handle<edm::View<Candidate> > CandidatesZEE;
  iEvent.getByLabel(RECOcollZEE.label(), CandidatesZEE);
  for ( edm::View<Candidate>::const_iterator hIter=CandidatesZEE->begin(); hIter!= CandidatesZEE->end(); ++hIter ){
    if (hIter->mass()>40.) countZEEcut40++;
    if (hIter->mass()>12. && 
	(hIter->daughter(0)->p4().pt()>cutPt && fabs(hIter->daughter(0)->eta())<=cutEta) &&
	(hIter->daughter(1)->p4().pt()>cutPt && fabs(hIter->daughter(1)->eta())<=cutEta)
	) 
      countZEEcut12++;
  }
  
  
  
  // Make decision:
  if ( countOScut40  >=1 || countSScut40  >=1 )   keepSkimA = true;
  if ( countZMMcut12 >=1 || countZEEcut12 >=1 )   keepSkimB = true;
  
  auto_ptr<bool> flagacceptA ( new bool );
  *flagacceptA=keepSkimA;
  iEvent.put(flagacceptA,"SkimDiLeptonA");
  
  auto_ptr<bool> flagacceptB ( new bool );
  *flagacceptB=keepSkimB;
  iEvent.put(flagacceptB,"SkimDiLeptonB");
  
  //cout << keepSkimA  << " " << keepSkimB << endl;
}

void HiggsToZZ4LeptonsSkimDiLeptonProducer::endJob() {
}
