/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/06/18 15:32:55 $
 *  $Revision: 1.1 $
 *  \author S. Bolognesi, Eric - CERN
 */

#include "DQM/Physics/src/QcdHighPtDQM.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/METReco/interface/METCollection.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

#include <string>
#include <cmath>
using namespace std;
using namespace edm;
using namespace reco;
using namespace math;


//Get Jets and MET (no MET plots yet pending converging w/JetMET group)

QcdHighPtDQM::QcdHighPtDQM(const ParameterSet& iConfig): 
  jetLabel_(iConfig.getUntrackedParameter<edm::InputTag>("jetTag")),
  metLabel_(iConfig.getUntrackedParameter<edm::InputTag>("metTag"))
{

}

QcdHighPtDQM::~QcdHighPtDQM() { 
  

}


void QcdHighPtDQM::beginJob(EventSetup const& iSetup) {
 
  theDbe = Service<DQMStore>().operator->();

  //Book MEs
 
  theDbe->setCurrentFolder("Physics/QcdHighPt");  

  MEcontainer_["dijet_mass"] = theDbe->book1D("dijet_mass", "dijet resonance invariant mass", 100, 0, 1000);
  MEcontainer_["njets"] = theDbe->book1D("njets", "jet multiplicity", 10, 0, 10);
  MEcontainer_["etaphi"] = theDbe->book2D("etaphi", "eta/phi distribution", 83, -42, 42, 72, -M_PI, M_PI);
  MEcontainer_["njets30"] = theDbe->book1D("njets30", "jet multiplicity, pt > 30 GeV", 10, 0, 10);
  MEcontainer_["leading_jet_pt"] = theDbe->book1D("leading_jet_pt", "leading jet Pt", 100, 0, 1000);


}

void QcdHighPtDQM::endJob(void) {

}


void QcdHighPtDQM::analyze(const Event& iEvent, const EventSetup& iSetup) {
  
  //Get Jets
  edm::Handle<CaloJetCollection> jetHandle;
  iEvent.getByLabel(jetLabel_,jetHandle);
  const CaloJetCollection & jets = *jetHandle;
  CaloJetCollection::const_iterator jet_iter;


  int njets = 0;
  int njets30 = 0;
  float leading_jetpt = 0;

  //get bins in eta.  
  //Bins correspond to calotower regions.
  const float etabins[83] = {-5.191, -4.889, -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489, -3.314, -3.139, -2.964, -2.853, -2.650, -2.500, -2.322, -2.172, -2.043, -1.930, -1.830, -1.740, -1.653, -1.566, -1.479, -1.392, -1.305, -1.218, -1.131, -1.044, -.957, -.879, -.783, -.696, -.609, -.522, -.435, -.348, -.261, -.174, -.087, 0, .087, .174, .261, .348, .435, .522, .609, .696, .783, .879, .957, 1.044, 1.131, 1.218, 1.305, 1.392, 1.479, 1.566, 1.653, 1.740, 1.830, 1.930, 2.043, 2.172, 2.322, 2.500, 2.650, 2.853, 2.964, 3.139, 3.314, 3.489, 3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.889, 5.191};

  for(jet_iter = jets.begin(); jet_iter!= jets.end(); ++jet_iter){
    njets++;
  
    //get Jet stats
    float jet_pt = jet_iter->pt();
    float jet_eta = jet_iter->eta();
    float jet_phi = jet_iter->phi();

    if((jet_pt) > 30) njets30++;
    if(jet_pt > leading_jetpt){
      leading_jetpt = jet_pt;}

    //fill eta-phi plot
    for (int eit = 0; eit < 81; eit++){
      for(int pit = 0; pit < 72; pit++){
	float low_eta = etabins[eit];
	float high_eta = etabins[eit+1];
	float low_phi = (-M_PI) + pit*(M_PI/36);
	float high_phi = low_phi + (M_PI/36);
	if(jet_eta > low_eta && jet_eta < high_eta && jet_phi > low_phi && jet_phi < high_phi){
	  MEcontainer_["etaphi"]->Fill((eit - 41), jet_phi);}


      }
    }
  }
  MEcontainer_["leading_jet_pt"]->Fill(leading_jetpt);
  MEcontainer_["njets"]->Fill(njets);
  MEcontainer_["njets30"]->Fill(njets30);
  
  //if 2 or more non-trivial jet objects, reconstruct dijet resonance
  if(jets.size() >= 2){
    if(jets[0].energy() > 0 && jets[1].energy() > 0){
      math::XYZTLorentzVector DiJet = jets[0].p4() + jets[1].p4();
      float dijet_mass = DiJet.mass();
      MEcontainer_["dijet_mass"]->Fill(dijet_mass);
    }
    
  }


}











