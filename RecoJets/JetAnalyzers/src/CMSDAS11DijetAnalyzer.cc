// CMSDAS11DijetAnalyzer.cc
// Description: A basic dijet analyzer for the CMSDAS 2011
// Author: John Paul Chou
// Date: January 12, 2011

#include "RecoJets/JetAnalyzers/interface/CMSDAS11DijetAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include <TH1D.h>

CMSDAS11DijetAnalyzer::CMSDAS11DijetAnalyzer(edm::ParameterSet const& params) :
  edm::EDAnalyzer(),
  jetSrc(params.getParameter<edm::InputTag>("jetSrc")),
  vertexSrc(params.getParameter<edm::InputTag>("vertexSrc"))
{
  // setup file service
  edm::Service<TFileService> fs;
  
  // setup histograms
  hVertexZ = fs->make<TH1D>("hVertexZ", "Z position of the Vertex",50,-20,20);
  hJetCorrPt = fs->make<TH1D>("hJetCorrPt","Corrected Jet Pt",50,0,1000);
}

void CMSDAS11DijetAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  ////////////////////////////////////////////
  // Get event ID information
  ////////////////////////////////////////////
  int nrun=iEvent.id().run();
  int nlumi=iEvent.luminosityBlock();
  int nevent=iEvent.id().event();

  ////////////////////////////////////////////
  // Get Primary Vertex Information
  ////////////////////////////////////////////

  // magic to get the vertices from EDM
  edm::Handle< std::vector<reco::Vertex> > vertices_h;
  iEvent.getByLabel(vertexSrc, vertices_h);

  // require in the event that there is at least one reconstructed vertex
  if(vertices_h->size()<=0) return;

  // pick the first (i.e. highest sum pt) verte
  const reco::Vertex* theVertex=&(vertices_h->front());

  // require that the vertex meets certain criteria
  if(theVertex->ndof()<5) return;
  if(fabs(theVertex->z())<24.0) return;
  if(fabs(theVertex->position().rho())<2.0) return;

  ////////////////////////////////////////////
  // Get Jet Information
  ////////////////////////////////////////////

  // magic to get the jets from EDM
  edm::Handle<reco::CaloJetCollection> jets_h;
  iEvent.getByLabel(jetSrc, jets_h);

  // magic to get the jet energy corrections
  const JetCorrector* corrector = JetCorrector::getJetCorrector(jetCorrections,iSetup);

  // collection of selected jets
  std::vector<reco::CaloJet> selectedJets;

  // loop over the jet collection
  for(reco::CaloJetCollection::const_iterator j_it = jets_h->begin(); j_it!=jets_h->end(); j_it++) {
    reco::CaloJet jet = *j_it;

    // calculate and apply the correction
    double scale = corrector->correction(jet.p4());
    jet.scaleEnergy(scale);

    // select high pt, central, non-noise-like jets
    if(jet.pt()<50.0) continue;
    if(fabs(jet.eta())>2.5) continue;
    if(jet.emEnergyFraction()<0.01) continue;

    // put the selected jets into a collection
    selectedJets.push_back(jet);
  }

  // require at least two jets to continue
  if(selectedJets.size()<2) return;

  //sort by corrected pt (not the same order as raw pt, sometimes)
  sort(selectedJets.begin(), selectedJets.end(), compare_JetPt);

  //Get the mass of the two leading jets.  Needs their 4-vectors...
  float corMass = (selectedJets[0].p4()+selectedJets[1].p4()).M();
  
  ////////////////////////////////////////////
  // Make Plots
  ////////////////////////////////////////////
  
  hVertexZ->Fill(theVertex->z());
  for(unsigned int i=0; i<selectedJets.size(); i++) {
    hJetCorrPt->Fill(selectedJets[i].pt());
  }

  return;

}



DEFINE_FWK_MODULE(CMSDAS11DijetAnalyzer);
