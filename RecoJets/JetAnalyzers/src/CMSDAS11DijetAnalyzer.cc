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
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include <TH1D.h>

CMSDAS11DijetAnalyzer::CMSDAS11DijetAnalyzer(edm::ParameterSet const& params) :
  edm::EDAnalyzer(),
  jetSrc(params.getParameter<edm::InputTag>("jetSrc")),
  vertexSrc(params.getParameter<edm::InputTag>("vertexSrc")),
  jetCorrections(params.getParameter<std::string>("jetCorrections")),
  innerDeltaEta(params.getParameter<double>("innerDeltaEta")),
  outerDeltaEta(params.getParameter<double>("outerDeltaEta")),
  JESbias(params.getParameter<double>("JESbias"))
{
  // setup file service
  edm::Service<TFileService> fs;

  const int NBINS=36;
  Double_t BOUNDARIES[NBINS] = { 220, 244, 270, 296, 325, 354, 386, 419, 453,
				 489, 526, 565, 606, 649, 693, 740, 788, 838,
				 890, 944, 1000, 1058, 1118, 1181, 1246, 1313, 1383,
				 1455, 1530, 1607, 1687, 1770, 1856, 1945, 2037, 2132 };
  
  // setup histograms
  hVertexZ = fs->make<TH1D>("hVertexZ", "Z position of the Vertex",50,-20,20);
  hJetRawPt    = fs->make<TH1D>("hJetRawPt","Raw Jet Pt",50,0,1000);
  hJetCorrPt = fs->make<TH1D>("hJetCorrPt","Corrected Jet Pt",50,0,1000);
  hJet1Pt      = fs->make<TH1D>("hJet1Pt","Corrected Jet1 Pt",50,0,1000);
  hJet2Pt      = fs->make<TH1D>("hJet2Pt","Corrected Jet2 Pt",50,0,1000);

  hJetEta      = fs->make<TH1D>("hJetEta","Corrected Jet Eta",   10,-5,5);
  hJet1Eta      = fs->make<TH1D>("hJet1Eta","Corrected Jet1 Eta",10,-5,5);
  hJet2Eta      = fs->make<TH1D>("hJet2Eta","Corrected Jet2 Eta",10,-5,5);

  hJetPhi      = fs->make<TH1D>("hJetPhi","Corrected Jet Phi",   10,-3.1415,3.1415);
  hJet1Phi      = fs->make<TH1D>("hJet1Phi","Corrected Jet1 Phi",10,-3.1415,3.1415);
  hJet2Phi      = fs->make<TH1D>("hJet2Phi","Corrected Jet2 Phi",10,-3.1415,3.1415);

  hJetEMF	= fs->make<TH1D>("hJetEMF","EM Fraction of Jets",50,0,1);
  hJet1EMF	= fs->make<TH1D>("hJet1EMF","EM Fraction of Jet1",50,0,1);
  hJet2EMF	= fs->make<TH1D>("hJet2EMF","EM Fraction of Jet2",50,0,1);

  hCorDijetMass = fs->make<TH1D>("hCorDijetMass","Corrected Dijet Mass",NBINS-1,BOUNDARIES);
  hDijetDeltaPhi= fs->make<TH1D>("hDijetDeltaPhi","Dijet |#Delta #phi|",50,0,3.1415);
  hDijetDeltaEta= fs->make<TH1D>("hDijetDeltaEta","Dijet |#Delta #eta|",50,0,6);

  hInnerDijetMass = fs->make<TH1D>("hInnerDijetMass","Corrected Inner Dijet Mass",NBINS-1,BOUNDARIES);
  hOuterDijetMass = fs->make<TH1D>("hOuterDijetMass","Corrected Outer Dijet Mass",NBINS-1,BOUNDARIES);
}

void CMSDAS11DijetAnalyzer::endJob(void){
}


void CMSDAS11DijetAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  ////////Get Weight, if this is MC//////////////  
  double mWeight;
  edm::Handle<GenEventInfoProduct> hEventInfo;
  iEvent.getByLabel("generator", hEventInfo);
  if (hEventInfo.isValid())
    mWeight = hEventInfo->weight();
  else 
    mWeight = 1.;

  ////////////////////////////////////////////
  // Get event ID information
  ////////////////////////////////////////////
  //int nrun=iEvent.id().run();
  //int nlumi=iEvent.luminosityBlock();
  //int nevent=iEvent.id().event();

  ////////////////////////////////////////////
  // Get Primary Vertex Information
  ////////////////////////////////////////////

  // magic to get the vertices from EDM
  edm::Handle< std::vector<reco::Vertex> > vertices_h;
  iEvent.getByLabel(vertexSrc, vertices_h);
  if (!vertices_h.isValid()) {
    std::cout<<"Didja hear the one about the empty vertex collection?\n";
    return;
  }
  
  // require in the event that there is at least one reconstructed vertex
  if(vertices_h->size()<=0) return;

  // pick the first (i.e. highest sum pt) verte
  const reco::Vertex* theVertex=&(vertices_h->front());

  // require that the vertex meets certain criteria
  if(theVertex->ndof()<5) return;
  if(fabs(theVertex->z())>24.0) return;
  if(fabs(theVertex->position().rho())>2.0) return;

  ////////////////////////////////////////////
  // Get Jet Information
  ////////////////////////////////////////////

  // magic to get the jets from EDM
  edm::Handle<reco::CaloJetCollection> jets_h;
  iEvent.getByLabel(jetSrc, jets_h);
  if (!jets_h.isValid()) {
    std::cout<<"Didja hear the one about the empty jet collection?\n";
    return;
  }

  // magic to get the jet energy corrections
  const JetCorrector* corrector = JetCorrector::getJetCorrector(jetCorrections,iSetup);

  // collection of selected jets
  std::vector<reco::CaloJet> selectedJets;

  // loop over the jet collection
  for(reco::CaloJetCollection::const_iterator j_it = jets_h->begin(); j_it!=jets_h->end(); j_it++) {
    reco::CaloJet jet = *j_it;

    // calculate and apply the correction
    double scale = corrector->correction(jet.p4());

    // Introduce a purposeful bias to the correction, to show what happens
    scale *= JESbias;

    // fill the histograms
    hJetRawPt->Fill(jet.pt(),mWeight);
    hJetEta->Fill(jet.eta(),mWeight);
    hJetPhi->Fill(jet.phi(),mWeight);
    hJetEMF->Fill(jet.emEnergyFraction(),mWeight);

    // correct the jet energy
    jet.scaleEnergy(scale);
    
    // now fill the correct jet pt after the correction
    hJetCorrPt->Fill(jet.pt(),mWeight);

    // put the selected jets into a collection
    selectedJets.push_back(jet);
  }

  // require at least two jets to continue
  if(selectedJets.size()<2) return;

  //sort by corrected pt (not the same order as raw pt, sometimes)
  sort(selectedJets.begin(), selectedJets.end(), compare_JetPt);

  // select high pt, central, non-noise-like jets
  if (selectedJets[0].pt()<50.0) return;
  if (fabs(selectedJets[0].eta())>2.5) return;
  if (selectedJets[0].emEnergyFraction()<0.01) return;
  if (selectedJets[1].pt()<50.0) return;
  if (fabs(selectedJets[1].eta())>2.5) return;
  if (selectedJets[1].emEnergyFraction()<0.01) return;
  

  // fill histograms for the jets in our dijets, only
  hJet1Pt ->Fill(selectedJets[0].pt(),mWeight);
  hJet1Eta->Fill(selectedJets[0].eta(),mWeight);
  hJet1Phi->Fill(selectedJets[0].phi(),mWeight);
  hJet1EMF->Fill(selectedJets[0].emEnergyFraction(),mWeight);
  hJet2Pt ->Fill(selectedJets[1].pt(),mWeight);
  hJet2Eta->Fill(selectedJets[1].eta(),mWeight);
  hJet2Phi->Fill(selectedJets[1].phi(),mWeight);
  hJet2EMF->Fill(selectedJets[1].emEnergyFraction(),mWeight);

  //Get the mass of the two leading jets.  Needs their 4-vectors...
  double corMass = (selectedJets[0].p4()+selectedJets[1].p4()).M();
  double deltaEta = fabs(selectedJets[0].eta()-selectedJets[1].eta());
  if (corMass < 489) return;
  if (deltaEta > 1.3) return;
  hCorDijetMass->Fill(corMass,mWeight);
  hDijetDeltaPhi->Fill(fabs(selectedJets[0].phi()-selectedJets[1].phi()) ,mWeight);
  hDijetDeltaEta->Fill(deltaEta ,mWeight);

  //Fill the inner and outer dijet mass spectra, to make the ratio from
  if (deltaEta < innerDeltaEta) hInnerDijetMass->Fill(corMass,mWeight);
  else if (deltaEta < outerDeltaEta) hOuterDijetMass->Fill(corMass,mWeight);

  hVertexZ->Fill(theVertex->z(),mWeight);
  return;
}


DEFINE_FWK_MODULE(CMSDAS11DijetAnalyzer);
