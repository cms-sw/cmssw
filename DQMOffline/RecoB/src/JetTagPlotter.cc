#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMOffline/RecoB/interface/Tools.h"

#include <iostream>

using namespace std;
using namespace RecoBTag;


JetTagPlotter::JetTagPlotter (const std::string & tagName, const EtaPtBin & etaPtBin,
		       const edm::ParameterSet& pSet, const bool& mc, const bool& update, const bool& wf) :
		       BaseBTagPlotter(tagName, etaPtBin), discrBins(400),
                       discrStart_(pSet.getParameter<double>("discriminatorStart")), 
                       discrEnd_(pSet.getParameter<double>("discriminatorEnd")),
                       nBinEffPur_(pSet.getParameter<int>("nBinEffPur")),
                       startEffPur_(pSet.getParameter<double>("startEffPur")), 
                       endEffPur_(pSet.getParameter<double>("endEffPur")), 
                       finalized(false), mcPlots_(mc), willFinalize_(wf) {

  // to have a shorter name .....
  const std::string & es = theExtensionString;
  const std::string jetTagDir("JetTag" + es);


  if (mcPlots_){
    // jet flavour
    dJetFlav = new FlavourHistograms<int>
      ("jetFlavour" + es, "Jet Flavour", 22, -0.5, 21.5,
       false, false, false, "b", update,jetTagDir, mcPlots_);
    // associated parton momentum
    dJetPartonMomentum = new FlavourHistograms<double>
      ("associatedPartonMomentum" + es, "associated parton momentum",
       200, 0.0, 400.0, false, false, true, "b", update,jetTagDir, mcPlots_);
    
    // associated parton pt
    dJetPartonPt = new FlavourHistograms<double>
      ("associatedPartonPt" + es, "associated parton pt",
       200, 0.0, 400.0, false, false, true, "b", update,jetTagDir, mcPlots_);
    
    // associated parton eta
    dJetPartonPseudoRapidity = new FlavourHistograms<double>
      ("associatedPartonEta" + es, "associated parton eta",
       100, -3.5, 3.5, false, false, true, "b", update,jetTagDir, mcPlots_);
  }else {
    dJetFlav=0;
    dJetPartonMomentum = 0;
    dJetPartonPt = 0;
    dJetPartonPseudoRapidity = 0;
  }
  // track multiplicity in jet
  dJetMultiplicity = new FlavourHistograms<int>
	("jetMultiplicity" + es, "Jet Multiplicity", 31, -0.5, 30.5,
	false, true, true, "b", update,jetTagDir, mcPlots_);

    // Discriminator: again with reasonable binning
  dDiscriminator = new FlavourHistograms<double>
	("discr" + es, "Discriminator", 50, discrStart_, discrEnd_,
	false, true, true, "b", update,jetTagDir, mcPlots_);
  dDiscriminator->settitle("Discriminant");
    // reconstructed jet momentum
  dJetRecMomentum = new FlavourHistograms<double>
	("jetMomentum" + es, "jet momentum", 200, 0.0, 200.0,
	false, false, true, "b", update,jetTagDir, mcPlots_);

  // reconstructed jet transverse momentum
  dJetRecPt = new FlavourHistograms<double>
	("jetPt" + es, "jet pt", 200, 0.0, 200.0,
	false, false, true, "b", update,jetTagDir, mcPlots_);

  // reconstructed jet eta
  dJetRecPseudoRapidity = new FlavourHistograms<double>
	("jetEta" + es, "jet eta", 100, -3.5, 3.5,
	false, false, true, "b", update,jetTagDir, mcPlots_);

  // reconstructed jet phi
  dJetRecPhi = new FlavourHistograms<double>
	("jetPhi" + es, "jet phi", 100, -3.15, 3.15,
	false, false, true, "b", update,jetTagDir, mcPlots_);

  if (willFinalize_) createPlotsForFinalize();

}  
  
  
JetTagPlotter::~JetTagPlotter () {
  delete dJetFlav;
  delete dJetMultiplicity;
  delete dDiscriminator;
  delete dJetRecMomentum;
  delete dJetRecPt;
  delete dJetRecPseudoRapidity;
  delete dJetRecPhi;
  delete dJetPartonMomentum;
  delete dJetPartonPt;
  delete dJetPartonPseudoRapidity;
  if (finalized) {
    delete effPurFromHistos;
  }
}

void JetTagPlotter::epsPlot(const std::string & name)
{
  dJetFlav->epsPlot(name);
  dJetMultiplicity->epsPlot(name);
  dDiscriminator->epsPlot(name);
  dJetRecMomentum->epsPlot(name);
  dJetRecPt->epsPlot(name);
  dJetRecPseudoRapidity->epsPlot(name);
  dJetRecPhi->epsPlot(name);
  dJetPartonMomentum->epsPlot(name);
  dJetPartonPt->epsPlot(name);
  dJetPartonPseudoRapidity->epsPlot(name);
  if (finalized) {
    effPurFromHistos->epsPlot(name);
  }
}

void JetTagPlotter::psPlot(const std::string & name)
{
  std::string cName = "JetTagPlots"+ theExtensionString;
  setTDRStyle()->cd();
  TCanvas canvas(cName.c_str(), cName.c_str(), 600, 900);
  canvas.UseCurrentStyle();

  canvas.Divide(2,3);
  canvas.Print((name + cName + ".ps[").c_str());
  canvas.cd(1);
  dJetFlav->plot();
  canvas.cd(2);
  dJetMultiplicity->plot();
  canvas.cd(3);
  dDiscriminator->plot();
  canvas.cd(4);
  dJetRecMomentum->plot();
  canvas.cd(5);
  dJetRecPt->plot();
  canvas.cd(6);
  dJetRecPseudoRapidity->plot();
  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();
  canvas.Divide(2,3);

  canvas.cd(1);
  dJetRecPhi->plot();
  canvas.cd(2);
  dJetPartonMomentum->plot();
  canvas.cd(3);
  dJetPartonPt->plot();
  canvas.cd(4);
  dJetPartonPseudoRapidity->plot();
  if (finalized) {
    canvas.cd(5);
    effPurFromHistos->discriminatorNoCutEffic()->plot();
    canvas.cd(6);
    effPurFromHistos->discriminatorCutEfficScan()->plot();
    canvas.Print((name + cName + ".ps").c_str());
    canvas.Clear();
    canvas.Divide(2,3);
    canvas.cd(1);
    effPurFromHistos->plot();
  }
  canvas.Print((name + cName + ".ps").c_str());
  canvas.Print((name + cName + ".ps]").c_str());
}


void JetTagPlotter::analyzeTag(const reco::Jet & jet,
			       const float& discriminator,
                               const int& jetFlavour)  
{

  if (mcPlots_) {
  dJetFlav->fill(jetFlavour, jetFlavour);
//   dJetPartonMomentum->fill(jetFlav, jetFlavour.underlyingParton4Vec().P() );
//   dJetPartonPt->fill(jetFlav, jetFlavour.underlyingParton4Vec().Pt() );
//   dJetPartonPseudoRapidity->fill(jetFlav, jetFlavour.underlyingParton4Vec().Eta() );
  }
  //  dJetMultiplicity->fill(jetFlavourour, jetTag.tracks().size()); //fixme
  dDiscriminator->fill(jetFlavour, discriminator );
  dJetRecMomentum->fill(jetFlavour, jet.p() );
  dJetRecPt->fill(jetFlavour, jet.pt() );
  dJetRecPseudoRapidity->fill(jetFlavour, jet.eta() );
  dJetRecPhi->fill(jetFlavour, jet.phi());
  
}


void JetTagPlotter::analyzeTag(const reco::JetTag & jetTag,
	const int & jetFlavour)
{

  if (mcPlots_) {
  dJetFlav->fill(jetFlavour, jetFlavour);
//   dJetPartonMomentum->fill(jetFlav, jetFlavour.underlyingParton4Vec().P() );
//   dJetPartonPt->fill(jetFlav, jetFlavour.underlyingParton4Vec().Pt() );
//   dJetPartonPseudoRapidity->fill(jetFlav, jetFlavour.underlyingParton4Vec().Eta() );
  }
  //  dJetMultiplicity->fill(jetFlavourour, jetTag.tracks().size()); //fixme
  dDiscriminator->fill(jetFlavour, jetTag.second );
  dJetRecMomentum->fill(jetFlavour, jetTag.first->p() );
  dJetRecPt->fill(jetFlavour, jetTag.first->pt() );
  dJetRecPseudoRapidity->fill(jetFlavour, jetTag.first->eta() );
  dJetRecPhi->fill(jetFlavour, jetTag.first->phi());
  
}




void JetTagPlotter::createPlotsForFinalize(){
    effPurFromHistos = new EffPurFromHistos ( dDiscriminator,"JetTag"+theExtensionString,mcPlots_, 
					    nBinEffPur_, startEffPur_, endEffPur_);

}

void JetTagPlotter::finalize()
{
  //
  // final processing:
  // produce the misid. vs. eff histograms
  //

  effPurFromHistos = new EffPurFromHistos ( dDiscriminator,"JetTag"+theExtensionString,mcPlots_, 
					    nBinEffPur_, startEffPur_, endEffPur_);
  effPurFromHistos->compute();
  finalized = true;
}


