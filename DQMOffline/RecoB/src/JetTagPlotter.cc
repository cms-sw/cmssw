#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <iostream>

using namespace std;
using namespace RecoBTag;


JetTagPlotter::JetTagPlotter (const std::string & tagName, const EtaPtBin & etaPtBin,
			      const edm::ParameterSet& pSet, const unsigned int& mc, 
			      const bool& update, const bool& wf, DQMStore::IBooker & ibook) :
		       BaseBTagPlotter(tagName, etaPtBin), discrBins(400),
                       discrStart_(pSet.getParameter<double>("discriminatorStart")), 
                       discrEnd_(pSet.getParameter<double>("discriminatorEnd")),
                       nBinEffPur_(pSet.getParameter<int>("nBinEffPur")),
                       startEffPur_(pSet.getParameter<double>("startEffPur")), 
                       endEffPur_(pSet.getParameter<double>("endEffPur")), 
                       finalized(false), mcPlots_(mc), willFinalize_(wf), ibook_(ibook) {

  // to have a shorter name .....
  const std::string & es = theExtensionString;
  const std::string jetTagDir(es.substr(1));

  //added to count the number of jets by event : 0=DATA or NI, 1to5=quarks u,d,s,c,b , 6=gluon
  int nFl = 1;
  if(mcPlots_) nFl = 8;
  nJets = new int [nFl];
  for(int i = 0; i < nFl; i++){
    nJets[i]=0;
  }
  
  if (mcPlots_){
    // jet flavour
    dJetFlav = new FlavourHistograms<int>
      ("jetFlavour" + es, "Jet Flavour", 22, -0.5, 21.5,
       false, false, false, "b", update,jetTagDir, mcPlots_, ibook);
    // associated parton momentum : commented, not really necessary
    //dJetPartonMomentum = new FlavourHistograms<double>
    //  ("associatedPartonMomentum" + es, "associated parton momentum",
    //   200, 0.0, 400.0, false, false, true, "b", update,jetTagDir, mcPlots_, ibook);
    
    // associated parton pt : commented, not really necessary
    //dJetPartonPt = new FlavourHistograms<double>
    //  ("associatedPartonPt" + es, "associated parton pt",
    //   200, 0.0, 400.0, false, false, true, "b", update,jetTagDir, mcPlots_, ibook);
    
    // associated parton eta : commented, not really necessary
    //dJetPartonPseudoRapidity = new FlavourHistograms<double>
    //  ("associatedPartonEta" + es, "associated parton eta",
    //   100, -3.5, 3.5, false, false, true, "b", update,jetTagDir, mcPlots_, ibook);
  }else {
    dJetFlav=0;
    //dJetPartonMomentum = 0;
    //dJetPartonPt = 0;
    //dJetPartonPseudoRapidity = 0;
  }

  // jet multiplicity
  JetMultiplicity = new FlavourHistograms<int>
    ("jetMultiplicity" + es, "Jet Multiplicity", 11, -0.5, 10.5,
     false, true, true, "b", update,jetTagDir, mcPlots_, ibook);

  // track multiplicity in jet 
  //dJetTrackMultiplicity = new FlavourHistograms<int>
  //	("jetTrackMultiplicity" + es, "Jet Track Multiplicity", 31, -0.5, 30.5,
  //	false, true, true, "b", update,jetTagDir, mcPlots_, ibook);

    // Discriminator: again with reasonable binning
  dDiscriminator = new FlavourHistograms<double>
	("discr" + es, "Discriminator", 102, discrStart_, discrEnd_,
	false, true, true, "b", update,jetTagDir, mcPlots_, ibook);
  dDiscriminator->settitle("Discriminant");
    // reconstructed jet momentum
  dJetRecMomentum = new FlavourHistograms<double>
	("jetMomentum" + es, "jet momentum", 350, 0.0, 350.0,
	false, false, true, "b", update,jetTagDir, mcPlots_, ibook);

  // reconstructed jet transverse momentum
  dJetRecPt = new FlavourHistograms<double>
	("jetPt" + es, "jet pt", 350, 0.0, 350.0,
	false, false, true, "b", update,jetTagDir, mcPlots_, ibook);

  // reconstructed jet eta
  dJetRecPseudoRapidity = new FlavourHistograms<double>
	("jetEta" + es, "jet eta", 100, -3.0, 3.0,
	false, false, true, "b", update,jetTagDir, mcPlots_, ibook);

  // reconstructed jet phi
  dJetRecPhi = new FlavourHistograms<double>
	("jetPhi" + es, "jet phi", 100, -3.15, 3.15,
	false, false, true, "b", update,jetTagDir, mcPlots_, ibook);

  if (willFinalize_) createPlotsForFinalize(ibook);

}  
  
  
JetTagPlotter::~JetTagPlotter () {
  delete dJetFlav;
  delete JetMultiplicity;
  //delete dJetTrackMultiplicity;
  delete dDiscriminator;
  delete dJetRecMomentum;
  delete dJetRecPt;
  delete dJetRecPseudoRapidity;
  delete dJetRecPhi;
  //delete dJetPartonMomentum;
  //delete dJetPartonPt;
  //delete dJetPartonPseudoRapidity;
  if (finalized) {
    delete effPurFromHistos;
  }
}

void JetTagPlotter::epsPlot(const std::string & name)
{
  dJetFlav->epsPlot(name);
  JetMultiplicity->epsPlot(name);
  //dJetTrackMultiplicity->epsPlot(name);
  dDiscriminator->epsPlot(name);
  dJetRecMomentum->epsPlot(name);
  dJetRecPt->epsPlot(name);
  dJetRecPseudoRapidity->epsPlot(name);
  dJetRecPhi->epsPlot(name);
  //dJetPartonMomentum->epsPlot(name);
  //dJetPartonPt->epsPlot(name);
  //dJetPartonPseudoRapidity->epsPlot(name);
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
  //dJetTrackMultiplicity->plot();
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

  JetMultiplicity->plot();
  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();

  canvas.cd(1);
  dJetRecPhi->plot();
  canvas.cd(2);
  //dJetPartonMomentum->plot();
  canvas.cd(3);
  //dJetPartonPt->plot();
  canvas.cd(4);
  //dJetPartonPseudoRapidity->plot();
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

void JetTagPlotter::analyzeTag() //here jetFlavour not needed
{
  //to use on data
  JetMultiplicity->fill(-1, nJets[0]);
  nJets[0] = 0; //reset to 0 before the next event
}

void JetTagPlotter::analyzeTag(const float& w)
{
  if (mcPlots_) {
  //to use with MC
    int totNJets = 0;
    int udsNJets = 0;
    int udsgNJets = 0;
    for(int i = 0; i < 8; i++){
      totNJets += nJets[i];
      if(i > 0 && i < 4) udsNJets += nJets[i];
      if((i > 0 && i < 4) || i == 6) udsgNJets += nJets[i];
      if(i <= 5 && i >= 1) JetMultiplicity->fill(i, nJets[i], w);
      else if (i==6) JetMultiplicity->fill(21, nJets[i], w);
      else if (i==7) JetMultiplicity->fill(20, nJets[i], w);
      else JetMultiplicity->fill(0, nJets[i], w);
      nJets[i] = 0; //reset to 0 before the next event
    }
    JetMultiplicity->fill(-1, totNJets, w); //total number of jets in the event
    JetMultiplicity->fill(123, udsNJets, w);
    JetMultiplicity->fill(12321, udsgNJets, w);
  }
  else 
    {
      int totNJets = 0;
      for(int i = 0; i < 8; i++){
	totNJets += nJets[i];
	nJets[i] = 0;
      }
      JetMultiplicity->fill(-1, totNJets, w);
    }
}

void JetTagPlotter::analyzeTag(const reco::Jet & jet, 
			       const double & jec,
			       const float& discriminator,
                               const int& jetFlavour)  
{
  if (mcPlots_) {
    dJetFlav->fill(jetFlavour, jetFlavour);
//   dJetPartonMomentum->fill(jetFlavour, jetFlavour.underlyingParton4Vec().P() );
//   dJetPartonPt->fill(jetFlavour, jetFlavour.underlyingParton4Vec().Pt() );
//   dJetPartonPseudoRapidity->fill(jetFlavour, jetFlavour.underlyingParton4Vec().Eta() );
    if(abs(jetFlavour)>0 && abs(jetFlavour)<6) nJets[abs(jetFlavour)]+=1; //quarks 1 to 5
    else if(abs(jetFlavour)==21) nJets[6]+=1; //gluons
    else if(jetFlavour==20) nJets[7]+=1; //PU
    else nJets[0]+=1; //NI
  }
  else{
    nJets[0]+=1;
  }

  //  dJetTrackMultiplicity->fill(jetFlavour, jetTag.tracks().size()); //fixme
  if (edm::isNotFinite(discriminator) ) dDiscriminator->fill(jetFlavour, -999.0 );
  else dDiscriminator->fill(jetFlavour, discriminator );
  dJetRecMomentum->fill(jetFlavour, jet.p()*jec );
  dJetRecPt->fill(jetFlavour, jet.pt()*jec );
  dJetRecPseudoRapidity->fill(jetFlavour, jet.eta() );
  dJetRecPhi->fill(jetFlavour, jet.phi());
}

void JetTagPlotter::analyzeTag(const reco::Jet & jet, 
			       const double & jec,
			       const float& discriminator,
                               const int& jetFlavour,
			       const float& w)  
{
  if (mcPlots_) {
    dJetFlav->fill(jetFlavour, jetFlavour , w );
//   dJetPartonMomentum->fill(jetFlavour, jetFlavour.underlyingParton4Vec().P() );
//   dJetPartonPt->fill(jetFlavour, jetFlavour.underlyingParton4Vec().Pt() );
//   dJetPartonPseudoRapidity->fill(jetFlavour, jetFlavour.underlyingParton4Vec().Eta() );
    if(abs(jetFlavour)>0 && abs(jetFlavour)<6) nJets[abs(jetFlavour)]+=1; //quarks 1 to 5
    else if(abs(jetFlavour)==21) nJets[6]+=1; //gluons
    else if(jetFlavour==20) nJets[7]+=1; //PU
    else nJets[0]+=1; //NI
  }
  else{
    nJets[0]+=1;
  }

  //  dJetTrackMultiplicity->fill(jetFlavour, jetTag.tracks().size()); //fixme
  if (edm::isNotFinite(discriminator) ) dDiscriminator->fill(jetFlavour, -999.0 , w );
  else dDiscriminator->fill(jetFlavour, discriminator , w );
  dJetRecMomentum->fill(jetFlavour, jet.p()*jec , w);
  dJetRecPt->fill(jetFlavour, jet.pt()*jec , w);
  dJetRecPseudoRapidity->fill(jetFlavour, jet.eta() , w );
  dJetRecPhi->fill(jetFlavour, jet.phi() , w );
}


void JetTagPlotter::analyzeTag(const reco::JetTag & jetTag, 
			       const double & jec,
			       const int & jetFlavour)
{
  if (mcPlots_) {
  dJetFlav->fill(jetFlavour, jetFlavour);
//   dJetPartonMomentum->fill(jetFlavour, jetFlavour.underlyingParton4Vec().P() );
//   dJetPartonPt->fill(jetFlavour, jetFlavour.underlyingParton4Vec().Pt() );
//   dJetPartonPseudoRapidity->fill(jetFlavour, jetFlavour.underlyingParton4Vec().Eta() );
  if(abs(jetFlavour)>0 && abs(jetFlavour)<6) nJets[abs(jetFlavour)]+=1; //quarks 1 to 5
  else if(abs(jetFlavour)==21) nJets[6]+=1; //gluons
  else if(jetFlavour==20) nJets[7]+=1; //PU  
  else nJets[0]+=1; //NI
  }
  else{
    nJets[0]+=1;
  }

  //  dJetTrackMultiplicity->fill(jetFlavour, jetTag.tracks().size()); //fixme
  if (edm::isNotFinite(jetTag.second) ) dDiscriminator->fill(jetFlavour, -999.0 );
  else dDiscriminator->fill(jetFlavour, jetTag.second);
  dJetRecMomentum->fill(jetFlavour, jetTag.first->p()*jec );
  dJetRecPt->fill(jetFlavour, jetTag.first->pt()*jec );
  dJetRecPseudoRapidity->fill(jetFlavour, jetTag.first->eta() );
  dJetRecPhi->fill(jetFlavour, jetTag.first->phi());
}

void JetTagPlotter::analyzeTag(const reco::JetTag & jetTag, 
			       const double & jec,
			       const int & jetFlavour,
			       const float& w)
{
  if (mcPlots_) {
    dJetFlav->fill(jetFlavour, jetFlavour, w );
//   dJetPartonMomentum->fill(jetFlavour, jetFlavour.underlyingParton4Vec().P() );
//   dJetPartonPt->fill(jetFlavour, jetFlavour.underlyingParton4Vec().Pt() );
//   dJetPartonPseudoRapidity->fill(jetFlavour, jetFlavour.underlyingParton4Vec().Eta() );
    if(abs(jetFlavour)>0 && abs(jetFlavour)<6) nJets[abs(jetFlavour)]+=1; //quarks 1 to 5
    else if(abs(jetFlavour)==21) nJets[6]+=1; //gluons
    else if(jetFlavour==20) nJets[7]+=1; //PU  
    else nJets[0]+=1; //NI
  }
  else{
    nJets[0]+=1;
  }

  //  dJetTrackMultiplicity->fill(jetFlavour, jetTag.tracks().size()); //fixme
  if (edm::isNotFinite(jetTag.second) ) dDiscriminator->fill(jetFlavour, -999.0 , w );
  else dDiscriminator->fill(jetFlavour, jetTag.second , w );
  dJetRecMomentum->fill(jetFlavour, jetTag.first->p()*jec , w );
  dJetRecPt->fill(jetFlavour, jetTag.first->pt()*jec , w );
  dJetRecPseudoRapidity->fill(jetFlavour, jetTag.first->eta() , w );
  dJetRecPhi->fill(jetFlavour, jetTag.first->phi() , w );
}


void JetTagPlotter::createPlotsForFinalize(DQMStore::IBooker & ibook){
  effPurFromHistos = new EffPurFromHistos ( dDiscriminator,theExtensionString.substr(1),mcPlots_, ibook, 
					    nBinEffPur_, startEffPur_, endEffPur_);

}

void JetTagPlotter::finalize()
{
  //
  // final processing:
  // produce the misid. vs. eff histograms
  //
  
  effPurFromHistos = new EffPurFromHistos ( dDiscriminator,theExtensionString.substr(1), mcPlots_, ibook_, 
					    nBinEffPur_, startEffPur_, endEffPur_);
  effPurFromHistos->compute(ibook_);
  finalized = true;
}


