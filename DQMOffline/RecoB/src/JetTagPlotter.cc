#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <iostream>

using namespace std;
using namespace RecoBTag;


JetTagPlotter::JetTagPlotter (const std::string & tagName, const EtaPtBin & etaPtBin,
                  const edm::ParameterSet& pSet, unsigned int mc, 
                  bool wf, DQMStore::IBooker & ibook, bool doCTagPlots/*=false*/, 
                  bool doDifferentialPlots/*=false*/, double discrCut/*=-999.*/) :
                       BaseBTagPlotter(tagName, etaPtBin),
                       discrStart_(pSet.getParameter<double>("discriminatorStart")), 
                       discrEnd_(pSet.getParameter<double>("discriminatorEnd")),
                       nBinEffPur_(pSet.getParameter<int>("nBinEffPur")),
                       startEffPur_(pSet.getParameter<double>("startEffPur")), 
                       endEffPur_(pSet.getParameter<double>("endEffPur")),
                       mcPlots_(mc), willFinalize_(wf), doCTagPlots_(doCTagPlots), 
                       doDifferentialPlots_(doDifferentialPlots),
                       cutValue_(discrCut) {

  // to have a shorter name .....
  const std::string & es = theExtensionString;
  const std::string jetTagDir(es.substr(1));
    
  if (willFinalize_) return;

  //added to count the number of jets by event : 0=DATA or NI, 1to5=quarks u,d,s,c,b , 6=gluon
  int nFl = 1;
  if (mcPlots_) nFl = 8;
  nJets_.resize(nFl, 0);

  if (mcPlots_) {
    // jet flavour
    dJetFlav_ = std::make_unique<FlavourHistograms<int>>
      ("jetFlavour" + es, "Jet Flavour", 22, -0.5, 21.5,
       false, false, false, "b", jetTagDir, mcPlots_, ibook);
  }
  
  // jet multiplicity
  jetMultiplicity_ = std::make_unique<FlavourHistograms<int>>
    ("jetMultiplicity" + es, "Jet Multiplicity", 11, -0.5, 10.5,
     false, true, true, "b", jetTagDir, mcPlots_, ibook);
  
  // Discriminator: again with reasonable binning
  dDiscriminator_ = std::make_unique<FlavourHistograms<double>>
    ("discr" + es, "Discriminator", 102, discrStart_, discrEnd_,
     false, true, true, "b", jetTagDir, mcPlots_, ibook);
  dDiscriminator_->settitle("Discriminant");
  // reconstructed jet momentum
  dJetRecMomentum_ = std::make_unique<FlavourHistograms<double>>
    ("jetMomentum" + es, "jet momentum", 350, 0.0, 350.0,
     false, false, true, "b", jetTagDir, mcPlots_, ibook);
  
  // reconstructed jet transverse momentum
  dJetRecPt_ = std::make_unique<FlavourHistograms<double>>
    ("jetPt" + es, "jet pt", 350, 0.0, 350.0,
     false, false, true, "b", jetTagDir, mcPlots_, ibook);
  
  // reconstructed jet eta
  dJetRecPseudoRapidity_ = std::make_unique<FlavourHistograms<double>>
    ("jetEta" + es, "jet eta", 20, -etaPtBin.getEtaMax(), etaPtBin.getEtaMax(),
     false, false, true, "b", jetTagDir, mcPlots_, ibook);
  
  // reconstructed jet phi
  dJetRecPhi_ = std::make_unique<FlavourHistograms<double>>
    ("jetPhi" + es, "jet phi", 20, -M_PI, M_PI,
     false, false, true, "b", jetTagDir, mcPlots_, ibook); 
 
  if (doDifferentialPlots_) {
      // jet Phi larger than requested discrimnator cut
      dJetPhiDiscrCut_ = std::make_unique<FlavourHistograms<double>>("jetPhi_diffEff" + es, "Efficiency vs. jet Phi for discriminator above cut",
            20, -M_PI, M_PI, false, false, true, "b", jetTagDir, mcPlots_, ibook);
      
      // jet Eta larger than requested discrimnator cut
      dJetPseudoRapidityDiscrCut_ = std::make_unique<FlavourHistograms<double>>("jetEta_diffEff" + es, "Efficiency vs. jet eta for discriminator above cut",
            20, -etaPtBin.getEtaMax(), etaPtBin.getEtaMax(), false, false, true, "b", jetTagDir, mcPlots_, ibook);
  }
}  
  
  
JetTagPlotter::~JetTagPlotter() {}

void JetTagPlotter::epsPlot(const std::string & name)
{
  if (!willFinalize_) {
    dJetFlav_->epsPlot(name);
    jetMultiplicity_->epsPlot(name);
    dDiscriminator_->epsPlot(name);
    dJetRecMomentum_->epsPlot(name);
    dJetRecPt_->epsPlot(name);
    dJetRecPseudoRapidity_->epsPlot(name);
    dJetRecPhi_->epsPlot(name);
  }
  else {
    effPurFromHistos_->epsPlot(name);
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
  if (!willFinalize_) {
    canvas.cd(1);
    dJetFlav_->plot();
    canvas.cd(2);
    canvas.cd(3);
    dDiscriminator_->plot();
    canvas.cd(4);
    dJetRecMomentum_->plot();
    canvas.cd(5);
    dJetRecPt_->plot();
    canvas.cd(6);
    dJetRecPseudoRapidity_->plot();
    canvas.Print((name + cName + ".ps").c_str());
    canvas.Clear();
    canvas.Divide(2,3);
    
    jetMultiplicity_->plot();
    canvas.Print((name + cName + ".ps").c_str());
    canvas.Clear();
    
    canvas.cd(1);
    dJetRecPhi_->plot();
    canvas.cd(2);
    canvas.cd(3);
    canvas.cd(4);
  } else {
    canvas.cd(5);
    effPurFromHistos_->discriminatorNoCutEffic().plot();
    canvas.cd(6);
    effPurFromHistos_->discriminatorCutEfficScan().plot();
    canvas.Print((name + cName + ".ps").c_str());
    canvas.Clear();
    canvas.Divide(2,3);
    canvas.cd(1);
    effPurFromHistos_->plot();
  }
  canvas.Print((name + cName + ".ps").c_str());
  canvas.Print((name + cName + ".ps]").c_str());
}

void JetTagPlotter::analyzeTag() //here jetFlavour not needed
{
  //to use on data
  jetMultiplicity_->fill(-1, nJets_[0]);
  nJets_[0] = 0; //reset to 0 before the next event
}

void JetTagPlotter::analyzeTag(float w)
{
  if (mcPlots_) {
  //to use with MC
    int totNJets = 0;
    int udsNJets = 0;
    int udsgNJets = 0;
    for (int i = 0; i < 8; i++) {
      totNJets += nJets_[i];
      if(i > 0 && i < 4) udsNJets += nJets_[i];
      if((i > 0 && i < 4) || i == 6) udsgNJets += nJets_[i];
      if(i <= 5 && i >= 1) jetMultiplicity_->fill(i, nJets_[i], w);
      else if (i==6) jetMultiplicity_->fill(21, nJets_[i], w);
      else if (i==7) jetMultiplicity_->fill(20, nJets_[i], w);
      else jetMultiplicity_->fill(0, nJets_[i], w);
      nJets_[i] = 0; //reset to 0 before the next event
    }
    jetMultiplicity_->fill(-1, totNJets, w); //total number of jets in the event
    jetMultiplicity_->fill(123, udsNJets, w);
    jetMultiplicity_->fill(12321, udsgNJets, w);
  } else {
      int totNJets = 0;
      for (int i = 0; i < 8; i++) {
        totNJets += nJets_[i];
        nJets_[i] = 0;
      }
      jetMultiplicity_->fill(-1, totNJets, w);
    }
}

void JetTagPlotter::analyzeTag(const reco::Jet & jet, 
                    double jec,
                    float discriminator,
                    int jetFlavour,
                    float w/*=1*/)  
{
  if (mcPlots_) {
    dJetFlav_->fill(jetFlavour, jetFlavour, w);
    if (abs(jetFlavour) > 0 && abs(jetFlavour) < 6) nJets_[abs(jetFlavour)] += 1; //quarks 1 to 5
    else if (abs(jetFlavour) == 21) nJets_[6] += 1; //gluons
    else if (jetFlavour == 20) nJets_[7] += 1; //PU
    else nJets_[0] += 1; //NI
  } else {
    nJets_[0] += 1;
  }
  if (edm::isNotFinite(discriminator)) dDiscriminator_->fill(jetFlavour, -999.0, w);
  else dDiscriminator_->fill(jetFlavour, discriminator, w);
  dJetRecMomentum_->fill(jetFlavour, jet.p() * jec, w);
  dJetRecPt_->fill(jetFlavour, jet.pt() * jec, w);
  dJetRecPseudoRapidity_->fill(jetFlavour, jet.eta(), w);
  dJetRecPhi_->fill(jetFlavour, jet.phi(), w);
  if (doDifferentialPlots_) {
    if (edm::isFinite(discriminator) && discriminator > cutValue_) {
      dJetPhiDiscrCut_->fill(jetFlavour, jet.phi(), w);
      dJetPseudoRapidityDiscrCut_->fill(jetFlavour, jet.eta(), w);
    }
  }
}


void JetTagPlotter::analyzeTag(const reco::JetTag & jetTag, 
                   double jec,
                   int jetFlavour,
                   float w/*=1*/)
{
  if (mcPlots_) {
    dJetFlav_->fill(jetFlavour, jetFlavour, w);
    if (abs(jetFlavour) > 0 && abs(jetFlavour) < 6) nJets_[abs(jetFlavour)] += 1; //quarks 1 to 5
    else if (abs(jetFlavour) == 21) nJets_[6] += 1; //gluons
    else if (jetFlavour == 20) nJets_[7] += 1; //PU
    else nJets_[0] += 1; //NI
  } else {
    nJets_[0] += 1;
  }
  const auto& discriminator = jetTag.second;
  if (edm::isNotFinite(discriminator)) dDiscriminator_->fill(jetFlavour, -999.0, w);
  else dDiscriminator_->fill(jetFlavour, discriminator, w);
  dJetRecMomentum_->fill(jetFlavour, jetTag.first->p() * jec, w);
  dJetRecPt_->fill(jetFlavour, jetTag.first->pt() * jec, w);
  dJetRecPseudoRapidity_->fill(jetFlavour, jetTag.first->eta(), w);
  dJetRecPhi_->fill(jetFlavour, jetTag.first->phi(), w);
  if (doDifferentialPlots_) {
    if (edm::isFinite(discriminator) && discriminator > cutValue_) {
      dJetPhiDiscrCut_->fill(jetFlavour, jetTag.first->phi(), w);
      dJetPseudoRapidityDiscrCut_->fill(jetFlavour, jetTag.first->eta(), w);
    }
  }
}

void JetTagPlotter::finalize(DQMStore::IBooker & ibook_, DQMStore::IGetter & igetter_)
{
  //
  // final processing:
  // produce the misid. vs. eff histograms
  //
  const std::string & es = theExtensionString;
  const std::string jetTagDir(es.substr(1));
  dDiscriminator_ = std::make_unique<FlavourHistograms<double>>("discr" + es, "Discriminator", 102, discrStart_, discrEnd_, "b", jetTagDir, mcPlots_, igetter_);
  
  effPurFromHistos_ = std::make_unique<EffPurFromHistos>(*dDiscriminator_, jetTagDir, mcPlots_, ibook_, nBinEffPur_, startEffPur_, endEffPur_);
  effPurFromHistos_->doCTagPlots(doCTagPlots_);
  effPurFromHistos_->compute(ibook_);

  // Produce the differentiel efficiency vs. kinematical variables
  if (doDifferentialPlots_) {
    dJetRecPhi_ = std::make_unique<FlavourHistograms<double>>("jetPhi" + es, "jet phi", 20, -M_PI, M_PI, "b", jetTagDir, mcPlots_, igetter_); 
    dJetPhiDiscrCut_ = std::make_unique<FlavourHistograms<double>>("jetPhi_diffEff" + es, "Efficiency vs. jet Phi for discriminator above cut", 20, -M_PI, M_PI, "b", jetTagDir, mcPlots_, igetter_); 
    dJetPhiDiscrCut_->divide(*dJetRecPhi_);
    dJetPhiDiscrCut_->setEfficiencyFlag();
  
    dJetRecPseudoRapidity_ = std::make_unique<FlavourHistograms<double>>("jetEta" + es, "jet eta", 20, -etaPtBin_.getEtaMax(), etaPtBin_.getEtaMax(), "b", jetTagDir, mcPlots_, igetter_); 
    dJetPseudoRapidityDiscrCut_ = std::make_unique<FlavourHistograms<double>>("jetEta_diffEff" + es, "Efficiency vs. jet eta for discriminator above cut", 20, -etaPtBin_.getEtaMax(), etaPtBin_.getEtaMax(), "b", jetTagDir, mcPlots_, igetter_); 
    dJetPseudoRapidityDiscrCut_->divide(*dJetRecPseudoRapidity_);
    dJetPseudoRapidityDiscrCut_->setEfficiencyFlag();
  }
}

