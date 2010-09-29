#include <map>
#include <string>

#include "TH1.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

/// maximal number of bins used for the jet
/// response plots
static const unsigned int MAXBIN=8;
/// binning used for the jet response plots 
/// (NOTE BINS must have a length of MAXBIN
/// +1)
static const float BINS[]={30., 40., 50., 60., 70., 80., 100., 125., 150.};

/**
   \class   PatJetAnalyzer PatJetAnalyzer.h "PhysicsTools/PatAlgos/plugins/PatJetAnalyzer.h"

   \brief   Module to analyze pat::Jets in the context of a more complex exercise.

   Basic quantities of jets like the transverse momentum, eta and phi as well as the 
   invariant dijet mass are plotted. Basic histograms for a jet energy response plot 
   as a function of the pt of the reference object are filled. As reference matched 
   partons are chosen. Input parameters are:

    - src       --> input for the pat jet collection (edm::InputTag).

    - corrLevel --> string for the pat jet correction level.
*/

class PatJetAnalyzer : public edm::EDAnalyzer {

public:
  /// default contructor
  explicit PatJetAnalyzer(const edm::ParameterSet& cfg);
  /// default destructor
  ~PatJetAnalyzer(){};
  
private:
  /// everything that needs to be done during the even loop
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);

  /// check if histogram was booked
  bool booked(const std::string histName) const { return hists_.find(histName.c_str())!=hists_.end(); };
  /// fill histogram if it had been booked before
  void fill(const std::string histName, double value) const { if(booked(histName.c_str())) hists_.find(histName.c_str())->second->Fill(value); };

private:  
  /// correction level for pat jet
  std::string corrLevel_;
  /// pat jets
  edm::InputTag jets_;
  /// management of 1d histograms
  std::map<std::string,TH1F*> hists_; 
};

#include "DataFormats/PatCandidates/interface/Jet.h"

PatJetAnalyzer::PatJetAnalyzer(const edm::ParameterSet& cfg):
  corrLevel_(cfg.getParameter<std::string>("corrLevel")),
  jets_(cfg.getParameter<edm::InputTag>("src"))
{
  // register TFileService
  edm::Service<TFileService> fs;

  // jet multiplicity
  hists_["mult" ]=fs->make<TH1F>("mult" , "N_{Jet}"          ,   15,   0.,   15.);
  // jet pt (for all jets)
  hists_["pt"   ]=fs->make<TH1F>("pt"   , "p_{T}(Jet) [GeV]" ,   60,   0.,  300.);
  // jet eta (for all jets)
  hists_["eta"  ]=fs->make<TH1F>("eta"  , "#eta (Jet)"       ,   60,  -3.,    3.);
  // jet phi (for all jets)
  hists_["phi"  ]=fs->make<TH1F>("phi"  , "#phi (Jet)"       ,   60,  3.2,   3.2);
  // dijet mass (if available)
  hists_["mass" ]=fs->make<TH1F>("mass" , "M_{jj} [GeV]"     ,   50,   0.,  500.);
  // basic histograms for jet energy response
  for(unsigned int idx=0; idx<MAXBIN; ++idx){
    char buffer [10]; sprintf (buffer, "jes_%i", idx);
    char title  [50]; sprintf (title , "p_{T}^{rec}/p_{T}^{gen} [%i GeV - %i GeV]", (int)BINS[idx], (int)BINS[idx+1]);
    hists_[buffer]=fs->make<TH1F>(buffer, title,  100, 0., 2.);
  }  
}

void
PatJetAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
  // recieve jet collection label
  edm::Handle<edm::View<pat::Jet> > jets;
  event.getByLabel(jets_,jets);

  // loop jets
  for(edm::View<pat::Jet>::const_iterator jet=jets->begin(); jet!=jets->end(); ++jet){
    // fill basic kinematics
    fill( "pt" , jet->correctedJet(corrLevel_).pt());
    fill( "eta", jet->eta());
    fill( "phi", jet->phi());
    // basic plots for jet responds plot as a function of pt
    if( jet->genJet() ){
      double resp=jet->correctedJet(corrLevel_).pt()/jet->genJet()->pt();
      for(unsigned int idx=0; idx<MAXBIN; ++idx){
	if(BINS[idx]<=jet->genJet()->pt() && jet->genJet()->pt()<BINS[idx+1]){
	  char buffer [10]; sprintf (buffer, "jes_%i", idx);
	  fill( buffer, resp );
	}
      }
    }
  }
  // jet multiplicity
  fill( "mult" , jets->size());
  // invariane dijet mass
  if(jets->size()>1){ fill( "mass", ((*jets)[0].p4()+(*jets)[1].p4()).mass());}
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatJetAnalyzer);
