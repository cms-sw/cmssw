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
/// binning used fro the jet response plots 
/// (NOTE BINS must have a length of MAXBIN
/// +1)
static const float BINS[]={0., 10., 20., 40., 60., 80., 100., 125., 150.};

/**
   \class   PatJetAnalyzer PatJetAnalyzer.h "PhysicsTools/PatAlgos/plugins/PatJetAnalyzer.h"

   \brief   module to analyze pat::Jets in the contect of a more complex exercise (detailed below).

   Exercise 1:

   (a)
   Make yourself familiar with the JetCorrFactors module of PAT. Inspect it using the standard 
   input file you used during the morning session and python in interactive mode (NB: use 
   python -i myfile_cfg.py). Make sure to understand the use and meaning of the parameters of 
   the module. Find out where the corresponding cfi file is located within the PatAlgos package.  

   (b)
   Make sure you understand how to retrieve jets with transvers momentum (pt) at different 
   correction levels of the jet energy scale (JES) from a pat::Jet. 

   (c)
   With the standard ttbar input sample you used during the morning session, compare the pt 
   of all pat::Jets with the pt of all reco::Jets at the following correction levels of the jet 
   energy scale (JES): Raw, L2Relative, L3Absolute. Use the most actual corrections for 7 TeV 
   data.

   (d)
   With the standard ttbar input sample you used during the morning session make a jet pt 
   response plot at the following correction levels of the JES: Raw, L2Relative, L3Absolute, 
   L5Flavor, L7Parton. Use the most actual corrections for 7 TeV data. Choose the L5Flavor and 
   L7Parton corrections to be determined from a ttbar sample instead of a QCD dijet sample 
   (which is the default configuration in the standard workflow of PAT). For the response 
   compare the pat::Jet to the matched generator jet or to a matched parton of status 3. You 
   may use the PatBasicExample to start from.
   
   As an extension you may distinguish between b-jets and light quark jets (based on the pdgId 
   of the matched status 3 parton) when plotting the jet response.


   Solution  :

   (c)
   We choose a simple implementation of an EDAnalyzer, which takes the following parameters: 
    - src       : input for the pat  jet collection (edm::InputTag).
    - reco      : input for the reco jet collection (edm::InputTag).
    - corrLevel : string for the pat jet correction level.
   The corrLevel string is expected to be of the form corrType:flavorType. The parameter 
   reco is optional; it can be omitted in the configuration file if not needed. We neglect 
   a complex parsing to check for allowed concrete substrings for the correction level or 
   correction flavor for the sake of simplicity; the user should take care of a proper input 
   here. Illegal strings will lead to an edm::Exception of the jet correction service. In a 
   corresponding cff file this module will be cloned for each correction level as mentioned 
   in the exercise.

   (d)
   For the sake of simplicity we restrict ourselves to the example of partons. The partons 
   are restricted to quarks (with masses below the top quark) only. A variable binning to 
   fill the basic histograms and the number of bins are defined as static const's outside 
   the class definition. Both, parton and generator jet matching are already provided to 
   best CMS knowledge by the configuration of the pat::Jet (check the configuration of the 
   patJetPartonMatch and the patJetGenJetMatch module and the WorkBookMCTruthMatch TWiki for 
   more details). We clone and re-use the module for each correction level mentioned in the 
   exercise.
*/

class PatJetAnalyzer : public edm::EDAnalyzer {

public:
  /// default contructor
  explicit PatJetAnalyzer(const edm::ParameterSet& cfg);
  /// default destructor
  ~PatJetAnalyzer(){};
  
private:
  /// everything that needs to be done before the event loop
  virtual void beginJob();
  /// everything that needs to be done during the even loop
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  /// everything that needs to be done after the event loop
  virtual void endJob();
  /// deduce correction level for pat::Jet; label is 
  /// expected to be of type 'corrLevel:flavorType'
  std::string corrLevel() { return corrLevel_.substr(0, corrLevel_.find(':')); };  
  /// deduce potential flavor type for pat::Jet; label
  /// is expected to be of type 'corrLevel:flavorType' 
  std::string corrFlavor() { return corrLevel_.substr(corrLevel_.find(':')+1); }; 

private:  
  /// simple map to contain all histograms; 
  /// histograms are booked in the beginJob() 
  /// method (for 1-dim histograms)
  std::map<std::string,TH1F*> hist1D_; 
  /// correction level for pat jet
  std::string corrLevel_;
  /// pat jets
  edm::InputTag jetsPat_;
  /// reco jets
  edm::InputTag jetsReco_;
};

#include "DataFormats/PatCandidates/interface/Jet.h"

PatJetAnalyzer::PatJetAnalyzer(const edm::ParameterSet& cfg) : hist1D_(),
  corrLevel_(cfg.getParameter<std::string>("corrLevel")),
  jetsPat_(cfg.getParameter<edm::InputTag>("src"))
{
  if(cfg.existsAs<std::string>("reco")){
    // can be omitted in the cfi file
    jetsReco_=cfg.getParameter<edm::InputTag>("reco");
  }
}

void
PatJetAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
  edm::Handle<edm::View<pat::Jet> > jetsPat;
  event.getByLabel(jetsPat_,jetsPat);

  size_t nPat =0;
  for(edm::View<pat::Jet>::const_iterator jet=jetsPat->begin(); jet!=jetsPat->end(); ++jet){
    hist1D_["jetPtPat"]->Fill(jet->correctedJet(corrLevel(), corrFlavor()).pt());
    if(jet->correctedJet(corrLevel(), corrFlavor()).pt()>20){ ++nPat; }

    if( jet->genParton() && abs(jet->genParton()->pdgId())<6 ){
      double resp=( jet->pt()-jet->genParton()->pt() )/jet->genParton()->pt();
      for(unsigned int idx=0; idx<MAXBIN; ++idx){
	if(BINS[idx]<=jet->genParton()->pt() && jet->genParton()->pt()<BINS[idx+1]){
	  char buffer [10]; sprintf (buffer, "jes_%i", idx);
	  hist1D_[buffer]->Fill( resp );
	}
      }
    }
  }
  hist1D_["jetMultPat"]->Fill(nPat);

  if(!jetsReco_.label().empty()){
    edm::Handle<edm::View<reco::Jet> > jetsReco;
    event.getByLabel(jetsReco_,jetsReco);
    
    size_t nReco=0;
    for(edm::View<reco::Jet>::const_iterator jet=jetsReco->begin(); jet!=jetsReco->end(); ++jet){
      hist1D_["jetPtReco"]->Fill(jet->pt());
      if(jet->pt()>20){ ++nReco; }
    }
    hist1D_["jetMultReco"]->Fill(nReco);
  }
}

void 
PatJetAnalyzer::beginJob()
{
  // register TFileService
  edm::Service<TFileService> fs;

  for(unsigned int idx=0; idx<MAXBIN; ++idx){
    char buffer [10]; sprintf (buffer, "jes_%i", idx);
    hist1D_[buffer]=fs->make<TH1F>(buffer, "(pt_{rec}-pt_{gen})/pt_{rec}",  80, 10., 10.);
  }  
  hist1D_["jetMultPat" ]=fs->make<TH1F>("jetMultPat" , "N_{>20}(jet)" ,   10, 0.,  10.);
  hist1D_["jetPtPat"   ]=fs->make<TH1F>("jetPtPat"   , "pt_{all}(jet)",  150, 0., 300.);
  if(jetsReco_.label().empty()) return;

  hist1D_["jetMultReco"]=fs->make<TH1F>("jetMultReco", "N_{>20}(jet)" ,   10, 0.,  10.);
  hist1D_["jetPtReco"  ]=fs->make<TH1F>("jetPtReco"  , "pt_{all}(jet)",  150, 0., 300.);
}

void 
PatJetAnalyzer::endJob() 
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatJetAnalyzer);
