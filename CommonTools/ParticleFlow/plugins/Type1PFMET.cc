
// user include files
#include "CommonTools/ParticleFlow/plugins/Type1PFMET.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/JetReco/interface/PFJet.h"

//using namespace std;

using namespace reco;

// PRODUCER CONSTRUCTORS ------------------------------------------
Type1PFMET::Type1PFMET( const edm::ParameterSet& iConfig )
{
  tokenUncorMet  = consumes<METCollection>(iConfig.getParameter<edm::InputTag>("inputUncorMetLabel"));
  tokenUncorJets = consumes<PFJetCollection>(iConfig.getParameter<edm::InputTag>("inputUncorJetsTag"));
  correctorToken = consumes<JetCorrector>(iConfig.getParameter<edm::InputTag>("corrector"));
  jetPTthreshold      = iConfig.getParameter<double>("jetPTthreshold");
  jetEMfracLimit      = iConfig.getParameter<double>("jetEMfracLimit");
  jetMufracLimit      = iConfig.getParameter<double>("jetMufracLimit");
  produces<METCollection>();
}
Type1PFMET::Type1PFMET()  {}

// PRODUCER DESTRUCTORS -------------------------------------------
Type1PFMET::~Type1PFMET() {}

// PRODUCER METHODS -----------------------------------------------
  void Type1PFMET::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  using namespace edm;
  Handle<PFJetCollection> inputUncorJets;
  iEvent.getByToken( tokenUncorJets, inputUncorJets );
  Handle<JetCorrector> corrector;
  iEvent.getByToken( correctorToken, corrector );
  Handle<METCollection> inputUncorMet;                     //Define Inputs
  iEvent.getByToken( tokenUncorMet,  inputUncorMet );     //Get Inputs
  std::auto_ptr<METCollection> output( new METCollection() );  //Create empty output
  run( *(inputUncorMet.product()), *(corrector.product()), *(inputUncorJets.product()),
       jetPTthreshold, jetEMfracLimit, jetMufracLimit,
       &*output );                                         //Invoke the algorithm
  iEvent.put( output );                                        //Put output into Event
}

void Type1PFMET::run(const METCollection& uncorMET,
		     const reco::JetCorrector& corrector,
		     const PFJetCollection& uncorJet,
		     double jetPTthreshold,
		     double jetEMfracLimit,
		     double jetMufracLimit,
		     METCollection* corMET)
{
  if (!corMET) {
    std::cerr << "Type1METAlgo_run-> undefined output MET collection. Stop. " << std::endl;
    return;
  }

  double DeltaPx = 0.0;
  double DeltaPy = 0.0;
  double DeltaSumET = 0.0;
  // ---------------- Calculate jet corrections, but only for those uncorrected jets
  // ---------------- which are above the given threshold.  This requires that the
  // ---------------- uncorrected jets be matched with the corrected jets.
  for( PFJetCollection::const_iterator jet = uncorJet.begin(); jet != uncorJet.end(); ++jet) {
    if( jet->pt() > jetPTthreshold ) {
      double emEFrac =
	jet->chargedEmEnergyFraction() + jet->neutralEmEnergyFraction();
      double muEFrac = jet->chargedMuEnergyFraction();
      if( emEFrac < jetEMfracLimit
	  && muEFrac < jetMufracLimit ) {
	double corr = corrector.correction (*jet) - 1.; // correction itself
	DeltaPx +=  jet->px() * corr;
	DeltaPy +=  jet->py() * corr;
	DeltaSumET += jet->et() * corr;
      }
    }
  }
  //----------------- Calculate and set deltas for new MET correction
  CorrMETData delta;
  delta.mex   =  - DeltaPx;    //correction to MET (from Jets) is negative,
  delta.mey   =  - DeltaPy;    //since MET points in direction opposite of jets
  delta.sumet =  DeltaSumET;
  //----------------- Fill holder with corrected MET (= uncorrected + delta) values
  const MET* u = &(uncorMET.front());
  double corrMetPx = u->px()+delta.mex;
  double corrMetPy = u->py()+delta.mey;
  MET::LorentzVector correctedMET4vector( corrMetPx, corrMetPy, 0.,
					  sqrt (corrMetPx*corrMetPx + corrMetPy*corrMetPy)
					  );
  //----------------- get previous corrections and push into new corrections
  std::vector<CorrMETData> corrections = u->mEtCorr();
  corrections.push_back( delta );
  //----------------- Push onto MET Collection
  MET result = MET(u->sumEt()+delta.sumet,
		   corrections,
		   correctedMET4vector,
		   u->vertex() );
  corMET->push_back(result);

  return;
}

//  DEFINE_FWK_MODULE(Type1PFMET);  //define this as a plug-in


