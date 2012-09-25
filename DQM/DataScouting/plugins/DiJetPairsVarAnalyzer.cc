#include "DQM/DataScouting/plugins/DiJetPairsVarAnalyzer.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include <cmath>

//------------------------------------------------------------------------------
// A simple constructor which takes as inoput only the name of the PF jet collection
DiJetPairsVarAnalyzer::DiJetPairsVarAnalyzer( const edm::ParameterSet & conf ):
  ScoutingAnalyzerBase(conf),
  jetPtCut_              (conf.getParameter<double>("jetPtCut")),
  htCut_                 (conf.getParameter<double>("htCut")),
  delta_                 (conf.getParameter<double>("delta")),
  jetPtCollectionTag_    (conf.getUntrackedParameter<edm::InputTag>("jetPtCollectionTag")),
  dijetPtCollectionTag_   (conf.getUntrackedParameter<edm::InputTag>("dijetPtCollectionTag")),
  dijetdRCollectionTag_   (conf.getUntrackedParameter<edm::InputTag>("dijetdRCollectionTag")),
  dijetMassCollectionTag_ (conf.getUntrackedParameter<edm::InputTag>("dijetMassCollectionTag")){
}

//------------------------------------------------------------------------------
// Nothing to destroy: the DQM service thinks about everything
DiJetPairsVarAnalyzer::~DiJetPairsVarAnalyzer(){}

//------------------------------------------------------------------------------
// Usual analyze method
void DiJetPairsVarAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & c ){

  edm::Handle<std::vector<double> > jetPtvar_handle;
  edm::Handle<std::vector<double> > dijetMassvar_handle;
  edm::Handle<std::vector<double> > dijetSumPtvar_handle;
  edm::Handle<std::vector<double> > dijetdRvar_handle;
  iEvent.getByLabel(jetPtCollectionTag_,jetPtvar_handle);
  iEvent.getByLabel(dijetPtCollectionTag_,dijetSumPtvar_handle);
  iEvent.getByLabel(dijetMassCollectionTag_,dijetMassvar_handle);
  iEvent.getByLabel(dijetdRCollectionTag_,dijetdRvar_handle);

  me_Njets->Fill(int(jetPtvar_handle->size()));
  if(int(jetPtvar_handle->size()) > 3 && int(dijetMassvar_handle->size()) > 5 &&
     int(dijetSumPtvar_handle->size()) > 5 && int(dijetdRvar_handle->size()) > 5){

    const double lowJetpt = jetPtvar_handle->at(3);
    double Ht = 0.0;
    for (int i =0; i < int(jetPtvar_handle->size()); ++i) Ht += jetPtvar_handle->at(i);
    me_Ht->Fill(Ht);
    me_fourthJetPt->Fill(lowJetpt);
    
    if (lowJetpt > jetPtCut_ && Ht > htCut_){
      //DD: Fill NJet dist
      //select best dijet pair
      double selDijetPt1  = 0.0;
      double selDijetPt2  = 0.0;
      double selAvgDijetM = 0.0;
      bool passDelta      = false;
      double selDeltaM    = 0.80;
      for (int i = 0; i < 3; ++i){
	int j = i +3;
	double dijetdR1= dijetdRvar_handle->at(i);
	double dijetdR2= dijetdRvar_handle->at(j);
	if (dijetdR1 < 0.7 || dijetdR2 < 0.7) continue;
	double dijetM1 = dijetMassvar_handle->at(i);
	double dijetM2 = dijetMassvar_handle->at(j);
	double dijetPt1= dijetSumPtvar_handle->at(i);
	double dijetPt2= dijetSumPtvar_handle->at(j);
	double deltaM  = fabs(dijetM1 - dijetM2);
	double avgM    = fabs(dijetM1 + dijetM2)/2.0;
	if (!(avgM > 0.0)) continue;
	//Need to check if this is a fractional cut req... 
	if (deltaM/avgM > 0.15 || deltaM/avgM > selDeltaM) continue;
	passDelta = false;
	if ((dijetPt1 - avgM > delta_) && (dijetPt2 - avgM > delta_))
	  passDelta = true;
	selDijetPt1 = dijetPt1;
	selDijetPt2 = dijetPt2;
	selAvgDijetM= avgM;
	selDeltaM   = deltaM/avgM;
      }
      //take best dijet pair and apply Delta cut for each jet:
      me_MassDiff->Fill(selDeltaM);
      if (selDeltaM < 0.75){
	//DD: Fill me_DeltavsAvgMass
	double delta1 = selDijetPt1 - selAvgDijetM;
	double delta2 = selDijetPt2 - selAvgDijetM;
	me_DeltavsAvgMass->Fill(selAvgDijetM,delta1);
	me_DeltavsAvgMass->Fill(selAvgDijetM,delta2);
	if (passDelta)
	  me_AvgDiJetMass->Fill(selAvgDijetM);
	//DD: Fill me_AvgDiJetMass
      }
    }
  }
}

void DiJetPairsVarAnalyzer::endRun( edm::Run const &, edm::EventSetup const & ){
}

//------------------------------------------------------------------------------
// Function to book the Monitoring Elements.
void DiJetPairsVarAnalyzer::bookMEs(){

  //4th jet pT
  me_fourthJetPt = bookH1withSumw2("fourthJetPt",
				  "4^{TH} jet p_{T} (GeV)",
				  250,0.,500.,
				  "4^{TH} jet p_{T} (GeV)");
  //HT distribution
  me_Ht         = bookH1withSumw2("Ht",
				  "H_{T}",
				  100, 0.,2000.,
				  "H_{T} (GeV)");
  //Njet distribution
  me_Njets      = bookH1withSumw2("Njets",
				  "Number of Jets",
				  16,0.,16.,
				  "N_{JETS} / Event");
  //avg dijet mass distribution
  me_MassDiff   = bookH1withSumw2("MassDiff",
				      "Paired Dijet Fractional Mass Difference",
				      50,0.0,0.5,
				      "Paired Dijet Fractional Mass Difference");
  //avg dijet mass distribution
  me_AvgDiJetMass   = bookH1withSumw2("AvgDiJetMass",
				      "Paired Dijet Average Mass",
				      1500,0.,3000.,
				      "Paired Dijet Average Mass (GeV)");
  //2D triplet pt vs triplet mass
  me_DeltavsAvgMass= bookH2withSumw2("DeltavsAvgMass",
				     "#Delta vs Paired Dijet Average Mass",
				     300,0.,3000.,
				     80,-200.,200.,
				     "#Delta = #Sum_{i=1,2}(P_{T})_{i} - m_{AVG} (GeV)",
				     "Paired Dijet Average Mass (GeV)");
}

