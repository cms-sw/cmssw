//
// Jet Analyzer class for heavy ion jets. for DQM jet analysis monitoring 
// For CMSSW_7_5_X, especially reading background subtracted jets - jet matching 1 - 1 
// author: Raghav Kunnawalkam Elayavalli,
//         April 6th 2015 
//         Rutgers University, email: raghav.k.e at CERN dot CH 
//


#include "DQMOffline/JetMET/interface/JetAnalyzer_HeavyIons_matching.h"

using namespace edm;
using namespace reco;
using namespace std;

// declare the constructors:


JetAnalyzer_HeavyIons_matching::JetAnalyzer_HeavyIons_matching(const edm::ParameterSet& iConfig) :
  mInputJet1Collection           (iConfig.getParameter<edm::InputTag>       ("src_Jet1")),
  mInputJet2Collection           (iConfig.getParameter<edm::InputTag>       ("src_Jet2")),
  JetType1                       (iConfig.getUntrackedParameter<std::string>("Jet1")),
  JetType2                       (iConfig.getUntrackedParameter<std::string>("Jet2")),
  mRecoJetPtThreshold            (iConfig.getParameter<double>              ("recoJetPtThreshold")),
  mRecoDelRMatch                 (iConfig.getParameter<double>              ("recoDelRMatch")),
  mRecoJetEtaCut                 (iConfig.getParameter<double>              ("recoJetEtaCut"))
{
  std::string inputCollectionLabelJet1(mInputJet1Collection.label());
  std::string inputCollectionLabelJet2(mInputJet2Collection.label());
  
  //consumes

  if(std::string("VsCalo") == JetType1) caloJet1Token_ = consumes<reco::CaloJetCollection>(mInputJet1Collection);
  if(std::string("VsPF") == JetType1) pfJetsToken_ = consumes<reco::PFJetCollection>(mInputJet1Collection);
  if(std::string("PuCalo") == JetType1) caloJet2Token_ = consumes<reco::CaloJetCollection>(mInputJet1Collection);
  if(std::string("PuPF") == JetType1) basicJetsToken_ = consumes<reco::BasicJetCollection>(mInputJet1Collection);

  if(std::string("VsCalo") == JetType2) caloJet1Token_ = consumes<reco::CaloJetCollection>(mInputJet2Collection);
  if(std::string("VsPF") == JetType2) pfJetsToken_ = consumes<reco::PFJetCollection>(mInputJet2Collection);
  if(std::string("PuCalo") == JetType2) caloJet2Token_ = consumes<reco::CaloJetCollection>(mInputJet2Collection);
  if(std::string("PuPF") == JetType2) basicJetsToken_ = consumes<reco::BasicJetCollection>(mInputJet2Collection);

  // initialize the Jet matching histograms

  mpT_ratio_Jet1Jet2 = 0;
  mpT_Jet1_matched = 0;
  mpT_Jet2_matched = 0;
  mpT_Jet1_unmatched = 0;
  mpT_Jet2_unmatched = 0;

  // we need to add histograms which will hold the hadronic and electromagnetic energy content for the unmatched Jets.
  if(std::string("VsCalo") == JetType1 || std::string("PuCalo") == JetType1){
    mHadEnergy_Jet1_unmatched = 0;
    mEmEnergy_Jet1_unmatched = 0; 
  }
  if(std::string("VsCalo") == JetType2 || std::string("PuCalo") == JetType2) {
    mHadEnergy_Jet2_unmatched = 0;
    mEmEnergy_Jet2_unmatched = 0; 
  }

  if(std::string("VsPF") == JetType1){
    mChargedHadronEnergy_Jet1_unmatched = 0; 
    mNeutralHadronEnergy_Jet1_unmatched = 0;
    mChargedEmEnergy_Jet1_unmatched = 0;
    mNeutralEmEnergy_Jet1_unmatched = 0;
    mChargedMuEnergy_Jet1_unmatched = 0;
  }
  
  if(std::string("VsPF") == JetType2){
    mChargedHadronEnergy_Jet2_unmatched = 0; 
    mNeutralHadronEnergy_Jet2_unmatched = 0;
    mChargedEmEnergy_Jet2_unmatched = 0;
    mNeutralEmEnergy_Jet2_unmatched = 0;
    mChargedMuEnergy_Jet2_unmatched = 0;
  }
  
  
}
   
void JetAnalyzer_HeavyIons_matching::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun,edm::EventSetup const &) 
  {

    ibooker.setCurrentFolder("JetMET/HIJetValidation/"+mInputJet1Collection.label()+mInputJet2Collection.label());
    
    mpT_ratio_Jet1Jet2 = ibooker.book1D("Ratio_Jet1pT_vs_Jet2pT","",100, 0, 10);
    mpT_Jet1_matched = ibooker.book1D("Jet1_matched_jet_Spectra","",100, 0, 1000);
    mpT_Jet2_matched = ibooker.book1D("Jet2_matched_jet_Spectra","",100, 0, 1000);
    mpT_Jet1_unmatched = ibooker.book1D("Jet1_unmatched_jet_Spectra","",100, 0, 1000);
    mpT_Jet2_unmatched = ibooker.book1D("Jet2_unmatched_jet_Spectra","",100, 0, 1000);
    
    if(std::string("VsCalo") == JetType1 || std::string("PuCalo") == JetType1){
      mHadEnergy_Jet1_unmatched = ibooker.book1D("HadEnergy_Jet1_unmatched", "HadEnergy_Jet1_unmatched", 50, 0, 200);
      mEmEnergy_Jet1_unmatched = ibooker.book1D("EmEnergy_Jet1_unmatched", "EmEnergy_Jet1_unmatched", 50, 0, 200);
    }
    
    if(std::string("VsCalo") == JetType2 || std::string("PuCalo") == JetType2){
      mHadEnergy_Jet2_unmatched = ibooker.book1D("HadEnergy_Jet2_unmatched", "HadEnergy_Jet2_unmatched", 50, 0, 200);
      mEmEnergy_Jet2_unmatched = ibooker.book1D("EmEnergy_Jet2_unmatched", "EmEnergy_Jet2_unmatched", 50, 0, 200);
    }

    if(std::string("VsPF") == JetType1){
      mChargedHadronEnergy_Jet1_unmatched = ibooker.book1D("ChargedHadronEnergy_Jet1_unmatched", "charged HAD energy unmatched Jet1",    50, 0, 100);
      mNeutralHadronEnergy_Jet1_unmatched = ibooker.book1D("ChargedHadronEnergy_Jet1_unmatched", "charged HAD energy unmatched Jet1",    50, 0, 100);
      mChargedEmEnergy_Jet1_unmatched = ibooker.book1D("ChargedEmEnergy_Jet1_unmatched", "charged EM energy unmatched Jet1",    50, 0, 100);
      mNeutralEmEnergy_Jet1_unmatched = ibooker.book1D("ChargedEmEnergy_Jet1_unmatched", "charged EM energy unmatched Jet1",    50, 0, 100);
      mChargedMuEnergy_Jet1_unmatched = ibooker.book1D("ChargedMuEnergy_Jet1_unmatched", "charged Mu energy unmatched Jet1",    50, 0, 100);
    }

    if(std::string("VsPF") == JetType2){
      mChargedHadronEnergy_Jet2_unmatched = ibooker.book1D("ChargedHadronEnergy_Jet2_unmatched", "charged HAD energy unmatched Jet2",    50, 0, 100);
      mNeutralHadronEnergy_Jet2_unmatched = ibooker.book1D("ChargedHadronEnergy_Jet2_unmatched", "charged HAD energy unmatched Jet2",    50, 0, 100);
      mChargedEmEnergy_Jet2_unmatched = ibooker.book1D("ChargedEmEnergy_Jet2_unmatched", "charged EM energy unmatched Jet2",    50, 0, 100);
      mNeutralEmEnergy_Jet2_unmatched = ibooker.book1D("ChargedEmEnergy_Jet2_unmatched", "charged EM energy unmatched Jet2",    50, 0, 100);
      mChargedMuEnergy_Jet2_unmatched = ibooker.book1D("ChargedMuEnergy_Jet2_unmatched", "charged Mu energy unmatched Jet2",    50, 0, 100);
    }
    
    if (mOutputFile.empty ()) 
      LogInfo("OutputInfo") << " Histograms will NOT be saved";
    else 
      LogInfo("OutputInfo") << " Histograms will be saved to file:" << mOutputFile;


  }



//------------------------------------------------------------------------------
// ~JetAnalyzer_HeavyIons
//------------------------------------------------------------------------------
JetAnalyzer_HeavyIons_matching::~JetAnalyzer_HeavyIons_matching() {}


//------------------------------------------------------------------------------
// beginJob
//------------------------------------------------------------------------------
//void JetAnalyzer_HeavyIons_matching::beginJob() {
//  std::cout<<"inside the begin job function"<<endl;
//}


//------------------------------------------------------------------------------
// endJob
//------------------------------------------------------------------------------
//void JetAnalyzer_HeavyIons_matching::endJob()
//{
//  if (!mOutputFile.empty() && &*edm::Service<DQMStore>())
//    {
//      edm::Service<DQMStore>()->save(mOutputFile);
//    }
//}


//------------------------------------------------------------------------------
// analyze
//------------------------------------------------------------------------------
void JetAnalyzer_HeavyIons_matching::analyze(const edm::Event& mEvent, const edm::EventSetup& mSetup)
{

  // Get the Jet collection
  //----------------------------------------------------------------------------
  
  std::vector<Jet> recoJet1;
  recoJet1.clear();
  std::vector<Jet> recoJet2;
  recoJet2.clear();
  
  edm::Handle<CaloJetCollection>  caloJet1;
  edm::Handle<CaloJetCollection>  caloJet2;
  edm::Handle<JPTJetCollection>   jptJets;
  edm::Handle<PFJetCollection>    pfJets;
  edm::Handle<BasicJetCollection> basicJets;

  if(std::string("VsCalo") == JetType1) {
    mEvent.getByToken(caloJet1Token_, caloJet1);
    for (unsigned ijet=0; ijet<caloJet1->size(); ++ijet) recoJet1.push_back((*caloJet1)[ijet]);
  }
  if(std::string("PuCalo") == JetType1) {
    mEvent.getByToken(caloJet2Token_, caloJet1);
    for (unsigned ijet=0; ijet<caloJet1->size(); ++ijet) recoJet1.push_back((*caloJet1)[ijet]);
  }
  if(std::string("VsPF") == JetType1) {
    mEvent.getByToken(pfJetsToken_, pfJets);
    for (unsigned ijet=0; ijet<pfJets->size(); ++ijet) recoJet1.push_back((*pfJets)[ijet]);
  }
  if(std::string("PuPF") == JetType1) {
    mEvent.getByToken(basicJetsToken_, basicJets);
    for (unsigned ijet=0; ijet<basicJets->size(); ++ijet) recoJet1.push_back((*basicJets)[ijet]);
  }

  if(std::string("VsCalo") == JetType2) {
    mEvent.getByToken(caloJet1Token_, caloJet2);
    for (unsigned ijet=0; ijet<caloJet2->size(); ++ijet) recoJet2.push_back((*caloJet2)[ijet]);
  }
  if(std::string("PuCalo") == JetType2) {
    mEvent.getByToken(caloJet2Token_, caloJet2);
    for (unsigned ijet=0; ijet<caloJet2->size(); ++ijet) recoJet2.push_back((*caloJet2)[ijet]);
  }
  if(std::string("VsPF") == JetType2) {
    mEvent.getByToken(pfJetsToken_, pfJets);
    for (unsigned ijet=0; ijet<pfJets->size(); ++ijet) recoJet2.push_back((*pfJets)[ijet]);
  }
  if(std::string("PuPF") == JetType2) {
    mEvent.getByToken(basicJetsToken_, basicJets);
    for (unsigned ijet=0; ijet<basicJets->size(); ++ijet) recoJet2.push_back((*basicJets)[ijet]);
  }

  // start to perform the matching - between recoJet1 and recoJet2.

  Int_t Jet1_nref = recoJet1.size();
  Int_t Jet2_nref = recoJet2.size();

  int jet1 = 0;
  int jet2 = 0; 
  
  std::vector <MyJet> vJet1, vJet2;
  std::vector <int> Jet1_ID(Jet1_nref), Jet2_ID(Jet2_nref);
  
  if(Jet1_nref == 0 || Jet2_nref == 0) return;
  
  for(unsigned ijet1 = 0; ijet1 < recoJet1.size(); ++ijet1){


    if(recoJet1[ijet1].pt() < mRecoJetPtThreshold) continue;
    if(fabs(recoJet1[ijet1].eta()) < mRecoJetEtaCut) continue;

    MyJet JET1;
    JET1.eta = recoJet1[ijet1].eta();
    JET1.phi = recoJet1[ijet1].phi();
    JET1.pt  = recoJet1[ijet1].pt();
    JET1.id  = ijet1; 

    vJet1.push_back(JET1);
    jet1++;

  }// first jet loop

  for(unsigned ijet2 = 0; ijet2 < recoJet2.size(); ++ijet2){

    if(recoJet2[ijet2].pt() < mRecoJetPtThreshold) continue;
    if(fabs(recoJet2[ijet2].eta()) < mRecoJetEtaCut) continue;

    MyJet JET2;
    JET2.eta = recoJet1[ijet2].eta();
    JET2.phi = recoJet1[ijet2].phi();
    JET2.pt  = recoJet1[ijet2].pt();
    JET2.id  = ijet2; 

    vJet2.push_back(JET2);
    jet2++;

  }// second jet loop

  bool onlyJet1     = (jet1>0  && jet2==0)  ? true : false;
  bool onlyJet2     = (jet1==0 && jet2 >0)  ? true : false;
  bool bothJet1Jet2 = (jet1>0  && jet2 >0)  ? true : false;
  
  int matchedJets   = 0;
  int unmatchedJet1 = 0;
  int unmatchedJet2 = 0;  

  std::vector < MyJet >::const_iterator iJet, jJet;

  if(onlyJet1) {

    for(iJet = vJet1.begin(); iJet != vJet1.end(); ++iJet){

      int pj = (*iJet).id;
      
      mpT_Jet1_unmatched->Fill(recoJet1[pj].pt());

    }

  }else if(onlyJet2) {

    for(iJet = vJet2.begin(); iJet != vJet2.end(); ++iJet){

      int cj = (*iJet).id;

      mpT_Jet2_unmatched->Fill(recoJet2[cj].pt());
    }
    
  }else if (bothJet1Jet2){

    ABMatchedJets mABMatchedJets; 

    for(iJet = vJet1.begin(); iJet != vJet1.end(); ++iJet){
      for(jJet = vJet2.begin(); jJet != vJet2.end(); ++jJet){
	mABMatchedJets.insert(std::make_pair(*iJet, *jJet));	
      }
    }

    ABItr itr;
    // matched Jets matching Jet 1 to Jet 2 
    for(itr = mABMatchedJets.begin(); itr != mABMatchedJets.end(); ++itr){

      ABJetPair jetpair = (*itr);
      MyJet Aj = jetpair.first;
      MyJet Bj = jetpair.second;

      float delr = JetAnalyzer_HeavyIons_matching::deltaRR(Bj.eta, Bj.phi, Aj.eta, Aj.phi);

      if( delr < mRecoDelRMatch && Jet1_ID[Aj.id] == 0){

	mpT_ratio_Jet1Jet2->Fill((Float_t) recoJet2[Bj.id].pt()/recoJet1[Aj.id].pt());

	mpT_Jet1_matched->Fill(recoJet1[Aj.id].pt());
	mpT_Jet2_matched->Fill(recoJet2[Bj.id].pt());
	
	Jet1_ID[Aj.id] = 1;
	Jet2_ID[Bj.id] = 1;

	matchedJets++; 


      }

    }

    // for unmatched Jets 
    for(itr = mABMatchedJets.begin(); itr != mABMatchedJets.end(); ++itr){

      ABJetPair jetpair = (*itr);

      MyJet Aj = jetpair.first;
      MyJet Bj = jetpair.second;

      if(Jet1_ID[Aj.id] == 0) {

	mpT_Jet1_unmatched->Fill(recoJet1[Aj.id].pt());
	unmatchedJet1++;
	Jet1_ID[Aj.id] = 1;

	if(std::string("VsCalo") == JetType1 || std::string("PuCalo") == JetType1){
	  mHadEnergy_Jet1_unmatched->Fill((*caloJet1)[Aj.id].hadEnergyInHO()
					  + (*caloJet1)[Aj.id].hadEnergyInHB()
					  + (*caloJet1)[Aj.id].hadEnergyInHF()
					  + (*caloJet1)[Aj.id].hadEnergyInHE()
					  );
	  mEmEnergy_Jet1_unmatched->Fill((*caloJet1)[Aj.id].emEnergyInEB()
					 + (*caloJet1)[Aj.id].emEnergyInEE()
					 + (*caloJet1)[Aj.id].emEnergyInHF()
					 );
	}

	if(std::string("VsPF") == JetType1){
	  mChargedHadronEnergy_Jet1_unmatched->Fill((*pfJets)[Aj.id].chargedHadronEnergy());
	  mNeutralHadronEnergy_Jet1_unmatched->Fill((*pfJets)[Aj.id].neutralHadronEnergy());
	  mChargedEmEnergy_Jet1_unmatched->Fill((*pfJets)[Aj.id].chargedEmEnergy());
	  mNeutralEmEnergy_Jet1_unmatched->Fill((*pfJets)[Aj.id].neutralEmEnergy());
	  mChargedMuEnergy_Jet1_unmatched->Fill((*pfJets)[Aj.id].chargedMuEnergy());
	}
	
      }

      if(Jet2_ID[Bj.id] == 0) {

	mpT_Jet2_unmatched->Fill(recoJet2[Bj.id].pt());
	unmatchedJet2++;
	Jet2_ID[Bj.id] = 2;
	if(std::string("VsCalo") == JetType2 || std::string("PuCalo") == JetType2){
	  mHadEnergy_Jet2_unmatched->Fill((*caloJet2)[Bj.id].hadEnergyInHO()
					  + (*caloJet2)[Bj.id].hadEnergyInHB()
					  + (*caloJet2)[Bj.id].hadEnergyInHF()
					  + (*caloJet2)[Bj.id].hadEnergyInHE()
					  );
	  mEmEnergy_Jet2_unmatched->Fill((*caloJet2)[Bj.id].emEnergyInEB()
					 + (*caloJet2)[Bj.id].emEnergyInEE()
					 + (*caloJet2)[Bj.id].emEnergyInHF()
					 );
	}
		

	if(std::string("VsPF") == JetType2){
	  mChargedHadronEnergy_Jet2_unmatched->Fill((*pfJets)[Bj.id].chargedHadronEnergy());
	  mNeutralHadronEnergy_Jet2_unmatched->Fill((*pfJets)[Bj.id].neutralHadronEnergy());
	  mChargedEmEnergy_Jet2_unmatched->Fill((*pfJets)[Bj.id].chargedEmEnergy());
	  mNeutralEmEnergy_Jet2_unmatched->Fill((*pfJets)[Bj.id].neutralEmEnergy());
	  mChargedMuEnergy_Jet2_unmatched->Fill((*pfJets)[Bj.id].chargedMuEnergy());
	}
       
      }
      
    }
    
  }// both Jet1 and Jet2 in the event. 


}


  
