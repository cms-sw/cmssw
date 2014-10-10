#include "RecoParticleFlow/Benchmark/interface/PFJetBenchmark.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// preprocessor macro for booking 1d histos with DQMStore -or- bare Root
#define BOOK1D(name,title,nbinsx,lowx,highx) \
  h##name = dbe_ ? dbe_->book1D(#name,title,nbinsx,lowx,highx)->getTH1F() \
    : new TH1F(#name,title,nbinsx,lowx,highx)

// preprocessor macro for booking 2d histos with DQMStore -or- bare Root
#define BOOK2D(name,title,nbinsx,lowx,highx,nbinsy,lowy,highy) \
  h##name = dbe_ ? dbe_->book2D(#name,title,nbinsx,lowx,highx,nbinsy,lowy,highy)->getTH2F() \
    : new TH2F(#name,title,nbinsx,lowx,highx,nbinsy,lowy,highy)

//macros for building barrel and endcap histos with one call
#define DBOOK1D(name,title,nbinsx,lowx,highx) \
  BOOK1D(B##name,"Barrel "#title,nbinsx,lowx,highx); BOOK1D(E##name,"Endcap "#title,nbinsx,lowx,highx); BOOK1D(F##name,"Forward "#title,nbinsx,lowx,highx);
#define DBOOK2D(name,title,nbinsx,lowx,highx,nbinsy,lowy,highy) \
  BOOK2D(B##name,"Barrel "#title,nbinsx,lowx,highx,nbinsy,lowy,highy); BOOK2D(E##name,"Endcap "#title,nbinsx,lowx,highx,nbinsy,lowy,highy); BOOK2D(F##name,"Forward "#title,nbinsx,lowx,highx,nbinsy,lowy,highy);

// all versions OK
// preprocesor macro for setting axis titles
#define SETAXES(name,xtitle,ytitle) \
  h##name->GetXaxis()->SetTitle(xtitle); h##name->GetYaxis()->SetTitle(ytitle)

//macro for setting the titles for barrel and endcap together
#define DSETAXES(name,xtitle,ytitle) \
  SETAXES(B##name,xtitle,ytitle);SETAXES(E##name,xtitle,ytitle);SETAXES(F##name,xtitle,ytitle);
/*#define SET2AXES(name,xtitle,ytitle) \
  hE##name->GetXaxis()->SetTitle(xtitle); hE##name->GetYaxis()->SetTitle(ytitle);  hB##name->GetXaxis()->SetTitle(xtitle); hB##name->GetYaxis()->SetTitle(ytitle)
*/

#define PT (plotAgainstReco_)?"reconstructed P_{T}" :"generated P_{T}"
#define P (plotAgainstReco_)?"generated P" :"generated P"

using namespace reco;
using namespace std;

PFJetBenchmark::PFJetBenchmark() : file_(0), entry_(0) {}

PFJetBenchmark::~PFJetBenchmark() {
  if(file_) file_->Close();
}

void PFJetBenchmark::write() {
   // Store the DAQ Histograms 
  if (outputFile_.size() != 0) {
    if (dbe_)
          dbe_->save(outputFile_.c_str());
    // use bare Root if no DQM (FWLite applications)
    else if (file_) {
       file_->Write(outputFile_.c_str());
       cout << "Benchmark output written to file " << outputFile_.c_str() << endl;
       file_->Close();
       }
  }
  else 
    cout << "No output file specified ("<<outputFile_<<"). Results will not be saved!" << endl;
    
} 

void PFJetBenchmark::setup(
			   string Filename,
			   bool debug, 
			   bool plotAgainstReco,
			   bool onlyTwoJets,
			   double deltaRMax, 
			   string benchmarkLabel_, 
			   double recPt, 
			   double maxEta, 
			   DQMStore * dbe_store) {
  debug_ = debug; 
  plotAgainstReco_ = plotAgainstReco;
  onlyTwoJets_ = onlyTwoJets;
  deltaRMax_ = deltaRMax;
  outputFile_=Filename;
  recPt_cut = recPt;
  maxEta_cut= maxEta;
  file_ = NULL;
  dbe_ = dbe_store;
  // print parameters
  cout<< "PFJetBenchmark Setup parameters =============================================="<<endl;
  cout << "Filename to write histograms " << Filename<<endl;
  cout << "PFJetBenchmark debug " << debug_<< endl;
  cout << "plotAgainstReco " << plotAgainstReco_ << endl;
  cout << "onlyTwoJets " << onlyTwoJets_ << endl;
  cout << "deltaRMax " << deltaRMax << endl;
  cout << "benchmarkLabel " << benchmarkLabel_ << endl;
  cout << "recPt_cut " << recPt_cut << endl;
  cout << "maxEta_cut " << maxEta_cut << endl;
  
  // Book histogram

  // Establish DQM Store
  string path = "PFTask/Benchmarks/"+ benchmarkLabel_ + "/";
  if (plotAgainstReco) path += "Reco"; else path += "Gen";
  if (dbe_) {
    dbe_->setCurrentFolder(path.c_str());
  }
  else {
    file_ = new TFile(outputFile_.c_str(), "recreate");
//    TTree * tr = new TTree("PFTast");
//    tr->Branch("Benchmarks/ParticleFlow")
    cout << "Info: DQM is not available to provide data storage service. Using TFile to save histograms. "<<endl;
  }
  // Jets inclusive  distributions  (Pt > 20 or specified recPt GeV)
  char cutString[35];
  sprintf(cutString,"Jet multiplicity P_{T}>%4.1f GeV", recPt_cut);
  BOOK1D(Njets,cutString,50, 0, 50);

  BOOK1D(jetsPt,"Jets P_{T} Distribution",100, 0, 500);

  sprintf(cutString,"Jets #eta Distribution |#eta|<%4.1f", maxEta_cut);
  BOOK1D(jetsEta,cutString,100, -5, 5);
	
  BOOK2D(RPtvsEta,"#DeltaP_{T}/P_{T} vs #eta",200, -5., 5., 200,-1,1); 
  BOOK2D(RNeutvsEta,"R_{Neutral} vs #eta",200, -5., 5., 200,-1,1); 
  BOOK2D(RNEUTvsEta,"R_{HCAL+ECAL} vs #eta",200, -5., 5., 200,-1,1); 
  BOOK2D(RNONLvsEta,"R_{HCAL+ECAL - Hcal Only} vs #eta",200, -5., 5., 200,-1,1); 
  BOOK2D(RHCALvsEta,"R_{HCAL} vs #eta",200, -5., 5., 200,-1,1); 
  BOOK2D(RHONLvsEta,"R_{HCAL only} vs #eta",200, -5., 5., 200,-1,1); 
  BOOK2D(RCHEvsEta,"R_{Charged} vs #eta",200, -5., 5., 200,-1,1); 
  BOOK2D(NCHvsEta,"N_{Charged} vs #eta",200, -5., 5., 200,0.,200.);
  BOOK2D(NCH0vsEta,"N_{Charged} vs #eta, iter 0",200, -5., 5., 200,0.,200.);
  BOOK2D(NCH1vsEta,"N_{Charged} vs #eta, iter 1",200, -5., 5., 200,0.,200.);
  BOOK2D(NCH2vsEta,"N_{Charged} vs #eta, iter 2",200, -5., 5., 200,0.,200.);
  BOOK2D(NCH3vsEta,"N_{Charged} vs #eta, iter 3",200, -5., 5., 200,0.,200.);
  BOOK2D(NCH4vsEta,"N_{Charged} vs #eta, iter 4",200, -5., 5., 200,0.,200.);
  BOOK2D(NCH5vsEta,"N_{Charged} vs #eta, iter 5",200, -5., 5., 200,0.,200.);
  BOOK2D(NCH6vsEta,"N_{Charged} vs #eta, iter 6",200, -5., 5., 200,0.,200.);
  BOOK2D(NCH7vsEta,"N_{Charged} vs #eta, iter 7",200, -5., 5., 200,0.,200.);
  // delta Pt or E quantities for Barrel
  DBOOK1D(RPt,#DeltaP_{T}/P_{T},80,-1,1);
  DBOOK1D(RCHE,#DeltaE/E (charged had),80,-2,2);
  DBOOK1D(RNHE,#DeltaE/E (neutral had),80,-2,2);
  DBOOK1D(RNEE,#DeltaE/E (neutral em),80,-2,2);
  DBOOK1D(Rneut,#DeltaE/E (neutral),80,-2,2);
  DBOOK1D(NCH, #N_{charged},200,0.,200.);
  DBOOK2D(RPtvsPt,#DeltaP_{T}/P_{T} vs P_{T},250, 0, 500, 200,-1,1);       //used to be 50 bin for each in x-direction
  DBOOK2D(RCHEvsPt,#DeltaE/E (charged had) vs P_{T},250, 0, 500, 120,-1,2);
  DBOOK2D(RNHEvsPt,#DeltaE/E (neutral had) vs P_{T},250, 0, 500, 120,-1,2);
  DBOOK2D(RNEEvsPt,#DeltaE/E (neutral em) vs P_{T},250, 0, 500, 120,-1,2);
  DBOOK2D(RneutvsPt,#DeltaE/E (neutral) vs P_{T},250, 0, 500, 120,-1,2);
  DBOOK2D(NCHvsPt,N_{charged} vs P_{T},250,0,500,200,0.,200.);
  DBOOK2D(NCH0vsPt, N_{charged} vs P_{T} iter 0,250,0,500,200,0.,200.);
  DBOOK2D(NCH1vsPt, N_{charged} vs P_{T} iter 1,250,0,500,200,0.,200.);
  DBOOK2D(NCH2vsPt, N_{charged} vs P_{T} iter 2,250,0,500,200,0.,200.);
  DBOOK2D(NCH3vsPt, N_{charged} vs P_{T} iter 3,250,0,500,200,0.,200.);
  DBOOK2D(NCH4vsPt, N_{charged} vs P_{T} iter 4,250,0,500,200,0.,200.);
  DBOOK2D(NCH5vsPt, N_{charged} vs P_{T} iter 5,250,0,500,200,0.,200.);
  DBOOK2D(NCH6vsPt, N_{charged} vs P_{T} iter 6,250,0,500,200,0.,200.);
  DBOOK2D(NCH7vsPt, N_{charged} vs P_{T} iter 7,250,0,500,200,0.,200.);
  

  DBOOK2D(RNEUTvsP,#DeltaE/E (ECAL+HCAL) vs P,250, 0, 1000, 150,-1.5,1.5);
  DBOOK2D(RNONLvsP,#DeltaE/E (ECAL+HCAL-only) vs P,250, 0, 1000, 150,-1.5,1.5);
  DBOOK2D(RHCALvsP,#DeltaE/E (HCAL) vs P,250, 0, 1000, 150,-1.5,1.5);
  DBOOK2D(RHONLvsP,#DeltaE/E (HCAL only) vs P,250, 0, 1000, 150,-1.5,1.5);
  DBOOK1D(RPt20_40,#DeltaP_{T}/P_{T},80,-1,1);
  DBOOK1D(RPt40_60,#DeltaP_{T}/P_{T},80,-1,1);
  DBOOK1D(RPt60_80,#DeltaP_{T}/P_{T},80,-1,1);
  DBOOK1D(RPt80_100,#DeltaP_{T}/P_{T},80,-1,1);
  DBOOK1D(RPt100_150,#DeltaP_{T}/P_{T},80,-1,1);
  DBOOK1D(RPt150_200,#DeltaP_{T}/P_{T},80,-1,1);
  DBOOK1D(RPt200_250,#DeltaP_{T}/P_{T},80,-1,1);
  DBOOK1D(RPt250_300,#DeltaP_{T}/P_{T},80,-1,1);
  DBOOK1D(RPt300_400,#DeltaP_{T}/P_{T},160,-1,1);
  DBOOK1D(RPt400_500,#DeltaP_{T}/P_{T},160,-1,1);
  DBOOK1D(RPt500_750,#DeltaP_{T}/P_{T},160,-1,1);
  DBOOK1D(RPt750_1250,#DeltaP_{T}/P_{T},160,-1,1);
  DBOOK1D(RPt1250_2000,#DeltaP_{T}/P_{T},160,-1,1);
  DBOOK1D(RPt2000_5000,#DeltaP_{T}/P_{T},160,-1,1);

  DBOOK2D(DEtavsPt,#Delta#eta vs P_{T},1000,0,2000,500,-0.5,0.5);
  DBOOK2D(DPhivsPt,#Delta#phi vs P_{T},1000,0,2000,500,-0.5,0.5);
  BOOK2D(DEtavsEta,"#Delta#eta vs P_{T}",1000,-5.,+5.,500,-0.5,0.5);
  BOOK2D(DPhivsEta,"#Delta#phi vs P_{T}",1000,-5.,+5.,500,-0.5,0.5);
	
 // Set Axis Titles
 
 // Jets inclusive  distributions  (Pt > 20 GeV)
  SETAXES(Njets,"","Multiplicity");
  SETAXES(jetsPt, PT, "Number of Events");
  SETAXES(jetsEta, "#eta", "Number of Events");
  SETAXES(RNeutvsEta, "#eta", "#DeltaE/E (Neutral)");
  SETAXES(RNEUTvsEta, "#eta", "#DeltaE/E (ECAL+HCAL)");
  SETAXES(RNONLvsEta, "#eta", "#DeltaE/E (ECAL+HCAL-only)");
  SETAXES(RHCALvsEta, "#eta", "#DeltaE/E (HCAL)");
  SETAXES(RHONLvsEta, "#eta", "#DeltaE/E (HCAL Only)");
  SETAXES(RCHEvsEta, "#eta", "#DeltaE/E (Charged)");
  SETAXES(RPtvsEta, "#eta", "#DeltaP_{T}/P_{T}");
  SETAXES(DEtavsEta, "#eta", "#Delta#eta");
  SETAXES(DPhivsEta,"#eta", "#Delta#phi");
  // delta Pt or E quantities for Barrel and Endcap
  DSETAXES(RPt, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt20_40, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt40_60, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt60_80, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt80_100, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt100_150, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt150_200, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt200_250, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt250_300, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt300_400, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt400_500, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt500_750, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt750_1250, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt1250_2000, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RPt2000_5000, "#DeltaP_{T}/P_{T}", "Events");
  DSETAXES(RCHE, "#DeltaE/E(charged had)", "Events");
  DSETAXES(RNHE, "#DeltaE/E(neutral had)", "Events");
  DSETAXES(RNEE, "#DeltaE/E(neutral em)", "Events");
  DSETAXES(Rneut, "#DeltaE/E(neutral)", "Events");
  DSETAXES(RPtvsPt, PT, "#DeltaP_{T}/P_{T}");
  DSETAXES(RCHEvsPt, PT, "#DeltaE/E(charged had)");
  DSETAXES(RNHEvsPt, PT, "#DeltaE/E(neutral had)");
  DSETAXES(RNEEvsPt, PT, "#DeltaE/E(neutral em)");
  DSETAXES(RneutvsPt, PT, "#DeltaE/E(neutral)");
  DSETAXES(RHCALvsP, P, "#DeltaE/E(HCAL)");
  DSETAXES(RHONLvsP, P, "#DeltaE/E(HCAL-only)");
  DSETAXES(RNEUTvsP, P, "#DeltaE/E(ECAL+HCAL)");
  DSETAXES(RNONLvsP, P, "#DeltaE/E(ECAL+HCAL-only)");
  DSETAXES(DEtavsPt, PT, "#Delta#eta");
  DSETAXES(DPhivsPt, PT, "#Delta#phi");

}


void PFJetBenchmark::process(const reco::PFJetCollection& pfJets, const reco::GenJetCollection& genJets) {
  // loop over reco  pf  jets
  resPtMax_ = 0.;
  resChargedHadEnergyMax_ = 0.;
  resNeutralHadEnergyMax_ = 0.;
  resNeutralEmEnergyMax_ = 0.; 
  int NPFJets = 0;
	
  for(unsigned i=0; i<pfJets.size(); i++) {   

    // Count the number of jets with a larger energy
    unsigned highJets = 0;
    for(unsigned j=0; j<pfJets.size(); j++) { 
      if ( j != i && pfJets[j].pt() > pfJets[i].pt() ) highJets++;
    }
    if ( onlyTwoJets_ && highJets > 1 ) continue;
		
		
    const reco::PFJet& pfj = pfJets[i];
    double rec_pt = pfj.pt();
    double rec_eta = pfj.eta();
    double rec_phi = pfj.phi();

    // skip PFjets with pt < recPt_cut GeV
    if (rec_pt<recPt_cut and recPt_cut != -1.) continue;
    // skip PFjets with eta > maxEta_cut
    if (fabs(rec_eta)>maxEta_cut and maxEta_cut != -1.) continue;

    NPFJets++;
		
    // fill inclusive PFjet distribution pt > 20 GeV
    hNjets->Fill(NPFJets);
    hjetsPt->Fill(rec_pt);
    hjetsEta->Fill(rec_eta);

    // separate Barrel PFJets from Endcap PFJets
    bool Barrel = false;
    bool Endcap = false;
    bool Forward = false;
    if (std::abs(rec_eta) < 1.4 ) Barrel = true;
    if (std::abs (rec_eta) > 1.6 && std::abs (rec_eta) < 2.4 ) Endcap = true;
    if (std::abs (rec_eta) > 2.5 && std::abs (rec_eta) < 2.9 ) Forward = true;
    if (std::abs (rec_eta) > 3.1 && std::abs (rec_eta) < 4.7 ) Forward = true;

    // do only barrel for now
    //  if(!Barrel) continue;

    // look for the closets gen Jet : truth
    const GenJet *truth = algo_->matchByDeltaR(&pfj,&genJets);
    if(!truth) continue;   
    double deltaR = algo_->deltaR(&pfj, truth);
    // check deltaR is small enough
    if(deltaR < deltaRMax_ || (abs(rec_eta)>2.5 && deltaR < 0.2) || deltaRMax_ == -1.0 ) {//start case deltaR < deltaRMax

      // generate histograms comparing the reco and truth candidate (truth = closest in delta-R) 
      // get the quantities to place on the denominator and/or divide by
      double pt_denom;
      double true_E = truth->p();
      double true_pt = truth->pt();
      double true_eta = truth->eta();
      double true_phi = truth->phi();

      if (plotAgainstReco_) {pt_denom = rec_pt;}
      else {pt_denom = true_pt;}
      // get true specific quantities
      double true_ChargedHadEnergy;
      double true_NeutralHadEnergy;
      double true_NeutralEmEnergy;
      gettrue (truth, true_ChargedHadEnergy, true_NeutralHadEnergy, true_NeutralEmEnergy);
      double true_NeutralEnergy = true_NeutralHadEnergy + true_NeutralEmEnergy;
      double rec_ChargedHadEnergy = pfj.chargedHadronEnergy();
      double rec_NeutralHadEnergy = pfj.neutralHadronEnergy();
      double rec_NeutralEmEnergy = pfj.neutralEmEnergy();
      double rec_NeutralEnergy = rec_NeutralHadEnergy + rec_NeutralEmEnergy;
      double rec_ChargedMultiplicity = pfj.chargedMultiplicity();
      std::vector <PFCandidatePtr> constituents = pfj.getPFConstituents ();
      std::vector <unsigned int> chMult(9, static_cast<unsigned int>(0)); 
      for (unsigned ic = 0; ic < constituents.size (); ++ic) {
	if ( constituents[ic]->particleId() > 3 ) continue;
	reco::TrackRef trackRef = constituents[ic]->trackRef();
	if ( trackRef.isNull() ) {
	  //std::cout << "Warning in entry " << entry_ 
	  //	    << " : a track with Id " << constituents[ic]->particleId() 
	  //	    << " has no track ref.." << std::endl;
	  continue;
	}
	unsigned int iter = 0; 
	switch (trackRef->algo()) {
	case TrackBase::ctf:
	case TrackBase::initialStep:
	  iter = 0;
	  break;
	case TrackBase::lowPtTripletStep:
	  iter = 1;
	  break;
	case TrackBase::pixelPairStep:
	  iter = 2;
	  break;
	case TrackBase::detachedTripletStep:
	  iter = 3;
	  break;
	case TrackBase::mixedTripletStep:
	  iter = 4;
	  break;
	case TrackBase::pixelLessStep:
	  iter = 5;
	  break;
	case TrackBase::tobTecStep:
	  iter = 6;
	  break;
	case TrackBase::conversionStep:
	  iter = 7;
	  //std::cout << "Warning in entry " << entry_ << " : iter = " << trackRef->algo() << std::endl;
	  //std::cout << ic << " " << *(constituents[ic]) << std::endl;
	  break;
	default:
	  iter = 8;
	  std::cout << "Warning in entry " << entry_ << " : iter = " << trackRef->algo() << std::endl;
	  std::cout << ic << " " << *(constituents[ic]) << std::endl;
	  break;
	}
	++(chMult[iter]);
      }

      bool plot1 = false;
      bool plot2 = false;
      bool plot3 = false;
      bool plot4 = false;
      bool plot5 = false;
      bool plot6 = false;
      bool plot7 = false;
      double cut1 = 0.0001;
      double cut2 = 0.0001;
      double cut3 = 0.0001;
      double cut4 = 0.0001;
      double cut5 = 0.0001;
      double cut6 = 0.0001;
      double cut7 = 0.0001;
      double resPt =0.;
      double resChargedHadEnergy= 0.;
      double resNeutralHadEnergy= 0.;
      double resNeutralEmEnergy= 0.;
      double resNeutralEnergy= 0.;

      double resHCALEnergy = 0.;
      double resNEUTEnergy = 0.;
      if ( rec_NeutralHadEnergy > cut6 && rec_ChargedHadEnergy < cut1 ) { 
	double true_NEUTEnergy = true_NeutralHadEnergy + true_NeutralEmEnergy;
	double true_HCALEnergy = true_NEUTEnergy - rec_NeutralEmEnergy;
	double rec_NEUTEnergy = rec_NeutralHadEnergy+rec_NeutralEmEnergy; 
	double rec_HCALEnergy = rec_NeutralHadEnergy; 
	resHCALEnergy = (rec_HCALEnergy-true_HCALEnergy)/rec_HCALEnergy;
	resNEUTEnergy = (rec_NEUTEnergy-true_NEUTEnergy)/rec_NEUTEnergy;
	if ( rec_NeutralEmEnergy > cut7 ) {
	  plot6 = true;
	} else {
	  plot7 = true;
	}
      }

      // get relative delta quantities (protect against division by zero!)
      if (true_pt > 0.0001){
	resPt = (rec_pt -true_pt)/true_pt ; 
	plot1 = true;}
      if (true_ChargedHadEnergy > cut1){
	resChargedHadEnergy = (rec_ChargedHadEnergy- true_ChargedHadEnergy)/true_ChargedHadEnergy;
	plot2 = true;}
      if (true_NeutralHadEnergy > cut2){
	resNeutralHadEnergy = (rec_NeutralHadEnergy- true_NeutralHadEnergy)/true_NeutralHadEnergy;
	plot3 = true;}
      else 
	if (rec_NeutralHadEnergy > cut3){
	  resNeutralHadEnergy = (rec_NeutralHadEnergy- true_NeutralHadEnergy)/rec_NeutralHadEnergy;
	  plot3 = true;}
      if (true_NeutralEmEnergy > cut4){
	resNeutralEmEnergy = (rec_NeutralEmEnergy- true_NeutralEmEnergy)/true_NeutralEmEnergy;
	plot4 = true;}
      if (true_NeutralEnergy > cut5){
	resNeutralEnergy = (rec_NeutralEnergy- true_NeutralEnergy)/true_NeutralEnergy;
	plot5 = true;}
      
      //double deltaEta = algo_->deltaEta(&pfj, truth);
      //double deltaPhi = algo_->deltaPhi(&pfj, truth);

      // Print outliers for further debugging
      if ( ( resPt > 0.2 && true_pt > 100. ) || 
	   ( resPt < -0.5 && true_pt > 100. ) ) {
	//if ( ( true_pt > 50. && 
	//     ( ( truth->eta()>3.0 && rec_eta-truth->eta() < -0.1 ) || 
	//       ( truth->eta()<-3.0 && rec_eta-truth->eta() > 0.1 ) ))) {
	std::cout << "Entry " << entry_ 
		  << " resPt = " << resPt
		  <<" resCharged  " << resChargedHadEnergy
		  <<" resNeutralHad  " << resNeutralHadEnergy
		  << " resNeutralEm  " << resNeutralEmEnergy
		  << " pT (T/R) " << true_pt << "/" << rec_pt 
		  << " Eta (T/R) " << truth->eta() << "/" << rec_eta 
		  << " Phi (T/R) " << truth->phi() << "/" << rec_phi 
		  << std::endl;

	// check overlapping PF jets
	const reco::PFJet* pfoj = 0; 
	double dRo = 1E9;
	for(unsigned j=0; j<pfJets.size(); j++) { 
	  const reco::PFJet& pfo = pfJets[j];
	  if ( j != i &&  algo_->deltaR(&pfj,&pfo) < dRo && pfo.pt() > 0.25*pfj.pt()) { 
	    dRo = algo_->deltaR(&pfj,&pfo);	
	    pfoj = &pfo;
	  }
	}
	
	// Check overlapping Gen Jet 
	math::XYZTLorentzVector overlappinGenJet(0.,0.,0.,0.);
	const reco::GenJet* genoj = 0;
	double dRgo = 1E9;
	for(unsigned j=0; j<genJets.size(); j++) { 
	  const reco::GenJet* gjo = &(genJets[j]);
	  if ( gjo != truth && algo_->deltaR(truth,gjo) < dRgo && gjo->pt() > 0.25*truth->pt() ) { 
	    dRgo = algo_->deltaR(truth,gjo);
	    genoj = gjo;
	  }
	}
	
	if ( dRo < 0.8 && dRgo < 0.8 && algo_->deltaR(genoj,pfoj) < 2.*deltaRMax_ ) 
	  std::cout << "Excess probably due to overlapping jets (DR = " <<   algo_->deltaR(genoj,pfoj) << "),"
		    << " at DeltaR(T/R) = " << dRgo << "/" << dRo  
		    << " with pT(T/R) " << genoj->pt() << "/" << pfoj->pt()
		    << " and Eta (T/R) " << genoj->eta() << "/" << pfoj->eta()
		    << " and Phi (T/R) " << genoj->phi() << "/" << pfoj->phi()
		    << std::endl;
      }

      if(std::abs(resPt) > std::abs(resPtMax_)) resPtMax_ = resPt;
      if(std::abs(resChargedHadEnergy) > std::abs(resChargedHadEnergyMax_) ) resChargedHadEnergyMax_ = resChargedHadEnergy;
      if(std::abs(resNeutralHadEnergy) > std::abs(resNeutralHadEnergyMax_) ) resNeutralHadEnergyMax_ = resNeutralHadEnergy;
      if(std::abs(resNeutralEmEnergy) > std::abs(resNeutralEmEnergyMax_) ) resNeutralEmEnergyMax_ = resNeutralEmEnergy;
      if (debug_) {
	cout << i <<"  =========PFJet Pt "<< rec_pt
	     << " eta " << rec_eta
	     << " phi " << rec_phi
	     << " Charged Had Energy " << rec_ChargedHadEnergy
	     << " Neutral Had Energy " << rec_NeutralHadEnergy
	     << " Neutral elm Energy " << rec_NeutralEmEnergy << endl;
	cout << " matching Gen Jet Pt " << true_pt
	     << " eta " << truth->eta()
	     << " phi " << truth->phi()
	     << " Charged Had Energy " << true_ChargedHadEnergy
	     << " Neutral Had Energy " << true_NeutralHadEnergy
	     << " Neutral elm Energy " << true_NeutralEmEnergy << endl;
	printPFJet(&pfj);
	//      cout<<pfj.print()<<endl;
	printGenJet(truth);
	//cout <<truth->print()<<endl;
				
	cout << "==============deltaR " << deltaR << "  resPt " << resPt
	     << " resChargedHadEnergy " << resChargedHadEnergy
	     << " resNeutralHadEnergy " << resNeutralHadEnergy
	     << " resNeutralEmEnergy " << resNeutralEmEnergy
	     << endl;
      }
			

      if(plot1) {
	if ( rec_eta > 0. ) 
	  hDEtavsEta->Fill(true_eta,rec_eta-true_eta);
	else
	  hDEtavsEta->Fill(true_eta,-rec_eta+true_eta);
	hDPhivsEta->Fill(true_eta,rec_phi-true_phi);

	hRPtvsEta->Fill(true_eta, resPt);
	hNCHvsEta->Fill(true_eta, rec_ChargedMultiplicity);
	hNCH0vsEta->Fill(true_eta,chMult[0]);
	hNCH1vsEta->Fill(true_eta,chMult[1]);
	hNCH2vsEta->Fill(true_eta,chMult[2]);
	hNCH3vsEta->Fill(true_eta,chMult[3]);
	hNCH4vsEta->Fill(true_eta,chMult[4]);
	hNCH5vsEta->Fill(true_eta,chMult[5]);
	hNCH6vsEta->Fill(true_eta,chMult[6]);
	hNCH7vsEta->Fill(true_eta,chMult[7]);
      }
      if(plot2)hRCHEvsEta->Fill(true_eta, resChargedHadEnergy);
      if(plot5)hRNeutvsEta->Fill(true_eta, resNeutralEnergy);
      if(plot6) { 
	hRHCALvsEta->Fill(true_eta, resHCALEnergy);
	hRNEUTvsEta->Fill(true_eta, resNEUTEnergy);
      }
      if(plot7) {  
	hRHONLvsEta->Fill(true_eta, resHCALEnergy);
	hRNONLvsEta->Fill(true_eta, resNEUTEnergy);
      }

      // fill histograms for relative delta quantitites of matched jets
      // delta Pt or E quantities for Barrel
      if (Barrel){
	if(plot1) { 
	  hBRPt->Fill (resPt);
	  if ( pt_denom >  20. && pt_denom <  40. ) hBRPt20_40->Fill (resPt);
	  if ( pt_denom >  40. && pt_denom <  60. ) hBRPt40_60->Fill (resPt);
	  if ( pt_denom >  60. && pt_denom <  80. ) hBRPt60_80->Fill (resPt);
	  if ( pt_denom >  80. && pt_denom < 100. ) hBRPt80_100->Fill (resPt);
	  if ( pt_denom > 100. && pt_denom < 150. ) hBRPt100_150->Fill (resPt);
	  if ( pt_denom > 150. && pt_denom < 200. ) hBRPt150_200->Fill (resPt);
	  if ( pt_denom > 200. && pt_denom < 250. ) hBRPt200_250->Fill (resPt);
	  if ( pt_denom > 250. && pt_denom < 300. ) hBRPt250_300->Fill (resPt);
	  if ( pt_denom > 300. && pt_denom < 400. ) hBRPt300_400->Fill (resPt);
	  if ( pt_denom > 400. && pt_denom < 500. ) hBRPt400_500->Fill (resPt);
	  if ( pt_denom > 500. && pt_denom < 750. ) hBRPt500_750->Fill (resPt);
	  if ( pt_denom > 750. && pt_denom < 1250. ) hBRPt750_1250->Fill (resPt);
	  if ( pt_denom > 1250. && pt_denom < 2000. ) hBRPt1250_2000->Fill (resPt);
	  if ( pt_denom > 2000. && pt_denom < 5000. ) hBRPt2000_5000->Fill (resPt);
	  hBNCH->Fill(rec_ChargedMultiplicity);
	  hBNCH0vsPt->Fill(pt_denom,chMult[0]);
	  hBNCH1vsPt->Fill(pt_denom,chMult[1]);
	  hBNCH2vsPt->Fill(pt_denom,chMult[2]);
	  hBNCH3vsPt->Fill(pt_denom,chMult[3]);
	  hBNCH4vsPt->Fill(pt_denom,chMult[4]);
	  hBNCH5vsPt->Fill(pt_denom,chMult[5]);
	  hBNCH6vsPt->Fill(pt_denom,chMult[6]);
	  hBNCH7vsPt->Fill(pt_denom,chMult[7]);
	  hBNCHvsPt->Fill(pt_denom,rec_ChargedMultiplicity);
	  if ( rec_eta > 0. ) 
	    hBDEtavsPt->Fill(pt_denom,rec_eta-true_eta);
	  else
	    hBDEtavsPt->Fill(pt_denom,-rec_eta+true_eta);
	  hBDPhivsPt->Fill(pt_denom,rec_phi-true_phi);
	}
	if(plot2)hBRCHE->Fill(resChargedHadEnergy);
	if(plot3)hBRNHE->Fill(resNeutralHadEnergy);
	if(plot4)hBRNEE->Fill(resNeutralEmEnergy);
	if(plot5)hBRneut->Fill(resNeutralEnergy);
	if(plot1)hBRPtvsPt->Fill(pt_denom, resPt);
	if(plot2)hBRCHEvsPt->Fill(pt_denom, resChargedHadEnergy);
	if(plot3)hBRNHEvsPt->Fill(pt_denom, resNeutralHadEnergy);
	if(plot4)hBRNEEvsPt->Fill(pt_denom, resNeutralEmEnergy);
	if(plot5)hBRneutvsPt->Fill(pt_denom, resNeutralEnergy);
	if(plot6) { 
	  hBRHCALvsP->Fill(true_E, resHCALEnergy);
	  hBRNEUTvsP->Fill(true_E, resNEUTEnergy);
	}
	if(plot7) { 
	  hBRHONLvsP->Fill(true_E, resHCALEnergy);
	  hBRNONLvsP->Fill(true_E, resNEUTEnergy);
	}

      }
      // delta Pt or E quantities for Endcap
      if (Endcap){
	if(plot1) {
	  hERPt->Fill (resPt);
	  if ( pt_denom >  20. && pt_denom <  40. ) hERPt20_40->Fill (resPt);
	  if ( pt_denom >  40. && pt_denom <  60. ) hERPt40_60->Fill (resPt);
	  if ( pt_denom >  60. && pt_denom <  80. ) hERPt60_80->Fill (resPt);
	  if ( pt_denom >  80. && pt_denom < 100. ) hERPt80_100->Fill (resPt);
	  if ( pt_denom > 100. && pt_denom < 150. ) hERPt100_150->Fill (resPt);
	  if ( pt_denom > 150. && pt_denom < 200. ) hERPt150_200->Fill (resPt);
	  if ( pt_denom > 200. && pt_denom < 250. ) hERPt200_250->Fill (resPt);
	  if ( pt_denom > 250. && pt_denom < 300. ) hERPt250_300->Fill (resPt);
	  if ( pt_denom > 300. && pt_denom < 400. ) hERPt300_400->Fill (resPt);
	  if ( pt_denom > 400. && pt_denom < 500. ) hERPt400_500->Fill (resPt);
	  if ( pt_denom > 500. && pt_denom < 750. ) hERPt500_750->Fill (resPt);
	  if ( pt_denom > 750. && pt_denom < 1250. ) hERPt750_1250->Fill (resPt);
	  if ( pt_denom > 1250. && pt_denom < 2000. ) hERPt1250_2000->Fill (resPt);
	  if ( pt_denom > 2000. && pt_denom < 5000. ) hERPt2000_5000->Fill (resPt);
	  hENCH->Fill(rec_ChargedMultiplicity);
	  hENCHvsPt->Fill(pt_denom,rec_ChargedMultiplicity);
	  hENCH0vsPt->Fill(pt_denom,chMult[0]);
	  hENCH1vsPt->Fill(pt_denom,chMult[1]);
	  hENCH2vsPt->Fill(pt_denom,chMult[2]);
	  hENCH3vsPt->Fill(pt_denom,chMult[3]);
	  hENCH4vsPt->Fill(pt_denom,chMult[4]);
	  hENCH5vsPt->Fill(pt_denom,chMult[5]);
	  hENCH6vsPt->Fill(pt_denom,chMult[6]);
	  hENCH7vsPt->Fill(pt_denom,chMult[7]);
	  if ( rec_eta > 0. ) 
	    hEDEtavsPt->Fill(pt_denom,rec_eta-true_eta);
	  else
	    hEDEtavsPt->Fill(pt_denom,-rec_eta+true_eta);
	  hEDPhivsPt->Fill(pt_denom,rec_phi-true_phi);
	}
	if(plot2)hERCHE->Fill(resChargedHadEnergy);
	if(plot3)hERNHE->Fill(resNeutralHadEnergy);
	if(plot4)hERNEE->Fill(resNeutralEmEnergy);
	if(plot5)hERneut->Fill(resNeutralEnergy);
	if(plot1)hERPtvsPt->Fill(pt_denom, resPt);
	if(plot2)hERCHEvsPt->Fill(pt_denom, resChargedHadEnergy);
	if(plot3)hERNHEvsPt->Fill(pt_denom, resNeutralHadEnergy);
	if(plot4)hERNEEvsPt->Fill(pt_denom, resNeutralEmEnergy);
	if(plot5)hERneutvsPt->Fill(pt_denom, resNeutralEnergy);
	if(plot6) {
	  hERHCALvsP->Fill(true_E, resHCALEnergy);
	  hERNEUTvsP->Fill(true_E, resNEUTEnergy);
	}
	if(plot7) {
	  hERHONLvsP->Fill(true_E, resHCALEnergy);
	  hERNONLvsP->Fill(true_E, resNEUTEnergy);
	}
      }						
      // delta Pt or E quantities for Forward
      if (Forward){
	if(plot1) {
	  hFRPt->Fill (resPt);
	  if ( pt_denom >  20. && pt_denom <  40. ) hFRPt20_40->Fill (resPt);
	  if ( pt_denom >  40. && pt_denom <  60. ) hFRPt40_60->Fill (resPt);
	  if ( pt_denom >  60. && pt_denom <  80. ) hFRPt60_80->Fill (resPt);
	  if ( pt_denom >  80. && pt_denom < 100. ) hFRPt80_100->Fill (resPt);
	  if ( pt_denom > 100. && pt_denom < 150. ) hFRPt100_150->Fill (resPt);
	  if ( pt_denom > 150. && pt_denom < 200. ) hFRPt150_200->Fill (resPt);
	  if ( pt_denom > 200. && pt_denom < 250. ) hFRPt200_250->Fill (resPt);
	  if ( pt_denom > 250. && pt_denom < 300. ) hFRPt250_300->Fill (resPt);
	  if ( pt_denom > 300. && pt_denom < 400. ) hFRPt300_400->Fill (resPt);
	  if ( pt_denom > 400. && pt_denom < 500. ) hFRPt400_500->Fill (resPt);
	  if ( pt_denom > 500. && pt_denom < 750. ) hFRPt500_750->Fill (resPt);
	  if ( pt_denom > 750. && pt_denom < 1250. ) hFRPt750_1250->Fill (resPt);
	  if ( pt_denom > 1250. && pt_denom < 2000. ) hFRPt1250_2000->Fill (resPt);
	  if ( pt_denom > 2000. && pt_denom < 5000. ) hFRPt2000_5000->Fill (resPt);
	  if ( rec_eta > 0. ) 
	    hFDEtavsPt->Fill(pt_denom,rec_eta-true_eta);
	  else
	    hFDEtavsPt->Fill(pt_denom,-rec_eta+true_eta);
	  hFDPhivsPt->Fill(pt_denom,rec_phi-true_phi);
	}
	if(plot2)hFRCHE->Fill(resChargedHadEnergy);
	if(plot3)hFRNHE->Fill(resNeutralHadEnergy);
	if(plot4)hFRNEE->Fill(resNeutralEmEnergy);
	if(plot5)hFRneut->Fill(resNeutralEnergy);
	if(plot1)hFRPtvsPt->Fill(pt_denom, resPt);
	if(plot2)hFRCHEvsPt->Fill(pt_denom, resChargedHadEnergy);
	if(plot3)hFRNHEvsPt->Fill(pt_denom, resNeutralHadEnergy);
	if(plot4)hFRNEEvsPt->Fill(pt_denom, resNeutralEmEnergy);
	if(plot5)hFRneutvsPt->Fill(pt_denom, resNeutralEnergy);
	if(plot6) {
	  hFRHCALvsP->Fill(true_E, resHCALEnergy);
	  hFRNEUTvsP->Fill(true_E, resNEUTEnergy);
	}
	if(plot7) {
	  hFRHONLvsP->Fill(true_E, resHCALEnergy);
	  hFRNONLvsP->Fill(true_E, resNEUTEnergy);
	}
      }						
    } // end case deltaR < deltaRMax
		
  } // i loop on pf Jets	

  // Increment counter
  entry_++;
}

void PFJetBenchmark::gettrue (const reco::GenJet* truth, double& true_ChargedHadEnergy, 
			      double& true_NeutralHadEnergy, double& true_NeutralEmEnergy){
  std::vector <const GenParticle*> mcparts = truth->getGenConstituents ();
  true_NeutralEmEnergy = 0.;
  true_ChargedHadEnergy = 0.;
  true_NeutralHadEnergy = 0.;
  // for each MC particle in turn  
  for (unsigned i = 0; i < mcparts.size (); i++) {
    const GenParticle* mcpart = mcparts[i];
    int PDG = std::abs( mcpart->pdgId());
    double e = mcpart->energy(); 
    switch(PDG){  // start PDG switch
    case 22: // photon
      true_NeutralEmEnergy += e;
      break;
    case 211: // pi
    case 321: // K
    case 2212: // p
    case 11: //electrons (until recognised)
      true_ChargedHadEnergy += e;
      break;
    case 310: // K_S0
    case 130: // K_L0
    case 3122: // Lambda0
    case 2112: // n0
      true_NeutralHadEnergy += e;
    default:
      break;
    }  // end PDG switch		
  }  // end loop on constituents.
}

void PFJetBenchmark::printPFJet(const reco::PFJet* pfj){
  cout<<setiosflags(ios::right);
  cout<<setiosflags(ios::fixed);
  cout<<setprecision(3);

  cout << "PFJet  p/px/py/pz/pt: " << pfj->p() << "/" << pfj->px () 
       << "/" << pfj->py() << "/" << pfj->pz() << "/" << pfj->pt() << endl
       << "    eta/phi: " << pfj->eta () << "/" << pfj->phi () << endl   		
       << "    PFJet specific:" << std::endl
       << "      charged/neutral hadrons energy: " << pfj->chargedHadronEnergy () << '/' << pfj->neutralHadronEnergy () << endl
       << "      charged/neutral em energy: " << pfj->chargedEmEnergy () << '/' << pfj->neutralEmEnergy () << endl
       << "      charged muon energy: " << pfj->chargedMuEnergy () << '/' << endl
       << "      charged/neutral multiplicity: " << pfj->chargedMultiplicity () << '/' << pfj->neutralMultiplicity () << endl;
  
  // And print the constituents
  std::cout << pfj->print() << std::endl;

  cout<<resetiosflags(ios::right|ios::fixed);
}


void PFJetBenchmark::printGenJet (const reco::GenJet* truth){
  std::vector <const GenParticle*> mcparts = truth->getGenConstituents ();
  cout << "GenJet p/px/py/pz/pt: " << truth->p() << '/' << truth->px () 
       << '/' << truth->py() << '/' << truth->pz() << '/' << truth->pt() << endl
       << "    eta/phi: " << truth->eta () << '/' << truth->phi () << endl
       << "    # of constituents: " << mcparts.size() << endl;
  cout << "    constituents:" << endl;
  for (unsigned i = 0; i < mcparts.size (); i++) {
    const GenParticle* mcpart = mcparts[i];
    cout << "      #" << i << "  PDG code:" << mcpart->pdgId() 
	 << ", p/pt/eta/phi: " << mcpart->p() << '/' << mcpart->pt() 
	 << '/' << mcpart->eta() << '/' << mcpart->phi() << endl;	
  }    
}

