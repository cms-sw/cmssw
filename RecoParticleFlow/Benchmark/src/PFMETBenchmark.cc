#include "RecoParticleFlow/Benchmark/interface/PFMETBenchmark.h"
//#include "TCanvas.h"
#include "TProfile.h"
#include "TF1.h"
#include "TH1F.h"

// preprocessor macro for booking 1d histos with DQMStore -or- bare Root
#define BOOK1D(name,title,nbinsx,lowx,highx) \
  h##name = dbe_ ? dbe_->book1D(#name,title,nbinsx,lowx,highx)->getTH1F() \
    : new TH1F(#name,title,nbinsx,lowx,highx)

// preprocessor macro for booking 2d histos with DQMStore -or- bare Root
#define BOOK2D(name,title,nbinsx,lowx,highx,nbinsy,lowy,highy) \
  h##name = dbe_ ? dbe_->book2D(#name,title,nbinsx,lowx,highx,nbinsy,lowy,highy)->getTH2F() \
    : new TH2F(#name,title,nbinsx,lowx,highx,nbinsy,lowy,highy)

// all versions OK
// preprocesor macro for setting axis titles
#define SETAXES(name,xtitle,ytitle) \
  h##name->GetXaxis()->SetTitle(xtitle); h##name->GetYaxis()->SetTitle(ytitle)


/*#define SET2AXES(name,xtitle,ytitle) \
  hE##name->GetXaxis()->SetTitle(xtitle); hE##name->GetYaxis()->SetTitle(ytitle);  hB##name->GetXaxis()->SetTitle(xtitle); hB##name->GetYaxis()->SetTitle(ytitle)
*/

#define PT (plotAgainstReco_)?"reconstructed P_{T}" :"generated P_{T}"

using namespace reco;
using namespace std;

class MonitorElement;

PFMETBenchmark::PFMETBenchmark() : file_(0) {}

PFMETBenchmark::~PFMETBenchmark() {
  if(file_) file_->Close();
}

void PFMETBenchmark::write() {
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

void PFMETBenchmark::setup(
			   string Filename,
			   bool debug, 
			   bool plotAgainstReco,
			   string benchmarkLabel_, 
			   DQMStore * dbe_store) {
  debug_ = debug; 
  plotAgainstReco_ = plotAgainstReco;
  outputFile_=Filename;
  file_ = NULL;
  dbe_ = dbe_store;
  // print parameters
  //cout<< "PFMETBenchmark Setup parameters =============================================="<<endl;
  cout << "Filename to write histograms " << Filename<<endl;
  cout << "PFMETBenchmark debug " << debug_<< endl;
  cout << "plotAgainstReco " << plotAgainstReco_ << endl;
  cout << "benchmarkLabel " << benchmarkLabel_ << endl;
  
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

  // delta Pt or E quantities for Barrel
  BOOK1D(MEX,"Particle Flow",400,-200,200);
  BOOK1D(DeltaMEX,"Particle Flow",400,-200,200);
  BOOK1D(DeltaMET,"Particle Flow",400,-200,200);
  BOOK1D(DeltaPhi,"Particle Flow", 1000, -3.2, 3.2);
  BOOK1D(DeltaSET,"Particle Flow",400,-200,200);
  BOOK2D(SETvsDeltaMET,"Particle Flow",200, 0.0, 1000.0, 400, -200.0, 200.0);        
  BOOK2D(SETvsDeltaSET,"Particle Flow",200, 0.0, 1000.0, 400, -200.0, 200.0);       
  profileSETvsSETresp = new TProfile("#DeltaPFSET / trueSET vs trueSET", "", 200, 0.0, 1000.0, -1.0, 1.0);
  profileMETvsMETresp = new TProfile("#DeltaPFMET / trueMET vs trueMET", "", 50, 0.0,  200.0, -1.0, 1.0);
	
  BOOK1D(CaloMEX,"Calorimeter",400,-200,200);
  BOOK1D(DeltaCaloMEX,"Calorimeter",400,-200,200);
  BOOK1D(DeltaCaloMET,"Calorimeter",400,-200,200);
  BOOK1D(DeltaCaloPhi,"Calorimeter", 1000, -3.2, 3.2);
  BOOK1D(DeltaCaloSET,"Calorimeter",400,-200,200);
  BOOK2D(CaloSETvsDeltaCaloMET,"Calorimeter",200, 0.0, 1000.0, 400, -200.0, 200.0);        
  BOOK2D(CaloSETvsDeltaCaloSET,"Calorimeter",200, 0.0, 1000.0, 400, -200.0, 200.0);       
  profileCaloSETvsCaloSETresp = new TProfile("#DeltaCaloSET / trueSET vs trueSET", "", 200, 0.0, 1000.0, -1.0, 1.0);
  profileCaloMETvsCaloMETresp = new TProfile("#DeltaCaloMET / trueMET vs trueMET", "", 200, 0.0,  200.0, -1.0, 1.0);

  BOOK2D(DeltaPFMETvstrueMET,"Particle Flow", 500, 0.0, 1000.0, 400, -200.0, 200.0);      
  BOOK2D(DeltaCaloMETvstrueMET,"Calorimeter", 500, 0.0, 1000.0, 400, -200.0, 200.0);      
  BOOK2D(DeltaPFPhivstrueMET,"Particle Flow", 500, 0.0, 1000.0, 400, -3.2, 3.2);      
  BOOK2D(DeltaCaloPhivstrueMET,"Calorimeter", 500, 0.0, 1000.0, 400, -3.2, 3.2);      
  BOOK2D(CaloMETvstrueMET,"Calorimeter", 500, 0.0, 1000.0, 500, 0.0, 1000.0);      
  BOOK2D(PFMETvstrueMET,"Particle Flow", 500, 0.0, 1000.0, 500, 0.0, 1000.0);

  BOOK2D(DeltaCaloMEXvstrueSET,"Calorimeter",200, 0.0, 1000.0, 400, -200.0, 200.0);
  BOOK2D(DeltaPFMEXvstrueSET,"Particle Flow",200, 0.0, 1000.0, 400, -200.0, 200.0);
   
  BOOK1D(TrueMET,"MC truth", 500, 0.0, 1000.0);      
  BOOK1D(CaloMET,"Calorimeter", 500, 0.0, 1000.0);
  BOOK1D(PFMET,"Particle Flow", 500, 0.0, 1000.0);

  BOOK1D(TCMEX,"Track Corrected",400,-200,200);
  BOOK1D(DeltaTCMEX,"Track Corrected",400,-200,200);
  BOOK1D(DeltaTCMET,"Track Corrected",400,-200,200);
  BOOK1D(DeltaTCPhi,"Track Corrected", 1000, -3.2, 3.2);
  BOOK1D(DeltaTCSET,"Track Corrected",400,-200,200);
  BOOK2D(TCSETvsDeltaTCMET,"Track Corrected",200, 0.0, 1000.0, 400, -200.0, 200.0);        
  BOOK2D(TCSETvsDeltaTCSET,"Track Corrected",200, 0.0, 1000.0, 400, -200.0, 200.0);       
  profileTCSETvsTCSETresp = new TProfile("#DeltaTCSET / trueSET vs trueSET", "", 200, 0.0, 1000.0, -1.0, 1.0);
  profileTCMETvsTCMETresp = new TProfile("#DeltaTCMET / trueMET vs trueMET", "", 200, 0.0,  200.0, -1.0, 1.0);
	
  BOOK1D(TCMET,"Track Corrected",500,0,1000);
  BOOK2D(DeltaTCMETvstrueMET,"Track Corrected", 500, 0.0, 1000.0, 400, -200.0, 200.0);
  BOOK2D(DeltaTCPhivstrueMET,"Track Corrected", 500, 0.0, 1000.0, 400, -3.2, 3.2);

  BOOK2D(DeltaTCMEXvstrueSET,"Track Corrected", 200, 0.0, 1000.0, 400, -200.0, 200.0);
  BOOK2D(TCMETvstrueMET,"Track Corrected", 200, 0.0, 1000.0, 500, 0.0, 1000.0);

  //BOOK1D(meanPF,    "Mean PFMEX", 100, 0.0, 1600.0);
  //BOOK1D(meanCalo,  "Mean CaloMEX", 100, 0.0, 1600.0);
  //BOOK1D(sigmaPF,   "#sigma(PFMEX)", 100, 0.0, 1600.0);
  //BOOK1D(sigmaCalo, "#sigma(CaloMEX)", 100, 0.0, 1600.0);
  //BOOK1D(rmsPF,     "RMS(PFMEX)", 100, 0.0, 1600.0);
  //BOOK1D(rmsCalo,   "RMS(CaloMEX)", 100, 0.0, 1600.0);

  // Set Axis Titles
  // delta Pt or E quantities for Barrel and Endcap
  SETAXES(MEX, "MEX",  "Events");
  SETAXES(DeltaMEX, "#DeltaMEX",  "Events");
  SETAXES(DeltaMET, "#DeltaMET",  "Events");
  SETAXES(DeltaPhi, "#Delta#phi", "Events");
  SETAXES(DeltaSET, "#DeltaSET",  "Events");
  SETAXES(SETvsDeltaMET, "SET", "#DeltaMET");
  SETAXES(SETvsDeltaSET, "SET", "#DeltaSET");

  SETAXES(DeltaPFMETvstrueMET, "trueMET", "#DeltaMET");
  SETAXES(DeltaCaloMETvstrueMET, "trueMET", "#DeltaCaloMET");
  SETAXES(DeltaPFPhivstrueMET, "trueMET", "#DeltaPFPhi");
  SETAXES(DeltaCaloPhivstrueMET, "trueMET", "#DeltaCaloPhi");
  SETAXES(CaloMETvstrueMET, "trueMET", "CaloMET");
  SETAXES(PFMETvstrueMET, "trueMET", "PFMET");
  SETAXES(DeltaCaloMEXvstrueSET, "trueSET", "#DeltaCaloMEX");
  SETAXES(DeltaPFMEXvstrueSET, "trueSET", "#DeltaPFMEX");
  SETAXES(TrueMET, "trueMET", "Events");
  SETAXES(CaloMET, "CaloMET", "Events");
  SETAXES(PFMET, "PFMET", "Events");
  SETAXES(TCMET, "TCMET", "Events");

  SETAXES(CaloMEX, "MEX",  "Events");
  SETAXES(DeltaCaloMEX, "#DeltaMEX",  "Events");
  SETAXES(DeltaCaloMET, "#DeltaMET",  "Events");
  SETAXES(DeltaCaloPhi, "#Delta#phi", "Events");
  SETAXES(DeltaCaloSET, "#DeltaSET",  "Events");
  SETAXES(CaloSETvsDeltaCaloMET, "SET", "#DeltaMET");
  SETAXES(CaloSETvsDeltaCaloSET, "SET", "#DeltaSET");

  SETAXES(TCMEX, "MEX",  "Events");
  SETAXES(DeltaTCMEX, "#DeltaMEX",  "Events");
  SETAXES(DeltaTCMET, "#DeltaMET",  "Events");
  SETAXES(DeltaTCPhi, "#Delta#phi", "Events");
  SETAXES(DeltaTCSET, "#DeltaSET",  "Events");
  SETAXES(TCSETvsDeltaTCMET, "SET", "#DeltaMET");
  SETAXES(TCSETvsDeltaTCSET, "SET", "#DeltaSET");

  SETAXES(DeltaTCMETvstrueMET, "trueMET", "#DeltaTCMET");
  SETAXES(DeltaTCPhivstrueMET, "trueMET", "#DeltaTCPhi");

  SETAXES(DeltaTCMEXvstrueSET, "trueSET", "#DeltaTCMEX");
  SETAXES(TCMETvstrueMET, "trueMET", "TCMET");
}


//void PFMETBenchmark::process(const reco::PFMETCollection& pfMets, const reco::GenMETCollection& genMets) {
void PFMETBenchmark::process( const reco::PFMETCollection& pfMets, 
			      const reco::GenParticleCollection& genParticleList, 
			      const reco::CaloMETCollection& caloMets,
			      const reco::METCollection& tcMets ) {
  calculateQuantities(pfMets, genParticleList, caloMets, tcMets);
  if (debug_) {
    cout << "  =========PFMET  " << rec_met  << ", " << rec_phi  << endl;
    cout << "  =========GenMET " << true_met << ", " << true_phi << endl;
    cout << "  =========CaloMET " << calo_met << ", " << calo_phi << endl;
    cout << "  =========TCMET " << tc_met << ", " << tc_phi << endl;
  }			
  // fill histograms
  hTrueMET->Fill(true_met);
  // delta Pt or E quantities
  // PF
  hDeltaMET->Fill( rec_met - true_met );
  hMEX->Fill( rec_mex );
  hMEX->Fill( rec_mey );
  hDeltaMEX->Fill( rec_mex - true_mex );
  hDeltaMEX->Fill( rec_mey - true_mey );
  hDeltaPhi->Fill( rec_phi - true_phi );
  hDeltaSET->Fill( rec_set - true_set );
  if( true_met > 5.0 ) hSETvsDeltaMET->Fill( rec_set, rec_met - true_met );
  else                 hSETvsDeltaMET->Fill( rec_set, rec_mex - true_mex );
  hSETvsDeltaSET->Fill( rec_set, rec_set - true_set );
  if( true_met > 5.0 ) profileMETvsMETresp->Fill(true_met, (rec_met-true_met)/true_met);
  profileSETvsSETresp->Fill(true_set, (rec_set-true_set)/true_set);

  hDeltaPFMETvstrueMET->Fill(true_met, rec_met - true_met);
  hDeltaPFPhivstrueMET->Fill(true_met, rec_phi - true_phi);
  hPFMETvstrueMET->Fill(true_met, rec_met);
  hPFMET->Fill(rec_met);

  hDeltaPFMEXvstrueSET->Fill(true_set, rec_mex - true_mex);
  hDeltaPFMEXvstrueSET->Fill(true_set, rec_mey - true_mey);

  // Calo
  hDeltaCaloMET->Fill( calo_met - true_met );
  hCaloMEX->Fill( calo_mex );
  hCaloMEX->Fill( calo_mey );
  hDeltaCaloMEX->Fill( calo_mex - true_mex);
  hDeltaCaloMEX->Fill( calo_mey - true_mey);
  hDeltaCaloPhi->Fill( calo_phi - true_phi );
  hDeltaCaloSET->Fill( calo_set - true_set );
  if( true_met > 5.0 ) hCaloSETvsDeltaCaloMET->Fill( calo_set, calo_met - true_met );
  else                 hCaloSETvsDeltaCaloMET->Fill( calo_set, calo_mex - true_mex);
  hCaloSETvsDeltaCaloSET->Fill( calo_set, calo_set - true_set );
  if( true_met > 5.0 ) profileCaloMETvsCaloMETresp->Fill(true_met, (calo_met-true_met)/true_met);
  profileCaloSETvsCaloSETresp->Fill(true_set, (calo_set-true_set)/true_set);

  hDeltaCaloMETvstrueMET->Fill(true_met, calo_met - true_met);
  hDeltaCaloPhivstrueMET->Fill(true_met, calo_phi - true_phi);
  hCaloMETvstrueMET->Fill(true_met, calo_met);
  hCaloMET->Fill(calo_met);

  hDeltaCaloMEXvstrueSET->Fill(true_set, calo_mex - true_mex);
  hDeltaCaloMEXvstrueSET->Fill(true_set, calo_mey - true_mey);

  // TC
  hDeltaTCMET->Fill( tc_met - true_met );
  hTCMET->Fill( tc_met );
  hTCMEX->Fill( tc_mex );
  hTCMEX->Fill( tc_mey );
  hDeltaTCMEX->Fill( tc_mex - true_mex);
  hDeltaTCMEX->Fill( tc_mey - true_mey);
  hDeltaTCPhi->Fill( tc_phi - true_phi );
  hDeltaTCSET->Fill( tc_set - true_set );
  if( true_met > 5.0 ) hTCSETvsDeltaTCMET->Fill( tc_set, tc_met - true_met );
  else                 hTCSETvsDeltaTCMET->Fill( tc_set, tc_mex - true_mex);
  hTCSETvsDeltaTCSET->Fill( tc_set, tc_set - true_set );
  if( true_met > 5.0 ) profileTCMETvsTCMETresp->Fill(true_met, (tc_met-true_met)/true_met);
  profileTCSETvsTCSETresp->Fill(true_set, (tc_set-true_set)/true_set);

  hDeltaTCMETvstrueMET->Fill( true_met, tc_met - true_met);
  hDeltaTCPhivstrueMET->Fill( true_met, tc_phi - true_phi);
  hDeltaTCMEXvstrueSET->Fill( true_set, tc_mex - true_mex);
  hDeltaTCMEXvstrueSET->Fill( true_set, tc_mey - true_mey);
  hTCMETvstrueMET->Fill( true_met, tc_met );

}

void PFMETBenchmark::calculateQuantities( const reco::PFMETCollection& pfMets, 
					  const reco::GenParticleCollection& genParticleList, 
					  const reco::CaloMETCollection& caloMets,
					  const reco::METCollection& tcMets) 
{
  const reco::PFMET&    pfm = pfMets[0];
  const reco::CaloMET&  cm  = caloMets[0];
  const reco::MET&  tcm  = tcMets[0];

  double trueMEY  = 0.0;
  double trueMEX  = 0.0;;
  true_set  = 0.0;;

  //  for( genParticle = genParticleList.begin(); genParticle != genParticleList.end(); genParticle++ )
  for( unsigned i = 0; i < genParticleList.size(); i++ ) {
    if( genParticleList[i].status() == 1 && fabs(genParticleList[i].eta()) < 5.0 ) { 
      if( std::abs(genParticleList[i].pdgId()) == 12 ||
	  std::abs(genParticleList[i].pdgId()) == 14 ||
	  std::abs(genParticleList[i].pdgId()) == 16 || 
	  std::abs(genParticleList[i].pdgId()) < 7   || 
	  std::abs(genParticleList[i].pdgId()) == 21 ) {
	trueMEX += genParticleList[i].px();
	trueMEY += genParticleList[i].py();
      } else {
	true_set += genParticleList[i].pt();
      }
    }
  }
  true_mex = -trueMEX;
  true_mey = -trueMEY;
  true_met = sqrt( trueMEX*trueMEX + trueMEY*trueMEY );
  true_phi = atan2(trueMEY,trueMEX);
  rec_met  = pfm.pt();
  rec_mex  = pfm.px();
  rec_mex  = pfm.py();
  rec_phi  = pfm.phi();
  rec_set  = pfm.sumEt();
  calo_met = cm.pt();
  calo_mex = cm.px();
  calo_mey = cm.py();
  calo_phi = cm.phi();
  calo_set = cm.sumEt();
  tc_met = tcm.pt();
  tc_mex = tcm.px();
  tc_mey = tcm.py();
  tc_phi = tcm.phi();
  tc_set = tcm.sumEt();

  if (debug_) {
    cout << "  =========PFMET  " << rec_met  << ", " << rec_phi  << endl;
    cout << "  =========trueMET " << true_met << ", " << true_phi << endl;
    cout << "  =========CaloMET " << calo_met << ", " << calo_phi << endl;
    cout << "  =========TCMET " << tc_met << ", " << tc_phi << endl;
  }			
}

void PFMETBenchmark::calculateQuantities( const reco::PFMETCollection& pfMets, 
					  const reco::GenParticleCollection& genParticleList, 
					  const reco::CaloMETCollection& caloMets,
					  const reco::METCollection& tcMets,
					  const std::vector<reco::CaloJet>& caloJets,
					  const std::vector<reco::CaloJet>& corr_caloJets) 
{
  const reco::PFMET&    pfm = pfMets[0];
  const reco::CaloMET&  cm  = caloMets[0];
  const reco::MET&  tcm  = tcMets[0];

  double trueMEY  = 0.0;
  double trueMEX  = 0.0;;
  true_set  = 0.0;;

  //  for( genParticle = genParticleList.begin(); genParticle != genParticleList.end(); genParticle++ )
  for( unsigned i = 0; i < genParticleList.size(); i++ ) {
    if( genParticleList[i].status() == 1 && fabs(genParticleList[i].eta()) < 5.0 ) { 
      if( std::abs(genParticleList[i].pdgId()) == 12 ||
	  std::abs(genParticleList[i].pdgId()) == 14 ||
	  std::abs(genParticleList[i].pdgId()) == 16 || 
	  std::abs(genParticleList[i].pdgId()) < 7   || 
	  std::abs(genParticleList[i].pdgId()) == 21 ) {
	trueMEX += genParticleList[i].px();
	trueMEY += genParticleList[i].py();
      } else {
	true_set += genParticleList[i].pt();
      }
    }
  }
  true_mex = -trueMEX;
  true_mey = -trueMEY;
  true_met = sqrt( trueMEX*trueMEX + trueMEY*trueMEY );
  true_phi = atan2(trueMEY,trueMEX);
  rec_met  = pfm.pt();
  rec_mex  = pfm.px();
  rec_mex  = pfm.py();
  rec_phi  = pfm.phi();
  rec_set  = pfm.sumEt();

  // propagation of the JEC to the caloMET:
  double caloJetCorPX = 0.0;
  double caloJetCorPY = 0.0;

  for(unsigned int caloJetc=0;caloJetc<caloJets.size();++caloJetc)
  {
    //std::cout << "caloJets[" << caloJetc << "].pt() = " << caloJets[caloJetc].pt() << std::endl;
    //std::cout << "caloJets[" << caloJetc << "].phi() = " << caloJets[caloJetc].phi() << std::endl;
    //std::cout << "caloJets[" << caloJetc << "].eta() = " << caloJets[caloJetc].eta() << std::endl;
    //}
    for(unsigned int corr_caloJetc=0;corr_caloJetc<corr_caloJets.size();++corr_caloJetc)
    {
      //std::cout << "corr_caloJets[" << corr_caloJetc << "].pt() = " << corr_caloJets[corr_caloJetc].pt() << std::endl;
      //std::cout << "corr_caloJets[" << corr_caloJetc << "].phi() = " << corr_caloJets[corr_caloJetc].phi() << std::endl;
      //std::cout << "corr_caloJets[" << corr_caloJetc << "].eta() = " << corr_caloJets[corr_caloJetc].eta() << std::endl;
      //}
      Float_t DeltaPhi = corr_caloJets[corr_caloJetc].phi() - caloJets[caloJetc].phi();
      Float_t DeltaEta = corr_caloJets[corr_caloJetc].eta() - caloJets[caloJetc].eta();
      Float_t DeltaR2 = DeltaPhi*DeltaPhi + DeltaEta*DeltaEta;
      if( DeltaR2 < 0.0001 && caloJets[caloJetc].pt() > 20.0 ) 
      {
	caloJetCorPX += (corr_caloJets[corr_caloJetc].px() - caloJets[caloJetc].px());
	caloJetCorPY += (corr_caloJets[corr_caloJetc].py() - caloJets[caloJetc].py());
      }
    }
  }
  double corr_calomet=sqrt((cm.px()-caloJetCorPX)*(cm.px()-caloJetCorPX)+(cm.py()-caloJetCorPY)*(cm.py()-caloJetCorPY));
  calo_met = corr_calomet;
  calo_mex = cm.px()-caloJetCorPX;
  calo_mey = cm.py()-caloJetCorPY;
  calo_phi = atan2((cm.py()-caloJetCorPY),(cm.px()-caloJetCorPX));
  //calo_met = cm.pt();
  //calo_mex = cm.px();
  //calo_mey = cm.py();
  //calo_phi = cm.phi();

  calo_set = cm.sumEt();
  tc_met = tcm.pt();
  tc_mex = tcm.px();
  tc_mey = tcm.py();
  tc_phi = tcm.phi();
  tc_set = tcm.sumEt();

  if (debug_) {
    cout << "  =========PFMET  " << rec_met  << ", " << rec_phi  << endl;
    cout << "  =========trueMET " << true_met << ", " << true_phi << endl;
    cout << "  =========CaloMET " << calo_met << ", " << calo_phi << endl;
    cout << "  =========TCMET " << tc_met << ", " << tc_phi << endl;
  }			
}

//double fitf(double *x, double *par)
//{
//  double fitval = sqrt( par[0]*par[0] + 
//			par[1]*par[1]*(x[0]-par[3]) + 
//			par[2]*par[2]*(x[0]-par[3])*(x[0]-par[3]) );
//  return fitval;
//}

void PFMETBenchmark::analyse() 
{
  //Define fit functions and histograms
  //TF1* func1 = new TF1("fit1", fitf, 0, 40, 4);
  //TF1* func2 = new TF1("fit2", fitf, 0, 40, 4);
  //TF1* func3 = new TF1("fit3", fitf, 0, 40, 4);
  //TF1* func4 = new TF1("fit4", fitf, 0, 40, 4);

  //fit gaussian to Delta MET corresponding to different slices in MET, store fit values (mean,sigma) in histos
  ////FitSlicesInY(hSETvsDeltaMET, hmeanPF, hrmsPF, false, 1); //set option flag for RMS or gaussian
  ////FitSlicesInY(hSETvsDeltaMET, hmeanPF, hsigmaPF, true, 1); //set option flag for RMS or gaussian
  ////FitSlicesInY(hCaloSETvsDeltaCaloMET, hmeanCalo, hrmsCalo, false, 2); 
  ////FitSlicesInY(hCaloSETvsDeltaCaloMET, hmeanCalo, hsigmaCalo, true, 2); 

  ////SETAXES(meanPF,    "SET", "Mean(MEX)");
  ////SETAXES(meanCalo,  "SET", "Mean(MEX)");
  ////SETAXES(sigmaPF,   "SET", "#sigma(MEX)");
  ////SETAXES(sigmaCalo, "SET", "#sigma(MEX)");
  ////SETAXES(rmsPF,     "SET", "RMS(MEX)");
  ////SETAXES(rmsCalo,   "SET", "RMS(MEX)");

  // Make the MET resolution versus SET plot
  /*
  TCanvas* canvas_MetResVsRecoSet = new TCanvas("MetResVsRecoSet", "MET Sigma vs Reco SET", 500,500);
  hsigmaPF->SetStats(0); 
  func1->SetLineColor(1); 
  func1->SetParNames("Noise", "Stochastic", "Constant", "Offset");
  func1->SetParameters(10.0, 0.8, 0.1, 100.0);
  hsigmaPF->Fit("fit1", "", "", 100.0, 900.0);
  func2->SetLineColor(2); 
  func2->SetParNames("Noise", "Stochastic", "Constant", "Offset");
  func2->SetParameters(10.0, 0.8, 0.1, 100.0);
  hsigmaCalo->Fit("fit2", "", "", 100.0, 900.0);
  func3->SetLineColor(4); 
  func3->SetParNames("Noise", "Stochastic", "Constant", "Offset");
  func3->SetParameters(10.0, 0.8, 0.1, 100.0);
  hrmsPF->Fit("fit3", "", "", 100.0, 900.0);
  func4->SetLineColor(6); 
  func4->SetParNames("Noise", "Stochastic", "Constant", "Offset");
  func4->SetParameters(10.0, 0.8, 0.1, 100.0);
  hrmsCalo->Fit("fit4", "", "", 100.0, 900.0);
  (hsigmaPF->GetYaxis())->SetRangeUser( 0.0, 50.0);
  hsigmaPF->SetLineWidth(2); 
  hsigmaPF->SetLineColor(1); 
  hsigmaPF->Draw();
  hsigmaCalo->SetLineWidth(2);
  hsigmaCalo->SetLineColor(2);
  hsigmaCalo->Draw("SAME");
  hrmsPF->SetLineWidth(2);
  hrmsPF->SetLineColor(4);
  hrmsPF->Draw("SAME");  
  hrmsCalo->SetLineWidth(2);
  hrmsCalo->SetLineColor(6);
  hrmsCalo->Draw("SAME");
  */

  // Make the SET response versus SET plot
  /*
  TCanvas* canvas_SetRespVsTrueSet = new TCanvas("SetRespVsTrueSet", "SET Response vs True SET", 500,500);
  profileSETvsSETresp->SetStats(0); 
  profileSETvsSETresp->SetStats(0); 
  (profileSETvsSETresp->GetYaxis())->SetRangeUser(-1.0, 1.0);
  profileSETvsSETresp->SetLineWidth(2); 
  profileSETvsSETresp->SetLineColor(4); 
  profileSETvsSETresp->Draw();
  profileCaloSETvsCaloSETresp->SetLineWidth(2); 
  profileCaloSETvsCaloSETresp->SetLineColor(2); 
  profileCaloSETvsCaloSETresp->Draw("SAME");
  */

  // Make the MET response versus MET plot
  /*
  TCanvas* canvas_MetRespVsTrueMet = new TCanvas("MetRespVsTrueMet", "MET Response vs True MET", 500,500);
  profileMETvsMETresp->SetStats(0); 
  profileMETvsMETresp->SetStats(0); 
  (profileMETvsMETresp->GetYaxis())->SetRangeUser(-1.0, 1.0);
  profileMETvsMETresp->SetLineWidth(2); 
  profileMETvsMETresp->SetLineColor(4); 
  profileMETvsMETresp->Draw();
  profileCaloMETvsCaloMETresp->SetLineWidth(2); 
  profileCaloMETvsCaloMETresp->SetLineColor(2); 
  profileCaloMETvsCaloMETresp->Draw("SAME");
  */

  //print the resulting plots to file
  /*
  canvas_MetResVsRecoSet->Print("MetResVsRecoSet.ps");
  canvas_SetRespVsTrueSet->Print("SetRespVsTrueSet.ps");
  canvas_MetRespVsTrueMet->Print("MetRespVsTrueMet.ps");  
  */
}

//void PFMETBenchmark::FitSlicesInY(TH2F* h, TH1F* mean, TH1F* sigma, bool doGausFit, int type )
//{
//  TAxis *fXaxis = h->GetXaxis();
//  TAxis *fYaxis = h->GetYaxis();
//  Int_t nbins  = fXaxis->GetNbins();
//
//  //std::cout << "nbins = " << nbins << std::endl;
//
//  Int_t binmin = 1;
//  Int_t binmax = nbins;
//  TString option = "QNR";
//  TString opt = option;
//  opt.ToLower();
//  Float_t ngroup = 1;
//  ngroup = 1;
//
//  //std::cout << "fYaxis->GetXmin() = " << fYaxis->GetXmin() << std::endl;
//  //std::cout << "fYaxis->GetXmax() = " << fYaxis->GetXmax() << std::endl;
//
//  //default is to fit with a gaussian
//  TF1 *f1 = 0;
//  if (f1 == 0) 
//    {
//      //f1 = (TF1*)gROOT->GetFunction("gaus");
//      if (f1 == 0) f1 = new TF1("gaus","gaus", fYaxis->GetXmin(), fYaxis->GetXmax());
//      else         f1->SetRange( fYaxis->GetXmin(), fYaxis->GetXmax());
//    }
//  Int_t npar = f1->GetNpar();
//
//  //std::cout << "npar = " << npar << std::endl;
//
//  if (npar <= 0) return;
//
//  Double_t *parsave = new Double_t[npar];
//  f1->GetParameters(parsave);
//
////  //Create one histogram for each function parameter
//  Int_t ipar;
//  TH1F **hlist = new TH1F*[npar];
//  char *name   = new char[2000];
//  char *title  = new char[2000];
//  const TArrayD *bins = fXaxis->GetXbins();
//  for( ipar=0; ipar < npar; ipar++ ) 
//    {
//      if( ipar == 1 ) 
//	if( type == 1 )   sprintf(name,"meanPF");
//	else              sprintf(name,"meanCalo");
//      else
//	if( doGausFit ) 
//	  if( type == 1 ) sprintf(name,"sigmaPF");
//	  else            sprintf(name,"sigmaCalo");
//	else 
//	  if( type == 1 ) sprintf(name,"rmsPF");
//	  else            sprintf(name,"rmsCalo");
//      if( type == 1 )     sprintf(title,"Particle Flow");
//      else                sprintf(title,"Calorimeter");
//      delete gDirectory->FindObject(name);
//      if (bins->fN == 0) 
//	hlist[ipar] = new TH1F(name,title, nbins, fXaxis->GetXmin(), fXaxis->GetXmax());
//      else
//	hlist[ipar] = new TH1F(name,title, nbins,bins->fArray);
//      hlist[ipar]->GetXaxis()->SetTitle(fXaxis->GetTitle());
//    }
//  sprintf(name,"test_chi2");
//  delete gDirectory->FindObject(name);
//  TH1F *hchi2 = new TH1F(name,"chisquare", nbins, fXaxis->GetXmin(), fXaxis->GetXmax());
//  hchi2->GetXaxis()->SetTitle(fXaxis->GetTitle());
//
//  //Loop on all bins in X, generate a projection along Y
//  Int_t bin;
//  Int_t nentries;
//  for( bin = (Int_t) binmin; bin <= (Int_t) binmax; bin += (Int_t)ngroup ) 
//    {
//      TH1F *hpy = (TH1F*) h->ProjectionY("_temp", (Int_t) bin, (Int_t) (bin + ngroup - 1), "e");
//      if(hpy == 0) continue;
//      nentries = Int_t( hpy->GetEntries() );
//      if(nentries == 0 ) {delete hpy; continue;}
//      f1->SetParameters(parsave);
//      hpy->Fit( f1, opt.Data() );
//      Int_t npfits = f1->GetNumberFitPoints(); 
//      //cout << "bin = " << bin << "; Npfits = " << npfits << "; npar = " << npar << endl;
//      if( npfits > npar ) 
//	{
//	  Int_t biny = bin + (Int_t)ngroup/2;
//	  for( ipar=0; ipar < npar; ipar++ ) 
//	    {
//	      if( doGausFit ) hlist[ipar]->Fill( fXaxis->GetBinCenter(biny), f1->GetParameter(ipar) );
//	      else            hlist[ipar]->Fill( fXaxis->GetBinCenter(biny), hpy->GetRMS() );
//	      //cout << "bin[" << bin << "]: RMS = " << hpy->GetRMS() << "; sigma = " << f1->GetParameter(ipar) << endl;
//	      hlist[ipar]->SetBinError( biny, f1->GetParError(ipar) );
//	    }
//	  hchi2->Fill( fXaxis->GetBinCenter(biny), f1->GetChisquare()/(npfits-npar) );
//	}
//      delete hpy;
//      ngroup += ngroup*0.2;//0.1  //used for non-uniform binning
//    }
//  *mean = *hlist[1];
//  *sigma = *hlist[2];
//  cout << "Entries = " << hlist[0]->GetEntries() << endl;
//}

double   
PFMETBenchmark::mpi_pi(double angle) {

  const double pi = 3.14159265358979323;
  const double pi2 = pi*2.;
  while(angle>pi) angle -= pi2;
  while(angle<-pi) angle += pi2;
  return angle;

}
