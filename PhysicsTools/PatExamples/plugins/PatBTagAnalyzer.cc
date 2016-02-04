#include <map>
#include <string>

#include "TH1D.h"
#include "TH2D.h"
#include "TGraphErrors.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "PhysicsTools/PatUtils/interface/bJetSelector.h"
#include "PhysicsTools/PatExamples/interface/BTagPerformance.h"
#include "PhysicsTools/PatExamples/interface/PatBTagCommonHistos.h"


class PatBTagAnalyzer : public edm::EDAnalyzer {

public:

  explicit PatBTagAnalyzer(const edm::ParameterSet&);
  ~PatBTagAnalyzer();
    
private:

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  
  edm::InputTag jetLabel_;
  edm::ParameterSet PatBjet_;

  std::string  BTagpurity_;
  std::string  BTagmethod_;
  std::string  BTagdiscriminator_;

  bool    BTagverbose;
  double  BTagdisccut_;
  double  BTagdiscmax_;


  std::string  discname[10];   
  std::string  bname   [10];   
  std::string  cname   [10];   
  BTagPerformance BTagPerf[10];
  std::map<int,std::string> udsgname;   

  /// simple map to contain all histograms; histograms are booked in 
  /// beginJob()
  std::map<std::string,TH1D*> histocontainer_;   
  /// simple map to contain 2D  histograms; histograms are booked in 
  /// beginJob()
  std::map<std::string,TH2D*> h2_; 
  /// simple map to contain all graphs; graphs are booked in 
  /// beginJob()
  std::map<std::string,TGraph*> graphcontainer_; 
  /// simple map to contain all graphs; graphs are booked in 
  /// beginJob()
  std::map<std::string,TGraphErrors*> grapherrorscontainer_; 

  bJetSelector BTagger;
  PatBTagCommonHistos BTagHistograms;
};

PatBTagAnalyzer::PatBTagAnalyzer(const edm::ParameterSet& iConfig):
  jetLabel_(iConfig.getUntrackedParameter<edm::InputTag>("jetTag")),
  PatBjet_(iConfig.getParameter< edm::ParameterSet >("BjetTag")),
  BTagpurity_(PatBjet_.getParameter<std::string>("purity")),
  BTagmethod_(PatBjet_.getUntrackedParameter<std::string>("tagger","TC2")),
  BTagdiscriminator_(PatBjet_.getParameter<std::string>("discriminator")),
  BTagverbose(PatBjet_.getUntrackedParameter<bool>("verbose",false)),
  BTagdisccut_(PatBjet_.getUntrackedParameter<double>("mindiscriminatorcut",5.0)),
  BTagdiscmax_(PatBjet_.getUntrackedParameter<double>("maxdiscriminatorcut",15.0)),
  BTagger(iConfig.getParameter< edm::ParameterSet >("BJetOperatingPoints")),
  BTagHistograms(iConfig)
{
   //now do what ever initialization is needed
}


PatBTagAnalyzer::~PatBTagAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

// ------------ method called to for each event  ------------
void
PatBTagAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // first: get all objects from the event.

   edm::Handle<edm::View<pat::Jet> > jetHandle;
   iEvent.getByLabel(jetLabel_,jetHandle);
   edm::View<pat::Jet> jets = *jetHandle; // get JETS

   // LOOP over all jets
   
   for(edm::View<pat::Jet>::const_iterator jet_iter = jets.begin(); jet_iter!=jets.end(); ++jet_iter){

       float bdiscriminator = jet_iter->bDiscriminator(BTagdiscriminator_);
       int flavor           = jet_iter->partonFlavour();
       //
       // Fill in for performance standard pt(uncorrected) >10 and abs(eta)<2.4 
       if( jet_iter->correctedJet("raw").pt() > 10  &&
		   fabs(jet_iter->eta()) < 2.4 ) {
		   
		   BTagPerf[0].Add(bdiscriminator,abs(flavor));

       }
       
	   //Fill histograms
	   BTagHistograms.Fill(jet_iter,"all");
	   if (flavor ==  0  ) BTagHistograms.Fill(jet_iter,"no_flavor");
	   if (flavor ==  5 || flavor ==  -5 ) BTagHistograms.Fill(jet_iter,"b");
	   if (flavor ==  4 || flavor ==  -4 ) BTagHistograms.Fill(jet_iter,"c");
	   if ((-4 < flavor && flavor < 4 && flavor != 0 )||(flavor == 21 || flavor == -21 ))
		   BTagHistograms.Fill(jet_iter,"udsg");
	   

   }//end loop over jets 

}
// ------------ method called once each job just before starting event loop  ------------
void 
PatBTagAnalyzer::beginJob()
{
  //
  // define some histograms using the framework tfileservice. Define the output file name in your .cfg.
  //
  edm::Service<TFileService> fs;
  
  TString suffix1="_test";

  //set performance variables collector
  for (int i=0; i < 10; i++){
    BTagPerf[i].Set(BTagmethod_);
    BTagPerf[i].SetMinDiscriminator(BTagdisccut_);
    BTagPerf[i].SetMaxDiscriminator(BTagdiscmax_);
  }

  histocontainer_["njets"]=fs->make<TH1D>("njets","jet multiplicity for jets with p_{T} > 50 GeV/c",10,0,10);
// Std. 30 pt uncorr cut for performance
  discname[0]="disc"+BTagmethod_+"_udsg";   
  bname[0]   ="g"+BTagmethod_+"_b";   
  cname[0]   ="g"+BTagmethod_+"_c";   
  udsgname[0]="g"+BTagmethod_+"_udsg";   

// 10 pt uncorr for performance + all,>0,>1,>2 tracks
  discname[1]="Uncor10_disc"+BTagmethod_+"_udsg";
  bname[1]   ="Uncor10_g"+BTagmethod_+"_b";   
  cname[1]   ="Uncor10_g"+BTagmethod_+"_c";   
  udsgname[1]="Uncor10_g"+BTagmethod_+"_udsg";
  discname[2]="Uncor10t0_disc"+BTagmethod_+"_udsg";
  bname[2]   ="Uncor10t0_g"+BTagmethod_+"_b";   
  cname[2]   ="Uncor10t0_g"+BTagmethod_+"_c";   
  udsgname[2]="Uncor10t0_g"+BTagmethod_+"_udsg";
  discname[3]="Uncor10t1_disc"+BTagmethod_+"_udsg";
  bname[3]   ="Uncor10t1_g"+BTagmethod_+"_b";   
  cname[3]   ="Uncor10t1_g"+BTagmethod_+"_c";   
  udsgname[3]="Uncor10t1_g"+BTagmethod_+"_udsg";
  discname[4]="Uncor10t2_disc"+BTagmethod_+"_udsg";
  bname[4]   ="Uncor10t2_g"+BTagmethod_+"_b";   
  cname[4]   ="Uncor10t2_g"+BTagmethod_+"_c";   
  udsgname[4]="Uncor10t2_g"+BTagmethod_+"_udsg";

// 30 pt corr for performance + all,>0,>1,>2 tracks   
  discname[5]="Corr30_disc"+BTagmethod_+"_udsg";
  bname[5]   ="Corr30_g"+BTagmethod_+"_b";   
  cname[5]   ="Corr30_g"+BTagmethod_+"_c";   
  udsgname[5]="Corr30_g"+BTagmethod_+"_udsg";
  discname[6]="Corr30t0_disc"+BTagmethod_+"_udsg";
  bname[6]   ="Corr30t0_g"+BTagmethod_+"_b";   
  cname[6]   ="Corr30t0_g"+BTagmethod_+"_c";   
  udsgname[6]="Corr30t0_g"+BTagmethod_+"_udsg";
  discname[7]="Corr30t1_disc"+BTagmethod_+"_udsg";
  bname[7]   ="Corr30t1_g"+BTagmethod_+"_b";   
  cname[7]   ="Corr30t1_g"+BTagmethod_+"_c";   
  udsgname[7]="Corr30t1_g"+BTagmethod_+"_udsg";
  discname[8]="Corr30t2_disc"+BTagmethod_+"_udsg";
  bname[8]   ="Corr30t2_g"+BTagmethod_+"_b";   
  cname[8]   ="Corr30t2_g"+BTagmethod_+"_c";   
  udsgname[8]="Corr30t2_g"+BTagmethod_+"_udsg";

// check filter
  discname[9]="check_disc"+BTagmethod_+"_udsg";   
  bname[9]   ="check_g"+BTagmethod_+"_b";   
  cname[9]   ="check_g"+BTagmethod_+"_c";   
  udsgname[9]="check_g"+BTagmethod_+"_udsg";   

  for(int i=1; i<10;i++){
    graphcontainer_[discname[i]]      =fs->make<TGraph>(BTagPerf[i].GetN());       graphcontainer_[discname[i]]->SetName(discname[i].c_str());
    grapherrorscontainer_[bname[i]]   =fs->make<TGraphErrors>(BTagPerf[i].GetN()); grapherrorscontainer_[bname[i]]   ->SetName(bname[i].c_str());
    grapherrorscontainer_[cname[i]]   =fs->make<TGraphErrors>(BTagPerf[i].GetN()); grapherrorscontainer_[cname[i]]   ->SetName(cname[i].c_str());
    grapherrorscontainer_[udsgname[i]]=fs->make<TGraphErrors>(BTagPerf[i].GetN()); grapherrorscontainer_[udsgname[i]]->SetName(udsgname[i].c_str());   
  }
  //Define histograms
  BTagHistograms.Set("all");  
  BTagHistograms.Set("no_flavor");  
  BTagHistograms.Set("b");  
  BTagHistograms.Set("c");  
  BTagHistograms.Set("udsg");  

  // Set to save histogram errors
  BTagHistograms.Sumw2();  

}

// ------------ method called once each job just after ending the event loop  ------------
void 
PatBTagAnalyzer::endJob() {
//ed++
   edm::Service<TFileService> fs;

// Save performance plots as Tgraphs


   for (int i=1;i<10;i++){
      BTagPerf[i].Eval();
      for (int n=0; n<BTagPerf[i].GetN();  n++ ){
         graphcontainer_[discname[i]]       ->SetPoint(n,BTagPerf[i].GetArray("udsg")[n],BTagPerf[i].GetArray("discriminator")[n]);
         grapherrorscontainer_[bname[i]]    ->SetPoint(n,BTagPerf[i].GetArray("b")[n],BTagPerf[i].GetArray("b")[n]);
         grapherrorscontainer_[cname[i]]    ->SetPoint(n,BTagPerf[i].GetArray("b")[n],BTagPerf[i].GetArray("c")[n]);
         grapherrorscontainer_[udsgname[i]] ->SetPoint(n,BTagPerf[i].GetArray("b")[n],BTagPerf[i].GetArray("udsg")[n]);
         grapherrorscontainer_[bname[i]]    ->SetPointError(n,BTagPerf[i].GetArray("bErr")[n],BTagPerf[i].GetArray("bErr")[n]);
         grapherrorscontainer_[cname[i]]    ->SetPointError(n,BTagPerf[i].GetArray("bErr")[n],BTagPerf[i].GetArray("cErr")[n]);
         grapherrorscontainer_[udsgname[i]] ->SetPointError(n,BTagPerf[i].GetArray("bErr")[n],BTagPerf[i].GetArray("udsgErr")[n]);
      }//end for over BTagPerf[i] elements
      graphcontainer_[discname[i]]     ->SetTitle("Jet udsg-mistagging");
      grapherrorscontainer_[bname[i]]   ->SetTitle("Jet b-efficiency");
      grapherrorscontainer_[cname[i]]   ->SetTitle("Jet c-mistagging");
      grapherrorscontainer_[udsgname[i]]->SetTitle("discriminator vs udsg-mistagging");
   }//end for over [i]


// Save default cut performance plot
   BTagPerf[0].Eval();

//  TFileDirectory TaggerDir = fs->mkdir(BTagmethod_);
//  TGraphErrors *BTagger_b    = new TGraphErrors(BTagTool.GetN(),
  TGraphErrors *BTagger_b    = fs->mkdir(BTagmethod_).make<TGraphErrors>(BTagPerf[0].GetN(),
                                            BTagPerf[0].GetArray("b").GetArray(),BTagPerf[0].GetArray("b").GetArray(),
                                            BTagPerf[0].GetArray("bErr").GetArray(),BTagPerf[0].GetArray("bErr").GetArray());
        
  TGraphErrors *BTagger_c    = new TGraphErrors(BTagPerf[0].GetN(),
                                            BTagPerf[0].GetArray("b").GetArray(),BTagPerf[0].GetArray("c").GetArray(),
                                            BTagPerf[0].GetArray("bErr").GetArray(),BTagPerf[0].GetArray("cErr").GetArray());
                
  TGraphErrors *BTagger_udsg = new TGraphErrors(BTagPerf[0].GetN(),
                                               BTagPerf[0].GetArray("b").GetArray(),BTagPerf[0].GetArray("udsg").GetArray(),
                                               BTagPerf[0].GetArray("bErr").GetArray(),BTagPerf[0].GetArray("udsgErr").GetArray());
  TGraph *discBTagger_udsg   = new TGraph(BTagPerf[0].GetN(),
                                   BTagPerf[0].GetArray("udsg").GetArray(),
                                   BTagPerf[0].GetArray("discriminator").GetArray());
 
  BTagger_b->SetName(bname[0].c_str());
  BTagger_c->SetName(cname[0].c_str());
  BTagger_udsg->SetName(udsgname[0].c_str());
  discBTagger_udsg->SetName(discname[0].c_str());

  BTagger_b->SetTitle("Jet b-efficiency");
  BTagger_c->SetTitle("Jet c-mistagging");
  BTagger_udsg->SetTitle("Jet udsg-mistagging");
  discBTagger_udsg->SetTitle("discriminator vs udsg-mistagging");


  for (int i=1;i<10;i++){
   graphcontainer_[discname[i]]      ->Write();
   grapherrorscontainer_[bname[i]]   ->Write();
   grapherrorscontainer_[cname[i]]   ->Write();
   grapherrorscontainer_[udsgname[i]]->Write();
  }
  
  BTagger_b->Write();
  BTagger_c->Write();
  BTagger_udsg->Write();
  discBTagger_udsg->Write();
    

}

//define this as a plug-in
DEFINE_FWK_MODULE(PatBTagAnalyzer);
