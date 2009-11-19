#include "../interface/KinematicsChecker.h"

KinematicsChecker::KinematicsChecker(const edm::ParameterSet& iConfig, std::string relativePath, std::string label)
{
  //now do what ever initialization is needed
  dqmStore_ = edm::Service<DQMStore>().operator->();
  NbOfEvents        = 0;
  relativePath_     = relativePath;
  label_            = label;
}

KinematicsChecker::~KinematicsChecker()
{
  delete dqmStore_;
}

void
KinematicsChecker::analyze(const std::vector<reco::CaloJet>& jets, const std::vector<reco::CaloMET>& mets, const std::vector<reco::Muon>& muons, const std::vector<reco::GsfElectron>& electrons)
{
  using namespace edm;
  NbOfEvents++;

  //Check if branches are available
  /*
  if (!jets.isValid()){
    edm::LogWarning  ("LinkBroken_noJetsFound") << "My warning message - No Jets Found"; 
    return; //throw cms::Exception("ProductNotFound") <<"jet collection not found"<<std::endl;
  }
  if (!muons.isValid()){
    edm::LogWarning  ("LinkBroken_NoMuonsFound") << "My warning message - No Muons Found"; 
    return; //throw cms::Exception("ProductNotFound") <<"muon collection not found"<<std::endl;
  }
  if (!mets.isValid()){
    edm::LogWarning  ("LinkBroken_NoMetsFound") << "My warning message - No Mets Found";
    return;// throw cms::Exception("ProductNotFound") <<"MET collection not found"<<std::endl;
  }
  */

  //Create TLorentzVector objets from reco objects
  std::vector< reco::Particle::LorentzVector > ObjectP4s[4];
  for( unsigned int i=0;i<jets.size();i++)      {ObjectP4s[0].push_back((jets)[i].p4());}
  for( unsigned int i=0;i<mets.size();i++)      {ObjectP4s[1].push_back((mets)[i].p4());}
  for( unsigned int i=0;i<muons.size();i++)     {ObjectP4s[2].push_back((muons)[i].p4());}
  for( unsigned int i=0;i<electrons.size();i++) {ObjectP4s[3].push_back((electrons)[i].p4());}

  //Fill Histograms
  for(unsigned int i=0;i<4;i++){
    hists_[i]["number"]->Fill(ObjectP4s[i].size()); 
    for(unsigned int j=0;j<ObjectP4s[i].size();j++) hists_[i]["pt"]    ->Fill(ObjectP4s[i][j].Pt());  
    for(unsigned int j=0;j<ObjectP4s[i].size();j++) hists_[i]["eta"]   ->Fill(ObjectP4s[i][j].Eta()); 
    for(unsigned int j=0;j<ObjectP4s[i].size();j++) hists_[i]["et"]    ->Fill(ObjectP4s[i][j].Et());
    for(unsigned int j=0;j<ObjectP4s[i].size();j++) hists_[i]["energy"]->Fill(ObjectP4s[i][j].E());
    for(unsigned int j=0;j<ObjectP4s[i].size();j++) hists_[i]["theta"] ->Fill(ObjectP4s[i][j].Theta());
    for(unsigned int j=0;j<ObjectP4s[i].size();j++) hists_[i]["phi"]   ->Fill(ObjectP4s[i][j].Phi());
  }
   
   
  for(unsigned int i=0;i<4;i++){
    ObjectP4s[i].clear();
  }
   

}

void 
KinematicsChecker::begin(const edm::EventSetup&)
{
  std::string ObjectNames[4] = {"CaloJets","CaloMETs","Muons","Electrons"};

  for(unsigned int i=0;i<4;i++){  
    dqmStore_->setCurrentFolder(relativePath_+"/"+ObjectNames[i]+"_"+label_);
    
    hists_[i]["number"] = dqmStore_->book1D("number" ,"number of objects",31,-0.5,30.5);
    hists_[i]["number"] ->setAxisTitle("nof "+ObjectNames[i],1);
    hists_[i]["pt"]     = dqmStore_->book1D("pt" ,"pt",50,0,200);
    hists_[i]["pt"]     ->setAxisTitle("Pt of "+ObjectNames[i],1);
    hists_[i]["et"]     = dqmStore_->book1D("et" ,"et",50,0,200);
    hists_[i]["et"]     ->setAxisTitle("Et of "+ObjectNames[i],1);
    hists_[i]["eta"]    = dqmStore_->book1D("eta" ,"eta",50,-6,6);
    hists_[i]["eta"]    ->setAxisTitle("Eta of "+ObjectNames[i],1);
    hists_[i]["energy"] = dqmStore_->book1D("energy" ,"energy",50,0,400);
    hists_[i]["energy"] ->setAxisTitle("Energy of "+ObjectNames[i],1);
    hists_[i]["theta"]  = dqmStore_->book1D("theta" ,"theta",50,0,3.2);
    hists_[i]["theta"]  ->setAxisTitle("Theta of "+ObjectNames[i],1);
    hists_[i]["phi"]    = dqmStore_->book1D("phi" ,"phi",50,-3.2,3.2);
    hists_[i]["phi"]    ->setAxisTitle("Phi of "+ObjectNames[i],1);
  }
}

void 
KinematicsChecker::end() 
{
}

