//
// Original Author:  Olga Kodolova, September 2007
//
// ZSP Jet Corrector
//
#include "JetMETCorrections/Algorithms/interface/ZSPJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleZSPJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL1OffsetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

using namespace std;


ZSPJetCorrector::ZSPJetCorrector (const edm::ParameterSet& fConfig) {

 iPU = fConfig.getParameter<int>("PU");
 fixedPU = fConfig.getParameter<int>("FixedPU"); 

 if( iPU >= 0 || fixedPU > 0 ) {
 theFilesL1Offset = fConfig.getParameter <vector<string> > ("tagNameOffset");
 for(vector<string>::iterator it=theFilesL1Offset.begin(); it != theFilesL1Offset.end(); it++) {
   std::string file="CondFormats/JetMETObjects/data/"+(*it)+".txt";
//   cout<<" File name "<<file<<endl;
   edm::FileInPath f2(file);
   mSimpleCorrectorOffset.push_back(new SimpleL1OffsetCorrector (f2.fullPath()));
 }
 }

 theFilesZSP = fConfig.getParameter <vector<string> > ("tagName");
 for(vector<string>::iterator it=theFilesZSP.begin(); it != theFilesZSP.end(); it++) {
   std::string file="CondFormats/JetMETObjects/data/"+(*it)+".txt";
   edm::FileInPath f1(file);
   mSimpleCorrector.push_back(new SimpleZSPJetCorrector (f1.fullPath()));
 }

//   cout<<" Size of correctors "<<mSimpleCorrector.size()<<" "<<mSimpleCorrectorOffset.size()<<endl;
}

ZSPJetCorrector::~ZSPJetCorrector () {
//  delete mSimpleCorrector;
//  delete mSimpleCorrectorOffset; 
} 

double ZSPJetCorrector::correction (const LorentzVector& fJet) const {
  double a = mSimpleCorrector[fixedPU]->correctionPtEtaPhiE (fJet.Pt(), fJet.Eta(), fJet.Phi(),fJet.E());
 // std::cout<<" Simple First correction "<<a<<std::endl;
  double b = a;
  if(iPU >= 0) {
  if(mSimpleCorrectorOffset.size()>0) {
  b = mSimpleCorrectorOffset[fixedPU]->correctionEnEta (a*fJet.E(), fJet.Eta());
  // std::cout<<"Simple  Second correction "<<b<<std::endl;
  } else {
 // std::cout<<" No offset files but iPU "<<iPU<<" check configuration JetMETCorrections/Configuration/python/ZSPOffsetJetCorrections219_cff.py "<<std::endl;
  }
  }
  return b;
}

double ZSPJetCorrector::correction (const reco::Jet& fJet) const {
  return correction (fJet.p4 ());
}
double ZSPJetCorrector::correction( const reco::Jet& fJet, const edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
   double b=1.;
   int nPU = 0;
//   cout<<" Event correction "<<iPU<<endl;
   if(iPU > 0) {
// Look for the Lumi section
//     LuminosityBlock lbID = iEvent.getLuminosityBlock();
//       cout<<" Not implemented yet "<<iEvent.run()<<endl;
       nPU=setPU();
   } else { if(iPU==0) nPU=setPU(); } 


  double a = mSimpleCorrector[nPU]->correctionPtEtaPhiE (fJet.p4().Pt(), fJet.p4().Eta(), fJet.p4().Phi(),fJet.p4().E());
//  std::cout<<" Lumi section First correction "<<a<<" "<<nPU<<std::endl;
  b = a;
  if(iPU >= 0) {
  if(mSimpleCorrectorOffset.size()>0) {
  b = mSimpleCorrectorOffset[nPU]->correctionEnEta (a*fJet.p4().E(), fJet.p4().Eta());
 // std::cout<<" Lumi section Second correction "<<b<<" "<<nPU<<std::endl;
  } else {
 // std::cout<<" No offset files but iPU "<<iPU<<" check configuration JetMETCorrections/Configuration/python/ZSPOffsetJetCorrections219_cff.py "<<std::endl;
  }
  }

   return b;
}

