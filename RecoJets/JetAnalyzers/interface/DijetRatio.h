// -*- C++ -*-
//
// Package:    DijetRatio
// Class:      DijetRatio
// 
/**\class DijetRatio DijetRatio.cc RecoJets/DijetRatio/src/DijetRatio.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Manoj Jha
//         Created:  Thu Apr 12 15:04:37 CDT 2007
// Kalanand Mishra (November 22, 2009): 
//        Modified and cleaned up to work in 3.3.X
//
//

// system include files
#ifndef DIJETRATIO_HH
#define DIJETRATIO_HH

#include <memory>
#include <string>
#include <iostream>
#include <map>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "CLHEP/Vector/LorentzVector.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

const int histoSize = 5;

//Histo Initializations
inline void hInit(TH1F* hJet[], const char* name){
  int const binSize = 35;
  float massBin[binSize+1] = { 100, 113, 132, 153, 176, 201, 
			       229, 259, 292, 327, 366, 400, 
			       453, 501, 553, 609, 669, 733, 
			       802, 875, 954, 1038, 1127, 1222, 
			       1323, 1431, 1546, 1667, 1796, 1934,
			       2079, 2233, 2396, 2569, 2752,3000};


  // (jetEta1 > 0 && jetEta1 < 0.7),  (jetEta2 > 0 && jetEta2 < 0.7 )
  std::string tit = std::string(name) + "_Eta_innerEtaCut_outerEtaCut";
  hJet[0] =  new TH1F(tit.c_str(), "DiJet Mass", binSize, massBin); 	


  // (jetEta1 > 0.7 && jetEta1 < 1.3),  (jetEta2 > 0.7 && jetEta2 < 1.3 )
  tit = std::string(name) + "_Eta_0_innerEtaCut";
  hJet[1] =  new TH1F(tit.c_str(), "DiJet Mass", binSize, massBin);		

  tit = std::string(name) + "_LeadJetEta";
  hJet[2] =  new TH1F(tit.c_str(), "1^{st} Leading Jet #eta", 120, -6., 6.);
  tit = std::string(name) + "_SecondJetEta";
  hJet[3] =  new TH1F(tit.c_str(), "2^{nd} Leading Jet #eta", 120, -6., 6.);
  tit = std::string(name) + "_numEvents";
  hJet[4] =  new TH1F(tit.c_str(), "No. of events", 10, 0.,10.);
   
  return ;
}



template <class R>
void histoFill(TH1F* jetHisto[], edm::Handle<R> jetsRec, double eta1, double eta2)
{
  //For no. of events
  jetHisto[4]->Fill(1.);

  if ((*jetsRec).size() >=2){
    double px1 = (*jetsRec)[0].px(); 
    double py1 = (*jetsRec)[0].py(); 
    double pz1 = (*jetsRec)[0].pz(); 
    double e1 = (*jetsRec)[0].energy(); 
    double jetEta1 = (*jetsRec)[0].eta(); 
    jetHisto[2]->Fill(jetEta1);
	   
    double px2 = (*jetsRec)[1].px(); 
    double py2 = (*jetsRec)[1].py(); 
    double pz2 = (*jetsRec)[1].pz(); 
    double e2 = (*jetsRec)[1].energy(); 
    double jetEta2 = (*jetsRec)[1].eta();
    jetHisto[3]->Fill(jetEta2);

    CLHEP::HepLorentzVector v1(px1,py1,pz1,e1);	   
    CLHEP::HepLorentzVector v2(px2,py2,pz2,e2);	   
    CLHEP::HepLorentzVector v(0.,0.,0.,0.);	   
    v = v1 + v2; 
    float mass = v.m();

    if ( fabs(jetEta1) > 0.0 &&  fabs(jetEta1) < eta1) 
      if ( fabs(jetEta2) > 0.0 &&  fabs(jetEta2) < eta1) 
	jetHisto[0]->Fill(mass);
	   
    if ( fabs(jetEta1) > eta1 && fabs(jetEta1) < eta2) 
      if ( fabs(jetEta2) > eta1 && fabs(jetEta2) < eta2)
	jetHisto[1]->Fill(mass);
	  
  }
}//histoFill


//
// class decleration
//
template<class Jet>
class DijetRatio : public edm::EDAnalyzer {
public:
  explicit DijetRatio(const edm::ParameterSet&);
  ~DijetRatio();


  typedef std::vector<Jet> JetCollection;
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;


  // ----------member data ---------------------------
  std::string fOutputFileName ;
     
  // names of modules, producing object collections
  std::string m_Mid5CorRecJetsSrc;
  std::string m_Mid5CaloJetsSrc;

  // eta limit 
  double  m_eta3;  // eta limit for numerator
  double  m_eta4;  // eta limit for denominator
  
  //Jet Kinematics for leading Jet
  static  const int hisotNumber = 10;
  
  TH1F*  hCalo[hisotNumber];
  TH1F*  hCor[hisotNumber];
     
  // Root file for saving histo
  TFile*      hOutputFile ;
};

#endif

