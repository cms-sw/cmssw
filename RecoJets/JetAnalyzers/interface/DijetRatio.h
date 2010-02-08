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
// $Id$
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
#include "FWCore/Framework/interface/Selector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "CLHEP/Vector/LorentzVector.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace TMath;

//Histo Initializations
inline void hInit(TH1F* hJet[], char* name){
	int const binSize = 133;
	float massBin[binSize+1] = {0.,6.75021,26.7502,34.7215,43.9963,54.6599,66.8013,80.5137,95.8941,113.044,132.071,153.087,176.208,201.558,229.267,259.469,292.306,327.929,366.493,408.164,453.112,501.52,553.576,609.481,669.442,733.679,802.422,875.911,954.4,1038.15,1127.45,1222.58,1323.84,1431.57,1546.09,1667.76,1796.95,1934.03,2079.43,2233.55,2396.86,2569.81,2752.89,2946.61,3151.52,3368.16,3597.14,3839.07,4094.59,4364.38,4649.15,4949.65,5266.64,5600.94,5953.41,6324.94,6716.47,7128.96,7563.46,8021.03,8502.79,9009.92,9543.66,10105.3,10696.2,11317.7,11971.4,12658.7,13381.4,14141,14939.5,15778.5,16660.2,17586.5,18559.5,19581.5,20654.8,21781.9,22965.3,24207.7,25511.8,26880.7,28317.4,29825,31407,33066.9,34808.2,36635,38551.1,40560.8,42668.6,44878.9,47196.7,49626.9,52174.9,54846.1,57646.3,60581.7,63658.4,66883.1,70262.6,73804.4,77515.8,81404.8,85479.7,89749.2,94222.2,98908.4,103817,108960,114346,119989,125898,132087,138569,145358,152466,159910,167705,175868,184414,193362,202730,212538,222806,233556,244810,256590,268921,281829,295340,309481,324283,339775};

	const int histoSize = 10;
	char title[histoSize][50];
	for (int i =0; i < histoSize; i++){
		char buff[50];
		sprintf(buff, "%s%d", name,i);
		for (int k =0; k < 50; k++)
			title[i][k] = buff[k];
	}
       // (jetEta1 > eta2 && jetEta1 < eta3),  (jetEta2 > eta2 && jetEta2 < eta3 )
	hJet[0] =  new TH1F(title[0], "DiJet Mass", binSize, massBin); 
	
       // (jetEta1 > eta1 && jetEta1 < eta4),  (jetEta2 > eta1 && jetEta2 < eta4 )
	hJet[1] =  new TH1F(title[1], "DiJet Mass", binSize, massBin);
	
       // (jetEta1 > eta3 && jetEta1 < eta4),  (jetEta2 > eta3 && jetEta2 < eta4 )
	hJet[2] =  new TH1F(title[2], "DiJet Mass", binSize, massBin); 
	
       // (jetEta1 > eta1 && jetEta1 < eta2),  (jetEta2 > eta1 && jetEta2 < eta2 )
	hJet[3] =  new TH1F(title[3], "DiJet Mass", binSize, massBin);
	
       // (jetEta1 > eta3 && jetEta1 < eta4),  (jetEta2 > eta1 && jetEta2 < eta2 )
	hJet[4] =  new TH1F(title[4], "DiJet Mass", binSize, massBin);
	
       // (jetEta1 > eta1 && jetEta1 < eta2),  (jetEta3 > eta2 && jetEta2 < eta4 )
	hJet[5] =  new TH1F(title[5], "DiJet Mass", binSize, massBin);
	
       // (jetEta1 > eta3 && jetEta1 < eta4),  (jetEta2 > eta3 && jetEta2 < eta4 )
	hJet[6] =  new TH1F(title[6], "DiJet Mass", binSize, massBin);
	
	hJet[7] =  new TH1F(title[7], "1^{st} Leading Jet #eta", 120, -6., 6.);
	hJet[8] =  new TH1F(title[8], "2^{nd} Leading Jet #eta", 120, -6., 6.);
	hJet[9] =  new TH1F(title[9], "No. of generated events", 10, 0.,10.);
   
   return ;
}


template <class R>
void histoFill(TH1F* jetHisto[], Handle<R> jetsRec, double eta1, double eta2, double eta3, double eta4)
{
	//For no. of events
	jetHisto[9]->Fill(1.);

	if ((*jetsRec).size() >=2){
	   double px1 = (*jetsRec)[0].px(); 
	   double py1 = (*jetsRec)[0].py(); 
	   double pz1 = (*jetsRec)[0].pz(); 
	   double e1 = (*jetsRec)[0].energy(); 
	   double jetEta1 = (*jetsRec)[0].eta(); 
	   jetHisto[7]->Fill(jetEta1);
	   
	   double px2 = (*jetsRec)[1].px(); 
	   double py2 = (*jetsRec)[1].py(); 
	   double pz2 = (*jetsRec)[1].pz(); 
	   double e2 = (*jetsRec)[1].energy(); 
	   double jetEta2 = (*jetsRec)[1].eta();
	   jetHisto[8]->Fill(jetEta2);

	   CLHEP::HepLorentzVector v1(px1,py1,pz1,e1);	   
	   CLHEP::HepLorentzVector v2(px2,py2,pz2,e2);	   
	   CLHEP::HepLorentzVector v(0.,0.,0.,0.);	   
	   v = v1 + v2; 
	   float mass = v.m();

           if (jetEta1 > eta2 && jetEta1 < eta3) 
		   if (jetEta2 > eta2 && jetEta2 < eta3) 
			  jetHisto[0]->Fill(mass);
	   
           if (jetEta1 > eta1 && jetEta1 < eta4) 
		   if (jetEta2 > eta1 && jetEta2 < eta4) 
			  jetHisto[1]->Fill(mass);
	   
           if (jetEta1 > eta3 && jetEta1 < eta4) 
		   if (jetEta2 > eta3 && jetEta2 < eta4) 
			  jetHisto[2]->Fill(mass);
	   
           if (jetEta1 > eta1 && jetEta1 < eta2) 
		   if (jetEta2 > eta1 && jetEta2 < eta2) 
			  jetHisto[3]->Fill(mass);
	   
           if (jetEta1 > eta3 && jetEta1 < eta4) 
		   if (jetEta2 > eta1 && jetEta2 < eta2) 
			  jetHisto[4]->Fill(mass);
	   
           if (jetEta1 > eta1 && jetEta1 < eta2) 
		   if (jetEta2 > eta3 && jetEta2 < eta4) 
			  jetHisto[5]->Fill(mass);
	   
	   if (abs(jetEta1) > eta3 && abs(jetEta1) < eta4)
		   if (abs(jetEta2) > eta3 && abs(jetEta2) < eta4)
			  jetHisto[6]->Fill(mass);
	   
	  
	}//(*jetsRec).size() >=2
}//histoFill


//
// class decleration
//

class DijetRatio : public edm::EDAnalyzer {
   public:
      explicit DijetRatio(const edm::ParameterSet&);
      ~DijetRatio();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
     string fOutputFileName ;
     
     // names of modules, producing object collections
     string m_Mid5CorRecJetsSrc;
     string m_Mid5GenJetsSrc;
     string m_Mid5CaloJetsSrc;

     // eta limit 
     double  m_eta3;  // eta limit for numerator
     double  m_eta4;  // eta limit for denominator
  
     //Jet Kinematics for leading Jet
     static  const int hisotNumber = 10;
  
     TH1F*  hGen[hisotNumber];
     TH1F*  hCalo[hisotNumber];
     TH1F*  hCor[hisotNumber];
     
     // Root file for saving histo
     TFile*      hOutputFile ;
};

#endif

