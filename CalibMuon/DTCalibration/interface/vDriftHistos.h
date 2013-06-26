#ifndef vDriftHistos_H
#define vDriftHistos_H

#include "TH1.h"
#include "TString.h"
#include "TFile.h"
#include "DTTMax.h"
#include <string>

// A set of histograms on chamber angle and position
class h4DSegm{
 public:
  h4DSegm(std::string name_){
    TString N = name_.c_str();
    name=name_.c_str();
    h4DSegmXPosInCham     = new TH1F(N+"_h4DSegmXPosInCham", 
				    "4D Segment x position (cm) in Chamber RF", 200, -200, 200); 
    h4DSegmYPosInCham     = new TH1F(N+"_h4DSegmYPosInCham", 
				    "4D Segment y position (cm) in Chamber RF", 200, -200, 200); 
    h4DSegmPhiAngleInCham   = new TH1F(N+"_h4DSegmPhiAngleInCham",  
 				    "4D Segment phi angle (rad) in Chamber RF", 180, -180, 180); 
    h4DSegmThetaAngleInCham   = new TH1F(N+"_h4DSegmThetaAngleInCham",  
 				    "4D Segment theta angle (rad) in Chamber RF", 180, -180, 180); 
    h4DSegmImpactAngleInCham   = new TH1F(N+"_h4DSegmImpactAngleInCham",  
 				    "4D Segment impact angle (rad) in Chamber RF", 180, -180, 180); 
  }
 h4DSegm(const TString& name_, TFile* file){
    name=name_;
    h4DSegmXPosInCham  = (TH1F *) file->Get(name+"_h4DSegmXPosInCham"); 
    h4DSegmYPosInCham  = (TH1F *) file->Get(name+"_h4DSegmYPosInCham"); 
    h4DSegmPhiAngleInCham  = (TH1F *) file->Get(name+"_h4DSegmPhiAngleInCham"); 
    h4DSegmThetaAngleInCham  = (TH1F *) file->Get(name+"_h4DSegmThetaAngleInCham"); 
    h4DSegmImpactAngleInCham  = (TH1F *) file->Get(name+"_h4DSegmImpactAngleInCham"); 
 }
 ~h4DSegm(){
    delete h4DSegmXPosInCham;     
    delete h4DSegmYPosInCham;     
    delete h4DSegmPhiAngleInCham;   
    delete h4DSegmThetaAngleInCham;   
    delete h4DSegmImpactAngleInCham;   
 } 
void Fill(float x, float y, float phi, float theta, float impact) {
    h4DSegmXPosInCham->Fill(x); 
    h4DSegmYPosInCham->Fill(y); 
    h4DSegmPhiAngleInCham->Fill(phi);   
    h4DSegmThetaAngleInCham->Fill(theta);   
    h4DSegmImpactAngleInCham->Fill(impact);   
} 
void Fill(float x, float phi) {
    h4DSegmXPosInCham->Fill(x); 
    h4DSegmPhiAngleInCham->Fill(phi);   
} 
 void Write() {
    h4DSegmXPosInCham->Write();     
    h4DSegmYPosInCham->Write();     
    h4DSegmPhiAngleInCham->Write();   
    h4DSegmThetaAngleInCham->Write();   
    h4DSegmImpactAngleInCham->Write();   
  }
 public:

  TH1F *h4DSegmXPosInCham;     
  TH1F *h4DSegmYPosInCham;     
  TH1F *h4DSegmPhiAngleInCham;   
  TH1F *h4DSegmThetaAngleInCham;   
  TH1F *h4DSegmImpactAngleInCham;   

  TString name;
};


// A set of histograms on SL angle and position
class h2DSegm{
 public:
  h2DSegm(std::string name_){
    TString N = name_.c_str();
    name=name_.c_str();
    h2DSegmPosInCham     = new TH1F(N+"_h2DSegmPosInCham", 
				    "2D Segment position (cm) in Chamber RF", 200, -200, 200); 
    h2DSegmAngleInCham   = new TH1F(N+"_h2DSegmAngleInCham",  
 				    "2D Segment angle (rad) in Chamber RF", 200, -2, 2); 
    h2DSegmCosAngleInCham   = new TH1F(N+"_h2DSegmCosAngleInCham",  
 				       "2D Segment cos(angle) in Chamber RF", 200, -2, 2); 
  }
  h2DSegm(const TString& name_, TFile* file){
    name=name_;

    h2DSegmPosInCham  = (TH1F *) file->Get(name+"_h2DSegmPosInCham"); 
    h2DSegmAngleInCham  = (TH1F *) file->Get(name+"_h2DSegmAngleInCham"); 
    h2DSegmCosAngleInCham  = (TH1F *) file->Get(name+"_h2DSegmCosAngleInCham"); 
  }
  ~h2DSegm(){
    delete h2DSegmPosInCham;     
    delete h2DSegmAngleInCham;   
    delete h2DSegmCosAngleInCham;   
  }
  void Fill(float pos, float localAngle) {

    h2DSegmPosInCham->Fill(pos); 
    h2DSegmAngleInCham->Fill(atan(localAngle));   
    h2DSegmCosAngleInCham->Fill(cos(atan(localAngle)));   
  }
  void Write() {
    h2DSegmPosInCham->Write();     
    h2DSegmAngleInCham->Write();   
    h2DSegmCosAngleInCham->Write();   
  }
 public:

  TH1F *h2DSegmPosInCham;     
  TH1F *h2DSegmAngleInCham;   
  TH1F *h2DSegmCosAngleInCham;   

  TString name;
};

// A set of histograms on SL Tmax
class hTMaxCell{
 public:
  hTMaxCell(const TString& name_){
    name = name_;

    // book TMax histograms 
    hTmax123      = new TH1F (name+"_Tmax123", "Tmax123 value", 2000, -1000., 1000.);
    hTmax124s72   = new TH1F (name+"_Tmax124_s72", "Tmax124 sigma=sqrt(7/2) value", 2000, -1000., 1000.);
    hTmax124s78   = new TH1F (name+"_Tmax124_s78", "Tmax124 sigma=sqrt(7/8) value", 2000, -1000., 1000.);
    hTmax134s72   = new TH1F (name+"_Tmax134_s72", "Tmax134 sigma=sqrt(7/2) value", 2000, -1000., 1000.);
    hTmax134s78   = new TH1F (name+"_Tmax134_s78", "Tmax134 sigma=sqrt(7/8) value", 2000, -1000., 1000.);
    hTmax234      = new TH1F (name+"_Tmax234", "Tmax234 value", 2000, -1000., 1000.);
    hTmax_3t0  = new TH1F (name+"_3t0", "Tmax+3*Delta(t0)", 2000, -1000., 1000.); 
    hTmax_3t0_0  = new TH1F (name+"_3t0_0", "Tmax+3*Delta(t0); 3 hits", 2000, -1000., 1000.); 
    hTmax_3t0_1  = new TH1F (name+"_3t0_1", "Tmax+3*Delta(t0); one t<5ns", 2000, -1000., 1000.); 
    hTmax_3t0_2  = new TH1F (name+"_3t0_2", "Tmax+3*Delta(t0); one t<10ns", 2000, -1000., 1000.); 
    hTmax_3t0_3  = new TH1F (name+"_3t0_3", "Tmax+3*Delta(t0); one t<20ns", 2000, -1000., 1000.); 
    hTmax_3t0_4  = new TH1F (name+"_3t0_4", "Tmax+3*Delta(t0); one t<50ns", 2000, -1000., 1000.); 
    hTmax_3t0_5  = new TH1F (name+"_3t0_5", "Tmax+3*Delta(t0); all t>50ns", 2000, -1000., 1000.); 
    hTmax_2t0  = new TH1F (name+"_2t0", "Tmax+2*Delta(t0)", 2000, -1000., 1000.);
    hTmax_2t0_0  = new TH1F (name+"_2t0_0", "Tmax+2*Delta(t0); 3 hits", 2000, -1000., 1000.); 
    hTmax_2t0_1  = new TH1F (name+"_2t0_1", "Tmax+2*Delta(t0); one t<5ns", 2000, -1000., 1000.); 
    hTmax_2t0_2  = new TH1F (name+"_2t0_2", "Tmax+2*Delta(t0); one t<10ns", 2000, -1000., 1000.); 
    hTmax_2t0_3  = new TH1F (name+"_2t0_3", "Tmax+2*Delta(t0); one t<20ns", 2000, -1000., 1000.); 
    hTmax_2t0_4  = new TH1F (name+"_2t0_4", "Tmax+2*Delta(t0); one t<50ns", 2000, -1000., 1000.); 
    hTmax_2t0_5  = new TH1F (name+"_2t0_5", "Tmax+2*Delta(t0); all t>50ns", 2000, -1000., 1000.); 
    hTmax_t0   = new TH1F (name+"_t0", "Tmax+Delta(t0)", 2000, -1000., 1000.);
    hTmax_t0_0  = new TH1F (name+"_t0_0", "Tmax+Delta(t0); 3 hits", 2000, -1000., 1000.); 
    hTmax_t0_1  = new TH1F (name+"_t0_1", "Tmax+Delta(t0); one t<5ns", 2000, -1000., 1000.); 
    hTmax_t0_2  = new TH1F (name+"_t0_2", "Tmax+Delta(t0); one t<10ns", 2000, -1000., 1000.); 
    hTmax_t0_3  = new TH1F (name+"_t0_3", "Tmax+Delta(t0); one t<20ns", 2000, -1000., 1000.); 
    hTmax_t0_4  = new TH1F (name+"_t0_4", "Tmax+Delta(t0); one t<50ns", 2000, -1000., 1000.); 
    hTmax_t0_5  = new TH1F (name+"_t0_5", "Tmax+Delta(t0); all t>50ns", 2000, -1000., 1000.); 
    hTmax_0    = new TH1F (name+"_0", "Tmax", 2000, -1000., 1000.);
  }


  hTMaxCell (const TString& name_, TFile* file){
    name=name_;
    hTmax123      = (TH1F *) file->Get(name+"_Tmax123");
    hTmax124s72   = (TH1F *) file->Get(name+"_Tmax124_s72");
    hTmax124s78   = (TH1F *) file->Get(name+"_Tmax124_s78");
    hTmax134s72   = (TH1F *) file->Get(name+"_Tmax134_s72");
    hTmax134s78   = (TH1F *) file->Get(name+"_Tmax134_s78");
    hTmax234      = (TH1F *) file->Get(name+"_Tmax234");
    hTmax_3t0  = (TH1F *) file->Get(name+"_3t0");
    hTmax_3t0_0  = (TH1F *) file->Get(name+"_3t0_0");
    hTmax_3t0_1  = (TH1F *) file->Get(name+"_3t0_1");
    hTmax_3t0_2  = (TH1F *) file->Get(name+"_3t0_2");
    hTmax_3t0_3  = (TH1F *) file->Get(name+"_3t0_3");
    hTmax_3t0_4  = (TH1F *) file->Get(name+"_3t0_4");
    hTmax_3t0_5  = (TH1F *) file->Get(name+"_3t0_5");
    hTmax_2t0  = (TH1F *) file->Get(name+"_2t0");
    hTmax_2t0_0  = (TH1F *) file->Get(name+"_2t0_0");
    hTmax_2t0_1  = (TH1F *) file->Get(name+"_2t0_1");
    hTmax_2t0_2  = (TH1F *) file->Get(name+"_2t0_2");
    hTmax_2t0_3  = (TH1F *) file->Get(name+"_2t0_3");
    hTmax_2t0_4  = (TH1F *) file->Get(name+"_2t0_4");
    hTmax_2t0_5  = (TH1F *) file->Get(name+"_2t0_5");
    hTmax_t0  = (TH1F *) file->Get(name+"_t0");
    hTmax_t0_1  = (TH1F *) file->Get(name+"_t0_1");
    hTmax_t0_2  = (TH1F *) file->Get(name+"_t0_2");
    hTmax_t0_3  = (TH1F *) file->Get(name+"_t0_3");
    hTmax_t0_4  = (TH1F *) file->Get(name+"_t0_4");
    hTmax_t0_5  = (TH1F *) file->Get(name+"_t0_5");
    hTmax_0  = (TH1F *) file->Get(name+"_0");

  }

 
  ~hTMaxCell(){
    delete hTmax123;
    delete hTmax124s72;
    delete hTmax124s78;
    delete hTmax134s72;
    delete hTmax134s78;
    delete hTmax234;
    delete hTmax_3t0;
    delete hTmax_3t0_0;
    delete hTmax_3t0_1;
    delete hTmax_3t0_2;
    delete hTmax_3t0_3;
    delete hTmax_3t0_4;
    delete hTmax_3t0_5;
    delete hTmax_2t0;
    delete hTmax_2t0_0;
    delete hTmax_2t0_1;
    delete hTmax_2t0_2;
    delete hTmax_2t0_3;
    delete hTmax_2t0_4;
    delete hTmax_2t0_5;
    delete hTmax_t0;
    delete hTmax_t0_0;
    delete hTmax_t0_1;
    delete hTmax_t0_2;
    delete hTmax_t0_3;
    delete hTmax_t0_4;
    delete hTmax_t0_5;
    delete hTmax_0;

  }

  void Fill(float tmax123, float tmax124, float tmax134, float tmax234,
	    dttmaxenums::SigmaFactor s124,// Give the factor relating the width of the TMax distribution
	    dttmaxenums::SigmaFactor s134,// and the cell resolution (for tmax123 and tmax234 is always= sqrt(3/2)).
	    unsigned t0_123, // Give the "quantity" of Delta(t0) included in the tmax 
	    unsigned t0_124, // formula used for Tmax123 or Tma124 or Tma134 or Tma234
	    unsigned t0_134,
	    unsigned t0_234,
	    unsigned hSubGroup//different t0 hists(at least one hit within a given distance from the wire)
	    ) {
 
    if(tmax123 > 0.) {
      hTmax123->Fill(tmax123);
      if(t0_123==1) {
	hTmax_t0->Fill(tmax123);
	switch(hSubGroup) {
	case 0: hTmax_t0_0->Fill(tmax123); break;
	case 1: hTmax_t0_1->Fill(tmax123); break;
	case 2: hTmax_t0_2->Fill(tmax123); break;
	case 3: hTmax_t0_3->Fill(tmax123); break;
	case 4: hTmax_t0_4->Fill(tmax123); break;
	case 99: hTmax_t0_5->Fill(tmax123); break;
	}
      }
      else {
	hTmax_2t0->Fill(tmax123);
	switch(hSubGroup) {
	case 0: hTmax_2t0_0->Fill(tmax123); break;
	case 1: hTmax_2t0_1->Fill(tmax123); break;
	case 2: hTmax_2t0_2->Fill(tmax123); break;
	case 3: hTmax_2t0_3->Fill(tmax123); break;
	case 4: hTmax_2t0_4->Fill(tmax123); break;
	case 99: hTmax_2t0_5->Fill(tmax123); break;
	}
      }
    }
    if(tmax124 > 0.) {
      (s124==dttmaxenums::r72)? hTmax124s72->Fill(tmax124):hTmax124s78->Fill(tmax124);
      if(t0_124==0)  
	hTmax_0->Fill(tmax124);
      else if(t0_124==1) {
	hTmax_t0->Fill(tmax124); 
	switch(hSubGroup) {
	case 0: hTmax_t0_0->Fill(tmax124); break;
	case 1: hTmax_t0_1->Fill(tmax124); break;
	case 2: hTmax_t0_2->Fill(tmax124); break;
	case 3: hTmax_t0_3->Fill(tmax124); break;
	case 4: hTmax_t0_4->Fill(tmax124); break;
	case 99: hTmax_t0_5->Fill(tmax124); break;
	}
      }
      else if(t0_124== 2) {
	hTmax_2t0->Fill(tmax124);
	switch(hSubGroup) {
	case 0: hTmax_2t0_0->Fill(tmax124); break;
	case 1: hTmax_2t0_1->Fill(tmax124); break;
	case 2: hTmax_2t0_2->Fill(tmax124); break;
	case 3: hTmax_2t0_3->Fill(tmax124); break;
	case 4: hTmax_2t0_4->Fill(tmax124); break;
	case 99: hTmax_2t0_5->Fill(tmax124); break;
	}
      }
      else if(t0_124==3) {
	hTmax_3t0->Fill(tmax124); 
	switch(hSubGroup) {
	case 0: hTmax_3t0_0->Fill(tmax124); break;
	case 1: hTmax_3t0_1->Fill(tmax124); break;
	case 2: hTmax_3t0_2->Fill(tmax124); break;
	case 3: hTmax_3t0_3->Fill(tmax124); break;
	case 4: hTmax_3t0_4->Fill(tmax124); break;
	case 99: hTmax_3t0_5->Fill(tmax124);break;
	}      
      }
    }
    if(tmax134 > 0.) {
      (s134==dttmaxenums::r72)? hTmax134s72->Fill(tmax134):hTmax134s78->Fill(tmax134);
      if(t0_134==0)  
	hTmax_0->Fill(tmax134);
      else if(t0_134==1) {
	hTmax_t0->Fill(tmax134); 
	switch(hSubGroup) {
	case 0: hTmax_t0_0->Fill(tmax134); break;
	case 1: hTmax_t0_1->Fill(tmax134); break;
	case 2: hTmax_t0_2->Fill(tmax134); break;
	case 3: hTmax_t0_3->Fill(tmax134); break;
	case 4: hTmax_t0_4->Fill(tmax134); break;
	case 99: hTmax_t0_5->Fill(tmax134); break;
	}
      }
      else if(t0_134== 2) {
	hTmax_2t0->Fill(tmax134);
	switch(hSubGroup) {
	case 0: hTmax_2t0_0->Fill(tmax134); break;
	case 1: hTmax_2t0_1->Fill(tmax134); break;
	case 2: hTmax_2t0_2->Fill(tmax134); break;
	case 3: hTmax_2t0_3->Fill(tmax134); break;
	case 4: hTmax_2t0_4->Fill(tmax134); break;
	case 99: hTmax_2t0_5->Fill(tmax134); break;
	}
      }
      else if(t0_134==3) {
	hTmax_3t0->Fill(tmax134); 
	switch(hSubGroup) {
	case 0: hTmax_3t0_0->Fill(tmax134); break;
	case 1: hTmax_3t0_1->Fill(tmax134); break;
	case 2: hTmax_3t0_2->Fill(tmax134); break;
	case 3: hTmax_3t0_3->Fill(tmax134); break;
	case 4: hTmax_3t0_4->Fill(tmax134); break;
	case 99: hTmax_3t0_5->Fill(tmax134); break;
	}      
      }
    }
    if(tmax234 > 0.) {
      hTmax234->Fill(tmax234);
      if(t0_234==1) {
	hTmax_t0->Fill(tmax234);
	switch(hSubGroup) {
	case 0: hTmax_t0_0->Fill(tmax234); break;
	case 1: hTmax_t0_1->Fill(tmax234); break;
	case 2: hTmax_t0_2->Fill(tmax234); break;
	case 3: hTmax_t0_3->Fill(tmax234); break;
	case 4: hTmax_t0_4->Fill(tmax234); break;
	case 99: hTmax_t0_5->Fill(tmax234);break;
	}
      }
      else {
	hTmax_2t0->Fill(tmax234);
	switch(hSubGroup) {
	case 0: hTmax_2t0_0->Fill(tmax234); break;
	case 1: hTmax_2t0_1->Fill(tmax234); break;
	case 2: hTmax_2t0_2->Fill(tmax234); break;
	case 3: hTmax_2t0_3->Fill(tmax234); break;
	case 4: hTmax_2t0_4->Fill(tmax234); break;
	case 99: hTmax_2t0_5->Fill(tmax234); break;
	}
      }
    }      
  }

  void Write() { 
    // write the Tmax histograms 
    hTmax123->Write();
    hTmax124s72->Write();
    hTmax124s78->Write();
    hTmax134s72->Write();
    hTmax134s78->Write();
    hTmax234->Write(); 
    hTmax_3t0->Write();
    hTmax_3t0_0->Write();
    hTmax_3t0_1->Write();
    hTmax_3t0_2->Write();
    hTmax_3t0_3->Write();
    hTmax_3t0_4->Write();
    hTmax_3t0_5->Write();
    hTmax_2t0->Write();
    hTmax_2t0_0->Write();
    hTmax_2t0_1->Write();
    hTmax_2t0_2->Write();
    hTmax_2t0_3->Write();
    hTmax_2t0_4->Write();
    hTmax_2t0_5->Write();
    hTmax_t0->Write();
    hTmax_t0_0->Write();
    hTmax_t0_1->Write();
    hTmax_t0_2->Write();
    hTmax_t0_3->Write();
    hTmax_t0_4->Write();
    hTmax_t0_5->Write();
    hTmax_0->Write();

  }

  int GetT0Factor(TH1F* hist) {
    unsigned t0 = 999;

    if (hist == hTmax_3t0)
      t0 = 3;
    else if(hist == hTmax_2t0)
      t0 = 2;
    else if(hist ==  hTmax_t0)
      t0 = 1;    
    else if (hist == hTmax_0)
      t0 = 0;
       
    return t0;
  }
 
  TH1F* hTmax123;
  TH1F* hTmax124s72;
  TH1F* hTmax124s78;
  TH1F* hTmax134s72;
  TH1F* hTmax134s78;
  TH1F* hTmax234;
  TH1F* hTmax_3t0;
  TH1F* hTmax_3t0_0;
  TH1F* hTmax_3t0_1;
  TH1F* hTmax_3t0_2;
  TH1F* hTmax_3t0_3;
  TH1F* hTmax_3t0_4;
  TH1F* hTmax_3t0_5;
  TH1F* hTmax_2t0;
  TH1F* hTmax_2t0_0;
  TH1F* hTmax_2t0_1;
  TH1F* hTmax_2t0_2;
  TH1F* hTmax_2t0_3;
  TH1F* hTmax_2t0_4;
  TH1F* hTmax_2t0_5;
  TH1F* hTmax_t0;
  TH1F* hTmax_t0_0;
  TH1F* hTmax_t0_1;
  TH1F* hTmax_t0_2;
  TH1F* hTmax_t0_3;
  TH1F* hTmax_t0_4;
  TH1F* hTmax_t0_5;
  TH1F* hTmax_0;

  TString name;  
};

#endif
