//
// $Id: ObjectResolutionCalc.cc,v 1.5 2008/10/08 19:19:25 gpetrucc Exp $
//

#include "PhysicsTools/PatUtils/interface/ObjectResolutionCalc.h"


using namespace pat;


// constructor with path; default should not be used
ObjectResolutionCalc::ObjectResolutionCalc(TString resopath, bool useNN = false): useNN_(useNN) {
  edm::LogVerbatim("ObjectResolutionCalc") << ("ObjectResolutionCalc") << "=== Constructing a TopObjectResolutionCalc...";
  resoFile_ = new TFile(resopath);
  if (!resoFile_) edm::LogError("ObjectResolutionCalc") << "No resolutions fits for this file available: "<<resopath<<"...";
  TString  resObsName[8] = {"_ares","_bres","_cres","_dres","_thres","_phres","_etres","_etares"};
  
  TList* keys = resoFile_->GetListOfKeys();
  TIter nextitem(keys);
  TKey* key = NULL;
  while((key = (TKey*)nextitem())) {
    TString name = key->GetName();
    if(useNN_) {
      for(Int_t ro=0; ro<8; ro++) {
        TString obsName = resObsName[ro]; obsName += "_NN";
        if(name.Contains(obsName)){
	  network_[ro] = (TMultiLayerPerceptron*) resoFile_->GetKey(name)->ReadObj();
	}
      }
    }
    else 
    { 
      if(name.Contains("etabin") && (!name.Contains("etbin"))) {
        for(int p=0; p<8; p++){
          if(name.Contains(resObsName[p])){
            TString etabin = name; etabin.Remove(0,etabin.Index("_")+1); etabin.Remove(0,etabin.Index("_")+7);
            int etaBin = etabin.Atoi();
            TH1F *tmp = (TH1F*) (resoFile_->GetKey(name)->ReadObj());
            fResVsEt_[p][etaBin] = (TF1)(*(tmp -> GetFunction("F_"+name)));
	  }
        }
      }
    }
  }
  // find etabin values
  TH1F *tmpEta = (TH1F*) (resoFile_->GetKey("hEtaBins")->ReadObj());
  for(int b=1; b<=tmpEta->GetNbinsX(); b++) etaBinVals_.push_back(tmpEta->GetXaxis()->GetBinLowEdge(b));
  etaBinVals_.push_back(tmpEta->GetXaxis()->GetBinUpEdge(tmpEta->GetNbinsX()));
  edm::LogVerbatim("ObjectResolutionCalc") << "Found "<<etaBinVals_.size()-1  << " eta-bins with edges: ( ";
  for(size_t u=0; u<etaBinVals_.size(); u++) edm::LogVerbatim("ObjectResolutionCalc") << etaBinVals_[u]<<", ";
  edm::LogVerbatim("ObjectResolutionCalc") << "\b\b )"<<std::endl;
  
  edm::LogVerbatim("ObjectResolutionCalc") << "=== done." << std::endl;
}


// destructor
ObjectResolutionCalc::~ObjectResolutionCalc() {
  delete resoFile_;
}


float ObjectResolutionCalc::obsRes(int obs, int eta, float eT) {
  if (useNN_) throw edm::Exception( edm::errors::LogicError, 
                                   "TopObjectResolutionCalc::obsRes should never be called when using a NN for resolutions." );
  float res = fResVsEt_[obs][eta].Eval(eT);
  return res;
}


int ObjectResolutionCalc::etaBin(float eta) {
  int nrEtaBins = etaBinVals_.size()-1;
  int bin = nrEtaBins-1;
  for(int i=0; i<nrEtaBins; i++) {
    if(fabs(eta) > etaBinVals_[i] && fabs(eta) < etaBinVals_[i+1]) bin = i;
  }
  return bin;
}

#if OBSOLETE
void ObjectResolutionCalc::operator()(Electron & obj) {
  if (useNN_) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResolutionA(     network_[0]->Evaluate(0,v ));
    obj.setResolutionB(     network_[1]->Evaluate(0,v ));
    obj.setResolutionC(     network_[2]->Evaluate(0,v ));
    obj.setResolutionD(     network_[3]->Evaluate(0,v ));	
    obj.setResolutionTheta( network_[4]->Evaluate(0,v ));	 
    obj.setResolutionPhi(   network_[5]->Evaluate(0,v ));	
    obj.setResolutionEt(    network_[6]->Evaluate(0,v ));	
    obj.setResolutionEta(   network_[7]->Evaluate(0,v ));
  } else {
    int bin = this->etaBin(obj.eta());
    obj.setResolutionA(     this->obsRes(0,bin,obj.et()) );
    obj.setResolutionB(     this->obsRes(1,bin,obj.et()) );
    obj.setResolutionC(     this->obsRes(2,bin,obj.et()) ); 
    obj.setResolutionD(     this->obsRes(3,bin,obj.et()) ); 
    obj.setResolutionTheta( this->obsRes(4,bin,obj.et()) );  
    obj.setResolutionPhi(   this->obsRes(5,bin,obj.et()) ); 
    obj.setResolutionEt(    this->obsRes(6,bin,obj.et()) ); 
    obj.setResolutionEta(   this->obsRes(7,bin,obj.et()) );
  }
}


void ObjectResolutionCalc::operator()(Muon & obj) {
  if (useNN_) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResolutionA(     network_[0]->Evaluate(0,v ));
    obj.setResolutionB(     network_[1]->Evaluate(0,v ));
    obj.setResolutionC(     network_[2]->Evaluate(0,v ));
    obj.setResolutionD(     network_[3]->Evaluate(0,v ));	
    obj.setResolutionTheta( network_[4]->Evaluate(0,v ));	 
    obj.setResolutionPhi(   network_[5]->Evaluate(0,v ));	
    obj.setResolutionEt(    network_[6]->Evaluate(0,v ));	
    obj.setResolutionEta(   network_[7]->Evaluate(0,v ));
  } else {
    int bin = this->etaBin(obj.eta());
    obj.setResolutionA(     this->obsRes(0,bin,obj.et()) );
    obj.setResolutionB(     this->obsRes(1,bin,obj.et()) );
    obj.setResolutionC(     this->obsRes(2,bin,obj.et()) ); 
    obj.setResolutionD(     this->obsRes(3,bin,obj.et()) ); 
    obj.setResolutionTheta( this->obsRes(4,bin,obj.et()) );  
    obj.setResolutionPhi(   this->obsRes(5,bin,obj.et()) ); 
    obj.setResolutionEt(    this->obsRes(6,bin,obj.et()) ); 
    obj.setResolutionEta(   this->obsRes(7,bin,obj.et()) );
  }
}


void ObjectResolutionCalc::operator()(Jet & obj) {
  if (useNN_) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResolutionA(     network_[0]->Evaluate(0,v ));
    obj.setResolutionB(     network_[1]->Evaluate(0,v ));
    obj.setResolutionC(     network_[2]->Evaluate(0,v ));
    obj.setResolutionD(     network_[3]->Evaluate(0,v ));	
    obj.setResolutionTheta( network_[4]->Evaluate(0,v ));	 
    obj.setResolutionPhi(   network_[5]->Evaluate(0,v ));	
    obj.setResolutionEt(    network_[6]->Evaluate(0,v ));	
    obj.setResolutionEta(   network_[7]->Evaluate(0,v ));
  } else {
    int bin = this->etaBin(obj.eta());
    obj.setResolutionA(     this->obsRes(0,bin,obj.et()) );
    obj.setResolutionB(     this->obsRes(1,bin,obj.et()) );
    obj.setResolutionC(     this->obsRes(2,bin,obj.et()) ); 
    obj.setResolutionD(     this->obsRes(3,bin,obj.et()) ); 
    obj.setResolutionTheta( this->obsRes(4,bin,obj.et()) );  
    obj.setResolutionPhi(   this->obsRes(5,bin,obj.et()) ); 
    obj.setResolutionEt(    this->obsRes(6,bin,obj.et()) ); 
    obj.setResolutionEta(   this->obsRes(7,bin,obj.et()) );
  }
}


void ObjectResolutionCalc::operator()(MET & obj) {
  if (useNN_) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResolutionA(     network_[0]->Evaluate(0,v ));
    obj.setResolutionB(     network_[1]->Evaluate(0,v ));
    obj.setResolutionC(     network_[2]->Evaluate(0,v ));
    obj.setResolutionD(     network_[3]->Evaluate(0,v ));
    obj.setResolutionTheta( 1000000.  );   			// Total freedom	
    obj.setResolutionPhi(   network_[5]->Evaluate(0,v ));	
    obj.setResolutionEt(    network_[6]->Evaluate(0,v ));	
    obj.setResolutionEta(   1000000.  );    			// Total freedom
  } else {
    obj.setResolutionA(     this->obsRes(0,0,obj.et())  );
    obj.setResolutionC(     this->obsRes(1,0,obj.et())  );
    obj.setResolutionB(     this->obsRes(2,0,obj.et())  );
    obj.setResolutionD(     this->obsRes(3,0,obj.et())  );
    obj.setResolutionTheta( 1000000.  );   			// Total freedom
    obj.setResolutionPhi(   this->obsRes(5,0,obj.et())  );
    obj.setResolutionEt(    this->obsRes(6,0,obj.et())  );
    obj.setResolutionEta(   1000000.  );    			// Total freedom
  }
}


void ObjectResolutionCalc::operator()(Tau & obj) {
  if (useNN_) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResolutionA(     network_[0]->Evaluate(0,v ));
    obj.setResolutionB(     network_[1]->Evaluate(0,v ));
    obj.setResolutionC(     network_[2]->Evaluate(0,v ));
    obj.setResolutionD(     network_[3]->Evaluate(0,v ));	
    obj.setResolutionTheta( network_[4]->Evaluate(0,v ));	 
    obj.setResolutionPhi(   network_[5]->Evaluate(0,v ));	
    obj.setResolutionEt(    network_[6]->Evaluate(0,v ));	
    obj.setResolutionEta(   network_[7]->Evaluate(0,v ));
  } else {
    int bin = this->etaBin(obj.eta());
    obj.setResolutionA(     this->obsRes(0,bin,obj.et()) );
    obj.setResolutionB(     this->obsRes(1,bin,obj.et()) );
    obj.setResolutionC(     this->obsRes(2,bin,obj.et()) ); 
    obj.setResolutionD(     this->obsRes(3,bin,obj.et()) ); 
    obj.setResolutionTheta( this->obsRes(4,bin,obj.et()) );  
    obj.setResolutionPhi(   this->obsRes(5,bin,obj.et()) ); 
    obj.setResolutionEt(    this->obsRes(6,bin,obj.et()) ); 
    obj.setResolutionEta(   this->obsRes(7,bin,obj.et()) );
  }
}
#endif
