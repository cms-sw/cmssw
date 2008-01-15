//
// $Id: ObjectResolutionCalc.cc,v 1.1 2008/01/07 11:48:27 lowette Exp $
//

#include "PhysicsTools/PatUtils/interface/ObjectResolutionCalc.h"


using namespace pat;


// constructor with path; default should not be used
ObjectResolutionCalc::ObjectResolutionCalc(TString resopath, bool useNN = false): useNN_(useNN) {
  std::cout << "=== Constructing a TopObjectResolutionCalc... " << std::endl; 
  resoFile_ = new TFile(resopath);
  if (!resoFile_) std::cout<<"### No resolutions fits for this file available: "<<resopath<<"... ###"<<std::endl;
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
            fResVsET_[p][etaBin] = (TF1)(*(tmp -> GetFunction("F_"+name)));
	  }
        }
      }
    }
  }
  // find etabin values
  TH1F *tmpEta = (TH1F*) (resoFile_->GetKey("hEtaBins")->ReadObj());
  for(int b=1; b<=tmpEta->GetNbinsX(); b++) etaBinVals_.push_back(tmpEta->GetXaxis()->GetBinLowEdge(b));
  etaBinVals_.push_back(tmpEta->GetXaxis()->GetBinUpEdge(tmpEta->GetNbinsX()));
  std::cout<<"Found "<<etaBinVals_.size()-1<< " eta-bins with edges: ( ";
  for(size_t u=0; u<etaBinVals_.size(); u++) std::cout<<etaBinVals_[u]<<", ";
  std::cout<<"\b\b )"<<std::endl;
  
  std::cout << "=== done." << std::endl;
}


// destructor
ObjectResolutionCalc::~ObjectResolutionCalc() {
  delete resoFile_;
}


float ObjectResolutionCalc::getObsRes(int obs, int eta, float eT) {
  if (useNN_) throw edm::Exception( edm::errors::LogicError, 
                                   "TopObjectResolutionCalc::getObsRes should never be called when using a NN for resolutions." );
  float res = fResVsET_[obs][eta].Eval(eT);
  return res;
}


int ObjectResolutionCalc::getEtaBin(float eta) {
  int nrEtaBins = etaBinVals_.size()-1;
  int bin = nrEtaBins-1;
  for(int i=0; i<nrEtaBins; i++) {
    if(fabs(eta) > etaBinVals_[i] && fabs(eta) < etaBinVals_[i+1]) bin = i;
  }
  return bin;
}


void ObjectResolutionCalc::operator()(Electron & obj) {
  if (useNN_) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResA(     network_[0]->Evaluate(0,v ));
    obj.setResB(     network_[1]->Evaluate(0,v ));
    obj.setResC(     network_[2]->Evaluate(0,v ));
    obj.setResD(     network_[3]->Evaluate(0,v ));	
    obj.setResTheta( network_[4]->Evaluate(0,v ));	 
    obj.setResPhi(   network_[5]->Evaluate(0,v ));	
    obj.setResET(    network_[6]->Evaluate(0,v ));	
    obj.setResEta(   network_[7]->Evaluate(0,v ));
  } else {
    int bin = this->getEtaBin(obj.eta());
    obj.setResA(     this->getObsRes(0,bin,obj.et()) );
    obj.setResB(     this->getObsRes(1,bin,obj.et()) );
    obj.setResC(     this->getObsRes(2,bin,obj.et()) );	
    obj.setResD(     this->getObsRes(3,bin,obj.et()) );	
    obj.setResTheta( this->getObsRes(4,bin,obj.et()) );	 
    obj.setResPhi(   this->getObsRes(5,bin,obj.et()) );	
    obj.setResET(    this->getObsRes(6,bin,obj.et()) );	
    obj.setResEta(   this->getObsRes(7,bin,obj.et()) );
  }
}


void ObjectResolutionCalc::operator()(Muon & obj) {
  if (useNN_) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResA(     network_[0]->Evaluate(0,v ));
    obj.setResB(     network_[1]->Evaluate(0,v ));
    obj.setResC(     network_[2]->Evaluate(0,v ));
    obj.setResD(     network_[3]->Evaluate(0,v ));	
    obj.setResTheta( network_[4]->Evaluate(0,v ));	 
    obj.setResPhi(   network_[5]->Evaluate(0,v ));	
    obj.setResET(    network_[6]->Evaluate(0,v ));	
    obj.setResEta(   network_[7]->Evaluate(0,v ));
  } else {
    int bin = this->getEtaBin(obj.eta());
    obj.setResA(     this->getObsRes(0,bin,obj.et()) );
    obj.setResB(     this->getObsRes(1,bin,obj.et()) );
    obj.setResC(     this->getObsRes(2,bin,obj.et()) );	
    obj.setResD(     this->getObsRes(3,bin,obj.et()) );	
    obj.setResTheta( this->getObsRes(4,bin,obj.et()) );	 
    obj.setResPhi(   this->getObsRes(5,bin,obj.et()) );	
    obj.setResET(    this->getObsRes(6,bin,obj.et()) );	
    obj.setResEta(   this->getObsRes(7,bin,obj.et()) );
  }
}


void ObjectResolutionCalc::operator()(Jet & obj) {
  if (useNN_) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResA(     network_[0]->Evaluate(0,v ));
    obj.setResB(     network_[1]->Evaluate(0,v ));
    obj.setResC(     network_[2]->Evaluate(0,v ));
    obj.setResD(     network_[3]->Evaluate(0,v ));	
    obj.setResTheta( network_[4]->Evaluate(0,v ));	 
    obj.setResPhi(   network_[5]->Evaluate(0,v ));	
    obj.setResET(    network_[6]->Evaluate(0,v ));	
    obj.setResEta(   network_[7]->Evaluate(0,v ));
  } else {
    int bin = this->getEtaBin(obj.eta());
    obj.setResA(     this->getObsRes(0,bin,obj.et()) );
    obj.setResB(     this->getObsRes(1,bin,obj.et()) );
    obj.setResC(     this->getObsRes(2,bin,obj.et()) );	
    obj.setResD(     this->getObsRes(3,bin,obj.et()) );	
    obj.setResTheta( this->getObsRes(4,bin,obj.et()) );	 
    obj.setResPhi(   this->getObsRes(5,bin,obj.et()) );	
    obj.setResET(    this->getObsRes(6,bin,obj.et()) );	
    obj.setResEta(   this->getObsRes(7,bin,obj.et()) );
  }
}


void ObjectResolutionCalc::operator()(MET & obj) {
  if (useNN_) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResA(     network_[0]->Evaluate(0,v ));
    obj.setResB(     network_[1]->Evaluate(0,v ));
    obj.setResC(     network_[2]->Evaluate(0,v ));
    obj.setResD(     network_[3]->Evaluate(0,v ));
    obj.setResTheta( 1000000.  );   			// Total freedom	
    obj.setResPhi(   network_[5]->Evaluate(0,v ));	
    obj.setResET(    network_[6]->Evaluate(0,v ));	
    obj.setResEta(   1000000.  );    			// Total freedom
  } else {
    obj.setResA(     this->getObsRes(0,0,obj.et())  );
    obj.setResC(     this->getObsRes(1,0,obj.et())  );
    obj.setResB(     this->getObsRes(2,0,obj.et())  );
    obj.setResD(     this->getObsRes(3,0,obj.et())  );
    obj.setResTheta( 1000000.  );   			// Total freedom
    obj.setResPhi(   this->getObsRes(5,0,obj.et())  );
    obj.setResET(    this->getObsRes(6,0,obj.et())  );
    obj.setResEta(   1000000.  );    			// Total freedom
  }
}


void ObjectResolutionCalc::operator()(Tau & obj) {
  if (useNN_) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResA(     network_[0]->Evaluate(0,v ));
    obj.setResB(     network_[1]->Evaluate(0,v ));
    obj.setResC(     network_[2]->Evaluate(0,v ));
    obj.setResD(     network_[3]->Evaluate(0,v ));	
    obj.setResTheta( network_[4]->Evaluate(0,v ));	 
    obj.setResPhi(   network_[5]->Evaluate(0,v ));	
    obj.setResET(    network_[6]->Evaluate(0,v ));	
    obj.setResEta(   network_[7]->Evaluate(0,v ));
  } else {
    int bin = this->getEtaBin(obj.eta());
    obj.setResA(     this->getObsRes(0,bin,obj.et()) );
    obj.setResB(     this->getObsRes(1,bin,obj.et()) );
    obj.setResC(     this->getObsRes(2,bin,obj.et()) );	
    obj.setResD(     this->getObsRes(3,bin,obj.et()) );	
    obj.setResTheta( this->getObsRes(4,bin,obj.et()) );	 
    obj.setResPhi(   this->getObsRes(5,bin,obj.et()) );	
    obj.setResET(    this->getObsRes(6,bin,obj.et()) );	
    obj.setResEta(   this->getObsRes(7,bin,obj.et()) );
  }
}
