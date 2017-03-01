#include "PhysicsTools/PatUtils/interface/RazorComputer.h"
#include "TLorentzVector.h"

RazorBox::RazorBox( const CachingVariable::CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC) :
  CachingVariable("RazorBox",arg.n,arg.iConfig,iC){

}

void RazorBox::compute( const edm::Event & iEvent) const{
  
}

RazorComputer::RazorComputer( const CachingVariable::CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC) : VariableComputer(arg,iC){
  jet_ = edm::Service<InputTagDistributorService>()->retrieve("jet",arg.iConfig);
  met_ = edm::Service<InputTagDistributorService>()->retrieve("met",arg.iConfig);
  jetToken_ = iC.consumes<std::vector<pat::Jet>>(jet_);
  metToken_ = iC.consumes<std::vector<pat::MET>>(met_);
  pt_ = 40.;
  eta_=2.4;

  declare("MRT",iC);
  declare("MR",iC);
  declare("R2",iC);
  declare("R",iC);

}

namespace razor{
  typedef reco::Candidate::LorentzVector LorentzVector;

  double CalcMR(const LorentzVector &ja, const LorentzVector &jb){
    double A = ja.P();
    double B = jb.P();
    double az = ja.Pz();
    double bz = jb.Pz();

    double temp = sqrt((A+B)*(A+B)-(az+bz)*(az+bz));

    return temp;
  }
  double CalcMTR(const LorentzVector & j1, const LorentzVector & j2, const pat::MET & met){

    double temp = met.et()*(j1.Pt()+j2.Pt()) - met.px()*(j1.X()+j2.X()) - met.py()*(j1.Y()+j2.Y());
    temp /= 2.;

    temp = sqrt(temp);

    return temp;
  }
};


void RazorComputer::compute(const edm::Event & iEvent) const{
  //std::cout<<"getting into computation"<<std::endl;

  edm::Handle<std::vector<pat::Jet>> jetH;
  edm::Handle<std::vector<pat::MET>> metH;
  iEvent.getByToken( jetToken_, jetH);
  iEvent.getByToken( metToken_, metH);
  
  typedef reco::Candidate::LorentzVector LorentzVector;


  std::vector<LorentzVector> jets;
  jets.reserve( jetH.product()->size() );
  for (std::vector<pat::Jet>::const_iterator jetit = jetH.product()->begin();
       jetit!= jetH.product()->end(); ++ jetit){
    if (jetit->et() > pt_ && fabs(jetit->eta()) < eta_){
      jets.push_back( jetit->p4());
    }
  }

  reco::Candidate::LorentzVector HEM_1,HEM_2;

  HEM_1.SetPxPyPzE(0.0,0.0,0.0,0.0);
  HEM_2.SetPxPyPzE(0.0,0.0,0.0,0.0);
    
  if(jets.size() < 2) {
  }
  else{
    unsigned int N_comb = 1;
    for(unsigned int i = 0; i < jets.size(); i++){
      N_comb *= 2;
    }   
  
    double M_temp;

    double M_min = 9999999999.0;
    int j_count;
    
    for(unsigned int i = 1; i < N_comb-1; i++){
      LorentzVector j_temp1, j_temp2;
      int itemp = i;
      j_count = N_comb/2;
      int count = 0;
      while(j_count > 0){
	if(itemp/j_count == 1){
	  j_temp1 += jets[count];
	} else {
	  j_temp2 += jets[count];
	}
	itemp -= j_count*(itemp/j_count);
	j_count /= 2;
	count++;
      }

      M_temp = j_temp1.M2()+j_temp2.M2();

      if(M_temp < M_min){
	M_min = M_temp;
	HEM_1 = j_temp1;
	HEM_2 = j_temp2;
      }
      
    }
    
   
    if(HEM_2.Pt() > HEM_1.Pt()){
      LorentzVector temp = HEM_1;
      HEM_1 = HEM_2;
      HEM_2 = temp;
    }
  }

  double mrt=razor::CalcMTR( HEM_1, HEM_2, metH.product()->at(0));
  double mr=razor::CalcMR( HEM_1, HEM_2);
  

  assign("MRT",mrt);
  assign("MR",mr);
  double r=-1,r2=-1;
  if (mr!=0){
    r=mrt/mr;
    r2=r*r;
  }
  assign("R",r);
  assign("R2",r2);

  //std::cout<<"MR,R2 "<<mr<<" , "<<r2<<std::endl;
}
