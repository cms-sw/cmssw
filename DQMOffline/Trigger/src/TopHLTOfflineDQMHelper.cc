#include "DQMOffline/Trigger/interface/TopHLTOfflineDQMHelper.h"
#include <iostream>
/*Originally from DQM/Physics package, written by Roger Wolf and Jeremy Andrea*/

using namespace std;

/// apply selection (w/o using the template class Object), override for jets
template <> 
bool SelectionStepHLT<reco::Jet>::select(const edm::Event& event, const edm::EventSetup& setup)
{
  // fetch input collection
  edm::Handle<edm::View<reco::Jet> > src; 
  if( !event.getByToken(src_, src) ) return false;

  // load btag collection if configured such
  // NOTE that the JetTagCollection needs an
  // edm::View to reco::Jets; we have to add
  // another Handle bjets for this purpose
  edm::Handle<edm::View<reco::Jet> > bjets; 
  edm::Handle<reco::JetTagCollection> btagger;
  edm::Handle<edm::View<reco::Vertex> > pvertex; 
  if(!btagLabel_.isUninitialized()){ 
    if( !event.getByToken(src_, bjets) ) return false;
    if( !event.getByToken(btagLabel_, btagger) ) return false;
    if( !event.getByToken(pvs_, pvertex) ) return false;
  }

  // load jetID value map if configured such 
  edm::Handle<reco::JetIDValueMap> jetID;
  if(jetIDSelect_){
    if( !event.getByToken(jetIDLabel_, jetID) ) return false;

  }

  // load jet corrector if configured such
  const JetCorrector* corrector=0;
  if(!jetCorrector_.empty()){
    // check whether a jet correcto is in the event setup or not
    if(setup.find( edm::eventsetup::EventSetupRecordKey::makeKey<JetCorrectionsRecord>() )){
      corrector = JetCorrector::getJetCorrector(jetCorrector_, setup);
    }
    else{
      edm::LogVerbatim( "TopHLTOfflineDQMHelper" ) 
        << "\n"
        << "------------------------------------------------------------------------------------- \n"
        << " No JetCorrectionsRecord available from EventSetup:                                   \n" 
        << "  - Jets will not be corrected.                                                       \n"
        << "  - If you want to change this add the following lines to your cfg file               \n"
        << "                                                                                      \n"
        << "  ## load jet corrections                                                             \n"
        << "  process.load(\"JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff\") \n"
        << "  process.prefer(\"ak5CaloL2L3\")                                                     \n"
        << "                                                                                      \n"
        << "------------------------------------------------------------------------------------- \n";
    }
  }
  // determine multiplicity of selected objects
  int n=0;
  for(typename edm::View<reco::Jet>::const_iterator obj=src->begin(); obj!=src->end(); ++obj){
    // check for chosen btag discriminator to be above the 
    // corresponding working point if configured such 
    unsigned int idx = obj-src->begin();
    if( btagLabel_.isUninitialized() ? true : (*btagger)[bjets->refAt(idx)]>btagWorkingPoint_ ){   
      bool passedJetID=true;
      // check jetID for calo jets
      if( jetIDSelect_ && dynamic_cast<const reco::CaloJet*>(src->refAt(idx).get())){
	passedJetID=(*jetIDSelect_)((*jetID)[src->refAt(idx)]);
      }
      if(passedJetID){
	// scale jet energy if configured such
        reco::Jet jet=*obj; jet.scaleEnergy(corrector ? corrector->correction(*obj) : 1.);
	if(select_(jet))++n;
      }
    }
  }
  bool accept=(min_>=0 ? n>=min_:true) && (max_>=0 ? n<=max_:true);
  return (min_<0 && max_<0) ? (n>0):accept;
}


CalculateHLT::CalculateHLT(int maxNJets, double wMass): 
  failed_(false), maxNJets_(maxNJets), wMass_(wMass), massWBoson_(-1.), massTopQuark_(-1.),tmassWBoson_(-1),tmassTopQuark_(-1),mlb_(-1)
{
}


double
CalculateHLT::massWBoson(const std::vector<reco::Jet>& jets)
{
  if(!failed_&& massWBoson_<0) operator()(jets); return massWBoson_;
}


double 
CalculateHLT::massTopQuark(const std::vector<reco::Jet>& jets) 
{ 
  if(!failed_&& massTopQuark_<0) operator()(jets); return massTopQuark_; 
}

/*
double 
Calculate::tmassWBoson(const T& mu, const reco::MET& met, const reco::Jet& b)
{
  if(!failed_&& tmassWBoson_<0) operator()(b,mu,met); return tmassWBoson_;
}


double
Calculate::masslb(const T& mu, const reco::MET& met, const reco::Jet& b)
{
  if(!failed_&& mlb_<0) operator()(b,mu,met); return mlb_;
}


double
Calculate::tmassTopQuark(const T& lepton, const reco::MET& met, const reco::Jet& b)
{
  if(!failed_&& tmassTopQuark_<0) operator()(b,lepton,met); return tmassTopQuark_;
}


void Calculate::operator()( const reco::Jet& bJet, const T& lepton, const reco::MET& met){
  reco::Particle::LorentzVector WT = lepton.p4() + met.p4();
  tmassWBoson_ = sqrt((WT.px()*WT.px()) + (WT.py()*WT.py()));	
  reco::Particle::LorentzVector topT = WT + bJet.p4();
  tmassTopQuark_ = sqrt((topT.px()*topT.px()) + (topT.py()*topT.py()));
  reco::Particle::LorentzVector lb = bJet.p4() + lepton.p4();  
  mlb_ = lb.mass();
}*/


double 
CalculateHLT::tmassWBoson(reco::RecoCandidate* mu, const reco::MET& met, const reco::Jet& b)
{
  if(!failed_&& tmassWBoson_<0) operator()(b,mu,met); return tmassWBoson_;
}


double
CalculateHLT::masslb(reco::RecoCandidate* mu, const reco::MET& met, const reco::Jet& b)
{
  if(!failed_&& mlb_<0) operator()(b,mu,met); return mlb_;
}


double
CalculateHLT::tmassTopQuark(reco::RecoCandidate* lepton, const reco::MET& met, const reco::Jet& b)
{
  if(!failed_&& tmassTopQuark_<0) operator()(b,lepton,met); return tmassTopQuark_;
}


void CalculateHLT::operator()( const reco::Jet& bJet, reco::RecoCandidate* lepton, const reco::MET& met){
  double metT = sqrt(pow(met.px(),2) + pow(met.py(),2));
  double lepT = sqrt(pow(lepton->px(),2) + pow(lepton->py(),2));
  double bT   = sqrt(pow(bJet.px(),2) + pow(bJet.py(),2));
  reco::Particle::LorentzVector WT = lepton->p4() + met.p4();
  //  cout<<"in calculate:\n\t"<<bJet.pt()<<"\t"<<lepton->pt()<<"\t"<<met.pt()<<endl;
  tmassWBoson_ = sqrt(pow(metT+lepT,2) - (WT.px()*WT.px()) - (WT.py()*WT.py()));	
  reco::Particle::LorentzVector topT = WT + bJet.p4();
  tmassTopQuark_ = sqrt(pow((metT+lepT+bT),2) - (topT.px()*topT.px()) - (topT.py()*topT.py()));
  reco::Particle::LorentzVector lb = bJet.p4() + lepton->p4();  
  mlb_ = lb.mass();
}


void
CalculateHLT::operator()(const std::vector<reco::Jet>& jets)
{
  
  if(maxNJets_<0) maxNJets_=jets.size();
  failed_= jets.size()<(unsigned int) maxNJets_;
  if( failed_){ return; }

  // associate those jets with maximum pt of the vectorial 
  // sum to the hadronic decay chain
  double maxPt=-1.;
  std::vector<int> maxPtIndices;
  maxPtIndices.push_back(-1);
  maxPtIndices.push_back(-1);
  maxPtIndices.push_back(-1);
  for(int idx=0; idx<maxNJets_; ++idx){
    for(int jdx=0; jdx<maxNJets_; ++jdx){ if(jdx<=idx) continue;
      for(int kdx=0; kdx<maxNJets_; ++kdx){ if(kdx==idx || kdx==jdx) continue;
	reco::Particle::LorentzVector sum = jets[idx].p4()+jets[jdx].p4()+jets[kdx].p4();
	if( maxPt<0. || maxPt<sum.pt() ){
	  maxPt=sum.pt();
	  maxPtIndices.clear();
	  maxPtIndices.push_back(idx);
	  maxPtIndices.push_back(jdx);
	  maxPtIndices.push_back(kdx);
	}
      }
    }
  }
  massTopQuark_= (jets[maxPtIndices[0]].p4()+
		  jets[maxPtIndices[1]].p4()+
		  jets[maxPtIndices[2]].p4()).mass();

  // associate those jets that get closest to the W mass
  // with their invariant mass to the W boson
  double wDist =-1.;
  std::vector<int> wMassIndices;
  wMassIndices.push_back(-1);
  wMassIndices.push_back(-1);
  for(unsigned idx=0; idx<maxPtIndices.size(); ++idx){  
    for(unsigned jdx=0; jdx<maxPtIndices.size(); ++jdx){  
      if( jdx==idx || maxPtIndices[idx]>maxPtIndices[jdx] ) continue;
	reco::Particle::LorentzVector sum = jets[maxPtIndices[idx]].p4()+jets[maxPtIndices[jdx]].p4();
	if( wDist<0. || wDist>fabs(sum.mass()-wMass_) ){
	  wDist=fabs(sum.mass()-wMass_);
	  wMassIndices.clear();
	  wMassIndices.push_back(maxPtIndices[idx]);
	  wMassIndices.push_back(maxPtIndices[jdx]);
	}
    }
  }
  massWBoson_= (jets[wMassIndices[0]].p4()+
		jets[wMassIndices[1]].p4()).mass();
}



