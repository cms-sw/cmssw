
#include "RecoJets/JetProducers/interface/PileUpSubtractor.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <map>
using namespace std;

PileUpSubtractor::PileUpSubtractor(const edm::ParameterSet& iConfig, edm::ConsumesCollector && iC) :
  reRunAlgo_ (iConfig.getUntrackedParameter<bool>("reRunAlgo",false)),
  doAreaFastjet_ (iConfig.getParameter<bool>         ("doAreaFastjet")),
  doRhoFastjet_  (iConfig.getParameter<bool>         ("doRhoFastjet")),
  jetPtMin_(iConfig.getParameter<double>       ("jetPtMin")),
  nSigmaPU_(iConfig.getParameter<double>("nSigmaPU")),
  radiusPU_(iConfig.getParameter<double>("radiusPU")),
  geo_(0)
{
  if (iConfig.exists("puPtMin"))
    puPtMin_=iConfig.getParameter<double>       ("puPtMin");
  else{
    puPtMin_=10;
    edm::LogWarning("MisConfiguration")<<"the parameter puPtMin is now necessary for PU substraction. setting it to "<<puPtMin_;
  }
   if ( doAreaFastjet_ || doRhoFastjet_ ) {
      // default Ghost_EtaMax should be 5
      double ghostEtaMax = iConfig.getParameter<double>("Ghost_EtaMax");
      // default Active_Area_Repeats 1
      int    activeAreaRepeats = iConfig.getParameter<int> ("Active_Area_Repeats");
      // default GhostArea 0.01
      double ghostArea = iConfig.getParameter<double> ("GhostArea");
      fjActiveArea_ =  ActiveAreaSpecPtr(new fastjet::ActiveAreaSpec(ghostEtaMax,
       								     activeAreaRepeats,
       								     ghostArea));
      fjRangeDef_ = RangeDefPtr( new fastjet::RangeDefinition(ghostEtaMax) );
   } 
}

void PileUpSubtractor::reset(std::vector<edm::Ptr<reco::Candidate> >& input,
			     std::vector<fastjet::PseudoJet>& towers,
			     std::vector<fastjet::PseudoJet>& output){
  
  inputs_ = &input;
  fjInputs_ = &towers;
  fjJets_ = &output;
  fjOriginalInputs_ = (*fjInputs_);
  for (unsigned int i = 0; i < fjInputs_->size(); ++i){
    fjOriginalInputs_[i].set_user_index((*fjInputs_)[i].user_index());
  }
  
}

void PileUpSubtractor::setDefinition(JetDefPtr const & jetDef){
  fjJetDefinition_ = JetDefPtr( new fastjet::JetDefinition( *jetDef ) );
}

void PileUpSubtractor::setupGeometryMap(edm::Event& iEvent,const edm::EventSetup& iSetup)
{

  LogDebug("PileUpSubtractor")<<"The subtractor setting up geometry...\n";

  if(geo_ == 0) {
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    geo_ = pG.product();
    std::vector<DetId> alldid =  geo_->getValidDetIds();

    int ietaold = -10000;
    ietamax_ = -10000;
    ietamin_ = 10000;
    for(std::vector<DetId>::const_iterator did=alldid.begin(); did != alldid.end(); did++){
      if( (*did).det() == DetId::Hcal ){
        HcalDetId hid = HcalDetId(*did);
        if( (hid).depth() == 1 ) {
	  allgeomid_.push_back(*did);

          if((hid).ieta() != ietaold){
            ietaold = (hid).ieta();
            geomtowers_[(hid).ieta()] = 1;
            if((hid).ieta() > ietamax_) ietamax_ = (hid).ieta();
            if((hid).ieta() < ietamin_) ietamin_ = (hid).ieta();
          }
          else{
            geomtowers_[(hid).ieta()]++;
          }
        }
      }
    }
  }

  for (int i = ietamin_; i<ietamax_+1; i++) {
    ntowersWithJets_[i] = 0;
  }
}

//
// Calculate mean E and sigma from jet collection "coll".  
//
void PileUpSubtractor::calculatePedestal( vector<fastjet::PseudoJet> const & coll )
{
  LogDebug("PileUpSubtractor")<<"The subtractor calculating pedestals...\n";
  map<int,double> emean2;
  map<int,int> ntowers;
    
  int ietaold = -10000;
  int ieta0 = -100;
   
  // Initial values for emean_, emean2, esigma_, ntowers

  for(int i = ietamin_; i < ietamax_+1; i++)
    {
      emean_[i] = 0.;
      emean2[i] = 0.;
      esigma_[i] = 0.;
      ntowers[i] = 0;
    }
    
  for (vector<fastjet::PseudoJet>::const_iterator input_object = coll.begin (),
	 fjInputsEnd = coll.end();  
       input_object != fjInputsEnd; ++input_object) {
     const reco::CandidatePtr & originalTower=(*inputs_)[ input_object->user_index()];
    ieta0 = ieta( originalTower );
    double Original_Et = originalTower->et();
  if( ieta0-ietaold != 0 )
      {
        emean_[ieta0] = emean_[ieta0]+Original_Et;
        emean2[ieta0] = emean2[ieta0]+Original_Et*Original_Et;
        ntowers[ieta0] = 1;
        ietaold = ieta0;
      }
  else
      {
        emean_[ieta0] = emean_[ieta0]+Original_Et;
        emean2[ieta0] = emean2[ieta0]+Original_Et*Original_Et;
        ntowers[ieta0]++;
      }
  }

  for(map<int,int>::const_iterator gt = geomtowers_.begin(); gt != geomtowers_.end(); gt++)    
    {
      int it = (*gt).first;
       
      double e1 = (*(emean_.find(it))).second;
      double e2 = (*emean2.find(it)).second;
      int nt = (*gt).second - (*(ntowersWithJets_.find(it))).second;

      LogDebug("PileUpSubtractor")<<" ieta : "<<it<<" number of towers : "<<nt<<" e1 : "<<e1<<" e2 : "<<e2<<"\n";
      if(nt > 0) {
	emean_[it] = e1/nt;
	double eee = e2/nt - e1*e1/(nt*nt);    
	if(eee<0.) eee = 0.;
	esigma_[it] = nSigmaPU_*sqrt(eee);
      }
      else
	{
          emean_[it] = 0.;
          esigma_[it] = 0.;
	}
      LogDebug("PileUpSubtractor")<<" ieta : "<<it<<" Pedestals : "<<emean_[it]<<"  "<<esigma_[it]<<"\n";
    }
}


//
// Subtract mean and sigma from fjInputs_
//    
void PileUpSubtractor::subtractPedestal(vector<fastjet::PseudoJet> & coll)
{

  LogDebug("PileUpSubtractor")<<"The subtractor subtracting pedestals...\n";

  int it = -100;
  for (vector<fastjet::PseudoJet>::iterator input_object = coll.begin (),
	 fjInputsEnd = coll.end(); 
       input_object != fjInputsEnd; ++input_object) {
    
     reco::CandidatePtr const & itow =  (*inputs_)[ input_object->user_index() ];
    
    it = ieta( itow );
    
    double etnew = itow->et() - (*(emean_.find(it))).second - (*(esigma_.find(it))).second;
    float mScale = etnew/input_object->Et(); 
    if(etnew < 0.) mScale = 0.;
    
    math::XYZTLorentzVectorD towP4(input_object->px()*mScale, input_object->py()*mScale,
				   input_object->pz()*mScale, input_object->e()*mScale);
    
    int index = input_object->user_index();
    input_object->reset_momentum ( towP4.px(),
				   towP4.py(),
				   towP4.pz(),
				   towP4.energy() );
    input_object->set_user_index(index);
  }
}

void PileUpSubtractor::calculateOrphanInput(vector<fastjet::PseudoJet> & orphanInput) 
{

  LogDebug("PileUpSubtractor")<<"The subtractor calculating orphan input...\n";

  (*fjInputs_) = fjOriginalInputs_;

  vector<int> jettowers; // vector of towers indexed by "user_index"
  vector<pair<int,int> >  excludedTowers; // vector of excluded ieta, iphi values

  vector <fastjet::PseudoJet>::iterator pseudojetTMP = fjJets_->begin (),
    fjJetsEnd = fjJets_->end();
  for (; pseudojetTMP != fjJetsEnd ; ++pseudojetTMP) {
    if(pseudojetTMP->perp() < puPtMin_) continue;

    // find towers within radiusPU_ of this jet
    for(vector<HcalDetId>::const_iterator im = allgeomid_.begin(); im != allgeomid_.end(); im++)
      {
	double dr = reco::deltaR(geo_->getPosition((DetId)(*im)),(*pseudojetTMP));
	vector<pair<int,int> >::const_iterator exclude = find(excludedTowers.begin(),excludedTowers.end(),pair<int,int>(im->ieta(),im->iphi()));
	if( dr < radiusPU_ && exclude == excludedTowers.end()) {
	  ntowersWithJets_[(*im).ieta()]++;     
	  excludedTowers.push_back(pair<int,int>(im->ieta(),im->iphi()));
	}
      }
    vector<fastjet::PseudoJet>::const_iterator it = fjInputs_->begin(),
      fjInputsEnd = fjInputs_->end();
      
    for (; it != fjInputsEnd; ++it ) {
      int index = it->user_index();
      int ie = ieta((*inputs_)[index]);
      int ip = iphi((*inputs_)[index]);
      vector<pair<int,int> >::const_iterator exclude = find(excludedTowers.begin(),excludedTowers.end(),pair<int,int>(ie,ip));
      if(exclude != excludedTowers.end()) {
	jettowers.push_back(index);
      } //dr < radiusPU_
    } // initial input collection  
  } // pseudojets
  
  //
  // Create a new collections from the towers not included in jets 
  //
  for(vector<fastjet::PseudoJet>::const_iterator it = fjInputs_->begin(),
	fjInputsEnd = fjInputs_->end(); it != fjInputsEnd; ++it ) {
    int index = it->user_index();
    vector<int>::const_iterator itjet = find(jettowers.begin(),jettowers.end(),index);
    if( itjet == jettowers.end() ){
      const reco::CandidatePtr& originalTower = (*inputs_)[index];
      fastjet::PseudoJet orphan(originalTower->px(),originalTower->py(),originalTower->pz(),originalTower->energy());
      orphan.set_user_index(index);

      orphanInput.push_back(orphan); 
    }
  }
}


void PileUpSubtractor::offsetCorrectJets() 
{
  LogDebug("PileUpSubtractor")<<"The subtractor correcting jets...\n";
  jetOffset_.clear();
  using namespace reco;
  
  //    
  // Reestimate energy of jet (energy of jet with initial map)
  //
  jetOffset_.reserve(fjJets_->size());
  vector<fastjet::PseudoJet>::iterator pseudojetTMP = fjJets_->begin (),
    jetsEnd = fjJets_->end();
  for (; pseudojetTMP != jetsEnd; ++pseudojetTMP) {
    int ijet = pseudojetTMP - fjJets_->begin();
    jetOffset_[ijet] = 0;
    
    std::vector<fastjet::PseudoJet> towers =
      fastjet::sorted_by_pt( pseudojetTMP->constituents() );
    double newjetet = 0.;	
    for(vector<fastjet::PseudoJet>::const_iterator ito = towers.begin(),
	  towEnd = towers.end(); 
	ito != towEnd; 
	++ito)
      {
	const reco::CandidatePtr& originalTower = (*inputs_)[ito->user_index()];
	int it = ieta( originalTower );
	double Original_Et = originalTower->et();
	double etnew = Original_Et - (*emean_.find(it)).second - (*esigma_.find(it)).second; 
	if(etnew < 0.) etnew = 0;
	newjetet = newjetet + etnew;
	jetOffset_[ijet] += Original_Et - etnew;
      }
    double mScale = newjetet/pseudojetTMP->Et();
    LogDebug("PileUpSubtractor")<<"pseudojetTMP->Et() : "<<pseudojetTMP->Et()<<"\n";
    LogDebug("PileUpSubtractor")<<"newjetet : "<<newjetet<<"\n";
    LogDebug("PileUpSubtractor")<<"jetOffset_[ijet] : "<<jetOffset_[ijet]<<"\n";
    LogDebug("PileUpSubtractor")<<"pseudojetTMP->Et() - jetOffset_[ijet] : "<<pseudojetTMP->Et() - jetOffset_[ijet]<<"\n";
    LogDebug("PileUpSubtractor")<<"Scale is : "<<mScale<<"\n";
    int cshist = pseudojetTMP->cluster_hist_index();
    pseudojetTMP->reset_momentum(pseudojetTMP->px()*mScale, pseudojetTMP->py()*mScale,
				 pseudojetTMP->pz()*mScale, pseudojetTMP->e()*mScale);
    pseudojetTMP->set_cluster_hist_index(cshist);
    
  }
}

double PileUpSubtractor::getCone(double cone, double eta, double phi, double& et, double& pu){
  pu = 0;
  
  for(vector<HcalDetId>::const_iterator im = allgeomid_.begin(); im != allgeomid_.end(); im++){
     if( im->depth() != 1 ) continue;    
    const GlobalPoint& point = geo_->getPosition((DetId)(*im));
    double dr = reco::deltaR(point.eta(),point.phi(),eta,phi);      
    if( dr < cone){
      pu += (*emean_.find(im->ieta())).second+(*esigma_.find(im->ieta())).second;
    }
  }
  
  return pu;
}

double PileUpSubtractor::getMeanAtTower(const reco::CandidatePtr & in) const{
  int it = ieta(in);
  return (*emean_.find(it)).second;
}

double PileUpSubtractor::getSigmaAtTower(const reco::CandidatePtr & in) const {
   int it = ieta(in);
   return (*esigma_.find(it)).second;
}

double PileUpSubtractor::getPileUpAtTower(const reco::CandidatePtr & in) const {
  int it = ieta(in);
  return (*emean_.find(it)).second + (*esigma_.find(it)).second;
}

int PileUpSubtractor::getN(const reco::CandidatePtr & in) const {
   int it = ieta(in);
   
   int n = (*(geomtowers_.find(it))).second - (*(ntowersWithJets_.find(it))).second;
   return n;

}

int PileUpSubtractor::getNwithJets(const reco::CandidatePtr & in) const {
   int it = ieta(in);
   int n = (*(ntowersWithJets_.find(it))).second;
   return n;

}


int PileUpSubtractor::ieta(const reco::CandidatePtr & in) const {
  int it = 0;
  const CaloTower* ctc = dynamic_cast<const CaloTower*>(in.get());
  if(ctc){
    it = ctc->id().ieta();
  } else
    {
      throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of CaloTower type";
    }
  return it;
}

int PileUpSubtractor::iphi(const reco::CandidatePtr & in) const {
  int it = 0;
  const CaloTower* ctc = dynamic_cast<const CaloTower*>(in.get());
  if(ctc){
    it = ctc->id().iphi();
  } else
    {
      throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of CaloTower type";
    }
  return it;
}


#include "FWCore/PluginManager/interface/PluginFactory.h"
EDM_REGISTER_PLUGINFACTORY(PileUpSubtractorFactory,"PileUpSubtractorFactory");



