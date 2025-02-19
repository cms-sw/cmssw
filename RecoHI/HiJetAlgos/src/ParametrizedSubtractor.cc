#include "RecoHI/HiJetAlgos/interface/ParametrizedSubtractor.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "string"
#include "iostream"
using namespace std;

void ParametrizedSubtractor::rescaleRMS(double s){
   for ( std::map<int, double>::iterator iter = esigma_.begin();
	 iter != esigma_.end(); ++iter ){
      iter->second = s*(iter->second);
   }
}


ParametrizedSubtractor::ParametrizedSubtractor(const edm::ParameterSet& iConfig) : 
   PileUpSubtractor(iConfig),
   dropZeroTowers_(iConfig.getUntrackedParameter<bool>("dropZeroTowers",true)),
   cbins_(0)
{
   centTag_ = iConfig.getUntrackedParameter<edm::InputTag>("centTag",edm::InputTag("hiCentrality","","RECO"));

   interpolate_ = iConfig.getParameter<bool>("interpolate");
   sumRecHits_ = iConfig.getParameter<bool>("sumRecHits");

   std::string ifname = "RecoHI/HiJetAlgos/data/PU_DATA.root";
   TFile* inf = new TFile(edm::FileInPath(ifname).fullPath().data());
   fPU = (TF1*)inf->Get("fPU");
   fMean = (TF1*)inf->Get("fMean");
   fRMS = (TF1*)inf->Get("fRMS");
   hC = (TH1D*)inf->Get("hC");

   for(int i = 0; i < 40; ++i){
      hEta.push_back((TH1D*)inf->Get(Form("hEta_%d",i)));
      hEtaMean.push_back((TH1D*)inf->Get(Form("hEtaMean_%d",i)));
      hEtaRMS.push_back((TH1D*)inf->Get(Form("hEtaRMS_%d",i)));
   }

}


void ParametrizedSubtractor::setupGeometryMap(edm::Event& iEvent,const edm::EventSetup& iSetup){
   LogDebug("PileUpSubtractor")<<"The subtractor setting up geometry...\n";

   //   if(!cbins_) getCentralityBinsFromDB(iSetup);

   edm::Handle<reco::Centrality> cent;
   iEvent.getByLabel(centTag_,cent);
   
   centrality_ = cent->EtHFhitSum();
   bin_ = 40-hC->FindBin(centrality_);
   if(bin_ > 39) bin_ = 39;
   if(bin_ < 0) bin_ = 0;

   if(!geo_) {
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
      emean_[i] = 0.;
      esigma_[i] = 0.;
      ntowersWithJets_[i] = 0;
   }
}


void ParametrizedSubtractor::calculatePedestal( vector<fastjet::PseudoJet> const & coll ){
   return;
}

void ParametrizedSubtractor::subtractPedestal(vector<fastjet::PseudoJet> & coll)
{
   if(0){
      return;
   }else{
      LogDebug("PileUpSubtractor")<<"The subtractor subtracting pedestals...\n";

   int it = -100;
   vector<fastjet::PseudoJet> newcoll;
        
   for (vector<fastjet::PseudoJet>::iterator input_object = coll.begin (),
	   fjInputsEnd = coll.end(); 
	input_object != fjInputsEnd; ++input_object) {
    
      reco::CandidatePtr const & itow =  (*inputs_)[ input_object->user_index() ];
    
      it = ieta( itow );
      iphi( itow );

      double Original_Et = itow->et();
      if(sumRecHits_){
	 Original_Et = getEt(itow);
      }

      double etnew = Original_Et - getPU(it,1,1);
      float mScale = etnew/input_object->Et(); 
      if(etnew < 0.) mScale = 0.;
    
      math::XYZTLorentzVectorD towP4(input_object->px()*mScale, input_object->py()*mScale,
				     input_object->pz()*mScale, input_object->e()*mScale);
    
      int index = input_object->user_index();
      input_object->reset ( towP4.px(),
			    towP4.py(),
			    towP4.pz(),
			    towP4.energy() );
      input_object->set_user_index(index);
      if(etnew > 0. && dropZeroTowers_) newcoll.push_back(*input_object);
   }
   if(dropZeroTowers_) coll = newcoll;

   }
}


void ParametrizedSubtractor::calculateOrphanInput(vector<fastjet::PseudoJet> & orphanInput) 
{
   orphanInput = *fjInputs_;
}



void ParametrizedSubtractor::offsetCorrectJets()
{

  LogDebug("PileUpSubtractor")<<"The subtractor correcting jets...\n";
  jetOffset_.clear();

  using namespace reco;
  
  (*fjInputs_) = fjOriginalInputs_;
  rescaleRMS(nSigmaPU_);
  subtractPedestal(*fjInputs_);

  if(0){
     const fastjet::JetDefinition& def = fjClusterSeq_->jet_def();
     if ( !doAreaFastjet_ && !doRhoFastjet_) {
	fastjet::ClusterSequence newseq( *fjInputs_, def );
	(*fjClusterSeq_) = newseq;
     } else {
	fastjet::ClusterSequenceArea newseq( *fjInputs_, def , *fjActiveArea_ );
	(*fjClusterSeq_) = newseq;
     }
     
     (*fjJets_) = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(jetPtMin_));
  }
  
  jetOffset_.reserve(fjJets_->size());
  
  vector<fastjet::PseudoJet>::iterator pseudojetTMP = fjJets_->begin (),
    jetsEnd = fjJets_->end();
  for (; pseudojetTMP != jetsEnd; ++pseudojetTMP) {
    
    int ijet = pseudojetTMP - fjJets_->begin();
    jetOffset_[ijet] = 0;
    
    std::vector<fastjet::PseudoJet> towers =
      sorted_by_pt(fjClusterSeq_->constituents(*pseudojetTMP));
    
    double newjetet = 0.;
    for(vector<fastjet::PseudoJet>::const_iterator ito = towers.begin(),
	  towEnd = towers.end();
	ito != towEnd;
	++ito)
      {
	const reco::CandidatePtr& originalTower = (*inputs_)[ito->user_index()];
        int it = ieta( originalTower );
        double Original_Et = originalTower->et();
      
	if(sumRecHits_){
	   Original_Et = getEt(originalTower);
	}

	double etnew = Original_Et - getPU(it,1,1);
	if(etnew < 0.) etnew = 0;
	newjetet = newjetet + etnew;
	jetOffset_[ijet] += Original_Et - etnew;
      }

    if(sumRecHits_){       
       double mScale = newjetet/pseudojetTMP->Et();
       int cshist = pseudojetTMP->cluster_hist_index();
       pseudojetTMP->reset(pseudojetTMP->px()*mScale, pseudojetTMP->py()*mScale,
			   pseudojetTMP->pz()*mScale, pseudojetTMP->e()*mScale);
       pseudojetTMP->set_cluster_hist_index(cshist);
    }

  }
}


double ParametrizedSubtractor::getEt(const reco::CandidatePtr & in) const {
   const CaloTower* ctc = dynamic_cast<const CaloTower*>(in.get());
   const GlobalPoint& pos=geo_->getPosition(ctc->id());
   double energy = ctc->emEnergy() + ctc->hadEnergy();

   if(0){
      energy = 0;
      const std::vector<DetId>& hitids = ctc->constituents();
      for(unsigned int i = 0; i< hitids.size(); ++i){

      }
   }

   double et = energy*sin(pos.theta());
   return et;
}

double ParametrizedSubtractor::getEta(const reco::CandidatePtr & in) const {
   const CaloTower* ctc = dynamic_cast<const CaloTower*>(in.get());
   const GlobalPoint& pos=geo_->getPosition(ctc->id());
   double eta = pos.eta();
   return eta;
}

double ParametrizedSubtractor::getMeanAtTower(const reco::CandidatePtr & in) const{
   int it = ieta(in);
   return getPU(it,1,0);
}

double ParametrizedSubtractor::getSigmaAtTower(const reco::CandidatePtr & in) const {
   int it = ieta(in);
   return getPU(it,0,1);
}

double ParametrizedSubtractor::getPileUpAtTower(const reco::CandidatePtr & in) const {
   int it = ieta(in);
   return getPU(it,1,1);
}

double ParametrizedSubtractor::getPU(int ieta,bool addMean, bool addSigma) const {   

  //double e = hEta[bin_]->GetBinContent(hEta[bin_]->FindBin(ieta));
  //double c = fPU->Eval(centrality_);

   double em = hEtaMean[bin_]->GetBinContent(hEtaMean[bin_]->FindBin(ieta));
   double cm = fMean->Eval(centrality_);

   double er = hEtaRMS[bin_]->GetBinContent(hEtaRMS[bin_]->FindBin(ieta));
   double cr = fRMS->Eval(centrality_);

   if(interpolate_){
      double n = 0;
      int hbin = 40-bin_;
      double centerweight =  (centrality_ - hC->GetBinCenter(hbin));
      double lowerweight = (centrality_ - hC->GetBinLowEdge(hbin));
      double upperweight = (centrality_ - hC->GetBinLowEdge(hbin+1));

      em *= lowerweight*upperweight;
      er *= lowerweight*upperweight;
      n += lowerweight*upperweight;

      if(bin_ > 0){
	 em += upperweight*centerweight*hEtaMean[bin_]->GetBinContent(hEtaMean[bin_-1]->FindBin(ieta));
	 er += upperweight*centerweight*hEtaRMS[bin_]->GetBinContent(hEtaRMS[bin_-1]->FindBin(ieta));
	 n += upperweight*centerweight;
      }

      if(bin_ < 39){
	 em += lowerweight*centerweight*hEtaMean[bin_]->GetBinContent(hEtaMean[bin_+1]->FindBin(ieta));
         er += lowerweight*centerweight*hEtaRMS[bin_]->GetBinContent(hEtaRMS[bin_+1]->FindBin(ieta));
	 n += lowerweight*centerweight;
      }
      em /= n;
      er /= n;
   }

   //   return e*c;
   return addMean*em*cm + addSigma*nSigmaPU_*er*cr;
}


