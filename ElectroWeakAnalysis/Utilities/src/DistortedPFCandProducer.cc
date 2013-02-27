#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"

//
// class declaration
//
class DistortedPFCandProducer : public edm::EDProducer {
   public:
      explicit DistortedPFCandProducer(const edm::ParameterSet&);
      ~DistortedPFCandProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

      edm::InputTag muonTag_;
      edm::InputTag pfTag_;
      edm::InputTag genMatchMapTag_;
      std::vector<double> etaBinEdges_;

      std::vector<double> shiftOnOneOverPt_; // in [1/GeV]
      std::vector<double> relativeShiftOnPt_;
      std::vector<double> uncertaintyOnOneOverPt_; // in [1/GeV]
      std::vector<double> relativeUncertaintyOnPt_;

      std::vector<double> efficiencyRatioOverMC_;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include <CLHEP/Random/RandFlat.h>
#include <CLHEP/Random/RandGauss.h>

/////////////////////////////////////////////////////////////////////////////////////
DistortedPFCandProducer::DistortedPFCandProducer(const edm::ParameterSet& pset) {

  // What is being produced
      produces<std::vector<reco::PFCandidate> >();

  // Input products
      muonTag_ = pset.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"));
      pfTag_ = pset.getUntrackedParameter<edm::InputTag> ("PFTag", edm::InputTag("particleFlow"));
      genMatchMapTag_ = pset.getUntrackedParameter<edm::InputTag> ("GenMatchMapTag", edm::InputTag("genMatchMap"));

  // Eta edges
      std::vector<double> defEtaEdges;
      defEtaEdges.push_back(-999999.);
      defEtaEdges.push_back(999999.);
      etaBinEdges_ = pset.getUntrackedParameter<std::vector<double> > ("EtaBinEdges",defEtaEdges);
      unsigned int ninputs_expected = etaBinEdges_.size()-1;

  // Distortions in muon momentum
      std::vector<double> defDistortion;
      defDistortion.push_back(0.);

      shiftOnOneOverPt_ = pset.getUntrackedParameter<std::vector<double> > ("ShiftOnOneOverPt",defDistortion); // in [1/GeV]
      if (shiftOnOneOverPt_.size()==1 && ninputs_expected>1) {
            for (unsigned int i=1; i<ninputs_expected; i++){ shiftOnOneOverPt_.push_back(shiftOnOneOverPt_[0]);}
      }

      relativeShiftOnPt_ = pset.getUntrackedParameter<std::vector<double> > ("RelativeShiftOnPt",defDistortion);
      if (relativeShiftOnPt_.size()==1 && ninputs_expected>1) {
            for (unsigned int i=1; i<ninputs_expected; i++){ relativeShiftOnPt_.push_back(relativeShiftOnPt_[0]);}
      }

      uncertaintyOnOneOverPt_ = pset.getUntrackedParameter<std::vector<double> > ("UncertaintyOnOneOverPt",defDistortion); // in [1/GeV]
      if (uncertaintyOnOneOverPt_.size()==1 && ninputs_expected>1) {
            for (unsigned int i=1; i<ninputs_expected; i++){ uncertaintyOnOneOverPt_.push_back(uncertaintyOnOneOverPt_[0]);}
      }

      relativeUncertaintyOnPt_ = pset.getUntrackedParameter<std::vector<double> > ("RelativeUncertaintyOnPt",defDistortion);
      if (relativeUncertaintyOnPt_.size()==1 && ninputs_expected>1) {
            for (unsigned int i=1; i<ninputs_expected; i++){ relativeUncertaintyOnPt_.push_back(relativeUncertaintyOnPt_[0]);}
      }

  // Data/MC efficiency ratios
      std::vector<double> defEfficiencyRatio;
      defEfficiencyRatio.push_back(1.);
      efficiencyRatioOverMC_ = pset.getUntrackedParameter<std::vector<double> > ("EfficiencyRatioOverMC",defEfficiencyRatio);
      if (efficiencyRatioOverMC_.size()==1 && ninputs_expected>1) {
            for (unsigned int i=1; i<ninputs_expected; i++){ efficiencyRatioOverMC_.push_back(efficiencyRatioOverMC_[0]);}
      }

  // Send a warning if there are inconsistencies in vector sizes !!
      bool effWrong = efficiencyRatioOverMC_.size()!=ninputs_expected;
      bool momWrong =    shiftOnOneOverPt_.size()!=ninputs_expected 
                      || relativeShiftOnPt_.size()!=ninputs_expected 
                      || uncertaintyOnOneOverPt_.size()!=ninputs_expected 
                      || relativeUncertaintyOnPt_.size()!=ninputs_expected;
      if ( effWrong and momWrong) {
           edm::LogError("") << "WARNING: DistortedPFCandProducer : Size of some parameters do not match the EtaBinEdges vector!!";
      }

} 

/////////////////////////////////////////////////////////////////////////////////////
DistortedPFCandProducer::~DistortedPFCandProducer(){
}

/////////////////////////////////////////////////////////////////////////////////////
void DistortedPFCandProducer::beginJob() {
}

/////////////////////////////////////////////////////////////////////////////////////
void DistortedPFCandProducer::endJob(){
}

/////////////////////////////////////////////////////////////////////////////////////
void DistortedPFCandProducer::produce(edm::Event& ev, const edm::EventSetup& iSetup) {

      if (ev.isRealData()) return;

      // Muon collection
      edm::Handle<edm::View<reco::Muon> > muonCollection;
      if (!ev.getByLabel(muonTag_, muonCollection)) {
            edm::LogError("") << ">>> Muon collection does not exist !!!";
            return;
      }

      edm::Handle<reco::GenParticleMatch> genMatchMap;
      if (!ev.getByLabel(genMatchMapTag_, genMatchMap)) {
            edm::LogError("") << ">>> Muon-GenParticle match map does not exist !!!";
            return;
      }
  

      // Get PFCandidate collection
      edm::Handle<edm::View<reco::PFCandidate> > pfCollection;
      if (!ev.getByLabel(pfTag_, pfCollection)) {
            edm::LogError("") << ">>> PFCandidate collection does not exist !!!";
            return;
      }

      unsigned int muonCollectionSize = muonCollection->size();
      unsigned int pfCollectionSize = pfCollection->size();
      
      if (pfCollectionSize<1) return;


      // Ask for PfMuon consistency
      bool pfMuonFound = false;



      std::auto_ptr<reco::PFCandidateCollection> newmuons (new reco::PFCandidateCollection);


      // Loop on all PF candidates
      for (unsigned int j=0; j<pfCollectionSize; j++) {
        edm::RefToBase<reco::PFCandidate> pf = pfCollection->refAt(j);


        // New PF muon
        double ptmu = pf->pt();


        for (unsigned int i=0; i<muonCollectionSize; i++) {
          edm::RefToBase<reco::Muon> mu = muonCollection->refAt(i);


          // Check the muon is in the PF collection
          if (pf->particleId()==reco::PFCandidate::mu) {
            reco::MuonRef muref = pf->muonRef();
            if (muref.isNonnull()) {
              if (muref.key()==mu.key()) {
       	           if ( mu->isStandAloneMuon() && ptmu == muref->standAloneMuon()->pt()  && (
                         ( !mu->isGlobalMuon()  || ( mu->isGlobalMuon() &&  ptmu != muref->combinedMuon()->pt() ) ) &&
                         ( !mu->isTrackerMuon() || ( mu->isTrackerMuon() && ptmu != mu->track()->pt() ) ) )
                       ) {
                        pfMuonFound = false; 
                    }
	            else if ( !mu->isTrackerMuon() ){
                        pfMuonFound = false;
                    }
                    else{
         		pfMuonFound = true;}
              }
              else {pfMuonFound = false; }

            }
          }

          // do nothing if StandAlone muon
          //const reco::Track& track = *trackRef;
          
          if ( !pfMuonFound) continue;

	  double ptgen = pf->pt();
	  double etagen = pf->eta();


	  reco::GenParticleRef gen = (*genMatchMap)[mu];
	  if( !gen.isNull()) {
	    ptgen = gen->pt();
	    etagen = gen->eta();
	    LogTrace("") << ">>> Muon-GenParticle match found; ptmu= " << pf->pt() << ", ptgen= " << ptgen;
	  } else {
	    LogTrace("") << ">>> MUON-GENPARTICLE MATCH NOT FOUND!!!";
	  }
     

	  // Initialize parameters
	  double effRatio = 0.;
	  double shift1 = 0.;
	  double shift2 = 0.;
	  double sigma1 = 0.;
	  double sigma2 = 0.;

	  // Find out which eta bin should be used
	  unsigned int nbins = etaBinEdges_.size()-1;
	  unsigned int etaBin = nbins;
	  if (etagen>etaBinEdges_[0] && etagen<etaBinEdges_[nbins]) {
	    for (unsigned int j=1; j<=nbins; ++j) {
	      if (etagen>etaBinEdges_[j]) continue;
	      etaBin = j-1;
	      break;
	    }
	  }
	  if (etaBin<nbins) {
	    LogTrace("") << ">>> etaBin: " << etaBin << ", for etagen =" << etagen;
	  } else {
	    // Muon is rejected if outside the considered eta range
	    LogTrace("") << ">>> Muon outside eta range: reject it; etagen = " << etagen;
	    pfMuonFound = false;
	    continue;
	  }
	  
          if (!pfMuonFound) continue;

	  // Set shifts
	  shift1 = shiftOnOneOverPt_[etaBin];
	  shift2 = relativeShiftOnPt_[etaBin];
	  LogTrace("") << "\tshiftOnOneOverPt= " << shift1*100 << " [%]"; 
	  LogTrace("") << "\trelativeShiftOnPt= " << shift2*100 << " [%]"; 
	  
	  // Set resolutions
	  sigma1 = uncertaintyOnOneOverPt_[etaBin];
	  sigma2 = relativeUncertaintyOnPt_[etaBin];
	  LogTrace("") << "\tuncertaintyOnOneOverPt= " << sigma1 << " [1/GeV]"; 
	  LogTrace("") << "\trelativeUncertaintyOnPt= " << sigma2*100 << " [%]"; 
	  
	  // Set efficiency ratio
	  effRatio = efficiencyRatioOverMC_[etaBin];
	  LogTrace("") << "\tefficiencyRatioOverMC= " << effRatio;
	  
	  // Reject muons according to efficiency ratio
	  double rndf = CLHEP::RandFlat::shoot();
	  if (rndf>effRatio) continue;

	  // Gaussian Random numbers for smearing
	  double rndg1 = CLHEP::RandGauss::shoot();
	  double rndg2 = CLHEP::RandGauss::shoot();
            
            
	  // change here the pt of the candidate, if it is a muon 

	  ptmu += ptgen * ( shift1*ptgen + shift2 + sigma1*rndg1*ptgen + sigma2*rndg2);
	  pfMuonFound = false ;

	}

	reco::PFCandidate* newmu = pf->clone();
	newmu->setP4 (
		      reco::Particle::PolarLorentzVector (
							  ptmu, pf->eta(), pf->phi(), pf->mass()
							  )
		      );
	
	newmuons->push_back(*newmu);
      }

 
	ev.put(newmuons);


}

DEFINE_FWK_MODULE(DistortedPFCandProducer);
