#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"

//
// class declaration
//
class DistortedMuonProducerFromDB : public edm::EDProducer {
   public:
      explicit DistortedMuonProducerFromDB(const edm::ParameterSet&);
      ~DistortedMuonProducerFromDB();

   private:
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      edm::EDGetTokenT<edm::View<reco::Muon> > muonToken_;

      std::string dbScaleLabel_;
      std::string dbDataResolutionLabel_;
      std::string dbMCResolutionLabel_;

      std::auto_ptr<MomentumScaleCorrector> momCorrector_;
      std::auto_ptr<ResolutionFunction> momResolutionData_;
      std::auto_ptr<ResolutionFunction> momResolutionMC_;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include <CLHEP/Random/RandGauss.h>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/RecoMuonObjects/interface/MuScleFitDBobject.h"
#include "CondFormats/DataRecord/interface/MuScleFitDBobjectRcd.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/BaseFunction.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"

/////////////////////////////////////////////////////////////////////////////////////
DistortedMuonProducerFromDB::DistortedMuonProducerFromDB(const edm::ParameterSet& pset) {

  // What is being produced
      produces<std::vector<reco::Muon> >();

  // Input products
      muonToken_ = consumes<edm::View<reco::Muon> >(pset.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons")));
      dbScaleLabel_ = pset.getUntrackedParameter<std::string> ("DBScaleLabel", "scale");
      dbDataResolutionLabel_ = pset.getUntrackedParameter<std::string> ("DBDataResolutionLabel", "datareso");
      dbMCResolutionLabel_ = pset.getUntrackedParameter<std::string> ("DBMCResolutionLabel", "mcreso");

}

/////////////////////////////////////////////////////////////////////////////////////
DistortedMuonProducerFromDB::~DistortedMuonProducerFromDB(){
}

/////////////////////////////////////////////////////////////////////////////////////
void DistortedMuonProducerFromDB::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
      edm::ESHandle<MuScleFitDBobject> dbObject1;
      iSetup.get<MuScleFitDBobjectRcd>().get(dbScaleLabel_,dbObject1);
      momCorrector_.reset(new MomentumScaleCorrector(dbObject1.product()));

      LogTrace("") << ">>> Using database for momentum scale corrections !!";

      edm::ESHandle<MuScleFitDBobject> dbObject2;
      iSetup.get<MuScleFitDBobjectRcd>().get(dbDataResolutionLabel_, dbObject2);
      momResolutionData_.reset(new ResolutionFunction(dbObject2.product()));

      edm::ESHandle<MuScleFitDBobject> dbObject3;
      iSetup.get<MuScleFitDBobjectRcd>().get(dbMCResolutionLabel_, dbObject3);
      momResolutionMC_.reset(new ResolutionFunction(dbObject3.product()));

      LogTrace("") << ">>> Using database for momentum resolution corrections !!";
}

/////////////////////////////////////////////////////////////////////////////////////
void DistortedMuonProducerFromDB::endJob(){
}

/////////////////////////////////////////////////////////////////////////////////////
void DistortedMuonProducerFromDB::produce(edm::Event& ev, const edm::EventSetup& iSetup) {

      if (ev.isRealData()) return;

      // Muon collection
      edm::Handle<edm::View<reco::Muon> > muonCollection;
      if (!ev.getByToken(muonToken_, muonCollection)) {
            edm::LogError("") << ">>> Muon collection does not exist !!!";
            return;
      }
      unsigned int muonCollectionSize = muonCollection->size();

      std::auto_ptr<reco::MuonCollection> newmuons (new reco::MuonCollection);

      for (unsigned int i=0; i<muonCollectionSize; i++) {
            edm::RefToBase<reco::Muon> mu = muonCollection->refAt(i);

            // Set shift
            double shift = (*momCorrector_)(*mu) - mu->pt();
            LogTrace("") << "\tmomentumScaleShift= " << shift << " [GeV]";

            // Set resolutions
            double sigma = pow(momResolutionData_->sigmaPt(*mu),2) -
                              pow(momResolutionMC_->sigmaPt(*mu),2);
            if (sigma>0.) sigma = sqrt(sigma); else sigma = 0.;
            LogTrace("") << "\tPt additional smearing= " << sigma << " [GeV]";

            // Gaussian Random number for smearing
            double rndg = CLHEP::RandGauss::shoot();

            // New muon
            double ptmu = mu->pt();
            ptmu += shift + sigma*rndg;
            reco::Muon* newmu = mu->clone();
            newmu->setP4 (
                  reco::Particle::PolarLorentzVector (
                        ptmu, mu->eta(), mu->phi(), mu->mass()
                  )
            );
            newmuons->push_back(*newmu);

      }

      ev.put(newmuons);
}

DEFINE_FWK_MODULE(DistortedMuonProducerFromDB);
