/* class PFTauTransverseImpactParamters
 * EDProducer of the 
 * authors: Ian M. Nugent
 */


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterFwd.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "TMath.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFTauTransverseImpactParameters : public EDProducer {
 public:
  enum Alg{useInputPV=0, useFont};
  enum CMSSWPerigee{aCurv=0,aTheta,aPhi,aTip,aLip};
  explicit PFTauTransverseImpactParameters(const edm::ParameterSet& iConfig);
  ~PFTauTransverseImpactParameters();
  virtual void produce(edm::Event&,const edm::EventSetup&);
 private:
  edm::InputTag PFTauTag_;
  edm::InputTag PFTauPVATag_;
  edm::InputTag PFTauSVATag_;
  bool useFullCalculation_;
};

PFTauTransverseImpactParameters::PFTauTransverseImpactParameters(const edm::ParameterSet& iConfig):
  PFTauTag_(iConfig.getParameter<edm::InputTag>("PFTauTag")),
  PFTauPVATag_(iConfig.getParameter<edm::InputTag>("PFTauPVATag")),
  PFTauSVATag_(iConfig.getParameter<edm::InputTag>("PFTauSVATag")),
  useFullCalculation_(iConfig.getParameter<bool>("useFullCalculation"))
{
  produces<edm::AssociationVector<PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameter> > >(); 
}

PFTauTransverseImpactParameters::~PFTauTransverseImpactParameters(){

}

void PFTauTransverseImpactParameters::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){
  // Obtain Collections
  edm::ESHandle<TransientTrackBuilder> transTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",transTrackBuilder);
  
  edm::Handle<std::vector<reco::PFTau> > Tau;
  iEvent.getByLabel(PFTauTag_,Tau);

  edm::Handle<edm::AssociationVector<PFTauRefProd, std::vector<reco::VertexRef> > > PFTauPVA;
  iEvent.getByLabel(PFTauPVATag_,PFTauPVA);

  edm::Handle<edm::AssociationVector<PFTauRefProd,std::vector<std::vector<reco::VertexRef> > > > PFTauSVA;
  iEvent.getByLabel(PFTauSVATag_,PFTauSVA);

  // Set Association Map
  auto_ptr<edm::AssociationVector<PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameter> > > AVPFTauTIP(new edm::AssociationVector<PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameter> >(PFTauRefProd(Tau)));

  // For each Tau Run Algorithim
  if(Tau.isValid()) {
    for(reco::PFTauCollection::size_type iPFTau = 0; iPFTau < Tau->size(); iPFTau++) {
      reco::PFTauRef RefPFTau(Tau, iPFTau);
      const reco::VertexRef PV=PFTauPVA->value(RefPFTau.key());
      const std::vector<reco::VertexRef> SV=PFTauSVA->value(RefPFTau.key());
      double dxy(-999), dxy_err(-999);
      reco::Vertex::Point poca(0,0,0);
      if(RefPFTau->leadPFChargedHadrCand().isNonnull()){
	if(RefPFTau->leadPFChargedHadrCand()->trackRef().isNonnull()){
	  if(useFullCalculation_){
	    reco::TransientTrack transTrk=transTrackBuilder->build(RefPFTau->leadPFChargedHadrCand()->trackRef());
	    GlobalPoint pv(PV->position().x(),PV->position().y(),PV->position().z());
	    dxy=-transTrk.trajectoryStateClosestToPoint(pv).perigeeParameters().vector()(aTip);
	    dxy_err=transTrk.trajectoryStateClosestToPoint(pv).perigeeError().covarianceMatrix()(aTip,aTip);
	    GlobalPoint pos=transTrk.trajectoryStateClosestToPoint(pv).position();
	    poca=reco::Vertex::Point(pos.x(),pos.y(),pos.x());
	  }
	  else{
	    dxy_err=RefPFTau->leadPFChargedHadrCand()->trackRef()->d0Error();
	    dxy=RefPFTau->leadPFChargedHadrCand()->trackRef()->dxy(PV->position());
	  }
	}
      }
      if(SV.size()>0){
	reco::Vertex::CovarianceMatrix cov;
	reco::Vertex::Point v(SV.at(0)->x()-PV->x(),SV.at(0)->y()-PV->y(),SV.at(0)->z()-PV->z());
	for(int i=0;i<reco::Vertex::dimension;i++){
	  for(int j=0;j<reco::Vertex::dimension;j++){
	    cov(i,j)=SV.at(0)->covariance(i,j)+PV->covariance(i,j);
	  }
	}
	reco::PFTauTransverseImpactParameter TIPV(poca,dxy,dxy_err,PV,v,cov,SV.at(0));
	std::cout << "FlightLength " << TIPV.FlightLength().Mag() << " sig- " << TIPV.FlightLengthSig() << std::endl;
	std::cout << "Vertex Quality " <<  TIPV.SecondaryVertex()->ndof() << " " << TIPV.SecondaryVertex()->chi2() << " " 
		  << TMath::Prob(TIPV.SecondaryVertex()->chi2(),TIPV.SecondaryVertex()->ndof()) << std::endl;
	AVPFTauTIP->setValue(iPFTau,TIPV);
      }
      else{
	reco::PFTauTransverseImpactParameter TIPV(poca,dxy,dxy_err,PV);
	AVPFTauTIP->setValue(iPFTau,TIPV);
      }
    }
  }
  iEvent.put(AVPFTauTIP);
}

DEFINE_FWK_MODULE(PFTauTransverseImpactParameters);
