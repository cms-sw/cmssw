/* class PFBaseTauTransverseImpactParameters
 * EDProducer of the 
 * authors: Ian M. Nugent
 * This work is based on the impact parameter work by Rosamaria Venditti and reconstructing the 3 prong taus.
 * The idea of the fully reconstructing the tau using a kinematic fit comes from
 * Lars Perchalla and Philip Sauerland Theses under Achim Stahl supervision. This
 * work was continued by Ian M. Nugent and Vladimir Cherepanov.
 * Thanks goes to Christian Veelken and Evan Klose Friis for their help and suggestions.
 */


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalTrajectoryExtrapolatorToLine.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"

#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"

#include "DataFormats/TauReco/interface/PFBaseTau.h"
#include "DataFormats/TauReco/interface/PFBaseTauFwd.h"
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
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "TMath.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFBaseTauTransverseImpactParameters : public edm::stream::EDProducer<> {
 public:
  enum Alg{useInputPV=0, useFont};
  enum CMSSWPerigee{aCurv=0,aTheta,aPhi,aTip,aLip};
  explicit PFBaseTauTransverseImpactParameters(const edm::ParameterSet& iConfig);
  ~PFBaseTauTransverseImpactParameters() override;
  void produce(edm::Event&,const edm::EventSetup&) override;
 private:
  edm::EDGetTokenT<std::vector<reco::PFBaseTau> > PFTauToken_;
  std::auto_ptr<tau::RecoTauVertexAssociator> vertexAssociator_;
  bool useFullCalculation_;
};

PFBaseTauTransverseImpactParameters::PFBaseTauTransverseImpactParameters(const edm::ParameterSet& iConfig):
  PFTauToken_(consumes<std::vector<reco::PFBaseTau> >(iConfig.getParameter<edm::InputTag>("PFTauTag"))),
  useFullCalculation_(iConfig.getParameter<bool>("useFullCalculation"))
{
  produces<edm::AssociationVector<PFBaseTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> > >(); 
  produces<PFTauTransverseImpactParameterCollection>("PFTauTIP");

  vertexAssociator_.reset(
      new tau::RecoTauVertexAssociator(iConfig, this->consumesCollector()));
}

PFBaseTauTransverseImpactParameters::~PFBaseTauTransverseImpactParameters(){

}

namespace {
  const reco::Track* getTrack(const reco::Candidate& cand) {
    const pat::PackedCandidate* pCand = dynamic_cast<const pat::PackedCandidate*>(&cand);
    if (pCand && pCand->hasTrackDetails())
    	return &pCand->pseudoTrack();
    return nullptr;
  }
}

void PFBaseTauTransverseImpactParameters::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){
  // Obtain Collections
  edm::ESHandle<TransientTrackBuilder> transTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",transTrackBuilder);
  
  edm::Handle<std::vector<reco::PFBaseTau> > Tau;
  iEvent.getByToken(PFTauToken_,Tau);

  vertexAssociator_->setEvent(iEvent);

  // Set Association Map
  auto AVPFTauTIP = std::make_unique< edm::AssociationVector<PFBaseTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>>(PFBaseTauRefProd(Tau));
  auto TIPCollection_out = std::make_unique<PFTauTransverseImpactParameterCollection>();
  reco::PFTauTransverseImpactParameterRefProd TIPRefProd_out = iEvent.getRefBeforePut<reco::PFTauTransverseImpactParameterCollection>("PFTauTIP");


  // For each Tau Run Algorithim
  if(Tau.isValid()) {
    for(reco::PFBaseTauCollection::size_type iPFTau = 0; iPFTau < Tau->size(); iPFTau++) {
      reco::PFBaseTauRef RefPFTau(Tau, iPFTau);
      const reco::VertexRef PV = vertexAssociator_->associatedVertex(*RefPFTau);
      const std::vector<reco::VertexRef> SV;
      double dxy(-999), dxy_err(-999);
      reco::Vertex::Point poca(0,0,0);
      double ip3d(-999), ip3d_err(-999);
      reco::Vertex::Point ip3d_poca(0,0,0);
      if(RefPFTau->leadPFChargedHadrCand().isNonnull()) {
        const reco::Track* track = getTrack(*RefPFTau->leadPFChargedHadrCand());

	if(track){
	  if(useFullCalculation_){
	    reco::TransientTrack transTrk=transTrackBuilder->build(*track);
	    GlobalVector direction(RefPFTau->p4().px(), RefPFTau->p4().py(), RefPFTau->p4().pz()); //To compute sign of IP
	    std::pair<bool,Measurement1D> signed_IP2D = IPTools::signedTransverseImpactParameter(transTrk, direction, (*PV));
	    dxy=signed_IP2D.second.value();
	    dxy_err=signed_IP2D.second.error();
	    std::pair<bool,Measurement1D> signed_IP3D = IPTools::signedImpactParameter3D(transTrk, direction, (*PV));
	    ip3d=signed_IP3D.second.value();
	    ip3d_err=signed_IP3D.second.error();
	    TransverseImpactPointExtrapolator extrapolator(transTrk.field());
	    GlobalPoint pos  = extrapolator.extrapolate(transTrk.impactPointState(), RecoVertex::convertPos(PV->position())).globalPosition();
	    poca=reco::Vertex::Point(pos.x(),pos.y(),pos.z());
	    AnalyticalImpactPointExtrapolator extrapolator3D(transTrk.field());
	    GlobalPoint pos3d = extrapolator3D.extrapolate(transTrk.impactPointState(),RecoVertex::convertPos(PV->position())).globalPosition();
	    ip3d_poca=reco::Vertex::Point(pos3d.x(),pos3d.y(),pos3d.z());
	  }
	  else{
	    dxy_err=track->d0Error();
	    dxy=track->dxy(PV->position());
	    ip3d_err=track->dzError(); //store dz, ip3d not available
	    ip3d=track->dz(PV->position()); //store dz, ip3d not available 
	  }
	}
      }
      if( !SV.empty() ){
	reco::Vertex::CovarianceMatrix cov;
	reco::Vertex::Point v(SV.at(0)->x()-PV->x(),SV.at(0)->y()-PV->y(),SV.at(0)->z()-PV->z());
	for(int i=0;i<reco::Vertex::dimension;i++){
	  for(int j=0;j<reco::Vertex::dimension;j++){
	    cov(i,j)=SV.at(0)->covariance(i,j)+PV->covariance(i,j);
	  }
	}
	GlobalVector direction(RefPFTau->px(),RefPFTau->py(),RefPFTau->pz());
	double vSig = SecondaryVertex::computeDist3d(*PV,*SV.at(0),direction,true).significance();
	reco::PFTauTransverseImpactParameter TIPV(poca,dxy,dxy_err,ip3d_poca,ip3d,ip3d_err,PV,v,vSig,SV.at(0));
	reco::PFTauTransverseImpactParameterRef TIPVRef=reco::PFTauTransverseImpactParameterRef(TIPRefProd_out,TIPCollection_out->size());
        TIPCollection_out->push_back(TIPV);
        AVPFTauTIP->setValue(iPFTau,TIPVRef);
      }
      else{ 
	reco::PFTauTransverseImpactParameter TIPV(poca,dxy,dxy_err,ip3d_poca,ip3d,ip3d_err,PV);
	reco::PFTauTransverseImpactParameterRef TIPVRef=reco::PFTauTransverseImpactParameterRef(TIPRefProd_out,TIPCollection_out->size());
	TIPCollection_out->push_back(TIPV);
	AVPFTauTIP->setValue(iPFTau,TIPVRef);
      }
    }
  }
  iEvent.put(std::move(TIPCollection_out),"PFTauTIP");
  iEvent.put(std::move(AVPFTauTIP));
}

DEFINE_FWK_MODULE(PFBaseTauTransverseImpactParameters);
