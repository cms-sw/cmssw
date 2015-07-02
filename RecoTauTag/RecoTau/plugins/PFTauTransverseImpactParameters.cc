/* class PFTauTransverseImpactParamters
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
#include "FWCore/Framework/interface/EDProducer.h"
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

class PFTauTransverseImpactParameters : public edm::stream::EDProducer<> {
 public:
  enum Alg{useInputPV=0, useFont};
  enum CMSSWPerigee{aCurv=0,aTheta,aPhi,aTip,aLip};
  explicit PFTauTransverseImpactParameters(const edm::ParameterSet& iConfig);
  ~PFTauTransverseImpactParameters();
  virtual void produce(edm::Event&,const edm::EventSetup&);
 private:
  edm::EDGetTokenT<std::vector<reco::PFTau> > PFTauToken_;
  edm::EDGetTokenT<edm::AssociationVector<PFTauRefProd, std::vector<reco::VertexRef> > > PFTauPVAToken_;
  edm::EDGetTokenT<edm::AssociationVector<PFTauRefProd,std::vector<std::vector<reco::VertexRef> > > > PFTauSVAToken_;
  bool useFullCalculation_;
};

PFTauTransverseImpactParameters::PFTauTransverseImpactParameters(const edm::ParameterSet& iConfig):
  PFTauToken_(consumes<std::vector<reco::PFTau> >(iConfig.getParameter<edm::InputTag>("PFTauTag"))),
  PFTauPVAToken_(consumes<edm::AssociationVector<PFTauRefProd, std::vector<reco::VertexRef> > >(iConfig.getParameter<edm::InputTag>("PFTauPVATag"))),
  PFTauSVAToken_(consumes<edm::AssociationVector<PFTauRefProd,std::vector<std::vector<reco::VertexRef> > > >(iConfig.getParameter<edm::InputTag>("PFTauSVATag"))),
  useFullCalculation_(iConfig.getParameter<bool>("useFullCalculation"))
{
  produces<edm::AssociationVector<PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> > >(); 
  produces<PFTauTransverseImpactParameterCollection>("PFTauTIP");
}

PFTauTransverseImpactParameters::~PFTauTransverseImpactParameters(){

}

void PFTauTransverseImpactParameters::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){
  // Obtain Collections
  edm::ESHandle<TransientTrackBuilder> transTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",transTrackBuilder);
  
  edm::Handle<std::vector<reco::PFTau> > Tau;
  iEvent.getByToken(PFTauToken_,Tau);

  edm::Handle<edm::AssociationVector<PFTauRefProd, std::vector<reco::VertexRef> > > PFTauPVA;
  iEvent.getByToken(PFTauPVAToken_,PFTauPVA);

  edm::Handle<edm::AssociationVector<PFTauRefProd,std::vector<std::vector<reco::VertexRef> > > > PFTauSVA;
  iEvent.getByToken(PFTauSVAToken_,PFTauSVA);

  // Set Association Map
  auto_ptr<edm::AssociationVector<PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> > > AVPFTauTIP(new edm::AssociationVector<PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> >(PFTauRefProd(Tau)));
  std::auto_ptr<PFTauTransverseImpactParameterCollection>  TIPCollection_out= std::auto_ptr<PFTauTransverseImpactParameterCollection>(new PFTauTransverseImpactParameterCollection());
  reco::PFTauTransverseImpactParameterRefProd TIPRefProd_out = iEvent.getRefBeforePut<reco::PFTauTransverseImpactParameterCollection>("PFTauTIP");


  // For each Tau Run Algorithim
  if(Tau.isValid()) {
    for(reco::PFTauCollection::size_type iPFTau = 0; iPFTau < Tau->size(); iPFTau++) {
      reco::PFTauRef RefPFTau(Tau, iPFTau);
      const reco::VertexRef PV=PFTauPVA->value(RefPFTau.key());
      const std::vector<reco::VertexRef> SV=PFTauSVA->value(RefPFTau.key());
      double dxy(-999), dxy_err(-999);
      reco::Vertex::Point poca(0,0,0);
      double ip3d(-999), ip3d_err(-999);
      reco::Vertex::Point ip3d_poca(0,0,0);
      if(RefPFTau->leadPFChargedHadrCand().isNonnull()){
	if(RefPFTau->leadPFChargedHadrCand()->trackRef().isNonnull()){
	  if(useFullCalculation_){
	    reco::TransientTrack transTrk=transTrackBuilder->build(RefPFTau->leadPFChargedHadrCand()->trackRef());
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
	    dxy_err=RefPFTau->leadPFChargedHadrCand()->trackRef()->d0Error();
	    dxy=RefPFTau->leadPFChargedHadrCand()->trackRef()->dxy(PV->position());
	    ip3d_err=RefPFTau->leadPFChargedHadrCand()->trackRef()->dzError(); //store dz, ip3d not available
	    ip3d=RefPFTau->leadPFChargedHadrCand()->trackRef()->dz(PV->position()); //store dz, ip3d not available 
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
  iEvent.put(TIPCollection_out,"PFTauTIP");
  iEvent.put(AVPFTauTIP);
}

DEFINE_FWK_MODULE(PFTauTransverseImpactParameters);
