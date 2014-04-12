//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                      //
//                                    WMuNuCandidate Producer                                                           //
//                                                                                                                      //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                      //
//    Productor of WMuNuCandidates for Analysis                                                                         //
//    --> Creates a WMuNuCandidateCollection                                                                            //
//    --> One Candidate is created per muon in the event, combinig the information with a selected kind of Met          //
//        (met kind configurable via cfg)                                                                               //
//    --> All WMuNuCandidates are stored in the event, ordered by muon pt.                                              //
//    --> The WMuNuCandidate to be used for the Inclusive analysis is then the first one! (Highest Pt)                  //
//                                                                                                                      //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TH1D.h"
#include <map>
// system include files
#include <memory>
#include <vector>

#include "AnalysisDataFormats/EWK/interface/WMuNuCandidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/METReco/interface/MET.h"


class WMuNuProducer : public edm::EDProducer {
public:
  WMuNuProducer(const edm::ParameterSet&);
  ~WMuNuProducer();


private:

  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void beginJob() override;
  virtual void endJob() override;

  edm::EDGetTokenT<edm::View<reco::Muon> > muonToken_;
  edm::EDGetTokenT<edm::View<reco::MET> > metToken_;

  struct ComparePt {
            bool operator()(reco::WMuNuCandidate w1, reco::WMuNuCandidate w2 ) const {
                  double pt1 = w1.getMuon().pt();
                  double pt2 = w2.getMuon().pt();
                  return (pt1> pt2);
            }
   };
  ComparePt ptComparator;

  unsigned int nall;


};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/GeometryVector/interface/Phi.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/Common/interface/View.h"




using namespace edm;
using namespace std;
using namespace reco;

WMuNuProducer::WMuNuProducer( const ParameterSet & cfg ) :
      // Input collections
      muonToken_(consumes<View<Muon> >(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons")))),
      metToken_(consumes<View<MET> >(cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met"))))
{
  produces< WMuNuCandidateCollection >();
}

void WMuNuProducer::beginJob() {
}

void WMuNuProducer::endJob() {
   LogTrace("")<<"WMuNuCandidateCollection Stored in the Event";
}


WMuNuProducer::~WMuNuProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


void WMuNuProducer::produce (Event & ev, const EventSetup &) {

      // Muon collection
      Handle<View<Muon> > muonCollection;
      if (!ev.getByToken(muonToken_, muonCollection)) {
            LogError("") << ">>> Muon collection does not exist !!!";
            return;
      }
      int muonCollectionSize = muonCollection->size();

      // MET
      Handle<View<MET> > metCollection;
      if (!ev.getByToken(metToken_, metCollection)) {
            LogError("") << ">>> MET collection does not exist !!!";
            return;
      }
      //const MET& Met = metCollection->at(0);
      edm::Ptr<reco::MET> met(metCollection,0);


      if (muonCollectionSize<1) return;

      auto_ptr< WMuNuCandidateCollection > WMuNuCandidates(new WMuNuCandidateCollection );


     // Fill Collection with n muons --> n W Candidates ordered by pt

     for (int indx=0; indx<muonCollectionSize; indx++){
            edm::Ptr<reco::Muon> muon(muonCollection,indx);
            if (!muon->isGlobalMuon()) continue;
            if (muon->globalTrack().isNull()) continue;
            if (muon->innerTrack().isNull()) continue;

      // Build WMuNuCandidate
      LogTrace("")<<"Building WMuNu Candidate!";
      WMuNuCandidate* WCand = new WMuNuCandidate(muon,met);
      LogTrace("") << "\t... W mass, W_et: "<<WCand->massT()<<", "<<WCand->eT()<<"[GeV]";
      LogTrace("") << "\t... W_px, W_py: "<<WCand->px()<<", "<< WCand->py() <<"[GeV]";
      LogTrace("") << "\t... acop:  " << WCand->acop();
      LogTrace("") << "\t... Muon pt, px, py, pz: "<<WCand->getMuon().pt()<<", "<<WCand->getMuon().px()<<", "<<WCand->getMuon().py()<<", "<< WCand->getMuon().pz()<<" [GeV]";
      LogTrace("") << "\t... Met  met_et, met_px, met_py : "<<WCand->getNeutrino().pt()<<", "<<WCand->getNeutrino().px()<<", "<<WCand->getNeutrino().py()<<" [GeV]";
  	WMuNuCandidates->push_back(*WCand);
       }

      std::sort(WMuNuCandidates->begin(),WMuNuCandidates->end(),ptComparator);

      ev.put(WMuNuCandidates);

}

DEFINE_FWK_MODULE(WMuNuProducer);
