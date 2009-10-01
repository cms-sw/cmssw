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



#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TH1D.h"
#include <map>
// system include files
#include <memory>


class WMuNuProducer : public edm::EDProducer {
public:
  WMuNuProducer(const edm::ParameterSet&);
  ~WMuNuProducer();
  
private:

  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();

  edm::InputTag muonTag_;
  edm::InputTag metTag_;
  const std::string WMuNuCollectionTag_;


  unsigned int nall;


};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/Common/interface/View.h"

#include "DataFormats/Candidate/src/CompositeCandidate.cc"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "ElectroWeakAnalysis/WMuNu/interface/WMuNuCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"


  
using namespace edm;
using namespace std;
using namespace reco;

WMuNuProducer::WMuNuProducer( const ParameterSet & cfg ) :
      // Input collections
      muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met")))
{
  produces< WMuNuCandidateCollection >();
}

void WMuNuProducer::beginJob(const EventSetup &) {
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
      if (!ev.getByLabel(muonTag_, muonCollection)) {
            LogError("") << ">>> Muon collection does not exist !!!";
            return;
      }
      unsigned int muonCollectionSize = muonCollection->size();

      // MET
      Handle<View<MET> > metCollection;
      if (!ev.getByLabel(metTag_, metCollection)) {
            LogError("") << ">>> MET collection does not exist !!!";
            return;
      }
      //const MET& Met = metCollection->at(0);
      edm::Ptr<reco::MET> met(metCollection,0);


      if (muonCollectionSize<1) return;
      if (muonCollectionSize>10) {LogError("")<<"I am not prepared for events with 10 muons!! Fix me! (and check this event! :-) )"; return;}

      double MaxPt[10]={0,0,0,0,0,0,0,0,0,0};
      int Maxi[10]={0,0,0,0,0,0,0,0,0,0};
      int nmuons=0;
      // Now order remaining Muons by Pt  - there should be only one if preselection has been applied
       
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            if (mu.globalTrack().isNull()) continue;
            if (mu.innerTrack().isNull()) continue;

            reco::TrackRef gm = mu.globalTrack();

            double pt = mu.pt();
          
            nmuons++; 
           
            bool foundPosition=false; int j=0;
            do{   if (pt > MaxPt[j]) { MaxPt[j]=pt; Maxi[j]=i; foundPosition=true;}
                  j++;
            }while(!foundPosition);
               
     }
     if (nmuons<0) return;

     auto_ptr< WMuNuCandidateCollection > WMuNuCandidates(new WMuNuCandidateCollection );


     // Fill Collection with n muons --> n W Candidates ordered by pt
 
     for (int indx=0; indx<nmuons; indx++){ 
     int MuonIndx=Maxi[indx]; 
     edm::Ptr<reco::Muon> muon(muonCollection,MuonIndx);
   
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

      ev.put(WMuNuCandidates);

}

DEFINE_FWK_MODULE(WMuNuProducer);
