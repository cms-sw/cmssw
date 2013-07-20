
//
// $Id: MuonCleanerBySegments.cc,v 1.2 2013/02/27 20:42:45 wmtan Exp $
//

/**
  \class    modules::MuonCleanerBySegmentsT MuonCleanerBySegmentsT.h "MuonAnalysis/MuonAssociators/interface/MuonCleanerBySegmentsT.h"
  \brief    Removes duplicates from a muon collection using segment references.

            This module removes duplicates from a muon collection using segment references.

            All muons that don't pass the preselection are discarded first

            Then, for each pair of muons that share at least a given fraction of segments,
            the worse one is flagged as ghost.

            Finally, all muons that are not flagged as ghosts, or which pass a 'passthrough' selection,
            are saved in the output.
            
  \author   Giovanni Petrucciani
  \version  $Id: MuonCleanerBySegments.cc,v 1.2 2013/02/27 20:42:45 wmtan Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"


namespace modules {

  template<typename T>
  class MuonCleanerBySegmentsT : public edm::EDProducer {
    public:
      explicit MuonCleanerBySegmentsT(const edm::ParameterSet & iConfig);
      virtual ~MuonCleanerBySegmentsT() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      bool isSameMuon(const T &mu1, const T &mu2) const {
        return (& mu1 == & mu2)  ||
               (mu1.reco::Muon::innerTrack().isNonnull() ?
                       mu1.reco::Muon::innerTrack() == mu2.reco::Muon::innerTrack() :
                       mu1.reco::Muon::outerTrack() == mu2.reco::Muon::outerTrack());
      }
      bool isBetterMuon(const T &mu1, const T &mu2) const ;
    private:
      /// Labels for input collections
      edm::InputTag src_;

      /// Preselection cut
      StringCutObjectSelector<T> preselection_;
      /// Always-accept cut
      StringCutObjectSelector<T> passthrough_;
   
      /// Fraction of shared segments
      double sharedFraction_;

      /// Use default criteria to choose the best muon
      bool defaultBestMuon_;
      /// Cut on the pair of objects together
      typedef std::pair<const reco::Muon *, const reco::Muon *> MuonPointerPair;
      StringCutObjectSelector<MuonPointerPair, true> bestMuonSelector_;
  };

  template<>
  bool
  MuonCleanerBySegmentsT<pat::Muon>::isSameMuon(const pat::Muon &mu1, const pat::Muon &mu2) const ;
} // namespace

template<typename T>
modules::MuonCleanerBySegmentsT<T>::MuonCleanerBySegmentsT(const edm::ParameterSet & iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    preselection_(iConfig.existsAs<std::string>("preselection") ? iConfig.getParameter<std::string>("preselection") : ""),
    passthrough_(iConfig.existsAs<std::string>("passthrough") ? iConfig.getParameter<std::string>("passthrough") : "0"),
    sharedFraction_(iConfig.getParameter<double>("fractionOfSharedSegments")),
    defaultBestMuon_(!iConfig.existsAs<std::string>("customArbitration")),
    bestMuonSelector_(defaultBestMuon_ ? std::string("") : iConfig.getParameter<std::string>("customArbitration"))
{
    // this is the basic output (edm::Association is not generic)
    produces<std::vector<T> >(); 
}

template<typename T>
void 
modules::MuonCleanerBySegmentsT<T>::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<T> > src;
    auto_ptr<vector<T> > out(new vector<T>());

    iEvent.getByLabel(src_, src);
    unsigned int nsrc = src->size();
    out->reserve(nsrc);
    std::vector<int> good(nsrc, true);
    for (unsigned int i = 0; i < nsrc; ++i) {
        const T &mu1 = (*src)[i];
        if (!preselection_(mu1)) good[i] = false; 
        if (!good[i]) continue;
        int  nSegments1 = mu1.numberOfMatches(reco::Muon::SegmentArbitration);
        for (unsigned int j = i+1; j < nsrc; ++j) {
            const T &mu2 = (*src)[j];
            if (isSameMuon(mu1,mu2)) continue;
            if (!good[j] || !preselection_(mu2)) continue;
            int nSegments2 = mu2.numberOfMatches(reco::Muon::SegmentArbitration);
            if (nSegments2 == 0 || nSegments1 == 0) continue;
            double sf = muon::sharedSegments(mu1,mu2)/std::min<double>(nSegments1,nSegments2);
            if (sf > sharedFraction_) {
                if (isBetterMuon(mu1,mu2)) {
                    good[j] = false;
                } else {
                    good[i] = false;
                }
            }
        }
    }
    for (unsigned int i = 0; i < nsrc; ++i) {
        const T &mu1 = (*src)[i];
        if (good[i] || passthrough_(mu1)) out->push_back(mu1);
    }
    iEvent.put(out);
}

template<typename T>
bool
modules::MuonCleanerBySegmentsT<T>::isBetterMuon(const T &mu1, const T &mu2) const {
    if (!defaultBestMuon_) {
        MuonPointerPair pair = { &mu1, &mu2 };
        return bestMuonSelector_(pair);
    }
    if (mu2.track().isNull()) return true;
    if (mu1.track().isNull()) return false;
    if (mu1.isPFMuon()     != mu2.isPFMuon())     return mu1.isPFMuon();
    if (mu1.isGlobalMuon() != mu2.isGlobalMuon()) return mu1.isGlobalMuon();
    if (mu1.charge() == mu2.charge() && deltaR2(mu1,mu2) < 0.0009) {
        return mu1.track()->ptError()/mu1.track()->pt() < mu2.track()->ptError()/mu2.track()->pt();
    } else {
        int nm1 = mu1.numberOfMatches(reco::Muon::SegmentArbitration);
        int nm2 = mu2.numberOfMatches(reco::Muon::SegmentArbitration);
        return (nm1 != nm2 ? nm1 > nm2 : mu1.pt() > mu2.pt());
    }
}

template<>
bool
modules::MuonCleanerBySegmentsT<pat::Muon>::isSameMuon(const pat::Muon &mu1, const pat::Muon &mu2) const {
    return (& mu1 == & mu2)  ||
           (mu1.originalObjectRef() == mu2.originalObjectRef()) ||
           (mu1.reco::Muon::innerTrack().isNonnull() ?
                   mu1.reco::Muon::innerTrack() == mu2.reco::Muon::innerTrack() :
                   mu1.reco::Muon::outerTrack() == mu2.reco::Muon::outerTrack());
}

namespace modules {
    typedef modules::MuonCleanerBySegmentsT<reco::Muon>  MuonCleanerBySegments;
    typedef modules::MuonCleanerBySegmentsT<pat::Muon>   PATMuonCleanerBySegments;
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace modules;
DEFINE_FWK_MODULE(MuonCleanerBySegments);
DEFINE_FWK_MODULE(PATMuonCleanerBySegments);
