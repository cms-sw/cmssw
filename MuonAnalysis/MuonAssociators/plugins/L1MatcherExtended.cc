// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "MuonAnalysis/MuonAssociators/interface/L1MuonMatcherAlgo.h"
#include "MuonAnalysis/MuonAssociators/interface/SegmentLCTMatchBox.h"

#include <iostream>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

class L1MatcherExtended : public edm::EDProducer {
    public:
        explicit L1MatcherExtended(const edm::ParameterSet&);
        ~L1MatcherExtended();

    private:
        virtual void produce(edm::Event&, const edm::EventSetup&);
        virtual void beginRun(edm::Run&, const edm::EventSetup&);
        /// The RECO objects
        edm::InputTag muons_;

        edm::InputTag l1extra_;

        reco::Muon::ArbitrationType arbitration_;

        edm::InputTag csctfDigis_, csctfLcts_, cscSegments_;

        L1MuonMatcherAlgo  matcherGeom_;
        SegmentLCTMatchBox matcherCsc_;

        /// Write a ValueMap<int> in the event
        template<typename T>
        void writeValueMap(edm::Event &iEvent,
                const edm::Handle<edm::View<reco::Muon> > & handle,
                const std::vector<T> & values,
                const std::string    & label) const ;


};

L1MatcherExtended::L1MatcherExtended(const edm::ParameterSet & iConfig) :
    muons_(iConfig.getParameter<edm::InputTag>("muons")),
    l1extra_(iConfig.getParameter<edm::InputTag>("l1extra")),
    csctfDigis_(iConfig.getParameter<edm::InputTag>("csctfDigis")),
    csctfLcts_(iConfig.getParameter<edm::InputTag>("csctfLcts")),
    /*cscSegments_(iConfig.getParameter<edm::InputTag>("cscSegments")),*/
    matcherGeom_(iConfig.getParameter<edm::ParameterSet>("matcherGeom")),
    matcherCsc_(4,4,4,iConfig.getUntrackedParameter<int>("matcherCscPrintLevel", -1))
{
    std::string arbitration = iConfig.getParameter<std::string>("segmentArbitration");
    if (arbitration == "SegmentArbitration") { 
        arbitration_ = reco::Muon::SegmentArbitration;
    } else if (arbitration == "SegmentAndTrackArbitration") {
        arbitration_ = reco::Muon::SegmentAndTrackArbitration;
    } else if (arbitration == "NoArbitration") {
        arbitration_ = reco::Muon::NoArbitration;
    } else throw cms::Exception("Configuration") << "Parameter 'segmentArbitration' must be one of SegmentArbitration, SegmentAndTrackArbitration, NoArbitration\n";

    produces<edm::ValueMap<int> >();
    produces<edm::ValueMap<int> >("cscMode");
    produces<edm::ValueMap<int> >("canPropagate");
    produces<edm::ValueMap<int> >("l1q");
    produces<edm::ValueMap<float> >("deltaR");
    produces<edm::ValueMap<float> >("deltaPhi");
    produces<edm::ValueMap<float> >("deltaEta");
    produces<edm::ValueMap<float> >("l1pt");
}

L1MatcherExtended::~L1MatcherExtended()
{
}

void 
L1MatcherExtended::beginRun(edm::Run & iRun,  const edm::EventSetup & iSetup) 
{
    matcherGeom_.init(iSetup);
}

void 
L1MatcherExtended::produce(edm::Event & iEvent,  const edm::EventSetup & iSetup) 
{
    using namespace edm;
    Handle<View<reco::Muon> > muons;
    iEvent.getByLabel(muons_, muons);

    Handle<std::vector<l1extra::L1MuonParticle> > l1s;
    iEvent.getByLabel(l1extra_, l1s);

    Handle<L1CSCTrackCollection> csctfDigis;
    Handle<CSCCorrelatedLCTDigiCollection> csctfLcts;
    iEvent.getByLabel(csctfDigis_, csctfDigis);
    iEvent.getByLabel(csctfLcts_, csctfLcts);

    size_t nmu = muons->size();
    std::vector<int> ret(nmu, 0), cscMode(nmu, 0), canPropagate(nmu, 0), l1q(nmu, -1);
    std::vector<float> dPhi(nmu, 999.), dEta(nmu, 999.), dR(nmu, 999.), l1pt(nmu, -1);

    for (size_t i = 0; i < nmu; ++i) {
        const reco::Muon &mu = (*muons)[i];
        //std::cout << "\n\nSearching for matches for muon of pt " << mu.pt() << ", eta " << mu.eta() << ", nseg = " << mu.numberOfMatches(arbitration_) << std::endl;

        TrajectoryStateOnSurface propagated;
        int matchGeom = matcherGeom_.match(mu, *l1s, dR[i], dPhi[i], propagated);
        if (propagated.isValid()) {
            canPropagate[i] = 1;
            //std::cout << "   can reach MB2/ME2" << std::endl;
        }
        if (matchGeom != -1) {
            const l1extra::L1MuonParticle &l1 = (*l1s)[matchGeom];
            l1q[i]  = l1.gmtMuonCand().quality();
            l1pt[i] = l1.pt();
            dEta[i] = propagated.globalPosition().eta() - l1.eta();
            //std::cout << "   found geometrical match " << matchGeom <<": q = " << l1q[i] << ", pt = " << l1pt[i] << ", dphi = " << dPhi[i] << ", deta = " << dEta[i] << std::endl;
            ret[i] = 10;
        }

        int cscrank = 0;
        foreach (const reco::MuonChamberMatch & chMatch, mu.matches()) {
            if (chMatch.detector() != MuonSubdetId::CSC) continue;
            foreach (const reco::MuonSegmentMatch & segMatch, chMatch.segmentMatches) {
                CSCSegmentRef segmentCSC = segMatch.cscSegmentRef;
                if (segmentCSC.isNull()) continue;
            
                if(arbitration_ == reco::Muon::SegmentArbitration) {
                    if(!segMatch.isMask(reco::MuonSegmentMatch::BestInChamberByDR)) {
                        continue;
                    }
                } else if(arbitration_ == reco::Muon::SegmentAndTrackArbitration) {
                    if(!segMatch.isMask(reco::MuonSegmentMatch::BestInChamberByDR) ||
                       !segMatch.isMask(reco::MuonSegmentMatch::BelongsToTrackByDR)) {
                        continue;
                    }
                }

                int mode = matcherCsc_.whichMode(*segmentCSC, csctfDigis, csctfLcts);
                // According to GianP. and sorted by "quality"
                // 1  = bad phi road 
                // 11 = single
                // 15 = halo
                // 2-10, 12-14: coinc.
                int thisrank = 0;
                if ((2 <= mode && mode <= 10) || (12 <= mode && mode <= 14)) {
                    thisrank = 4;
                } else if (mode == 15) {
                    thisrank = 3;
                } else if (mode == 11) {
                    thisrank = 2;
                } else if (mode == 1) {
                    thisrank = 1;
                }
                if (thisrank > cscrank) {
                    cscrank = thisrank;
                    cscMode[i] = mode;
                }
                //std::cout << "   found match with mode " << mode << ", final mode is " << cscMode[i] << " (rank " << thisrank << ")" << std::endl;
            }
            ret[i] += cscrank;
        }
    }

    writeValueMap(iEvent, muons, ret, "");
    writeValueMap(iEvent, muons, cscMode,      "cscMode");
    writeValueMap(iEvent, muons, canPropagate, "canPropagate");
    writeValueMap(iEvent, muons, l1q,  "l1q");
    writeValueMap(iEvent, muons, l1pt, "l1pt");
    writeValueMap(iEvent, muons, dR,   "deltaR");
    writeValueMap(iEvent, muons, dEta, "deltaEta");
    writeValueMap(iEvent, muons, dPhi, "deltaPhi");
}

template<typename T>
void
L1MatcherExtended::writeValueMap(edm::Event &iEvent,
        const edm::Handle<edm::View<reco::Muon> > & handle,
        const std::vector<T> & values,
        const std::string    & label) const 
{
    using namespace edm; 
    using namespace std;
    auto_ptr<ValueMap<T> > valMap(new ValueMap<T>());
    typename edm::ValueMap<T>::Filler filler(*valMap);
    filler.insert(handle, values.begin(), values.end());
    filler.fill();
    iEvent.put(valMap, label);
}



//define this as a plug-in
DEFINE_FWK_MODULE(L1MatcherExtended);
