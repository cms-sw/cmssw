#include "MuonAnalysis/MuonAssociators/interface/MatcherByPullsAlgorithm.h"

// user include files
#include "DataFormats/Math/interface/deltaR.h"

/*     ____                _                   _             
 *    / ___|___  _ __  ___| |_ _ __ _   _  ___| |_ ___  _ __ 
 *   | |   / _ \| '_ \/ __| __| '__| | | |/ __| __/ _ \| '__|
 *   | |__| (_) | | | \__ \ |_| |  | |_| | (__| || (_) | |   
 *    \____\___/|_| |_|___/\__|_|   \__,_|\___|\__\___/|_|   
 *                                                           
 */  
MatcherByPullsAlgorithm::MatcherByPullsAlgorithm(const edm::ParameterSet &iConfig) :
    dr2_(std::pow(iConfig.getParameter<double>("maxDeltaR"),2)),
    cut_(iConfig.getParameter<double>("maxPull")),
    diagOnly_(iConfig.getParameter<bool>("diagonalElementsOnly")),
    useVertex_(iConfig.getParameter<bool>("useVertexVariables"))
{
    std::string track = iConfig.getParameter<std::string>("track");
    if      (track == "standAloneMuon") track_ = StaTrack;
    else if (track == "combinedMuon")   track_ = GlbTrack;
    else if (track == "track")          track_ = TrkTrack;
    else throw cms::Exception("Configuration") << "MatcherByPullsAlgorithm: track '"<<track<<"' is not known\n" << 
               "Allowed values are: 'track', 'combinedMuon', 'standAloneMuon' (as per reco::RecoCandidate object)\n";
}

MatcherByPullsAlgorithm::~MatcherByPullsAlgorithm()
{
}

/*    __  __       _       _                   
 *   |  \/  | __ _| |_ ___| |__   ___ _ __ ___ 
 *   | |\/| |/ _` | __/ __| '_ \ / _ \ '__/ __|
 *   | |  | | (_| | || (__| | | |  __/ |  \__ \
 *   |_|  |_|\__,_|\__\___|_| |_|\___|_|  |___/
 *                                             
 */  
std::pair<bool,float> 
MatcherByPullsAlgorithm::match(const reco::Track &tk, const reco::Candidate &c, const AlgebraicSymMatrix55 &invCov) const {
    if (::deltaR2(tk,c) <= dr2_) {
        AlgebraicVector5 diff(tk.qoverp() - c.charge()/c.p(), 
                              tk.theta() - c.theta(), 
                              ::deltaPhi(tk.phi(), c.phi()),
                              tk.dxy(c.vertex()),
                              tk.dsz(c.vertex())); 
        double pull = ROOT::Math::Similarity(diff, invCov);
#if 0
        std::cout << "Tk charge/pt/eta/phi/vx/vy/vz " << tk.charge() << "\t" << tk.pt() << "\t" << tk.eta() << "\t" << tk.phi() << "\t" << tk.vx() << "\t" << tk.vy() << "\t" << tk.vz() << std::endl;
        std::cout << "MC charge/pt/eta/phi/vx/vy/vz " << c.charge() << "\t" << c.pt() << "\t" << c.eta() << "\t" << c.phi() << "\t" << c.vx() << "\t" << c.vy() << "\t" << c.vz() << std::endl;
        std::cout << "Delta: " << diff << std::endl;
        std::cout << "Sigmas: ";
        for (size_t i = 0; i < 5; ++i) {
            if (invCov(i,i) == 0) std::cout << "---\t";
            else std::cout << std::sqrt(1.0/invCov(i,i)) << "\t";
        }
        std::cout << std::endl;
        std::cout << "Items: ";
        for (size_t i = 0; i < 5; ++i) {
            if (invCov(i,i) == 0) std::cout << "---\t";
            else std::cout << diff(i)*std::sqrt(invCov(i,i)) << "\t";
        }
        std::cout << std::endl;
        std::cout << "Pull: "  << pull << std::endl;
#endif
        return std::pair<bool,float>(pull < cut_, pull);
    }
    return std::pair<bool,float>(false,9e9);
}



std::pair<int,float>
MatcherByPullsAlgorithm::match(const reco::RecoCandidate &src, 
                      const std::vector<reco::GenParticle> &cands,
                      const std::vector<uint8_t>       &good) const 
{
    const reco::Track * tk = track(src);
    return (tk == nullptr ? 
            std::pair<int,float>(-1,9e9) : 
            match(*tk, cands, good));
}

std::pair<int,float>
MatcherByPullsAlgorithm::match(const reco::Track &tk, 
                      const std::vector<reco::GenParticle> &cands,
                      const std::vector<uint8_t>       &good) const 
{
    std::pair<int,float> best(-1,9e9);

    AlgebraicSymMatrix55 invCov; fillInvCov(tk, invCov);
    for (int i = 0, n = cands.size(); i < n; ++i) {
        if (!good[i]) continue;
        std::pair<bool,float> m = match(tk, cands[i], invCov);
        if (m.first && (m.second < best.second)) {
            best.first  = i;
            best.second = m.second;
        } 
    }
    return best;
}

void
MatcherByPullsAlgorithm::matchMany(const reco::RecoCandidate &src,
                const std::vector<reco::GenParticle> &cands,
                const std::vector<uint8_t>           &good,
                std::vector<std::pair<double,int> >  &matchesToFill) const
{
    const reco::Track * tk = track(src);
    if (tk != nullptr) matchMany(*tk, cands, good, matchesToFill);
}

void
MatcherByPullsAlgorithm::matchMany(const reco::Track &tk,
                const std::vector<reco::GenParticle> &cands,
                const std::vector<uint8_t>           &good,
                std::vector<std::pair<double,int> >  &matchesToFill) const
{

    AlgebraicSymMatrix55 invCov; fillInvCov(tk, invCov);
    for (int i = 0, n = cands.size(); i < n; ++i) {
        if (!good[i]) continue;
        std::pair<bool,double> m = match(tk, cands[i], invCov);
        if (m.first) matchesToFill.push_back(std::make_pair(m.second,i));
    }
    std::sort(matchesToFill.begin(),matchesToFill.end());
}

/*    _   _ _   _ _ _ _   _           
 *   | | | | |_(_) (_) |_(_) ___  ___ 
 *   | | | | __| | | | __| |/ _ \/ __|
 *   | |_| | |_| | | | |_| |  __/\__ \
 *    \___/ \__|_|_|_|\__|_|\___||___/
 *                                    
 */  
const reco::Track * 
MatcherByPullsAlgorithm::track(const reco::RecoCandidate &muon) const {
    switch (track_) {
        case StaTrack : return muon.standAloneMuon().isNonnull() ? muon.standAloneMuon().get() : nullptr;
        case GlbTrack : return muon.combinedMuon().isNonnull()   ? muon.combinedMuon().get()   : nullptr;
        case TrkTrack : return muon.track().isNonnull()          ? muon.track().get()          : nullptr;
    }
    assert(false);
}


void
MatcherByPullsAlgorithm::fillInvCov(const reco::Track &tk, AlgebraicSymMatrix55 &invCov) const 
{
    if (useVertex_) {
        invCov = tk.covariance();
        if (diagOnly_) {
            for (size_t i = 0; i < 5; ++i) { for (size_t j = i+1; j < 5; ++j) { 
                invCov(i,j) = 0;
            } }
        }
        invCov.Invert();
    } else {
        AlgebraicSymMatrix33 momCov = tk.covariance().Sub<AlgebraicSymMatrix33>(0,0); // get 3x3 matrix
        if (diagOnly_) { momCov(0,1) = 0; momCov(0,2) = 0; momCov(1,2) = 0; }
        momCov.Invert();
        invCov.Place_at(momCov,0,0);
    }
}
