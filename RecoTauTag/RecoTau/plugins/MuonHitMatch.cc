/*! Match hits in the muon system.
*/

#include "RecoTauTag/RecoTau/interface/MuonHitMatch.h"

namespace tau_analysis {

const std::vector<int>& MuonHitMatch::ConsideredSubdets()
{
    static const std::vector<int> subdets = { MuonSubdetId::DT, MuonSubdetId::CSC, MuonSubdetId::RPC };
    return subdets;
}

const std::string& MuonHitMatch::SubdetName(int subdet)
{
    static const std::map<int, std::string> subdet_names = {
        { MuonSubdetId::DT, "DT" }, { MuonSubdetId::CSC, "CSC" }, { MuonSubdetId::RPC, "RPC" }
    };
    if(!subdet_names.count(subdet))
        throw cms::Exception("MuonHitMatch") << "Subdet name for subdet id " << subdet << " not found.";
    return subdet_names.at(subdet);
}

size_t MuonHitMatch::GetStationIndex(int station, bool throw_exception)
{
    if(station < first_station_id || station > last_station_id) {
        if(throw_exception)
            throw cms::Exception("MuonHitMatch") << "Station id is out of range";
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(station - 1);
}

MuonHitMatch::MuonHitMatch(const pat::Muon& muon)
{
    for(int subdet : ConsideredSubdets()) {
        n_matches[subdet].fill(0);
        n_hits[subdet].fill(0);
    }

    CountMatches(muon, n_matches);
    CountHits(muon, n_hits);
}

void MuonHitMatch::CountMatches(const pat::Muon& muon, CountMap& n_matches)
{
    for(const auto& segment : muon.matches()) {
        if(segment.segmentMatches.empty() && segment.rpcMatches.empty()) continue;
        if(n_matches.count(segment.detector())) {
            const size_t station_index = GetStationIndex(segment.station(), true);
            ++n_matches.at(segment.detector()).at(station_index);
        }
    }
}

void MuonHitMatch::CountHits(const pat::Muon& muon, CountMap& n_hits)
{
    if(muon.outerTrack().isNonnull()) {
        const auto& hit_pattern = muon.outerTrack()->hitPattern();
        for(int hit_index = 0; hit_index < hit_pattern.numberOfAllHits(reco::HitPattern::TRACK_HITS); ++hit_index) {
            auto hit_id = hit_pattern.getHitPattern(reco::HitPattern::TRACK_HITS, hit_index);
            if(hit_id == 0) break;
            if(hit_pattern.muonHitFilter(hit_id) && (hit_pattern.getHitType(hit_id) == TrackingRecHit::valid
                                                     || hit_pattern.getHitType(hit_id) == TrackingRecHit::bad)) {
                const size_t station_index = GetStationIndex(hit_pattern.getMuonStation(hit_id), false);
                if(station_index < n_muon_stations) {
                    CountArray* muon_n_hits = nullptr;
                    if(hit_pattern.muonDTHitFilter(hit_id))
                        muon_n_hits = &n_hits.at(MuonSubdetId::DT);
                    else if(hit_pattern.muonCSCHitFilter(hit_id))
                        muon_n_hits = &n_hits.at(MuonSubdetId::CSC);
                    else if(hit_pattern.muonRPCHitFilter(hit_id))
                        muon_n_hits = &n_hits.at(MuonSubdetId::RPC);

                    if(muon_n_hits)
                        ++muon_n_hits->at(station_index);
                }
            }
        }
    }
}

unsigned MuonHitMatch::NMatches(int subdet, int station) const
{
    if(!n_matches.count(subdet))
        throw cms::Exception("MuonHitMatch") << "Subdet " << subdet << " not found.";
    const size_t station_index = GetStationIndex(station, true);
    return n_matches.at(subdet).at(station_index);
}

unsigned MuonHitMatch::NHits(int subdet, int station) const
{
    if(!n_hits.count(subdet))
        throw cms::Exception("MuonHitMatch") << "Subdet " << subdet << " not found.";
    const size_t station_index = GetStationIndex(station, true);
    return n_hits.at(subdet).at(station_index);
}

unsigned MuonHitMatch::CountMuonStationsWithMatches(int first_station, int last_station) const
{
    static const std::map<int, std::vector<bool>> masks = {
        { MuonSubdetId::DT, { false, false, false, false } },
        { MuonSubdetId::CSC, { true, false, false, false } },
        { MuonSubdetId::RPC, { false, false, false, false } },
    };
    const size_t first_station_index = GetStationIndex(first_station, true);
    const size_t last_station_index = GetStationIndex(last_station, true);
    unsigned cnt = 0;
    for(size_t n = first_station_index; n <= last_station_index; ++n) {
        for(const auto& match : n_matches) {
            if(!masks.at(match.first).at(n) && match.second.at(n) > 0) ++cnt;
        }
    }
    return cnt;
}

unsigned MuonHitMatch::CountMuonStationsWithHits(int first_station, int last_station) const
{
    static const std::map<int, std::vector<bool>> masks = {
        { MuonSubdetId::DT, { false, false, false, false } },
        { MuonSubdetId::CSC, { false, false, false, false } },
        { MuonSubdetId::RPC, { false, false, false, false } },
    };

    const size_t first_station_index = GetStationIndex(first_station, true);
    const size_t last_station_index = GetStationIndex(last_station, true);
    unsigned cnt = 0;
    for(size_t n = first_station_index; n <= last_station_index; ++n) {
        for(const auto& hit : n_hits) {
            if(!masks.at(hit.first).at(n) && hit.second.at(n) > 0) ++cnt;
        }
    }
    return cnt;
}

} // namespace tau_analysis
