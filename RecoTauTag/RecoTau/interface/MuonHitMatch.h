/*! Match hits in the muon system.
*/

#pragma once

#include "DataFormats/PatCandidates/interface/Muon.h"

namespace tau_analysis {

namespace MuonSubdetId {
enum { DT = 1, CSC = 2, RPC = 3, GEM = 4, ME0 = 5 };
}

struct MuonHitMatch {
    static constexpr size_t n_muon_stations = 4;
    static constexpr int first_station_id = 1;
    static constexpr int last_station_id = first_station_id + n_muon_stations - 1;
    using CountArray = std::array<unsigned, n_muon_stations>;
    using CountMap = std::map<int, CountArray>;

    static const std::vector<int>& ConsideredSubdets();
    static const std::string& SubdetName(int subdet);

    static size_t GetStationIndex(int station, bool throw_exception);
    static void CountMatches(const pat::Muon& muon, CountMap& n_matches);
    static void CountHits(const pat::Muon& muon, CountMap& n_hits);

    MuonHitMatch(const pat::Muon& muon);

    unsigned CountMuonStationsWithMatches(int first_station, int last_station) const;
    unsigned CountMuonStationsWithHits(int first_station, int last_station) const;

    unsigned NMatches(int subdet, int station) const;
    unsigned NHits(int subdet, int station) const;

private:
    CountMap n_matches, n_hits;
};

} // namespace tau_analysis
