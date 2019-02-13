#include "CommonTools/RecoAlgos/interface/TrackPUIDMVA.h"

TrackPUIDMVA::TrackPUIDMVA(std::string weights_file, bool is4D):
    is4D_(is4D)
{
    //---3D
    vars_.push_back(std::make_tuple("pt", 0.));
    vars_.push_back(std::make_tuple("eta", 0.));
    vars_.push_back(std::make_tuple("phi", 0.));
    vars_.push_back(std::make_tuple("dx", 0.));
    vars_.push_back(std::make_tuple("dy", 0.));
    vars_.push_back(std::make_tuple("dz", 0.));
    vars_.push_back(std::make_tuple("dzErr", 0.));
    vars_.push_back(std::make_tuple("dxyErr", 0.));
    vars_.push_back(std::make_tuple("chi2", 0.));
    vars_.push_back(std::make_tuple("ndof", 0.));
    vars_.push_back(std::make_tuple("numberOfValidHits", 0.));
    vars_.push_back(std::make_tuple("numberOfValidPixelBarrelHits", 0.));
    vars_.push_back(std::make_tuple("numberOfValidPixelEndcapHits", 0.));        
    //---4D
    if(is4D)
    {
        vars_.push_back(std::make_tuple("dt", 0.));
        vars_.push_back(std::make_tuple("sigmat0", 0.));
        vars_.push_back(std::make_tuple("btlMatchChi2", 0.));
        vars_.push_back(std::make_tuple("btlMatchTimeChi2", 0.));        
        vars_.push_back(std::make_tuple("etlMatchChi2", 0.));
        vars_.push_back(std::make_tuple("etlMatchTimeChi2", 0.));        
        vars_.push_back(std::make_tuple("mtdt", 0.));
        vars_.push_back(std::make_tuple("path_len", 0.));
    }

    mva_ = MVAComputer(&vars_, weights_file);
}

float TrackPUIDMVA::operator() (const reco::TrackRef& trk, const reco::Vertex& vtx)
{
    const auto& pattern = trk->hitPattern();                
    
    std::get<1>(vars_[0]) = trk->pt();
    std::get<1>(vars_[1]) = trk->eta();
    std::get<1>(vars_[2]) = trk->phi();
    std::get<1>(vars_[3]) = std::abs(trk->vx()-vtx.x());
    std::get<1>(vars_[4]) = std::abs(trk->vy()-vtx.y());
    std::get<1>(vars_[5]) = std::abs(trk->vz()-vtx.z());
    std::get<1>(vars_[6]) = trk->dzError();
    std::get<1>(vars_[7]) = trk->dxyError();
    std::get<1>(vars_[8]) = trk->chi2();
    std::get<1>(vars_[9]) = trk->ndof();
    std::get<1>(vars_[10]) = trk->numberOfValidHits();
    std::get<1>(vars_[11]) = pattern.numberOfValidPixelBarrelHits();
    std::get<1>(vars_[12]) = pattern.numberOfValidPixelEndcapHits();

    return mva_();
}

float TrackPUIDMVA::operator() (const reco::TrackRef& trk, const reco::TrackRef& ext_trk, const reco::Vertex& vtx,
                                edm::ValueMap<float>& t0s,
                                edm::ValueMap<float>& sigma_t0s,
                                edm::ValueMap<float>& btl_chi2s,
                                edm::ValueMap<float>& btl_time_chi2s,                      
                                edm::ValueMap<float>& etl_chi2s,
                                edm::ValueMap<float>& etl_time_chi2s,
                                edm::ValueMap<float>& tmtds,                                
                                edm::ValueMap<float>& trk_lengths)
{
    const auto& pattern = ext_trk->hitPattern();                
    
    std::get<1>(vars_[0]) = ext_trk->pt();
    std::get<1>(vars_[1]) = ext_trk->eta();
    std::get<1>(vars_[2]) = ext_trk->phi();
    std::get<1>(vars_[3]) = std::abs(ext_trk->vx()-vtx.x());
    std::get<1>(vars_[4]) = std::abs(ext_trk->vy()-vtx.y());
    std::get<1>(vars_[5]) = std::abs(ext_trk->vz()-vtx.z());
    std::get<1>(vars_[6]) = ext_trk->dzError();
    std::get<1>(vars_[7]) = ext_trk->dxyError();
    std::get<1>(vars_[8]) = ext_trk->chi2();
    std::get<1>(vars_[9]) = ext_trk->ndof();
    std::get<1>(vars_[10]) = ext_trk->numberOfValidHits();
    std::get<1>(vars_[11]) = pattern.numberOfValidPixelBarrelHits();
    std::get<1>(vars_[12]) = pattern.numberOfValidPixelEndcapHits();
    std::get<1>(vars_[13]) = t0s.contains(trk.id()) ? std::abs(t0s[trk]-vtx.t()) : std::abs(-1-vtx.t());
    std::get<1>(vars_[14]) = sigma_t0s.contains(trk.id()) ? sigma_t0s[trk] : -1;    
    std::get<1>(vars_[15]) = btl_chi2s.contains(ext_trk.id()) ? btl_chi2s[ext_trk] : -1;
    std::get<1>(vars_[16]) = btl_time_chi2s.contains(ext_trk.id()) ? btl_time_chi2s[ext_trk] : -1;    
    std::get<1>(vars_[17]) = etl_chi2s.contains(ext_trk.id()) ? etl_chi2s[ext_trk] : -1;
    std::get<1>(vars_[18]) = etl_time_chi2s.contains(ext_trk.id()) ? etl_time_chi2s[ext_trk] : -1;    
    std::get<1>(vars_[19]) = tmtds[ext_trk];
    std::get<1>(vars_[20]) = trk_lengths[ext_trk];

    return mva_();
}
