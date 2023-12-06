#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DataUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/TemplateUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/DuplicateRemovalLayer.h"

using namespace emtf::phase2;
using namespace emtf::phase2::algo;

DuplicateRemovalLayer::DuplicateRemovalLayer(
        const EMTFContext& context
):
    context_(context)
{
    // Do Nothing
}

DuplicateRemovalLayer::~DuplicateRemovalLayer() {
    // Do Nothing
}

void DuplicateRemovalLayer::apply(
        std::vector<track_t>& tracks
) const {
    // ===========================================================================
    // Unpack model
    // ---------------------------------------------------------------------------
    const auto& model = context_.model_;
    const auto& model_reduced_sites = model.reduced_sites_;

    // ===========================================================================
    // Build reduced tracks
    // ---------------------------------------------------------------------------
    std::vector<reduced_track_t> reduced_tracks;

    for (const auto& track : tracks) { // Begin loop tracks
        // Fetch reduced track
        auto& rtrk       = reduced_tracks.emplace_back();
        auto& rtrk_valid = rtrk.valid;

        // Initialize valid state
        rtrk_valid = track.valid;

        // Fill reduced track with segments
        for (const auto& model_rsite : model_reduced_sites) { // Begin loop reduced model sites            
            
            // Get reduced site
            int model_rsite_id = static_cast<int>(model_rsite.id);

            auto& rsite_seg = rtrk.site_segs[model_rsite_id];
            auto& rsite_bit = rtrk.site_mask[model_rsite_id];

            // Init reduced site
            rsite_seg = 0;
            rsite_bit = 0;

            // Select the first segment available for the reduced site
            for (const auto& model_rs_ts : model_rsite.trk_sites) { // Begin loop reduced site track sites
                int trk_site_id = static_cast<int>(model_rs_ts);

                const auto& trk_site_seg = track.site_segs[trk_site_id];
                const auto& trk_site_bit = track.site_mask[trk_site_id];

                if (trk_site_bit == 0) {
                    continue;
                }

                // Attach segment
                // If even one segment is attached 
                // the reduced track is considered valid
                rtrk_valid = 1           ;                
                rsite_seg  = trk_site_seg;
                rsite_bit  = 1           ;

                break;
            } // End loop reduced site track sites
        } // End loop reduced model sites
    } // End loop tracks

    // ===========================================================================
    // Find and invalidate duplicate tracks
    // ---------------------------------------------------------------------------
    for (unsigned int i_rtrk = 0; i_rtrk < reduced_tracks.size(); ++i_rtrk) { // Begin loop reduced tracks i

        auto& trk_i = tracks[i_rtrk];
        const auto& rtrk_i = reduced_tracks[i_rtrk];
        
        if (rtrk_i.valid == 1) {
            for (unsigned int j_rtrk = (i_rtrk + 1); j_rtrk < reduced_tracks.size(); ++j_rtrk) { // Begin loop reduced tracks j
                
                auto& rtrk_j = reduced_tracks[j_rtrk];
                
                // If the reduced track is already invalid, move on
                if (rtrk_j.valid == 0)
                    continue;

                // Compare reduced track sites
                for (int k_rsite = 0; k_rsite < v3::kNumTrackSitesRM; ++k_rsite) { // Begin loop reduced sites k
                    const auto& rtrk_site_mask_ik = rtrk_i.site_mask[k_rsite];
                    const auto& rtrk_site_mask_jk = rtrk_j.site_mask[k_rsite];

                    // If one or both of the sites are missing, move on
                    if (!(rtrk_site_mask_ik & rtrk_site_mask_jk))
                        continue;

                    // Compare segment_ids
                    const auto& rtrk_seg_id_ik = rtrk_i.site_segs[k_rsite];
                    const auto& rtrk_seg_id_jk = rtrk_j.site_segs[k_rsite];

                    // If segment ids are differente, move on
                    if (rtrk_seg_id_ik != rtrk_seg_id_jk)
                        continue; 
                    
                    // If there's even one collision, invalidate the track
                    rtrk_j.valid = 0;
                } // End loop reduced sites k 
            } // End loop reduced tracks j
        }

        // Propagate invalidation
        trk_i.valid = rtrk_i.valid;

        // DEBUG
        if (CONFIG.verbosity_ > 1) {
            if (trk_i.valid) {
                std::cout
                    << "Unique Track"
                    << " zone "  << trk_i.zone
                    << " col "   << trk_i.col
                    << " pat "   << trk_i.pattern
                    << " qual "  << trk_i.quality
                    << " phi "   << trk_i.phi
                    << " theta " << trk_i.theta
                    << " valid " << trk_i.valid
                    << std::endl;
            }    
        }
    } // End loop reduced tracks i
}

