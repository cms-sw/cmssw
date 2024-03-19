#ifndef L1Trigger_L1TMuonEndCapPhase2_EMTFModel_h
#define L1Trigger_L1TMuonEndCapPhase2_EMTFModel_h

#include <array>
#include <vector>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"

namespace emtf::phase2 {

  // Forward Declarations
  namespace model {
    namespace zones {
      namespace hitmap {
        struct chamber_t;
        struct site_t;
        typedef std::vector<site_t> row_t;
      }  // namespace hitmap

      namespace pattern {
        struct row_t;
      }

      typedef std::vector<hitmap::row_t> hitmap_t;
      typedef std::vector<pattern::row_t> pattern_t;
      typedef std::vector<unsigned int> quality_lut_t;
    }  // namespace zones

    namespace theta_medians {
      struct site_t;
      typedef std::vector<site_t> group_t;
    }  // namespace theta_medians

    namespace reduced_sites {
      struct reduced_site_t;
    }

    struct zone_t;
    struct feature_t;

    typedef std::vector<theta_medians::group_t> theta_median_t;
    typedef std::vector<reduced_sites::reduced_site_t> reduced_sites_t;
  }  // namespace model

  // Definitions
  class EMTFModel {
  public:
    EMTFModel(const EMTFContext&);

    ~EMTFModel();

    std::vector<model::zone_t> zones_;
    std::vector<model::feature_t> features_;
    std::vector<model::theta_median_t> theta_medians_;
    model::reduced_sites_t reduced_sites_;

  private:
    const EMTFContext& context_;
  };

  namespace model {
    // Define Zone Structs
    struct zone_t {
      zones::hitmap_t hitmap;

      // Prompt
      std::vector<zones::pattern_t> prompt_patterns;
      zones::quality_lut_t prompt_quality_lut;

      // Displaced
      std::vector<zones::pattern_t> disp_patterns;
      zones::quality_lut_t disp_quality_lut;
    };

    namespace zones {
      namespace hitmap {
        struct site_t {
          site_id_t id;
          std::vector<chamber_t> chambers;
        };

        struct chamber_t {
          unsigned int id;
          unsigned int begin;
          unsigned int end;
        };
      }  // namespace hitmap

      namespace pattern {
        struct row_t {
          unsigned int begin;
          unsigned int center;
          unsigned int end;
        };
      }  // namespace pattern
    }    // namespace zones

    // Define Feature Structs
    struct feature_t {
      feature_id_t id;
      std::vector<site_id_t> sites;
    };

    // Define Theta Median Structs
    namespace theta_medians {
      struct site_t {
        site_id_t id;
        theta_id_t theta_id;
      };
    }  // namespace theta_medians

    // Define Reduced Site Structs
    namespace reduced_sites {
      struct reduced_site_t {
        reduced_site_id_t id;
        std::vector<site_id_t> trk_sites;
      };
    }  // namespace reduced_sites
  }    // namespace model
}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_EMTFModel_h not defined
