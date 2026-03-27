#ifndef middle_buffer_multififo_regionizer_ref_h
#define middle_buffer_multififo_regionizer_ref_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/multififo_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"
#include <memory>
#include <deque>

namespace l1ct {
  class MiddleBufferMultififoRegionizerEmulator : public RegionizerEmulator {
  public:
    MiddleBufferMultififoRegionizerEmulator(unsigned int nclocks,
                                            unsigned int nbuffers,
                                            unsigned int etabufferDepth,
                                            unsigned int ntklinks,
                                            unsigned int nHCalLinks,
                                            unsigned int nECalLinks,
                                            unsigned int ntk,
                                            unsigned int ncalo,
                                            unsigned int nem,
                                            unsigned int nmu,
                                            bool streaming,
                                            unsigned int outii,
                                            unsigned int pauseii,
                                            bool useAlsoVtxCoords,
                                            bool tmux6GCTinput = false);
    // note: this one will work only in CMSSW
    MiddleBufferMultififoRegionizerEmulator(const edm::ParameterSet& iConfig);

    ~MiddleBufferMultififoRegionizerEmulator() override;

    static edm::ParameterSetDescription getParameterSetDescription();

    void initSectorsAndRegions(const RegionizerDecodedInputs& in, const std::vector<PFInputRegion>& out) override;

    void run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) override;

    // link emulation from decoded inputs (for simulation)
    void fillLinks(unsigned int iclock,
                   const RegionizerDecodedInputs& in,
                   std::vector<l1ct::TkObjEmu>& links,
                   std::vector<bool>& valid);
    void fillLinks(unsigned int iclock,
                   const RegionizerDecodedInputs& in,
                   std::vector<l1ct::CommonCaloObjEmu>& links,
                   std::vector<bool>& valid);
    void fillLinks(unsigned int iclock,
                   const RegionizerDecodedInputs& in,
                   std::vector<l1ct::HadCaloObjEmu>& links,
                   std::vector<bool>& valid);
    void fillLinks(unsigned int iclock,
                   const RegionizerDecodedInputs& in,
                   std::vector<l1ct::EmCaloObjEmu>& links,
                   std::vector<bool>& valid);
    void fillLinks(unsigned int iclock,
                   const RegionizerDecodedInputs& in,
                   std::vector<l1ct::MuObjEmu>& links,
                   std::vector<bool>& valid);
    template <typename T>
    void fillLinks(unsigned int iclock, const RegionizerDecodedInputs& in, std::vector<T>& links) {
      std::vector<bool> unused;
      fillLinks(iclock, in, links, unused);
    }

    void destream(int iclock,
                  const std::vector<l1ct::TkObjEmu>& tk_out,
                  const std::vector<l1ct::EmCaloObjEmu>& em_out,
                  const std::vector<l1ct::HadCaloObjEmu>& calo_out,
                  const std::vector<l1ct::MuObjEmu>& mu_out,
                  PFInputRegion& out);

    // clock-cycle emulation
    bool step(bool newEvent,
              const std::vector<l1ct::TkObjEmu>& links_tk,
              const std::vector<l1ct::HadCaloObjEmu>& links_hadCalo,
              const std::vector<l1ct::EmCaloObjEmu>& links_emCalo,
              const std::vector<l1ct::CommonCaloObjEmu>& links_commonCalo,
              const std::vector<l1ct::MuObjEmu>& links_mu,
              std::vector<l1ct::TkObjEmu>& out_tk,
              std::vector<l1ct::HadCaloObjEmu>& out_hadCalo,
              std::vector<l1ct::EmCaloObjEmu>& out_emCalo,
              std::vector<l1ct::MuObjEmu>& out_mu,
              bool /*unused*/);

    template <typename TEmu, typename TFw>
    void toFirmware(const std::vector<TEmu>& emu, TFw fw[]) {
      for (unsigned int i = 0, n = emu.size(); i < n; ++i) {
        fw[i] = emu[i];
      }
    }

    void reset();

    template <typename T>
    static void encode(const T& from, l1ct::CommonCaloObjEmu& to) {
      to.convertFrom(from);
      to.src = from.src;
    }

    void convert_GCTinput_tmux(const RegionizerDecodedInputs& in_tm6, RegionizerDecodedInputs& in_tm18) const;
    void init_GCT_tmux18sectors(std::vector<l1ct::DetectorSector<l1ct::HadCaloObjEmu>>& gct_tmux18_hadcalo,
                                std::vector<l1ct::DetectorSector<l1ct::EmCaloObjEmu>>& gct_tmux18_emcalo) const;

  protected:
    const unsigned int NTK_SECTORS, NCALO_SECTORS;
    const unsigned int NTK_LINKS, HCAL_LINKS, ECAL_LINKS, NMU_LINKS;
    unsigned int nclocks_, nbuffers_, etabuffer_depth_, ntk_, ncalo_, nem_, nmu_, outii_, pauseii_, nregions_pre_,
        nregions_post_;
    bool streaming_;
    bool init_;
    unsigned int iclock_;
    bool tmux6GCTinput_;
    std::vector<l1ct::PFRegionEmu> mergedRegions_, outputRegions_;
    multififo_regionizer::Regionizer<l1ct::TkObjEmu> tkRegionizerPre_, tkRegionizerPost_;
    multififo_regionizer::Regionizer<l1ct::CommonCaloObjEmu> commonCaloRegionizerPre_;
    multififo_regionizer::Regionizer<l1ct::HadCaloObjEmu> hadCaloRegionizerPre_, hadCaloRegionizerPost_;
    multififo_regionizer::Regionizer<l1ct::EmCaloObjEmu> emCaloRegionizerPre_, emCaloRegionizerPost_;
    multififo_regionizer::Regionizer<l1ct::MuObjEmu> muRegionizerPre_, muRegionizerPost_;
    std::vector<l1ct::multififo_regionizer::Route> tkRoutes_, caloRoutes_, emCaloRoutes_, muRoutes_;
    std::vector<l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::TkObjEmu>> tkBuffers_;
    std::vector<l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::HadCaloObjEmu>> hadCaloBuffers_;
    std::vector<l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::EmCaloObjEmu>> emCaloBuffers_;
    std::vector<l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::MuObjEmu>> muBuffers_;
    std::vector<PFRegionEmu> gct_slr_regions_;
    std::vector<l1ct::DetectorSector<l1ct::HadCaloObjEmu>> gct_tmux18_hadcalo_;
    std::vector<l1ct::DetectorSector<l1ct::EmCaloObjEmu>> gct_tmux18_emcalo_;

    template <typename T>
    void fillCaloLinks_(unsigned int iclock,
                        const std::vector<DetectorSector<T>>& in,
                        std::vector<T>& links,
                        std::vector<bool>& valid);

    void fillSharedCaloLinks(unsigned int iclock,
                             const std::vector<DetectorSector<l1ct::EmCaloObjEmu>>& em_in,
                             const std::vector<DetectorSector<l1ct::HadCaloObjEmu>>& had_in,
                             std::vector<l1ct::CommonCaloObjEmu>& links,
                             std::vector<bool>& valid);

    void run_worker(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out);
    void init_GCT_slrs(std::vector<l1ct::PFRegionEmu>& gct_slr_regions) const;

    template <typename T>
    void convert_GCTsector_tmux(const std::vector<l1ct::DetectorSector<T>>& tm6_sectors,
                                std::vector<l1ct::DetectorSector<T>>& tm18_sectors) const {
      static constexpr std::array<unsigned int, 12> gct_slr_tmux18sector_mapping = {0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2};
      for (const auto& sec : tm6_sectors) {
        // std::cout << "Processing TMUX6 hadcalo sector: " << isec << " #cl: " << sec.obj.size() << std::endl;
        for (auto cl : sec.obj) {
          if (cl.hwPt == 0)
            continue;  // skip empty objects
          glbeta_t gl_hweta = sec.region.hwGlbEta(cl.hwEta);
          glbphi_t gl_hwphi = sec.region.hwGlbPhi(cl.hwPhi);
          for (unsigned int islr = 0; islr < gct_slr_regions_.size(); ++islr) {
            if (gct_slr_regions_[islr].containsHw(gl_hweta, gl_hwphi)) {
              auto itmux18 = gct_slr_tmux18sector_mapping[islr];
              cl.hwEta = l1ct::Scales::makeEta(tm18_sectors[itmux18].region.localEta(sec.region.floatGlbEtaOf(cl)));
              cl.hwPhi = l1ct::Scales::makePhi(tm18_sectors[itmux18].region.localPhi(sec.region.floatGlbPhiOf(cl)));
              tm18_sectors[itmux18].obj.push_back(cl);
              break;
            }
          }
        }
      }
    };
  };
}  // namespace l1ct

#endif
