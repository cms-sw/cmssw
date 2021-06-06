#include "L1Trigger/Phase2L1ParticleFlow/interface/BitwisePFAlgo.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "L1Trigger/Phase2L1ParticleFlow/src/dbgPrintf.h"

//#define REG_HGCal
#include "ref/pfalgo2hgc_ref.h"
#include "ref/pfalgo3_ref.h"
#include "utils/DiscretePF2Firmware.h"
#include "utils/Firmware2DiscretePF.h"

using namespace l1tpf_impl;

BitwisePFAlgo::BitwisePFAlgo(const edm::ParameterSet &iConfig) : PFAlgoBase(iConfig), config_(nullptr) {
  const edm::ParameterSet &bitwiseConfig = iConfig.getParameter<edm::ParameterSet>("bitwiseConfig");
  const std::string &algo = iConfig.getParameter<std::string>("bitwiseAlgo");
  debug_ = iConfig.getUntrackedParameter<int>("debugBitwisePFAlgo", iConfig.getUntrackedParameter<int>("debug", 0));
  if (algo == "pfalgo3") {
    algo_ = AlgoChoice::algo3;
    config_ = std::make_shared<pfalgo3_config>(bitwiseConfig.getParameter<uint32_t>("NTRACK"),
                                               bitwiseConfig.getParameter<uint32_t>("NEMCALO"),
                                               bitwiseConfig.getParameter<uint32_t>("NCALO"),
                                               bitwiseConfig.getParameter<uint32_t>("NMU"),
                                               bitwiseConfig.getParameter<uint32_t>("NPHOTON"),
                                               bitwiseConfig.getParameter<uint32_t>("NSELCALO"),
                                               bitwiseConfig.getParameter<uint32_t>("NALLNEUTRAL"),
                                               bitwiseConfig.getParameter<uint32_t>("DR2MAX_TK_MU"),
                                               bitwiseConfig.getParameter<uint32_t>("DR2MAX_TK_EM"),
                                               bitwiseConfig.getParameter<uint32_t>("DR2MAX_EM_CALO"),
                                               bitwiseConfig.getParameter<uint32_t>("DR2MAX_TK_CALO"),
                                               bitwiseConfig.getParameter<uint32_t>("TK_MAXINVPT_LOOSE"),
                                               bitwiseConfig.getParameter<uint32_t>("TK_MAXINVPT_TIGHT"));
  } else if (algo == "pfalgo2hgc") {
    algo_ = AlgoChoice::algo2hgc;
    config_ = std::make_shared<pfalgo_config>(bitwiseConfig.getParameter<uint32_t>("NTRACK"),
                                              bitwiseConfig.getParameter<uint32_t>("NCALO"),
                                              bitwiseConfig.getParameter<uint32_t>("NMU"),
                                              bitwiseConfig.getParameter<uint32_t>("NSELCALO"),
                                              bitwiseConfig.getParameter<uint32_t>("DR2MAX_TK_MU"),
                                              bitwiseConfig.getParameter<uint32_t>("DR2MAX_TK_CALO"),
                                              bitwiseConfig.getParameter<uint32_t>("TK_MAXINVPT_LOOSE"),
                                              bitwiseConfig.getParameter<uint32_t>("TK_MAXINVPT_TIGHT"));
  } else {
    throw cms::Exception("Configuration", "Unsupported bitwiseAlgo " + algo);
  }
}

BitwisePFAlgo::~BitwisePFAlgo() {}

void BitwisePFAlgo::runPF(Region &r) const {
  initRegion(r);

  std::unique_ptr<HadCaloObj[]> calo(new HadCaloObj[config_->nCALO]);
  std::unique_ptr<TkObj[]> track(new TkObj[config_->nTRACK]);
  std::unique_ptr<MuObj[]> mu(new MuObj[config_->nMU]);
  std::unique_ptr<PFChargedObj[]> outch(new PFChargedObj[config_->nTRACK]);
  std::unique_ptr<PFNeutralObj[]> outne(new PFNeutralObj[config_->nSELCALO]);
  std::unique_ptr<PFChargedObj[]> outmu(new PFChargedObj[config_->nMU]);

  dpf2fw::convert(config_->nTRACK, r.track, track.get());
  dpf2fw::convert(config_->nCALO, r.calo, calo.get());
  dpf2fw::convert(config_->nMU, r.muon, mu.get());

  if (debug_) {
    dbgPrintf(
        "BitwisePF\nBitwisePF region eta [ %+5.2f , %+5.2f ], phi [ %+5.2f , %+5.2f ], fiducial eta [ %+5.2f , %+5.2f "
        "], phi [ %+5.2f , %+5.2f ], algo = %d\n",
        r.etaMin - r.etaExtra,
        r.etaMax + r.etaExtra,
        r.phiCenter - r.phiHalfWidth - r.phiExtra,
        r.phiCenter + r.phiHalfWidth + r.phiExtra,
        r.etaMin,
        r.etaMax,
        r.phiCenter - r.phiHalfWidth,
        r.phiCenter + r.phiHalfWidth,
        static_cast<int>(algo_));
    dbgPrintf("BitwisePF \t N(track) %3lu   N(em) %3lu   N(calo) %3lu   N(mu) %3lu\n",
              r.track.size(),
              r.emcalo.size(),
              r.calo.size(),
              r.muon.size());
    for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
      const auto &tk = r.track[itk];
      dbgPrintf(
          "BitwisePF \t track %3d: pt %7.2f +- %5.2f  vtx eta %+5.2f  vtx phi %+5.2f  calo eta %+5.2f  calo phi %+5.2f "
          " fid %1d  calo ptErr %7.2f stubs %2d chi2 %7.1f\n",
          itk,
          tk.floatPt(),
          tk.floatPtErr(),
          tk.floatVtxEta(),
          tk.floatVtxPhi(),
          tk.floatEta(),
          tk.floatPhi(),
          int(r.fiducialLocal(tk.floatEta(), tk.floatPhi())),
          tk.floatCaloPtErr(),
          int(tk.hwStubs),
          tk.hwChi2 * 0.1f);
    }
    for (int iem = 0, nem = r.emcalo.size(); iem < nem; ++iem) {
      const auto &em = r.emcalo[iem];
      dbgPrintf(
          "BitwisePF \t EM    %3d: pt %7.2f +- %5.2f  vtx eta %+5.2f  vtx phi %+5.2f  calo eta %+5.2f  calo phi %+5.2f "
          " fid %1d  calo ptErr %7.2f\n",
          iem,
          em.floatPt(),
          em.floatPtErr(),
          em.floatEta(),
          em.floatPhi(),
          em.floatEta(),
          em.floatPhi(),
          int(r.fiducialLocal(em.floatEta(), em.floatPhi())),
          em.floatPtErr());
    }
    for (int ic = 0, nc = r.calo.size(); ic < nc; ++ic) {
      auto &calo = r.calo[ic];
      dbgPrintf(
          "BitwisePF \t calo  %3d: pt %7.2f +- %5.2f  vtx eta %+5.2f  vtx phi %+5.2f  calo eta %+5.2f  calo phi %+5.2f "
          " fid %1d  calo ptErr %7.2f em pt %7.2f \n",
          ic,
          calo.floatPt(),
          calo.floatPtErr(),
          calo.floatEta(),
          calo.floatPhi(),
          calo.floatEta(),
          calo.floatPhi(),
          int(r.fiducialLocal(calo.floatEta(), calo.floatPhi())),
          calo.floatPtErr(),
          calo.floatEmPt());
    }
    for (int im = 0, nm = r.muon.size(); im < nm; ++im) {
      auto &mu = r.muon[im];
      dbgPrintf(
          "BitwisePF \t muon  %3d: pt %7.2f           vtx eta %+5.2f  vtx phi %+5.2f  calo eta %+5.2f  calo phi %+5.2f "
          " fid %1d \n",
          im,
          mu.floatPt(),
          mu.floatEta(),
          mu.floatPhi(),
          mu.floatEta(),
          mu.floatPhi(),
          int(r.fiducialLocal(mu.floatEta(), mu.floatPhi())));
    }
  }
  switch (algo_) {
    case AlgoChoice::algo3: {
      pfalgo3_config *config3 = static_cast<pfalgo3_config *>(config_.get());
      std::unique_ptr<EmCaloObj[]> emcalo(new EmCaloObj[config3->nEMCALO]);
      std::unique_ptr<PFNeutralObj[]> outpho(new PFNeutralObj[config3->nPHOTON]);

      dpf2fw::convert(config3->nEMCALO, r.emcalo, emcalo.get());
      pfalgo3_ref(*config3,
                  emcalo.get(),
                  calo.get(),
                  track.get(),
                  mu.get(),
                  outch.get(),
                  outpho.get(),
                  outne.get(),
                  outmu.get(),
                  debug_);

      fw2dpf::convert(config3->nTRACK, outch.get(), r.track, r.pf);  // FIXME works only with a 1-1 mapping
      fw2dpf::convert(config3->nPHOTON, outpho.get(), r.pf);
      fw2dpf::convert(config3->nSELCALO, outne.get(), r.pf);
    } break;
    case AlgoChoice::algo2hgc: {
      pfalgo2hgc_ref(*config_, calo.get(), track.get(), mu.get(), outch.get(), outne.get(), outmu.get(), debug_);
      fw2dpf::convert(config_->nTRACK, outch.get(), r.track, r.pf);  // FIXME works only with a 1-1 mapping
      fw2dpf::convert(config_->nSELCALO, outne.get(), r.pf);
    } break;
  };

  if (debug_) {
    dbgPrintf("BitwisePF \t Output N(ch) %3u/%3u   N(nh) %3u/%3u   N(ph) %3u/%u   [all/fiducial]\n",
              r.nOutput(l1tpf_impl::Region::charged_type, false, false),
              r.nOutput(l1tpf_impl::Region::charged_type, false, true),
              r.nOutput(l1tpf_impl::Region::neutral_hadron_type, false, false),
              r.nOutput(l1tpf_impl::Region::neutral_hadron_type, false, true),
              r.nOutput(l1tpf_impl::Region::photon_type, false, false),
              r.nOutput(l1tpf_impl::Region::photon_type, false, true));
    for (int ipf = 0, npf = r.pf.size(); ipf < npf; ++ipf) {
      const auto &pf = r.pf[ipf];
      dbgPrintf(
          "BitwisePF \t pf    %3d: pt %7.2f pid %d   vtx eta %+5.2f  vtx phi %+5.2f  calo eta %+5.2f  calo phi %+5.2f  "
          "fid %1d\n",
          ipf,
          pf.floatPt(),
          int(pf.hwId),
          pf.floatVtxEta(),
          pf.floatVtxPhi(),
          pf.floatEta(),
          pf.floatPhi(),
          int(r.fiducialLocal(pf.floatEta(), pf.floatPhi())));
    }
  }
}
