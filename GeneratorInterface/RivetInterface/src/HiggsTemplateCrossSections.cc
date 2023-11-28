#ifndef TRUTHRIVETTOOLS_HIGGSTEMPLATECROSSSECTIONS_CC
#define TRUTHRIVETTOOLS_HIGGSTEMPLATECROSSSECTIONS_CC

// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Particle.hh"
#include "Rivet/Projections/FastJets.hh"

// Definition of the StatusCode and Category enums
//#include "HiggsTemplateCrossSections.h"
#include "SimDataFormats/HTXS/interface/HiggsTemplateCrossSections.h"  //

#include <atomic>

namespace Rivet {

  /// @class HiggsTemplateCrossSections
  /// @brief  Rivet routine for classifying MC events according to the Higgs template cross section categories
  /// @author Jim Lacey (DESY) <james.lacey@cern.ch,jlacey@desy.de>
  /// @author Dag Gillberg (Carleton University) <dag.gillberg@cern.ch>
  class HiggsTemplateCrossSections : public Analysis {
  public:
    // Constructor
    HiggsTemplateCrossSections() : Analysis("HiggsTemplateCrossSections"), m_HiggsProdMode(HTXS::UNKNOWN) {}

  public:
    /// @name Utility methods
    /// Methods to identify the Higgs boson and
    /// associated vector boson and to build jets
    /// @{

    /// follow a "propagating" particle and return its last instance
    Particle getLastInstance(Particle ptcl) {
      if (ptcl.genParticle()->end_vertex()) {
        if (!hasChild(ptcl.genParticle(), ptcl.pid()))
          return ptcl;
        else
          return getLastInstance(ptcl.children()[0]);
      }
      return ptcl;
    }

    /// @brief Whether particle p originate from any of the ptcls
    bool originateFrom(const Particle &p, const Particles &ptcls) {
      const ConstGenVertexPtr prodVtx = p.genParticle()->production_vertex();
      if (prodVtx == nullptr)
        return false;
      // for each ancestor, check if it matches any of the input particles
      for (const auto &ancestor : HepMCUtils::particles(prodVtx, HepMC::ancestors)) {
        for (const auto &part : ptcls)
          if (ancestor == part.genParticle())
            return true;
      }
      // if we get here, no ancetor matched any input particle
      return false;
    }

    /// @brief Whether particle p originates from p2
    bool originateFrom(const Particle &p, const Particle &p2) {
      Particles ptcls = {p2};
      return originateFrom(p, ptcls);
    }

    /// @brief Checks whether the input particle has a child with a given PDGID
    bool hasChild(const ConstGenParticlePtr &ptcl, int pdgID) {
      for (const auto &child : Particle(*ptcl).children()) {
        if (child.pid() == pdgID) {
          return true;
        }
      }
      return false;
    }

    /// @brief Checks whether the input particle has a parent with a given PDGID
    bool hasParent(const ConstGenParticlePtr &ptcl, int pdgID) {
      for (auto parent : HepMCUtils::particles(ptcl->production_vertex(), HepMC::parents))
        if (parent->pdg_id() == pdgID)
          return true;
      return false;
    }

    /// @brief Return true is particle decays to quarks
    bool quarkDecay(const Particle &p) {
      for (const auto &child : p.children()) {
        if (PID::isQuark(child.pid())) {
          return true;
        }
      }
      return false;
    }

    /// @brief Return true if particle decays to charged leptons.
    bool ChLeptonDecay(const Particle &p) {
      for (const auto &child : p.children()) {
        if (PID::isChLepton(child.pid())) {
          return true;
        }
      }
      return false;
    }

    /// @brief Returns the classification object with the error code set.
    ///        Prints an warning message, and keeps track of number of errors
    HiggsClassification error(HiggsClassification &cat,
                              HTXS::ErrorCode err,
                              std::string msg = "",
                              int NmaxWarnings = 20) {
      // Set the error, and keep statistics
      cat.errorCode = err;
      ++m_errorCount[err];

      // Print warning message to the screen/log
      static std::atomic<int> Nwarnings{0};
      if (!msg.empty() && ++Nwarnings < NmaxWarnings)
        MSG_WARNING(msg);

      return cat;
    }
    /// @}

    /// @brief Main classificaion method.
    HiggsClassification classifyEvent(const Event &event, const HTXS::HiggsProdMode prodMode) {
      if (m_HiggsProdMode == HTXS::UNKNOWN)
        m_HiggsProdMode = prodMode;

      // the classification object
      HiggsClassification cat;
      cat.prodMode = prodMode;
      cat.errorCode = HTXS::UNDEFINED;
      cat.stage0_cat = HTXS::Stage0::UNKNOWN;
      cat.stage1_cat_pTjet25GeV = HTXS::Stage1::UNKNOWN;
      cat.stage1_cat_pTjet30GeV = HTXS::Stage1::UNKNOWN;
      cat.stage1_1_cat_pTjet25GeV = HTXS::Stage1_1::UNKNOWN;
      cat.stage1_1_cat_pTjet30GeV = HTXS::Stage1_1::UNKNOWN;
      cat.stage1_1_fine_cat_pTjet25GeV = HTXS::Stage1_1_Fine::UNKNOWN;
      cat.stage1_1_fine_cat_pTjet30GeV = HTXS::Stage1_1_Fine::UNKNOWN;
      cat.stage1_2_cat_pTjet25GeV = HTXS::Stage1_2::UNKNOWN;
      cat.stage1_2_cat_pTjet30GeV = HTXS::Stage1_2::UNKNOWN;
      cat.stage1_2_fine_cat_pTjet25GeV = HTXS::Stage1_2_Fine::UNKNOWN;
      cat.stage1_2_fine_cat_pTjet30GeV = HTXS::Stage1_2_Fine::UNKNOWN;

      if (prodMode == HTXS::UNKNOWN)
        return error(cat,
                     HTXS::PRODMODE_DEFINED,
                     "Unkown Higgs production mechanism. Cannot classify event."
                     " Classification for all events will most likely fail.");

      /*****
       * Step 1. 
       *  Idenfify the Higgs boson and the hard scatter vertex
       *  There should be only one of each.
       */

      ConstGenVertexPtr HSvtx = event.genEvent()->signal_process_vertex();
      int Nhiggs = 0;
      for (const ConstGenParticlePtr ptcl : HepMCUtils::particles(event.genEvent())) {
        // a) Reject all non-Higgs particles
        if (!PID::isHiggs(ptcl->pdg_id()))
          continue;
        // b) select only the final Higgs boson copy, prior to decay
        if (ptcl->end_vertex() && !hasChild(ptcl, PID::HIGGS)) {
          cat.higgs = Particle(ptcl);
          ++Nhiggs;
        }
        // c) if HepMC::signal_proces_vertex is missing
        //    set hard-scatter vertex based on first Higgs boson
        if (HSvtx == nullptr && ptcl->production_vertex() && !hasParent(ptcl, PID::HIGGS))
          HSvtx = ptcl->production_vertex();
      }

      // Make sure things are in order so far
      if (Nhiggs != 1)
        return error(cat,
                     HTXS::HIGGS_IDENTIFICATION,
                     "Current event has " + std::to_string(Nhiggs) + " Higgs bosons. There must be only one.");
      if (cat.higgs.children().size() < 2)
        return error(cat, HTXS::HIGGS_DECAY_IDENTIFICATION, "Could not identify Higgs boson decay products.");

      if (HSvtx == nullptr)
        return error(cat, HTXS::HS_VTX_IDENTIFICATION, "Cannot find hard-scatter vertex of current event.");

      /*****
       * Step 2. 
       *   Identify associated vector bosons
       */

      // Find associated vector bosons
      bool is_uncatdV = false;
      Particles uncatV_decays;
      FourMomentum uncatV_p4(0, 0, 0, 0);
      FourVector uncatV_v4(0, 0, 0, 0);
      int nWs = 0, nZs = 0;
      if (isVH(prodMode)) {
        for (auto ptcl : HepMCUtils::particles(HSvtx, HepMC::children)) {
          if (PID::isW(ptcl->pdg_id())) {
            ++nWs;
            cat.V = Particle(ptcl);
          }
          if (PID::isZ(ptcl->pdg_id())) {
            ++nZs;
            cat.V = Particle(ptcl);
          }
        }
        if (nWs + nZs > 0)
          cat.V = getLastInstance(cat.V);
        else {
          for (auto ptcl : HepMCUtils::particles(HSvtx, HepMC::children)) {
            if (!PID::isHiggs(ptcl->pdg_id())) {
              uncatV_decays += Particle(ptcl);
              uncatV_p4 += Particle(ptcl).momentum();
              // uncatV_v4 += Particle(ptcl).origin();
            }
          }
          // is_uncatdV = true; cat.V = Particle(24,uncatV_p4,uncatV_v4);
          is_uncatdV = true;
          cat.V = Particle(24, uncatV_p4);
        }
      }

      if (!is_uncatdV) {
        if (isVH(prodMode) && !cat.V.genParticle()->end_vertex())
          return error(cat, HTXS::VH_DECAY_IDENTIFICATION, "Vector boson does not decay!");

        if (isVH(prodMode) && cat.V.children().size() < 2)
          return error(cat, HTXS::VH_DECAY_IDENTIFICATION, "Vector boson does not decay!");

        if ((prodMode == HTXS::WH && (nZs > 0 || nWs != 1)) ||
            ((prodMode == HTXS::QQ2ZH || prodMode == HTXS::GG2ZH) && (nZs != 1 || nWs > 0)))
          return error(cat,
                       HTXS::VH_IDENTIFICATION,
                       "Found " + std::to_string(nWs) + " W-bosons and " + std::to_string(nZs) +
                           " Z-bosons. Inconsitent with VH expectation.");
      }

      // Find and store the W-bosons from ttH->WbWbH
      Particles Ws;
      if (prodMode == HTXS::TTH || prodMode == HTXS::TH) {
        // loop over particles produced in hard-scatter vertex
        for (auto ptcl : HepMCUtils::particles(HSvtx, HepMC::children)) {
          if (!PID::isTop(ptcl->pdg_id()))
            continue;
          Particle top = getLastInstance(Particle(ptcl));
          if (top.genParticle()->end_vertex())
            for (const auto &child : top.children())
              if (PID::isW(child.pid()))
                Ws += getLastInstance(child);
        }
      }

      // Make sure result make sense
      if ((prodMode == HTXS::TTH && Ws.size() < 2) || (prodMode == HTXS::TH && Ws.empty()))
        return error(cat, HTXS::TOP_W_IDENTIFICATION, "Failed to identify W-boson(s) from t-decay!");

      /*****
       * Step 3.
       *   Build jets
       *   Make sure all stable particles are present
       */

      // Create a list of the vector bosons that decay leptonically
      // Either the vector boson produced in association with the Higgs boson,
      // or the ones produced from decays of top quarks produced with the Higgs
      Particles leptonicVs;
      if (!is_uncatdV) {
        if (isVH(prodMode) && !quarkDecay(cat.V))
          leptonicVs += cat.V;
      } else
        leptonicVs = uncatV_decays;
      for (const auto &W : Ws)
        if (W.genParticle()->end_vertex() && !quarkDecay(W))
          leptonicVs += W;

      // Obtain all stable, final-state particles
      const Particles FS = apply<FinalState>(event, "FS").particles();
      Particles hadrons;
      FourMomentum sum(0, 0, 0, 0), vSum(0, 0, 0, 0), hSum(0, 0, 0, 0);
      for (const Particle &p : FS) {
        // Add up the four momenta of all stable particles as a cross check
        sum += p.momentum();
        // ignore particles from the Higgs boson
        if (originateFrom(p, cat.higgs)) {
          hSum += p.momentum();
          continue;
        }
        // Cross-check the V decay products for VH
        if (isVH(prodMode) && !is_uncatdV && originateFrom(p, Ws))
          vSum += p.momentum();
        // ignore final state particles from leptonic V decays
        if (!leptonicVs.empty() && originateFrom(p, leptonicVs))
          continue;
        // All particles reaching here are considered hadrons and will be used to build jets
        hadrons += p;
      }

      cat.p4decay_higgs = hSum;
      cat.p4decay_V = vSum;

      FinalState fps_temp;
      FastJets jets(fps_temp, FastJets::ANTIKT, 0.4);
      jets.calc(hadrons);

      cat.jets25 = jets.jetsByPt(Cuts::pT > 25.0);
      cat.jets30 = jets.jetsByPt(Cuts::pT > 30.0);

      // check that four mometum sum of all stable particles satisfies momentum consevation
      /*
      if ( sum.pt()>0.1 )
	return error(cat,HTXS::MOMENTUM_CONSERVATION,"Four vector sum does not amount to pT=0, m=E=sqrt(s), but pT="+
		     std::to_string(sum.pt())+" GeV and m = "+std::to_string(sum.mass())+" GeV");
*/
      // check if V-boson was not included in the event record but decay particles were
      // EFT contact interaction: return UNKNOWN for category but set all event/particle kinematics
      if (is_uncatdV)
        return error(cat, HTXS::VH_IDENTIFICATION, "Failed to identify associated V-boson!");

      /*****
       * Step 4.
       *   Classify and save output
       */

      // Apply the categorization categorization
      cat.isZ2vvDecay = false;
      if ((prodMode == HTXS::GG2ZH || prodMode == HTXS::QQ2ZH) && !quarkDecay(cat.V) && !ChLeptonDecay(cat.V))
        cat.isZ2vvDecay = true;
      cat.stage0_cat = getStage0Category(prodMode, cat.higgs, cat.V);
      cat.stage1_cat_pTjet25GeV = getStage1Category(prodMode, cat.higgs, cat.jets25, cat.V);
      cat.stage1_cat_pTjet30GeV = getStage1Category(prodMode, cat.higgs, cat.jets30, cat.V);
      cat.stage1_1_cat_pTjet25GeV = getStage1_1_Category(prodMode, cat.higgs, cat.jets25, cat.V);
      cat.stage1_1_cat_pTjet30GeV = getStage1_1_Category(prodMode, cat.higgs, cat.jets30, cat.V);
      cat.stage1_1_fine_cat_pTjet25GeV = getStage1_1_Fine_Category(prodMode, cat.higgs, cat.jets25, cat.V);
      cat.stage1_1_fine_cat_pTjet30GeV = getStage1_1_Fine_Category(prodMode, cat.higgs, cat.jets30, cat.V);
      cat.stage1_2_cat_pTjet25GeV = getStage1_2_Category(prodMode, cat.higgs, cat.jets25, cat.V);
      cat.stage1_2_cat_pTjet30GeV = getStage1_2_Category(prodMode, cat.higgs, cat.jets30, cat.V);
      cat.stage1_2_fine_cat_pTjet25GeV = getStage1_2_Fine_Category(prodMode, cat.higgs, cat.jets25, cat.V);
      cat.stage1_2_fine_cat_pTjet30GeV = getStage1_2_Fine_Category(prodMode, cat.higgs, cat.jets30, cat.V);

      cat.errorCode = HTXS::SUCCESS;
      ++m_errorCount[HTXS::SUCCESS];
      ++m_sumevents;

      return cat;
    }

    /// @name Categorization methods
    /// Methods to assign the truth category based
    /// on the identified Higgs boson and associated
    /// vector bosons and/or reconstructed jets
    /// @{

    /// @brief Return bin index of x given the provided bin edges. 0=first bin, -1=underflow bin.
    int getBin(double x, std::vector<double> bins) {
      for (size_t i = 1; i < bins.size(); ++i)
        if (x < bins[i])
          return i - 1;
      return bins.size() - 1;
    }

    /// @brief VBF topolog selection
    /// 0 = fail loose selction: m_jj > 400 GeV and Dy_jj > 2.8
    /// 1 pass loose, but fail additional cut pT(Hjj)<25. 2 pass tight selection
    int vbfTopology(const Jets &jets, const Particle &higgs) {
      if (jets.size() < 2)
        return 0;
      const FourMomentum &j1 = jets[0].momentum(), &j2 = jets[1].momentum();
      bool VBFtopo = (j1 + j2).mass() > 400.0 && std::abs(j1.rapidity() - j2.rapidity()) > 2.8;
      return VBFtopo ? (j1 + j2 + higgs.momentum()).pt() < 25 ? 2 : 1 : 0;
    }

    /// @brief VBF topology selection Stage1.1 and Stage1.2
    /// 0 = fail loose selection: m_jj > 350 GeV
    /// 1 pass loose, but fail additional cut pT(Hjj)<25. 2 pass pT(Hjj)>25 selection
    /// 3 pass tight (m_jj>700 GeV), but fail additional cut pT(Hjj)<25. 4 pass pT(Hjj)>25 selection
    int vbfTopology_Stage1_X(const Jets &jets, const Particle &higgs) {
      if (jets.size() < 2)
        return 0;
      const FourMomentum &j1 = jets[0].momentum(), &j2 = jets[1].momentum();
      double mjj = (j1 + j2).mass();
      if (mjj > 350 && mjj <= 700)
        return (j1 + j2 + higgs.momentum()).pt() < 25 ? 1 : 2;
      else if (mjj > 700)
        return (j1 + j2 + higgs.momentum()).pt() < 25 ? 3 : 4;
      else
        return 0;
    }

    /// @brief VBF topology selection for Stage1.1 and Stage 1.2 Fine
    /// 0 = fail loose selection: m_jj > 350 GeV
    /// 1 pass loose, but fail additional cut pT(Hjj)<25. 2 pass pT(Hjj)>25 selection
    /// 3 pass 700<m_jj<1000 GeV, but fail additional cut pT(Hjj)<25. 4 pass pT(Hjj)>25 selection
    /// 5 pass 1000<m_jj<1500 GeV, but fail additional cut pT(Hjj)<25. 6 pass pT(Hjj)>25 selection
    /// 7 pass m_jj>1500 GeV, but fail additional cut pT(Hjj)<25. 8 pass pT(Hjj)>25 selection
    int vbfTopology_Stage1_X_Fine(const Jets &jets, const Particle &higgs) {
      if (jets.size() < 2)
        return 0;
      const FourMomentum &j1 = jets[0].momentum(), &j2 = jets[1].momentum();
      double mjj = (j1 + j2).mass();
      if (mjj > 350 && mjj <= 700)
        return (j1 + j2 + higgs.momentum()).pt() < 25 ? 1 : 2;
      else if (mjj > 700 && mjj <= 1000)
        return (j1 + j2 + higgs.momentum()).pt() < 25 ? 3 : 4;
      else if (mjj > 1000 && mjj <= 1500)
        return (j1 + j2 + higgs.momentum()).pt() < 25 ? 5 : 6;
      else if (mjj > 1500)
        return (j1 + j2 + higgs.momentum()).pt() < 25 ? 7 : 8;
      else
        return 0;
    }

    /// @brief Whether the Higgs is produced in association with a vector boson (VH)
    bool isVH(HTXS::HiggsProdMode p) { return p == HTXS::WH || p == HTXS::QQ2ZH || p == HTXS::GG2ZH; }

    /// @brief Stage-0 HTXS categorization
    HTXS::Stage0::Category getStage0Category(const HTXS::HiggsProdMode prodMode,
                                             const Particle &higgs,
                                             const Particle &V) {
      using namespace HTXS::Stage0;
      int ctrlHiggs = std::abs(higgs.rapidity()) < 2.5;
      // Special cases first, qq→Hqq
      if ((prodMode == HTXS::WH || prodMode == HTXS::QQ2ZH) && quarkDecay(V)) {
        return ctrlHiggs ? VH2HQQ : VH2HQQ_FWDH;
      } else if (prodMode == HTXS::GG2ZH && quarkDecay(V)) {
        return Category(HTXS::GGF * 10 + ctrlHiggs);
      }
      // General case after
      return Category(prodMode * 10 + ctrlHiggs);
    }

    /// @brief Stage-1 categorization
    HTXS::Stage1::Category getStage1Category(const HTXS::HiggsProdMode prodMode,
                                             const Particle &higgs,
                                             const Jets &jets,
                                             const Particle &V) {
      using namespace HTXS::Stage1;
      int Njets = jets.size(), ctrlHiggs = std::abs(higgs.rapidity()) < 2.5, fwdHiggs = !ctrlHiggs;
      double pTj1 = !jets.empty() ? jets[0].momentum().pt() : 0;
      int vbfTopo = vbfTopology(jets, higgs);

      // 1. GGF Stage 1 categories
      //    Following YR4 write-up: XXXXX
      if (prodMode == HTXS::GGF || (prodMode == HTXS::GG2ZH && quarkDecay(V))) {
        if (fwdHiggs)
          return GG2H_FWDH;
        if (Njets == 0)
          return GG2H_0J;
        else if (Njets == 1)
          return Category(GG2H_1J_PTH_0_60 + getBin(higgs.pt(), {0, 60, 120, 200}));
        else if (Njets >= 2) {
          // events with pT_H>200 get priority over VBF cuts
          if (higgs.pt() <= 200) {
            if (vbfTopo == 2)
              return GG2H_VBFTOPO_JET3VETO;
            else if (vbfTopo == 1)
              return GG2H_VBFTOPO_JET3;
          }
          // Njets >= 2jets without VBF topology
          return Category(GG2H_GE2J_PTH_0_60 + getBin(higgs.pt(), {0, 60, 120, 200}));
        }
      }
      // 2. Electroweak qq->Hqq Stage 1 categories
      else if (prodMode == HTXS::VBF || (isVH(prodMode) && quarkDecay(V))) {
        if (std::abs(higgs.rapidity()) > 2.5)
          return QQ2HQQ_FWDH;
        if (pTj1 > 200)
          return QQ2HQQ_PTJET1_GT200;
        if (vbfTopo == 2)
          return QQ2HQQ_VBFTOPO_JET3VETO;
        if (vbfTopo == 1)
          return QQ2HQQ_VBFTOPO_JET3;
        double mjj = jets.size() > 1 ? (jets[0].mom() + jets[1].mom()).mass() : 0;
        if (60 < mjj && mjj < 120)
          return QQ2HQQ_VH2JET;
        return QQ2HQQ_REST;
      }
      // 3. WH->Hlv categories
      else if (prodMode == HTXS::WH) {
        if (fwdHiggs)
          return QQ2HLNU_FWDH;
        else if (V.pt() < 150)
          return QQ2HLNU_PTV_0_150;
        else if (V.pt() > 250)
          return QQ2HLNU_PTV_GT250;
        // 150 < pTV/GeV < 250
        return jets.empty() ? QQ2HLNU_PTV_150_250_0J : QQ2HLNU_PTV_150_250_GE1J;
      }
      // 4. qq->ZH->llH categories
      else if (prodMode == HTXS::QQ2ZH) {
        if (fwdHiggs)
          return QQ2HLL_FWDH;
        else if (V.pt() < 150)
          return QQ2HLL_PTV_0_150;
        else if (V.pt() > 250)
          return QQ2HLL_PTV_GT250;
        // 150 < pTV/GeV < 250
        return jets.empty() ? QQ2HLL_PTV_150_250_0J : QQ2HLL_PTV_150_250_GE1J;
      }
      // 5. gg->ZH->llH categories
      else if (prodMode == HTXS::GG2ZH) {
        if (fwdHiggs)
          return GG2HLL_FWDH;
        if (V.pt() < 150)
          return GG2HLL_PTV_0_150;
        else if (jets.empty())
          return GG2HLL_PTV_GT150_0J;
        return GG2HLL_PTV_GT150_GE1J;
      }
      // 6.ttH,bbH,tH categories
      else if (prodMode == HTXS::TTH)
        return Category(TTH_FWDH + ctrlHiggs);
      else if (prodMode == HTXS::BBH)
        return Category(BBH_FWDH + ctrlHiggs);
      else if (prodMode == HTXS::TH)
        return Category(TH_FWDH + ctrlHiggs);
      return UNKNOWN;
    }

    /// @brief Stage-1.1 categorization
    HTXS::Stage1_1::Category getStage1_1_Category(const HTXS::HiggsProdMode prodMode,
                                                  const Particle &higgs,
                                                  const Jets &jets,
                                                  const Particle &V) {
      using namespace HTXS::Stage1_1;
      int Njets = jets.size(), ctrlHiggs = std::abs(higgs.rapidity()) < 2.5, fwdHiggs = !ctrlHiggs;
      int vbfTopo = vbfTopology_Stage1_X(jets, higgs);

      // 1. GGF Stage 1 categories
      //    Following YR4 write-up: XXXXX
      if (prodMode == HTXS::GGF || (prodMode == HTXS::GG2ZH && quarkDecay(V))) {
        if (fwdHiggs)
          return GG2H_FWDH;
        if (higgs.pt() > 200)
          return GG2H_PTH_GT200;
        if (Njets == 0)
          return higgs.pt() < 10 ? GG2H_0J_PTH_0_10 : GG2H_0J_PTH_GT10;
        if (Njets == 1)
          return Category(GG2H_1J_PTH_0_60 + getBin(higgs.pt(), {0, 60, 120, 200}));
        if (Njets > 1) {
          //VBF topology
          if (vbfTopo)
            return Category(GG2H_MJJ_350_700_PTHJJ_0_25 + vbfTopo - 1);
          //Njets >= 2jets without VBF topology (mjj<350)
          return Category(GG2H_GE2J_MJJ_0_350_PTH_0_60 + getBin(higgs.pt(), {0, 60, 120, 200}));
        }
      }

      // 2. Electroweak qq->Hqq Stage 1.1 categories
      else if (prodMode == HTXS::VBF || (isVH(prodMode) && quarkDecay(V))) {
        if (std::abs(higgs.rapidity()) > 2.5)
          return QQ2HQQ_FWDH;
        int Njets = jets.size();
        if (Njets == 0)
          return QQ2HQQ_0J;
        else if (Njets == 1)
          return QQ2HQQ_1J;
        else if (Njets >= 2) {
          double mjj = (jets[0].mom() + jets[1].mom()).mass();
          if (mjj < 60)
            return QQ2HQQ_MJJ_0_60;
          else if (60 < mjj && mjj < 120)
            return QQ2HQQ_MJJ_60_120;
          else if (120 < mjj && mjj < 350)
            return QQ2HQQ_MJJ_120_350;
          else if (mjj > 350) {
            if (higgs.pt() > 200)
              return QQ2HQQ_MJJ_GT350_PTH_GT200;
            if (vbfTopo)
              return Category(QQ2HQQ_MJJ_GT350_PTH_GT200 + vbfTopo);
          }
        }
      }
      // 3. WH->Hlv categories
      else if (prodMode == HTXS::WH) {
        if (fwdHiggs)
          return QQ2HLNU_FWDH;
        else if (V.pt() < 75)
          return QQ2HLNU_PTV_0_75;
        else if (V.pt() < 150)
          return QQ2HLNU_PTV_75_150;
        else if (V.pt() > 250)
          return QQ2HLNU_PTV_GT250;
        // 150 < pTV/GeV < 250
        return jets.empty() ? QQ2HLNU_PTV_150_250_0J : QQ2HLNU_PTV_150_250_GE1J;
      }
      // 4. qq->ZH->llH categories
      else if (prodMode == HTXS::QQ2ZH) {
        if (fwdHiggs)
          return QQ2HLL_FWDH;
        else if (V.pt() < 75)
          return QQ2HLL_PTV_0_75;
        else if (V.pt() < 150)
          return QQ2HLL_PTV_75_150;
        else if (V.pt() > 250)
          return QQ2HLL_PTV_GT250;
        // 150 < pTV/GeV < 250
        return jets.empty() ? QQ2HLL_PTV_150_250_0J : QQ2HLL_PTV_150_250_GE1J;
      }
      // 5. gg->ZH->llH categories
      else if (prodMode == HTXS::GG2ZH) {
        if (fwdHiggs)
          return GG2HLL_FWDH;
        else if (V.pt() < 75)
          return GG2HLL_PTV_0_75;
        else if (V.pt() < 150)
          return GG2HLL_PTV_75_150;
        else if (V.pt() > 250)
          return GG2HLL_PTV_GT250;
        return jets.empty() ? GG2HLL_PTV_150_250_0J : GG2HLL_PTV_150_250_GE1J;
      }
      // 6.ttH,bbH,tH categories
      else if (prodMode == HTXS::TTH)
        return Category(TTH_FWDH + ctrlHiggs);
      else if (prodMode == HTXS::BBH)
        return Category(BBH_FWDH + ctrlHiggs);
      else if (prodMode == HTXS::TH)
        return Category(TH_FWDH + ctrlHiggs);
      return UNKNOWN;
    }

    /// @brief Stage-1_1 categorization
    HTXS::Stage1_1_Fine::Category getStage1_1_Fine_Category(const HTXS::HiggsProdMode prodMode,
                                                            const Particle &higgs,
                                                            const Jets &jets,
                                                            const Particle &V) {
      using namespace HTXS::Stage1_1_Fine;
      int Njets = jets.size(), ctrlHiggs = std::abs(higgs.rapidity()) < 2.5, fwdHiggs = !ctrlHiggs;
      int vbfTopo = vbfTopology_Stage1_X_Fine(jets, higgs);

      // 1. GGF Stage 1.1 categories
      //    Following YR4 write-up: XXXXX
      if (prodMode == HTXS::GGF || (prodMode == HTXS::GG2ZH && quarkDecay(V))) {
        if (fwdHiggs)
          return GG2H_FWDH;
        if (higgs.pt() > 200)
          return GG2H_PTH_GT200;
        if (Njets == 0)
          return higgs.pt() < 10 ? GG2H_0J_PTH_0_10 : GG2H_0J_PTH_GT10;
        if (Njets == 1)
          return Category(GG2H_1J_PTH_0_60 + getBin(higgs.pt(), {0, 60, 120, 200}));
        if (Njets > 1) {
          //double mjj = (jets[0].mom()+jets[1].mom()).mass();
          double pTHjj = (jets[0].momentum() + jets[1].momentum() + higgs.momentum()).pt();
          //VBF topology
          if (vbfTopo)
            return Category(GG2H_MJJ_350_700_PTHJJ_0_25 + vbfTopo - 1);
          //Njets >= 2jets without VBF topology (mjj<350)
          if (pTHjj < 25)
            return Category(GG2H_GE2J_MJJ_0_350_PTH_0_60_PTHJJ_0_25 + getBin(higgs.pt(), {0, 60, 120, 200}));
          else
            return Category(GG2H_GE2J_MJJ_0_350_PTH_0_60_PTHJJ_GT25 + getBin(higgs.pt(), {0, 60, 120, 200}));
        }
      }

      // 2. Electroweak qq->Hqq Stage 1.1 categories
      else if (prodMode == HTXS::VBF || (isVH(prodMode) && quarkDecay(V))) {
        if (std::abs(higgs.rapidity()) > 2.5)
          return QQ2HQQ_FWDH;
        int Njets = jets.size();
        if (Njets == 0)
          return QQ2HQQ_0J;
        else if (Njets == 1)
          return QQ2HQQ_1J;
        else if (Njets >= 2) {
          double mjj = (jets[0].mom() + jets[1].mom()).mass();
          double pTHjj = (jets[0].momentum() + jets[1].momentum() + higgs.momentum()).pt();
          if (mjj < 350) {
            if (pTHjj < 25)
              return Category(QQ2HQQ_MJJ_0_60_PTHJJ_0_25 + getBin(mjj, {0, 60, 120, 350}));
            else
              return Category(QQ2HQQ_MJJ_0_60_PTHJJ_GT25 + getBin(mjj, {0, 60, 120, 350}));
          } else {  //mjj>350 GeV
            if (higgs.pt() < 200) {
              return Category(QQ2HQQ_MJJ_350_700_PTHJJ_0_25 + vbfTopo - 1);
            } else {
              return Category(QQ2HQQ_PTH_GT200_MJJ_350_700_PTHJJ_0_25 + vbfTopo - 1);
            }
          }
        }
      }
      // 3. WH->Hlv categories
      else if (prodMode == HTXS::WH) {
        if (fwdHiggs)
          return QQ2HLNU_FWDH;
        int Njets = jets.size();
        if (Njets == 0)
          return Category(QQ2HLNU_PTV_0_75_0J + getBin(V.pt(), {0, 75, 150, 250, 400}));
        if (Njets == 1)
          return Category(QQ2HLNU_PTV_0_75_1J + getBin(V.pt(), {0, 75, 150, 250, 400}));
        return Category(QQ2HLNU_PTV_0_75_GE2J + getBin(V.pt(), {0, 75, 150, 250, 400}));
      }
      // 4. qq->ZH->llH categories
      else if (prodMode == HTXS::QQ2ZH) {
        if (fwdHiggs)
          return QQ2HLL_FWDH;
        int Njets = jets.size();
        if (Njets == 0)
          return Category(QQ2HLL_PTV_0_75_0J + getBin(V.pt(), {0, 75, 150, 250, 400}));
        if (Njets == 1)
          return Category(QQ2HLL_PTV_0_75_1J + getBin(V.pt(), {0, 75, 150, 250, 400}));
        return Category(QQ2HLL_PTV_0_75_GE2J + getBin(V.pt(), {0, 75, 150, 250, 400}));
      }
      // 5. gg->ZH->llH categories
      else if (prodMode == HTXS::GG2ZH) {
        if (fwdHiggs)
          return GG2HLL_FWDH;
        int Njets = jets.size();
        if (Njets == 0)
          return Category(GG2HLL_PTV_0_75_0J + getBin(V.pt(), {0, 75, 150, 250, 400}));
        if (Njets == 1)
          return Category(GG2HLL_PTV_0_75_1J + getBin(V.pt(), {0, 75, 150, 250, 400}));
        return Category(GG2HLL_PTV_0_75_GE2J + getBin(V.pt(), {0, 75, 150, 250, 400}));
      }
      // 6.ttH,bbH,tH categories
      else if (prodMode == HTXS::TTH)
        return Category(TTH_FWDH + ctrlHiggs);
      else if (prodMode == HTXS::BBH)
        return Category(BBH_FWDH + ctrlHiggs);
      else if (prodMode == HTXS::TH)
        return Category(TH_FWDH + ctrlHiggs);
      return UNKNOWN;
    }

    /// @brief Stage-1.2 categorization
    HTXS::Stage1_2::Category getStage1_2_Category(const HTXS::HiggsProdMode prodMode,
                                                  const Particle &higgs,
                                                  const Jets &jets,
                                                  const Particle &V) {
      using namespace HTXS::Stage1_2;
      int Njets = jets.size(), ctrlHiggs = std::abs(higgs.rapidity()) < 2.5, fwdHiggs = !ctrlHiggs;
      int vbfTopo = vbfTopology_Stage1_X(jets, higgs);

      // 1. GGF Stage 1 categories
      //    Following YR4 write-up: XXXXX
      if (prodMode == HTXS::GGF || (prodMode == HTXS::GG2ZH && quarkDecay(V))) {
        if (fwdHiggs)
          return GG2H_FWDH;
        if (higgs.pt() > 200)
          return Category(GG2H_PTH_200_300 + getBin(higgs.pt(), {200, 300, 450, 650}));
        if (Njets == 0)
          return higgs.pt() < 10 ? GG2H_0J_PTH_0_10 : GG2H_0J_PTH_GT10;
        if (Njets == 1)
          return Category(GG2H_1J_PTH_0_60 + getBin(higgs.pt(), {0, 60, 120, 200}));
        if (Njets > 1) {
          //VBF topology
          if (vbfTopo)
            return Category(GG2H_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25 + vbfTopo - 1);
          //Njets >= 2jets without VBF topology (mjj<350)
          return Category(GG2H_GE2J_MJJ_0_350_PTH_0_60 + getBin(higgs.pt(), {0, 60, 120, 200}));
        }
      }

      // 2. Electroweak qq->Hqq Stage 1.2 categories
      else if (prodMode == HTXS::VBF || (isVH(prodMode) && quarkDecay(V))) {
        if (std::abs(higgs.rapidity()) > 2.5)
          return QQ2HQQ_FWDH;
        int Njets = jets.size();
        if (Njets == 0)
          return QQ2HQQ_0J;
        else if (Njets == 1)
          return QQ2HQQ_1J;
        else if (Njets >= 2) {
          double mjj = (jets[0].mom() + jets[1].mom()).mass();
          if (mjj < 60)
            return QQ2HQQ_GE2J_MJJ_0_60;
          else if (60 < mjj && mjj < 120)
            return QQ2HQQ_GE2J_MJJ_60_120;
          else if (120 < mjj && mjj < 350)
            return QQ2HQQ_GE2J_MJJ_120_350;
          else if (mjj > 350) {
            if (higgs.pt() > 200)
              return QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200;
            if (vbfTopo)
              return Category(QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200 + vbfTopo);
          }
        }
      }
      // 3. WH->Hlv categories
      else if (prodMode == HTXS::WH) {
        if (fwdHiggs)
          return QQ2HLNU_FWDH;
        else if (V.pt() < 75)
          return QQ2HLNU_PTV_0_75;
        else if (V.pt() < 150)
          return QQ2HLNU_PTV_75_150;
        else if (V.pt() > 250)
          return QQ2HLNU_PTV_GT250;
        // 150 < pTV/GeV < 250
        return jets.empty() ? QQ2HLNU_PTV_150_250_0J : QQ2HLNU_PTV_150_250_GE1J;
      }
      // 4. qq->ZH->llH categories
      else if (prodMode == HTXS::QQ2ZH) {
        if (fwdHiggs)
          return QQ2HLL_FWDH;
        else if (V.pt() < 75)
          return QQ2HLL_PTV_0_75;
        else if (V.pt() < 150)
          return QQ2HLL_PTV_75_150;
        else if (V.pt() > 250)
          return QQ2HLL_PTV_GT250;
        // 150 < pTV/GeV < 250
        return jets.empty() ? QQ2HLL_PTV_150_250_0J : QQ2HLL_PTV_150_250_GE1J;
      }
      // 5. gg->ZH->llH categories
      else if (prodMode == HTXS::GG2ZH) {
        if (fwdHiggs)
          return GG2HLL_FWDH;
        else if (V.pt() < 75)
          return GG2HLL_PTV_0_75;
        else if (V.pt() < 150)
          return GG2HLL_PTV_75_150;
        else if (V.pt() > 250)
          return GG2HLL_PTV_GT250;
        return jets.empty() ? GG2HLL_PTV_150_250_0J : GG2HLL_PTV_150_250_GE1J;
      }
      // 6.ttH,bbH,tH categories
      else if (prodMode == HTXS::TTH) {
        if (fwdHiggs)
          return TTH_FWDH;
        else
          return Category(TTH_PTH_0_60 + getBin(higgs.pt(), {0, 60, 120, 200, 300}));
      } else if (prodMode == HTXS::BBH)
        return Category(BBH_FWDH + ctrlHiggs);
      else if (prodMode == HTXS::TH)
        return Category(TH_FWDH + ctrlHiggs);
      return UNKNOWN;
    }

    /// @brief Stage-1.2 Fine categorization
    HTXS::Stage1_2_Fine::Category getStage1_2_Fine_Category(const HTXS::HiggsProdMode prodMode,
                                                            const Particle &higgs,
                                                            const Jets &jets,
                                                            const Particle &V) {
      using namespace HTXS::Stage1_2_Fine;
      int Njets = jets.size(), ctrlHiggs = std::abs(higgs.rapidity()) < 2.5, fwdHiggs = !ctrlHiggs;
      int vbfTopo = vbfTopology_Stage1_X_Fine(jets, higgs);

      // 1. GGF Stage 1.2 categories
      //    Following YR4 write-up: XXXXX
      if (prodMode == HTXS::GGF || (prodMode == HTXS::GG2ZH && quarkDecay(V))) {
        if (fwdHiggs)
          return GG2H_FWDH;
        if (higgs.pt() > 200) {
          if (Njets > 0) {
            double pTHj = (jets[0].momentum() + higgs.momentum()).pt();
            if (pTHj / higgs.pt() > 0.15)
              return Category(GG2H_PTH_200_300_PTHJoverPTH_GT15 + getBin(higgs.pt(), {200, 300, 450, 650}));
            else
              return Category(GG2H_PTH_200_300_PTHJoverPTH_0_15 + getBin(higgs.pt(), {200, 300, 450, 650}));
          } else
            return Category(GG2H_PTH_200_300_PTHJoverPTH_0_15 + getBin(higgs.pt(), {200, 300, 450, 650}));
        }
        if (Njets == 0)
          return higgs.pt() < 10 ? GG2H_0J_PTH_0_10 : GG2H_0J_PTH_GT10;
        if (Njets == 1)
          return Category(GG2H_1J_PTH_0_60 + getBin(higgs.pt(), {0, 60, 120, 200}));
        if (Njets > 1) {
          //double mjj = (jets[0].mom()+jets[1].mom()).mass();
          double pTHjj = (jets[0].momentum() + jets[1].momentum() + higgs.momentum()).pt();
          //VBF topology
          if (vbfTopo)
            return Category(GG2H_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25 + vbfTopo - 1);
          //Njets >= 2jets without VBF topology (mjj<350)
          if (pTHjj < 25)
            return Category(GG2H_GE2J_MJJ_0_350_PTH_0_60_PTHJJ_0_25 + getBin(higgs.pt(), {0, 60, 120, 200}));
          else
            return Category(GG2H_GE2J_MJJ_0_350_PTH_0_60_PTHJJ_GT25 + getBin(higgs.pt(), {0, 60, 120, 200}));
        }
      }

      // 2. Electroweak qq->Hqq Stage 1.2 categories
      else if (prodMode == HTXS::VBF || (isVH(prodMode) && quarkDecay(V))) {
        if (std::abs(higgs.rapidity()) > 2.5)
          return QQ2HQQ_FWDH;
        int Njets = jets.size();
        if (Njets == 0)
          return QQ2HQQ_0J;
        else if (Njets == 1)
          return QQ2HQQ_1J;
        else if (Njets >= 2) {
          double mjj = (jets[0].mom() + jets[1].mom()).mass();
          double pTHjj = (jets[0].momentum() + jets[1].momentum() + higgs.momentum()).pt();
          if (mjj < 350) {
            if (pTHjj < 25)
              return Category(QQ2HQQ_GE2J_MJJ_0_60_PTHJJ_0_25 + getBin(mjj, {0, 60, 120, 350}));
            else
              return Category(QQ2HQQ_GE2J_MJJ_0_60_PTHJJ_GT25 + getBin(mjj, {0, 60, 120, 350}));
          } else {  //mjj>350 GeV
            if (higgs.pt() < 200) {
              return Category(QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25 + vbfTopo - 1);
            } else {
              return Category(QQ2HQQ_GE2J_MJJ_350_700_PTH_GT200_PTHJJ_0_25 + vbfTopo - 1);
            }
          }
        }
      }
      // 3. WH->Hlv categories
      else if (prodMode == HTXS::WH) {
        if (fwdHiggs)
          return QQ2HLNU_FWDH;
        int Njets = jets.size();
        if (Njets == 0)
          return Category(QQ2HLNU_PTV_0_75_0J + getBin(V.pt(), {0, 75, 150, 250, 400}));
        if (Njets == 1)
          return Category(QQ2HLNU_PTV_0_75_1J + getBin(V.pt(), {0, 75, 150, 250, 400}));
        return Category(QQ2HLNU_PTV_0_75_GE2J + getBin(V.pt(), {0, 75, 150, 250, 400}));
      }
      // 4. qq->ZH->llH categories
      else if (prodMode == HTXS::QQ2ZH) {
        if (fwdHiggs)
          return QQ2HLL_FWDH;
        int Njets = jets.size();
        if (Njets == 0)
          return Category(QQ2HLL_PTV_0_75_0J + getBin(V.pt(), {0, 75, 150, 250, 400}));
        if (Njets == 1)
          return Category(QQ2HLL_PTV_0_75_1J + getBin(V.pt(), {0, 75, 150, 250, 400}));
        return Category(QQ2HLL_PTV_0_75_GE2J + getBin(V.pt(), {0, 75, 150, 250, 400}));
      }
      // 5. gg->ZH->llH categories
      else if (prodMode == HTXS::GG2ZH) {
        if (fwdHiggs)
          return GG2HLL_FWDH;
        int Njets = jets.size();
        if (Njets == 0)
          return Category(GG2HLL_PTV_0_75_0J + getBin(V.pt(), {0, 75, 150, 250, 400}));
        if (Njets == 1)
          return Category(GG2HLL_PTV_0_75_1J + getBin(V.pt(), {0, 75, 150, 250, 400}));
        return Category(GG2HLL_PTV_0_75_GE2J + getBin(V.pt(), {0, 75, 150, 250, 400}));
      }
      // 6.ttH,bbH,tH categories
      else if (prodMode == HTXS::TTH) {
        if (fwdHiggs)
          return TTH_FWDH;
        else
          return Category(TTH_PTH_0_60 + getBin(higgs.pt(), {0, 60, 120, 200, 300, 450}));
      } else if (prodMode == HTXS::BBH)
        return Category(BBH_FWDH + ctrlHiggs);
      else if (prodMode == HTXS::TH)
        return Category(TH_FWDH + ctrlHiggs);
      return UNKNOWN;
    }

    /// @}

    /// @name Default Rivet analysis methods and steering methods
    /// @{

    /// @brief Sets the Higgs production mode
    void setHiggsProdMode(HTXS::HiggsProdMode prodMode) { m_HiggsProdMode = prodMode; }

    /// @brief default Rivet Analysis::init method
    /// Booking of histograms, initializing Rivet projection
    /// Extracts Higgs production mode from shell variable if not set manually using setHiggsProdMode
    void init() override {
      printf("==============================================================\n");
      printf("========     HiggsTemplateCrossSections Initialization     =========\n");
      printf("==============================================================\n");
      // check that the production mode has been set
      // if running in standalone Rivet the production mode is set through an env variable
      if (m_HiggsProdMode == HTXS::UNKNOWN) {
        char *pm_env = std::getenv("HIGGSPRODMODE");
        string pm(pm_env == nullptr ? "" : pm_env);
        if (pm == "GGF")
          m_HiggsProdMode = HTXS::GGF;
        else if (pm == "VBF")
          m_HiggsProdMode = HTXS::VBF;
        else if (pm == "WH")
          m_HiggsProdMode = HTXS::WH;
        else if (pm == "ZH")
          m_HiggsProdMode = HTXS::QQ2ZH;
        else if (pm == "QQ2ZH")
          m_HiggsProdMode = HTXS::QQ2ZH;
        else if (pm == "GG2ZH")
          m_HiggsProdMode = HTXS::GG2ZH;
        else if (pm == "TTH")
          m_HiggsProdMode = HTXS::TTH;
        else if (pm == "BBH")
          m_HiggsProdMode = HTXS::BBH;
        else if (pm == "TH")
          m_HiggsProdMode = HTXS::TH;
        else {
          MSG_WARNING("No HIGGSPRODMODE shell variable found. Needed when running Rivet stand-alone.");
        }
      }

      // Projections for final state particles
      const FinalState FS;
      declare(FS, "FS");

      // initialize the histograms with for each of the stages
      initializeHistos();
      m_sumw = 0.0;
      m_sumevents = 0;
      printf("==============================================================\n");
      printf("========             Higgs prod mode %d              =========\n", m_HiggsProdMode);
      printf("========          Sucessful Initialization           =========\n");
      printf("==============================================================\n");
    }

    // Perform the per-event analysis
    void analyze(const Event &event) override {
      // get the classification
      HiggsClassification cat = classifyEvent(event, m_HiggsProdMode);

      // Fill histograms: categorization --> linerize the categories
      const double weight = 1.0;
      m_sumw += weight;

      int F = cat.stage0_cat % 10, P = cat.stage1_cat_pTjet30GeV / 100;
      hist_stage0->fill(cat.stage0_cat / 10 * 2 + F, weight);

      // Stage 1 enum offsets for each production mode: GGF=12, VBF=6, WH= 5, QQ2ZH=5, GG2ZH=4, TTH=2, BBH=2, TH=2
      vector<int> offset({0, 1, 13, 19, 24, 29, 33, 35, 37, 39});
      int off = offset[P];
      hist_stage1_pTjet25->fill(cat.stage1_cat_pTjet25GeV % 100 + off, weight);
      hist_stage1_pTjet30->fill(cat.stage1_cat_pTjet30GeV % 100 + off, weight);

      // Stage 1_2 enum offsets for each production mode: GGF=17, VBF=11, WH= 6, QQ2ZH=6, GG2ZH=6, TTH=6, BBH=2, TH=2
      static vector<int> offset1_2({0, 1, 18, 29, 35, 41, 47, 53, 55, 57});
      int off1_2 = offset1_2[P];
      // Stage 1_2 Fine enum offsets for each production mode: GGF=28, VBF=25, WH= 16, QQ2ZH=16, GG2ZH=16, TTH=7, BBH=2, TH=2
      static vector<int> offset1_2f({0, 1, 29, 54, 70, 86, 102, 109, 111, 113});
      int off1_2f = offset1_2f[P];
      hist_stage1_2_pTjet25->fill(cat.stage1_2_cat_pTjet25GeV % 100 + off1_2, weight);
      hist_stage1_2_pTjet30->fill(cat.stage1_2_cat_pTjet30GeV % 100 + off1_2, weight);
      hist_stage1_2_fine_pTjet25->fill(cat.stage1_2_fine_cat_pTjet25GeV % 100 + off1_2f, weight);
      hist_stage1_2_fine_pTjet30->fill(cat.stage1_2_fine_cat_pTjet30GeV % 100 + off1_2f, weight);

      // Fill histograms: variables used in the categorization
      hist_pT_Higgs->fill(cat.higgs.pT(), weight);
      hist_y_Higgs->fill(cat.higgs.rapidity(), weight);
      hist_pT_V->fill(cat.V.pT(), weight);

      hist_Njets25->fill(cat.jets25.size(), weight);
      hist_Njets30->fill(cat.jets30.size(), weight);

      hist_isZ2vv->fill(cat.isZ2vvDecay, weight);

      // Jet variables. Use jet collection with pT threshold at 30 GeV
      if (!cat.jets30.empty())
        hist_pT_jet1->fill(cat.jets30[0].pt(), weight);
      if (cat.jets30.size() >= 2) {
        const FourMomentum &j1 = cat.jets30[0].momentum(), &j2 = cat.jets30[1].momentum();
        hist_deltay_jj->fill(std::abs(j1.rapidity() - j2.rapidity()), weight);
        hist_dijet_mass->fill((j1 + j2).mass(), weight);
        hist_pT_Hjj->fill((j1 + j2 + cat.higgs.momentum()).pt(), weight);
      }
    }

    void printClassificationSummary() {
      MSG_INFO(" ====================================================== ");
      MSG_INFO("      Higgs Template X-Sec Categorization Tool          ");
      MSG_INFO("                Status Code Summary                     ");
      MSG_INFO(" ====================================================== ");
      bool allSuccess = (m_sumevents == m_errorCount[HTXS::SUCCESS]);
      if (allSuccess)
        MSG_INFO("     >>>> All " << m_errorCount[HTXS::SUCCESS] << " events successfully categorized!");
      else {
        MSG_INFO("     >>>> " << m_errorCount[HTXS::SUCCESS] << " events successfully categorized");
        MSG_INFO("     >>>> --> the following errors occured:");
        MSG_INFO("     >>>> " << m_errorCount[HTXS::PRODMODE_DEFINED] << " had an undefined Higgs production mode.");
        MSG_INFO("     >>>> " << m_errorCount[HTXS::MOMENTUM_CONSERVATION] << " failed momentum conservation.");
        MSG_INFO("     >>>> " << m_errorCount[HTXS::HIGGS_IDENTIFICATION]
                              << " failed to identify a valid Higgs boson.");
        MSG_INFO("     >>>> " << m_errorCount[HTXS::HS_VTX_IDENTIFICATION]
                              << " failed to identify the hard scatter vertex.");
        MSG_INFO("     >>>> " << m_errorCount[HTXS::VH_IDENTIFICATION] << " VH: to identify a valid V-boson.");
        MSG_INFO("     >>>> " << m_errorCount[HTXS::TOP_W_IDENTIFICATION]
                              << " failed to identify valid Ws from top decay.");
      }
      MSG_INFO(" ====================================================== ");
      MSG_INFO(" ====================================================== ");
    }

    void finalize() override {
      printClassificationSummary();
      double sf = m_sumw > 0 ? 1.0 / m_sumw : 1.0;
      for (const auto &hist : {hist_stage0,
                               hist_stage1_pTjet25,
                               hist_stage1_pTjet30,
                               hist_stage1_2_pTjet25,
                               hist_stage1_2_pTjet30,
                               hist_stage1_2_fine_pTjet25,
                               hist_stage1_2_fine_pTjet30,
                               hist_Njets25,
                               hist_Njets30,
                               hist_pT_Higgs,
                               hist_y_Higgs,
                               hist_pT_V,
                               hist_pT_jet1,
                               hist_deltay_jj,
                               hist_dijet_mass,
                               hist_pT_Hjj})
        scale(hist, sf);
    }

    /*
     *  initialize histograms
     */

    void initializeHistos() {
      book(hist_stage0, "HTXS_stage0", 20, 0, 20);
      book(hist_stage1_pTjet25, "HTXS_stage1_pTjet25", 40, 0, 40);
      book(hist_stage1_pTjet30, "HTXS_stage1_pTjet30", 40, 0, 40);
      book(hist_stage1_2_pTjet25, "HTXS_stage1_2_pTjet25", 57, 0, 57);
      book(hist_stage1_2_pTjet30, "HTXS_stage1_2_pTjet30", 57, 0, 57);
      book(hist_stage1_2_fine_pTjet25, "HTXS_stage1_2_fine_pTjet25", 113, 0, 113);
      book(hist_stage1_2_fine_pTjet30, "HTXS_stage1_2_fine_pTjet30", 113, 0, 113);
      book(hist_pT_Higgs, "pT_Higgs", 80, 0, 400);
      book(hist_y_Higgs, "y_Higgs", 80, -4, 4);
      book(hist_pT_V, "pT_V", 80, 0, 400);
      book(hist_pT_jet1, "pT_jet1", 80, 0, 400);
      book(hist_deltay_jj, "deltay_jj", 50, 0, 10);
      book(hist_dijet_mass, "m_jj", 50, 0, 2000);
      book(hist_pT_Hjj, "pT_Hjj", 50, 0, 250);
      book(hist_Njets25, "Njets25", 10, 0, 10);
      book(hist_Njets30, "Njets30", 10, 0, 10);
      book(hist_isZ2vv, "isZ2vv", 2, 0, 2);
    }
    /// @}

    /*
     *    initialize private members used in the classification procedure
     */

  private:
    double m_sumw;
    size_t m_sumevents;
    HTXS::HiggsProdMode m_HiggsProdMode;
    std::map<HTXS::ErrorCode, size_t> m_errorCount;
    Histo1DPtr hist_stage0;
    Histo1DPtr hist_stage1_pTjet25, hist_stage1_pTjet30;
    Histo1DPtr hist_stage1_2_pTjet25, hist_stage1_2_pTjet30;
    Histo1DPtr hist_stage1_2_fine_pTjet25, hist_stage1_2_fine_pTjet30;
    Histo1DPtr hist_pT_Higgs, hist_y_Higgs;
    Histo1DPtr hist_pT_V, hist_pT_jet1;
    Histo1DPtr hist_deltay_jj, hist_dijet_mass, hist_pT_Hjj;
    Histo1DPtr hist_Njets25, hist_Njets30;
    Histo1DPtr hist_isZ2vv;
  };

  // the PLUGIN only needs to be decleared when running standalone Rivet
  // and causes compilation / linking issues if included in Athena / RootCore
  //check for Rivet environment variable RIVET_ANALYSIS_PATH
#ifdef RIVET_ANALYSIS_PATH
  // The hook for the plugin system
  DECLARE_RIVET_PLUGIN(HiggsTemplateCrossSections);
#endif

}  // namespace Rivet

#endif
