#include "RecoJets/JetAlgorithms/interface/HEPTopTaggerV2.h"

// Do not change next line, it's needed by the sed-code that makes the tagger CMSSW-compatible.
namespace external {

  //optimal_R fit
  double R_opt_calc_funct(double pt_filt) { return 327. / pt_filt; }

  void HEPTopTaggerV2_fixed_R::print_banner() {
    if (!_first_time) {
      return;
    }
    _first_time = false;

    edm::LogInfo("HEPTopTaggerV2") << "#--------------------------------------------------------------------------\n";
    edm::LogInfo("HEPTopTaggerV2") << "#                   HEPTopTaggerV2 - under construction                      \n";
    edm::LogInfo("HEPTopTaggerV2") << "#                                                                          \n";
    edm::LogInfo("HEPTopTaggerV2") << "# Please cite JHEP 1010 (2010) 078 [arXiv:1006.2833 [hep-ph]]              \n";
    edm::LogInfo("HEPTopTaggerV2") << "# and Phys.Rev. D89 (2014) 074047 [arXiv:1312.1504 [hep-ph]]               \n";
    edm::LogInfo("HEPTopTaggerV2") << "#--------------------------------------------------------------------------\n";
  }

  //pt wrt a reference vector
  double HEPTopTaggerV2_fixed_R::perp(const fastjet::PseudoJet& vec, const fastjet::PseudoJet& ref) {
    double ref_ref = ref.px() * ref.px() + ref.py() * ref.py() + ref.pz() * ref.pz();
    double vec_ref = vec.px() * ref.px() + vec.py() * ref.py() + vec.pz() * ref.pz();
    double per_per = vec.px() * vec.px() + vec.py() * vec.py() + vec.pz() * vec.pz();
    if (ref_ref > 0.)
      per_per -= vec_ref * vec_ref / ref_ref;
    if (per_per < 0.)
      per_per = 0.;
    return sqrt(per_per);
  }

  //modified Jade distance
  double HEPTopTaggerV2_fixed_R::djademod(const fastjet::PseudoJet& subjet_i,
                                          const fastjet::PseudoJet& subjet_j,
                                          const fastjet::PseudoJet& ref) {
    double dj = -1.0;
    double delta_phi = subjet_i.delta_phi_to(subjet_j);
    double delta_eta = subjet_i.eta() - subjet_j.eta();
    double delta_R = sqrt(delta_eta * delta_eta + delta_phi * delta_phi);
    dj = perp(subjet_i, ref) * perp(subjet_j, ref) * pow(delta_R, 4.);
    return dj;
  }

  //minimal |(m_ij / m_123) / (m_w/ m_t) - 1|
  double HEPTopTaggerV2_fixed_R::f_rec() {
    double m12 = (_top_subs[0] + _top_subs[1]).m();
    double m13 = (_top_subs[0] + _top_subs[2]).m();
    double m23 = (_top_subs[1] + _top_subs[2]).m();
    double m123 = (_top_subs[0] + _top_subs[1] + _top_subs[2]).m();

    double fw12 = fabs((m12 / m123) / (_mwmass / _mtmass) - 1);
    double fw13 = fabs((m13 / m123) / (_mwmass / _mtmass) - 1);
    double fw23 = fabs((m23 / m123) / (_mwmass / _mtmass) - 1);

    return std::min(fw12, std::min(fw13, fw23));
  }

  //find hard substructures
  void HEPTopTaggerV2_fixed_R::FindHardSubst(const PseudoJet& this_jet, std::vector<fastjet::PseudoJet>& t_parts) {
    PseudoJet parent1(0, 0, 0, 0), parent2(0, 0, 0, 0);
    if (this_jet.m() < _max_subjet_mass || !this_jet.validated_cs()->has_parents(this_jet, parent1, parent2)) {
      t_parts.push_back(this_jet);
    } else {
      if (parent1.m() < parent2.m())
        std::swap(parent1, parent2);
      FindHardSubst(parent1, t_parts);
      if (parent1.m() < _mass_drop_threshold * this_jet.m())
        FindHardSubst(parent2, t_parts);
    }
  }

  //store subjets as vector<PseudoJet> with [0]->b [1]->W-jet 1 [2]->W-jet 2
  void HEPTopTaggerV2_fixed_R::store_topsubjets(const std::vector<PseudoJet>& top_subs) {
    _top_subjets.resize(0);
    double m12 = (top_subs[0] + top_subs[1]).m();
    double m13 = (top_subs[0] + top_subs[2]).m();
    double m23 = (top_subs[1] + top_subs[2]).m();
    double dm12 = fabs(m12 - _mwmass);
    double dm13 = fabs(m13 - _mwmass);
    double dm23 = fabs(m23 - _mwmass);

    if (dm23 <= dm12 && dm23 <= dm13) {
      _top_subjets.push_back(top_subs[0]);
      _top_subjets.push_back(top_subs[1]);
      _top_subjets.push_back(top_subs[2]);
    } else if (dm13 <= dm12 && dm13 < dm23) {
      _top_subjets.push_back(top_subs[1]);
      _top_subjets.push_back(top_subs[0]);
      _top_subjets.push_back(top_subs[2]);
    } else if (dm12 < dm23 && dm12 < dm13) {
      _top_subjets.push_back(top_subs[2]);
      _top_subjets.push_back(top_subs[0]);
      _top_subjets.push_back(top_subs[1]);
    }
    _W = _top_subjets[1] + _top_subjets[2];
    return;
  }

  //check mass plane cuts
  bool HEPTopTaggerV2_fixed_R::check_mass_criteria(const std::vector<PseudoJet>& top_subs) const {
    bool is_passed = false;
    double m12 = (top_subs[0] + top_subs[1]).m();
    double m13 = (top_subs[0] + top_subs[2]).m();
    double m23 = (top_subs[1] + top_subs[2]).m();
    double m123 = (top_subs[0] + top_subs[1] + top_subs[2]).m();
    double atan1312 = atan(m13 / m12);
    double m23_over_m123 = m23 / m123;
    double m23_over_m123_square = m23_over_m123 * m23_over_m123;
    double rmin_square = _rmin * _rmin;
    double rmax_square = _rmax * _rmax;
    double m13m12_square_p1 = (1 + (m13 / m12) * (m13 / m12));
    double m12m13_square_p1 = (1 + (m12 / m13) * (m12 / m13));
    if ((atan1312 > _m13cutmin && _m13cutmax > atan1312 && (m23_over_m123 > _rmin && _rmax > m23_over_m123)) ||
        ((m23_over_m123_square < 1 - rmin_square * m13m12_square_p1) &&
         (m23_over_m123_square > 1 - rmax_square * m13m12_square_p1) && (m23_over_m123 > _m23cut)) ||
        ((m23_over_m123_square < 1 - rmin_square * m12m13_square_p1) &&
         (m23_over_m123_square > 1 - rmax_square * m12m13_square_p1) && (m23_over_m123 > _m23cut))) {
      is_passed = true;
    }
    return is_passed;
  }

  double HEPTopTaggerV2_fixed_R::nsub(
      fastjet::PseudoJet jet, int order, fastjet::contrib::Njettiness::AxesMode axes, double beta, double R0) {
    fastjet::contrib::Nsubjettiness nsub(order, axes, beta, R0);
    return nsub.result(jet);
  }

  HEPTopTaggerV2_fixed_R::HEPTopTaggerV2_fixed_R()
      : _do_qjets(false),
        _mass_drop_threshold(0.8),
        _max_subjet_mass(30.),
        _mode(Mode(0)),
        _mtmass(172.3),
        _mwmass(80.4),
        _mtmin(150.),
        _mtmax(200.),
        _rmin(0.85 * 80.4 / 172.3),
        _rmax(1.15 * 80.4 / 172.3),
        _m23cut(0.35),
        _m13cutmin(0.2),
        _m13cutmax(1.3),
        _minpt_tag(200.),
        _nfilt(5),
        _Rfilt(0.3),
        _jet_algorithm_filter(fastjet::cambridge_algorithm),
        _minpt_subjet(0.),
        _jet_algorithm_recluster(fastjet::cambridge_algorithm),
        _zcut(0.1),
        _rcut_factor(0.5),
        _q_zcut(0.1),
        _q_dcut_fctr(0.5),
        _q_exp_min(0.),
        _q_exp_max(0.),
        _q_rigidity(0.1),
        _q_truncation_fctr(0.0),
        _rnEngine(nullptr),
        //_qjet_plugin(_q_zcut, _q_dcut_fctr, _q_exp_min, _q_exp_max, _q_rigidity, _q_truncation_fctr),
        _debug(false),
        _first_time(true) {
    _djsum = 0.;
    _delta_top = 1000000000000.0;
    _pruned_mass = 0.;
    _unfiltered_mass = 0.;
    _top_candidate.reset(0., 0., 0., 0.);
    _parts_size = 0;
    _is_maybe_top = _is_masscut_passed = _is_ptmincut_passed = false;
    _top_subs.clear();
    _top_subjets.clear();
    _top_hadrons.clear();
    _top_parts.clear();
    _qweight = -1.;
  }

  HEPTopTaggerV2_fixed_R::HEPTopTaggerV2_fixed_R(const fastjet::PseudoJet jet)
      : _do_qjets(false),
        _jet(jet),
        _initial_jet(jet),
        _mass_drop_threshold(0.8),
        _max_subjet_mass(30.),
        _mode(Mode(0)),
        _mtmass(172.3),
        _mwmass(80.4),
        _mtmin(150.),
        _mtmax(200.),
        _rmin(0.85 * 80.4 / 172.3),
        _rmax(1.15 * 80.4 / 172.3),
        _m23cut(0.35),
        _m13cutmin(0.2),
        _m13cutmax(1.3),
        _minpt_tag(200.),
        _nfilt(5),
        _Rfilt(0.3),
        _jet_algorithm_filter(fastjet::cambridge_algorithm),
        _minpt_subjet(0.),
        _jet_algorithm_recluster(fastjet::cambridge_algorithm),
        _zcut(0.1),
        _rcut_factor(0.5),
        _q_zcut(0.1),
        _q_dcut_fctr(0.5),
        _q_exp_min(0.),
        _q_exp_max(0.),
        _q_rigidity(0.1),
        _q_truncation_fctr(0.0),
        _fat(jet),
        _rnEngine(nullptr),
        _debug(false),
        _first_time(true) {}

  HEPTopTaggerV2_fixed_R::HEPTopTaggerV2_fixed_R(const fastjet::PseudoJet jet, double mtmass, double mwmass)
      : _do_qjets(false),
        _jet(jet),
        _initial_jet(jet),
        _mass_drop_threshold(0.8),
        _max_subjet_mass(30.),
        _mode(Mode(0)),
        _mtmass(mtmass),
        _mwmass(mwmass),
        _rmin(0.85 * 80.4 / 172.3),
        _rmax(1.15 * 80.4 / 172.3),
        _m23cut(0.35),
        _m13cutmin(0.2),
        _m13cutmax(1.3),
        _minpt_tag(200.),
        _nfilt(5),
        _Rfilt(0.3),
        _jet_algorithm_filter(fastjet::cambridge_algorithm),
        _minpt_subjet(0.),
        _jet_algorithm_recluster(fastjet::cambridge_algorithm),
        _zcut(0.1),
        _rcut_factor(0.5),
        _q_zcut(0.1),
        _q_dcut_fctr(0.5),
        _q_exp_min(0.),
        _q_exp_max(0.),
        _q_rigidity(0.1),
        _q_truncation_fctr(0.0),
        _fat(jet),
        _rnEngine(nullptr),
        _debug(false),
        _first_time(true) {}

  void HEPTopTaggerV2_fixed_R::run() {
    print_banner();

    if ((_mode != EARLY_MASSRATIO_SORT_MASS) && (_mode != LATE_MASSRATIO_SORT_MASS) &&
        (_mode != EARLY_MASSRATIO_SORT_MODDJADE) && (_mode != LATE_MASSRATIO_SORT_MODDJADE) &&
        (_mode != TWO_STEP_FILTER)) {
      edm::LogError("HEPTopTaggerV2") << "ERROR: UNKNOWN MODE" << std::endl;
      return;
    }

    //Qjets
    QjetsPlugin _qjet_plugin(_q_zcut, _q_dcut_fctr, _q_exp_min, _q_exp_max, _q_rigidity, _q_truncation_fctr);
    _qjet_def = fastjet::JetDefinition(&_qjet_plugin);
    _qweight = -1;
    vector<fastjet::PseudoJet> _q_constits;
    ClusterSequence* _qjet_seq;
    PseudoJet _qjet;
    if (_do_qjets) {
      _q_constits = _initial_jet.associated_cluster_sequence()->constituents(_initial_jet);
      _qjet_seq = new ClusterSequence(_q_constits, _qjet_def);
      _qjet = sorted_by_pt(_qjet_seq->inclusive_jets())[0];
      _qjet_seq->delete_self_when_unused();
      const QjetsBaseExtras* ext = dynamic_cast<const QjetsBaseExtras*>(_qjet_seq->extras());
      _qweight = ext->weight();
      _jet = _qjet;
      _fat = _qjet;
      _qjet_plugin.SetRNEngine(_rnEngine);
    }

    //initialization
    _djsum = 0.;
    _delta_top = 1000000000000.0;
    _pruned_mass = 0.;
    _unfiltered_mass = 0.;
    _top_candidate.reset(0., 0., 0., 0.);
    _parts_size = 0;
    _is_maybe_top = _is_masscut_passed = _is_ptmincut_passed = false;
    _top_subs.clear();
    _top_subjets.clear();
    _top_hadrons.clear();
    _top_parts.clear();

    //find hard substructures
    FindHardSubst(_jet, _top_parts);

    if (_top_parts.size() < 3) {
      if (_debug) {
        edm::LogInfo("HEPTopTaggerV2") << "< 3 hard substructures " << std::endl;
      }
      return;  //such events are not interesting
    }

    // Sort subjets-after-unclustering by pT.
    // Necessary so that two-step-filtering can use the leading-three.
    _top_parts = sorted_by_pt(_top_parts);

    // loop over triples
    _top_parts = sorted_by_pt(_top_parts);
    for (unsigned rr = 0; rr < _top_parts.size(); rr++) {
      for (unsigned ll = rr + 1; ll < _top_parts.size(); ll++) {
        for (unsigned kk = ll + 1; kk < _top_parts.size(); kk++) {
          // two-step filtering
          // This means that we only look at the triplet formed by the
          // three leading-in-pT subjets-after-unclustering.
          if ((_mode == TWO_STEP_FILTER) && rr > 0)
            continue;
          if ((_mode == TWO_STEP_FILTER) && ll > 1)
            continue;
          if ((_mode == TWO_STEP_FILTER) && kk > 2)
            continue;

          //pick triple
          PseudoJet triple = join(_top_parts[rr], _top_parts[ll], _top_parts[kk]);

          //filtering
          double filt_top_R = std::min(_Rfilt,
                                       0.5 * sqrt(std::min(_top_parts[kk].squared_distance(_top_parts[ll]),
                                                           std::min(_top_parts[rr].squared_distance(_top_parts[ll]),
                                                                    _top_parts[kk].squared_distance(_top_parts[rr])))));
          JetDefinition filtering_def(_jet_algorithm_filter, filt_top_R);
          fastjet::Filter filter(filtering_def,
                                 fastjet::SelectorNHardest(_nfilt) * fastjet::SelectorPtMin(_minpt_subjet));
          PseudoJet topcandidate = filter(triple);

          //mass window cut
          if (topcandidate.m() < _mtmin || _mtmax < topcandidate.m())
            continue;

          // Sanity cut: can't recluster less than 3 objects into three subjets
          if (topcandidate.pieces().size() < 3)
            continue;

          // Recluster to 3 subjets and apply mass plane cuts
          JetDefinition reclustering(_jet_algorithm_recluster, 3.14);
          ClusterSequence* cs_top_sub = new ClusterSequence(topcandidate.constituents(), reclustering);
          std::vector<PseudoJet> top_subs = sorted_by_pt(cs_top_sub->exclusive_jets(3));
          cs_top_sub->delete_self_when_unused();

          // Require the third subjet to be above the pT threshold
          if (top_subs[2].perp() < _minpt_subjet)
            continue;

          // Modes with early 2d-massplane cuts
          if (_mode == EARLY_MASSRATIO_SORT_MASS && !check_mass_criteria(top_subs)) {
            continue;
          }
          if (_mode == EARLY_MASSRATIO_SORT_MODDJADE && !check_mass_criteria(top_subs)) {
            continue;
          }

          //is this candidate better than the other? -> update
          double deltatop = fabs(topcandidate.m() - _mtmass);
          double djsum = djademod(top_subs[0], top_subs[1], topcandidate) +
                         djademod(top_subs[0], top_subs[2], topcandidate) +
                         djademod(top_subs[1], top_subs[2], topcandidate);
          bool better = false;

          // Modes 0 and 1 sort by top mass
          if ((_mode == EARLY_MASSRATIO_SORT_MASS) || (_mode == LATE_MASSRATIO_SORT_MASS)) {
            if (deltatop < _delta_top)
              better = true;
          }
          // Modes 2 and 3 sort by modified jade distance
          else if ((_mode == EARLY_MASSRATIO_SORT_MODDJADE) || (_mode == LATE_MASSRATIO_SORT_MODDJADE)) {
            if (djsum > _djsum)
              better = true;
          }
          // Mode 4 is the two-step filtering. No sorting necessary as
          // we just look at the triplet of highest pT objects after
          // unclustering
          else if (_mode == TWO_STEP_FILTER) {
            better = true;
          } else {
            edm::LogError("HEPTopTaggerV2") << "ERROR: UNKNOWN MODE (IN DISTANCE MEASURE SELECTION)" << std::endl;
            return;
          }

          if (better) {
            _djsum = djsum;
            _delta_top = deltatop;
            _is_maybe_top = true;
            _top_candidate = topcandidate;
            _top_subs = top_subs;
            store_topsubjets(top_subs);
            _top_hadrons = topcandidate.constituents();
            // Pruning
            double _Rprun = _initial_jet.validated_cluster_sequence()->jet_def().R();
            JetDefinition jet_def_prune(fastjet::cambridge_algorithm, _Rprun);
            fastjet::Pruner pruner(jet_def_prune, _zcut, _rcut_factor);
            PseudoJet prunedjet = pruner(triple);
            _pruned_mass = prunedjet.m();
            _unfiltered_mass = triple.m();

            //are all criteria fulfilled?
            _is_masscut_passed = false;
            if (check_mass_criteria(top_subs)) {
              _is_masscut_passed = true;
            }
            _is_ptmincut_passed = false;
            if (_top_candidate.pt() > _minpt_tag) {
              _is_ptmincut_passed = true;
            }
          }  //end better
        }    //end kk
      }      //end ll
    }        //end rr

    return;
  }

  void HEPTopTaggerV2_fixed_R::get_info() const {
    edm::LogInfo("HEPTopTaggerV2") << "#--------------------------------------------------------------------------\n";
    edm::LogInfo("HEPTopTaggerV2") << "#                          HEPTopTaggerV2 Result" << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "#" << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# is top candidate: " << _is_maybe_top << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# mass plane cuts passed: " << _is_masscut_passed << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# top candidate mass: " << _top_candidate.m() << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# top candidate (pt, eta, phi): (" << _top_candidate.perp() << ", "
                                   << _top_candidate.eta() << ", " << _top_candidate.phi_std() << ")" << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# top hadrons: " << _top_hadrons.size() << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# hard substructures: " << _parts_size << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# |m - mtop| : " << _delta_top << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# djsum : " << _djsum << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# is consistency cut passed: " << _is_ptmincut_passed << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "#--------------------------------------------------------------------------\n";
    return;
  }

  void HEPTopTaggerV2_fixed_R::get_setting() const {
    edm::LogInfo("HEPTopTaggerV2") << "#--------------------------------------------------------------------------\n";
    edm::LogInfo("HEPTopTaggerV2") << "#                         HEPTopTaggerV2 Settings" << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "#" << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# mode: " << _mode << " (0 = EARLY_MASSRATIO_SORT_MASS) " << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "#        "
                                   << " (1 = LATE_MASSRATIO_SORT_MASS)  " << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "#        "
                                   << " (2 = EARLY_MASSRATIO_SORT_MODDJADE)  " << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "#        "
                                   << " (3 = LATE_MASSRATIO_SORT_MODDJADE)  " << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "#        "
                                   << " (4 = TWO_STEP_FILTER)  " << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# top mass: " << _mtmass << "    ";
    edm::LogInfo("HEPTopTaggerV2") << "W mass: " << _mwmass << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# top mass window: [" << _mtmin << ", " << _mtmax << "]" << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# W mass ratio: [" << _rmin << ", " << _rmax << "] (["
                                   << _rmin * _mtmass / _mwmass << "%, " << _rmax * _mtmass / _mwmass << "%])"
                                   << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# mass plane cuts: (m23cut, m13min, m13max) = (" << _m23cut << ", " << _m13cutmin
                                   << ", " << _m13cutmax << ")" << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# mass_drop_threshold: " << _mass_drop_threshold << "    ";
    edm::LogInfo("HEPTopTaggerV2") << "max_subjet_mass: " << _max_subjet_mass << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# R_filt: " << _Rfilt << "    ";
    edm::LogInfo("HEPTopTaggerV2") << "n_filt: " << _nfilt << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# minimal subjet pt: " << _minpt_subjet << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# minimal reconstructed pt: " << _minpt_tag << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "# internal jet algorithms (0 = kt, 1 = C/A, 2 = anti-kt): " << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "#   filtering: " << _jet_algorithm_filter << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "#   reclustering: " << _jet_algorithm_recluster << std::endl;
    edm::LogInfo("HEPTopTaggerV2") << "#--------------------------------------------------------------------------\n";

    return;
  }

  //uncluster a fat jet to subjets of given cone size
  void HEPTopTaggerV2::UnclusterFatjets(const vector<fastjet::PseudoJet>& big_fatjets,
                                        vector<fastjet::PseudoJet>& small_fatjets,
                                        const ClusterSequence& cseq,
                                        const double small_radius) {
    for (unsigned i = 0; i < big_fatjets.size(); i++) {
      const PseudoJet& this_jet = big_fatjets[i];
      PseudoJet parent1(0, 0, 0, 0), parent2(0, 0, 0, 0);
      bool test = cseq.has_parents(this_jet, parent1, parent2);
      double dR = 100;

      if (test)
        dR = sqrt(parent1.squared_distance(parent2));

      if (!test || dR < small_radius) {
        small_fatjets.push_back(this_jet);
      } else {
        vector<fastjet::PseudoJet> parents;
        parents.push_back(parent1);
        parents.push_back(parent2);
        UnclusterFatjets(parents, small_fatjets, cseq, small_radius);
      }
    }
  }

  HEPTopTaggerV2::HEPTopTaggerV2()
      : _do_optimalR(false),
        _do_qjets(false),
        _mass_drop_threshold(0.8),
        _max_subjet_mass(30.),
        _mode(Mode(0)),
        _mtmass(172.3),
        _mwmass(80.4),
        _mtmin(150.),
        _mtmax(200.),
        _rmin(0.85 * 80.4 / 172.3),
        _rmax(1.15 * 80.4 / 172.3),
        _m23cut(0.35),
        _m13cutmin(0.2),
        _m13cutmax(1.3),
        _minpt_tag(200.),
        _nfilt(5),
        _Rfilt(0.3),
        _jet_algorithm_filter(fastjet::cambridge_algorithm),
        _minpt_subjet(0.),
        _jet_algorithm_recluster(fastjet::cambridge_algorithm),
        _zcut(0.1),
        _rcut_factor(0.5),
        _max_fatjet_R(1.8),
        _min_fatjet_R(0.5),
        _step_R(0.1),
        _optimalR_threshold(0.2),
        _R_filt_optimalR_calc(0.2),
        _N_filt_optimalR_calc(10),
        _r_min_exp_function(&R_opt_calc_funct),
        _optimalR_mmin(150.),
        _optimalR_mmax(200.),
        _optimalR_fw(0.175),
        _R_opt_diff(0.3),
        _R_opt_reject_min(false),
        _R_filt_optimalR_pass(0.2),
        _N_filt_optimalR_pass(5),
        _R_filt_optimalR_fail(0.3),
        _N_filt_optimalR_fail(3),
        _q_zcut(0.1),
        _q_dcut_fctr(0.5),
        _q_exp_min(0.),
        _q_exp_max(0.),
        _q_rigidity(0.1),
        _q_truncation_fctr(0.0),
        _rnEngine(nullptr),
        _debug(false) {}

  HEPTopTaggerV2::HEPTopTaggerV2(const fastjet::PseudoJet& jet)
      : _do_optimalR(false),
        _do_qjets(false),
        _jet(jet),
        _initial_jet(jet),
        _mass_drop_threshold(0.8),
        _max_subjet_mass(30.),
        _mode(Mode(0)),
        _mtmass(172.3),
        _mwmass(80.4),
        _mtmin(150.),
        _mtmax(200.),
        _rmin(0.85 * 80.4 / 172.3),
        _rmax(1.15 * 80.4 / 172.3),
        _m23cut(0.35),
        _m13cutmin(0.2),
        _m13cutmax(1.3),
        _minpt_tag(200.),
        _nfilt(5),
        _Rfilt(0.3),
        _jet_algorithm_filter(fastjet::cambridge_algorithm),
        _minpt_subjet(0.),
        _jet_algorithm_recluster(fastjet::cambridge_algorithm),
        _zcut(0.1),
        _rcut_factor(0.5),
        _max_fatjet_R(jet.validated_cluster_sequence()->jet_def().R()),
        _min_fatjet_R(0.5),
        _step_R(0.1),
        _optimalR_threshold(0.2),
        _R_filt_optimalR_calc(0.2),
        _N_filt_optimalR_calc(10),
        _r_min_exp_function(&R_opt_calc_funct),
        _optimalR_mmin(150.),
        _optimalR_mmax(200.),
        _optimalR_fw(0.175),
        _R_opt_diff(0.3),
        _R_opt_reject_min(false),
        _R_filt_optimalR_pass(0.2),
        _N_filt_optimalR_pass(5),
        _R_filt_optimalR_fail(0.3),
        _N_filt_optimalR_fail(3),
        _q_zcut(0.1),
        _q_dcut_fctr(0.5),
        _q_exp_min(0.),
        _q_exp_max(0.),
        _q_rigidity(0.1),
        _q_truncation_fctr(0.0),
        _fat(jet),
        _rnEngine(nullptr),
        _debug(false) {}

  HEPTopTaggerV2::HEPTopTaggerV2(const fastjet::PseudoJet& jet, double mtmass, double mwmass)
      : _do_optimalR(false),
        _do_qjets(false),
        _jet(jet),
        _initial_jet(jet),
        _mass_drop_threshold(0.8),
        _max_subjet_mass(30.),
        _mode(Mode(0)),
        _mtmass(mtmass),
        _mwmass(mwmass),
        _mtmin(150.),
        _mtmax(200.),
        _rmin(0.85 * 80.4 / 172.3),
        _rmax(1.15 * 80.4 / 172.3),
        _m23cut(0.35),
        _m13cutmin(0.2),
        _m13cutmax(1.3),
        _minpt_tag(200.),
        _nfilt(5),
        _Rfilt(0.3),
        _jet_algorithm_filter(fastjet::cambridge_algorithm),
        _minpt_subjet(0.),
        _jet_algorithm_recluster(fastjet::cambridge_algorithm),
        _zcut(0.1),
        _rcut_factor(0.5),
        _max_fatjet_R(jet.validated_cluster_sequence()->jet_def().R()),
        _min_fatjet_R(0.5),
        _step_R(0.1),
        _optimalR_threshold(0.2),
        _R_filt_optimalR_calc(0.2),
        _N_filt_optimalR_calc(10),
        _r_min_exp_function(&R_opt_calc_funct),
        _optimalR_mmin(150.),
        _optimalR_mmax(200.),
        _optimalR_fw(0.175),
        _R_opt_diff(0.3),
        _R_opt_reject_min(false),
        _R_filt_optimalR_pass(0.2),
        _N_filt_optimalR_pass(5),
        _R_filt_optimalR_fail(0.3),
        _N_filt_optimalR_fail(3),
        _q_zcut(0.1),
        _q_dcut_fctr(0.5),
        _q_exp_min(0.),
        _q_exp_max(0.),
        _q_rigidity(0.1),
        _q_truncation_fctr(0.0),
        _fat(jet),
        _rnEngine(nullptr),
        _debug(false) {}

  void HEPTopTaggerV2::run() {
    QjetsPlugin _qjet_plugin(_q_zcut, _q_dcut_fctr, _q_exp_min, _q_exp_max, _q_rigidity, _q_truncation_fctr);
    int maxR = int(_max_fatjet_R * 10);
    int minR = int(_min_fatjet_R * 10);
    int stepR = int(_step_R * 10);
    _qweight = -1;
    _qjet_plugin.SetRNEngine(_rnEngine);

    if (!_do_optimalR) {
      HEPTopTaggerV2_fixed_R htt(_jet);
      htt.set_mass_drop_threshold(_mass_drop_threshold);
      htt.set_max_subjet_mass(_max_subjet_mass);
      htt.set_filtering_n(_nfilt);
      htt.set_filtering_R(_Rfilt);
      htt.set_filtering_minpt_subjet(_minpt_subjet);
      htt.set_filtering_jetalgorithm(_jet_algorithm_filter);
      htt.set_reclustering_jetalgorithm(_jet_algorithm_recluster);
      htt.set_mode(_mode);
      htt.set_mt(_mtmass);
      htt.set_mw(_mwmass);
      htt.set_top_mass_range(_mtmin, _mtmax);
      htt.set_mass_ratio_range(_rmin, _rmax);
      htt.set_mass_ratio_cut(_m23cut, _m13cutmin, _m13cutmax);
      htt.set_top_minpt(_minpt_tag);
      htt.set_pruning_zcut(_zcut);
      htt.set_pruning_rcut_factor(_rcut_factor);
      htt.set_debug(_debug);
      htt.set_qjets(_q_zcut, _q_dcut_fctr, _q_exp_min, _q_exp_max, _q_rigidity, _q_truncation_fctr);
      htt.run();

      _HEPTopTaggerV2[maxR] = htt;
      _Ropt = maxR;
      _qweight = htt.q_weight();
      _HEPTopTaggerV2_opt = _HEPTopTaggerV2[_Ropt];
    } else {
      _qjet_def = fastjet::JetDefinition(&_qjet_plugin);
      vector<fastjet::PseudoJet> _q_constits;
      ClusterSequence* _qjet_seq;
      PseudoJet _qjet;
      const ClusterSequence* _seq;
      _seq = _initial_jet.validated_cluster_sequence();
      if (_do_qjets) {
        _q_constits = _initial_jet.associated_cluster_sequence()->constituents(_initial_jet);
        _qjet_seq = new ClusterSequence(_q_constits, _qjet_def);
        _qjet = sorted_by_pt(_qjet_seq->inclusive_jets())[0];
        _qjet_seq->delete_self_when_unused();
        const QjetsBaseExtras* ext = dynamic_cast<const QjetsBaseExtras*>(_qjet_seq->extras());
        _qweight = ext->weight();
        _jet = _qjet;
        _seq = _qjet_seq;
        _fat = _qjet;
      }

      // Do MultiR procedure
      vector<fastjet::PseudoJet> big_fatjets;
      vector<fastjet::PseudoJet> small_fatjets;

      big_fatjets.push_back(_jet);
      _Ropt = 0;

      for (int R = maxR; R >= minR; R -= stepR) {
        UnclusterFatjets(big_fatjets, small_fatjets, *_seq, R / 10.);

        if (_debug) {
          edm::LogInfo("HEPTopTaggerV2") << "R = " << R << " -> n_small_fatjets = " << small_fatjets.size() << endl;
        }

        _n_small_fatjets[R] = small_fatjets.size();

        // We are sorting by pt - so start with a negative dummy
        double dummy = -99999;

        for (unsigned i = 0; i < small_fatjets.size(); i++) {
          HEPTopTaggerV2_fixed_R htt(small_fatjets[i]);
          htt.set_mass_drop_threshold(_mass_drop_threshold);
          htt.set_max_subjet_mass(_max_subjet_mass);
          htt.set_filtering_n(_nfilt);
          htt.set_filtering_R(_Rfilt);
          htt.set_filtering_minpt_subjet(_minpt_subjet);
          htt.set_filtering_jetalgorithm(_jet_algorithm_filter);
          htt.set_reclustering_jetalgorithm(_jet_algorithm_recluster);
          htt.set_mode(_mode);
          htt.set_mt(_mtmass);
          htt.set_mw(_mwmass);
          htt.set_top_mass_range(_mtmin, _mtmax);
          htt.set_mass_ratio_range(_rmin, _rmax);
          htt.set_mass_ratio_cut(_m23cut, _m13cutmin, _m13cutmax);
          htt.set_top_minpt(_minpt_tag);
          htt.set_pruning_zcut(_zcut);
          htt.set_pruning_rcut_factor(_rcut_factor);
          htt.set_debug(_debug);
          htt.set_qjets(_q_zcut, _q_dcut_fctr, _q_exp_min, _q_exp_max, _q_rigidity, _q_truncation_fctr);

          htt.run();

          if (htt.t().perp() > dummy) {
            dummy = htt.t().perp();
            _HEPTopTaggerV2[R] = htt;
          }
        }  //End of loop over small_fatjets

        // Only check if we have not found Ropt yet
        if (_Ropt == 0 && R < maxR) {
          // If the new mass is OUTSIDE the window ..
          if (_HEPTopTaggerV2[R].t().m() < (1 - _optimalR_threshold) * _HEPTopTaggerV2[maxR].t().m())
            // .. set _Ropt to the previous mass
            _Ropt = R + stepR;
        }

        big_fatjets = small_fatjets;
        small_fatjets.clear();
      }  //End of loop over R

      // if we did not find Ropt in the loop:
      if (_Ropt == 0 && _HEPTopTaggerV2[maxR].t().m() > 0) {
        // either pick the last value (_R_opt_reject_min=false)
        // or leace Ropt at zero (_R_opt_reject_min=true)
        if (_R_opt_reject_min == false)
          _Ropt = minR;
      }

      //for the case that there is no tag at all (< 3 hard substructures)
      if (_Ropt == 0 && _HEPTopTaggerV2[maxR].t().m() == 0)
        _Ropt = maxR;

      _HEPTopTaggerV2_opt = _HEPTopTaggerV2[_Ropt];

      Filter filter_optimalR_calc(_R_filt_optimalR_calc, SelectorNHardest(_N_filt_optimalR_calc));
      _R_opt_calc = _r_min_exp_function(filter_optimalR_calc(_fat).pt());
      _pt_for_R_opt_calc = filter_optimalR_calc(_fat).pt();

      Filter filter_optimalR_pass(_R_filt_optimalR_pass, SelectorNHardest(_N_filt_optimalR_pass));
      Filter filter_optimalR_fail(_R_filt_optimalR_fail, SelectorNHardest(_N_filt_optimalR_fail));
      if (optimalR_type() == 1) {
        _filt_fat = filter_optimalR_pass(_fat);
      } else {
        _filt_fat = filter_optimalR_fail(_fat);
      }
    }
  }

  //optimal_R type
  int HEPTopTaggerV2::optimalR_type() {
    if (_HEPTopTaggerV2_opt.t().m() < _optimalR_mmin || _HEPTopTaggerV2_opt.t().m() > _optimalR_mmax) {
      return 0;
    }
    if (_HEPTopTaggerV2_opt.f_rec() > _optimalR_fw) {
      return 0;
    }
    if (_Ropt / 10. - _R_opt_calc > _R_opt_diff) {
      return 0;
    }
    return 1;
  }

  double HEPTopTaggerV2::nsub_unfiltered(int order,
                                         fastjet::contrib::Njettiness::AxesMode axes,
                                         double beta,
                                         double R0) {
    fastjet::contrib::Nsubjettiness nsub(order, axes, beta, R0);
    return nsub.result(_fat);
  }

  double HEPTopTaggerV2::nsub_filtered(int order, fastjet::contrib::Njettiness::AxesMode axes, double beta, double R0) {
    fastjet::contrib::Nsubjettiness nsub(order, axes, beta, R0);
    return nsub.result(_filt_fat);
  }

  HEPTopTaggerV2::~HEPTopTaggerV2() {}
  // Do not change next line, it's needed by the sed-code that makes the tagger CMSSW-compatible.
};  // namespace external
