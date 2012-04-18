#ifndef HLTrigger_JetMET_AlphaT_h
#define HLTrigger_JetMET_AlphaT_h

struct AlphaT {

  // -----------------------------------------------------------------------------
  //
  struct fabs_less {
    bool operator()( const double x, const double y ) const { return std::abs(x) < std::abs(y); }
  };

  // -----------------------------------------------------------------------------
  //
  template<class LorentzV>
  double operator()( const std::vector<LorentzV const *>& p4,
         bool use_et = true ) const {

    if ( p4.size() == 0 ) { return 0; }

    std::vector<double> et;
    std::vector<double> px;
    std::vector<double> py;

    transform( p4.begin(), p4.end(), back_inserter(et), ( use_et ? std::mem_fun(&LorentzV::Et) : std::mem_fun(&LorentzV::Pt) ) );
    transform( p4.begin(), p4.end(), back_inserter(px), std::mem_fun(&LorentzV::Px) );
    transform( p4.begin(), p4.end(), back_inserter(py), std::mem_fun(&LorentzV::Py) );

    return value( et, px, py );

  }

  // -----------------------------------------------------------------------------
  //
  template<class LorentzV>
  double operator()( const std::vector<LorentzV const *>& p4,
         std::vector<bool>& pseudo_jet1,
         bool use_et = true ) const {

    if ( p4.size() == 0 ) { return 0; }

    std::vector<double> et;
    std::vector<double> px;
    std::vector<double> py;
    pseudo_jet1.clear();

    transform( p4.begin(), p4.end(), back_inserter(et), std::mem_fun( use_et ? &LorentzV::Et : &LorentzV::Pt ) );
    transform( p4.begin(), p4.end(), back_inserter(px), std::mem_fun(&LorentzV::Px) );
    transform( p4.begin(), p4.end(), back_inserter(py), std::mem_fun(&LorentzV::Py) );

    return value( et, px, py, pseudo_jet1 );

  }

  // -----------------------------------------------------------------------------
  //
  template<class LorentzV>
  double operator()( const std::vector<LorentzV>& p4,
         bool use_et = true ) const {

    if ( p4.size() == 0 ) { return 0; }

    std::vector<double> et;
    std::vector<double> px;
    std::vector<double> py;

    transform( p4.begin(), p4.end(), back_inserter(et), std::mem_fun_ref( use_et ? &LorentzV::Et : &LorentzV::Pt ) );
    transform( p4.begin(), p4.end(), back_inserter(px), std::mem_fun_ref(&LorentzV::Px) );
    transform( p4.begin(), p4.end(), back_inserter(py), std::mem_fun_ref(&LorentzV::Py) );

    return value( et, px, py );

  }


  // -----------------------------------------------------------------------------
  //
  template<class LorentzV>
  double operator()( const std::vector<LorentzV>& p4,
         std::vector<bool>& pseudo_jet1,
         bool use_et = true ) const {

    if ( p4.size() == 0 ) { return 0; }

    std::vector<double> et;
    std::vector<double> px;
    std::vector<double> py;

    transform( p4.begin(), p4.end(), back_inserter(et), std::mem_fun_ref( use_et ? &LorentzV::Et : &LorentzV::Pt ) );
    transform( p4.begin(), p4.end(), back_inserter(px), std::mem_fun_ref(&LorentzV::Px) );
    transform( p4.begin(), p4.end(), back_inserter(py), std::mem_fun_ref(&LorentzV::Py) );

    return value( et, px, py, pseudo_jet1 );

  }

  // -----------------------------------------------------------------------------
  //
  static double value( const std::vector<double>& et,
           const std::vector<double>& px,
           const std::vector<double>& py,
           std::vector<bool>& pseudo_jet1,
           bool list = true ) {

    // Clear pseudo-jet container
    pseudo_jet1.clear();
    pseudo_jet1.resize(et.size());

    // check the size of the input collection
    if (et.size() == 0)
      // empty jet collection, return AlphaT = 0
      return 0.;

    if (et.size() > (unsigned int) std::numeric_limits<unsigned int>::digits)
      // too many jets, return AlphaT = a very large number
      return std::numeric_limits<double>::max();

    // Momentum sums in transverse plane
    const double sum_et = accumulate( et.begin(), et.end(), 0. );
    const double sum_px = accumulate( px.begin(), px.end(), 0. );
    const double sum_py = accumulate( py.begin(), py.end(), 0. );

    // Minimum Delta Et for two pseudo-jets
    double min_delta_sum_et = sum_et;

    for (unsigned int i = 0; i < (1U << (et.size() - 1)); i++) { //@@ iterate through different combinations
      double delta_sum_et = 0.;
      for (unsigned int j = 0; j < et.size(); ++j) { //@@ iterate through jets
        if (i & (1U << j))
          delta_sum_et -= et[j];
        else
          delta_sum_et += et[j];
      }
      delta_sum_et = std::abs(delta_sum_et);
      if (delta_sum_et < min_delta_sum_et) {
        min_delta_sum_et = delta_sum_et;
        if (list) {
          for (unsigned int j = 0; j < et.size(); ++j)
            pseudo_jet1[j] = ((i & (1U << j)) == 0);
        }
      }
    }

    // Alpha_T
    return (0.5 * (sum_et - min_delta_sum_et) / sqrt( sum_et*sum_et - (sum_px*sum_px+sum_py*sum_py) ));  
  }

  // -----------------------------------------------------------------------------
  //
  static double value( const std::vector<double>& et,
           const std::vector<double>& px,
           const std::vector<double>& py ) {
    std::vector<bool> pseudo_jet1;
    return value( et, px, py, pseudo_jet1, false );
  }

};

#endif // HLTrigger_JetMET_AlphaT_h
