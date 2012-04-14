#ifndef HLTAlphaTFilter_h
#define HLTAlphaTFilter_h

/** \class HLTAlphaTFilter
 *
 *  \author Bryn Mathias
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}


//
// class declaration
//
template<typename T>
class HLTAlphaTFilter : public HLTFilter {

   public:
      explicit HLTAlphaTFilter(const edm::ParameterSet&);
      ~HLTAlphaTFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
      
   private:
      edm::InputTag inputJetTag_; // input tag identifying jets
      edm::InputTag inputJetTagFastJet_; // input tag identifying a second collection of jets
      std::vector<double> minPtJet_;
      std::vector<double> etaJet_;
      unsigned int maxNJets_;
      double minHt_;
      double minAlphaT_;
      int triggerType_;
};



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

    // Momentum sums in transverse plane
    const double sum_et = accumulate( et.begin(), et.end(), 0. );
    const double sum_px = accumulate( px.begin(), px.end(), 0. );
    const double sum_py = accumulate( py.begin(), py.end(), 0. );

    // Minimum Delta Et for two pseudo-jets
    double min_delta_sum_et = -1.;
    for ( unsigned i=0; i < unsigned(1<<(et.size()-1)); i++ ) { //@@ iterate through different combinations
      double delta_sum_et = 0.;
      std::vector<bool> jet;
      for ( unsigned j=0; j < et.size(); j++ ) { //@@ iterate through jets
  delta_sum_et += et[j] * ( 1 - 2 * (int(i>>j)&1) );
  if ( list ) { jet.push_back( (int(i>>j)&1) == 0 ); }
      }
      if ( ( std::abs(delta_sum_et) < min_delta_sum_et || min_delta_sum_et < 0. ) ) {
  min_delta_sum_et = std::abs(delta_sum_et);
  if ( list && jet.size() == et.size() ) {
    pseudo_jet1.resize(jet.size());
    std::copy( jet.begin(), jet.end(), pseudo_jet1.begin() );
  }
      }
    }
    if ( min_delta_sum_et < 0. ) { return 0.; }

    // Alpha_T
    return ( 0.5 * ( sum_et - min_delta_sum_et ) / sqrt( sum_et*sum_et - (sum_px*sum_px+sum_py*sum_py) ) );

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






#endif //HLTAlphaTFilter_h
