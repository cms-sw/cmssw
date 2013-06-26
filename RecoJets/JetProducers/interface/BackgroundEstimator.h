#ifndef __FASTJET_BACKGROUND_EXTRACTOR_HH__
#define __FASTJET_BACKGROUND_EXTRACTOR_HH__

#include <fastjet/ClusterSequenceAreaBase.hh>
#include <fastjet/RangeDefinition.hh>
#include <iostream>

namespace fastjet{


/// \class BackgroundEstimator
/// Class to estimate the density of the background per unit area
///
/// The default behaviour of this class is to compute the global 
/// properties of the background as it is done in ClusterSequenceArea.
/// On top of that, we provide methods to specify an explicit set of
/// jets to use or a list of jets to exclude.
/// We also provide all sorts of additional information regarding
/// the background estimation like the jets that have been used or
/// the number of pure-ghost jets.
///
/// Default behaviour:
///   by default the list of included jets is the inclusive jets from
///   the given ClusterSequence; the list of explicitly excluded jets 
///   is empty; we use 4-vector area
///
/// Beware: 
///   by default, to correctly handle partially empty events, the
///   class attempts to calculate an "empty area", based
///   (schematically) on
///
///          range.total_area() - sum_{jets_in_range} jets.area()
///  
///   For ranges with small areas, this can be innacurate (particularly 
///   relevant in dense events where empty_area should be zero and ends
///   up not being zero).
///
///   This calculation of empty area can be avoided if you supply a
///   ClusterSequenceArea class with explicit ghosts
///   (ActiveAreaExplicitGhosts). This is _recommended_!
///
 class BackgroundEstimator{
public:
  /// default ctor
  /// \param csa      the ClusterSequenceArea to use
  /// \param range    the range over which jets will be considered
  BackgroundEstimator(const ClusterSequenceAreaBase &csa, const RangeDefinition &range);
  
  /// default dtor
  ~BackgroundEstimator();

  // retrieving information
  //-----------------------  

  /// get the median rho
  double median_rho(){ 
    _recompute_if_needed();
    return _median_rho;
  }

  /// synonym for median rho [[do we have this? Both?]]
  double rho() {return median_rho();}

  /// get the sigma
  double sigma() {
    _recompute_if_needed();
    return _sigma;
  }
  
  /// get the median area of the jets used to actually compute the background properties
  double mean_area(){
    _recompute_if_needed();
    return _mean_area;
  }
  
  /// get the number of jets used to actually compute the background properties
  unsigned int n_jets_used(){
    _recompute_if_needed();
    return _n_jets_used;
  }
  
  /// get the number of jets (within the given range) that have been
  /// explicitly excluded when computing the background properties
  unsigned int n_jets_excluded(){
    _recompute_if_needed();
    return _n_jets_excluded;
  }
  
  /// get the number of empty jets used when computing the background properties;
  /// (it is deduced from the empty area with an assumption about the average
  /// area of jets)
  double n_empty_jets(){
    _recompute_if_needed();
    return _n_empty_jets;
  }

  /// returns the estimate of the area (within Range) that is not occupied
  /// by the jets (excluded jets are removed from this count)
  double empty_area(){
    _recompute_if_needed();
    return _empty_area;
  }

  // configuring behaviour
  //----------------------  

  /// reset to default values
  /// set the list of included jets to the inclusive jets and clear the excluded ones
  void reset();

  /// specify if one uses the scalar or 4-vector area
  ///  \param use_it             whether one uses the 4-vector area or not (true by default)
  void set_use_area_4vector(bool use_it = true){
    _use_area_4vector = use_it;
  }
  
  /// set the list of included jets
  ///  \param included_jets      the list of jets to include
  ///  \param all_from_included  when true, we'll assume that the cluster sequence inclusive jets 
  ///                            give all the potential jets in the range. In practice this means 
  ///                            that the empty area will be computed from the inclusive jets rather
  ///                            than from the 'included_jets'. You can overwrite the default value 
  ///                            and send it to 'false' e.g. when the included_jets you provide are
  ///                            themselves a list of inclusive jets.
  void set_included_jets(const std::vector<PseudoJet> &included_jets, bool all_from_inclusive = true){
    _included_jets = included_jets;
    _all_from_inclusive = all_from_inclusive;
    _uptodate = false;
  }
  
  /// set the list of explicitly excluded jets
  ///  \param excluded_jets      the list of jets that have to be explicitly excluded
  void set_excluded_jets(const std::vector<PseudoJet> &excluded_jets){
    _excluded_jets = excluded_jets;
    _uptodate = false;  
  }
  
private:
  /// do the actual job
  void _compute();
  
  /// check if the properties need to be recomputed 
  /// and do so if needed
  void _recompute_if_needed(){
    if (!_uptodate)
      _compute();
    _uptodate = true;
  }
  
  // the information needed to do the computation
  const ClusterSequenceAreaBase &_csa;      ///< cluster sequence to get jets and areas from
  const RangeDefinition &_range;            ///< range to compute the background in
  std::vector<PseudoJet> _included_jets;    ///< jets to be used
  std::vector<PseudoJet> _excluded_jets;    ///< jets to be excluded
  bool _all_from_inclusive;                 ///< when true, we'll assume that the incl jets are the complete set
  bool _use_area_4vector;
  
  // the actual results of the computation
  double _median_rho;		            ///< background estimated density per unit area
  double _sigma;		            ///< background estimated fluctuations
  double _mean_area;		            ///< mean area of the jets used to estimate the background
  unsigned int _n_jets_used;                ///< number of jets used to estimate the background
  unsigned int _n_jets_excluded;            ///< number of jets that have explicitly been excluded
  double _n_empty_jets;                     ///< number of empty (pure-ghost) jets
  double _empty_area;                       ///< the empty (pure-ghost/unclustered) area!

  // internal variables
  bool _uptodate;                           ///< true when the background computation is up-to-date
};

} // namespace

#endif
