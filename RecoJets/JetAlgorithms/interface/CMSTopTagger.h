// Original JHTopTagger code copyright (c) 2011 Matteo Cacciari, Gregory Soyez,
//  and Gavin Salam.
// Modifications to make it CMSTopTagger copyright (c) 2011 Christopher
//  Vermilion.
//
//----------------------------------------------------------------------
//  This file is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 3 of the License, or
//  (at your option) any later version.
//
//  This file is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  The GNU General Public License is available at
//  http://www.gnu.org/licenses/gpl.html or you can write to the Free Software
//  Foundation, Inc.:
//      59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//----------------------------------------------------------------------

#ifndef __CMSTOPTAGGER_HH__
#define __CMSTOPTAGGER_HH__

#include <fastjet/tools/JHTopTagger.hh>
#include <fastjet/CompositeJetStructure.hh>
#include <fastjet/LimitedWarning.hh>
#include <fastjet/Error.hh>
#include <fastjet/JetDefinition.hh>
#include <fastjet/ClusterSequence.hh>

#include <sstream>
#include <limits>

FASTJET_BEGIN_NAMESPACE

/// An implementation of the "CMS Top Tagger", as described in CMS-PAS-JME-10-013,
/// which is based on the "Johns Hopkins" top tagger (arXiv:0806.0848; Kaplan,
/// Rehermann, Schwartz and Tweedie).  An earlier version of this tagger was
/// described in CMS-PAS-JME-09-001; both implementations can be used via
/// different values passed to the constructor (see below).
///
/// This code is based on JHTopTagger.{hh,cc}, part of the FastJet package,
/// written by Matteo Cacciari, Gavin P. Salam and Gregory Soyez, and released
/// under the GNU General Public License.
///
/// The CMS top tagger is extremely similar to JH; the difference is in the
/// choice of parameters and in how the W is identified.  Accordingly I reuse
/// the JHTopTaggerStructure.
class CMSTopTaggerStructure;


class CMSTopTagger : public TopTaggerBase {
public:
  /// The parameters are the following:
  ///  \param delta_p        fractional pt cut imposed on the subjets
  ///                         (computed as a fraction of the original jet)
  ///  \param delta_r        minimum distance between 2 subjets
  ///                         (computed as sqrt((y1-y2)^2+(phi1-phi2)^2))
  ///  \param A              the actual DeltaR cut is (delta_r - A * pT_child)
  ///
  /// The default values of these parameters are taken from
  /// CMS-PAS-JME-10-013.  For the older tagger described in CMS-PAS-JME-09-001,
  /// use delta_p=0.05, delta_r=0.0, A=0.0
  CMSTopTagger(double delta_p=0.05, double delta_r=0.4, double A=0.0004);

  /// returns a textual description of the tagger
  virtual std::string description() const;

  /// runs the tagger on the given jet and
  /// returns the tagged PseudoJet if successful, or a PseudoJet==0 otherwise
  /// (standard access is through operator()).
  ///  \param jet   the PseudoJet to tag
  virtual PseudoJet result(const PseudoJet & jet) const;

  // the type of the associated structure
  typedef CMSTopTaggerStructure StructureType;
        
protected:
  /// runs the Johns Hopkins decomposition procedure
  std::vector<PseudoJet> _split_once(const PseudoJet & jet_to_split,
                                     const PseudoJet & reference_jet) const;

  /// find the indices corresponding to the minimum mass pairing in subjets
  /// only considers the hardest 3
  void _find_min_mass(const std::vector<PseudoJet>& subjets, int& i, int& j) const;

  double _delta_p, _delta_r, _A;
  mutable LimitedWarning _warnings_nonca;
};

/// Basically just a copy of JHTopTaggerStructure, but this way CMSTopTagger can
/// be a friend.
class CMSTopTaggerStructure : public JHTopTaggerStructure {
public:
  CMSTopTaggerStructure(const std::vector<PseudoJet>& pieces,
      const JetDefinition::Recombiner *recombiner = 0)
    : JHTopTaggerStructure(pieces, recombiner) {}

protected:
  friend class CMSTopTagger;
};




//----------------------------------------------------------------------
inline CMSTopTagger::CMSTopTagger(double delta_p, double delta_r, double A)
  : _delta_p(delta_p), _delta_r(delta_r), _A(A) {}

//------------------------------------------------------------------------
// description of the tagger
inline std::string CMSTopTagger::description() const{ 
  std::ostringstream oss;
  oss << "CMSTopTagger with delta_p=" << _delta_p << ", delta_r=" << _delta_r
      << ", and A=" << _A;
  oss << description_of_selectors();
  return oss.str();
}


//------------------------------------------------------------------------
// returns the tagged PseudoJet if successful, 0 otherwise
//  - jet   the PseudoJet to tag
inline PseudoJet CMSTopTagger::result(const PseudoJet & jet) const{
  // make sure that there is a "regular" cluster sequence associated
  // with the jet. Note that we also check it is valid (to avoid a
  // more criptic error later on)
  if (!jet.has_valid_cluster_sequence()){
    throw Error("CMSTopTagger can only be applied on jets having an associated (and valid) ClusterSequence");
  }

  // warn if the jet has not been clustered with a Cambridge/Aachen
  // algorithm
  if (! jet.validated_cs()->jet_def().jet_algorithm() == cambridge_algorithm)
    _warnings_nonca.warn("CMSTopTagger should only be applied on jets from a Cambridge/Aachen clustering; use it with other algorithms at your own risk.");


  // do the first splitting
  std::vector<PseudoJet> split0 = _split_once(jet, jet);
  if (split0.size() == 0) return PseudoJet();

  // now try a second splitting on each of the resulting objects
  std::vector<PseudoJet> subjets;
  for (unsigned i = 0; i < 2; i++) {
    std::vector<PseudoJet> split1 = _split_once(split0[i], jet);
    if (split1.size() > 0) {
      subjets.push_back(split1[0]);
      subjets.push_back(split1[1]);
    } else {
      subjets.push_back(split0[i]);
    }
  }

  // make sure things make sense
  if (subjets.size() < 3) return PseudoJet();

  // now find the pair of subjets with minimum mass (only taking hardest three)
  int ii=-1, jj=-1;
  _find_min_mass(subjets, ii, jj);

  // order the subjets in the following order:
  //  - hardest of the W subjets
  //  - softest of the W subjets
  //  - hardest of the remaining subjets
  //  - softest of the remaining subjets (if any)
  if (ii>0) std::swap(subjets[ii], subjets[0]);
  if (jj>1) std::swap(subjets[jj], subjets[1]);
  if (subjets[0].perp2() < subjets[1].perp2()) std::swap(subjets[0], subjets[1]);
  if ((subjets.size()>3) && (subjets[2].perp2() < subjets[3].perp2())) 
    std::swap(subjets[2], subjets[3]);
  
  // create the result and its structure
  const JetDefinition::Recombiner *rec
    = jet.associated_cluster_sequence()->jet_def().recombiner();

  PseudoJet W = join(subjets[0], subjets[1], *rec);
  PseudoJet non_W;
  if (subjets.size()>3) {
    non_W = join(subjets[2], subjets[3], *rec);
  } else {
    non_W = join(subjets[2], *rec);
  }
  PseudoJet result = join<CMSTopTaggerStructure>(W, non_W, *rec);
  CMSTopTaggerStructure *s = (CMSTopTaggerStructure*) result.structure_non_const_ptr();
  s->_cos_theta_w = _cos_theta_W(result);

  // Note that we could perhaps ensure this cut before constructing
  // the result structure but this has the advantage that the top
  // 4-vector is already available and does not have to de re-computed
  if (!_top_selector.pass(result) || ! _W_selector.pass(W)) {
    result *= 0.0;
  }

  result = join(subjets); //Added by J. Pilot to combine the (up to 4) subjets identified in the decomposition instead of just the W and non_W components

  return result;
}


// runs the Johns Hopkins decomposition procedure
inline std::vector<PseudoJet> CMSTopTagger::_split_once(const PseudoJet & jet_to_split,
                                           const PseudoJet & reference_jet) const{
  PseudoJet this_jet = jet_to_split;
  PseudoJet p1, p2;
  std::vector<PseudoJet> result;
  while (this_jet.has_parents(p1, p2)) {
    if (p2.perp2() > p1.perp2()) std::swap(p1,p2); // order with hardness
    if (p1.perp() < _delta_p * reference_jet.perp()) break; // harder is too soft wrt original jet
    double DR = p1.delta_R(p2);
    if (DR < _delta_r - _A * this_jet.perp()) break; // distance is too small
    if (p2.perp() < _delta_p * reference_jet.perp()) {
      this_jet = p1; // softer is too soft wrt original, so ignore it
      continue; 
    }
    //result.push_back(this_jet);
    result.push_back(p1);
    result.push_back(p2);
    break;
  }
  return result;
}


// find the indices corresponding to the minimum mass pairing in subjets
inline void CMSTopTagger::_find_min_mass(const std::vector<PseudoJet>& subjets, int& i, int& j) const{
  assert(subjets.size() > 1 && subjets.size() < 5);
  
  // if four subjets found, only consider three hardest
  unsigned softest = 5; // invalid value
  if (subjets.size() == 4) {
    double min_pt = std::numeric_limits<double>::max();;
    for (unsigned ii = 0; ii < subjets.size(); ++ii) {
      if (subjets[ii].perp() < min_pt) {
        min_pt = subjets[ii].perp();
        softest = ii;
      }
    }
  }
  
  double min_mass = std::numeric_limits<double>::max();
  for (unsigned ii = 0; ii+1 < subjets.size(); ++ii) { // don't do size()-1: (unsigned(0)-1 != -1) !!
    if (ii == softest) continue;
    for (unsigned jj = ii + 1; jj < subjets.size(); ++jj) {
      if (jj == softest) continue;
      if ((subjets[ii]+subjets[jj]).m() < min_mass) {
        min_mass = (subjets[ii]+subjets[jj]).m();
        i = ii;
        j = jj;
      }
    }
  }
}


FASTJET_END_NAMESPACE

#endif // __CMSTOPTAGGER_HH__
