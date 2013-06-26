// Copyright (c) 2011 Christopher Vermilion
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

#ifndef __HEPTOPTAGGER_WRAPPER_HH__
#define __HEPTOPTAGGER_WRAPPER_HH__

#include <fastjet/tools/TopTaggerBase.hh>
#include <fastjet/CompositeJetStructure.hh>
#include <sstream>

FASTJET_BEGIN_NAMESPACE

/// A fastjet::Transformer wrapper for the HEP top tagger.  All of the work
/// is done by the HEPTopTagger class in HEPTopTagger.hh, written by
/// Tilman Plehn, Gavin Salam, Michael Spannowsky, and Michihisa Takeuchi.  The
/// HEP top tagger was described in Phys. Rev. Lett. 104 (2010) 111801
/// [arXiv:0910.5472] and JHEP 1010 (2010) 078 [arXiv:1006.2833].
/// 
///
/// This code is based on JHTopTagger.{hh,cc}, part of the FastJet package,
/// written by Matteo Cacciari, Gavin P. Salam and Gregory Soyez, and released
/// under the GNU General Public License.
///
/// The HEP top tagger produces information similar to the Johns Hopkins tagger.
///  Accordingly I simply reuse the JHTopTaggerStructure.
class HEPTopTaggerStructure;


class HEPTopTagger : public TopTaggerBase {
public:
  /// Sets two of the algorithm parameters
  ///
  /// \param mass_drop_threshold    A splitting is hard if 
  ///                                max(subjet_m) < mass_drop_threshold * m_child
  /// \param max_subjet_mass        The tagger attempts to split subjets until
  ///                                remaining subjets have m_subjet < max_subjet_mass.
  /// \param use_subjet_mass_cuts   Whether to impose the subjet mass cuts described
  ///                                in arXiv:1006.2833 (default=false)
  /// Default values are taken from the original HepTopTagger.hh code.
  HEPTopTagger(double mass_drop_threshold=0.8, double max_subjet_mass=30.,
               bool use_subjet_mass_cuts=false)
    : _mass_drop_threshold(mass_drop_threshold),
      _max_subjet_mass(max_subjet_mass),
      _use_subjet_mass_cuts(use_subjet_mass_cuts)
  {}

  /// returns a textual description of the tagger
  virtual std::string description() const;

  /// runs the tagger on the given jet and
  /// returns the tagged PseudoJet if successful, or a PseudoJet==0 otherwise
  /// (standard access is through operator()).
  ///  \param jet   the PseudoJet to tag
  virtual PseudoJet result(const PseudoJet & jet) const;

  // the type of the associated structure
  typedef HEPTopTaggerStructure StructureType;

private:
  double _mass_drop_threshold;
  double _max_subjet_mass;
  bool _use_subjet_mass_cuts; ///< whether to include the is_masscut_passed() test
};


/// Basically just a copy of JHTopTaggerStructure, but this way HEPTopTagger can
/// be a friend.

//BEGIN COMMENTING OUT BY CHRISTOPHER SILKWORTH
/*
class HEPTopTaggerStructure : public JHTopTaggerStructure {
public:
  HEPTopTaggerStructure(std::vector<PseudoJet> pieces,
      const JetDefinition::Recombiner *recombiner = 0)
    : JHTopTaggerStructure(pieces, recombiner) {}

protected:
  friend class HEPTopTagger;
};
*/
//END COMMENTING OUT

//BEGIN ADDED BY CHRISTOPHER SILKWORTH


class HEPTopTaggerStructure : public CompositeJetStructure, public TopTaggerBaseStructure {
 public:
   /// ctor with pieces initialisation
   HEPTopTaggerStructure(const std::vector<PseudoJet>& pieces_in,
                  const JetDefinition::Recombiner *recombiner = 0) :
      CompositeJetStructure(pieces_in, recombiner), _cos_theta_w(0.0),W_rec(recombiner), 
      rW_()
	{}
 
   /// returns the W subjet
      inline PseudoJet const & W() const{ 
         rW_ = join(_pieces[0], _pieces[1], *W_rec);
         return rW_;
      }
 
      
      inline PseudoJet  W1() const{
         assert(W().pieces().size()>0);
         return W().pieces()[0];
      }
      
      /// returns the second W subjet
      inline PseudoJet W2() const{
         assert(W().pieces().size()>1);
         return W().pieces()[1];
      }


   /// returns the non-W subjet
   /// It will have 1 or 2 pieces depending on whether the tagger has
   /// found 3 or 4 pieces
   inline const PseudoJet & non_W() const{ 
     return _pieces[2];
   }
 
   /// returns the W helicity angle
   inline double cos_theta_W() const {return _cos_theta_w;}
 
 //  /// returns the original jet (before tagging)
 //  const PseudoJet & original() const {return _original_jet;}

 
 
 protected:
      double _cos_theta_w; ///< the W helicity angle
      const JetDefinition::Recombiner  * W_rec;
   //PseudoJet _W;             ///< the tagged W
   //PseudoJet _non_W;         ///< the remaining pieces
 //  PseudoJet _original_jet;  ///< the original jet (before tagging)
 
      mutable PseudoJet rW_;

   // allow the tagger to set these
   friend class HEPTopTagger;
 };


//END ADDED BY CHRISTOPHER SILKWORTH


//------------------------------------------------------------------------
// description of the tagger
inline std::string HEPTopTagger::description() const{ 
  std::ostringstream oss;
  oss << "HEPTopTagger with {max. subjet mass = " << _max_subjet_mass
      << ", mass-drop threshold = " << _mass_drop_threshold
      << ", and " << (_use_subjet_mass_cuts ? "using" : "not using") << " subjet mass cuts" << std::endl;
  oss << description_of_selectors();
  return oss.str();
}


FASTJET_END_NAMESPACE

#endif // __HEPTOPTAGGER_HH__
