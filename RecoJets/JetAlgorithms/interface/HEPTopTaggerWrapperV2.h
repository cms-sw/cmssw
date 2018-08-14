//  2011 Christopher Vermilion
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

#ifndef __HEPTOPTAGGER_WRAPPERV2_HH__
#define __HEPTOPTAGGER_WRAPPERV2_HH__

#include <fastjet/tools/TopTaggerBase.hh>
#include <fastjet/CompositeJetStructure.hh>
#include "CLHEP/Random/RandomEngine.h"
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

// Removed legacy comments by CHRISTOPHER SILKWORTH

class HEPTopTaggerV2Structure;


class HEPTopTaggerV2 : public TopTaggerBase {
public:
 HEPTopTaggerV2(bool DoOptimalR,
		bool DoQjets,
		double minSubjetPt, 
		double minCandPt, 
		double subjetMass, 
		double muCut, 
		double filtR,
		int filtN,
		int mode, 
		double minCandMass, 
		double maxCandMass, 
		double massRatioWidth, 
		double minM23Cut, 
		double minM13Cut, 
		double maxM13Cut,
		bool optRrejectMin) : DoOptimalR_(DoOptimalR),
    DoQjets_(DoQjets),
    minSubjetPt_(minSubjetPt),
    minCandPt_(minCandPt),
    subjetMass_(subjetMass),
    muCut_(muCut),
    filtR_(filtR),
    filtN_(filtN),
    mode_(mode),
    minCandMass_(minCandMass),
    maxCandMass_(maxCandMass),
    massRatioWidth_(massRatioWidth),
    minM23Cut_(minM23Cut),
    minM13Cut_(minM13Cut),
    maxM13Cut_(maxM13Cut),
    optRrejectMin_(optRrejectMin),
    engine_(nullptr)
  {}

  /// returns a textual description of the tagger
  std::string description() const override;

  /// runs the tagger on the given jet and
  /// returns the tagged PseudoJet if successful, or a PseudoJet==0 otherwise
  /// (standard access is through operator()).
  ///  \param jet   the PseudoJet to tag
  PseudoJet result(const PseudoJet & jet) const override;

  void set_rng(CLHEP::HepRandomEngine* engine){ engine_ = engine;}

  // the type of the associated structure
  typedef HEPTopTaggerV2Structure StructureType;

private:
    bool DoOptimalR_; // Use optimalR mode
    bool DoQjets_; // Use qjet mode

    double minSubjetPt_; // Minimal pT for subjets [GeV]
    double minCandPt_;   // Minimal pT to return a candidate [GeV]
 
    double subjetMass_; // Mass above which subjets are further unclustered
    double muCut_; // Mass drop threshold
    
    double filtR_; // maximal filtering radius
    int filtN_; // number of filtered subjets to use

    // HEPTopTagger Mode
    // 0: do 2d-plane, return candidate with delta m_top minimal
    // 1: return candidate with delta m_top minimal IF passes 2d plane
    // 2: do 2d-plane, return candidate with max dj_sum
    // 3: return candidate with max dj_sum IF passes 2d plane
    // 4: return candidate built from leading three subjets after unclustering IF passes 2d plane
    // Note: Original HTT was mode==1    
    int mode_; 

    // Top Quark mass window in GeV
    double minCandMass_;
    double maxCandMass_;
    
    double massRatioWidth_; // One sided width of the A-shaped window around m_W/m_top in %
    double minM23Cut_; // minimal value of m23/m123
    double minM13Cut_; // minimal value of atan(m13/m12)
    double maxM13Cut_; // maximal value of atan(m13/m12)
  
    bool optRrejectMin_; // set Ropt to zero for candidates that never leave the window around the initial mass
                         // otherwise (default) set them to R=0.5

    // Random engine for Q-jet HTT
    CLHEP::HepRandomEngine* engine_;
};


class HEPTopTaggerV2Structure : public CompositeJetStructure, public TopTaggerBaseStructure {

 public:
   /// ctor with pieces initialisation
   HEPTopTaggerV2Structure(const std::vector<PseudoJet>& pieces_in,
                  const JetDefinition::Recombiner *recombiner = nullptr) : CompositeJetStructure(pieces_in, recombiner),
    _fj_mass(0.0),
    _fj_pt(0.0),
    _fj_eta(0.0),
    _fj_phi(0.0),
    _top_mass(0.0),
    _unfiltered_mass(0.0),
    _pruned_mass(0.0),
    _fRec(-1.),
    _mass_ratio_passed(-1),
    _ptForRoptCalc(-1),
    _tau1Unfiltered(-1.),
    _tau2Unfiltered(-1.),
    _tau3Unfiltered(-1.),
    _tau1Filtered(-1.),
    _tau2Filtered(-1.),
    _tau3Filtered(-1.),
    _qweight(-1.),    
    _qepsilon(-1.),    
    _qsigmaM(-1.),    
    W_rec(recombiner), 
    rW_(){}
  
   // Return W subjet
   inline PseudoJet const & W() const override{ 
     rW_ = join(_pieces[0], _pieces[1], *W_rec);
     return rW_;
   }
     
   // Return leading subjet in W
   inline PseudoJet  W1() const{
     assert(!W().pieces().empty());
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
   inline const PseudoJet & non_W() const override{ 
     return _pieces[2];
   }
 
   /// return the mass of the initial fatjet
   inline double fj_mass() const {return _fj_mass;}

   /// return the pt of the initial fatjet
   inline double fj_pt() const {return _fj_pt;}

   /// return the eta of the initial fatjet
   inline double fj_eta() const {return _fj_eta;}

   /// return the phi of the initial fatjet
   inline double fj_phi() const {return _fj_phi;}

   /// returns the candidate mass
   inline double top_mass() const {return _top_mass;}

   /// returns the unfiltered mass
   inline double unfiltered_mass() const {return _unfiltered_mass;}

   /// returns the pruned mass
   inline double pruned_mass() const {return _pruned_mass;}

   /// returns fRec
   inline double fRec() const {return _fRec;}

   /// returns if 2d-mass plane cuts were passed
   inline double mass_ratio_passed() const {return _mass_ratio_passed;}

   /// returns Ropt
   inline double ropt() const {return _ropt;}

   /// returns calculated Ropt
   inline double roptCalc() const {return _roptCalc;}

   /// returns the filtered pT for calculating R_opt
   inline double ptForRoptCalc() const {return _ptForRoptCalc;}

   // Nsubjettiness and Q-jet variables
   inline double Tau1Unfiltered() const {return _tau1Unfiltered;}
   inline double Tau2Unfiltered() const {return _tau2Unfiltered;}
   inline double Tau3Unfiltered() const {return _tau3Unfiltered;}
   inline double Tau1Filtered() const {return _tau1Filtered;}
   inline double Tau2Filtered() const {return _tau2Filtered;}
   inline double Tau3Filtered() const {return _tau3Filtered;}

   inline double qweight() const {return _qweight;}
   inline double qepsilon() const {return _qepsilon;}
   inline double qsigmaM() const {return _qsigmaM;}   
    
 protected:

      double _fj_mass;
      double _fj_pt;
      double _fj_eta;
      double _fj_phi;

      double _top_mass;
      double _unfiltered_mass;
      double _pruned_mass;
      double _fRec;
      int _mass_ratio_passed;
      double _ptForRoptCalc;
      double _ropt;
      double _roptCalc;

      double _tau1Unfiltered;
      double _tau2Unfiltered;
      double _tau3Unfiltered;
      double _tau1Filtered;
      double _tau2Filtered;
      double _tau3Filtered;
      double _qweight;
      double _qepsilon;
      double _qsigmaM;

      const JetDefinition::Recombiner  * W_rec;
 
      mutable PseudoJet rW_;

   // allow the tagger to set these
   friend class HEPTopTaggerV2;
 };


//------------------------------------------------------------------------
// description of the tagger
inline std::string HEPTopTaggerV2::description() const{ 

  std::ostringstream oss;
  oss << "HEPTopTaggerV2 with: "
      << "minSubjetPt = " << minSubjetPt_ 
      << "minCandPt = " << minCandPt_ 
      << "subjetMass = " << subjetMass_ 
      << "muCut = " << muCut_ 
      << "filtR = " << filtR_ 
      << "filtN = " << filtN_     
      << "mode = " << mode_ 
      << "minCandMass = " << minCandMass_ 
      << "maxCandMass = " << maxCandMass_ 
      << "massRatioWidth = " << massRatioWidth_ 
      << "minM23Cut = " << minM23Cut_ 
      << "minM13Cut = " << minM13Cut_ 
      << "maxM13Cut = " << maxM13Cut_ << std::endl;
  return oss.str();
}


FASTJET_END_NAMESPACE

#endif // __HEPTOPTAGGER_HH__
