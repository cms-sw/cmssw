#ifndef __RecoJets_JetAlgorithms_CMSBoostedTauSeedingAlgorithm_h__
#define __RecoJets_JetAlgorithms_CMSBoostedTauSeedingAlgorithm_h__

// CMSBoostedTau Package
//
// Find subjets corresponding to decay products of tau lepton pair
// and produce data-formats neccessary to seed tau reconstruction.
//
// Questions/Comments? 
//    for physics : Christian.Veelken@cern.ch
//    for implementation : Salvatore.Rappoccio@cern.ch
//
//----------------------------------------------------------------------
// This file is part of FastJet contrib.
//
// It is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the
// Free Software Foundation; either version 2 of the License, or (at
// your option) any later version.
//
// It is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this code. If not, see <http://www.gnu.org/licenses/>.
//----------------------------------------------------------------------

#include <fastjet/internal/base.hh>
#include <fastjet/tools/Transformer.hh>
#include <fastjet/CompositeJetStructure.hh>

#include <fastjet/ClusterSequence.hh>
#include <fastjet/Error.hh>
#include <fastjet/JetDefinition.hh>
//#include "fastjet/FunctionOfPseudoJet.hh"

#include <map>
#include <sstream>
#include <string>

FASTJET_BEGIN_NAMESPACE      // defined in fastjet/internal/base.hh

namespace contrib{

  class CMSBoostedTauSeedingAlgorithmStructure;

  //------------------------------------------------------------------------
  /// \class CMSBoostedTauSeedingAlgorithm
  /// This class implements the CMS boosted tau algorithm
  ///
  class CMSBoostedTauSeedingAlgorithm : public Transformer {

  public:
    // constructors
    CMSBoostedTauSeedingAlgorithm(double ptMin, double muMin, double muMax, double yMin, double yMax, double dRMin, double dRMax, int maxDepth, int verbosity = 0);

    // destructor
    virtual ~CMSBoostedTauSeedingAlgorithm(){}

    // standard usage
    virtual std::string description() const;

    virtual PseudoJet result(const PseudoJet & jet) const;

    // the type of the associated structure
    typedef CMSBoostedTauSeedingAlgorithmStructure StructureType;

  protected:
    void dumpSubJetStructure(const fastjet::PseudoJet& jet, int depth, int maxDepth, const std::string& depth_and_idx_string) const;
    std::pair<PseudoJet, PseudoJet> findSubjets(const PseudoJet& jet, int depth, bool& subjetsFound) const;

  private: 
    double ptMin_;  ///< minimum sub-jet pt
    double muMin_;  ///< the min value of the mass-drop parameter
    double muMax_;  ///< the max value of the mass-drop parameter
    double yMin_;   ///< the min value of the asymmetry parameter
    double yMax_;   ///< the max value of the asymmetry parameter    
    double dRMin_;  ///< the min value of the dR parameter
    double dRMax_;  ///< the max value of the dR parameter
    int maxDepth_;  ///< the max depth for descending into clustering sequence

    int verbosity_; ///< flag to enable/disable debug output
  };


  //------------------------------------------------------------------------
  /// @ingroup tools_taggers
  /// \class CMSBoostedTauSeedingAlgorithmStructure
  /// the structure returned by the CMSBoostedTauSeedingAlgorithm transformer.
  ///
  /// See the CMSBoostedTauSeedingAlgorithm class description for the details of what
  /// is inside this structure
  ///
  class CMSBoostedTauSeedingAlgorithmStructure : public CompositeJetStructure {
  public:
    /// ctor with initialisation
    ///  \param pieces  the pieces of the created jet
    ///  \param rec     the recombiner from the underlying cluster sequence
    CMSBoostedTauSeedingAlgorithmStructure(const PseudoJet& result_jet, const JetDefinition::Recombiner* rec = 0) 
      : CompositeJetStructure(result_jet.pieces(), rec), 
        _mu(0.0), _y(0.0), _dR(0.0), _pt(0.0) 
    {}

    /// returns the mass-drop ratio, pieces[0].m()/jet.m()
    inline double mu() const { return _mu; }

    /// returns the value of y = (squared kt distance) / (squared mass) for the
    /// splitting that triggered the mass-drop condition
    inline double y() const { return _y; }

    /// returns the value of dR
    inline double dR() const { return _dR; }

    /// returns the value of pt
    inline double pt() const { return _pt; }

    // /// returns the original jet (before tagging)
    //const PseudoJet& original() const { return _original_jet; }

  protected:
    double _mu;              ///< the value of the mass-drop parameter
    double _y;               ///< the value of the asymmetry parameter
    double _dR;              ///< the value of the dR parameter
    double _pt;              ///< the value of the pt parameter
    // allow the tagger to set these
    friend class CMSBoostedTauSeedingAlgorithm;
  };


} // namespace contrib

FASTJET_END_NAMESPACE

#endif  // __FASTJET_CONTRIB_CMSBOOSTEDTAUSEEDINGALGORITHM_HH__
