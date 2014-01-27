// CMSBoostedTauSeedingAlgorithm Package
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

#include "RecoJets/JetAlgorithms/interface/CMSBoostedTauSeedingAlgorithm.h"

FASTJET_BEGIN_NAMESPACE      // defined in fastjet/internal/base.hh

namespace contrib{


  /////////////////////////////
  // constructor
  CMSBoostedTauSeedingAlgorithm::CMSBoostedTauSeedingAlgorithm(double iminPt, 			       
							       double iminMassDrop, double imaxMassDrop,
							       double iminY, double imaxY,
							       double iminDeltaR, double imaxDeltaR,
							       int maxDepth, 
							       int verbosity) 
    : ptMin_(iminPt), 
      muMin_(iminMassDrop), muMax_(imaxMassDrop), 
      yMin_(iminY), yMax_(imaxY), 
      dRMin_(iminDeltaR), dRMax_(imaxDeltaR),
      maxDepth_(maxDepth),
      verbosity_(verbosity)
  {}

  /////////////////////////////
  // description
  std::string CMSBoostedTauSeedingAlgorithm::description() const {
    std::ostringstream oss;
    oss << "CMSBoostedTauSeedingAlgorithm algorithm";
    return oss.str();
  }


  /////////////////////////////
  void CMSBoostedTauSeedingAlgorithm::dumpSubJetStructure(const fastjet::PseudoJet& jet, int depth, int maxDepth, const std::string& depth_and_idx_string) const
  {
    if ( maxDepth != -1 && depth > maxDepth ) return;
    fastjet::PseudoJet subjet1, subjet2; 
    bool hasSubjets = jet.has_parents(subjet1, subjet2);
    if ( !hasSubjets ) return;
    std::string depth_and_idx_string_subjet1 = depth_and_idx_string;
    if ( depth_and_idx_string_subjet1.length() > 0 ) depth_and_idx_string_subjet1.append(".");
    depth_and_idx_string_subjet1.append("0");
    for ( int iSpace = 0; iSpace < depth; ++iSpace ) {
      std::cout << " ";
    }
    std::cout << " jetConstituent #" << depth_and_idx_string_subjet1 << " (depth = " << depth << "): Pt = " << subjet1.pt() << "," 
	      << " eta = " << subjet1.eta() << ", phi = " << subjet1.phi() << ", mass = " << subjet1.m() 
	      << " (constituents = " << subjet1.constituents().size() << ")" << std::endl;
    dumpSubJetStructure(subjet1, depth + 1, maxDepth, depth_and_idx_string_subjet1);
    std::string depth_and_idx_string_subjet2 = depth_and_idx_string;
    if ( depth_and_idx_string_subjet2.length() > 0 ) depth_and_idx_string_subjet2.append(".");
    depth_and_idx_string_subjet2.append("1");
    for ( int iSpace = 0; iSpace < depth; ++iSpace ) {
      std::cout << " ";
    }
    std::cout << " jetConstituent #" << depth_and_idx_string_subjet2 << " (depth = " << depth << "): Pt = " << subjet2.pt() << "," 
	      << " eta = " << subjet2.eta() << ", phi = " << subjet2.phi() << ", mass = " << subjet2.m() 
	      << " (constituents = " << subjet2.constituents().size() << ")" << std::endl;
    dumpSubJetStructure(subjet2, depth + 1, maxDepth, depth_and_idx_string_subjet2);
    for ( int iSpace = 0; iSpace < depth; ++iSpace ) {
      std::cout << " ";
    }
    double dR = subjet1.delta_R(subjet2);
    std::cout << " (mass-drop @ " << depth_and_idx_string << " = " << std::max(subjet1.m(), subjet2.m())/jet.m() << ", dR = " << dR << ")" << std::endl;
  }

  std::pair<PseudoJet, PseudoJet> CMSBoostedTauSeedingAlgorithm::findSubjets(const PseudoJet& jet, int depth, bool& subjetsFound) const 
  {
    if ( verbosity_ >= 2 ) {
      std::cout << "<CMSBoostedTauSeedingAlgorithm::findSubjets>:" << std::endl;
      std::cout << " jet: Pt = " << jet.pt() << ", eta = " << jet.eta() << ", phi = " << jet.phi() << ", mass = " << jet.m() << std::endl;
    }

    PseudoJet subjet1, subjet2;
    bool hasSubjets = jet.has_parents(subjet1, subjet2);
    if ( hasSubjets && (maxDepth_ == -1 || depth <= maxDepth_) ) {
      // make subjet1 the more massive jet
      if ( subjet1.m2() < subjet2.m2() ) {
	std::swap(subjet1, subjet2);
      }
      double dR = subjet1.delta_R(subjet2);
      double kT = subjet1.kt_distance(subjet2);
      double mu = ( jet.m() > 0. ) ? 
	sqrt(std::max(subjet1.m2(), subjet2.m2())/jet.m2()) : -1.;
      // check if subjets pass selection required for seeding boosted tau reconstruction
      if ( subjet1.pt() > ptMin_ && subjet2.pt() > ptMin_ && dR > dRMin_ && dR < dRMax_ && mu > muMin_ && mu < muMax_ && kT < (yMax_*jet.m2()) && kT > (yMin_*jet.m2()) ) {
	subjetsFound = true;
	return std::make_pair(subjet1, subjet2); 
      } else if ( subjet1.pt() > ptMin_ ) {
	return findSubjets(subjet1, depth + 1, subjetsFound);
      } else if ( subjet2.pt() > ptMin_ ) {
	return findSubjets(subjet2, depth + 1, subjetsFound);
      } 
    }
    subjetsFound = false;
    PseudoJet dummy_subjet1, dummy_subjet2;
    return std::make_pair(dummy_subjet1, dummy_subjet2); 
  }

  PseudoJet CMSBoostedTauSeedingAlgorithm::result(const PseudoJet& jet) const 
  {
    if ( verbosity_ >= 1 ) {
      std::cout << "<CMSBoostedTauSeedingAlgorithm::findSubjets>:" << std::endl;
      std::cout << " jet: Pt = " << jet.pt() << ", eta = " << jet.eta() << ", phi = " << jet.phi() << ", mass = " << jet.m() << std::endl;
    }

    if ( verbosity_ >= 2 ) {
      dumpSubJetStructure(jet, 0, maxDepth_, "");
    }

    bool subjetsFound = false;
    std::pair<PseudoJet, PseudoJet> subjets = findSubjets(jet, 0, subjetsFound);
    if ( subjetsFound ) {
      // fill structure for returning result
      PseudoJet subjet1 = subjets.first;
      PseudoJet subjet2 = subjets.second;
      if ( verbosity_ >= 1 ) {
	std::cout << "before recombination:" << std::endl;
	std::cout << " subjet #1: Pt = " << subjet1.pt() << ", eta = " << subjet1.eta() << ", phi = " << subjet1.phi() << ", mass = " << subjet1.m() << std::endl;
	std::cout << " subjet #2: Pt = " << subjet2.pt() << ", eta = " << subjet2.eta() << ", phi = " << subjet2.phi() << ", mass = " << subjet2.m() << std::endl;
      }

      const JetDefinition::Recombiner* rec = jet.associated_cluster_sequence()->jet_def().recombiner();
      PseudoJet result_local = join(subjet1, subjet2, *rec);
      if ( verbosity_ >= 1 ) {
	std::cout << "after recombination:" << std::endl;
	std::vector<fastjet::PseudoJet> subjets = result_local.pieces();
	int idx_subjet = 0;
	for ( std::vector<fastjet::PseudoJet>::const_iterator subjet = subjets.begin();
	      subjet != subjets.end(); ++subjet ) {
	  std::cout << " subjet #" << idx_subjet << ": Pt = " << subjet->pt() << ", eta = " << subjet->eta() << ", phi = " << subjet->phi() << ", mass = " << subjet->m() 
		    << " (#constituents = " << subjet->constituents().size() << ")" << std::endl;
	  std::vector<fastjet::PseudoJet> constituents = subjet->constituents();
	  int idx_constituent = 0;
	  for ( std::vector<fastjet::PseudoJet>::const_iterator constituent = constituents.begin();
		constituent != constituents.end(); ++constituent ) {
	    if ( constituent->pt() < 1.e-3 ) continue; // CV: skip ghosts
	    std::cout << "  constituent #" << idx_constituent << ": Pt = " << constituent->pt() << ", eta = " << constituent->eta() << ", phi = " << constituent->phi() << "," 
		      << " mass = " << constituent->m() << std::endl;
	    ++idx_constituent;
	  }
	  ++idx_subjet;
	}
      }

      CMSBoostedTauSeedingAlgorithmStructure* s = new CMSBoostedTauSeedingAlgorithmStructure(result_local);
      //s->_original_jet = jet;
      s->_mu = ( jet.m2() > 0. ) ? sqrt(std::max(subjet1.m2(), subjet2.m2())/jet.m2()) : 0.;
      s->_y  = ( jet.m2() > 0. ) ? subjet1.kt_distance(subjet2)/jet.m2() : 0.;
      s->_dR = subjet1.delta_R(subjet2);
      s->_pt = subjet2.pt();
            
      result_local.set_structure_shared_ptr(SharedPtr<PseudoJetStructureBase>(s));

      return result_local;
    } else {
      // no subjets for seeding boosted tau reconstruction found, return an empty PseudoJet
      if ( verbosity_  >= 1 ) {
	std::cout << "No subjets found." << std::endl;
      }
      return PseudoJet();
    }
  }


} // namespace contrib
 
FASTJET_END_NAMESPACE
