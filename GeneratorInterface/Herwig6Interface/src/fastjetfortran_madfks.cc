//STARTHEADER
// $Id: fastjetfortran_madfks.cc,v 1.1 2013/02/08 20:15:52 spadhi Exp $
//
// Copyright (c) 2005-2011, Matteo Cacciari, Gavin P. Salam and Gregory Soyez
//
//----------------------------------------------------------------------
// This file is part of FastJet.
//
//  FastJet is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  The algorithms that underlie FastJet have required considerable
//  development and are described in hep-ph/0512210. If you use
//  FastJet as part of work towards a scientific publication, please
//  include a citation to the FastJet paper.
//
//  FastJet is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with FastJet. If not, see <http://www.gnu.org/licenses/>.
//----------------------------------------------------------------------
//ENDHEADER

#include <iostream>
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/Selector.hh"
#include "fastjet/SISConePlugin.hh"

using namespace std;
using namespace fastjet;

FASTJET_BEGIN_NAMESPACE      // defined in fastjet/internal/base.hh

/// a namespace for the fortran-wrapper which contains commonly-used
/// structures and means to transfer fortran <-> C++
namespace fwrapper {
  vector<PseudoJet> input_particles, jets;
  auto_ptr<JetDefinition::Plugin> plugin;
  JetDefinition jet_def;
  auto_ptr<ClusterSequence> cs;

  /// helper routine to transfer fortran input particles into 
  void transfer_input_particles(const double * p, const int & npart) {
    input_particles.resize(0);
    input_particles.reserve(npart);
    for (int i=0; i<npart; i++) {
      valarray<double> mom(4); // mom[0..3]
      for (int j=0;j<=3; j++) {
         mom[j] = *(p++);
      // RF-MZ: reorder the arguments because in madfks energy goes first; in fastjet last.
        // mom[(j+3) % 4] = *(p++);
      }
      PseudoJet psjet(mom);
      psjet.set_user_index(i);
      input_particles.push_back(psjet);    
    }
  }

  /// helper routine to help transfer jets -> f77jets[4*ijet+0..3]
  void transfer_jets(double * f77jets, int & njets) {
    njets = jets.size();
    for (int i=0; i<njets; i++) {
      for (int j=0;j<=3; j++) {
        *f77jets = jets[i][j];
      // RF-MZ: reorder the arguments because in madfks energy goes first; in fastjet last.
        //*f77jets = jets[i][(j+3) % 4];
        f77jets++;
      } 
    }
  }
  
  /// helper routine packaging the transfers, the clustering
  /// and the extraction of the jets
  void transfer_cluster_transfer(const double * p, const int & npart, 
                                 const JetDefinition & jet_def,
                                 const double & ptmin,
				 double * f77jets, int & njets, int * whichjet,
				 const double & ghost_maxrap = 0.0,  
				 const int & nrepeat = 0, const double & ghost_area = 0.0) {

    // transfer p[4*ipart+0..3] -> input_particles[i]
    transfer_input_particles(p, npart);

    // perform the clustering
    if ( ghost_maxrap == 0.0 ) {
         // cluster without areas
	 cs.reset(new ClusterSequence(input_particles,jet_def));
    } else {
         // cluster with areas
         GhostedAreaSpec area_spec(ghost_maxrap,nrepeat,ghost_area);
         AreaDefinition area_def(active_area, area_spec);
	 cs.reset(new ClusterSequenceArea(input_particles,jet_def,area_def));
    }
    // extract jets (pt-ordered)
    jets = sorted_by_pt(cs->inclusive_jets(ptmin));

    // transfer jets -> f77jets[4*ijet+0..3]
    transfer_jets(f77jets, njets);
 
    // Determine which parton/particle ended-up in which jet
    // set all jet entrie to zero first
    for(int ii=0; ii<npart; ++ii) whichjet[ii]=0;       

    // Loop over jets and find constituents
    for (int kk=0; kk<njets; ++kk) {   
      vector<PseudoJet> constit = cs->constituents(jets[kk]);
      for(unsigned int ll=0; ll<constit.size(); ++ll)
             whichjet[constit[ll].user_index()]=kk+1;
    }
    
  }

}
FASTJET_END_NAMESPACE

using namespace fastjet::fwrapper;


extern "C" {   

/// f77 interface to SISCone (via fastjet), as defined in arXiv:0704.0292
/// [see below for the interface to kt, Cam/Aachen & kt]
//
// Corresponds to the following Fortran subroutine
// interface structure:
//
//   SUBROUTINE FASTJETSISCONE(P,NPART,R,F,F77JETS,NJETS,WHICHJET)
//   DOUBLE PRECISION P(4,*), R, F, F77JETS(4,*)
//   INTEGER          NPART, NJETS, WHICHJET(*)
// 
// where on input
//
//   P        the input particle 4-momenta
//   NPART    the number of input momenta
//   R        the radius parameter
//   F        the overlap threshold
//
// and on output 
//
//   F77JETS  the output jet momenta (whose second dim should be >= NPART)
//            sorted in order of decreasing p_t.
//   NJETS    the number of output jets 
//   WHICHJET(i) the jet of parton/particle 'i'
//
// NOTE: if you are interfacing fastjet to Pythia 6, Pythia stores its
// momenta as a matrix of the form P(4000,5), whereas this fortran
// interface to fastjet expects them as P(4,NPART), i.e. you must take
// the transpose of the Pythia array and drop the fifth component
// (particle mass).
//
void fastjetsiscone_(const double * p, const int & npart,                   
                     const double & R, const double & f,                   
                     double * f77jets, int & njets, int * whichjet) {
    
    // prepare jet def
    plugin.reset(new SISConePlugin(R,f));
    jet_def = plugin.get();

    // do everything
//    transfer_cluster_transfer(p,npart,jet_def,f77jets,njets,whichjet);
}



/// f77 interface to SISCone (via fastjet), as defined in arXiv:0704.0292
/// [see below for the interface to kt, Cam/Aachen & kt]
/// Also calculates the active area of the jets, as defined in  
/// arXiv.org:0802.1188
//
// Corresponds to the following Fortran subroutine
// interface structure:
//
//   SUBROUTINE FASTJETSISCONEWITHAREA(P,NPART,R,F,GHMAXRAP,NREP,GHAREA,F77JETS,NJETS,WHICHJET)
//   DOUBLE PRECISION P(4,*), R, F, F77JETS(4,*), GHMAXRAP, GHAREA
//   INTEGER          NPART, NJETS, NREP, WHICHJET(*)
// 
// where on input
//
//   P        the input particle 4-momenta
//   NPART    the number of input momenta
//   R        the radius parameter
//   F        the overlap threshold
//   GHMAXRAP the maximum (abs) rapidity covered by ghosts (FastJet default 6.0) 
//   NREP     the number of repetitions used to evaluate the area (FastJet default 1) 
//   GHAREA   the area of a single ghost (FastJet default 0.01) 
//
// and on output 
//
//   F77JETS  the output jet momenta (whose second dim should be >= NPART)
//            sorted in order of decreasing p_t.
//   NJETS    the number of output jets 
//   WHICHJET(i) the jet of parton/particle 'i'
//
// NOTE: if you are interfacing fastjet to Pythia 6, Pythia stores its
// momenta as a matrix of the form P(4000,5), whereas this fortran
// interface to fastjet expects them as P(4,NPART), i.e. you must take
// the transpose of the Pythia array and drop the fifth component
// (particle mass).
//
void fastjetsisconewitharea_(const double * p, const int & npart,                   
                     const double & R, const double & f,                   
                     const double & ghost_rapmax, const int & nrepeat, const double & ghost_area,
                     double * f77jets, int & njets, int * whichjet) {
    
    // prepare jet def
    plugin.reset(new SISConePlugin(R,f));
    jet_def = plugin.get();

    // do everything
//    transfer_cluster_transfer(p,npart,jet_def,f77jets,njets,whichjet,ghost_rapmax,nrepeat,ghost_area);
}



/// f77 interface to the pp generalised-kt (sequential recombination)
/// algorithms, as defined in arXiv.org:0802.1189, which includes
/// kt, Cambridge/Aachen and anti-kt as special cases.
//
// Corresponds to the following Fortran subroutine
// interface structure:
//
//   SUBROUTINE FASTJETPPGENKT(P,NPART,R,PALG,F77JETS,NJETS,WHICHJET)
//   DOUBLE PRECISION P(4,*), R, PALG, F, F77JETS(4,*)
//   INTEGER          NPART, NJETS, WHICHJET(*)
// 
// where on input
//
//   P        the input particle 4-momenta
//   NPART    the number of input momenta
//   R        the radius parameter
//   PALG     the power for the generalised kt alg 
//            (1.0=kt, 0.0=C/A,  -1.0 = anti-kt)
//
// and on output 
//
//   F77JETS  the output jet momenta (whose second dim should be >= NPART)
//            sorted in order of decreasing p_t.
//   NJETS    the number of output jets 
//   WHICHJET(i) the jet of parton/particle 'i'
//
// For the values of PALG that correspond to "standard" cases (1.0=kt,
// 0.0=C/A, -1.0 = anti-kt) this routine actually calls the direct
// implementation of those algorithms, whereas for other values of
// PALG it calls the generalised kt implementation.
//
// NOTE: if you are interfacing fastjet to Pythia 6, Pythia stores its
// momenta as a matrix of the form P(4000,5), whereas this fortran
// interface to fastjet expects them as P(4,NPART), i.e. you must take
// the transpose of the Pythia array and drop the fifth component
// (particle mass).
//
void fastjetppgenkt_(const double * p, const int & npart,                   
                     const double & R, const double & ptjetmin,
                     const double & palg,
                     double * f77jets, int & njets, int * whichjet) {

    // prepare jet def
    if (palg == 1.0) {
      jet_def = JetDefinition(kt_algorithm, R);
    }  else if (palg == 0.0) {
      jet_def = JetDefinition(cambridge_algorithm, R);
    }  else if (palg == -1.0) {
      jet_def = JetDefinition(antikt_algorithm, R);
    } else {
      jet_def = JetDefinition(genkt_algorithm, R, palg);
    }

    // do everything
    transfer_cluster_transfer(p,npart,jet_def,ptjetmin,f77jets,njets,whichjet);
}


/// f77 interface to the pp generalised-kt (sequential recombination)
/// algorithms, as defined in arXiv.org:0802.1189, which includes
/// kt, Cambridge/Aachen and anti-kt as special cases.
/// Also calculates the active area of the jets, as defined in  
/// arXiv.org:0802.1188
//
// Corresponds to the following Fortran subroutine
// interface structure:
//
//   SUBROUTINE FASTJETPPGENKTWITHAREA(P,NPART,R,PALG,GHMAXRAP,NREP,GHAREA,F77JETS,NJETS,WHICHJET)
//   DOUBLE PRECISION P(4,*), R, PALG, GHMAXRAP, GHAREA,  F77JETS(4,*)
//   INTEGER          NPART, NREP, NJETS, WHICHJET(*)
// 
// where on input
//
//   P        the input particle 4-momenta
//   NPART    the number of input momenta
//   R        the radius parameter
//   PALG     the power for the generalised kt alg 
//            (1.0=kt, 0.0=C/A,  -1.0 = anti-kt)
//   GHMAXRAP the maximum (abs) rapidity covered by ghosts (FastJet default 6.0) 
//   NREP     the number of repetitions used to evaluate the area (FastJet default 1) 
//   GHAREA   the area of a single ghost (FastJet default 0.01) 
//
// and on output 
//
//   F77JETS  the output jet momenta (whose second dim should be >= NPART)
//            sorted in order of decreasing p_t.
//   NJETS    the number of output jets 
//   WHICHJET(i) the jet of parton/particle 'i'
//
// For the values of PALG that correspond to "standard" cases (1.0=kt,
// 0.0=C/A, -1.0 = anti-kt) this routine actually calls the direct
// implementation of those algorithms, whereas for other values of
// PALG it calls the generalised kt implementation.
//
// NOTE: if you are interfacing fastjet to Pythia 6, Pythia stores its
// momenta as a matrix of the form P(4000,5), whereas this fortran
// interface to fastjet expects them as P(4,NPART), i.e. you must take
// the transpose of the Pythia array and drop the fifth component
// (particle mass).
//
void fastjetppgenktwitharea_(const double * p, const int & npart,                   
                             const double & R, const double & palg,
                             const double & ghost_rapmax, const int & nrepeat, const double & ghost_area,
                             double * f77jets, int & njets, int * whichjet) {
    
    // prepare jet def
    if (palg == 1.0) {
      jet_def = JetDefinition(kt_algorithm, R);
    }  else if (palg == 0.0) {
      jet_def = JetDefinition(cambridge_algorithm, R);
    }  else if (palg == -1.0) {
      jet_def = JetDefinition(antikt_algorithm, R);
    } else {
      jet_def = JetDefinition(genkt_algorithm, R, palg);
    }
        
    // do everything
//    transfer_cluster_transfer(p,npart,jet_def,f77jets,njets,whichjet,ghost_rapmax,nrepeat,ghost_area);
}


/// f77 interface to provide access to the constituents of a jet found
/// in the jet clustering with one of the above routines.
///
/// Given the index ijet of a jet (in the range 1...njets) obtained in
/// the last call to jet clustering, fill the array
/// constituent_indices, with nconstituents entries, with the indices
/// of the constituents that belong to that jet (which will be in the
/// range 1...npart)
//
// Corresponds to the following Fortran subroutine
// interface structure:
//
//   SUBROUTINE FASTJETCONSTITUENTS(IJET,CONSTITUENT_INDICES,NCONSTITUENTS)
//   INTEGER    IJET
//   INTEGER    CONSTITUENT_INDICES(*)
//   INTEGER    nconstituents
//
void fastjetconstituents_(const int & ijet, 
   	                  int * constituent_indices, int & nconstituents) {
  assert(cs.get() != 0);
  assert(ijet > 0 && ijet <= int(jets.size()));

  vector<PseudoJet> constituents = cs->constituents(jets[ijet-1]);

  nconstituents = constituents.size();
  for (int i = 0; i < nconstituents; i++) {
    constituent_indices[i] = constituents[i].cluster_hist_index()+1;
  }
}


/// f77 interface to provide access to the area of a jet found
/// in the jet clustering with one of the above "...witharea" routines.
///
/// Given the index ijet of a jet (in the range 1...njets) obtained in
/// the last call to jet clustering, return its area. If the jets have
/// not been obtained with a "...witharea" soutine it returns 0.
//
// Corresponds to the following Fortran subroutine
// interface structure:
//
//   FUNCTION FASTJETAREA(IJET)
//   DOUBLE PRECISION FASTJETAREA
//   INTEGER    IJET
//
double fastjetarea_(const int & ijet) {
  assert(ijet > 0 && ijet <= int(jets.size()));
  const ClusterSequenceAreaBase * csab =
                    dynamic_cast<const ClusterSequenceAreaBase *>(cs.get());
  if (csab != 0) {
    // we have areas and can use csab to access all the area-related info
    return csab->area(jets[ijet-1]);
  } else {
    return 0.;
//  Error("No area information associated to this jet."); 
  }
}


/// return the dmin corresponding to the recombination that went from
/// n+1 to n jets (sometimes known as d_{n n+1}).
//
// Corresponds to the following Fortran interface
// 
//   FUNCTION FASTJETDMERGE(N)
//   DOUBLE PRECISION FASTJETDMERGE
//   INTEGER N
//   
double fastjetdmerge_(const int & n) {
  assert(cs.get() != 0);
  return cs->exclusive_dmerge(n);
}


/// return the maximum of the dmin encountered during all recombinations 
/// up to the one that led to an n-jet final state; identical to
/// exclusive_dmerge, except in cases where the dmin do not increase
/// monotonically.
//
// Corresponds to the following Fortran interface
// 
//   FUNCTION FASTJETDMERGEMAX(N)
//   DOUBLE PRECISION FASTJETDMERGEMAX
//   INTEGER N
//   
double fastjetdmergemax_(const int & n) {
  assert(cs.get() != 0);
  return cs->exclusive_dmerge_max(n);
}


/// return the background transverse momentum density per unit scalar
/// area rho, its fluctuation sigma, and the mean area of the jets used for the
/// background estimation in a given event,
/// as evaluated in the range [rapmin,rapmax] in rapidity and [phimin,phimax] in azimuth
//
// Corresponds to the following Fortran interface
// 
//   SUBROUTINE FASTJETGLOBALRHOANDSIGMA(RAPMIN,RAPMAX,PHIMIN,PHIMAX,RHO,SIGMA,MEANAREA)
//   DOUBLE PRECISION RAPMIN,RAPMAX,PHIMIN,PHIMAX
//   DOUBLE PRECISION RHO,SIGMA,MEANAREA
//   
void fastjetglobalrhoandsigma_(const double & rapmin, const double & rapmax,
                               const double & phimin, const double & phimax,
			       double & rho, double & sigma, double & meanarea) {
  const ClusterSequenceAreaBase * csab =
                    dynamic_cast<const ClusterSequenceAreaBase *>(cs.get());
  if (csab != 0) {
      // we have areas and can use csab to access all the area-related info
    Selector range =  SelectorRapRange(rapmin,rapmax) && SelectorPhiRange(phimin,phimax);
      bool use_area_4vector = false;
      csab->get_median_rho_and_sigma(range,use_area_4vector,rho,sigma,meanarea);
  } else {
      Error("Clustering with area is necessary in order to be able to evaluate rho."); 
  }
}




}
