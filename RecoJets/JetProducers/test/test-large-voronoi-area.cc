//----------------------------------------------------------------------
/// \file
/// \page Example06 06 - using jet areas
///
/// fastjet example program for jet areas
/// It mostly illustrates the usage of the 
/// fastjet::AreaDefinition and fastjet::ClusterSequenceArea classes
///
/// run it with    : ./area_example < data.txt
///
/// Source code: area_example.cc
//----------------------------------------------------------------------

//STARTHEADER
//
//  2005-2011, Matteo Cacciari, Gavin Salam and Gregory Soyez
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
//  along with FastJet; if not, write to the Free Software
//  Foundation, Inc.:
//      59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//----------------------------------------------------------------------
//ENDHEADER

#include "fastjet/ClusterSequenceArea.hh"  // use this instead of the "usual" ClusterSequence to get area support
#include <iostream> // needed for io
#include <cstdio>   // needed for io

using namespace std;

/// an example program showing how to use fastjet
int main (int argc, char ** argv) {
  

  double inputEtaMax = 5.0;
  double rhoEtaMax = 6.0;
  int nev=0, iev=0;
  if ( argc > 1 ) {
    inputEtaMax = atof( argv[1] );
    cout << "inputEtaMax is now set to " << inputEtaMax << endl;
  }
  if ( argc > 2 ) {
    rhoEtaMax = atof( argv[2] );
    cout << "rhoEtaMax is now set to " << rhoEtaMax << endl;
  }
  if ( argc > 3 ) {
    nev = atoi(argv[3]);
    cout << "nev is now set to " << nev << endl;
  }
  
  while (iev < nev) {
  // read in input partdicles
  //----------------------------------------------------------
  double px, py , pz, E;
  string line;
  vector<fastjet::PseudoJet> input_particles;
  while (getline(cin, line)){
    if (line == "#END") break;
    if (line.substr(0,1) == "#") {continue;}
    istringstream istr(line);
    istr >> px >> py >> pz >> E;
    // create a fastjet::PseudoJet with these components and put it onto
    // back of the input_particles vector
    fastjet::PseudoJet j(px,py,pz,E);
    //std::cout << j.eta() << " " << j.phi() << " " << j.perp() << '\n';
    if ( fabs(j.rap()) < inputEtaMax )
      input_particles.push_back(fastjet::PseudoJet(px,py,pz,E)); 
  }
  if (input_particles.size() == 0) break;
  iev++;

  // create a jet definition: 
  // a jet algorithm with a given radius parameter
  //----------------------------------------------------------
  double R = 0.6;
  fastjet::JetDefinition jet_def(fastjet::kt_algorithm, R);


  // Now we also need an AreaDefinition to define the properties of the 
  // area we want
  //
  // This is made of 2 building blocks:
  //  - the area type:
  //    passive, active, active with explicit ghosts, or Voronoi area
  //  - the specifications:
  //    a VoronoiSpec or a GhostedAreaSpec for the 3 ghost-bases ones
  // 
  //---------------------------------------------------------- For
  // GhostedAreaSpec (as below), the minimal info you have to provide
  // is up to what rapidity ghosts are placed. 
  // Other commonm parameters (that mostly have an impact on the
  // precision on the area) include the number of repetitions
  // (i.e. the number of different sets of ghosts that are used) and
  // the ghost density (controlled through the ghost_area).
  // Other, more exotic, parameters (not shown here) control how ghosts
  // are placed.
  //
  // The ghost rapidity interval should be large enough to cover the
  // jets for which you want to calculate. E.g. if you want to
  // calculate the area of jets up to |y|=4, you need to put ghosts up
  // to at least 4+R (or, optionally, up to the largest particle
  // rapidity if this is smaller).
  double maxrap = rhoEtaMax;
  unsigned int n_repeat = 10; // default is 1
  double ghost_area = 0.01; // this is the default
  fastjet::GhostedAreaSpec area_spec(maxrap, n_repeat, ghost_area);
  fastjet::RangeDefinition range_def(maxrap);
  fastjet::AreaDefinition area_def(fastjet::active_area, area_spec);
  fastjet::VoronoiAreaSpec voronoiAreaSpec(0.9);

  // run the jet clustering with the above jet and area definitions
  //
  // The only change is the usage of a ClusterSequenceArea rather than
  //a ClusterSequence
  //----------------------------------------------------------
  //  fastjet::ClusterSequenceVoronoiArea clust_seq(input_particles, jet_def, voronoiAreaSpec);
  fastjet::ClusterSequenceVoronoiArea clust_seq(input_particles, jet_def,voronoiAreaSpec);
  //fastjet::ClusterSequenceArea clust_seq(input_particles, jet_def, area_def);


  // get the resulting jets ordered in pt
  //----------------------------------------------------------
  double ptmin = 0.0;
  vector<fastjet::PseudoJet> inclusive_jets = sorted_by_pt(clust_seq.inclusive_jets(ptmin));


//   double rho, sigma, mean_area;
//   clust_seq.get_median_rho_and_sigma( range_def, false, rho, sigma, mean_area );
// 
//   // tell the user what was done
//   //  - the description of the algorithm and area used
//   //  - extract the inclusive jets with pt > 5 GeV
//   //    show the output as 
//   //      {index, rap, phi, pt, number of constituents}
//   //----------------------------------------------------------
//   cout << endl;
//   cout << "Ran " << jet_def.description() << endl;
//   cout << "Area: " << voronoiAreaSpec.description() << endl;
//   cout << "Rho: " << rho << endl;


  double total_area_in_jets = 0., total_area_in_jets_in_range = 0;
  for (vector<fastjet::PseudoJet>::const_iterator it = inclusive_jets.begin();
       it != inclusive_jets.end(); ++it) {
    total_area_in_jets += clust_seq.area(*it);
    if (range_def.is_in_range(*it)) total_area_in_jets_in_range += clust_seq.area(*it);
  }

  cout << "Total area of range: " << range_def.area() << endl;
  cout << "Total area of jets: " << total_area_in_jets << endl;
  cout << "Total area of jets in range: " << total_area_in_jets_in_range << endl;
  cout << "Empty area in range: "
       << clust_seq.empty_area(range_def) << endl;
  cout << "Number of empty jets in range: "
       << clust_seq.n_empty_jets(range_def) << endl;
  
  double rho, sigma;
  clust_seq.get_median_rho_and_sigma(range_def, true, rho, sigma);
  cout << "rho, sigma = " << rho << " " << sigma << endl;


  
  cout << "Number of unclustered particles: " 
       << clust_seq.unclustered_particles().size() << endl;

  // label the columns
  printf("%5s %15s %15s %15s %15s %15s\n","jet #", "rapidity", "phi", "pt", "area", "area error");
 
  //   // print out the details for each jet
  for (unsigned int i = 0; i < inclusive_jets.size(); i++) {
    printf("%5u %15.8f %15.8f %15.8f %15.8f %15.8f\n", i,
	   inclusive_jets[i].rap(), inclusive_jets[i].phi(), inclusive_jets[i].perp(),
	   clust_seq.area(inclusive_jets[i]), clust_seq.area_error(inclusive_jets[i]) );
  }
  
  }
  return 0;
}
