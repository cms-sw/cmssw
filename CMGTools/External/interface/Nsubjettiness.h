//  Nsubjettiness Plugin
//  Jesse Thaler and Ken Van Tilburg
//  Version 0.3.2 (November 4, 2011)
//  Questions/Comments?  jthaler@jthaler.net

#ifndef __NJETTINESS_HH__
#define __NJETTINESS_HH__


#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include <cmath>
#include <vector>
#include <list>

///////
//
// Helper classes and enums
//
///////

// Choose Axes Mode
enum NsubAxesMode {
   nsub_kt_axes,  // exclusive kt axes
   nsub_ca_axes,  // exclusive ca axes
   nsub_antikt_0p2_axes,  // inclusive hardest axes with antikt-0.2
   nsub_min_axes, // axes that minimize N-subjettiness (100 passes by default)
   nsub_manual_axes, // set your own axes with setAxes()
   nsub_1pass_from_kt_axes, // one-pass minimization from kt starting point
   nsub_1pass_from_ca_axes, // one-pass minimization from ca starting point
   nsub_1pass_from_antikt_0p2_axes,  // one-pass minimization from antikt-0.2 starting point 
   nsub_1pass_from_manual_axes  // one-pass minimization from manual starting point
};

// Parameters that define Nsubjettiness
class NsubParameters {
private:
   double _beta;  // angular weighting exponent
   double _R0;    // characteristic jet radius (for normalization)
   double _Rcutoff;  // Cutoff scale for cone jet finding (default is large number such that no boundaries are used)
   
public:
   NsubParameters(double myBeta, double myR0, double myRcutoff=10000.0) :
   _beta(myBeta), _R0(myR0), _Rcutoff(myRcutoff) {}
   double beta() const {return _beta;}
   double R0() const {return _R0;}
   double Rcutoff() const {return _Rcutoff;}
};

// Parameters that change minimization procedure.
// Set automatically when you choose NsubAxesMode, but can be adjusted manually as well
class KmeansParameters {
private:
   int _n_iterations;  // Number of iterations to run  (0 for no minimization, 1 for one-pass, >>1 for global minimum)
   double _precision;  // Desired precision in axes alignment
   int _halt;          // maximum number of steps per iteration
   double _noise_range;// noise range for random initialization
   
public:
   KmeansParameters() : _n_iterations(0), _precision(0.0), _halt(0.0), _noise_range(0.0) {}
   KmeansParameters(int my_n_iterations, double my_precision, int my_halt, double my_noise_range) :
   _n_iterations(my_n_iterations),  _precision(my_precision), _halt(my_halt), _noise_range(my_noise_range) {}
   int n_iterations() const { return _n_iterations;}
   double precision() const {return _precision;}
   int halt() const {return _halt;}
   double noise_range() const {return _noise_range;}
};

// helper class for minimization
class LightLikeAxis {
private:
   double _rap, _phi, _weight, _mom;
   
public:
   LightLikeAxis(double my_rap, double my_phi, double my_weight, double my_mom) :
   _rap(my_rap), _phi(my_phi), _weight(my_weight), _mom(my_mom) {}
   double rap() {return _rap;}
   double phi() {return _phi;}
   double weight() {return _weight;}
   double mom() {return _mom;}
   void set_rap(double newRap) {_rap = newRap;}
   void set_phi(double newPhi) {_phi = newPhi;}
   void set_weight(double newWeight) {_weight = newWeight;}
   void set_mom(double newMom) {_mom = newMom;}
};

///////
//
// Functions for minimization.
// TODO:  Wrap these in N-subjettiness class
//
///////

// Calculates distance between two points in rapidity-azimuth plane
// TODO:  Convert to built in fastjet DeltaR distance
double Distance(double rap1, double phi1, double rap2, double phi2){ 
   double distRap = fabs(rap1-rap2); double distPhi = fabs(phi1-phi2);
   if (distPhi > M_PI) {distPhi = 2.0*M_PI - distPhi;}
   double distance = sqrt(distRap*distRap + distPhi*distPhi);
   return distance;
}

// Given starting axes, update to find better axes
std::vector<LightLikeAxis> UpdateAxes(std::vector <LightLikeAxis> old_axes, 
                                  const std::vector <fastjet::PseudoJet> & inputJets, NsubParameters paraNsub) {
   int n_jets = old_axes.size();
   
   double beta = paraNsub.beta();
   double Rcutoff = paraNsub.Rcutoff();
   
   /////////////// Assignment Step //////////////////////////////////////////////////////////
   std::vector<double> assignment_index(inputJets.size()); 
   int k_assign = -1;
   
   for (unsigned int i = 0; i < inputJets.size(); i++){
      double smallestDist = 1000.0;
      for (int k = 0; k < n_jets; k++) {
         double thisDist = Distance(inputJets[i].rap(),inputJets[i].phi(),old_axes[k].rap(),old_axes[k].phi());
         if (thisDist < smallestDist) {
            smallestDist = thisDist;
            k_assign = k;
         }
      }
      if (smallestDist > Rcutoff) {k_assign = -1;}
      assignment_index[i] = k_assign;
   }
   
   //////////////// Update Step /////////////////////////////////////////////////////////////
   std::vector <LightLikeAxis>  new_axes(n_jets,LightLikeAxis(0,0,0,0));
   std::vector< std::vector<double> > fourVecJets(4, std::vector<double>(n_jets,0));
   double distPhi, old_dist;
   for (unsigned int i = 0; i < inputJets.size(); i++){
      if (assignment_index[i] == -1) {continue;}
      old_dist = Distance(inputJets[i].rap(),inputJets[i].phi(),old_axes[assignment_index[i]].rap(), old_axes[assignment_index[i]].phi());
      old_dist = pow(old_dist, (beta-2));
      // rapidity sum
      new_axes[assignment_index[i]].set_rap(new_axes[assignment_index[i]].rap() + inputJets[i].perp() * inputJets[i].rap() * old_dist);
      // phi sum
      distPhi = inputJets[i].phi() - old_axes[assignment_index[i]].phi();
      if (fabs(distPhi) <= M_PI){
         new_axes[assignment_index[i]].set_phi( new_axes[assignment_index[i]].phi() + inputJets[i].perp() * inputJets[i].phi() * old_dist );
      } else if (distPhi > M_PI) {
         new_axes[assignment_index[i]].set_phi( new_axes[assignment_index[i]].phi() + inputJets[i].perp() * (-2*M_PI + inputJets[i].phi()) * old_dist );
      } else if (distPhi < -M_PI) {
         new_axes[assignment_index[i]].set_phi( new_axes[assignment_index[i]].phi() + inputJets[i].perp() * (+2*M_PI + inputJets[i].phi()) * old_dist );
      }
      // weights sum
      new_axes[assignment_index[i]].set_weight( new_axes[assignment_index[i]].weight() + inputJets[i].perp() * old_dist );
      // momentum magnitude sum
      fourVecJets[0][assignment_index[i]] += inputJets[i].px();
      fourVecJets[1][assignment_index[i]] += inputJets[i].py();
      fourVecJets[2][assignment_index[i]] += inputJets[i].pz();
      fourVecJets[3][assignment_index[i]] += inputJets[i].e();
   }
   // normalize sums
   for (int k = 0; k < n_jets; k++){
      if (new_axes[k].weight() == 0) {continue;}
      new_axes[k].set_rap( new_axes[k].rap() / new_axes[k].weight() );
      new_axes[k].set_phi( new_axes[k].phi() / new_axes[k].weight() );
      new_axes[k].set_phi( fmod(new_axes[k].phi() + 2*M_PI, 2*M_PI) );
      new_axes[k].set_mom( sqrt( pow(fourVecJets[0][k],2) + pow(fourVecJets[1][k],2) + pow(fourVecJets[2][k],2) ) );
   }
   return new_axes;
}

// Go from internal LightLikeAxis to PseudoJet
// TODO:  Make part of LightLikeAxis class.
std::vector<fastjet::PseudoJet> ConvertToPseudoJet(std::vector <LightLikeAxis> axes) {
   
   int n_jets = axes.size();
   
   double px, py, pz, E;
   std::vector<fastjet::PseudoJet> FourVecJets;
   for (int k = 0; k < n_jets; k++) {
      E = axes[k].mom();
      pz = (exp(2.0*axes[k].rap()) - 1.0) / (exp(2.0*axes[k].rap()) + 1.0) * E;
      px = cos(axes[k].phi()) * sqrt( pow(E,2) - pow(pz,2) );
      py = sin(axes[k].phi()) * sqrt( pow(E,2) - pow(pz,2) );
      fastjet::PseudoJet temp = fastjet::PseudoJet(px,py,pz,E);
      FourVecJets.push_back(temp);
   }
   return FourVecJets;
}

// N-subjettiness pieces
std::vector<double> ConstituentTauValue(const std::vector <fastjet::PseudoJet> & particles, const std::vector<fastjet::PseudoJet>& axes,const NsubParameters paraNsub) {// Returns the sub-tau values, i.e. a std::vector of the contributions to tau_N of each Voronoi region (or region within R_0)
   double beta = paraNsub.beta();
   double R0 = paraNsub.R0();
   double Rcutoff = paraNsub.Rcutoff();
   
   std::vector<double> tauNum(axes.size(),0.0),tau(axes.size());
   double tauDen = 0.0;
   for (unsigned int i = 0; i < particles.size(); i++) {
      // find minimum distance
      int j_min = -1;
      double minR = 10000.0; // large number
      for (unsigned int j = 0; j < axes.size(); j++) {
         double tempR = sqrt(particles[i].squared_distance(axes[j])); // delta R distance
         if (tempR < minR) {minR = tempR; j_min = j;}
      }
      if (minR > Rcutoff) {minR = Rcutoff;}
      tauNum[j_min] += particles[i].perp() * pow(minR,beta);
      tauDen += particles[i].perp() * pow(R0,beta);
   }
   for (unsigned int j = 0; j < axes.size(); j++) {
      tau[j] = tauNum[j]/tauDen;
   }
   return tau;
}

// N-subjettiness values
double TauValue(const std::vector <fastjet::PseudoJet>& particles, const std::vector<fastjet::PseudoJet>& axes,const NsubParameters paraNsub) {// Calculates tau_N
   std::vector<double> tau_vec = ConstituentTauValue(particles, axes,paraNsub);
   double tau = 0.0;
   for (unsigned int j = 0; j < tau_vec.size(); j++) {tau += tau_vec[j];}
   return tau;
}

// Get exclusive kT subjets
std::vector<fastjet::PseudoJet> GetKTAxes(const int n_jets, const std::vector <fastjet::PseudoJet> & inputJets) {
   fastjet::JetDefinition jet_def = fastjet::JetDefinition(fastjet::kt_algorithm,M_PI/2.0,fastjet::E_scheme,fastjet::Best);
   fastjet::ClusterSequence jet_clust_seq(inputJets, jet_def);
   return jet_clust_seq.exclusive_jets(n_jets);
}

// Get exclusive CA subjets
std::vector<fastjet::PseudoJet> GetCAAxes(const int n_jets, const std::vector <fastjet::PseudoJet> & inputJets) {
   fastjet::JetDefinition jet_def = fastjet::JetDefinition(fastjet::cambridge_algorithm,M_PI/2.0,fastjet::E_scheme,fastjet::Best);
   fastjet::ClusterSequence jet_clust_seq(inputJets, jet_def);
   return jet_clust_seq.exclusive_jets(n_jets);
}

// Get inclusive anti kT hardest subjets
std::vector<fastjet::PseudoJet> GetAntiKTAxes(const int n_jets, const double R0, const std::vector <fastjet::PseudoJet> & inputJets) {
   fastjet::JetDefinition jet_def = fastjet::JetDefinition(fastjet::antikt_algorithm,R0,fastjet::E_scheme,fastjet::Best);
   fastjet::ClusterSequence jet_clust_seq(inputJets, jet_def);
   std::vector<fastjet::PseudoJet> myJets = sorted_by_pt(jet_clust_seq.inclusive_jets());
   myJets.resize(n_jets);  // only keep n hardest
   return myJets;
}


// Get minimization axes
std::vector<fastjet::PseudoJet> GetMinimumAxes(std::vector <fastjet::PseudoJet> seedAxes, const std::vector <fastjet::PseudoJet> & inputJets, KmeansParameters para, 
                                          NsubParameters paraNsub) {
   int n_jets = seedAxes.size();
   double noise = 0, tau = 10000.0, tau_tmp, cmp;
   std::vector< LightLikeAxis > new_axes(n_jets, LightLikeAxis(0,0,0,0)), old_axes(n_jets, LightLikeAxis(0,0,0,0));
   std::vector<fastjet::PseudoJet> tmp_min_axes, min_axes;
   for (int l = 0; l < para.n_iterations(); l++) { // Do minimization procedure multiple times
      // Add noise to guess for the axes
      for (int k = 0; k < n_jets; k++) {
         if ( 0 == l ) { // Don't add noise on first pass
            old_axes[k].set_rap( seedAxes[k].rap() + noise );
            old_axes[k].set_phi( seedAxes[k].phi() + noise );
         } else {
            noise = ((double)rand()/(double)RAND_MAX) * para.noise_range() * 2 - para.noise_range();
            old_axes[k].set_rap( seedAxes[k].rap() + noise );
            noise = ((double)rand()/(double)RAND_MAX) * para.noise_range() * 2 - para.noise_range();
            old_axes[k].set_phi( seedAxes[k].phi() + noise );
         }
      }
      cmp = 100.0; int h = 0;
      while (cmp > para.precision() && h < para.halt()) { // Keep updating axes until near-convergence or too many update steps
         cmp = 0.0; h++;
         new_axes = UpdateAxes(old_axes, inputJets, paraNsub); // Update axes
         for (int k = 0; k < n_jets; k++) {
            cmp += Distance(new_axes[k].rap(),new_axes[k].phi(),old_axes[k].rap(),old_axes[k].phi());
         }
         cmp = cmp / ((double) n_jets);
         old_axes = new_axes;
      }
      tmp_min_axes = ConvertToPseudoJet(old_axes); // Convert axes directions into four-std::vectors
      tau_tmp = TauValue(inputJets, tmp_min_axes,paraNsub); 
      if (tau_tmp < tau) {tau = tau_tmp; min_axes = tmp_min_axes;} // Keep axes and tau only if they are best so far
   }	
   return min_axes;
}

///////
//
// Main Nsubjettiness Class
//
///////

class Nsubjettiness {
   
private:
   NsubAxesMode _axes;    //Axes mode choice
   NsubParameters _paraNsub;  //Parameters for Nsubjettiness
   KmeansParameters _paraKmeans;  //Parameters for Minimization Procedure (set by NsubAxesMode automatically, but can change manually if desired)

   std::vector<fastjet::PseudoJet> _currentAxes;
   
   void establishAxes(const int n_jets, const std::vector <fastjet::PseudoJet> & inputs);
   
public:
   Nsubjettiness(NsubAxesMode axes, NsubParameters paraNsub);
   
   void setParaKmeans(KmeansParameters newPara) {_paraKmeans = newPara;}
   void setParaNsub(NsubParameters newPara) {_paraNsub = newPara;}
   
   // setAxes for Manual mode
   void setAxes(std::vector<fastjet::PseudoJet> myAxes) {
      assert((_axes == nsub_manual_axes) || (_axes == nsub_1pass_from_manual_axes));
      _currentAxes = myAxes;
   }
   
   // The value of N-subjettiness
   double getTau(const int n_jets, const std::vector<fastjet::PseudoJet> & inputJets) {
      if (n_jets >= (int) inputJets.size()) {
         _currentAxes = inputJets;
         _currentAxes.resize(n_jets,fastjet::PseudoJet(0.0,0.0,0.0,0.0));
         return 0.0;
      }
      establishAxes(n_jets, inputJets);  // sets current Axes
      return TauValue(inputJets,_currentAxes,_paraNsub);
   }
   
   // get axes used by getTau.
   std::vector<fastjet::PseudoJet> currentAxes() {
      return _currentAxes;
   }
   
   // partition inputs by Voronoi (each vector stores indices corresponding to inputJets)
   std::vector<std::list<int> > getPartition(const std::vector<fastjet::PseudoJet> & inputJets);

   // partition inputs by Voronoi
   std::vector<fastjet::PseudoJet> getJets(const std::vector<fastjet::PseudoJet> & inputJets);

};


//Use NsubAxesMode to pick which type of axes to use
void Nsubjettiness::establishAxes(const int n_jets, const std::vector <fastjet::PseudoJet> & inputs) {
   switch (_axes) {
      case nsub_kt_axes:
         _currentAxes = GetKTAxes(n_jets,inputs);
         break;
      case nsub_ca_axes:
         _currentAxes = GetCAAxes(n_jets,inputs);
         break;
      case nsub_antikt_0p2_axes:
         _currentAxes = GetAntiKTAxes(n_jets,0.2,inputs);
         break;
      case nsub_1pass_from_kt_axes:
      case nsub_min_axes:
         _currentAxes = GetKTAxes(n_jets,inputs);
         _currentAxes = GetMinimumAxes(_currentAxes, inputs, _paraKmeans, _paraNsub);
         break;
      case nsub_1pass_from_ca_axes:
         _currentAxes = GetCAAxes(n_jets,inputs);
         _currentAxes = GetMinimumAxes(_currentAxes, inputs, _paraKmeans, _paraNsub);
         break;
      case nsub_1pass_from_antikt_0p2_axes:
         _currentAxes = GetAntiKTAxes(n_jets,0.2,inputs);
         _currentAxes = GetMinimumAxes(_currentAxes, inputs, _paraKmeans, _paraNsub);
         break;
      case nsub_1pass_from_manual_axes:
         assert((int) _currentAxes.size() == n_jets);
         _currentAxes = GetMinimumAxes(_currentAxes, inputs, _paraKmeans, _paraNsub);
         break;
      case nsub_manual_axes:
         assert((int) _currentAxes.size() == n_jets);
         break;
      default:
         assert(false);
         break;
   }      
}

//Constructor sets KmeansParameters from NsubAxesMode input
Nsubjettiness::Nsubjettiness(NsubAxesMode axes, NsubParameters paraNsub) : _axes(axes), _paraNsub(paraNsub), _paraKmeans() {
   switch (_axes) {
      case nsub_kt_axes:
      case nsub_ca_axes:
      case nsub_antikt_0p2_axes:
         _paraKmeans = KmeansParameters(0,0.0,0.0,0.0);
         break;
      case nsub_1pass_from_kt_axes:
      case nsub_1pass_from_ca_axes:
      case nsub_1pass_from_antikt_0p2_axes:
      case nsub_1pass_from_manual_axes:
         _paraKmeans = KmeansParameters(1,0.0001,1000.0,0.8);
         break;
      case nsub_min_axes:
         _paraKmeans = KmeansParameters(100,0.0001,1000.0,0.8);
         break;
      case nsub_manual_axes:
         _paraKmeans = KmeansParameters(0,0.0,0.0,0.0);
         break;
      default:
         assert(false);
         break;
   }
}


std::vector<std::list<int> > Nsubjettiness::getPartition(const std::vector<fastjet::PseudoJet> & particles) {
   double Rcutoff = _paraNsub.Rcutoff();
   
   std::vector<std::list<int> > partitions(_currentAxes.size());

   for (unsigned int i = 0; i < particles.size(); i++) {
      // find minimum distance
      int j_min = -1;
      double minR = 10000.0; // large number
      for (unsigned int j = 0; j < _currentAxes.size(); j++) {
         double tempR = sqrt(particles[i].squared_distance(_currentAxes[j])); // delta R distance
         if (tempR < minR) {
            minR = tempR;
            j_min = j;
         }
      }
      if (minR > Rcutoff) {
         // do nothing
      } else {
         partitions[j_min].push_back(i);
      }
   }
   return partitions;
}

// TODO:  Make this call getPartition
std::vector<fastjet::PseudoJet> Nsubjettiness::getJets(const std::vector<fastjet::PseudoJet> & particles) {
   double Rcutoff = _paraNsub.Rcutoff();
   
   std::vector<fastjet::PseudoJet> jets(_currentAxes.size());

   for (unsigned int i = 0; i < particles.size(); i++) {
      // find minimum distance
      int j_min = -1;
      double minR = 10000.0; // large number
      for (unsigned int j = 0; j < _currentAxes.size(); j++) {
         double tempR = sqrt(particles[i].squared_distance(_currentAxes[j])); // delta R distance
         if (tempR < minR) {
            minR = tempR;
            j_min = j;
         }
      }
      if (minR > Rcutoff) {
         // do nothing
      } else {
         jets[j_min] += particles[i];
      }
   }
   return jets;
}

#endif

