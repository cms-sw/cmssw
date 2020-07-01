
#include "RecoEcal/EgammaClusterAlgos/interface/EndcapPiZeroDiscriminatorAlgo.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <fstream>
#include <iostream>

using namespace std;

namespace {
  constexpr int Nfiles_EE = 5;
}

EndcapPiZeroDiscriminatorAlgo::EndcapPiZeroDiscriminatorAlgo(double stripEnergyCut, int nStripCut, const string& path)
    : preshStripEnergyCut_(stripEnergyCut), preshSeededNstr_(nStripCut) {
  // Read all Weight files
  constexpr std::array<char const*, Nfiles_EE> file_pt{{"20", "30", "40", "50", "60"}};
  constexpr std::array<char const*, 5> file_barrel_pt{{"20", "30", "40", "50", "60"}};

  for (auto ptName : file_pt) {
    string nn_paterns_file = "endcapPiZeroDiscriminatorWeights_et";
    nn_paterns_file += ptName;
    nn_paterns_file += ".wts";
    edm::FileInPath WFile(path + nn_paterns_file);
    readWeightFile(WFile.fullPath().c_str(), EE_Layers, EE_Indim, EE_Hidden, EE_Outdim);  // read the weights' file
  }

  for (auto ptName : file_barrel_pt) {
    string nn_paterns_file = "barrelPiZeroDiscriminatorWeights_et";
    nn_paterns_file += ptName;
    nn_paterns_file += ".wts";
    edm::FileInPath WFile(path + nn_paterns_file);
    readWeightFile(WFile.fullPath().c_str(), EB_Layers, EB_Indim, EB_Hidden, EB_Outdim);  // read the weights' file
  }
}

vector<float> EndcapPiZeroDiscriminatorAlgo::findPreshVector(ESDetId strip,
                                                             RecHitsMap* rechits_map,
                                                             CaloSubdetectorTopology* topology_p) {
  vector<float> vout_stripE;

  // skip if rechits_map contains no hits
  if (rechits_map->empty()) {
    edm::LogWarning("EndcapPiZeroDiscriminatorAlgo") << "RecHitsMap has size 0.";
    return vout_stripE;
  }

  vout_stripE.clear();

  vector<ESDetId> road_2d;
  road_2d.clear();

  int plane = strip.plane();

  LogTrace("EcalClusters")
      << "EndcapPiZeroDiscriminatorAlgo: findPreshVectors: Preshower Seeded Algorithm - looking for clusters"
      << "n"
      << "findPreshVectors: Preshower is intersected at strip " << strip.strip() << ", at plane " << plane;

  if (strip == ESDetId(0)) {  //works in case of no intersected strip found
    for (int i = 0; i < 11; i++) {
      vout_stripE.push_back(-100.);
    }
  }

  // Add to the road the central strip
  road_2d.push_back(strip);

  //Make a navigator, and set it to the strip cell.
  EcalPreshowerNavigator navigator(strip, topology_p);
  navigator.setHome(strip);
  //search for neighbours in the central road
  findPi0Road(strip, navigator, plane, road_2d);

  LogTrace("EcalClusters")
      << "EndcapPiZeroDiscriminatorAlgo:findPreshVectors: Total number of strips in the central road: "
      << road_2d.size();

  // Find the energy of each strip
  RecHitsMap::iterator final_strip = rechits_map->end();
  // very dangerous, added a protection on the rechits_map->size()
  // at the beginning of the method
  final_strip--;
  ESDetId last_stripID = final_strip->first;

  vector<ESDetId>::iterator itID;
  for (itID = road_2d.begin(); itID != road_2d.end(); itID++) {
    LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: findPreshVectors: ID = " << *itID;

    float E = 0.;
    RecHitsMap::iterator strip_it = rechits_map->find(*itID);
    if (goodPi0Strip(strip_it, last_stripID)) {  // continue if strip not found in rechit_map
      E = strip_it->second.energy();
    }
    vout_stripE.push_back(E);
    LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: findPreshVectors: E = " << E;
  }

  // ***ML beg***
  // vector of size=11, content of vout_stripE is copied into vout_ElevenStrips_Energy
  // to avoid problem in case number of strips is less than 11
  vector<float> vout_ElevenStrips_Energy;
  vout_ElevenStrips_Energy.reserve(11);

for (int i = 0; i < 11; i++) {
    vout_ElevenStrips_Energy.push_back(0.);
  }

  for (unsigned int i = 0; i < vout_stripE.size(); i++) {
    vout_ElevenStrips_Energy[i] = vout_stripE.at(i);
  }

  //return vout_stripE;
  return vout_ElevenStrips_Energy;
  // ***ML end***
}

// returns true if the candidate strip fulfills the requirements to be added to the cluster:
bool EndcapPiZeroDiscriminatorAlgo::goodPi0Strip(RecHitsMap::iterator candidate_it, ESDetId lastID) {
  RecHitsMap::iterator candidate_tmp = candidate_it;
  candidate_tmp--;

  // crystal should not be included...
  if (candidate_tmp->first == lastID)  // ...if it corresponds to a hit
  {
    LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: goodPi0Strip No such a strip in rechits_map ";
    return false;
  } else if (candidate_it->second.energy() <= preshStripEnergyCut_)  // ...if it has a negative or zero energy
  {
    LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: goodPi0Strip Strip energy "
                             << candidate_it->second.energy() << " is below threshold ";
    return false;
  }

  return true;
}

// find strips in the road of size +/- preshSeededNstr_ from the central strip
void EndcapPiZeroDiscriminatorAlgo::findPi0Road(ESDetId strip,
                                                EcalPreshowerNavigator& theESNav,
                                                int plane,
                                                vector<ESDetId>& vout) {
  if (strip == ESDetId(0))
    return;
  ESDetId next;
  theESNav.setHome(strip);
  LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: findPi0Road: starts from strip " << strip;

  if (plane == 1) {
    // east road
    int n_east = 0;
    LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: findPi0Road: Go to the East ";

    while (((next = theESNav.east()) != ESDetId(0) && next != strip)) {
      LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: findPi0Road: East: " << n_east << " current strip is "
                               << next;

      vout.push_back(next);
      ++n_east;
      if (n_east == preshSeededNstr_)
        break;
    }
    // west road
    int n_west = 0;
    LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: findPi0Road: Go to the West ";

    theESNav.home();
    while (((next = theESNav.west()) != ESDetId(0) && next != strip)) {
      LogTrace("EcalClusters") << "findPi0Road: West: " << n_west << " current strip is " << next;

      vout.push_back(next);
      ++n_west;
      if (n_west == preshSeededNstr_)
        break;
    }
    LogTrace("EcalClusters")
        << "EndcapPiZeroDiscriminatorAlgo: findPi0Road: Total number of strips found in the road at 1-st plane is "
        << n_east + n_west;

  } else if (plane == 2) {
    // north road
    int n_north = 0;
    LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: findPi0Road: Go to the North ";

    while (((next = theESNav.north()) != ESDetId(0) && next != strip)) {
      LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: findPi0Road: North: " << n_north
                               << " current strip is " << next;

      vout.push_back(next);
      ++n_north;
      if (n_north == preshSeededNstr_)
        break;
    }
    // south road
    int n_south = 0;
    LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: findPi0Road: Go to the South ";

    theESNav.home();
    while (((next = theESNav.south()) != ESDetId(0) && next != strip)) {
      LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: findPi0Road: South: " << n_south
                               << " current strip is " << next;

      vout.push_back(next);
      ++n_south;
      if (n_south == preshSeededNstr_)
        break;
    }
    LogTrace("EcalClusters")
        << "EndcapPiZeroDiscriminatorAlgo: findPi0Road: Total number of strips found in the road at 2-nd plane is "
        << n_south + n_north;

  } else {
    LogTrace("EcalClusters")
        << "EndcapPiZeroDiscriminatorAlgo: findPi0Road: Wrong plane number, null cluster will be returned! ";

  }  // end of if

  theESNav.home();
}

//===================================================================
// EndcapPiZeroDiscriminatorAlgo::readWeightFile(...), a method that reads the weigths of the NN
// INPUT: Weights_file
// OUTPUT: I_H_Weight, H_Thresh, H_O_Weight, O_Thresh arrays
//===================================================================
void EndcapPiZeroDiscriminatorAlgo::readWeightFile(
    const char* Weights_file, int& Layers, int& Indim, int& Hidden, int& Outdim) {
  FILE* weights = nullptr;

  char line[80];

  bool checkinit = false;
  // Open the weights file, generated by jetnet, and read
  // in the nodes and weights
  //*******************************************************
  weights = fopen(Weights_file, "r");
  LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: I opeded the Weights file  = " << Weights_file;
  if (weights == nullptr) {
    throw cms::Exception("MissingWeightFile") << "Could not open the weights file: " << Weights_file;
  }

  const auto I_H_W_offset = I_H_Weight_all.size();
  const auto H_O_W_offset = H_O_Weight_all.size();
  const auto H_T_offset = H_Thresh_all.size();
  const auto O_T_offset = O_Thresh_all.size();

  while (!feof(weights)) {
    fscanf(weights, "%s", line);
    if (line[0] == 'A') {              //Read in ANN nodes: Layers, input , Hidden, Output
      fscanf(weights, "%d", &Layers);  // # of NN Layers  used
      fscanf(weights, "%d", &Indim);   // # of Inputs actually used
      fscanf(weights, "%d", &Hidden);  // # of hidden nodes
      fscanf(weights, "%d", &Outdim);  // # of output nodes

      I_H_Weight_all.resize(I_H_W_offset + Indim * Hidden);
      H_Thresh_all.resize(H_T_offset + Hidden);
      H_O_Weight_all.resize(H_O_W_offset + Hidden * Outdim);
      O_Thresh_all.resize(O_T_offset + Outdim);
      checkinit = true;
    } else if (line[0] == 'B') {  // read in weights between hidden and intput nodes
      assert(checkinit);
      for (int i = 0; i < Indim; i++) {
        for (int j = 0; j < Hidden; j++) {
          fscanf(weights, "%f", &I_H_Weight_all[I_H_W_offset + i * Hidden + j]);
        }
      }
    } else if (line[0] == 'C') {  // Read in the thresholds for hidden nodes
      assert(checkinit);
      for (int i = 0; i < Hidden; i++) {
        fscanf(weights, "%f", &H_Thresh_all[H_T_offset + i]);
      }
    } else if (line[0] == 'D') {  // read in weights between hidden and output nodes
      assert(checkinit);
      for (int i = 0; i < Hidden * Outdim; i++) {
        fscanf(weights, "%f", &H_O_Weight_all[H_O_W_offset + i]);
      }
    } else if (line[0] == 'E') {  // read in the threshold for the output nodes
      assert(checkinit);
      for (int i = 0; i < Outdim; i++) {
        fscanf(weights, "%f", &O_Thresh_all[O_T_offset + i]);
      }
    } else {
      edm::LogError("EEPi0Discrim") << "EndcapPiZeroDiscriminatorAlgo: Not a Net file of Corrupted Net file " << endl;
    }
  }
  fclose(weights);
}

//=====================================================================================
// EndcapPiZeroDiscriminatorAlgo::getNNoutput(int sel_wfile), a method that calculated the NN output
// INPUT: sel_wfile -> Weight file selection
// OUTPUT : nnout -> the NN output
//=====================================================================================

float EndcapPiZeroDiscriminatorAlgo::getNNoutput(
    int sel_wfile, int Layers, int Indim, int Hidden, int Outdim, int barrelstart) const {
  float nnout = 0.0;
  int mij;

  std::vector<float> I_SUM(size_t(Hidden), 0.0);
  std::vector<float> OUT(size_t(Outdim), 0.0);

  for (int h = 0; h < Hidden; h++) {
    mij = h - Hidden;
    for (int i = 0; i < Indim; i++) {
      mij = mij + Hidden;
      I_SUM[h] += I_H_Weight_all[mij + sel_wfile * Indim * Hidden + barrelstart * Nfiles_EE * EE_Indim * EE_Hidden] *
                  input_var[i];
    }
    I_SUM[h] += H_Thresh_all[h + sel_wfile * Hidden + barrelstart * Nfiles_EE * EE_Hidden];
    for (int o1 = 0; o1 < Outdim; o1++) {
      OUT[o1] += H_O_Weight_all[barrelstart * Nfiles_EE * EE_Outdim * EE_Hidden + h * Outdim + o1 +
                                sel_wfile * Outdim * Hidden] *
                 Activation_fun(I_SUM[h]);
    }
  }
  for (int o2 = 0; o2 < Outdim; o2++) {
    OUT[o2] += O_Thresh_all[barrelstart * Nfiles_EE * EE_Outdim + o2 + sel_wfile * Outdim];
  }
  nnout = Activation_fun(OUT[0]);
  LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: getNNoutput :: -> NNout = " << nnout;

  return (nnout);
}

float EndcapPiZeroDiscriminatorAlgo::Activation_fun(float SUM) const { return (1.0 / (1.0 + exp(-2.0 * SUM))); }
//=====================================================================================
// EndcapPiZeroDiscriminatorAlgo::calculateNNInputVariables(...), a method that calculates the 25 input variables
// INPUTS:
// vph1 -> vector of the stip energies in 1st Preshower plane
// vph2 -> vector of the stip energies in 2nd Preshower plane
// pS1_max -> E1
// pS9_max -> E9
// pS25_max -> E25
// OUTPUT:
// input_var[25] -> the 25 input to the NN variables array
//=====================================================================================
bool EndcapPiZeroDiscriminatorAlgo::calculateNNInputVariables(
    vector<float>& vph1, vector<float>& vph2, float pS1_max, float pS9_max, float pS25_max, int EScorr) {
  input_var.resize(EE_Indim);
  bool valid_NNinput = true;

  /*   
   for(int i = 0; i<11;i++) {
   LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: Energies of the Preshower Strips in X plane = " << vph1[i] ;
   }
   
   for(int i = 0; i<11;i++) {
   LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: Energies of the Preshower Strips in Y plane = " << vph2[i] ;
   } 
   */

  // check if all Preshower info is availabla - If NOT use remaning info
  for (int k = 0; k < 11; k++) {
    if (vph1[k] < 0) {
      LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: Oops!!! Preshower Info for strip : " << k
                               << " of X plane Do not exists";

      vph1[k] = 0.0;
    }
    if (vph2[k] < 0) {
      LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: Oops!!! Preshower Info for strip : " << k
                               << " of Y plane Do not exists";

      vph2[k] = 0.0;
    }
  }

  /*
   for(int i = 0; i<11;i++) {  
     LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: After: Energies of the Preshower Strips in X plane = " << vph1[i] ;
   }

   for(int i = 0; i<11;i++) {
 
     LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: After: Energies of the Preshower Strips in Y plane = " << vph2[i] ;
   }
   */

  // FIRST : Produce the 22 NN variables related with the Preshower
  // --------------------------------------------------------------
  // New normalization of the preshower strip energies Aris 8/11/2004
  for (int kk = 0; kk < 11; kk++) {
    input_var[kk] = fabs(vph1[kk] / 0.01);
    input_var[kk + 11] = fabs(vph2[kk] / 0.02);
    if (input_var[kk] < 0.0001)
      input_var[kk] = 0.;
    if (input_var[kk + 11] < 0.0001)
      input_var[kk + 11] = 0.;
  }
  input_var[0] = fabs(input_var[0] / 2.);
  input_var[1] = fabs(input_var[1] / 2.);
  input_var[6] = fabs(input_var[6] / 2.);
  input_var[11] = fabs(input_var[11] / 2.);
  input_var[12] = fabs(input_var[12] / 2.);
  input_var[17] = fabs(input_var[17] / 2.);

  // correction for version > CMSSW_3_1_0_pre5 where extra enegry is given to the ES strips
  // Aris 18/5/2009
  if (EScorr == 1) {
    input_var[0] -= 0.05;
    input_var[1] -= 0.035;
    input_var[2] -= 0.035;
    input_var[3] -= 0.02;
    input_var[4] -= 0.015;
    input_var[5] -= 0.0075;
    input_var[6] -= 0.035;
    input_var[7] -= 0.035;
    input_var[8] -= 0.02;
    input_var[9] -= 0.015;
    input_var[10] -= 0.0075;

    input_var[11] -= 0.05;
    input_var[12] -= 0.035;
    input_var[13] -= 0.035;
    input_var[14] -= 0.02;
    input_var[15] -= 0.015;
    input_var[16] -= 0.0075;
    input_var[17] -= 0.035;
    input_var[18] -= 0.035;
    input_var[19] -= 0.02;
    input_var[20] -= 0.015;
    input_var[21] -= 0.0075;

    for (int kk1 = 0; kk1 < 22; kk1++) {
      if (input_var[kk1] < 0)
        input_var[kk1] = 0.0;
    }
  }
  // SECOND: Take the final NN variable related to the ECAL
  // -----------------------------------------------
  float ECAL_norm_factor = 500.;
  if (pS25_max > 500 && pS25_max <= 1000)
    ECAL_norm_factor = 1000;
  if (pS25_max > 1000)
    ECAL_norm_factor = 7000;

  input_var[22] = pS1_max / ECAL_norm_factor;
  input_var[23] = pS9_max / ECAL_norm_factor;
  input_var[24] = pS25_max / ECAL_norm_factor;

  LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: S1/ECAL_norm_factor = " << input_var[22];
  LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: S9/ECAL_norm_factor = " << input_var[23];
  LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: S25/ECAL_norm_factor = " << input_var[24];

  for (int i = 0; i < EE_Indim; i++) {
    if (input_var[i] > 1.0e+00) {
      valid_NNinput = false;
      break;
    }
  }

  LogTrace("EcalClusters") << " valid_NNinput = " << valid_NNinput;

  return valid_NNinput;
}

//=====================================================================================
// EndcapPiZeroDiscriminatorAlgo::calculateBarrelNNInputVariables(...), a method that calculates
// the 12 barrel NN input
// OUTPUT:
// input_var[12] -> the 12 input to the barrel NN variables array
//=====================================================================================

void EndcapPiZeroDiscriminatorAlgo::calculateBarrelNNInputVariables(float et,
                                                                    double s1,
                                                                    double s9,
                                                                    double s25,
                                                                    double m2,
                                                                    double cee,
                                                                    double cep,
                                                                    double cpp,
                                                                    double s4,
                                                                    double s6,
                                                                    double ratio,
                                                                    double xcog,
                                                                    double ycog) {
  input_var.resize(EB_Indim);

  double lam, lam1, lam2;

  if (xcog < 0.) {
    input_var[0] = -xcog / s25;
  } else {
    input_var[0] = xcog / s25;
  }

  input_var[1] = cee / 0.0004;

  if (cpp < .001) {
    input_var[2] = cpp / .001;
  } else {
    input_var[2] = 0.;
  }

  if (s9 != 0.) {
    input_var[3] = s1 / s9;
    input_var[8] = s6 / s9;
    input_var[10] = (m2 + s1) / s9;
  } else {
    input_var[3] = 0.;
    input_var[8] = 0.;
    input_var[10] = 0.;
  }

  if (s25 - s1 > 0.) {
    input_var[4] = (s9 - s1) / (s25 - s1);
  } else {
    input_var[4] = 0.;
  }

  if (s25 > 0.) {
    input_var[5] = s4 / s25;
  } else {
    input_var[5] = 0.;
  }

  if (ycog < 0.) {
    input_var[6] = -ycog / s25;
  } else {
    input_var[6] = ycog / s25;
  }

  input_var[7] = ratio;

  lam = sqrt((cee - cpp) * (cee - cpp) + 4 * cep * cep);
  lam1 = (cee + cpp + lam) / 2;
  lam2 = (cee + cpp - lam) / 2;

  if (lam1 == 0) {
    input_var[9] = .0;
  } else {
    input_var[9] = lam2 / lam1;
  }
  if (s4 != 0.) {
    input_var[11] = (m2 + s1) / s4;
  } else {
    input_var[11] = 0.;
  }
}

//=====================================================================================
// EndcapPiZeroDiscriminatorAlgo::GetNNOutput(...), a method that calculates the NNoutput
// INPUTS: Super Cluster Energy
// OUTPUTS : NNoutput
//=====================================================================================
float EndcapPiZeroDiscriminatorAlgo::GetNNOutput(float EE_Et) {
  float nnout = -1;
  // Print the  NN input variables that are related to the Preshower + ECAL
  // ------------------------------------------------------------------------
  LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo::GetNNoutput :nn_invar_presh = ";

  LogTrace("EcalClusters").log([&](auto& lt) {
    for (auto const v : input_var) {
      lt << v << " ";
    }
  });
  LogTrace("EcalClusters") << " ";

  // select the appropriate Weigth file
  int sel_wfile;
  if (EE_Et < 25.0) {
    sel_wfile = 0;
  } else if (EE_Et >= 25.0 && EE_Et < 35.0) {
    sel_wfile = 1;
  } else if (EE_Et >= 35.0 && EE_Et < 45.0) {
    sel_wfile = 2;
  } else if (EE_Et >= 45.0 && EE_Et < 55.0) {
    sel_wfile = 3;
  } else {
    sel_wfile = 4;
  }

  LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: Et_SC = " << EE_Et
                           << " and I select Weight file Number = " << sel_wfile;

  nnout = getNNoutput(
      sel_wfile, EE_Layers, EE_Indim, EE_Hidden, EE_Outdim, 0);  // calculate the nnoutput for the given ECAL object

  LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: ===================> GetNNOutput : NNout = " << nnout;

  return nnout;
}

//=====================================================================================
// EndcapPiZeroDiscriminatorAlgo::GetBarrelNNOutput(...), a method that calculates the barrel NNoutput
// INPUTS: Super Cluster Energy
// OUTPUTS : NNoutput
//=====================================================================================
float EndcapPiZeroDiscriminatorAlgo::GetBarrelNNOutput(float EB_Et) {
  float nnout = -1;
  // Print the  NN input variables that are related to the ECAL Barrel
  // ------------------------------------------------------------------------
  LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo::GetBarrelNNoutput :nn_invar_presh = ";

  LogTrace("EcalCluster").log([&](auto& lt) {
    for (auto const v : input_var) {
      lt << v << " ";
    }
  });
  LogTrace("EcalClusters") << " ";

  // select the appropriate Weigth file
  int sel_wfile;
  if (EB_Et < 25.0) {
    sel_wfile = 0;
  } else if (EB_Et >= 25.0 && EB_Et < 35.0) {
    sel_wfile = 1;
  } else if (EB_Et >= 35.0 && EB_Et < 45.0) {
    sel_wfile = 2;
  } else if (EB_Et >= 45.0 && EB_Et < 55.0) {
    sel_wfile = 3;
  } else {
    sel_wfile = 4;
  }
  LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: E_SC = " << EB_Et
                           << " and I select Weight file Number = " << sel_wfile;

  nnout = getNNoutput(
      sel_wfile, EB_Layers, EB_Indim, EB_Hidden, EB_Outdim, 1);  // calculate the nnoutput for the given ECAL object

  LogTrace("EcalClusters") << "EndcapPiZeroDiscriminatorAlgo: ===================> GetNNOutput : NNout = " << nnout;

  return nnout;
}
