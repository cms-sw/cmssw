// HardwareSortingMethods.cc
// Authors: R. Alex Barbieri
//          Ben Kries
//
// This file should contain the C++ equivalents of the sorting
// algorithms used in Hardware. Most C++ methods originally written by
// Ben Kries.

#include <iostream>
#include "L1Trigger/L1TCalorimeter/interface/HardwareSortingMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"

using namespace std;

const bool verbose = true;

//Mappings between firmware phi to GT phi
int fw_to_gt_phi_map[] = {4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5};
int gt_to_fw_phi_map[] = {4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5};//obvious that this is the same? mind blown.

//For C++ convenience.
//In the firmware, we just have one vector of bits that holds this info.
struct jet {
  int energy;
  int phi;
  int eta;
};

void print2DVector(vector<vector<jet> > myVector){
  int nrows = myVector.size();
  int ncols = myVector[0].size();
  cout << endl;
  cout << "rows: " << nrows << endl;
  cout << "cols: " << ncols << endl;

  for(int r=0; r<nrows; r++){
    for(int c=0; c<ncols; c++){
      cout << setw(5) << myVector[r][c].energy << ' ';
    }
    cout << endl;
  }
}

// void setBits(unsigned int & myNumber, unsigned int value, int bitPosition, unsigned int numberOfBits){
//   unsigned int mask = -1;
//   for(unsigned int i=bitPosition; i<bitPosition+numberOfBits; i++){
//     mask &= ~(((unsigned int)1)<<bitPosition);
//   }

//   //cout << hex << mask << " " << myNumber << " " << (value << bitPosition) << endl;
//   myNumber &= mask;
//   myNumber += (value << bitPosition)&(~mask);
//   //cout << hex << mask << " " << myNumber << " " << (value << bitPosition) << endl;
// }

vector<jet> sort_array(vector<jet> inputArray){
  vector<jet> outputArray(inputArray.size());
  for(unsigned int i=0; i<inputArray.size(); i++){
    int rank=0;
    for(unsigned int j=0; j<inputArray.size(); j++){
      if( (inputArray[i].energy > inputArray[j].energy) || ( (inputArray[i].energy == inputArray[j].energy) && i<j) ) rank++;
    }//j
    outputArray[outputArray.size()-1-rank] = inputArray[i];
  }//i
  return outputArray;
}

vector<vector<jet> > presort(vector<vector<int> > energies, int rows, int cols, int detector=0){

  int row_block_length = energies.size() / cols;
  if(energies.size() % cols != 0) row_block_length++;
  //cout << row_block_length << endl;

  //Initialize output
  jet dummyJet;
  dummyJet.energy=0;
  dummyJet.phi=99;
  dummyJet.eta=99;
  vector<vector<jet> > sorted_energies (rows, vector<jet>(cols, dummyJet));
  if(verbose) print2DVector( sorted_energies );

  unsigned int row=0, col=0;
  vector<jet> energy_feeder (cols, dummyJet);
  vector<jet> energy_result (cols, dummyJet);
  for(int r=0; r<rows; r++){
    for(int c=0; c<cols; c++){

      row = (r % row_block_length)*cols+c;//row goes up to 19 and we pad with zeros
      //cout << "row, col = " << row << ", " << col << endl;

      if(row < energies.size()){
	jet myJet; //filled from energies[row][col]
	myJet.energy = energies[row][col];

	//Use GT convention for eta and phi
	// unsigned int eta_GT_convention = 99;
	// if(detector == 0){
	//   if(col <= 6){
	//     eta_GT_convention = 6-col;
	//     setBits(eta_GT_convention, 1, 3, 1);//minus
	//   }
	//   else if(col >= 7){
	//     eta_GT_convention = col-7;
	//     setBits(eta_GT_convention, 0, 3, 1);//plus
	//   }
	// }
	// else {assert(0);}
	unsigned eta_GT_convention = l1t::gtEta(col+4); // hardcoded
	//cout << (bitset<5>) eta_GT_convention << endl;
	myJet.eta = eta_GT_convention;
	myJet.phi = fw_to_gt_phi_map[row];

	energy_feeder[c] = myJet;
      }
      else{
	energy_feeder[c] = dummyJet;
      }

    }//c

    energy_result = sort_array(energy_feeder);//sort!

    sorted_energies[r] = energy_result;

    if(r % row_block_length == row_block_length - 1) col++;

  }//r
  if(verbose) print2DVector( sorted_energies );

  return sorted_energies;
}

vector<vector<jet> > extract_sub_jet_energy_position_matrix(vector<vector<jet> > input_matrix, unsigned int row_i, unsigned int row_f, unsigned int col_i, unsigned int col_f){
  vector<vector<jet> > output_matrix(row_f-row_i+1,vector<jet>(col_f-col_i+1));
  jet dummyJet;
  dummyJet.energy=0;
  dummyJet.phi=99;
  dummyJet.eta=99;

  for(unsigned int i=0; i<row_f-row_i+1; i++){
    for(unsigned int j=0; j<col_f-col_i+1; j++){
      if(row_i+i > input_matrix.size()-1) output_matrix[i][j] = dummyJet;
      else output_matrix[i][j] = input_matrix[row_i+i][col_i+j];
    }//j
  }//i
  return output_matrix;
}

vector<vector<jet> > sort_matrix_rows(vector<vector<jet> > input_matrix){
  vector<vector<jet> > output_matrix( input_matrix.size(), vector<jet> (input_matrix[0].size()));

  for(unsigned int i=0; i<input_matrix.size(); i++){
    int rank=0;
    for (unsigned int j=0; j<input_matrix.size(); j++){
      if( (input_matrix[i][0].energy > input_matrix[j][0].energy) || ((input_matrix[i][0].energy == input_matrix[j][0].energy) && i<j)) rank++;
    }//j
    output_matrix[input_matrix.size()-1-rank] = input_matrix[i];
  }//i

  return output_matrix;
}

vector<vector<jet> > sort_by_row_in_groups(vector<vector<jet> > input_matrix, int group_size){
  int n_groups = input_matrix.size()/group_size + (1 - input_matrix.size()/group_size*group_size/input_matrix.size()); //constants must make this an integer
  vector<vector<jet> > output_matrix(input_matrix.size()+(input_matrix.size() % group_size), vector<jet> (input_matrix[0].size()));

  for(int g=0; g<n_groups; g++){
    vector<vector<jet> > small_output_matrix = extract_sub_jet_energy_position_matrix(input_matrix, g*group_size, (g+1)*group_size-1, 0, input_matrix[0].size()-1 );
    small_output_matrix = sort_matrix_rows(small_output_matrix);

    for(unsigned int i=0; i<small_output_matrix.size(); i++){
      output_matrix[g*group_size+i]=small_output_matrix[i];
    }
  }

  if(verbose) print2DVector( output_matrix );
  return output_matrix;
}

vector<jet> array_from_row_sorted_matrix(vector<vector<jet> > input_matrix, unsigned int n_keep){
  vector<jet> output_array (n_keep*(n_keep+1)/2);
  unsigned int max_row = n_keep-1;
  unsigned int max_col = n_keep-1;

  //compute size
  if(input_matrix.size() < n_keep) max_row = input_matrix.size();
  if(input_matrix[0].size() < n_keep) max_col = input_matrix[0].size();

  unsigned int array_position = 0;
  for(unsigned int i=0; i<max_row; i++){
    for(unsigned int j=0; j<max_col-i; j++){
      //cout << input_matrix[i][j].energy << endl;
      output_array[array_position] = input_matrix[i][j];
      array_position++;
    }//j
  }//i

  //fill rest with zeros
  jet dummyJet;
  dummyJet.energy=0;
  dummyJet.phi=99;
  dummyJet.eta=99;
  for(unsigned int k=array_position; k<output_array.size(); k++){
    output_array[k]=dummyJet;
  }

  //printVector(output_array);
  return output_array;
}

vector<vector<jet> > super_sort_matrix_rows(vector<vector<jet> > input_matrix, unsigned int group_size, unsigned int n_keep){
  unsigned int n_groups = input_matrix.size()/group_size + (1 - input_matrix.size()/group_size*group_size/input_matrix.size()); //constants must make this an integer
  vector<vector<jet> > output_matrix(n_groups, vector<jet>(n_keep));

  for(unsigned int g=0; g<n_groups; g++){
    vector<vector<jet> > small_output_matrix = extract_sub_jet_energy_position_matrix(input_matrix, g*group_size, (g+1)*group_size-1, 0, input_matrix[0].size()-1 );
    vector<jet> unsorted_array = array_from_row_sorted_matrix(small_output_matrix, n_keep);
    vector<jet> unsorted_array_without_largest (unsorted_array.size()-1);//we know first element is the biggest
    for(unsigned int i=0; i<unsorted_array.size()-1; i++){
      unsorted_array_without_largest[i] = unsorted_array[1+i];
    }
    vector<jet> sorted_array_without_largest = sort_array(unsorted_array_without_largest);

    vector<jet> sorted_array (n_keep);
    sorted_array[0] = unsorted_array[0];
    for(unsigned int i=0; i<n_keep-1; i++){
      sorted_array[1+i]=sorted_array_without_largest[i];
    }

    output_matrix[g] = sorted_array;
  }//g

  if(verbose) print2DVector(output_matrix);
  return output_matrix;
}

namespace l1t{
  void SortJets(vector<Jet> * input,
		vector<Jet> * output){
    const int CENTRAL_ETA_SLICES = 14;
    const int N_PHI_GROUPS = 5;
    const int N_PRESORTED_ROWS_CENTRAL = CENTRAL_ETA_SLICES*N_PHI_GROUPS;
    const int PRESORT_DEPTH = 4;
    const int N_KEEP_CENTRAL = 4;
    const int N_ETA_GROUP_SIZE_CENTRAL = 4;
    const int N_ETA_GROUPS_CENTRAL = 4;

    const int cen_nrows = 18;
    const int cen_ncols = 14;
    vector<vector<int> > cen_input_energy (cen_nrows, vector<int>(cen_ncols));

    for (vector<Jet>::const_iterator injet = input->begin();
	 injet != input->end(); ++injet){
      if(injet->hwEta() < 4 || injet->hwEta() > 17 ) continue;
      unsigned int myrow = gt_to_fw_phi_map[injet->hwPhi()];
      unsigned int mycol = injet->hwEta()-4; //hopefully that's right, hardcoding is bad
      cen_input_energy[myrow][mycol] = injet->hwPt();
    }

    //Each CLK is one clock

    //CLK 1
    vector<vector<jet> > presorted_energies_matrix_sig = presort(cen_input_energy, N_PRESORTED_ROWS_CENTRAL, PRESORT_DEPTH);

    //CLK 2
    vector<vector<jet> > row_presorted_energies_matrix_sig = sort_by_row_in_groups(presorted_energies_matrix_sig, N_PHI_GROUPS);

    //CLK 3
    vector<vector<jet> > sorted_eta_slices_energies_matrix_sig = super_sort_matrix_rows(row_presorted_energies_matrix_sig, N_PHI_GROUPS, N_KEEP_CENTRAL);

    //CLK 4
    vector<vector<jet> > row_presorted_eta_slices_energies_matrix_sig = sort_by_row_in_groups(sorted_eta_slices_energies_matrix_sig, N_ETA_GROUP_SIZE_CENTRAL);

    //CLK 5
    vector<vector<jet> > sorted_eta_groups_energies_matrix_sig = super_sort_matrix_rows(row_presorted_eta_slices_energies_matrix_sig, N_ETA_GROUP_SIZE_CENTRAL, N_KEEP_CENTRAL);

    //CLK 6
    vector<vector<jet> > row_presorted_eta_groups_energies_matrix_sig = sort_by_row_in_groups(sorted_eta_groups_energies_matrix_sig, N_ETA_GROUPS_CENTRAL);

    //CLK 7
    vector<vector<jet> > sorted_final_energies_matrix_sig = super_sort_matrix_rows(row_presorted_eta_groups_energies_matrix_sig, N_ETA_GROUPS_CENTRAL, N_KEEP_CENTRAL);

    for(unsigned int i = 0; i < 4; ++i)
    {
      jet intjet = sorted_final_energies_matrix_sig[0][i];

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > jetLorentz(0,0,0,0);
      l1t::Jet outjet(*&jetLorentz, intjet.energy, intjet.eta, intjet.phi, 0);
      output->push_back(outjet);
    }

  }
}
