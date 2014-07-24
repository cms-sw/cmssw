// HardwareSortingMethods.cc
// Authors: R. Alex Barbieri
//          Ben Kreis
//
// This file should contain the C++ equivalents of the sorting
// algorithms used in Hardware. Most C++ methods originally written by
// Ben Kries.

#include <iostream>
#include "L1Trigger/L1TCalorimeter/interface/HardwareSortingMethods.h"

const bool verbose = true;

int fw_to_gt_phi_map[] = {4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5};
int gt_to_fw_phi_map[] = {4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5};

void print2DVector(std::vector<std::vector<l1t::Jet> > myVector){
  int nrows = myVector.size();
  int ncols = myVector[0].size();
  std::cout << std::endl;
  std::cout << "rows: " << nrows << std::endl;
  std::cout << "cols: " << ncols << std::endl;

  for(int r=0; r<nrows; r++){
    for(int c=0; c<ncols; c++){
      std::cout << std::setw(5) << myVector[r][c].hwPt() << ' ';
    }
    std::cout << std::endl;
  }
}

std::vector<l1t::Jet> sort_array(std::vector<l1t::Jet> inputArray){
  std::vector<l1t::Jet> outputArray(inputArray.size());
  for(unsigned int i=0; i<inputArray.size(); i++){
    int rank=0;
    for(unsigned int j=0; j<inputArray.size(); j++){
      if( (inputArray[i].hwPt() > inputArray[j].hwPt()) || ( (inputArray[i].hwPt() == inputArray[j].hwPt()) && i<j) ) rank++;
    }//j
    outputArray[outputArray.size()-1-rank] = inputArray[i];
  }//i
  return outputArray;
}

std::vector<std::vector<l1t::Jet> > presort(std::vector<std::vector<l1t::Jet> > energies, int rows, int cols, int detector=0){

  int row_block_length = energies.size() / cols;
  if(energies.size() % cols != 0) row_block_length++;

  l1t::Jet dummyJet;
  dummyJet.setHwPt(0);
  dummyJet.setHwPhi(99);
  dummyJet.setHwEta(99);
  std::vector<std::vector<l1t::Jet> > sorted_energies (rows, std::vector<l1t::Jet>(cols, dummyJet));
  if(verbose) print2DVector( sorted_energies );

  unsigned int row=0, col=0;
  std::vector<l1t::Jet> energy_feeder (cols, dummyJet);
  std::vector<l1t::Jet> energy_result (cols, dummyJet);
  for(int r=0; r<rows; r++){
    for(int c=0; c<cols; c++){

      row = (r % row_block_length)*cols+c;//row goes up to 19 and we pad with zeros

      if(row < energies.size()){
	energy_feeder[c] = energies[row][col];
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

std::vector<std::vector<l1t::Jet> > extract_sub_jet_energy_position_matrix(std::vector<std::vector<l1t::Jet> > input_matrix, unsigned int row_i, unsigned int row_f, unsigned int col_i, unsigned int col_f){
  std::vector<std::vector<l1t::Jet> > output_matrix(row_f-row_i+1,std::vector<l1t::Jet>(col_f-col_i+1));
  l1t::Jet dummyJet;
  dummyJet.setHwPt(0);
  dummyJet.setHwPhi(99);
  dummyJet.setHwEta(99);

  for(unsigned int i=0; i<row_f-row_i+1; i++){
    for(unsigned int j=0; j<col_f-col_i+1; j++){
      if(row_i+i > input_matrix.size()-1) output_matrix[i][j] = dummyJet;
      else output_matrix[i][j] = input_matrix[row_i+i][col_i+j];
    }//j
  }//i
  return output_matrix;
}

std::vector<std::vector<l1t::Jet> > sort_matrix_rows(std::vector<std::vector<l1t::Jet> > input_matrix){
  std::vector<std::vector<l1t::Jet> > output_matrix( input_matrix.size(), std::vector<l1t::Jet> (input_matrix[0].size()));

  for(unsigned int i=0; i<input_matrix.size(); i++){
    int rank=0;
    for (unsigned int j=0; j<input_matrix.size(); j++){
      if( (input_matrix[i][0].hwPt() > input_matrix[j][0].hwPt()) || ((input_matrix[i][0].hwPt() == input_matrix[j][0].hwPt()) && i<j)) rank++;
    }//j
    output_matrix[input_matrix.size()-1-rank] = input_matrix[i];
  }//i

  return output_matrix;
}

std::vector<std::vector<l1t::Jet> > sort_by_row_in_groups(std::vector<std::vector<l1t::Jet> > input_matrix, int group_size){
  int n_groups = input_matrix.size()/group_size + (1 - input_matrix.size()/group_size*group_size/input_matrix.size()); //constants must make this an integer
  std::vector<std::vector<l1t::Jet> > output_matrix(input_matrix.size()+(input_matrix.size() % group_size), std::vector<l1t::Jet> (input_matrix[0].size()));

  for(int g=0; g<n_groups; g++){
    std::vector<std::vector<l1t::Jet> > small_output_matrix = extract_sub_jet_energy_position_matrix(input_matrix, g*group_size, (g+1)*group_size-1, 0, input_matrix[0].size()-1 );
    small_output_matrix = sort_matrix_rows(small_output_matrix);

    for(unsigned int i=0; i<small_output_matrix.size(); i++){
      output_matrix[g*group_size+i]=small_output_matrix[i];
    }
  }

  if(verbose) print2DVector( output_matrix );
  return output_matrix;
}

std::vector<l1t::Jet> array_from_row_sorted_matrix(std::vector<std::vector<l1t::Jet> > input_matrix, unsigned int n_keep){
  std::vector<l1t::Jet> output_array (n_keep*(n_keep+1)/2);
  unsigned int max_row = n_keep-1;
  unsigned int max_col = n_keep-1;

  //compute size
  if(input_matrix.size() < n_keep) max_row = input_matrix.size();
  if(input_matrix[0].size() < n_keep) max_col = input_matrix[0].size();

  unsigned int array_position = 0;
  for(unsigned int i=0; i<max_row; i++){
    for(unsigned int j=0; j<max_col-i; j++){
      //cout << input_matrix[i][j].hwPt() << endl;
      output_array[array_position] = input_matrix[i][j];
      array_position++;
    }//j
  }//i

  //fill rest with zeros
  l1t::Jet dummyJet;
  dummyJet.setHwPt(0);
  dummyJet.setHwPhi(99);
  dummyJet.setHwEta(99);
  for(unsigned int k=array_position; k<output_array.size(); k++){
    output_array[k]=dummyJet;
  }

  //printVector(output_array);
  return output_array;
}

std::vector<std::vector<l1t::Jet> > super_sort_matrix_rows(std::vector<std::vector<l1t::Jet> > input_matrix, unsigned int group_size, unsigned int n_keep){
  unsigned int n_groups = input_matrix.size()/group_size + (1 - input_matrix.size()/group_size*group_size/input_matrix.size()); //constants must make this an integer
  std::vector<std::vector<l1t::Jet> > output_matrix(n_groups, std::vector<l1t::Jet>(n_keep));

  for(unsigned int g=0; g<n_groups; g++){
    std::vector<std::vector<l1t::Jet> > small_output_matrix = extract_sub_jet_energy_position_matrix(input_matrix, g*group_size, (g+1)*group_size-1, 0, input_matrix[0].size()-1 );
    std::vector<l1t::Jet> unsorted_array = array_from_row_sorted_matrix(small_output_matrix, n_keep);
    std::vector<l1t::Jet> unsorted_array_without_largest (unsorted_array.size()-1);//we know first element is the biggest
    for(unsigned int i=0; i<unsorted_array.size()-1; i++){
      unsorted_array_without_largest[i] = unsorted_array[1+i];
    }
    std::vector<l1t::Jet> sorted_array_without_largest = sort_array(unsorted_array_without_largest);

    std::vector<l1t::Jet> sorted_array (n_keep);
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
  void SortJets(std::vector<l1t::Jet> * input,
		std::vector<l1t::Jet> * output){
    const int CENTRAL_ETA_SLICES = 14;
    const int N_PHI_GROUPS = 5;
    const int N_PRESORTED_ROWS_CENTRAL = CENTRAL_ETA_SLICES*N_PHI_GROUPS;
    const int PRESORT_DEPTH = 4;
    const int N_KEEP_CENTRAL = 4;
    const int N_ETA_GROUP_SIZE_CENTRAL = 4;
    const int N_ETA_GROUPS_CENTRAL = 4;

    const int cen_nrows = 18;
    const int cen_ncols = 14;
    std::vector<std::vector<l1t::Jet> > cen_input_energy (cen_nrows, std::vector<l1t::Jet>(cen_ncols));

    for (std::vector<l1t::Jet>::const_iterator injet = input->begin();
	 injet != input->end(); ++injet){
      if(injet->hwEta() < 4 || injet->hwEta() > 17 ) continue;
      unsigned int myrow = gt_to_fw_phi_map[injet->hwPhi()];
      unsigned int mycol = injet->hwEta()-4; //hardcoding is bad
      cen_input_energy[myrow][mycol] = *injet;
    }

    //Each CLK is one clock

    //CLK 1
    std::vector<std::vector<l1t::Jet> > presorted_energies_matrix_sig = presort(cen_input_energy, N_PRESORTED_ROWS_CENTRAL, PRESORT_DEPTH);

    //CLK 2
    std::vector<std::vector<l1t::Jet> > row_presorted_energies_matrix_sig = sort_by_row_in_groups(presorted_energies_matrix_sig, N_PHI_GROUPS);

    //CLK 3
    std::vector<std::vector<l1t::Jet> > sorted_eta_slices_energies_matrix_sig = super_sort_matrix_rows(row_presorted_energies_matrix_sig, N_PHI_GROUPS, N_KEEP_CENTRAL);

    //CLK 4
    std::vector<std::vector<l1t::Jet> > row_presorted_eta_slices_energies_matrix_sig = sort_by_row_in_groups(sorted_eta_slices_energies_matrix_sig, N_ETA_GROUP_SIZE_CENTRAL);

    //CLK 5
    std::vector<std::vector<l1t::Jet> > sorted_eta_groups_energies_matrix_sig = super_sort_matrix_rows(row_presorted_eta_slices_energies_matrix_sig, N_ETA_GROUP_SIZE_CENTRAL, N_KEEP_CENTRAL);

    //CLK 6
    std::vector<std::vector<l1t::Jet> > row_presorted_eta_groups_energies_matrix_sig = sort_by_row_in_groups(sorted_eta_groups_energies_matrix_sig, N_ETA_GROUPS_CENTRAL);

    //CLK 7
    std::vector<std::vector<l1t::Jet> > sorted_final_energies_matrix_sig = super_sort_matrix_rows(row_presorted_eta_groups_energies_matrix_sig, N_ETA_GROUPS_CENTRAL, N_KEEP_CENTRAL);

    for(unsigned int i = 0; i < 4; ++i)
    {
      l1t::Jet intjet = sorted_final_energies_matrix_sig[0][i];
      output->push_back(intjet);
    }
  }
}
