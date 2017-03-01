// HardwareSortingMethods.cc
// Authors: R. Alex Barbieri
//          Ben Kreis
//
// This file should contain the C++ equivalents of the sorting
// algorithms used in Hardware. Most C++ methods originally written by
// Ben Kreis.

#include <iostream>
#include "L1Trigger/L1TCalorimeter/interface/HardwareSortingMethods.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool verbose = false;

int fw_to_gt_phi_map[] = {4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5};
int gt_to_fw_phi_map[] = {4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5};

namespace l1t{
  unsigned int pack15bits(int pt, int eta, int phi)
  {
    return( ((pt & 0x3f)) + ((eta & 0xf) << 6) + ((phi & 0x1f) << 10));
  }

  unsigned int pack16bits(int pt, int eta, int phi)
  {
    return( 0x8000 + ((pt & 0x3f)) + ((eta & 0xf) << 6) + ((phi & 0x1f) << 10));
  }

  unsigned int pack16bitsEgammaSpecial(int pt, int eta, int phi)
  {
    return( 0x8000 + ((pt & 0x3f) << 9) + ((eta & 0xf)) + ((phi & 0x1f) << 4));
  }
}

void print2DVector(std::vector<std::vector<l1t::L1Candidate> > myVector){
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

std::vector<l1t::L1Candidate> sort_array(std::vector<l1t::L1Candidate> inputArray){
  std::vector<l1t::L1Candidate> outputArray(inputArray.size());
  for(unsigned int i=0; i<inputArray.size(); i++){
    int rank=0;
    for(unsigned int j=0; j<inputArray.size(); j++){
      if( (inputArray[i].hwPt() > inputArray[j].hwPt()) || ( (inputArray[i].hwPt() == inputArray[j].hwPt()) && i<j) ) rank++;
    }//j
    outputArray[outputArray.size()-1-rank] = inputArray[i];
  }//i
  return outputArray;
}

std::vector<std::vector<l1t::L1Candidate> > presort(std::vector<std::vector<l1t::L1Candidate> > energies, int rows, int cols){

  int row_block_length = energies.size() / cols;
  if(energies.size() % cols != 0) row_block_length++;

  l1t::L1Candidate dummyJet;
  dummyJet.setHwPt(0);
  dummyJet.setHwPhi(99);
  dummyJet.setHwEta(99);
  dummyJet.setHwQual(0x10);
  std::vector<std::vector<l1t::L1Candidate> > sorted_energies (rows, std::vector<l1t::L1Candidate>(cols, dummyJet));
  if(verbose) print2DVector( sorted_energies );

  unsigned int row=0, col=0;
  std::vector<l1t::L1Candidate> energy_feeder (cols, dummyJet);
  std::vector<l1t::L1Candidate> energy_result (cols, dummyJet);
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

std::vector<std::vector<l1t::L1Candidate> > extract_sub_jet_energy_position_matrix(std::vector<std::vector<l1t::L1Candidate> > input_matrix, unsigned int row_i, unsigned int row_f, unsigned int col_i, unsigned int col_f){
  std::vector<std::vector<l1t::L1Candidate> > output_matrix(row_f-row_i+1,std::vector<l1t::L1Candidate>(col_f-col_i+1));
  l1t::L1Candidate dummyJet;
  dummyJet.setHwPt(0);
  dummyJet.setHwPhi(99);
  dummyJet.setHwEta(99);
  dummyJet.setHwQual(0x10);
  for(unsigned int i=0; i<row_f-row_i+1; i++){
    for(unsigned int j=0; j<col_f-col_i+1; j++){
      if(row_i+i > input_matrix.size()-1) output_matrix[i][j] = dummyJet;
      else output_matrix[i][j] = input_matrix[row_i+i][col_i+j];
    }//j
  }//i
  return output_matrix;
}

std::vector<std::vector<l1t::L1Candidate> > sort_matrix_rows(std::vector<std::vector<l1t::L1Candidate> > input_matrix){
  std::vector<std::vector<l1t::L1Candidate> > output_matrix( input_matrix.size(), std::vector<l1t::L1Candidate> (input_matrix[0].size()));

  for(unsigned int i=0; i<input_matrix.size(); i++){
    int rank=0;
    for (unsigned int j=0; j<input_matrix.size(); j++){
      if( (input_matrix[i][0].hwPt() > input_matrix[j][0].hwPt()) || ((input_matrix[i][0].hwPt() == input_matrix[j][0].hwPt()) && i<j)) rank++;
    }//j
    output_matrix[input_matrix.size()-1-rank] = input_matrix[i];
  }//i

  return output_matrix;
}

std::vector<std::vector<l1t::L1Candidate> > sort_by_row_in_groups(std::vector<std::vector<l1t::L1Candidate> > input_matrix, int group_size){
  int n_groups = input_matrix.size()/group_size + (1 - input_matrix.size()/group_size*group_size/input_matrix.size()); //constants must make this an integer
  //std::vector<std::vector<l1t::L1Candidate> > output_matrix(input_matrix.size()+(input_matrix.size() % group_size), std::vector<l1t::L1Candidate> (input_matrix[0].size()));
  std::vector<std::vector<l1t::L1Candidate> > output_matrix(input_matrix.size()
                                     +(group_size*(1 - ((input_matrix.size()/group_size)*group_size)/input_matrix.size()))
                                     -(input_matrix.size() % group_size), std::vector<l1t::L1Candidate> (input_matrix[0].size()));

  for(int g=0; g<n_groups; g++){
    std::vector<std::vector<l1t::L1Candidate> > small_output_matrix = extract_sub_jet_energy_position_matrix(input_matrix, g*group_size, (g+1)*group_size-1, 0, input_matrix[0].size()-1 );
    small_output_matrix = sort_matrix_rows(small_output_matrix);

    for(unsigned int i=0; i<small_output_matrix.size(); i++){
      output_matrix[g*group_size+i]=small_output_matrix[i];
    }
  }

  if(verbose) print2DVector( output_matrix );
  return output_matrix;
}

std::vector<l1t::L1Candidate> array_from_row_sorted_matrix(std::vector<std::vector<l1t::L1Candidate> > input_matrix, unsigned int n_keep){
  std::vector<l1t::L1Candidate> output_array (n_keep*(n_keep+1)/2);
  unsigned int max_row = n_keep-1;
  unsigned int max_col = n_keep-1;

  //compute size
  if(input_matrix.size() < n_keep) max_row = input_matrix.size()-1;
  if(input_matrix[0].size() < n_keep) max_col = input_matrix[0].size()-1;

  unsigned int array_position = 0;
  for(unsigned int i=0; i<=max_row; i++){
    for(unsigned int j=0; j<=max_col-i; j++){
      //cout << input_matrix[i][j].hwPt() << endl;
      output_array[array_position] = input_matrix[i][j];
      array_position++;
    }//j
  }//i

  //fill rest with zeros
  l1t::L1Candidate dummyJet;
  dummyJet.setHwPt(0);
  dummyJet.setHwPhi(99);
  dummyJet.setHwEta(99);
  dummyJet.setHwQual(0x10);
  for(unsigned int k=array_position; k<output_array.size(); k++){
    output_array[k]=dummyJet;
  }

  //printVector(output_array);
  return output_array;
}

std::vector<std::vector<l1t::L1Candidate> > super_sort_matrix_rows(std::vector<std::vector<l1t::L1Candidate> > input_matrix, unsigned int group_size, unsigned int n_keep){
  unsigned int n_groups = input_matrix.size()/group_size + (1 - input_matrix.size()/group_size*group_size/input_matrix.size()); //constants must make this an integer
  std::vector<std::vector<l1t::L1Candidate> > output_matrix(n_groups, std::vector<l1t::L1Candidate>(n_keep));

  for(unsigned int g=0; g<n_groups; g++){
    std::vector<std::vector<l1t::L1Candidate> > small_output_matrix = extract_sub_jet_energy_position_matrix(input_matrix, g*group_size, (g+1)*group_size-1, 0, input_matrix[0].size()-1 );
    std::vector<l1t::L1Candidate> unsorted_array = array_from_row_sorted_matrix(small_output_matrix, n_keep);
    std::vector<l1t::L1Candidate> unsorted_array_without_largest (unsorted_array.size()-1);//we know first element is the biggest
    for(unsigned int i=0; i<unsorted_array.size()-1; i++){
      unsorted_array_without_largest[i] = unsorted_array[1+i];
    }
    std::vector<l1t::L1Candidate> sorted_array_without_largest = sort_array(unsorted_array_without_largest);

    std::vector<l1t::L1Candidate> sorted_array (n_keep);
    sorted_array[0] = unsorted_array[0];
    for(unsigned int i=0; i<n_keep-1; i++){
      sorted_array[1+i]=sorted_array_without_largest[i];
    }

    output_matrix[g] = sorted_array;
  }//g

  if(verbose) print2DVector(output_matrix);
  return output_matrix;
}

std::vector<std::vector<l1t::L1Candidate> > presort_egamma(std::vector<l1t::L1Candidate> input_egamma, int rows, int cols){

  int row_block_length = input_egamma.size() / cols;
  if(input_egamma.size() % cols != 0) row_block_length++;

  //Initialize output
  l1t::L1Candidate dummyJet;
  dummyJet.setHwPt(0);
  dummyJet.setHwPhi(99);
  dummyJet.setHwEta(99);
  dummyJet.setHwQual(0x10);
  std::vector<std::vector<l1t::L1Candidate> > sorted_energies (rows, std::vector<l1t::L1Candidate>(cols, dummyJet));
  if(verbose) print2DVector( sorted_energies );

  unsigned int row=0, col=0;
  std::vector<l1t::L1Candidate> energy_feeder (cols, dummyJet);
  std::vector<l1t::L1Candidate> energy_result (cols, dummyJet);
  for(int r=0; r<rows; r++){
    for(int c=0; c<cols; c++){

      row = (r % row_block_length)*cols+c;//row goes up to 19 and we pad with zeros
      //cout << "row, col = " << row << ", " << col << endl;

      if(row < input_egamma.size()){
	energy_feeder[c] = input_egamma[row];
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


namespace l1t{
  void SortJets(std::vector<l1t::Jet> * input,
		std::vector<l1t::Jet> * output){
    //verbose = true;
    const int CENTRAL_ETA_SLICES = 14;
    const int N_PHI_GROUPS = 5;
    const int N_PRESORTED_ROWS_CENTRAL = CENTRAL_ETA_SLICES*N_PHI_GROUPS;
    const int PRESORT_DEPTH = 4;
    const int N_KEEP_CENTRAL = 4;
    const int N_ETA_GROUP_SIZE_CENTRAL = 4;
    const int N_ETA_GROUPS_CENTRAL = 4;

    const int HFM_ETA_SLICES = 4;
    const int HFP_ETA_SLICES = 4;
    const int N_PRESORTED_ROWS_HFM = HFM_ETA_SLICES*N_PHI_GROUPS;
    const int N_PRESORTED_ROWS_HFP = HFP_ETA_SLICES*N_PHI_GROUPS;
    const int N_KEEP_FORWARD = 4;

    const int cen_nrows = 18;
    const int cen_ncols = 14;
    const int hfm_nrows = 18, hfp_nrows = 18;
    const int hfm_ncols = 4, hfp_ncols = 4;

    std::vector<std::vector<l1t::L1Candidate> > cen_input_energy (cen_nrows, std::vector<l1t::L1Candidate>(cen_ncols));
    std::vector<std::vector<l1t::L1Candidate> > hfm_input_energy (hfm_nrows, std::vector<l1t::L1Candidate>(hfm_ncols));
    std::vector<std::vector<l1t::L1Candidate> > hfp_input_energy (hfp_nrows, std::vector<l1t::L1Candidate>(hfp_ncols));

    for (std::vector<l1t::Jet>::const_iterator injet = input->begin();
	 injet != input->end(); ++injet){
      if(injet->hwEta() >= 4 && injet->hwEta() <= 17 )
      {
	unsigned int myrow = gt_to_fw_phi_map[injet->hwPhi()];
	unsigned int mycol = injet->hwEta()-4; //hardcoding is bad
	cen_input_energy[myrow][mycol] = *injet;
      }
      else if(injet->hwEta() < 4)
      {
	unsigned int myrow = gt_to_fw_phi_map[injet->hwPhi()];
	unsigned int mycol = injet->hwEta(); //hardcoding is bad
	hfm_input_energy[myrow][mycol] = *injet;
      }
      else if(injet->hwEta() > 17)
      {
	unsigned int myrow = gt_to_fw_phi_map[injet->hwPhi()];
	unsigned int mycol = injet->hwEta()-18; //hardcoding is bad
	hfp_input_energy[myrow][mycol] = *injet;
      }
      else
	edm::LogError("HardwareJetSort") << "Region out of bounds: " << injet->hwEta();
    }

    for(int i = 0; i < cen_nrows; ++i)
      for(int j = 0; j < cen_ncols; ++j)
      {
	if(cen_input_energy[i][j].hwPt() == 0)
	{
	  cen_input_energy[i][j].setHwPhi(fw_to_gt_phi_map[i]);
	  cen_input_energy[i][j].setHwEta(4+j);
	}
      }

    for(int i = 0; i < hfm_nrows; ++i)
      for(int j = 0; j < hfm_ncols; ++j)
      {
	if(hfm_input_energy[i][j].hwPt() == 0)
	{
	  hfm_input_energy[i][j].setHwPhi(fw_to_gt_phi_map[i]);
	  hfm_input_energy[i][j].setHwEta(j);
	  hfm_input_energy[i][j].setHwQual(2);
	}
      }

    for(int i = 0; i < hfp_nrows; ++i)
      for(int j = 0; j < hfp_ncols; ++j)
      {
	if(hfp_input_energy[i][j].hwPt() == 0)
	{
	  hfp_input_energy[i][j].setHwPhi(fw_to_gt_phi_map[i]);
	  hfp_input_energy[i][j].setHwEta(j+18);
	  hfp_input_energy[i][j].setHwQual(2);
	}
      }

    //Each CLK is one clock

    //CLK 1
    std::vector<std::vector<l1t::L1Candidate> > presorted_energies_matrix_sig = presort(cen_input_energy, N_PRESORTED_ROWS_CENTRAL, PRESORT_DEPTH);
    std::vector<std::vector<l1t::L1Candidate> > hfm_presorted_energies_matrix_sig = presort(hfm_input_energy, N_PRESORTED_ROWS_HFM, PRESORT_DEPTH);
    std::vector<std::vector<l1t::L1Candidate> > hfp_presorted_energies_matrix_sig = presort(hfp_input_energy, N_PRESORTED_ROWS_HFP, PRESORT_DEPTH);

    //CLK 2
    std::vector<std::vector<l1t::L1Candidate> > row_presorted_energies_matrix_sig = sort_by_row_in_groups(presorted_energies_matrix_sig, N_PHI_GROUPS);
    std::vector<std::vector<l1t::L1Candidate> > hfm_row_presorted_energies_matrix_sig = sort_by_row_in_groups(hfm_presorted_energies_matrix_sig, N_PHI_GROUPS);
    std::vector<std::vector<l1t::L1Candidate> > hfp_row_presorted_energies_matrix_sig = sort_by_row_in_groups(hfp_presorted_energies_matrix_sig, N_PHI_GROUPS);

    //CLK 3
    std::vector<std::vector<l1t::L1Candidate> > sorted_eta_slices_energies_matrix_sig = super_sort_matrix_rows(row_presorted_energies_matrix_sig, N_PHI_GROUPS, N_KEEP_CENTRAL);
    std::vector<std::vector<l1t::L1Candidate> > hfm_sorted_eta_slices_energies_matrix_sig = super_sort_matrix_rows(hfm_row_presorted_energies_matrix_sig, N_PHI_GROUPS, N_KEEP_FORWARD);
    std::vector<std::vector<l1t::L1Candidate> > hfp_sorted_eta_slices_energies_matrix_sig = super_sort_matrix_rows(hfp_row_presorted_energies_matrix_sig, N_PHI_GROUPS, N_KEEP_FORWARD);

    //CLK 4
    std::vector<std::vector<l1t::L1Candidate> > row_presorted_eta_slices_energies_matrix_sig = sort_by_row_in_groups(sorted_eta_slices_energies_matrix_sig, N_ETA_GROUP_SIZE_CENTRAL);
    std::vector<std::vector<l1t::L1Candidate> > hfm_row_presorted_eta_slices_energies_matrix_sig = sort_by_row_in_groups(hfm_sorted_eta_slices_energies_matrix_sig, HFM_ETA_SLICES);
    std::vector<std::vector<l1t::L1Candidate> > hfp_row_presorted_eta_slices_energies_matrix_sig = sort_by_row_in_groups(hfp_sorted_eta_slices_energies_matrix_sig, HFP_ETA_SLICES);

    //CLK 5
    std::vector<std::vector<l1t::L1Candidate> > sorted_eta_groups_energies_matrix_sig = super_sort_matrix_rows(row_presorted_eta_slices_energies_matrix_sig, N_ETA_GROUP_SIZE_CENTRAL, N_KEEP_CENTRAL);
    std::vector<std::vector<l1t::L1Candidate> > hfm_sorted_final_energies_matrix_sig = super_sort_matrix_rows(hfm_row_presorted_eta_slices_energies_matrix_sig, HFM_ETA_SLICES, N_KEEP_FORWARD);
    std::vector<std::vector<l1t::L1Candidate> > hfp_sorted_final_energies_matrix_sig = super_sort_matrix_rows(hfp_row_presorted_eta_slices_energies_matrix_sig, HFP_ETA_SLICES, N_KEEP_FORWARD);

    //CLK 6
    std::vector<std::vector<l1t::L1Candidate> > row_presorted_eta_groups_energies_matrix_sig = sort_by_row_in_groups(sorted_eta_groups_energies_matrix_sig, N_ETA_GROUPS_CENTRAL);
    std::vector<std::vector<l1t::L1Candidate> > hf_merged_plus_minus_forward_energies_matrix_sig(2, std::vector<l1t::L1Candidate>(N_KEEP_FORWARD));
    hf_merged_plus_minus_forward_energies_matrix_sig[0] = hfm_sorted_final_energies_matrix_sig[0];
    hf_merged_plus_minus_forward_energies_matrix_sig[1] = hfp_sorted_final_energies_matrix_sig[0];
    std::vector<std::vector<l1t::L1Candidate> > hf_row_presorted_merged_plus_minus_forward_energies_matrix_sig = sort_by_row_in_groups(hf_merged_plus_minus_forward_energies_matrix_sig, 2);

    //CLK 7
    std::vector<std::vector<l1t::L1Candidate> > sorted_final_energies_matrix_sig = super_sort_matrix_rows(row_presorted_eta_groups_energies_matrix_sig, N_ETA_GROUPS_CENTRAL, N_KEEP_CENTRAL);
    std::vector<std::vector<l1t::L1Candidate> > hf_sorted_final_merged_plus_minus_forward_energies_matrix_sig = super_sort_matrix_rows(hf_row_presorted_merged_plus_minus_forward_energies_matrix_sig, 2, N_KEEP_FORWARD);

    for(unsigned int i = 0; i < 4; ++i)
    {
      l1t::Jet *intjet = static_cast<l1t::Jet *>( &sorted_final_energies_matrix_sig[0][i] );
      output->push_back(*intjet);
    }
    for(unsigned int i = 0; i < 4; ++i)
    {
      l1t::Jet *intjet = static_cast<l1t::Jet *>( &hf_sorted_final_merged_plus_minus_forward_energies_matrix_sig[0][i] );
      intjet->setHwQual(intjet->hwQual() | 2);
      output->push_back(*intjet);
    }
    //verbose = false;
  }

  void SortEGammas(std::vector<l1t::EGamma> * input,
		   std::vector<l1t::EGamma> * output)
  {
    //Initialize
    const int FIBER_PAIRS = 18;
    const int N_INPUT_EGAMMAS = 4;
    const int N_PRESORTED_ROWS_EGAMMA = 36;
    const int PRESORT_DEPTH = 4;
    const int N_EGAMMA_FIRST_GROUP_SIZE = 6;
    const int N_EGAMMA_SECOND_GROUP_SIZE = 6;
    const int N_EGAMMA_FIRST_GROUPS = 6;
    const int N_KEEP_EGAMMA = 4;

    //Input
    //Each egamma: RCT isolation, RCT order, phi index, eta index

    vector<l1t::L1Candidate> iso_egamma_array_p, iso_egamma_array_m; //reusing objects.  should probably rename to something like "object"
    vector<l1t::L1Candidate> noniso_egamma_array_p, noniso_egamma_array_m;

    for(int k=0; k<2*N_INPUT_EGAMMAS*FIBER_PAIRS; k++){
      l1t::L1Candidate dummyJet;
      dummyJet.setHwPt(0);
      dummyJet.setHwEta(99);
      dummyJet.setHwPhi(99);
      dummyJet.setHwQual(0x10);
      if(k<N_INPUT_EGAMMAS*FIBER_PAIRS){
	iso_egamma_array_p.push_back(dummyJet);
	noniso_egamma_array_p.push_back(dummyJet);
      }
      else{
	iso_egamma_array_m.push_back(dummyJet);
	noniso_egamma_array_m.push_back(dummyJet);
      }
    }

    for (std::vector<l1t::EGamma>::const_iterator ineg = input->begin();
	 ineg != input->end(); ++ineg){
      int fiberNum = (int) floor(gt_to_fw_phi_map[ineg->hwPhi()]/2);
      int index = ineg->hwQual();
      bool iso = ineg->hwIso();
      bool minus = (ineg->hwEta() < 11);

      // while waiting for firmware LUT, set all iso to true
      //iso = true;

      if(iso && minus)
	iso_egamma_array_m[8*fiberNum+index] = *ineg;
      else if (iso && !minus)
	iso_egamma_array_p[8*fiberNum+index] = *ineg;
      else if (!iso && minus)
	noniso_egamma_array_m[8*fiberNum+index] = *ineg;
      else if (!iso && !minus)
	noniso_egamma_array_p[8*fiberNum+index] = *ineg;

    }

    // std::cout << "iso_egamma_array_m" << std::endl;
    // for(int i = 0; i < (int)iso_egamma_array_m.size(); ++i)
    // {
    //   std::cout << iso_egamma_array_m[i].hwPt() << " "
    // 	   << iso_egamma_array_m[i].hwEta() << " "
    // 	   << iso_egamma_array_m[i].hwPhi() << std::endl;
    // }

    // std::cout << "iso_egamma_array_p" << std::endl;
    // for(int i = 0; i < (int)iso_egamma_array_p.size(); ++i)
    // {
    //   std::cout << iso_egamma_array_p[i].hwPt() << " "
    // 	   << iso_egamma_array_p[i].hwEta() << " "
    // 	   << iso_egamma_array_p[i].hwPhi() << std::endl;
    // }

    //verbose = true;
    //1
    std::vector<std::vector<l1t::L1Candidate> > presorted_iso_matrix_sig_p = presort_egamma(iso_egamma_array_p, N_PRESORTED_ROWS_EGAMMA/2, PRESORT_DEPTH);
    std::vector<std::vector<l1t::L1Candidate> > presorted_iso_matrix_sig_m = presort_egamma(iso_egamma_array_m, N_PRESORTED_ROWS_EGAMMA/2, PRESORT_DEPTH);
    std::vector<std::vector<l1t::L1Candidate> > presorted_non_iso_matrix_sig_p = presort_egamma(noniso_egamma_array_p, N_PRESORTED_ROWS_EGAMMA/2, PRESORT_DEPTH);
    std::vector<std::vector<l1t::L1Candidate> > presorted_non_iso_matrix_sig_m = presort_egamma(noniso_egamma_array_m, N_PRESORTED_ROWS_EGAMMA/2, PRESORT_DEPTH);

    //2
    std::vector<std::vector<l1t::L1Candidate> > iso_row_presorted_energies_matrix_sig_p = sort_by_row_in_groups(presorted_iso_matrix_sig_p, N_EGAMMA_FIRST_GROUP_SIZE);
    std::vector<std::vector<l1t::L1Candidate> > iso_row_presorted_energies_matrix_sig_m = sort_by_row_in_groups(presorted_iso_matrix_sig_m, N_EGAMMA_FIRST_GROUP_SIZE);
    std::vector<std::vector<l1t::L1Candidate> > non_iso_row_presorted_energies_matrix_sig_p = sort_by_row_in_groups(presorted_non_iso_matrix_sig_p, N_EGAMMA_FIRST_GROUP_SIZE);
    std::vector<std::vector<l1t::L1Candidate> > non_iso_row_presorted_energies_matrix_sig_m = sort_by_row_in_groups(presorted_non_iso_matrix_sig_m, N_EGAMMA_FIRST_GROUP_SIZE);

    //3
    std::vector<std::vector<l1t::L1Candidate> > iso_super_sorted_energies_matrix_sig_p = super_sort_matrix_rows(iso_row_presorted_energies_matrix_sig_p, N_EGAMMA_FIRST_GROUP_SIZE, N_KEEP_EGAMMA);
    std::vector<std::vector<l1t::L1Candidate> > iso_super_sorted_energies_matrix_sig_m = super_sort_matrix_rows(iso_row_presorted_energies_matrix_sig_m, N_EGAMMA_FIRST_GROUP_SIZE, N_KEEP_EGAMMA);
    std::vector<std::vector<l1t::L1Candidate> > non_iso_super_sorted_energies_matrix_sig_p = super_sort_matrix_rows(non_iso_row_presorted_energies_matrix_sig_p, N_EGAMMA_FIRST_GROUP_SIZE, N_KEEP_EGAMMA);
    std::vector<std::vector<l1t::L1Candidate> > non_iso_super_sorted_energies_matrix_sig_m = super_sort_matrix_rows(non_iso_row_presorted_energies_matrix_sig_m, N_EGAMMA_FIRST_GROUP_SIZE, N_KEEP_EGAMMA);
    //combine plus and minus
    std::vector<std::vector<l1t::L1Candidate> > iso_super_sorted_energies_matrix_sig (N_EGAMMA_FIRST_GROUPS, std::vector<l1t::L1Candidate>(N_KEEP_EGAMMA) );
    std::vector<std::vector<l1t::L1Candidate> > non_iso_super_sorted_energies_matrix_sig (N_EGAMMA_FIRST_GROUPS, std::vector<l1t::L1Candidate>(N_KEEP_EGAMMA) );
    for(int r=0; r<N_EGAMMA_FIRST_GROUPS/2; r++){
      iso_super_sorted_energies_matrix_sig[r] = iso_super_sorted_energies_matrix_sig_m[r];
      iso_super_sorted_energies_matrix_sig[r+N_EGAMMA_FIRST_GROUPS/2] = iso_super_sorted_energies_matrix_sig_p[r];
      non_iso_super_sorted_energies_matrix_sig[r] = non_iso_super_sorted_energies_matrix_sig_m[r];
      non_iso_super_sorted_energies_matrix_sig[r+N_EGAMMA_FIRST_GROUPS/2] = non_iso_super_sorted_energies_matrix_sig_p[r];
    }

    //4
    std::vector<std::vector<l1t::L1Candidate> > iso_stage2_row_sorted_matrix_sig = sort_by_row_in_groups(iso_super_sorted_energies_matrix_sig, N_EGAMMA_SECOND_GROUP_SIZE);
    std::vector<std::vector<l1t::L1Candidate> > non_iso_stage2_row_sorted_matrix_sig = sort_by_row_in_groups(non_iso_super_sorted_energies_matrix_sig, N_EGAMMA_SECOND_GROUP_SIZE);

    //5
    std::vector<std::vector<l1t::L1Candidate> > iso_stage2_super_sorted_matrix_sig = super_sort_matrix_rows(iso_stage2_row_sorted_matrix_sig, N_EGAMMA_SECOND_GROUP_SIZE, N_KEEP_EGAMMA);
    std::vector<std::vector<l1t::L1Candidate> > non_iso_stage2_super_sorted_matrix_sig = super_sort_matrix_rows(non_iso_stage2_row_sorted_matrix_sig, N_EGAMMA_SECOND_GROUP_SIZE, N_KEEP_EGAMMA);

    //Prepare output
    std::vector<l1t::L1Candidate> sorted_iso_egammas = iso_stage2_super_sorted_matrix_sig[0];
    std::vector<l1t::L1Candidate> sorted_noniso_egammas = non_iso_stage2_super_sorted_matrix_sig[0];

    //verbose = false;

    for(unsigned int i = 0; i < 4; ++i)
    {
      l1t::EGamma *ineg = static_cast<l1t::EGamma *>( &sorted_iso_egammas[i] );
      ineg->setHwIso(1);
      output->push_back(*ineg);
    }
    for(unsigned int i = 0; i < 4; ++i)
    {
      l1t::EGamma *ineg = static_cast<l1t::EGamma *>( &sorted_noniso_egammas[i] );
      output->push_back(*ineg);
    }
  }

  void SortTaus(std::vector<l1t::Tau> * input,
		std::vector<l1t::Tau> * output){
    const int CENTRAL_ETA_SLICES = 14;
    const int N_PHI_GROUPS = 5;
    const int N_PRESORTED_ROWS_CENTRAL = CENTRAL_ETA_SLICES*N_PHI_GROUPS;
    const int PRESORT_DEPTH = 4;
    const int N_KEEP_CENTRAL = 4;
    const int N_ETA_GROUP_SIZE_CENTRAL = 4;
    const int N_ETA_GROUPS_CENTRAL = 4;

    const int cen_nrows = 18;
    const int cen_ncols = 14;

    std::vector<std::vector<l1t::L1Candidate> > cen_input_energy (cen_nrows, std::vector<l1t::L1Candidate>(cen_ncols));

    for (std::vector<l1t::Tau>::const_iterator injet = input->begin();
	 injet != input->end(); ++injet){
      if(injet->hwEta() >= 4 && injet->hwEta() <= 17 )
      {
	unsigned int myrow = gt_to_fw_phi_map[injet->hwPhi()];
	unsigned int mycol = injet->hwEta()-4; //hardcoding is bad
	cen_input_energy[myrow][mycol] = *injet;
      }
      else
	edm::LogError("HardwareTauSort") << "Region out of bounds: " << injet->hwEta();
    }

    for(int i = 0; i < cen_nrows; ++i)
      for(int j = 0; j < cen_ncols; ++j)
      {
	if(cen_input_energy[i][j].hwPt() == 0)
	{
	  cen_input_energy[i][j].setHwPhi(fw_to_gt_phi_map[i]);
	  cen_input_energy[i][j].setHwEta(4+j);
	}
      }

    //Each CLK is one clock

    //CLK 1
    std::vector<std::vector<l1t::L1Candidate> > presorted_energies_matrix_sig = presort(cen_input_energy, N_PRESORTED_ROWS_CENTRAL, PRESORT_DEPTH);
    //CLK 2
    std::vector<std::vector<l1t::L1Candidate> > row_presorted_energies_matrix_sig = sort_by_row_in_groups(presorted_energies_matrix_sig, N_PHI_GROUPS);
    //CLK 3
    std::vector<std::vector<l1t::L1Candidate> > sorted_eta_slices_energies_matrix_sig = super_sort_matrix_rows(row_presorted_energies_matrix_sig, N_PHI_GROUPS, N_KEEP_CENTRAL);
    //CLK 4
    std::vector<std::vector<l1t::L1Candidate> > row_presorted_eta_slices_energies_matrix_sig = sort_by_row_in_groups(sorted_eta_slices_energies_matrix_sig, N_ETA_GROUP_SIZE_CENTRAL);
    //CLK 5
    std::vector<std::vector<l1t::L1Candidate> > sorted_eta_groups_energies_matrix_sig = super_sort_matrix_rows(row_presorted_eta_slices_energies_matrix_sig, N_ETA_GROUP_SIZE_CENTRAL, N_KEEP_CENTRAL);
    //CLK 6
    std::vector<std::vector<l1t::L1Candidate> > row_presorted_eta_groups_energies_matrix_sig = sort_by_row_in_groups(sorted_eta_groups_energies_matrix_sig, N_ETA_GROUPS_CENTRAL);
    //CLK 7
    std::vector<std::vector<l1t::L1Candidate> > sorted_final_energies_matrix_sig = super_sort_matrix_rows(row_presorted_eta_groups_energies_matrix_sig, N_ETA_GROUPS_CENTRAL, N_KEEP_CENTRAL);

    for(unsigned int i = 0; i < 4; ++i)
    {
      l1t::Tau *intjet = static_cast<l1t::Tau *>( &sorted_final_energies_matrix_sig[0][i] );
      output->push_back(*intjet);
    }
  }
}
