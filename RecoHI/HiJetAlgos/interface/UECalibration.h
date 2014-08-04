#ifndef __HiJetAlgos_UECalibration_h__
#define __HiJetAlgos_UECalibration_h__

/////////////////////////////////////////////////////////////////////
// SVD Block Predictor
#include <fstream>
#include <sstream>

#include "FWCore/ParameterSet/interface/FileInPath.h"

struct UECalibration{
  explicit UECalibration(bool isRealData = true, bool isCalo = false){
	   np[0] = 3;	// Number of reduced PF ID (track, ECAL, HCAL)
	   np[1] = 15;	// Number of pseudorapidity block
	   np[2] = 5;	// Fourier series order
	   np[3] = 2;	// Re or Im
	   np[4] = 82;	// Number of feature parameter

      ni0[0] = np[1];
      ni0[1] = 344;

      ni1[0] = np[1];
      ni1[1] = 344;

      ni2[0] = np[1];
      ni2[1] = 82;

      index = 0;

      unsigned int Nnp_full = np[0] * np[1] * np[2] * np[3] * np[4];
      unsigned int Nnp = np[0] * np[1] * (1 + (np[2] - 1) * np[3]) * np[4];
      unsigned int Nni0 = ni0[0]*ni0[1];
      unsigned int Nni1 = ni1[0]*ni1[1];
      unsigned int Nni2 = ni2[0]*ni2[1];

      memset(ue_predictor_pf, 0, Nnp_full * sizeof(float));
      memset(ue_interpolation_pf0, 0, Nni0 * sizeof(float));
      memset(ue_interpolation_pf1, 0, Nni1 * sizeof(float));
      memset(ue_interpolation_pf2, 0, Nni2 * sizeof(float));

      std::string calibrationFile;

      if(isCalo){
        if(isRealData) calibrationFile = "RecoHI/HiJetAlgos/data/ue_calibrations_calo_data.txt";
	if(!isRealData) calibrationFile = "RecoHI/HiJetAlgos/data/ue_calibrations_calo_mc.txt";
      }else{
	if(isRealData) calibrationFile = "RecoHI/HiJetAlgos/data/ue_calibrations_pf_data.txt";
	if(!isRealData) calibrationFile = "RecoHI/HiJetAlgos/data/ue_calibrations_pf_mc.txt";
      }

      edm::FileInPath ueData(calibrationFile.data());
      std::string qpDataName = ueData.fullPath();
      std::ifstream in( qpDataName.c_str() );
      std::string line;

      while( std::getline( in, line)){      
	 if(!line.size() || line[0]=='#') {
	    std::cout<<" continue "<<std::endl;
	    continue;
	 }
	 std::istringstream linestream(line);
	 float val;
	 int bin0, bin1, bin2, bin3, bin4;
	 if(index < Nnp){
	    //	    cout<<"predictor "<<bin0<<" "<<bin1<<" "<<bin2<<" "<<bin3<<" "<<bin4<<" "<<val<<endl;
	    linestream>>bin0>>bin1>>bin2>>bin3>>bin4>>val;
	    ue_predictor_pf[bin0][bin1][bin2][bin3][bin4] = val;
	 }else if(index < Nnp + Nni0){
	    //            cout<<"inter_0 "<<bin0<<" "<<bin1<<" "<<bin2<<" "<<bin3<<" "<<bin4<<" "<<val<<endl;
	    linestream>>bin0>>bin1>>val;
	    ue_interpolation_pf0[bin0][bin1] = val;
	 }else if(index < Nnp + Nni0 + Nni1){
	    //            cout<<"inter_1 "<<bin0<<" "<<bin1<<" "<<bin2<<" "<<bin3<<" "<<bin4<<" "<<val<<endl;
            linestream>>bin0>>bin1>>val;
            ue_interpolation_pf1[bin0][bin1] = val;
	 }else if(index < Nnp + Nni0 + Nni1 + Nni2){
	    //            cout<<"inter_2 "<<bin0<<" "<<bin1<<" "<<bin2<<" "<<bin3<<" "<<bin4<<" "<<val<<endl;
	    linestream>>bin0>>bin1>>val;
	    ue_interpolation_pf2[bin0][bin1] = val;
	 }
	 ++index;
      }
   }

   unsigned int index,
      np[5], 
      ni0[2], 
      ni1[2], 
      ni2[2];

   float ue_predictor_pf[3][15][5][2][82], 
      ue_interpolation_pf0[15][344], 
      ue_interpolation_pf1[15][344], 
      ue_interpolation_pf2[15][82];
   
};


#endif



