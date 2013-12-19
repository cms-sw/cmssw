#ifndef __HiJetAlgos_UECalibration_h__
#define __HiJetAlgos_UECalibration_h__

/////////////////////////////////////////////////////////////////////
// SVD Block Predictor
#include <fstream>
#include <sstream>

#include "FWCore/ParameterSet/interface/FileInPath.h"

struct UECalibration{
   UECalibration(bool isRealData = true){
      np[0] = 3;
      np[1] = 7;
      np[2] = 3;
      np[3] = 2;
      np[4] = 46;

      ni0[0] = 7;
      ni0[1] = 344;

      ni1[0] = 7;
      ni1[1] = 344;

      ni2[0] = 7;
      ni2[1] = 82;

      index = 0;

      unsigned int Nnp = np[0]*np[1]*np[2]*np[3]*np[4];
      unsigned int Nni0 = ni0[0]*ni0[1];
      unsigned int Nni1 = ni1[0]*ni1[1];
      unsigned int Nni2 = ni2[0]*ni2[1];

      std::string calibrationFile = "RecoHI/HiJetAlgos/data/ue_calibrations_data.txt";
      if(!isRealData) calibrationFile = "RecoHI/HiJetAlgos/data/ue_calibrations_mc.txt";
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

   float ue_predictor_pf[3][7][3][2][46], 
      ue_interpolation_pf0[7][344], 
      ue_interpolation_pf1[7][344], 
      ue_interpolation_pf2[7][82];
   
};


#endif



