#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterEnergyUncertainty.h"


float EcalClusterEnergyUncertainty::getValue( const reco::SuperCluster & superCluster, const int mode ) const
{
        checkInit();
	// mode  = -1 returns negative energy uncertainty 
	//       = +1 returns positive energy uncertainty 
	//       =  0 (default) returns overall  energy uncertainty 
	float en = superCluster.energy(); 
	float eta = fabs(superCluster.eta()); 
	float et = en/cosh(eta); 
	float brem = superCluster.phiWidth()/superCluster.etaWidth(); 
	
	int offset = 0; 
	
	if ( superCluster.algoID() == reco::hybrid ) offset = 0; 
	else if ( superCluster.algoID() == reco::multi5x5 ) offset = 36; 
	else { 
	  // not supported now 
	  //std::cout << "Not supported value " << superCluster.algoID() << std::endl;
	  //std::cout << "eta = " << superCluster.eta() << std::endl;
	  //std::cout << "phi = " << superCluster.phi() << std::endl;
	  //std::cout << "En  = " << superCluster.energy() << std::endl;
	  return -1; 
	}
	if ( mode ==  0 ) offset += 0; // total effective uncertainty 
	else if ( mode == -1 ) offset += 12;  // negative energy uncertainty 
	else if ( mode ==  1 ) offset += 24;  // positive energy uncertainty 
	else {
	  // wrong input
	  return 0;
	}

	float br0_0 = (*params_)[offset + 0]; 
	float br0_1 = (*params_)[offset + 1]; 
	float br0_2 = (*params_)[offset + 2]; 
	float br0_3 = (*params_)[offset + 3]; 
	
	float br1_0 = (*params_)[offset + 4];  
	float br1_1 = (*params_)[offset + 5];  
	float br1_2 = (*params_)[offset + 6];  
	float br1_3 = (*params_)[offset + 7];  
	
	float br2_0 = (*params_)[offset + 8];  
	float br2_1 = (*params_)[offset + 9];  
	float br2_2 = (*params_)[offset + 10];  
	float br2_3 = (*params_)[offset + 11];  
	
	float p0 = (br0_0 + br0_1*brem) + (br0_2 + br0_3*brem)/et;  
	float p1 = (br1_0 + br1_1*brem) + (br1_2 + br1_3*brem)/et;  
	float p2 = (br2_0 + br2_1*brem) + (br2_2 + br2_3*brem)/et;  
	//std::cout << "====================================================" << std::endl;
	//std::cout << "et = " << et << "\t eta = " << eta << std::endl;
	//std::cout << "p0 = " << p0 << "\t p1 = " << p1 << "\t p2 = " << p2 << std::endl;
	float uncertainty = en*(p0 + p1*fabs(eta) + p2*eta*eta);  
	//std::cout << uncertainty << std::endl;
	//std::cout << std::endl;
	return uncertainty;  


}

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
DEFINE_EDM_PLUGIN( EcalClusterFunctionFactory, EcalClusterEnergyUncertainty, "EcalClusterEnergyUncertainty");
