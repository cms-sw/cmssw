#include "JetMETCorrections/JetPlusTrack/interface/SingleParticleJetResponseTmp.h"
#include <iostream>
using namespace std;

SingleParticleJetResponseTmp::SingleParticleJetResponseTmp() {}

vector<double> SingleParticleJetResponseTmp::response(double echar, double energycluster, int algo) const 
{
  vector<double> response;
  double recal=0,rhcal=0;
  
 // threshold on energy of track
  double e_thEC_hint;
  double e_thHC_hint;
  double e_thHC1_mip;
  double e_thHC2_mip;
  
  double e_thEC1;
  double e_thEC2;
  double e_thHC1;

  // parameters for Ecal responces with interaction in Ecal
  double PAR1_EC_hint;
  double PAR2_EC_hint;
  double PAR3_EC_hint;
  double PAR4_EC_hint;

  // parameters for Hcal responces with interaction in Ecal
  double PAR1_HC_hint;
  double PAR2_HC_hint;
  double PAR3_HC_hint;
  double PAR4_HC_hint;

  // parameters for Hcal responces without interaction in Ecal (MIP)
  double PAR1_HC_mip;
  double PAR2_HC_mip;
  double PAR3_HC_mip;
  double PAR4_HC_mip;
  double PAR5_HC_mip;

//===>  
  // parameters for Ecal responces with (without) interaction in Ecal
  double PAR1_EC;
  double PAR2_EC;
  double PAR3_EC;
  double PAR4_EC;

  // parameters for Hcal responces with (without) interaction in Ecal
  double PAR1_HC;
  double PAR2_HC;
  double PAR3_HC;
  double PAR4_HC;
//===>

  if (algo == 0) // e/pi  
    {
      if(energycluster>0.5) 
	{
	  double epiecal=1.6/(1+0.6*0.11*log(0.4*echar));
	  double epihcal=1.39/(1+0.39*0.11*log(0.6*echar));
	  recal=0.4*echar/epiecal;
	  rhcal=0.6*echar/epihcal;
	} 
      else 
	{
	  double epihcal=1.39/(1+0.39*0.11*log(echar));
	  rhcal=echar/epihcal;
	  recal=energycluster;
	} // interact or not 
    }
  
  else if (algo == 1) // ORCA_6 
    {
      // threshold on energy of track
      e_thHC_hint = 65.;
      e_thHC1_mip = 14.;
      e_thHC2_mip = 54.;
      
      // parameters for Ecal responses with interaction in Ecal
      PAR1_EC_hint = 3.2181;
      PAR2_EC_hint = 4.8399;
      
      // parameters for Hcal responses with interaction in Ecal
      PAR1_HC_hint = 0.2496;
      PAR2_HC_hint =-1.1673;
      PAR3_HC_hint = 0.7506;
      
      // parameters for Hcal responses without interaction in Ecal (MIP)
      PAR1_HC_mip  = 0.4108;
      PAR2_HC_mip  =-0.3474;
      PAR3_HC_mip  = 0.0293;
      PAR4_HC_mip  = 0.8388;
      PAR5_HC_mip  = 1.0520;
      
      if(energycluster>0.5) 
	{
	  double fecal_hint = PAR1_EC_hint/(sqrt(echar) + PAR2_EC_hint);
	  recal = echar*fecal_hint;
	  
	  if(echar <= e_thHC_hint) 
	    {
	      double fhcal_hint = PAR1_HC_hint * (log(echar) + PAR2_HC_hint);
	      rhcal=echar*fhcal_hint;
	    } 
	  else 
	    {
	      rhcal =echar* PAR3_HC_hint;
	    } // hcal response, interact=1
	  
	  // MIP, interact=0
	} 
      else 
	{
	  recal=energycluster;
	  if(echar <= e_thHC1_mip) 
	    {
	      double fhcal_mip = PAR1_HC_mip * (log(echar) + PAR2_HC_mip);
	      rhcal=echar*fhcal_mip;
	    } 
	  else if (echar > e_thHC1_mip && echar <= e_thHC2_mip) 
	    {
	      double fhcal_mip = PAR3_HC_mip * sqrt(echar) + PAR4_HC_mip;
	      rhcal=echar*fhcal_mip;
	    } 
	  else 
	    {
	      rhcal=echar*PAR5_HC_mip;
	    } // hcal response, MIP
	  
	} // interact or not
    } // algo = 1
  
  else if (algo == 2) // cmsim133
    {
      // threshold on energy of track
      e_thEC_hint = 0.1;
      e_thHC_hint = 1000.;
      e_thHC1_mip = 15.5;
      e_thHC2_mip = 58.;
      
      // parameters for Ecal responses with interaction in Ecal
      PAR1_EC_hint = -0.99228E-01;
      PAR2_EC_hint = -8.2298;
      PAR3_EC_hint = -0.82878E-03;
      PAR4_EC_hint = 0.41902;
      
      // parameters for Hcal responses with interaction in Ecal
      PAR1_HC_hint = 0.90693E-01;
      PAR2_HC_hint = 0.86861E-01;
      PAR3_HC_hint = 0.089524;
      PAR4_HC_hint = 0.65067;
      
      // parameters for Hcal responses without interaction in Ecal (MIP)
      PAR1_HC_mip  = 0.29538;
      PAR2_HC_mip  = -0.12289;
      PAR3_HC_mip  = 0.14940E-01;
      PAR4_HC_mip  = 0.73503;
      PAR5_HC_mip  = 0.84801;    

     
      if(energycluster>0.5) 
	{
	  
	  if(echar <= e_thEC_hint) 
	    {
	      double fecal_hint = PAR1_EC_hint * (echar + PAR2_EC_hint);
	      recal = echar*fecal_hint;
	    }
	  else 
	    {
	      double fecal_hint = PAR3_EC_hint * echar + PAR4_EC_hint;
	      recal = echar*fecal_hint;
	    }

	  if(echar <= e_thHC_hint) 
	    {
	      double fhcal_hint = PAR1_HC_hint * (log(echar) + PAR2_HC_hint);
	      rhcal=echar*fhcal_hint;
	    }
	  else 
	    {
	      double fhcal_hint = PAR3_HC_hint * (log(echar) + PAR4_HC_hint);
	      rhcal=echar*fhcal_hint;
	    } // hcal response, interact=1
	  
	  // MIP, interact=0
	}
      else 
	{
	  recal=energycluster;
	  if(echar <= e_thHC1_mip) 
	    {
	      double fhcal_mip = PAR1_HC_mip * (log(echar) + PAR2_HC_mip);
	      rhcal=echar*fhcal_mip;
	    } 
	  else if (echar > e_thHC1_mip && echar <= e_thHC2_mip) 
	    {
	      double fhcal_mip = PAR3_HC_mip * sqrt(echar) + PAR4_HC_mip;
	      rhcal=echar*fhcal_mip;
	    } 
	  else 
	    {
	      rhcal=echar*PAR5_HC_mip;
	    } // hcal response, MIP
      
	} // interact or not
    } // algo = 2

  else if (algo == 3) // OSCAR_3_6_0
    {
//       std::cout<<" SingleParticleJetResponse for OSCAR360 "<<energycluster<<std::endl;
      // threshold on energy of track
      e_thEC_hint = 1000.;
      e_thHC_hint = 0.5;
      e_thHC1_mip = 12.;
      e_thHC2_mip = 30.;

      // parameters for Ecal responses with interaction in Ecal
      PAR1_EC_hint = -0.54951E-03;
      PAR2_EC_hint = 0.42609;
      PAR3_EC_hint = -2.8831;
      PAR4_EC_hint = 0.33487;
      
      // parameters for Hcal responses with interaction in Ecal
      PAR1_HC_hint = -0.028222;
      PAR2_HC_hint = 0.43868;
      PAR3_HC_hint = 0.12144;
      PAR4_HC_hint = -0.63474;

      // parameters for Hcal responses without interaction in Ecal (MIP)
      PAR1_HC_mip  = 0.86455E-01;
      PAR2_HC_mip  = -0.36659;
      PAR3_HC_mip  = 0.60879E-01;
      PAR4_HC_mip  = 0.64219;
      PAR5_HC_mip  = 0.96800;
     
      if(energycluster>0.5) 
	{
	  if(echar <= e_thEC_hint) 
	    {
	      double fecal_hint = PAR1_EC_hint * (echar) + PAR2_EC_hint;
	      recal = echar*fecal_hint;
	    }
	  else 
	    {
	      double fecal_hint = PAR3_EC_hint/(-echar) + PAR4_EC_hint;
	      recal = echar*fecal_hint;
	    }

	  if(echar <= e_thHC_hint) 
	    {
	      double fhcal_hint = PAR1_HC_hint * echar + PAR2_HC_hint;
	      rhcal=echar*fhcal_hint;
	    }
	  else 
	    {
	      double fhcal_hint = PAR3_HC_hint * (log(echar) + PAR4_HC_hint);
	      rhcal=echar*fhcal_hint;
	    } // hcal response, interact=1
	  
//	  std::cout<<" Interacted in ECAL Ecal resp / Hcal resp "<<recal<<" "<<rhcal<<std::endl;
	  // MIP, interact=0
	}
      else 
	{
	  recal=energycluster;
	  if(echar <= e_thHC1_mip) 
	    {
	      double fhcal_mip = PAR1_HC_mip * (echar + PAR2_HC_mip);
	      rhcal=echar*fhcal_mip;
	    } 
	  else if (echar > e_thHC1_mip && echar <= e_thHC2_mip) 
	    {
	      double fhcal_mip = PAR3_HC_mip * sqrt(echar) + PAR4_HC_mip;
	      rhcal=echar*fhcal_mip;
	    } else 
	    {
	      rhcal=echar*PAR5_HC_mip;
	    } // hcal response, MIP
	  
//	  std::cout<<" Non-Interacted in ECAL Ecal resp / Hcal resp "<<recal<<" "<<rhcal<<std::endl;
	  
	} // interact or not
    } // algo = 3
    
//=> algo = 4, CMSSW_130_pre3

  else if (algo == 4) // CMSSW_130_pre3 (with Ecal Zero Suppression)
    {
      // threshold on energy of track
      
      e_thHC1_mip = 35.;
      
      // parameters for Ecal responses with interaction in Ecal
      PAR1_EC_hint = 22.976;
      PAR2_EC_hint = 55.913;
      
      // parameters for Hcal responses with interaction in Ecal
      PAR1_HC_hint = 0.20229E-01;
      PAR2_HC_hint = 0.30293;
      
      // parameters for Hcal responses without interaction in Ecal (MIP)
      PAR1_HC_mip  = 0.52371E-01;
      PAR2_HC_mip  = 0.66230;
      PAR3_HC_mip  = 0.96957;
      
      if(energycluster>0.7) 
	{
	  double fecal_hint = PAR1_EC_hint / (sqrt(echar) + PAR2_EC_hint);
	  recal = echar*fecal_hint;
	  
	  double fhcal_hint = PAR1_HC_hint * sqrt(echar) + PAR2_HC_hint;
	  rhcal=echar*fhcal_hint;
	} // ecal, hcal responses, interact=1;
	
        // MIP, interact=0 
      else 
	{
	  recal=energycluster;
	  if(echar <= e_thHC1_mip) 
	    {
	      double fhcal_mip = PAR1_HC_mip * sqrt(echar) + PAR2_HC_mip;
	      rhcal=echar*fhcal_mip;
	    }  
	  else 
	    {
	      rhcal=echar*PAR3_HC_mip;
	    } // hcal response, MIP
	  
	} // interact or not
    } // algo = 4
          
  else if (algo == 5) // CMSSW_130_pre3 (without Ecal Zero Suppression)
                      // without MIP definition
    {
      // threshold on energy of track
      
      e_thEC1 = 3.;
      e_thEC2 = 15.;
      e_thHC1 = 18.;
      
      // parameters for Ecal responses with (without) interaction in Ecal
      PAR1_EC = 0.12560;
      PAR2_EC = 2.0539;
      PAR3_EC = 5.1346;
      PAR4_EC = 0.22829;
      
      // parameters for Hcal responses with (without) interaction in Ecal
      PAR1_HC = 0.15370;
      PAR2_HC = 0.79737;
      PAR3_HC = 0.69524E-01;
      PAR4_HC = 5.2839;

// Ecal responses
          if(echar <= e_thEC1) 
	    {
	      double fecal = PAR1_EC * echar;
	      recal=echar*fecal;
	    } 
	   else if (echar > e_thEC1 && echar <= e_thEC2) 
	    {
	      double fecal = PAR2_EC / (sqrt(echar) + PAR3_EC);
	      recal=echar*fecal;
	    } 
	   else 
	    {
	      recal=echar*PAR4_EC;
	    } // ecal response
            
//Hcal responses
	  if(echar <= e_thHC1) 
	    {
	      double fhcal = PAR1_HC * (log(echar) + PAR2_HC);
	      rhcal=echar*fhcal;
	    }  
	  else 
	    {
	      double fhcal = PAR3_HC * (log(echar) + PAR4_HC);
	      rhcal=echar*fhcal;
	    } // hcal response
	  

    } // algo = 5
//  cout<<"CMSSW130 Algo = "<<algo<<" Ecal response= "<<recal <<" Hcal response= "<<rhcal<<" Energy of track= "<<echar<<endl;            
  response.push_back(recal);
  response.push_back(rhcal);
  return response;
}
