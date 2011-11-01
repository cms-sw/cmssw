
#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterEnergyCorrectionObjectSpecific.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "TMath.h"

// Shower leakage corrections developed by Jungzhie et al. using TB data
// Developed for EB only!
float EcalClusterEnergyCorrectionObjectSpecific::fEta(float energy, float eta, int algorithm) const
{

  //std::cout << "fEta function" << std::endl;

  // this correction is setup only for EB
  if ( algorithm != 0 ) return energy;
  
  float ieta = fabs(eta)*(5/0.087);
  float p0 = (params_->params())[0];  // should be 40.2198
  float p1 = (params_->params())[1];  // should be -3.03103e-6

  //std::cout << "ieta=" << ieta << std::endl;

  float correctedEnergy = energy;
  if ( ieta < p0 ) correctedEnergy = energy;
  else             correctedEnergy = energy/(1.0 + p1*(ieta-p0)*(ieta-p0));
  //std::cout << "ECEC fEta = " << correctedEnergy << std::endl;
  return correctedEnergy;
}

float EcalClusterEnergyCorrectionObjectSpecific::fBremEta(float sigmaPhiSigmaEta, float eta, int algorithm) const
{

  const float etaCrackMin = 1.44; 
  const float etaCrackMax = 1.56; 
   
  //STD 
  const int    nBinsEta              = 14; 
  float       leftEta  [nBinsEta]   = { 0.02, 0.25, 0.46, 0.81, 0.91, 1.01, 1.16,           etaCrackMax,  1.653,  1.8, 2.0, 2.2, 2.3, 2.4 };  
  float       rightEta [nBinsEta]   = { 0.25, 0.42, 0.77, 0.91, 1.01, 1.13, etaCrackMin,    1.653,        1.8  ,  2.0, 2.2, 2.3, 2.4, 2.5 };  
		
  float xcorr[nBinsEta];
  
  float par0[nBinsEta];
  float par1[nBinsEta];
  float par2[nBinsEta];
  float par3[nBinsEta];
  float par4[nBinsEta];

  float sigmaPhiSigmaEtaMin = 0.8;
  float sigmaPhiSigmaEtaMax = 5.;

  float sigmaPhiSigmaEtaFit = -1;

  // extra protections																					   
  // fix sigmaPhiSigmaEta boundaries 
  if (sigmaPhiSigmaEta < sigmaPhiSigmaEtaMin)  sigmaPhiSigmaEta = sigmaPhiSigmaEtaMin; 
  if (sigmaPhiSigmaEta > sigmaPhiSigmaEtaMax  )  sigmaPhiSigmaEta = sigmaPhiSigmaEtaMax; 

  // eta = 0																						   
  if (TMath::Abs(eta)  <  leftEta[0]            ) { eta = 0.02 ; }																   
  // outside acceptance																					   
  if (TMath::Abs(eta)  >=  rightEta[nBinsEta-1] ) { eta = 2.49; } //if (DBG) std::cout << " WARNING [applyScCorrections]: TMath::Abs(eta)  >=  rightEta[nBinsEta-1] " << std::endl;}  
  
  int tmpEta = -1;                                                                                                                                                                         
  for (int iEta = 0; iEta < nBinsEta; ++iEta){								              								      	   
    if ( leftEta[iEta] <= TMath::Abs(eta) && TMath::Abs(eta) <rightEta[iEta] ){				       									      	   
      tmpEta = iEta;											       										   
    }													       										   
  }	


  if (algorithm==0){ //Electrons

    
    xcorr[0]=1.00227;
    xcorr[1]=1.00252;
    xcorr[2]=1.00225;
    xcorr[3]=1.00159;
    xcorr[4]=0.999475;
    xcorr[5]=0.997203;
    xcorr[6]=0.993886;
    xcorr[7]=0.971262;
    xcorr[8]=0.975922;
    xcorr[9]=0.979087;
    xcorr[10]=0.98495;
    xcorr[11]=0.98781;
    xcorr[12]=0.989546;
    xcorr[13]=0.989638;

    par0[0] = 1.00718;
    par1[0] = -0.00187886;
    par2[0] = 0 ;

    par0[1] = 1.00713;
    par1[1] = -0.00227574;
    par2[1] = 0 ;

    par0[2] = 1.00641;
    par1[2] = -0.00259935;
    par2[2] = 0 ;

    par0[3] = 1.00761;
    par1[3] = -0.00433692;
    par2[3] = 0 ;

    par0[4] = 1.00682;
    par1[4] = -0.00551324;
    par2[4] = 0 ;

    par0[5] = 1.0073;
    par1[5] = -0.00799669;
    par2[5] = 0 ;

    par0[6] = 1.00462;
    par1[6] = -0.00870057;
    par2[6] = 0 ;

    par0[7] = 0.972798;
    par1[7] = -0.000771577;
    par2[7] = -0.00276696;

    par0[8] = 0.981672;
    par1[8] = -0.00202028;
    par2[8] = -0.00471028;

    par0[9] = 0.98251;
    par1[9] = 0.00441308;
    par2[9] = -0.00809139;

    par0[10] = 0.986123;
    par1[10] = 0.00832913;
    par2[10] = -0.00944584;

    par0[11] = 0.990124;
    par1[11] = 0.00742879;
    par2[11] = -0.00960462;

    par0[12] = 0.990187;
    par1[12] = 0.0094608;
    par2[12] = -0.010172;

    par0[13] = 0.99372;
    par1[13] = 0.00560406;
    par2[13] = -0.00943169;

    sigmaPhiSigmaEtaFit = 1.2;

  }

  if (algorithm==1){ //Photons


    xcorr[0]=1.00506;
    xcorr[1]=1.00697;
    xcorr[2]=1.00595;
    xcorr[3]=1.00595;
    xcorr[4]=1.00595;
    xcorr[5]=1.00595;
    xcorr[6]=1.00595;
    xcorr[7]=0.966651;
    xcorr[8]=0.97381;
    xcorr[9]=0.976516;
    xcorr[10]=0.983254;
    xcorr[11]=0.98502;
    xcorr[12]=0.98502;
    xcorr[13]=0.978472;

    par0[0] = 0.00132382 ;
    par1[0] = 2.17664 ;
    par2[0] = -0.00467206 ;
    par3[0] = 0.988994 ;
    par4[0] = 17.5858 ;

    par0[1] = -0.00590257 ;
    par1[1] = 1.90733 ;
    par2[1] = 0.000684327 ;
    par3[1] = 0.986431 ;
    par4[1] = 16.6698 ;

    par0[2] = 0.00265109 ;
    par1[2] = 1.73272 ;
    par2[2] = -0.00107022 ;
    par3[2] = 0.989322 ;
    par4[2] = 15.4911 ;

    par0[3] = 0.00231631 ;
    par1[3] = 1.3463 ;
    par2[3] = -0.00369555 ;
    par3[3] = 0.987133 ;
    par4[3] = 10.9233 ;
    
    par0[4] = 0.00984253 ;
    par1[4] = 1.33889 ;
    par2[4] = -0.00392593 ;
    par3[4] = 0.979191 ;
    par4[4] = 9.35276 ;

    par0[5] = 0.023683 ;
    par1[5] = 1.31198 ;
    par2[5] = -0.00947317 ;
    par3[5] = 0.963352 ;
    par4[5] = 7.5597 ;
    
    par0[6] = 0.0851133 ;
    par1[6] = 1.38097 ;
    par2[6] = -0.0340201 ;
    par3[6] = 0.969502 ;
    par4[6] = 4.17983 ;

    par0[7] = 6.71705 ;
    par1[7] = 5034.26 ;
    par2[7] = -2.68669 ;
    par3[7] = 0.970174 ;
    par4[7] = 1.00288 ;

    par0[8] = 1306.82 ;
    par1[8] = 472004 ;
    par2[8] = -1.86145 ;
    par3[8] = 0.981714 ;
    par4[8] = -0.25644 ;
    
    par0[9] = 0.317121 ;
    par1[9] = 3.22717 ;
    par2[9] = -0.126848 ;
    par3[9] = 0.957792 ;
    par4[9] = 2.01028 ;

    par0[10] = 0.275225 ;
    par1[10] = 2.20686 ;
    par2[10] = -0.11009 ;
    par3[10] = 0.93922 ;
    par4[10] = 2.69958 ;

    par0[11] = 0.0639875 ;
    par1[11] = 1.40045 ;
    par2[11] = -0.0255853 ;
    par3[11] = 0.821566 ;
    par4[11] = 7.3297 ;

    par0[12] = 0.030488 ;
    par1[12] = 1.37842 ;
    par2[12] = -0.0121879 ;
    par3[12] = 0.8173 ;
    par4[12] = 9.29944 ;

    par0[13] = 0.213906 ;
    par1[13] = 1.67471 ;
    par2[13] = -0.0860589 ;
    par3[13] = 0.893636 ;
    par4[13] = 3.78218 ;
  
    sigmaPhiSigmaEtaFit = 1.;

  }

  
												       										           
  
  // Interpolation																					         
  float tmpInter = 1;																				         
  // In eta cracks/gaps 																				         
  if (tmpEta == -1 ) { // need to interpolate    
    for (int iEta = 0; iEta < nBinsEta-1; ++iEta){								       								         
      if (rightEta[iEta] <= TMath::Abs(eta) && TMath::Abs(eta) <leftEta[iEta+1] ){													         
	if (sigmaPhiSigmaEta >= sigmaPhiSigmaEtaFit)  {
	  if (algorithm==0){ //electron
	    tmpInter = ( par0[iEta] + sigmaPhiSigmaEta*par1[iEta] + sigmaPhiSigmaEta*sigmaPhiSigmaEta*par2[iEta] +  
			 par0[iEta+1] + sigmaPhiSigmaEta*par1[iEta+1] + sigmaPhiSigmaEta*sigmaPhiSigmaEta*par2[iEta+1]) / 2. ; 
	  }
	  if (algorithm==1){ //photon
	    tmpInter =   (par0[iEta  ]*(1.-exp(-(sigmaPhiSigmaEta-par4[iEta  ])/par1[iEta  ]))*par2[iEta  ]*sigmaPhiSigmaEta + par3[iEta  ]+
			  par0[iEta+1]*(1.-exp(-(sigmaPhiSigmaEta-par4[iEta+1])/par1[iEta+1]))*par2[iEta+1]*sigmaPhiSigmaEta + par3[iEta+1] ) /2.;
	  }
	}
	else tmpInter = (xcorr[iEta] + xcorr[iEta+1])/2.; 
      }																						         
    }																						         
    return tmpInter;																					         
  }  		
																					         
  if (sigmaPhiSigmaEta >= sigmaPhiSigmaEtaFit) {
    if (algorithm==0) return par0[tmpEta] + sigmaPhiSigmaEta*par1[tmpEta] + sigmaPhiSigmaEta*sigmaPhiSigmaEta*par2[tmpEta]; 
    if (algorithm==1) return par0[tmpEta  ]*(1.-exp(-(sigmaPhiSigmaEta-par4[tmpEta  ])/par1[tmpEta  ]))*par2[tmpEta  ]*sigmaPhiSigmaEta + par3[tmpEta  ];
  }
  else return xcorr[tmpEta]; 



  return 1.;
}

float EcalClusterEnergyCorrectionObjectSpecific::fEt(float ET, int algorithm) const
{

  float par0 =  -1; 
  float par1 =  -1; 
  float par2 =  -1;	   
  float par3 =  -1;	   
  float par4 =  -1;     
  float par5 =  -1;     
  float par6 =  -1;  

  if (algorithm==0){ //Electrons EB

    par0 = 0.97213; 
    par1 = 0.999528; 
    par2 = 5.61192e-06; 
    par3 = 0.0143269; 
    par4 = -17.1776; 

    if (ET > 200) ET =200;   		  
    if (             ET <    5 ) return         1.;  
    if (  5 <= ET && ET <   10 ) return         par0 ;  
    if ( 10 <= ET && ET <= 200 ) return         (par1  + ET*par2)*(1- par3*exp(ET/ par4));

  }


  if (algorithm==1){ //Electrons EE

    par0 = 0.930081; 
    par1 = 0.996683; 
    par2 = 3.54079e-05; 
    par3 = 0.0460187; 
    par4 = -23.2461; 
    
    if (ET > 200) ET =200;   		  
    if (             ET <    5 ) return         1.;  
    if (  5 <= ET && ET <   10 ) return         par0;  
    if ( 10 <= ET && ET <= 200 ) return         ( par1  + ET*par2)*(1-par3*exp(ET/par4));

  }

  

  if (algorithm==2){ //Photons EB

    par0 =  1; 
    par1 =  1.00348; 
    par2 =  1.001;	   
    par3 =  -9.17302e-06;	   
    par4 =  0.999688;     

    if (             ET <   5 ) return         1.;  
    if (  5 <= ET && ET <  10 ) return         par0 ;  
    if ( 10 <= ET && ET <  20 ) return         par1 ;  
    if ( 20 <= ET && ET < 140 ) return         par2 + par3*ET ;  
    if (140 <= ET             ) return         par4;  

  }


  if (algorithm==3){ //Photons EE
		  
    par0 =  1; 
    par1 =  0.996931; 
    par2 =  0.999497;	   
    par3 =  0.992617;	   
    par4 =  7.52128e-05;     
    par5 =  -1.2845e-07;     
    par6 =  1.00231;     

    if (             ET <   5 ) return         1.;  
    if (  5 <= ET && ET <  10 ) return          par0 ;  
    if ( 10 <= ET && ET <  20 ) return          par1 ;  
    if ( 20 <= ET && ET <  30 ) return          par2 ;  
    if ( 30 <= ET && ET < 200 ) return          par3 + par4 *ET + par5 *ET*ET ;  
    if ( 200 <= ET            ) return          par6 ;   	

  }


  return 1.;
}


float EcalClusterEnergyCorrectionObjectSpecific::fEnergy(float E, int algorithm) const
{

  float par0 = -1;               
  float par1 = -1; 
  float par2 = -1; 
  float par3 = -1; 
  float par4 = -1; 

  if (algorithm==0){ //Electrons EB
    return 1.;
  }


  if (algorithm==1){ //Electrons EE
			 	  
    par0 = 400;               
    par1 = 0.982475; 
    par2 = 4.95413e-05; 
    par3 = 0.16886; 
    par4 = -30.1517; 
   				 	  
    if (E > par0) E =par0;   		  
    if (             E <   0 ) return         1.;  
    if (  0 <= E && E <= par0 ) return         (par1 + E*par2 )*(1- par3*exp(E/par4 ));
 		
  }


  if (algorithm==2){ //Photons EB
    return 1.;
  }


  if (algorithm==3){ //Photons EE
			 	  
    par0 = 850;               
    par1 = 0.994169 ;	  
    par2 = 1.28629e-05 ;     
  				 	  
    if (E  > par0 ) E = par0 ;   		  
    if (            E <   0     ) return      1.;  
    if (  0 <= E && E <=  par0  ) return      par1 + E*par2; 

  }

  return 1.;
}



float EcalClusterEnergyCorrectionObjectSpecific::getValue( const reco::SuperCluster & superCluster, const int mode ) const
{

  float corr = 1.;
  float corr2 = 1.;
  float energy = 0;

  int subdet = superCluster.seed()->hitsAndFractions()[0].first.subdetId();
  //std::cout << "subdet="<< subdet<< std::endl;

  //std::cout << "rawEnergy=" << superCluster.rawEnergy() << " SCeta=" << superCluster.eta() << std::endl;
  
  if (subdet==EcalBarrel){
    float cetacorr = fEta(superCluster.rawEnergy(), superCluster.eta(), 0)/superCluster.rawEnergy();
    //std::cout << "cetacorr=" <<cetacorr<< std::endl;

    energy = superCluster.rawEnergy()*cetacorr; //previously in CMSSW
    //energy = superCluster.rawEnergy()*fEta(e5x5, superCluster.seed()->eta(), 0)/e5x5;
  }
  else if (subdet==EcalEndcap){
    energy = superCluster.rawEnergy()+superCluster.preshowerEnergy();
  }

  float newEnergy = energy;

  if (mode==0){ //Electron

    corr = fBremEta(superCluster.phiWidth()/superCluster.etaWidth(), superCluster.eta(), 0);

    float et = energy*TMath::Sin(2*TMath::ATan(TMath::Exp(-superCluster.eta())))/corr;

    if (subdet==EcalBarrel) corr2 = corr * fEt(et, 0);
    if (subdet==EcalEndcap) corr2 = corr * fEnergy(et, 1);

    newEnergy = energy/corr2; 

  }

  if (mode==1){ //low R9 Photons

    corr = fBremEta(superCluster.phiWidth()/superCluster.etaWidth(), superCluster.eta(), 1);

    float et = energy*TMath::Sin(2*TMath::ATan(TMath::Exp(-superCluster.eta())))/corr;

    if (subdet==EcalBarrel) corr2 = corr * fEt(et, 2);
    if (subdet==EcalEndcap) corr2 = corr * fEnergy(et, 3);

    newEnergy = energy/corr2; 

  }



  return newEnergy;
}

/*
float EcalClusterEnergyCorrectionObjectSpecific::getValue( const reco::GsfElectron & electron, const int mode) const
{

  return getValue(*(electron.superCluster()), 0);
}
*/
 /*
float EcalClusterEnergyCorrectionObjectSpecific::getValue( const reco::Photon & photon, const int mode) const
{

  float corr = 1.;
  float corr2 = 1.;
  float energy = 0;

  int subdet = photon.superCluster()->seed()->hitsAndFractions()[0].first.subdetId();

  float cetacorr = fEta(photon.superCluster()->rawEnergy(), photon.superCluster()->eta(), 0)/photon.superCluster()->rawEnergy();

  if (subdet==EcalBarrel){
    energy = photon.superCluster()->rawEnergy()*cetacorr; //previously in CMSSW
    //energy = superCluster.rawEnergy()*fEta(e5x5, superCluster.seed()->eta(), 0)/e5x5;
  }
  else if (subdet==EcalEndcap){
    energy = photon.superCluster()->rawEnergy()+photon.superCluster()->preshowerEnergy();
  }

  float r9 = photon.r9();
  float e5x5 = photon.e5x5();

    //float e3x3    =   EcalClusterTools::e3x3(  *(superCluster.seed()), &(*hits), &(*topology)); 
    //float e5x5    =   EcalClusterTools::e5x5( *(superCluster.seed()), &(*hits), &(*topology)); 
    //float r9 = e3x3/(superCluster.rawEnergy());

  float R9min;
  if (subdet==EcalBarrel) R9min=0.94;
  if (subdet==EcalEndcap) R9min=0.95;
    
  float newEnergy = energy;

  if (r9<R9min){

    corr = fBremEta(photon.superCluster()->phiWidth()/photon.superCluster()->etaWidth(), photon.superCluster()->eta(), 1);

    float et = energy*TMath::Sin(2*TMath::ATan(TMath::Exp(-photon.superCluster()->eta())))/corr;
    
    if (subdet==EcalBarrel) corr2 = corr * fEt(et, 2);
    if (subdet==EcalEndcap) corr2 = corr * fEnergy(et, 3);
    
    newEnergy = energy/corr2;
    
  }
  if (r9>R9min){
    
    //if uncorrected
    //if (subdet==EcalBarrel) newEnergy = e5x5*cetacorr;
    //if (subdet==EcalEndcap) newEnergy = e5x5 + superCluster.preshowerEnergy();
    
    //already corrected
    newEnergy = e5x5;
    
  }

  return newEnergy;

}
 */


#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
DEFINE_EDM_PLUGIN( EcalClusterFunctionFactory, EcalClusterEnergyCorrectionObjectSpecific, "EcalClusterEnergyCorrectionObjectSpecific");
