#ifndef CMGTools_H2TauTau_TriggerEfficiency_H
#define CMGTools_H2TauTau_TriggerEfficiency_H

#include <math.h> 
#include "TMath.h" 
#include <limits>
#include <TH1.h> 


class TriggerEfficiency {
public:
  TriggerEfficiency(){} ;
  
  ////Method for fitting turn-on curves
  bool fitEfficiency(const char* filename,float xmin=0.,float xmax=0.);
  double operator()(const double *xx ) const;
  TH1* chi2FunctorHisto;  
  float xmin_; float xmax_;


  //trigger lumi weighting done according to this table from:
  //https://twiki.cern.ch/twiki/bin/viewauth/CMS/HToTauTauPlusTwoJets
  // HLT_IsoMu12_LooseIsoPFTau10_v4 	        163269 - 163869 	168.6 	L1_SingleMu10 	 
  // HLT_IsoMu15_LooseIsoPFTau15_v2 	        165088 - 165633 	139.0 	L1_SingleMu10 	 
  // HLT_IsoMu15_LooseIsoPFTau15_v4 	        165970 - 167043 	545.1 	L1_SingleMu10 	w/o run 166346
  // HLT_IsoMu15_LooseIsoPFTau15_v5 	        166346          	4.3 	L1_SingleMu10 	one run only
  // HLT_IsoMu15_LooseIsoPFTau15_v6 	        167078 - 167913 	245.6 	L1_SingleMu10 	 
  // HLT_IsoMu15_LooseIsoPFTau15_v8 	        170249 - 173198 	785.7 	L1_SingleMu10 	 
  // HLT_IsoMu15_LooseIsoPFTau15_v9 	        173236 - 178380 	1945 	L1_SingleMu10 	 (Note! this trigger got prescaled in 2011B)    6.95  ?? 
  // and in 2011A, it seems off for the first part of the spill..
  // HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v1 	173236 - 178380 	1945 	L1_SingleMu14_Eta2p1 	ET(tau)>20 GeV, |eta(mu)|<2.1   1.692
  // HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v5 	178420 - 179889 	706.7 	L1_SingleMu14_Eta2p1 	ET(tau)>20 GeV, |eta(mu)|<2.1   0.695
  // HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v6 	179959 - 180252 	120.7 	L1_SingleMu14_Eta2p1 	end of 2011 run                 0.118

  
  //Below, measurement from Colin (we're now using these numbers to weight our efficiencies)
 
  //                                            2011A                        2011B
  // HLT_IsoMu12_LooseIsoPFTau10_v4 	        168.6
  // HLT_IsoMu15_LooseIsoPFTau15_v2 	        139.1
  // HLT_IsoMu15_LooseIsoPFTau15_v4 	        543.3
  // HLT_IsoMu15_LooseIsoPFTau15_v5 	          4.3
  // HLT_IsoMu15_LooseIsoPFTau15_v6 	        243.1
  // HLT_IsoMu15_LooseIsoPFTau15_v8 	        780.4 (368.04 + 412.36) 
  // HLT_IsoMu15_LooseIsoPFTau15_v9 	        246.527                      
  // HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v1                                   1698           
  // HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v5                                    694.8     
  // HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v6                                    117.6     


  double effTau2011A(double pt, double eta){
    float tau10w = 168.6;
    float tau15w = 139.1 + 543.3 + 4.3 + 243.1 + 780.4 + 246.527;
    return ( tau10w * effLooseTau10(pt,eta) + 
	     tau15w * effLooseTau15(pt,eta) ) / ( tau10w + tau15w);
      
    /*     return ((168.6)*effLooseTau10(pt,eta) */
    /* 	    //COLIN I think the 246.5 should be multiplied by effLooseTau20? or maybe not...  */
    /* 	    +(139.0+545.1+4.3+245.6+785.7+246.5)*effLooseTau15(pt,eta))/(168.6+139.0+545.1+4.3+245.6+785.7+246.527); */
    //last number 246.5 obtained from runs 172620-->173692 in Oct3ReReco(=PromtReco-v6)
  }
  

  double effTau2011B(double pt, double eta){
    return effLooseTau20(pt,eta);

      /*     return ((0.1)*effLooseTau15(pt,eta) */
      /* 	    +((1945-246.5-0.1)+706.7+120.7)*effLooseTau20(pt,eta))/(0.1+1945-246.5-0.1+706.7+120.7); */
      /*     //first number 0.1 is an approximation because this trigger got prescaled and most of data is actually with LooseTau20 */
  }

  double effTau2011AB(double pt, double eta){
    float tau10w = 168.6;
    float tau15w = 139.1 + 543.3 + 4.3 + 243.1 + 780.4 + 246.527;
    float tau20w = 1698 + 694.8 + 117.6;
    return ( tau10w * effLooseTau10(pt,eta) + 
	     tau15w * effLooseTau15(pt,eta) +
	     tau20w * effLooseTau20(pt,eta) ) / ( tau10w + tau15w + tau20w );

    /*     return ((168.6)*effLooseTau10(pt,eta) */
    /* 	    +(139.0+545.1+4.3+245.6+785.7+246.5+0.1)*effLooseTau15(pt,eta) */
    /* 	    +((1945-246.5-0.1)+706.7+120.7)*effLooseTau20(pt,eta))/(168.6+139.0+545.1+4.3+245.6+785.7+246.5+0.1+(1945-246.5-0.1)+706.7+120.7); */
  }
  
  double effMu2011A(double pt, double eta){
    float mu12w = 168.6;
    float mu15w = 139.1 + 543.3 + 4.3 + 243.1 + 780.4 + 246.527;
    return ( mu12w * effIsoMu12(pt,eta) + 
	     mu15w * effIsoMu15(pt,eta) ) / ( mu12w + mu15w);

/*     return ((168.6)*effIsoMu12(pt,eta) */
/* 	    +(139.0+545.1+4.3+245.6+785.7+246.527)*effIsoMu15(pt,eta))/(168.6+139.0+545.1+4.3+245.6+785.7+246.527); */
  }

  double effMu2011B(double pt, double eta){
    return effIsoMu15eta2p1(pt,eta);
/*     return ((1698)*effIsoMu15(pt,eta) */
/* 	    +(706.7+120.7)*effIsoMu15eta2p1(pt,eta))/(1698+706.7+120.7); */
  }

  double effMu2011AB(double pt, double eta){
    float mu12w = 168.6;
    float mu15w = 139.1 + 543.3 + 4.3 + 243.1 + 780.4 + 246.527;
    float mu15eta2p1 = 1698 + 694.8 + 117.6;
    return ( mu12w * effIsoMu12(pt,eta) + 
	     mu15w * effIsoMu15(pt,eta) + 
	     mu15eta2p1 * effIsoMu15eta2p1(pt,eta) ) / ( mu12w + mu15w + mu15eta2p1);

/*     return ((168.6)*effIsoMu12(pt,eta) */
/* 	    +(139.0+545.1+4.3+245.6+785.7+1945.)*effIsoMu15(pt,eta) */
/* 	    +(706.7+120.7)*effIsoMu15eta2p1(pt,eta))/(168.6+139.0+545.1+4.3+245.6+785.7+1945+706.7+120.7); */
  }

  /* HLT_Ele15_ _LooseIsoPFTau15_v1 	160404 - 161176 	6.7 	L1_SingleEG12 	buggy, lower eff */
  /* HLT_Ele15_ _LooseIsoPFTau15_v2 	161216 - 163261 	40.9 	L1_SingleEG12 	buggy, lower eff */
  /* HLT_Ele15_ _LooseIsoPFTau15_v4 	163269 - 163869 	168.6 	L1_SingleEG12 	  */
  /* HLT_Ele15_ _LooseIsoPFTau20_v6 	165088 - 165633 	139.0 	L1_SingleEG12 	  */
  /* HLT_Ele15_ _LooseIsoPFTau20_v8 	165970 - 166967 	526.7 	L1_SingleEG12 	  */
  /* HLT_Ele15_ _LooseIsoPFTau20_v9 	167039 - 167913 	268.3 	L1_SingleEG12 	  */
  /* HLT_Ele15_ _TightIsoPFTau20_v2 	170249 - 173198 	785.7 	L1_SingleEG12 	tight iso */
  /* HLT_Ele18_ _MediumIsoPFTau20_v1 	173236 - 178380 	1945 	L1_SingleEG15 	medium iso, ET(e)>18 GeV */
  /* HLT_Ele20_ _MediumIsoPFTau20_v5 	178420 - 179889 	706.7 	L1_SingleEG18 OR EG20 	medium iso, ET(e)>20 GeV */
  /* HLT_Ele20_ _MediumIsoPFTau20_v6 	179959 - 180252 	120.7 	L1_SingleEG18 OR EG20 	end of 2011 run  */

  //                                       2011A                                       2011B
  // HLT_Ele15_ _LooseIsoPFTau15_v1        not used - bugged	
  // HLT_Ele15_ _LooseIsoPFTau15_v2 	   not used - bugged
  // HLT_Ele15_ _LooseIsoPFTau15_v4 	   168.6    (May10ReReco)
  // HLT_Ele15_ _LooseIsoPFTau20_v6 	   139.1    (V4)
  // HLT_Ele15_ _LooseIsoPFTau20_v8 	   524.9    (V4)
  // HLT_Ele15_ _LooseIsoPFTau20_v9 	   265.7    (V4)
  // HLT_Ele15_ _TightIsoPFTau20_v2 	   368.04   (Aug5) + 412.36 (Oct3) = 780.4
  // HLT_Ele18_ _MediumIsoPFTau20_v1 	   246.527                                    1698
  // HLT_Ele20_ _MediumIsoPFTau20_v5 	                                               694.8
  // HLT_Ele20_ _MediumIsoPFTau20_v6 	                                               117.6


  double effEle2011A( double pt, double eta) {
    double ele15Weight = 168.6 + 139.1 + 524.9 + 265.7 + 780.4;
    double ele18Weight = 246.5; // warning overlap!
    return (ele15Weight * effEle15(pt, eta) + 
	    ele18Weight * effEle18(pt,eta)) / (ele15Weight+ele18Weight);
  }

  double effEle2011B( double pt, double eta) {
    double ele18Weight = 1698; 
    double ele20Weight = 694.8 + 117.6;
    return ( ele18Weight * effEle18(pt,eta) + 
	     ele20Weight * effEle20(pt,eta)) / (ele18Weight + ele20Weight);
  } 

  double effEle2011AB( double pt, double eta) {
    double ele15Weight = 168.6 + 139.1 + 524.9 + 265.7 + 780.4;
    double ele18Weight = 246.5 + 1698; 
    double ele20Weight = 694.8 + 117.6; 
    return ( ele15Weight * effEle15(pt, eta) + 
	     ele18Weight * effEle18(pt,eta) + 
	     ele20Weight * effEle20(pt,eta)) / (ele15Weight + ele18Weight + ele20Weight);    
  }

  double effTau2011A_TauEle( double pt, double eta) {
    double tau15Weight = 168.6;
    double tau20Weight = 139.1 + 524.9 + 265.7;
    double tightIsoTau20Weight = 780.4;
    double mediumIsoTau20Weight = 246.5; 

    return (tau15Weight * effLooseTau15(pt, eta) + 
	    tau20Weight * effLooseTau20_TauEle(pt,eta) + 
	    tightIsoTau20Weight * effTightIsoTau20(pt,eta) + 
	    mediumIsoTau20Weight * effMediumIsoTau20(pt,eta)) 
      / ( tau15Weight + tau20Weight + tightIsoTau20Weight + mediumIsoTau20Weight);
  }

  double effTau2011B_TauEle( double pt, double eta) {
    return effMediumIsoTau20(pt,eta);
  }

  double effTau2011AB_TauEle( double pt, double eta) {
    double tau20Weight = 168.6 + 139.1 + 524.9 + 265.7;
    double tightIsoTau20Weight = 780.4;
    double mediumIsoTau20Weight = 246.5 + 1698 + 694.8 + 117.6;

	  return (tau20Weight * effLooseTau20_TauEle(pt,eta) + 
	    tightIsoTau20Weight * effTightIsoTau20(pt,eta) + 
	    mediumIsoTau20Weight * effMediumIsoTau20(pt,eta)) 
      / ( tau20Weight + tightIsoTau20Weight + mediumIsoTau20Weight);
  }


  //****************
  //parameters taken from AN-11-390 v8
  //*****************
  double effLooseTau10(double pt, double eta){
    if(fabs(eta)<1.5) 
      return efficiency(pt,13.6046,1.66291,1.71551,141.929,0.910686);
    else 
      return efficiency(pt,-0.392211,7.90467,5.48228,134.599,0.925858);
  }
  double effLooseTau15(double pt, double eta){
    if(fabs(eta)<1.5) 
      return efficiency(pt,13.9694,0.084835,0.057743,1.50674,0.984976);
    else 
      return efficiency(pt,14.435,1.34952,2.43996,1.03631,1.79081);
  }
  double effLooseTau20(double pt, double eta){
    if(fabs(eta)<1.5) 
      return efficiency(pt,19.2102,1.26519,2.48994,1.04699,1.3492);
    else
      return efficiency(pt,19.2438,1.37298,1.76448,1.73935,0.901291);
  }
  double effLooseTau20_TauEle(double pt, double eta){
    if(fabs(eta)<1.5) 
      return efficiency(pt,19.3916,0.996964,1.70131,1.38002,0.903245);
    else
      return efficiency(pt,18.8166,0.526632,0.20666,6.80392,0.903245);
  }

  double effLooseTau15MC(double pt, double eta){//should correspond to Fall11 MC
    if(fabs(eta)<1.5) 
      return efficiency(pt,14.4601,0.0485272,0.03849,1.48324,0.965257);
    else 
      return efficiency(pt,14.4451,0.0790573,0.0732472,1.47046,0.942028);
  }


  //PG this is ok for TauEle
  double effTightIsoTau20(double pt, double eta){
    if(fabs(eta)<1.5) 
      return efficiency(pt, 19.6013, 0.987317, 1.08015, 1.88592, 0.776894);
    else 
      return efficiency(pt, 18.8859, 0.271301, 0.128008, 1.50993, 0.825122);
  }

  //PG this is ok for TauEle
  double effMediumIsoTau20(double pt, double eta){
    if(fabs(eta)<1.5) 
      return efficiency(pt, 19.5667, 1.15203, 1.68126, 1.40025, 0.848033);
    else 
      return efficiency(pt, 18.8476, 0.528963, 0.16717, 3.65814, 0.749759);
  }

  //PG this is ok for TauEle
  //Jose: should we add a _TauEle tag to this function? 
  double effMediumIsoTau20MC(double pt, double eta){
    if(fabs(eta)<1.5) 
      return efficiency(pt, 19.468, 0.0615381, 0.0349325, 1.59349, 0.860096);
    else 
      return efficiency(pt, 19.3862, 0.247148, 0.123187, 2.87108,	0.790894);
  }


  //****************
  //parameters taken from AN-11-390 v8
  //*****************

  //COLIN: putting the junction at 1.4 as discussed with Josh
  double effIsoMu12(double pt, double eta){
    if(fabs(eta)<0.8)
      return 0.920; //Barrel
    else if (fabs(eta) < 1.2)
      return 0.868;
    else 
      return 0.845; //EndCap
  }

  double effIsoMu15(double pt, double eta){
    if(fabs(eta)<0.8)
      return 0.917;
    else if (fabs(eta) < 1.2)
      return 0.871;
    else 
      return 0.864;
  }

  double effIsoMu15eta2p1(double pt, double eta){
    if(fabs(eta)<0.8)
      return efficiency(pt,15.9877,2.90938e-07,2.63922e-11,5.81194,0.906943);
    else if(fabs(eta)<1.2)
      return efficiency(pt,15.9995,1.35931e-07,7.88264e-11,4.60253,0.855461);
    else 
      return efficiency(pt,15.9084,2.27242e-12,8.77174e-14,1.00241,12.9909);
  }

  double effIsoMu15MC(double pt, double eta){//should correspond to Fall11 MC
    if(fabs(eta)<0.8)
      return 0.923;
    else if (fabs(eta)<1.2)
      return 0.879;
    else
      return 0.839;
  }


  //****************
  //parameters taken from AN-11-390 v8
  //*****************

  double effEle15(double pt, double eta) {
    if(fabs(eta)<1.479) 
      return efficiency(pt, 14.8772, 0.311255, 0.221021, 1.87734, 0.986665);
    else
      return efficiency(pt, 15.6629, 0.759192, 0.47756, 2.02154, 0.998816);
  }

  double effEle18(double pt, double eta) {
    if(fabs(eta)<1.479) 
      return efficiency(pt, 18.3193, 0.443703, 0.385554, 1.86523, 0.986514 );
    else
      return efficiency(pt, 19.6586, 0.682633, 0.279486, 2.66423, 0.973455 );
  }

  double effEle20(double pt, double eta) {
    if(fabs(eta)<1.479) 
      return efficiency(pt, 20.554, 0.683776, 0.855573, 1.45917, 1.03957 );
    else
      return efficiency(pt, 23.6386, 1.60775, 1.72093, 1.4131, 1.13962  );
  }

  double effEle18MC(double pt, double eta) {
    if(fabs(eta)<1.479) 
      return efficiency(pt, 15.1804, 2.43126, 3.85048, 1.72284, 0.998507 );
    else
      return efficiency(pt, 16.993, 0.0693958, 0.00695096, 1.9566, 1.00632 );
  }

  //****************
  //mu tau 2012 
  //*****************
  
  double eff2012ATau20(double pt, double eta){
    if(fabs(eta)<1.5) 
      return efficiency(pt, 18.52262128, 1.85879597, 3.48843815, 1.15491294, 1.02489024);
    else 
      return efficiency(pt, 18.90119559, 0.14025596, 0.14482632, 1.56126508, 0.81188198 );
  }

  double eff2012BTau20(double pt, double eta){
    if(fabs(eta)<1.5) 
      return efficiency(pt, 17.92648563, 1.96846742, 4.46406075, 1.02023992, 1.52260575);
    else 
      return efficiency(pt, 18.59856420, 2.49132550, 10.99643595, 1.50651123, 0.87952970 );
  }

  double eff2012MCTau20(double pt, double eta){
    if(fabs(eta)<1.5) 
      return efficiency(pt, 18.86257072, 0.25680380, 0.16916101, 2.42931257, 0.89590264);
    else 
      return efficiency(pt, 18.74764561, 1.82036845, 701.46994969, 101.57913480, 0.82547043);
  }

  double effTau2012MC(double pt, double eta) {
    return eff2012MCTau20(pt, eta);
  }

  double effTau2012A(double pt, double eta) {
    return eff2012ATau20(pt, eta);
  }

  double effTau2012B(double pt, double eta) {
    return eff2012BTau20(pt, eta);
  }

  double effTau2012AB(double pt, double eta) {
    // float weight_A = 696.09; 
    // float weight_B = 4327.0;
    float weight_A = 0.14; // Andrew's weighting
    float weight_B = 0.86;
    return (weight_A * eff2012ATau20(pt, eta) + weight_B * eff2012BTau20(pt, eta))/(weight_A+weight_B);
  }

  double eff2012AMu18(double pt, double eta) {
    if(fabs(eta)<1.2) 
      return efficiency(pt, 15.9998319, -0.39072829, 	0.28256338,	1.72861719, 	0.95769408);
    else
      return efficiency(pt, 18.49754887, 	-0.16941614, 	0.26076717, 	1.05494469, 	1.53819978);
  }
  
  double eff2012BMu17(double pt, double eta) {
    if(fabs(eta)<1.2) 
      return efficiency(pt, 17.21270264, 	0.54997112, 	1.02874912, 	1.29646487, 	0.96724273 );
    else
      return efficiency(pt, 15.98037640, 	0.12062946, 	0.02183977, 	2.84751010,	0.83985656 );
  }

  double effMu2012MC(double pt, double eta) {
    if(fabs(eta)<1.2) 
      return efficiency(pt, 16.99389526, 	-0.04080190, 	0.00794730, 	1.60377906, 	0.99626161  );
    else
      return efficiency(pt,  16.99065795, 	-0.11993730, 	0.01384991, 	2.38867304, 	0.86552275 );
  }
  
  double effMu2012A(double pt, double eta) {
    return eff2012AMu18(pt, eta);
  }
  
  double effMu2012B(double pt, double eta) {
    return eff2012BMu17(pt, eta);
  }
  
  double effMu2012AB(double pt, double eta) {
    // float weight_A = 696.09; 
    // float weight_B = 4327.0;
    float weight_A = 0.14; // Andrew's weighting
    float weight_B = 0.86;

    return (weight_A * eff2012AMu18(pt, eta) + weight_B * eff2012BMu17(pt, eta))/(weight_A+weight_B);
  }
  



  ////////////////////////////HCP 2012/////////////////////////////////////
  // ///2012C Data curve from Josh Friday 5pm
  // EffMuEB_Data   m0 = 16.00061526; sigma = 0.00737246; alpha = 0.00029014; n = 2.12854792; norm = 0.93371791;
  // EffMuEE_Data   m0 = 16.65093710; sigma = 0.48774518; alpha = 0.56076820; n = 1.73768135; norm = 0.86107187;
  double eff2012CMu17(double pt, double eta) {
    if(fabs(eta)<1.2) 
      return efficiency(pt,16.00061526,0.00737246,0.00029014,2.12854792,0.93371791);
    else
      return efficiency(pt,16.65093710,0.48774518,0.56076820,1.73768135,0.86107187);
  }

  /////Muon effiency for full 2012A+B+C
  double effMu2012ABC(double pt, double eta) {
    float weight_A = 696.09; 
    float weight_B = 4327.0;
    float weight_C = 7000.0;
    return (weight_A * eff2012AMu18(pt, eta) + weight_B * eff2012BMu17(pt, eta) + weight_C  * eff2012CMu17(pt, eta)  )/(weight_A+weight_B+weight_C);
  }

  //   ////53X MC numbers from Josh's fit Friday 5pm
  // EffMuEB_MC   m0 = 16.00073094; sigma = 0.00779095; alpha = 0.00029834; n = 2.13782323; norm = 0.95571348;
  // EffMuEE_MC   m0 = 17.03319591; sigma = 0.73033173; alpha = 1.02903291; n = 1.46732719; norm = 0.89420534;
  double effMu2012MC53X(double pt, double eta) {
    if(fabs(eta)<1.2) 
      return efficiency(pt,16.00073094,0.00779095,0.00029834,2.13782323,0.95571348);
    else
      return efficiency(pt,17.03319591,0.73033173,1.02903291,1.46732719,0.89420534);
  }


  ////////////Tau effiency for Full 2012A+B+C Data-set
  // 2012AllEB   m0 = 18.50940288; sigma = 1.62285299; alpha = 2.73232995; n = 1.79135412; norm = 0.91481432; 
  // 2012AllEE   m0 = 18.45678784; sigma = 0.68697618; alpha = 0.57008697; n = 3.73470825; norm = 0.84747211; 
  double effTau2012ABC(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,18.50940288,1.62285299,2.73232995,1.79135412,0.91481432);
    else               return efficiency(pt,18.45678784,0.68697618,0.57008697,3.73470825,0.84747211);
  }


  ///Tau Efficiency for 53X MC
  /// 53XMCEB   m0 = 18.80484409; sigma = 0.19082817; alpha = 0.19983010; n = 1.81979820; norm = 0.93270649; 
  /// 53XMCEE   m0 = 18.25975478; sigma = 1.32745225; alpha = 1.70380810; n = 149.18410074; norm = 0.87377770;
  double effTau2012MC53X(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,18.80484409,0.19082817,0.19983010,1.81979820,0.93270649);
    else               return efficiency(pt,18.25975478,1.32745225,1.70380810,149.18410074,0.87377770);
  }


//   double eff_2012_Rebecca_TauMu_IsoMu18A (double pt, double eta){
//     if (fabs (eta) < 1.2) return efficiency (pt, 17.0055, 0.007058, 0.0004652, 1.56169, 0.963324) ;
//     else                  return efficiency (pt, 17.5451, 0.416753, 0.374504 , 1.56159, 0.881856) ;
//   }

//   double eff_2012_Rebecca_TauMu_IsoMu17B (double pt, double eta){
//     if (fabs (eta) < 1.2) return efficiency (pt, 15.9477, 0.007510  , 0.0002642  , 2.09333, 0.932723) ;
//     else                  return efficiency (pt, 15.999 ,7.52516e-05, 6.77183e-08, 1.62932, 0.855128) ;
//   }

//   double eff_2012_Rebecca_TauMu_IsoMu17C (double pt, double eta){
//     if (fabs (eta) < 1.2) return efficiency (pt, 15.9162, 0.007649, 0.0002533, 2.14652, 0.933307) ;
//     else                  return efficiency (pt, 16.8844, 1.15563 , 1.96536  , 1.37044, 0.883499) ;
//   }

//   double eff_2012_Rebecca_TauMu_IsoMu17BC (double pt, double eta){
//     if (fabs (eta) < 1.2) return efficiency (pt, 15.9199, 0.007744  , 0.0002486  , 2.16741, 0.932644) ;
//     else                  return efficiency (pt, 15.998 ,7.76568e-05, 6.33412e-08, 1.65649, 0.861397) ;
//   }

//   double eff_2012_Rebecca_TauMu_IsoMu1753XMC (double pt, double eta){
//     if (fabs (eta) < 1.2) return efficiency (pt, 15.9562, 0.007615   , 0.0002460  , 2.09189, 0.955486) ;
//     else                  return efficiency (pt, 15.9951, 7.45044e-05, 3.06986e-08, 1.76431, 0.872391) ;
//   }


  double eff_2012_Rebecca_TauMu_IsoMu18A (double pt, double eta){
    if (eta < -1.2)              return efficiency (pt,16.9993,8.82202e-5,7.91529e-8,1.40792,0.928102) ;
    if (-1.2<eta && eta <= -0.8) return efficiency (pt,16.9824,0.0694986,0.0186614,1.66577,0.908218) ;
    if (-0.8<eta && eta <= 0.0)  return efficiency (pt,17.2736,0.13896,0.198452,1.13119,1.21897) ;
    if ( 0.0<eta && eta <= 0.8)  return efficiency (pt,17.9605,0.500059,0.865294,1.04633,1.69027) ;
    if ( 0.8<eta && eta <= 1.2)  return efficiency (pt,18.094,0.607997,0.89385,1.36337,0.92399) ;
    if ( 1.2<eta )               return efficiency (pt,16.9805,9.18396e-5,2.81836e-8,1.83783,0.858988) ;
    return 0.;
  }

  double eff_2012_Rebecca_TauMu_IsoMu17B (double pt, double eta){
    if (eta < -1.2)              return efficiency (pt,16.0015,5.59745e-07,1.3395e-07,1.37357,0.891284) ;
    if (-1.2<eta && eta <= -0.8) return efficiency (pt,18.015,0.0512973,0.0603545,1.36001,0.907481) ;
    if (-0.8<eta && eta <= 0.0)  return efficiency (pt,16.4569,0.214484,0.302707,1.42363,0.982643) ;
    if ( 0.0<eta && eta <= 0.8)  return efficiency (pt,15.9829,0.0435624,0.0196399,1.71605,0.967839) ;
    if ( 0.8<eta && eta <= 1.2)  return efficiency (pt,17.4688,0.0494554,0.0628053,1.34067,0.904989) ;
    if ( 1.2<eta )               return efficiency (pt,16.0029,4.01862e-5,6.62491e-8,1.42189,0.880251) ;
    return 0.;
  }

  double eff_2012_Rebecca_TauMu_IsoMu17C (double pt, double eta){
    if (eta < -1.2)              return efficiency (pt,15.9974,7.20337e-05,7.72238e-08,1.5461,0.87064) ;
    if (-1.2<eta && eta <= -0.8) return efficiency (pt,17.446,0.760355,1.58032,1.0623,1.10472) ;
    if (-0.8<eta && eta <= 0.0)  return efficiency (pt,15.9788,0.044455,0.0215911,1.71024,0.965673) ;
    if ( 0.0<eta && eta <= 0.8)  return efficiency (pt,15.9762,0.0552286,0.0231409,1.78576,0.96848) ;
    if ( 0.8<eta && eta <= 1.2)  return efficiency (pt,17.462,0.804351,1.62323,1.22776,0.900085) ;
    if ( 1.2<eta )               return efficiency (pt,16.0051,-4.10239e-05,1.15509e-08,1.82463,0.865417) ;
    return 0.;
  }
  double eff_2012_Rebecca_TauMu_IsoMu1753XMC (double pt, double eta){
    if (eta < -1.2)              return efficiency (pt,15.997,8.73042e-05,5.36172e-08,1.67934,0.871415) ;
    if (-1.2<eta && eta <= -0.8) return efficiency (pt,17.3339,0.768105,1.31172,1.35161,0.942887) ;
    if (-0.8<eta && eta <= 0.0)  return efficiency (pt,15.959,0.0229759,0.00597735,1.76124,0.980734) ;
    if ( 0.0<eta && eta <= 0.8)  return efficiency (pt,15.9618,0.0587497,0.0189749,1.94016,0.978294) ;
    if ( 0.8<eta && eta <= 1.2)  return efficiency (pt,16.7859,0.443337,0.571078,1.62214,0.919211) ;
    if ( 1.2<eta )               return efficiency (pt,15.9974,8.50572e-05,5.53033e-08,1.64714,0.888026) ;
    return 0.;
  }


  double effMu2012_Rebecca_TauMu_ABC(double pt, double eta) {
    float weight_A = 888.33; 
    float weight_B = 4420.0;    
    float weight_C = 6890.975;    
    return (weight_A * eff_2012_Rebecca_TauMu_IsoMu18A(pt, eta) + 
            weight_B * eff_2012_Rebecca_TauMu_IsoMu17B(pt, eta) + 
            weight_C * eff_2012_Rebecca_TauMu_IsoMu17C(pt, eta))/(weight_A+weight_B+weight_C);
  } 


  ////////////////////Moriond Top-Up/////////////////////////////
  double effMu_muTau_Data_2012D(double pt, double eta){
    if (fabs(eta)<0.8)                        return efficiency (pt,15.9852,0.0428581,0.0160247,1.69952,0.971443) ;
    else if (0.8<=fabs(eta) && fabs(eta)<1.2) return efficiency (pt,16.7041,0.383545,0.467605,1.59941,0.882451) ;
    else                                      return efficiency (pt,15.9994,7.37077e-05,7.21076e-08,1.58178,0.861339) ;
  }
  //for muon MC use same curve as for ABC, Rebecca did not measure a new one for 2012D alone.

  double effTau_muTau_Data_2012D(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,19.09,    0.236111,    0.140104,    2.361672,    0.9137);
    else               return efficiency(pt,19.49,    0.003359,    0.005832,    1.000378,    85.3401);
  }
  double effTau_muTau_MC_2012D(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,18.84,    0.962342,    2.103198,    1.014981,    1.8846);
    else               return efficiency(pt,19.01,    0.492647,    0.449299,    137.190323,    0.8850);
  }

  ///full dataset ABCD
  double effMu_muTau_Data_2012ABCD(double pt, double eta){
    if      (eta<-1.2)  return efficiency (pt,15.9825,7.90724e-05,5.49275e-08,1.6403,0.858285) ;
    else if (eta<-0.8)  return efficiency (pt,17.3283,0.707103,1.2047,1.3732,0.900519) ;
    else if (eta< 0.0)  return efficiency (pt,15.9828,0.0412999,0.0177441,1.66934,0.970097) ;
    else if (eta< 0.8)  return efficiency (pt,15.9802,0.0548775,0.020313,1.79791,0.968398) ;
    else if (eta< 1.2)  return efficiency (pt,16.8396,0.458636,0.633185,1.5706,0.8848) ;
    else                return efficiency (pt,15.9987,8.94398e-05,5.18549e-08,1.8342,0.854625) ;
  }

  double effMu_muTau_MC_2012ABCD(double pt, double eta){
    if      (eta<-1.2)  return efficiency (pt,16.0051,2.45144e-05,4.3335e-09,1.66134,0.87045) ;
    else if (eta<-0.8)  return efficiency (pt,17.3135,0.747636,1.21803,1.40611,0.934983) ;
    else if (eta< 0.0)  return efficiency (pt,15.9556,0.0236127,0.00589832,1.75409,0.981338) ;
    else if (eta< 0.8)  return efficiency (pt,15.9289,0.0271317,0.00448573,1.92101,0.978625 ) ;
    else if (eta< 1.2)  return efficiency (pt,16.5678,0.328333,0.354533,1.67085,0.91699) ;
    else                return efficiency (pt,15.997,7.90069e-05,4.40036e-08,1.66272,0.884502) ;
  }

  double effTau_muTau_Data_2012ABCD(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,18.52036251,1.47760312,2.53574445,1.71202550,0.93019930);
    else               return efficiency(pt,18.41225333,0.76598912,0.60544260,5.38350881,0.85870108);
  }
  double effTau_muTau_MC_2012ABCD(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,18.88740627,0.10718873,0.12277723,1.60581265,0.95041892);
    else               return efficiency(pt,18.30439676,1.44360240,3.79358997,1.07560564,0.93103925);
  }

  

  ////**************************************
  ///mu-Tau Summer 13 ReReco data-set
  ///****************************************
  double effMu_muTau_Data_2012ABCDSummer13(double pt, double eta){
    if      (eta<-1.2)  return efficiency (pt,15.9977,7.64004e-05,6.4951e-08,1.57403,0.865325);
    else if (eta<-0.8)  return efficiency (pt,17.3974,0.804001,1.47145,1.24295,0.928198) ;
    else if (eta< 0.0)  return efficiency (pt,16.4307,0.226312,0.265553,1.55756,0.974462) ;
    else if (eta< 0.8)  return efficiency (pt,17.313,0.662731,1.3412,1.05778,1.26624) ;
    else if (eta< 1.2)  return efficiency (pt,16.9966,0.550532,0.807863,1.55402,0.885134) ;
    else                return efficiency (pt,15.9962,0.000106195,4.95058e-08,1.9991,0.851294) ;
  }

  ////muon MC efficiency remains same as for Moriond

  double effTau_muTau_Data_2012ABCDSummer13(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,18.604910,    0.276042,    0.137039,    2.698437,    0.940721);                                            
    else               return efficiency(pt,18.701715,    0.216523,    0.148111,    2.245081,    0.895320);
  }
  double effTau_muTau_MC_2012ABCDSummer13(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,18.532997,    1.027880,    2.262950,    1.003322,    5.297292);
    else               return efficiency(pt,18.212782,    0.338119,    0.122828,    12.577926,    0.893975);
  }


  //********************************
  //e-tau 2012
  //********************************

  //Upto ICHEP
  double effEle2012AB(double pt, double eta) {
    float weight_A = 696.09; 
    float weight_B = 4327.0;    
    return (weight_A * eff2012AEle20(pt, eta) + weight_B * eff2012BEle22(pt, eta))/(weight_A+weight_B);
  } 
  //Upto ICHEP
  double effTau2012AB_TauEle(double pt, double eta) {
    float weight_A = 696.09; 
    float weight_B = 4327.0;
    return (weight_A * eff2012ATau20_TauEle(pt, eta) + weight_B * eff2012BTau20_TauEle(pt, eta))/(weight_A+weight_B);
  }


  // Electron (2012) //Upto ICHEP
  // Trigger 	                     m0 	  sigma 	   alpha 	      n 	  norm
  // Ele20 EB - Run < 193686 	20.97643939 	1.15196354 	2.27544602 	1.01743868 	2.04391816
  // Ele22 EB - Run >= 193686 	22.90752344 	1.32376429 	2.17813319 	1.03674051 	2.15454768
  // Ele 20 EB - 52X MC 	        20.58604584 	-1.89456806 	3.69311772 	1.05480046 	1.28655181
  
  // Ele20 EE - Run < 193686 	20.59874300 	1.25425435 	1.61098921 	1.00146962 	60.35067579
  // Ele22 EE - Run >= 193686 	22.14553261 	1.19913124 	1.75642067 	1.00826962 	9.04331617
  // Ele 20 EE - 52X MC 	        20.15425918 	0.75449122 	1.06027513 	1.01106686 	7.01956561 
  double eff2012AEle20(double pt, double eta) {
    if(fabs(eta)<1.479) 
      return efficiency(pt, 20.97643939 , 1.15196354 , 2.27544602, 1.01743868, 2.04391816);
    else
      return efficiency(pt, 20.59874300, 1.25425435, 1.61098921, 1.00146962, 60.35067579);
  }  
  double eff2012BEle22(double pt, double eta) {
    if(fabs(eta)<1.479) 
      return efficiency(pt, 22.90752344, 1.32376429 , 2.17813319, 1.03674051, 2.15454768);
    else
      return efficiency(pt, 22.14553261, 1.19913124 , 1.75642067, 1.00826962, 9.04331617);
  }
  double eff2012Ele20MC(double pt, double eta) {
    if(fabs(eta)<1.479)  
      return efficiency(pt, 20.58604584, -1.89456806 , 3.69311772, 1.05480046, 1.28655181);
    else
      return efficiency(pt, 20.15425918,  0.75449122 , 1.06027513, 1.01106686, 7.01956561);
  }


  // Tau (2012) //Upto ICHEP
  // Trigger (Tau) 	             m0 	   sigma 	  alpha              n    	norm
  // Loose Tau 20 - Run < 193686 	  18.84658959 	0.25958704 	0.17300958 	2.43491208 	0.85872017
  // Loose Tau 20 - Run >= 193686   18.48663118 	1.63417147 	20.25695815 	138.55422224 	0.89456038
  // Loose Tau 20 - 52X MC 	  18.77448606 	0.45765507 	0.26077509 	13.43372485 	0.88037836
  double eff2012ATau20_TauEle(double pt, double eta) {
    return efficiency(pt, 18.84658959 , 0.25958704,  0.17300958,   2.43491208, 0.85872017);
  }
  double eff2012BTau20_TauEle(double pt, double eta) {
    return efficiency(pt, 18.48663118 , 1.63417147, 20.25695815, 138.55422224, 0.89456038);
  }
  double eff2012Tau20MC_TauEle(double pt, double eta) {
    return efficiency(pt, 18.77448606 , 0.45765507,  0.26077509,  13.43372485, 0.88037836);
  }


  ////////////////////////////HCP 2012/////////////////////////////////////
  // EffEleEB_MC   m0 = 22.00666445; sigma = 0.00036058; alpha = 0.00000251; n = 1.38456083; norm = 1.02640579; 
  // EffEleEE_MC   m0 = 22.18226941; sigma = 1.07762306; alpha = 1.23712775; n = 1.27324238; norm = 1.15312185;


  double eff_2012_Rebecca_TauEle_Ele20A (double pt, double eta){
    if (fabs (eta) < 1.479) return efficiency (pt,20.4669, 1.20429, 1.84954, 1.38645, 0.891122) ; 
    else                    return efficiency (pt,21.4136, 1.93922, 2.43562, 1.00186, 51.947  ) ; 
  }

  double eff_2012_Rebecca_TauEle_Ele22B (double pt, double eta){
    if (fabs (eta) < 1.479) return efficiency (pt,22.8618, 0.844755,  1.07941, 1.27956, 1.07722 ) ;
    else                    return efficiency (pt,22.1045, 1.08481 , 0.780119, 1.91846, 0.962174) ;
  }

  double eff_2012_Rebecca_TauEle_Ele22C (double pt, double eta){
    if (fabs (eta) < 1.479) return efficiency (pt,22.8598, 0.855666, 1.02951 , 1.32713, 1.05486 ) ;
    else                    return efficiency (pt,21.7643, 1.45024 , 0.785753, 3.14722, 0.926788) ;
  }

  double eff_2012_Rebecca_TauEle_Ele22BC (double pt, double eta){
    if (fabs (eta) < 1.479) return efficiency (pt,22.8925, 0.86372, 1.13289, 1.22478, 1.13184 ) ;
    else                    return efficiency (pt,22.0292, 1.4626 , 0.97438, 2.47942, 0.937275) ;
  }

  double eff_2012_Rebecca_TauEle_Ele2253XMC (double pt, double eta){
    if (fabs (eta) < 1.479) return efficiency (pt,21.4136, 0.000422, 2.47314e-06, 1.42487, 1.00104) ;
    else                    return efficiency (pt,20.9985, 0.002918, 3.43131e-05, 1.41479, 1.06506) ;
  }

  double effEle2012_Rebecca_TauEle_ABC(double pt, double eta) {
    float weight_A = 888.33; // see python/proto/samples/run2012/data.py
    float weight_B = 4420.0;    
    float weight_C = 6890.975;    
    return (weight_A * eff_2012_Rebecca_TauEle_Ele20A(pt, eta) + 
            weight_B * eff_2012_Rebecca_TauEle_Ele22B(pt, eta) + 
            weight_C * eff_2012_Rebecca_TauEle_Ele22C(pt, eta))/(weight_A+weight_B+weight_C);
  } 

  // for 2012C Data
  // EffEleEB_Data   m0 = 23.05556088; sigma = 0.96047151; alpha = 1.24782044; n = 1.26042277; norm = 1.09675041; 
  // EffEleEE_Data   m0 = 21.99911375; sigma = 1.15806380; alpha = 0.80675262; n = 1.98765770; norm = 0.97138507; 
  double eff2012CEle22(double pt, double eta) {
    if(fabs(eta)<1.479) 
      return efficiency(pt,23.05556088,0.96047151,1.24782044,1.26042277,1.09675041);
    else
      return efficiency(pt,21.99911375,1.15806380,0.80675262,1.98765770,0.97138507);
  }

  double effEle2012ABC(double pt, double eta) {
    float weight_A = 696.09; 
    float weight_B = 4327.0;    
    float weight_C = 7000.;    
    return (weight_A * eff2012AEle20(pt, eta) + weight_B * eff2012BEle22(pt, eta) + weight_C * eff2012CEle22(pt, eta))/(weight_A+weight_B+weight_C);
  } 

  // This function is the old one used by Jose (from Valentina, probably):

  double effEle2012MC53X(double pt, double eta) {
    if(fabs(eta)<1.479) 
      return efficiency(pt,22.00666445,0.00036058,0.00000251,1.38456083,1.02640579);
    else
      return efficiency(pt,22.18226941,1.07762306,1.23712775,1.27324238,1.15312185);
  }

  
  //For Tau curves see email from Josh
//   // Tau, EleTau (2012)
//   // Loose Tau 20 - Run < 193686 	        18.84658959 	0.25958704 	0.17300958 	2.43491208 	0.85872017
//   // Loose Tau 20 - Run >= 193686 	18.48663118 	1.63417147 	20.25695815 	138.55422224 	0.89456038
//   //He has not measured the C period therefore just use from 2012B curve which has same trigger
//   double effTau2012ABC_TauEle(double pt, double eta) {
//     float weight_A = 696.09; 
//     float weight_B = 4327.0;
//     float weight_C = 7000.;
//     return (weight_A * eff2012ATau20_TauEle(pt, eta) + weight_B * eff2012BTau20_TauEle(pt, eta)+ weight_C * eff2012BTau20_TauEle(pt, eta))/(weight_A+weight_B+weight_C);
//   }
//   // And this for the MC:
//   // Eff53XMCET   m0 = 18.62733399; sigma = 0.51301539; alpha = 0.38517573; n = 5.68099833; norm = 0.91536401;
//   double eff2012Tau20MC53X_TauEle(double pt, double eta) {
//     return efficiency(pt, 18.62733399 , 0.51301539, 0.38517573 , 5.68099833 , 0.91536401);
//   }


// ///Email from Josh on October 15: measured one curve for whole 2012 period and separated Barrel and Endcap
// 2012AllETEB   m0 = 18.43442868; sigma = 2.08967536; alpha = 3.27357845; n = 6.96327309; norm = 0.85564484;
// 2012AllETEE   m0 = 18.16839440; sigma = 1.86184564; alpha = 4.39116712; n = 1.01410741; norm = 1.39240481;
// Eff53XMCETEB   m0 = 18.40815138; sigma = 1.53235636; alpha = 3.55989632; n = 1.74542709; norm = 0.90118450;
// Eff53XMCETEE   m0 = 18.29028052; sigma = 1.56239255; alpha = 11.03605631; n = 155.89290151; norm = 0.85683995; 
  double effTau2012ABC_TauEle(double pt, double eta) {
    if(fabs(eta)<1.5) return efficiency(pt,18.43442868,2.08967536,3.27357845,6.96327309,0.85564484);
    else              return efficiency(pt,18.16839440,1.86184564,4.39116712,1.01410741,1.39240481);
  }
  double eff2012Tau20MC53X_TauEle(double pt, double eta) {
     if(fabs(eta)<1.5) return efficiency(pt,18.40815138,1.53235636,3.55989632,1.74542709,0.90118450);
     else              return efficiency(pt,18.29028052,1.56239255,11.03605631,155.89290151,0.85683995);
  }



  ////////////////////Moriond Top-Up/////////////////////////////
  double effEle_eTau_Data_2012D(double pt, double eta){
    if (fabs(eta)<1.479)    return efficiency (pt,23.2037,0.947222,1.29024,1.09804,1.53015 ) ;
    else                    return efficiency (pt,21.86,0.979008,0.505753,2.2701,0.94213) ;
  }
  //for electron MC use curve for ABC

  double effTau_eTau_Data_2012D(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,18.73,    0.374578,    0.136068,    5.789410,    0.8638);
    else               return efficiency(pt,19.32,    0.146243,    0.123579,    3.126114,    0.8313);
  }
  double effTau_eTau_MC_2012D(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,19.22,    0.204905,    0.175676,    2.644803,    0.8974);
    else               return efficiency(pt,18.62,    0.037935,    0.002134,    95.090919,    0.8515);
  }
  
  //full data-set ABCD 
  double effEle_eTau_Data_2012ABCD(double pt, double eta){
    if(fabs(eta)<1.479)  return efficiency (pt,22.9041,1.04728,1.38544,1.22576,1.13019) ;
    else                 return efficiency (pt,21.9941,1.43419,1.01152,2.28622,0.939872) ;
  }
  double effEle_eTau_MC_2012ABCD(double pt, double eta){
    if(fabs(eta)<1.479)  return efficiency (pt,21.7243,0.619015,0.739301,1.34903,1.02594) ;
    else                 return efficiency (pt,22.1217,1.34054,1.8885,1.01855,4.7241) ;
  }
  double effTau_eTau_Data_2012ABCD(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,18.686211,    1.993524,    3.202713,    3.612693,    0.871640);
    else               return efficiency(pt,18.472954,    1.606388,    3.468975,    55.629620,    0.828977);
  }
  double effTau_eTau_MC_2012ABCD(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,18.431118,    1.572877,    3.301699,    4.760769,    0.899620);
    else               return efficiency(pt,18.257217,    1.632443,    9.283116,    40.219585,    0.858643);
  }


  ////**************************************
  ///e-Tau Summer 13 ReReco data-set  (done with Ivo's new WP for antiElectronMediumMVA3)
  ///****************************************
  double effEle_eTau_Data_2012ABCDSummer13(double pt, double eta){
    if(fabs(eta)<1.479)  return efficiency (pt,22.9704,1.0258,1.26889,1.31024,1.06409) ;
    else                 return efficiency (pt,21.9816,1.40993,0.978597,2.33144,0.937552) ;
  }
  //electron MC efficiency remains same as Moriond
  double effTau_eTau_Data_2012ABCDSummer13(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,18.538229,    0.651562,    0.324869,    13.099048,    0.902365);
    else               return efficiency(pt,18.756548,    0.230732,    0.142859,    3.358497,    0.851919);
  }
  double effTau_eTau_MC_2012ABCDSummer13(double pt, double eta) {
    if(fabs(eta)<1.5)  return efficiency(pt,18.605055,    0.264062,    0.139561,    4.792849,    0.915035);
    else               return efficiency(pt,18.557810,    0.280908,    0.119282,    17.749043,    0.865756);
  }



  //****************
  //first trigger turn-ons for the di-tau trigger from Simone
  //*****************
  

  double effIsoTau20(double pt, double eta){
    double p0 = 0.886928;
    double p1 = 28.1136;
    double p2 = 1.04502;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }

  double effIsoTau25(double pt, double eta){
    double p0 = 0.894481;
    double p1 = 32.7471;
    double p2 = 0.915929;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }

  double effIsoTau35(double pt, double eta){
    double p0 = 0.930435;
    double p1 = 43.3497;
    double p2 = 1.03643;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }

  double effIsoTau45(double pt, double eta){
    double p0 = 0.94552;
    double p1 = 56.6926;
    double p2 = 1.30613;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }

  double effTau1fb(double pt, double eta){
    float tau20w = 200.;
    float tau25w = 139.;
    float tau35w = 790.;
    return ( tau20w * effIsoTau20(pt,eta) + 
	     tau25w * effIsoTau25(pt,eta) + 
	     tau35w * effIsoTau35(pt,eta) ) / ( tau20w + tau25w + tau35w);
  }
  
  double effTau5fb(double pt, double eta){
    float tau20w = 200.;
    float tau25w = 139.;
    float tau35w = 790.;
    float tau45w = 3500.;
    return ( tau20w * effIsoTau20(pt,eta) + 
	     tau25w * effIsoTau25(pt,eta) + 
	     tau35w * effIsoTau35(pt,eta) + 
	     tau45w * effIsoTau45(pt,eta) ) / ( tau20w + tau25w + tau35w + tau45w);
  }
  
  double eff2012IsoTau25(double pt, double eta){
    double p0 = 0.86;
    double p1 = 36.0;
    double p2 = 1.15;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }

  double eff2012IsoTau30(double pt, double eta){
    double p0 = 0.839697;
    double p1 = 38.3468;
    double p2 = 1.0334;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }

  double eff2012IsoTau1_6fb(double pt, double eta){
    float tau25w = 400;
    float tau30w = 1200;
    return ( tau25w * eff2012IsoTau25(pt,eta) + 
	     tau30w * eff2012IsoTau30(pt,eta) ) / ( tau25w + tau30w);
  }
  
  double eff2012IsoTau5_1fb(double pt, double eta){
    float tau25w = 400;
    float tau30w = 4700;
    return ( tau25w * eff2012IsoTau25(pt,eta) + 
	     tau30w * eff2012IsoTau30(pt,eta) ) / ( tau25w + tau30w);
  }
  
  double eff2012Jet30(double pt, double eta){
    double p0 = 0.9714;
    double p1 = 34.56;
    double p2 = 1.143;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }

  double eff2012Jet5fb(double pt, double eta){
    double p0 = 0.989366;
    double p1 = 33.5362;
    double p2 = 1.27463;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }

  double eff2012IsoTau5fb(double pt, double eta){
    double p0 = 0.829767;
    double p1 = 38.4455;
    double p2 = 1.06633;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }
  
  double eff2012IsoTau5fbUp(double pt, double eta){
    double p0 = 0.829767;
    double p1 = 38.4455*1.03;
    double p2 = 1.06633;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }

  double eff2012IsoTau5fbDown(double pt, double eta){
    double p0 = 0.829767;
    double p1 = 38.4455/1.03;
    double p2 = 1.06633;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }
  
  double eff2012IsoTau5fbUpSlope(double pt, double eta){
    double p0 = 0.829767;
    double p1 = 38.4455;
    double p2 = 1.06633*1.06;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }

  double eff2012IsoTau5fbDownSlope(double pt, double eta){
    double p0 = 0.829767;
    double p1 = 38.4455;
    double p2 = 1.06633/1.06;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }
  
  double eff2012IsoTau5fbUpPlateau(double pt, double eta){
    double p0 = 0.829767*1.03;
    double p1 = 38.4455;
    double p2 = 1.06633;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }

  double eff2012IsoTau5fbDownPlateau(double pt, double eta){
    double p0 = 0.829767/1.03;
    double p1 = 38.4455;
    double p2 = 1.06633;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }
  
double crystalballfunc(double m, double m0, double sigma, double alpha, double n,
        double norm) {
    const double sqrtPiOver2 = 1.2533141373;
    const double sqrt2 = 1.4142135624;
    double sig = fabs((double) sigma);
    double t = (m - m0) / sig;
    if (alpha < 0)
        t = -t;
    double absAlpha = fabs(alpha / sig);
    double a = TMath::Power(n / absAlpha, n) * exp(-0.5 * absAlpha * absAlpha);
    double b = absAlpha - n / absAlpha;
    double ApproxErf;
    double arg = absAlpha / sqrt2;
    if (arg > 5.)
        ApproxErf = 1;
    else if (arg < -5.)
        ApproxErf = -1;
    else
        ApproxErf = TMath::Erf(arg);
    double leftArea = (1 + ApproxErf) * sqrtPiOver2;
    double rightArea = (a * 1 / TMath::Power(absAlpha - b, n - 1)) / (n - 1);
    double area = leftArea + rightArea;
    if (t <= absAlpha) {
        arg = t / sqrt2;
        if (arg > 5.)
            ApproxErf = 1;
        else if (arg < -5.)
            ApproxErf = -1;
        else
            ApproxErf = TMath::Erf(arg);
        return norm * (1 + ApproxErf) * sqrtPiOver2 / area;
    } else {
        return norm * (leftArea + a * (1 / TMath::Power(t - b, n - 1) - 1
                / TMath::Power(absAlpha - b, n - 1)) / (1 - n)) / area;
    }
}   

  double eff2012IsoTau5fbCrystalBall(double pt, double eta){
    double m0 = 37.6182;
    double sigma = 6.9402;
    double alpha = 14.4058;
    double n = 1.00844;
    double norm = 2.82596;
    return crystalballfunc(pt, m0, sigma, alpha, n, norm);
  }
  
  double eff2012IsoTau5fbFitFrom30(double pt, double eta){
    double p0 = 0.808613;
    double p1 = 37.7854;
    double p2 = 0.901174;
    return p0*0.5*(TMath::Erf((pt-p1)/2./p2/sqrt(pt))+1.);
  }
  
  double eff2012IsoTau12fb(double pt, double eta){
    return (808.411*(0.764166*0.5*(TMath::Erf((pt-33.2236)/2./0.97289/sqrt(pt))+1.))+
            4428.0*(0.802387*0.5*(TMath::Erf((pt-38.0971)/2./0.82842/sqrt(pt))+1.))+
            1783.003*(0.818051*0.5*(TMath::Erf((pt-37.3669)/2./0.74847/sqrt(pt))+1.))+
            5109.155*(0.796086*0.5*(TMath::Erf((pt-37.3302)/2./0.757558/sqrt(pt))+1.))
            )/(808.411+4428.0+1783.003+5109.155);
  }
  
  double eff2012Jet12fb(double pt, double eta){
    return (abs(eta)<=2.1)*
           ((808.411*(0.99212*0.5*(TMath::Erf((pt-31.3706)/2./1.22821/sqrt(pt))+1.))+
            4428.0*(0.99059*0.5*(TMath::Erf((pt-32.1104)/2./1.23292/sqrt(pt))+1.))+
            1783.003*(0.988256*0.5*(TMath::Erf((pt-31.3103)/2./1.18766/sqrt(pt))+1.))+
            5109.155*(0.988578*0.5*(TMath::Erf((pt-31.6391)/2./1.22826/sqrt(pt))+1.))
            )/(808.411+4428.0+1783.003+5109.155))+
	   (abs(eta)>2.1)*
	   ((808.411*(0.969591*0.5*(TMath::Erf((pt-36.8179)/2./0.904254/sqrt(pt))+1.))+
            4428.0*(0.975932*0.5*(TMath::Erf((pt-37.2121)/2./0.961693/sqrt(pt))+1.))+
            1783.003*(0.990305*0.5*(TMath::Erf((pt-36.3096)/2./0.979524/sqrt(pt))+1.))+
            5109.155*(0.971612*0.5*(TMath::Erf((pt-36.2294)/2./0.871726/sqrt(pt))+1.))
            )/(808.411+4428.0+1783.003+5109.155));
  }
  
  double eff2012Jet19fb(double pt, double eta){
    return (abs(eta)<=2.1)*
    ( ( 808.411 * ( 0.99212  * 0.5 * (TMath::Erf((pt-31.3706)/2./1.22821/sqrt(pt))+1.))
      + 4428.0  * ( 0.99059  * 0.5 * (TMath::Erf((pt-32.1104)/2./1.23292/sqrt(pt))+1.))
      + 1783.003* ( 0.988256 * 0.5 * (TMath::Erf((pt-31.3103)/2./1.18766/sqrt(pt))+1.))
      + 5109.155* ( 0.988578 * 0.5 * (TMath::Erf((pt-31.6391)/2./1.22826/sqrt(pt))+1.))
      + 4131.   * ( 0.989049 * 0.5 * (TMath::Erf((pt-31.9836)/2./1.23871/sqrt(pt))+1.))
      + 3143.   * ( 0.988047 * 0.5 * (TMath::Erf((pt-31.6975)/2./1.25372/sqrt(pt))+1.)))
    /(808.411+4428.0+1783.003+5109.155+4131+3143))+
    (abs(eta)>2.1)*
    ( ( 808.411 *( 0.969591  * 0.5 * (TMath::Erf((pt-36.8179)/2./0.904254/sqrt(pt))+1.))
      + 4428.0  *( 0.975932  * 0.5 * (TMath::Erf((pt-37.2121)/2./0.961693/sqrt(pt))+1.))
      + 1783.003*( 0.990305  * 0.5 * (TMath::Erf((pt-36.3096)/2./0.979524/sqrt(pt))+1.))
      + 5109.155*( 0.971612  * 0.5 * (TMath::Erf((pt-36.2294)/2./0.871726/sqrt(pt))+1.))
      + 4131.   *( 0.977958  * 0.5 * (TMath::Erf((pt-37.131 )/2./0.987523/sqrt(pt))+1.))
      + 3143.   *( 0.968457  * 0.5 * (TMath::Erf((pt-36.3159)/2./0.895031/sqrt(pt))+1.)))
    /(808.411+4428.0+1783.003+5109.155+4131+3143));
  }    

  double eff2012IsoTau19fb(double pt, double eta){
    return (  808.411  * ( 0.764166 * 0.5 * (TMath::Erf((pt-33.2236)/2./0.97289 /sqrt(pt))+1.))
            + 4428.0   * ( 0.802387 * 0.5 * (TMath::Erf((pt-38.0971)/2./0.82842 /sqrt(pt))+1.))
            + 1783.003 * ( 0.818051 * 0.5 * (TMath::Erf((pt-37.3669)/2./0.74847 /sqrt(pt))+1.))
            + 5109.155 * ( 0.796086 * 0.5 * (TMath::Erf((pt-37.3302)/2./0.757558/sqrt(pt))+1.))
            + 4131.    * ( 0.828182 * 0.5 * (TMath::Erf((pt-37.6596)/2./0.830682/sqrt(pt))+1.))
            + 3143.    * ( 0.833004 * 0.5 * (TMath::Erf((pt-37.634 )/2./0.777843/sqrt(pt))+1.)) )
            /(808.411+4428.0+1783.003+5109.155+4131+3143);
  }
  
  double eff2012IsoTau35Park(double pt, double eta){
    return (0.83*0.5*(TMath::Erf((pt-40.8)/2./1.41/sqrt(pt))+1.));
  }
  
  double eff2012IsoTau1prong12fb(double pt, double eta){
    return (808.411 *(0.73201 *0.5*(TMath::Erf((pt-41.0386)/2./0.663772/sqrt(pt))+1.))+
            4428.0  *(0.764371*0.5*(TMath::Erf((pt-41.2967)/2./0.71147 /sqrt(pt))+1.))+
            1783.003*(0.840278*0.5*(TMath::Erf((pt-41.7228)/2./0.754745/sqrt(pt))+1.))+
            5109.155*(0.7879  *0.5*(TMath::Erf((pt-40.9494)/2./0.66706 /sqrt(pt))+1.))
            )/(808.411+4428.0+1783.003+5109.155);
  }

  double eff2012IsoTau1prong19fb(double pt, double eta){
    return (808.411 *(0.73201 *0.5*(TMath::Erf((pt-41.0386)/2./0.663772/sqrt(pt))+1.))+
            4428.0  *(0.764371*0.5*(TMath::Erf((pt-41.2967)/2./0.71147 /sqrt(pt))+1.))+
            1783.003*(0.840278*0.5*(TMath::Erf((pt-41.7228)/2./0.754745/sqrt(pt))+1.))+
            5109.155*(0.7879  *0.5*(TMath::Erf((pt-40.9494)/2./0.66706 /sqrt(pt))+1.))+
            4131    *(0.811053*0.5*(TMath::Erf((pt-41.2314)/2./0.72215 /sqrt(pt))+1.))+
            3143    *(0.802065*0.5*(TMath::Erf((pt-41.0161)/2./0.654632/sqrt(pt))+1.))
            )/(808.411+4428.0+1783.003+5109.155+4131+3143);
  }








  double eff2012IsoTau19fb_Simone(double pt, double eta){

    // for real Taus mT<20
    if ( fabs(eta) < 1.4 )
    {
      return (  808.411  * ( 0.764166 * 0.5 * (TMath::Erf((pt-33.2236)/2./0.97289 /sqrt(pt))+1.))   // 2012A by Bastian not split in eta
              + 4428.0   * ( 0.75721  * 0.5 * (TMath::Erf((pt-39.0836)/2./1.07753 /sqrt(pt))+1.))   // 2012B
              + 6892.158 * ( 0.791464 * 0.5 * (TMath::Erf((pt-38.4932)/2./1.01232 /sqrt(pt))+1.))   // 2012C measured in v2 only
              + 7274.    * ( 0.779446 * 0.5 * (TMath::Erf((pt-38.4603)/2./1.01071 /sqrt(pt))+1.)) ) // 2012D measured in one go
              /( 808.411 + 4428.0 + 6892.158 + 7274. );
    }
    
    else
    {
      return (  808.411  * ( 0.764166 * 0.5 * (TMath::Erf((pt-33.2236)/2./0.97289 /sqrt(pt))+1.))   // 2012A by Bastian not split in eta
              + 4428.0   * ( 0.693788 * 0.5 * (TMath::Erf((pt-37.7719)/2./1.09202 /sqrt(pt))+1.))   // 2012B
              + 6892.158 * ( 0.698909 * 0.5 * (TMath::Erf((pt-36.5533)/2./1.05743 /sqrt(pt))+1.))   // 2012C measured in v2 only
              + 7274.    * ( 0.703532 * 0.5 * (TMath::Erf((pt-38.8609)/2./1.05514 /sqrt(pt))+1.)) ) // 2012D measured in one go
              /( 808.411 + 4428.0 + 6892.158 + 7274. );
    }
    
  }

  double eff2012IsoTau19fbMC_Simone(double pt, double eta){

    // for real Taus using ggH120
    if ( fabs(eta) < 1.4 )
    {
      return ( 0.807425 * 0.5 * (TMath::Erf((pt-35.2214)/2./1.04214  /sqrt(pt))+1.) ) ;
    }
    
    else
    {
      return ( 0.713068 * 0.5 * (TMath::Erf((pt-33.4584)/2./0.994692 /sqrt(pt))+1.) ) ;
    }
    
  }


  // Tau Parked with HLT_DoubleMediumIsoPFTau35_Trk*_eta2p1_v*
  double eff2012IsoParkedTau19fb_Simone(double pt, double eta){

    // for real Taus mT<20
    if ( fabs(eta) < 1.4 )
    {
      return (  0.883869 * 0.5 * (TMath::Erf((pt-43.8723)/2./0.946593 /sqrt(pt))+1.) ) ;// 2012CD measured in one go, take this for all as of may 15
              
    }
    
    else
    {
      return (  0.798480 * 0.5 * (TMath::Erf((pt-43.1362)/2./1.04861  /sqrt(pt))+1.) ) ;// 2012CD measured in one go, take this for all as of may 15
    }
    
  }


  double eff2012IsoParkedTau19fbMC_Simone(double pt, double eta){

    // for real Taus using ggH120
    if ( fabs(eta) < 1.4 )
    {
      return ( 0.814832 * 0.5 * (TMath::Erf((pt-40.1457)/2./0.856575  /sqrt(pt))+1.) ) ;
    }
    
    else
    {
      return ( 0.661991 * 0.5 * (TMath::Erf((pt-38.0195)/2./0.833499 /sqrt(pt))+1.) ) ;
    }
    
  }


 
 
private:

  //function definition taken from AN-11-390 v4
  double efficiency(double m, double m0, double sigma, double alpha,double n, double norm) const;
} ;
#endif 


