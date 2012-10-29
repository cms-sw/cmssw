/*
       Name                 EtaMin1 EtaMax1 EtaMin2 EtaMax2  Order   ResCoreMate 1        ResCorMate 2
 0 EvtPlaneFromTracksMidEta	 -0.75   0.75		       2        etHFm                    etHFp
 1 EvtPTracksPosEtaGap		     1	    2		       2        EPlaneFromTracksMidEta   EPTracksNegEtaGap
 2 EvtPTracksNegEtaGap		    -2	   -1	               2        EPlaneFromTracksMidEta   EPTracksPosEtaGap

 3 EPTracksMid3	                 -0.75   0.75		       3        etHFm3                   etHFp3
 4 EPTracksPos3		             1	    2		       3        EPTracksMid3             EPTracksNeg3
 5 EPTracksNeg3		            -2	   -1	               3        EPTracksMid3             EPTracksPos3

 6 EPTracksMid4	                 -0.75   0.75		       4        etHFm4                   etHFp4
 7 EPTracksPos4		             1	    2		       4        EPTracksMid4             EPTracksNeg4
 8 EPTracksNeg4		            -2	   -1	               4        EPTracksMid4             EPTracksPos4

 9 EPTracksMid5	                 -0.75   0.75		       5        etHFm5                   etHFp5
10 EPTracksPos5		             1	    2		       5        EPTracksMid5             EPTracksNeg5
11 EPTracksNeg5		            -2	   -1	               5        EPTracksMid5             EPTracksPos5

12 EPTracksMid6	                 -0.75   0.75		       6        etHFm6                   etHFp6
13 EPTracksPos6		             1	    2		       6        EPTracksMid6             EPTracksNeg6
14 EPTracksNeg6		            -2	   -1	               6        EPTracksMid6             EPTracksPos6

15 etEcal	                    -2.7  2.7		       2        etHFm                    etHFp
16 etEcalP	                     0.3  2.7		       2        etHFm                    etHFp
17 etEcalM	                    -2.7 -0.3		       2        etHFm                    etHFp

18 etHcal	                    -2.7  2.7		       2        etHFm                    etHFp
19 etHcalP	                     0.3  2.7		       2        etHFm                    etHFp
20 etHcalM	                    -2.7 -0.3		       2        etHFm                    etHFp

21 etHF	                            -5	   -3	   3	  5    2        EvtPTracksNegEtaGap      EvtPTracksPosEtaGap
22 etHFp	                     3	    5		       2        etHFm                    EvtPlaneFromTracksMidEta
23 etHFm	                    -5	   -3		       2        EvtPlaneFromTracksMidEta etHFp

24 etHF3	                    -5	   -3	   3	  5    3        EPTracksNeg3             EPTracksPos3 
25 etHFp3	                     3	    5		       3        etHFm3                   EPTracksMid3
26 etHFm3	                    -5	   -3		       3        EPTracksMid3             etHFp3

27 etHF4	                    -5	   -3	   3	  5    4        EPTracksNeg4             EPTracksPos4 
28 etHFp4	                     3	    5		       4        etHFm4                   EPTracksMid4
29 etHFm4	                    -5	   -3		       4        EPTracksMid4             etHFp4

30 etHF5	                    -5	   -3	   3	  5    5        EPTracksNeg5             EPTracksPos5 
31 etHFp5	                     3	    5		       5        etHFm5                   EPTracksMid5
32 etHFm5	                    -5	   -3		       5        EPTracksMid5             etHFp5

33 etHF6	                    -5	   -3	   3	  5    6        EPTracksNeg6             EPTracksPos6 
34 etHFp6	                     3	    5		       6        etHFm6                   EPTracksMid6
35 etHFm6	                    -5	   -3		       6        EPTracksMid6             etHFp6

36 etCaloHFP	                  0.25	    5		       2        etHFm                    etEcalM
37 etCaloHFM	                    -5  -0.25		       2        etHFp                    etEcalP

*/
#include <string>
namespace hi{
enum EPNamesInd {
                 EvtPlaneFromTracksMidEta,    EvtPTracksPosEtaGap,    EvtPTracksNegEtaGap,        
		 EPTracksMid3,                EPTracksPos3,           EPTracksNeg3,                
                 EPTracksMid4,                EPTracksPos4,           EPTracksNeg4,    
                 EPTracksMid5,                EPTracksPos5,           EPTracksNeg5,    
                 EPTracksMid6,                EPTracksPos6,           EPTracksNeg6,    
                 etEcal,                      etEcalP,                etEcalM,                
                 etHcal,                      etHcalP,                etHcalM,
		 etHF,                        etHFp,                  etHFm,                    
		 etHF3,                       etHFp3,                 etHFm3,                    
		 etHF4,                       etHFp4,                 etHFm4,                    
		 etHF5,                       etHFp5,                 etHFm5,                    
		 etHF6,                       etHFp6,                 etHFm6,                    
                 etCaloHFP,                   etCaloHFM  
};

const int RCMate1[] = {
                etHFm,                      EvtPlaneFromTracksMidEta, EvtPlaneFromTracksMidEta, 
                etHFm3,                     EPTracksMid3,             EPTracksMid3,               
                etHFm4,                     EPTracksMid4,             EPTracksMid4,    
                etHFm5,                     EPTracksMid5,             EPTracksMid5,    
                etHFm6,                     EPTracksMid6,             EPTracksMid6,    
                etHFm,                      etHFm,                    etHFm,                    
                etHFm,                      etHFm,                    etHFm,
                EvtPTracksNegEtaGap,        etHFm,                    EvtPlaneFromTracksMidEta, 
                EPTracksNeg3,               etHFm3,                   EPTracksMid3,
                EPTracksNeg4,               etHFm4,                   EPTracksMid4,
                EPTracksNeg5,               etHFm5,                   EPTracksMid5,
                EPTracksNeg6,               etHFm6,                   EPTracksMid6,
                etHFm,           etHFp };

const int RCMate2[] = {
                etHFp,                      EvtPTracksNegEtaGap,      EvtPTracksPosEtaGap,      
                etHFp3,                     EPTracksNeg3,             EPTracksPos3,               
                etHFp4,                     EPTracksNeg4,             EPTracksPos4,    
                etHFp5,                     EPTracksNeg5,             EPTracksPos5,    
                etHFp6,                     EPTracksNeg6,             EPTracksPos6,    
                etHFp,                      etHFp,                    etHFp,                    
                etHFp,                      etHFp,                    etHFp,
                EvtPTracksPosEtaGap,        EvtPlaneFromTracksMidEta, etHFp,
		EPTracksPos3,               EPTracksMid3,             etHFp3,
		EPTracksPos4,               EPTracksMid4,             etHFp4,
		EPTracksPos5,               EPTracksMid5,             etHFp5,
		EPTracksPos6,               EPTracksMid6,             etHFp6,      
                etEcalM,                    etEcalP };

const std::string EPNames[]={
                 "EvtPlaneFromTracksMidEta",    "EvtPTracksPosEtaGap",    "EvtPTracksNegEtaGap",        
		 "EPTracksMid3",                "EPTracksPos3",           "EPTracksNeg3",                
                 "EPTracksMid4",                "EPTracksPos4",           "EPTracksNeg4",    
                 "EPTracksMid5",                "EPTracksPos5",           "EPTracksNeg5",    
                 "EPTracksMid6",                "EPTracksPos6",           "EPTracksNeg6",    
                 "etEcal",                      "etEcalP",                "etEcalM",                
                 "etHcal",                      "etHcalP",                "etHcalM",
		 "etHF",                        "etHFp",                  "etHFm",                    
		 "etHF3",                        "etHFp3",                 "etHFm3",                    
		 "etHF4",                        "etHFp4",                 "etHFm4",                    
		 "etHF5",                        "etHFp5",                 "etHFm5",                    
		 "etHF6",                        "etHFp6",                 "etHFm6",                    
                 "etCaloHFP",                   "etCaloHFM"  
};


const int EPOrder[]={
                     2,2,2,
		     3,3,3,
		     4,4,4,
		     5,5,5,
		     6,6,6,
		     2,2,2,
		     2,2,2,
		     2,2,2,
		     3,3,3,
		     4,4,4,
		     5,5,5,
		     6,6,6,
		     2,2
};

const double EPEtaMin1[] = {
  -0.75,      1,    -2,   
  -0.75,      1,    -2,  
  -0.75,      1,    -2,    
  -0.75,      1,    -2,    
  -0.75,      1,    -2,    
  -2.7,     0.3,    -2.7,    
  -2.7,     0.3,    -2.7,
  -5.0,     3.0,    -5.0,
  -5.0,     3.0,    -5.0,
  -5.0,     3.0,    -5.0,
  -5.0,     3.0,    -5.0,
  -5.0,     3.0,    -5.0,
  0.25,    -5
};

const double EPEtaMax1[] = {
   0.75,      2,    -1,   
   0.75,      2,    -1,   
   0.75,      2,    -1,   
   0.75,      2,    -1,   
   0.75,      2,    -1,   
   2.7,     2.7,  -0.3,       
   2.7,     2.7,  -0.3,       
    -3,       5,    -3,       
    -3,       5,    -3,       
    -3,       5,    -3,       
    -3,       5,    -3,       
    -3,       5,    -3,       
     5,   -0.25   
};
const double EPEtaMin2[] = {
  0,0,0,
  0,0,0,
  0,0,0,
  0,0,0,
  0,0,0,
  0,0,0,
  0,0,0,
  3,0,0,
  3,0,0,
  3,0,0,
  3,0,0,
  3,0,0,
  0,0
};
const double EPEtaMax2[] = {
  0,0,0,
  0,0,0,
  0,0,0,
  0,0,0,
  0,0,0,
  0,0,0,
  0,0,0,
  5,0,0,
  5,0,0,
  5,0,0,
  5,0,0,
  5,0,0,
  0,0
};

static const int NumEPNames = 38;
}
