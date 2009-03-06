#ifndef BtagResult_h
#define BtagResult_h

class BtagResult {
 public:
  enum BtagResultType {BEFF=1, BERR=2, CEFF=3, 
		       CERR=4, LEFF=5, LERR=6, NBEFF=7, NBERR=8,
		       //
		       // add corrections in case the table is for weights and not efficiencies
		       //
		       BEFFCORR=9, BERRCORR=10, CEFFCORR=11, 
		       CERRCORR=12, LEFFCORR=13, LERRCORR=14, NBEFFCORR=15, NBERRCORR=16};
  
};

#endif
