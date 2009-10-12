#ifndef PerformanceResult_h
#define PerformanceResult_h

class PerformanceResult {
 public:
  enum ResultType {
    //
    // BTAG
    //
    BTAGBEFF=1001, BTAGBERR=1002, BTAGCEFF=1003, 
    BTAGCERR=1004, BTAGLEFF=1005, BTAGLERR=1006, BTAGNBEFF=1007, BTAGNBERR=1008,
    //
    // add corrections in case the table is for weights and not efficiencies
    //
    BTAGBEFFCORR=1009, BTAGBERRCORR=1010, BTAGCEFFCORR=1011, 
    BTAGCERRCORR=1012, BTAGLEFFCORR=1013, BTAGLERRCORR=1014, BTAGNBEFFCORR=1015, BTAGNBERRCORR=1016,
    //
    // MUONS
    //
    MUEFF=2001, MUERR=2002, MUFAKE=2003, MUEFAKE=2004
};
  
};

#endif
