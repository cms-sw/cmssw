#ifndef DQMOFFLINE_TRIGGER_EGHLTBINDATA
#define DQMOFFLINE_TRIGGER_EGHLTBINDATA

//this unsurprisingly stores the histogram bin values as read in from the config file

namespace edm {
  class ParameterSet;
}

namespace egHLT {
  
  struct BinData {
    
    struct Data1D {
      int nr;
      double min;
      double max;
      void setup(const edm::ParameterSet& conf);
    }; 
    struct Data2D {
      int nrX;
      double xMin;
      double xMax;
      int nrY;
      double yMin;
      double yMax;
      void setup(const edm::ParameterSet& conf);
    };


    BinData(){}
    explicit BinData(const edm::ParameterSet& conf){setup(conf);}
    void setup(const edm::ParameterSet& conf);
    
    Data1D energy;
    Data1D et;
    Data1D etHigh;
    Data1D eta;
    Data1D phi;
    Data1D charge;
    Data1D hOverE;
    Data1D dPhiIn;
    Data1D dEtaIn;
    Data1D sigEtaEta;
    Data1D e2x5;
    Data1D e1x5;
    //----Morse----
    //Data1D r9;
    Data1D minr9;
    Data1D maxr9;
    Data1D nVertex;
    Data1D HLTenergy;
    Data1D HLTphi;
    Data1D HLTeta;
    Data1D deltaE;
    //-----------
    Data1D isolEm;
    Data1D isolHad;
    Data1D isolPtTrks;
    Data1D isolNrTrks;
    Data1D mass;
    Data1D massHigh;  
    Data1D eOverP;
    Data1D invEInvP;

    Data2D etaVsPhi;
  };

}

#endif
