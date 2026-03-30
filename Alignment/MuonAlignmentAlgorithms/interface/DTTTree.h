#ifndef Alignment_MuonAlignmentAlgorithms_DTTTree_H
#define Alignment_MuonAlignmentAlgorithms_DTTTree_H
#define BADVAL -999.0

typedef struct DTLayerData {
  UChar_t wheel;
  UChar_t station;
  UChar_t sector;

  UInt_t nlayers;
  UInt_t nDT;
  UInt_t nCSC;
  UInt_t nTracker;

  Int_t charge;
  Int_t nEvent;

  Float_t pt;
  Float_t pz;
  Float_t eta;
  Float_t phi;

  Float_t v_hitx[8], v_hity[4];
  Float_t v_trackx[8], v_tracky[4], v_tracky_x_layer[8];

  Bool_t doFill;
  std::string cutType;

  DTLayerData() {
    charge = 0;
    wheel = 0;
    station = 0;
    sector = 0;
    nlayers = 0;
    nDT = 0;
    nCSC = 0;
    nTracker = 0;
    pt = BADVAL;
    pz = BADVAL;
    eta = BADVAL;
    phi = BADVAL;
    doFill = false;
    cutType = "";
  }

  DTLayerData& operator=(DTLayerData x) {
    charge = x.charge;
    wheel = x.wheel;
    station = x.station;
    sector = x.sector;
    nlayers = x.nlayers;
    nDT = x.nDT;
    nCSC = x.nCSC;
    nTracker = x.nTracker;
    pt = x.pt;
    pz = x.pz;
    eta = x.eta;
    phi = x.phi;
    doFill = x.doFill;
    cutType = x.cutType;
    return *this;
  }
} DTLayerData;

#endif
