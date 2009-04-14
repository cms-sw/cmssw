#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"

using namespace std;

class DBlmapWriter
{
  
 public:
  typedef struct _LMapRowHBEF
  {
    int    side;
    int    eta;
    int    phi;
    int    dphi;
    int    depth;
    string det;
    string rbx;
    int    wedge;
    int    rm;
    int    pixel;
    int    qie;
    int    adc;
    int    rm_fi;
    int    fi_ch;
    int    crate;
    int    htr;
    string fpga;
    int    htr_fi;
    int    dcc_sl;
    int    spigo;
    int    dcc;
    int    slb;
    string slbin;
    string slbin2;
    string slnam;
    int    rctcra;
    int    rctcar;
    int    rctcon;
    string rctnam;
    int    fedid;
  } LMapRowHBEF;

  typedef struct _LMapRowHO
  {
    int    sideO;
    int    etaO;
    int    phiO;
    int    dphiO;
    int    depthO;
    string detO;
    string rbxO;
    int    sectorO;
    int    rmO;
    int    pixelO;
    int    qieO;
    int    adcO;
    int    rm_fiO;
    int    fi_chO;
    string let_codeO;
    int    crateO;
    int    htrO;
    string fpgaO;
    int    htr_fiO;
    int    dcc_slO;
    int    spigoO;
    int    dccO;
    int    fedidO;
  } LMapRowHO;

  DBlmapWriter();
  ~DBlmapWriter();

  XMLDOMBlock * createLMapHBEFXMLBase( string templateFileName );
  XMLDOMBlock * createLMapHOXMLBase( string templateFileName );

  int addLMapHBEFDataset( XMLDOMBlock * doc, LMapRowHBEF * row, string templateFileName );
  int addLMapHODataset( XMLDOMBlock * doc, LMapRowHO * row, string templateFileName );

  int createLMap( void );

 protected:

 private:

};
