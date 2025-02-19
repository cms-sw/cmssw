#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"

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
    std::string det;
    std::string rbx;
    int    wedge;
    int    rm;
    int    pixel;
    int    qie;
    int    adc;
    int    rm_fi;
    int    fi_ch;
    int    crate;
    int    htr;
    std::string fpga;
    int    htr_fi;
    int    dcc_sl;
    int    spigo;
    int    dcc;
    int    slb;
    std::string slbin;
    std::string slbin2;
    std::string slnam;
    int    rctcra;
    int    rctcar;
    int    rctcon;
    std::string rctnam;
    int    fedid;
  } LMapRowHBEF;

  typedef struct _LMapRowHO
  {
    int    sideO;
    int    etaO;
    int    phiO;
    int    dphiO;
    int    depthO;
    std::string detO;
    std::string rbxO;
    int    sectorO;
    int    rmO;
    int    pixelO;
    int    qieO;
    int    adcO;
    int    rm_fiO;
    int    fi_chO;
    std::string let_codeO;
    int    crateO;
    int    htrO;
    std::string fpgaO;
    int    htr_fiO;
    int    dcc_slO;
    int    spigoO;
    int    dccO;
    int    fedidO;
  } LMapRowHO;

  DBlmapWriter();
  ~DBlmapWriter();

  XMLDOMBlock * createLMapHBEFXMLBase( std::string templateFileName );
  XMLDOMBlock * createLMapHOXMLBase( std::string templateFileName );

  int addLMapHBEFDataset( XMLDOMBlock * doc, LMapRowHBEF * row, std::string templateFileName );
  int addLMapHODataset( XMLDOMBlock * doc, LMapRowHO * row, std::string templateFileName );

  int createLMap( void );

 protected:

 private:

};
