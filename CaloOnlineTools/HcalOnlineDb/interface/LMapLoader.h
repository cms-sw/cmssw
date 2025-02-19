#ifndef HCALConfigDBTools_XMLTools_LMapLoader_h
#define HCALConfigDBTools_XMLTools_LMapLoader_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     LMapLoader
// 
/**\class LMapLoader LMapLoader.h CaloOnlineTools/HcalOnlineDb/interface/LMapLoader.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Aram Avetisyan avetisya@fnal.gov
//         Created:  Tue Nov 14 15:05:33 CDT 2007
// $Id: LMapLoader.h,v 1.3 2010/08/06 20:24:10 wmtan Exp $
//

// system include files

// user include files
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"

// forward declarations

class LMapLoader : public XMLDOMBlock
{
  
 public:
  
  //structs
  
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

  LMapLoader();
  LMapLoader( std::string templateLoaderBase );
  virtual ~LMapLoader();

  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  int createLMapHBEFXMLBase( void );
  int addLMapHBEFDataset( LMapRowHBEF * row, std::string templateFileName );
  int addLMapHODataset( LMapRowHO * row, std::string templateFileName );
  
 private:
  LMapLoader(const LMapLoader&); // stop default
  
  const LMapLoader& operator=(const LMapLoader&); // stop default
  
  // ---------- member data --------------------------------
  
};


#endif
