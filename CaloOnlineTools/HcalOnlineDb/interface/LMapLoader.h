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
// $Id: LMapLoader.h,v 1.1 2007/11/14 23:32:17 avetisya Exp $
//

// system include files

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"

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

  LMapLoader();
  LMapLoader( string templateLoaderBase );
  virtual ~LMapLoader();

  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  int createLMapHBEFXMLBase( void );
  int addLMapHBEFDataset( LMapRowHBEF * row, string templateFileName );
  int addLMapHODataset( LMapRowHO * row, string templateFileName );
  
 private:
  LMapLoader(const LMapLoader&); // stop default
  
  const LMapLoader& operator=(const LMapLoader&); // stop default
  
  // ---------- member data --------------------------------
  
};


#endif
