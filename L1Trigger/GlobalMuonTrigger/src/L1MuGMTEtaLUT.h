//-------------------------------------------------
//
//   Class: L1MuGMTEtaLUT
/**
 *   Description: Look-up table for GMT Eta projection unit
 *
 *                Caluclates float delta-eta from charge, eta and pT 
 *
 *                Simple static implementation with parametrization for
 *                CMS121 geometry
*/
//
//   $Date: 2007/03/23 18:51:35 $
//   $Revision: 1.2 $
//
//   Author :
//   H. Sakulin            CERN EP 
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTEtaLUT_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTEtaLUT_h

//---------------
// C++ Headers --
//---------------

#include <vector>

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuGMTEtaLUT {

  public:  

    /// constructor
    L1MuGMTEtaLUT();

    /// destructor
    virtual ~L1MuGMTEtaLUT();
    	
    /// look up delta-eta
    static float eta(int isys, int isISO, int icharge, float eta, float pt);

  private:
    static int etabin (float eta, int isys);

  private:
    static const int NSYS=4;
    static const int DT=0;
    static const int CSC=1;
    static const int bRPC=2;
    static const int fRPC=3;

    // 3-bit eta, in hardware 4th bit is reserved for
    // positive / negative endcap asymmetries
    static const unsigned int NETA=8;

    // 2 reference planes 0: calo, 1: vertex
    static const unsigned int NRP=2;  

    static float etabins[NSYS][NETA+1];
    static float fitparams_eta[NRP][NSYS][NETA][3];
};

#endif










