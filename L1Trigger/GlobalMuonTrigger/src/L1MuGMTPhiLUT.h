//-------------------------------------------------
//
//   Class: L1MuGMTPhiLUT
/**
 *   Description: Look-up table for GMT Phi projection unit
 *
 *                Caluclates float delta-phi from charge, eta and pT 
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
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTPhiLUT_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTPhiLUT_h

//---------------
// C++ Headers --
//---------------

#include <vector>
#include <cmath>

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuGMTPhiLUT {

  public:  

    /// constructor
    L1MuGMTPhiLUT();

    /// destructor
    virtual ~L1MuGMTPhiLUT();
    	
    //FIXME: two versions

    /// look up delta-phi with integer eta
    static float dphi(int isys, int isISO, int icharge, int ieta, float pt) ;

    /// look up delta-phi
    static float dphi(int isys, int isISO, int icharge, float eta, float pt){
      return dphi(isys, isISO, icharge, etabin( (float)fabs(eta), isys), pt); 
    };

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
    static float fitparams_phi[NRP][NSYS][NETA][2][3];
};

#endif
