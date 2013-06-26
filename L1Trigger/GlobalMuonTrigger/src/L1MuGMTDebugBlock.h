//-------------------------------------------------
//
//   \class L1MuGMTDebugBlock
/**
 *   Description: debug block for GMT
 *                it is filled during GMT processing
 *                and allows to retrieve intermediate 
 *                results, later (e.g. for comparison
 *                with the hardware model)
*/
//
//   $Date: 2013/04/22 17:03:01 $
//   $Revision: 1.5 $
//
//   Author :
//   H. Sakulin            HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTDebugBlock_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTDebugBlock_h

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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatrix.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuGMTDebugBlock {

  public:  
    static const int NumMatrices = 6;

    /// constructor
    L1MuGMTDebugBlock(int minbx=-10, int maxbx=10);

    /// destructor
    virtual ~L1MuGMTDebugBlock();

    /// Reset the debug block
    void reset ();

    /// Set the current bunch crossing
    void SetBX(int bx) { 
      if (bx < _minbx || bx > _maxbx) edm::LogWarning("RangeViolation") 
                                    << "L1MuGMTDebugBlock::SetBX(): bx out of range ";
      else _bx=bx;
    };

    // index: (0..31: 16*isFWD + 8*isISO + 4* isRPC + nr )

    /// Set projected phi positions for current bx
    void SetPhi(int idx, float phi) { _prophi[_bx - _minbx][idx]=phi; };

    /// Set projected eta positions for current bx
    void SetEta(int idx, float eta) { _proeta[_bx - _minbx][idx]=eta; };

    /// Set phi select bits for current bx
    void SetPhiSelBits(int idx, unsigned phisel) { _phisel[_bx - _minbx][idx]=phisel; };

    /// Set eta select bits for current bx
    void SetEtaSelBits(int idx, unsigned etasel) { _etasel[_bx - _minbx][idx]=etasel; };

    /// Set MIP/ISO bits for current bx
    void SetIsMIPISO(int idx, unsigned ismipiso) { _isMIPISO[_bx - _minbx][idx]=ismipiso; };

    /// Set pair matrices
    void SetPairMatrix(int idx, L1MuGMTMatrix<bool> pm) { _pairMatrices[_bx - _minbx][idx]=pm; };

    /// Set match quality matrices
    void SetMQMatrix(int idx, L1MuGMTMatrix<int> mqm) { _mqMatrices[_bx - _minbx][idx]=mqm; };

    /// Set cancel bits
    void SetCancelBits (int idx, const std::vector<bool>& mine, const std::vector<bool>& others);

    /// Set brl GMT Cands
    void SetBrlGMTCands (int idx, L1MuGMTExtendedCand const& cand) { _brlmuons[_bx - _minbx][idx]=cand; };

    /// Set fwd GMT Cands
    void SetFwdGMTCands (int idx, L1MuGMTExtendedCand const& cand) { _fwdmuons[_bx - _minbx][idx]=cand; };


    /// Get stored phi position
    float Phi(int bx, int idx) { return _prophi[bx - _minbx][idx]; };

    /// Get stored eta position
    float Eta(int bx, int idx) { return _proeta[bx - _minbx][idx]; };

    /// Get stored phi select bits
    unsigned PhiSel(int bx, int idx) { return _phisel[bx - _minbx][idx]; };

    /// Get stored eta select bits
    unsigned EtaSel(int bx, int idx) { return _etasel[bx - _minbx][idx]; };
    	
    /// Get stored MIP/ISO select bits
    unsigned IsMIPISO(int bx, int idx) { return _isMIPISO[bx - _minbx][idx]; };

    /// Get pair matrices
    L1MuGMTMatrix<bool> GetPairMatrix(int bx, int idx) { return _pairMatrices[bx - _minbx][idx]; };

    /// Get match quality matrices
    L1MuGMTMatrix<int> GetMQMatrix(int bx, int idx) { return _mqMatrices[bx - _minbx][idx]; };

    /// Get Cancel Bits
    unsigned GetCancelBits(int bx, int idx) { return _cancelbits[bx - _minbx][idx]; };

    /// Get brl Cands
    L1MuGMTExtendedCand const& GetBrlGMTCand(int bx, int idx) { return _brlmuons[bx - _minbx][idx];}

    /// Get fwd Cands
    L1MuGMTExtendedCand const& GetFwdGMTCand(int bx, int idx) { return _fwdmuons[bx - _minbx][idx];}

    /// Get stored phi position, four indices
    float Phi(int bx, int isFWD, int isISO, int isRPC, int nr) { 
      return _prophi[bx - _minbx][16*isFWD + 8*isISO + 4*isRPC + nr]; 
    };

    /// Get stored eta position, four indices
    float Eta(int bx, int isFWD, int isISO, int isRPC, int nr) { 
      return _proeta[bx - _minbx][16*isFWD + 8*isISO + 4*isRPC + nr];
    };
    	
    

  private:
    const int _minbx, _maxbx;
    int _bx;
    std::vector<std::vector<float> > _prophi;
    std::vector<std::vector<float> > _proeta;
    std::vector<std::vector<unsigned> > _phisel;
    std::vector<std::vector<unsigned> > _etasel;
    std::vector<std::vector<unsigned> > _isMIPISO;

    std::vector<std::vector<L1MuGMTMatrix<bool> > > _pairMatrices;
    std::vector<std::vector<L1MuGMTMatrix<int> > > _mqMatrices;

    std::vector<std::vector<unsigned> > _cancelbits;
    std::vector<std::vector<L1MuGMTExtendedCand> > _brlmuons;
    std::vector<std::vector<L1MuGMTExtendedCand> > _fwdmuons;
};

#endif









