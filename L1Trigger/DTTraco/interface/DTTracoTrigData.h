//-------------------------------------------------
//
/**  \class DTTracoTrigData
 *
 *   DTTracoChip Trigger Data
 *
 *
 *   $Date: 2008/06/30 13:42:21 $
 *   $Revision: 1.3 $
 *
 *   \author  C. Grandi
 */
//
//--------------------------------------------------
#ifndef DT_TRACO_TRIG_DATA_H
#define DT_TRACO_TRIG_DATA_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfig.h"
#include "DataFormats/MuonDetId/interface/DTTracoId.h"

//----------------------
// Base Class Headers --
//----------------------
#include "L1Trigger/DTUtilities/interface/DTTrigData.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef unsigned char myint8;

class DTTracoTrigData : public DTTrigData {

 public:

  // public methods

  //!  Constructor
  DTTracoTrigData() {}

  //!  Constructor
  DTTracoTrigData(DTTracoId, int);
  
  //!  Destructor 
  ~DTTracoTrigData() {}

  // Non-const methods
  
  //! Set the parent TRACO Identifier
  inline void setParent(DTTracoId tracoid) { 
    _tracoid = tracoid; 
  }

  //! Set trigger step
  inline void setStep(int step) {
    _step = step;
  }

  //! Set trigger preview parameters
  inline void setPV(int first, int code, int K, int ioflag) {
    _pvfirst = first; 
    _pvcode = code;
    _pvKval = K; 
    _pvIOflag = ioflag;
/*
    cout<<"setPV called, stored:"<<
    "  first=" << first <<
    "  code=" << code <<
    "  K=" << K <<
    " ioflag=" << ioflag << endl;
*/
   }

  //! Set trigger preview correlation bit
  inline void setPVCorr(int ic) {
    _pvCorr = ic; 
  }

  //! Set trigger code inner layer
  inline void setCodeIn(int code) {
    _codeIn = code;
  }
  //! Set trigger code outer layer
  inline void setCodeOut(int code) {
    _codeOut = code;
  }

  //! Set position of segment, inner layer
  inline void setPosIn(int pos) {
    _posIn = pos;
  }

  //! Set position of segment, outer layer
  inline void setPosOut(int pos) {
    _posOut = pos;
  }

  //! Set bti trigger equation of segment, inner layer
  inline void setEqIn(int eq) {
    _eqIn = eq;
  }

  //! Set bti trigger equation of segment, outer layer
  inline void setEqOut(int eq) {
    _eqOut = eq;
  }


  //! Set trigger K parameter
  inline void setK(int k) {
    _Kval = k;
  }

  //! Set trigger X parameter
  inline void setX(int x) {
    _Xval = x;
  }

  //! Set trigger angles
  inline void setAngles(int psi, int psir, int dpsir) { 
    if(psi & 0x200)
      psi |= 0xFFFFFC00;
    if(psir & 0x800)
      psir |= 0xFFFFF000;
    if(dpsir & 0x200)
      dpsir |= 0xFFFFFC00;

    _psi = psi;
    _psiR = psir;
    _dPsiR = dpsir; 
  }

  //! Reset all variables but preview
  void resetVar() {
    _codeIn = 0; 
    _codeOut = 0; 
    _posIn = 0;
    _posOut = 0;
    _eqIn = 0;
    _eqOut = 0;
    _Kval = 255; 
    _Xval = 0;
    /*
    _psi = -DTConfig::RESOLPSI;
    _psiR = -DTConfig::RESOLPSIR/2;
    _dPsiR = -DTConfig::RESOLPSI;
    */
    //SV notazione complemento a due:
    _psi = 0x1FF;
    _psiR = 0xFFF;
    _dPsiR = 0x1FF;

  }
  
  //! Reset preview variables
  void resetPV() {
    _pvfirst = 0;
    _pvcode = 0;
    _pvKval = 9999;
    _pvCorr = 0;
    _pvIOflag = 0;
  }
  
  //! Clear
  void clear() {
    resetVar();
    resetPV();
  }
  
  //! Return chamber identifier
  DTChamberId ChamberId() const {
    return _tracoid.ChamberId(); 
  }

  //! print
  void print() const;

  //! Return parent TRACO identifier
  inline DTTracoId parentId() const {
    return _tracoid; 
  }

  //! Return parent TRACO number
  inline int tracoNumber() const { 
    return _tracoid.traco(); 
  }

  //! Return step
  inline int step() const { 
    return _step; 
  }

  //! Return trigger code
  inline int code() const { 
    return _codeIn*10 + _codeOut; 
  }
  
  //! Return correlator output code (position of segments)
  inline int posMask() const { 
    return _posOut*1000 + _posIn; 
  }
  
  //! Return the position of inner segment
  inline int posIn() const { 
    return _posIn; 
  }

  //! Return the position of outer segment
  inline int posOut() const { 
    return _posOut; 
  }

  //! Return bti trigger equation of inner segment
  inline int eqIn() const { 
    return _eqIn; 
  }

  //! Return bti trigger equation of outer segment
  inline int eqOut() const { 
    return _eqOut; 
  }


  //! Return non 0 if the track is a first track
  inline int isFirst() const {
    return _pvfirst;
  }

  //! Return the preview code
  inline int pvCode() const { 
    return _pvcode; 
  }

  //! Return the preview K
  inline int pvK() const { 
    return _pvKval; 
  }

  //! Return the preview correaltion bit
  inline int pvCorr() const { 
    return _pvCorr; 
  }

  //! Return the preview i/o bit
  inline int pvIO() const { 
    return _pvIOflag; 
  }


  //! Return trigger K parameter
  inline int K() const { 
    return _Kval; 
  }

  //! Return trigger X parameter
  inline int X() const { 
    return _Xval; 
  }

  //! Return trigger K parameter converted to angle
  int psi() const {
    return _psi; 
  }

  //! Return trigger X parameter converted to angle
  int psiR() const {
    return _psiR; 
  }

  //! Return DeltaPsiR
  int DeltaPsiR() const {
    return _dPsiR; 
  }

  //! Return the trigger code in new format
  int qdec() const;

  private:

  // parent TRACO
  DTTracoId _tracoid; // this is 4 bytes

  // step number
  myint8 _step;

  // inner segment position
  myint8 _posIn;
  // outer segment position
  myint8 _posOut;
  // inner segment bti trigger equation
  myint8 _eqIn;
  // outer segment bti trigger equation
  myint8 _eqOut;

  // inner segment code
  myint8 _codeIn;
  // outer segment code
  myint8 _codeOut;

  // preview first/second track
  myint8 _pvfirst;

  // preview code for TS
  myint8 _pvcode;
  // preview K variable for TS
  int _pvKval;
  // preview correlation bit
  myint8 _pvCorr;
  // preview io bit
  myint8 _pvIOflag;


  // K variable value
  int _Kval;
  // X variable value
  int _Xval;

  // K converted to angle (psi)
  long int _psi;
  // X converted to angle (psi_r, phi)
  long int _psiR;
  // bending angle (delta(psi_r), phiB)
  long int _dPsiR;

};

#endif
