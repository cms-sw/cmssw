//-------------------------------------------------
//
/**  \class DTTracoTrig
 *
 *   TRACO Trigger Data.
 *   Has pointers to parent TRACO and BTI triggers
 *   which allow algorithm debugging
 *
 *
 *   $Date: 2006/07/19 10:24:02 $
 *   $Revision: 1.1 $
 *
 *   \author C. Grandi, S. Vanini
 */
//
//--------------------------------------------------
#ifndef DT_TRACO_TRIG_H
#define DT_TRACO_TRIG_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTBtiTrigData;
class DTTracoChip;

//----------------------
// Base Class Headers --
//----------------------
#include "L1Trigger/DTUtilities/interface/DTTrigData.h"
#include "L1Trigger/DTBti/interface/DTBtiTrig.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrigData.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTracoTrig : public DTTrigData {

  public:

    /// Constructors
    DTTracoTrig();

    /// Constructors
    DTTracoTrig(DTTracoChip*, int);

    /// Constructors
    DTTracoTrig(DTTracoChip*, DTTracoTrigData) ;
  
    /// Destructor 
    ~DTTracoTrig();

    /// Set the parent TRACO
    inline void setParent(DTTracoChip* parent) { 
      _tparent = parent; 
    }  

    /// Add a BTI trigger to the list
    inline void addDTBtiTrig(const DTBtiTrigData* btitrig) {
      _btitrig.push_back(btitrig); 
    }

    /// Set trigger preview parameters
    inline void setPV(int first, int code, int K, int io) { 
      _data.setPV(first,code,K,io); 
    }
  
    /// Set trigger preview correlation bit
    inline void setPVCorr(int ic) { 
      _data.setPVCorr(ic); 
    }
  
    /// Set trigger code, inner segment
    inline void setCodeIn(int code) {
      _data.setCodeIn(code);
    }

    /// Set trigger code, outer segment
    inline void setCodeOut(int code) {
      _data.setCodeOut(code);
    }
  
    /// Set position of segments, inner
    inline void setPosIn(int pos) {
      _data.setPosIn(pos);
    }
  
    /// Set position of segments, outer
    inline void setPosOut(int pos) {
      _data.setPosOut(pos);
    }
 
    /// Set bti trigger equation of segments, inner
    inline void setEqIn(int eq) {
      _data.setEqIn(eq);
    }
  
    /// Set bti trigger equation of segments, outer
    inline void setEqOut(int eq) {
      _data.setEqOut(eq);
    }
 
    /// Set trigger K parameter
    inline void setK(int k) {
      _data.setK(k);
    }
  
    /// Set trigger X parameter
    inline void setX(int x) {
      _data.setX(x);
    }
  
    /// Set trigger angles
    inline void setAngles(int psi, int psir, int dpsir) {
      _data.setAngles(psi,psir,dpsir);
    }
  
    /// Reset all variables but preview
    inline void resetVar() {
      _data.resetVar();
    }
  
    /// Reset preview variables
    inline void resetPV() {
      _data.resetPV();
    }
  
    /// Clear
    inline void clear() {
      _data.clear();
      _btitrig.clear();
    }
  
    /// Return chamber identifier
    inline DTChamberId ChamberId() const {
      return _data.ChamberId(); 
    }

    /// Print
    inline void print() const { 
      _data.print();
    }

    /// Return parent TRACO pointer
    inline DTTracoChip* tParent() const { 
      return _tparent; 
    }
  
    /// Return the data part
    inline DTTracoTrigData data() const {
      return _data;
    }

    /// Return parent TRACO number
    inline int tracoNumber() const { 
      return _data.tracoNumber(); 
    }

    /// Return step
    inline int step() const { 
      return _data.step(); 
    }
  
    /// Return trigger code
    inline int code() const { 
      return _data.code(); 
    }
  
    /// Return correlator output code (position of segments)
    inline int posMask() const { 
      return _data.posMask(); 
    }
  
    /// Return the position of inner segment
    inline int posIn() const { 
      return _data.posIn(); 
    }
  
    /// Return the position of outer segment
    inline int posOut() const { 
      return _data.posOut(); 
    }

    /// Return bti trigger equation of segments, inner
    inline int eqIn() {
      return _data.eqIn();
    }
  
    /// Return bti trigger equation of segments, outer
    inline int eqOut() {
      return _data.eqOut();
    }
 
    /// Return non 0 if the track is a first track
    inline int isFirst() const {
      return _data.isFirst();
    }

    /// Return the preview code
    inline int pvCode() const { 
      return _data.pvCode(); 
    }
  
    /// Return the preview K
    inline int pvK() const { 
      return _data.pvK(); 
    }
  
    /// Return the preview correaltion bit
    inline int pvCorr() const { 
      return _data.pvCorr(); 
    }
  
    /// Return trigger K parameter
    inline int K() const { 
      return _data.K(); 
    }
  
    /// Return trigger X parameter
    inline int X() const { 
      return _data.X(); 
    }
  
    /// Return trigger K parameter converted to angle
    inline int psi() const { 
      return _data.psi(); 
    }
  
    /// Return trigger X parameter converted to angle
    inline int psiR() const { 
      return _data.psiR(); 
    }
  
    /// Return DeltaPsiR
    inline int DeltaPsiR() const {
      return _data.DeltaPsiR(); 
    }
  
    /// Return the trigger code in new format
    inline int qdec() const { 
      return _data.qdec(); 
    }

    /// Return the BTI Triggers list
      std::vector<const DTBtiTrigData*> btiTrigList() const { 
      return _btitrig; 
    }

    /// comparison operator
    bool operator == (const DTTracoTrig &) const;


  private:

    // parent TRACO
    DTTracoChip* _tparent;

    // Trigger data component
    DTTracoTrigData _data;

    // vector of BTI triggers which produce the DTTracoChip trigger (1 or 2)
	std::vector<const DTBtiTrigData*> _btitrig;

};

#endif
