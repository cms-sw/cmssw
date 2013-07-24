//-------------------------------------------------
//
/**  \class DTTSM
 *    Implementation of TSM trigger algorithm
 *
 *
 *   $Date: 2008/09/05 15:59:57 $
 *   $Revision: 1.3 $
 *
 *   \author C. Grandi, D. Bonacorsi, S. Marcellini
 */
//
//--------------------------------------------------
#ifndef MU_DT_TSM_H
#define MU_DT_TSM_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTTracoTrigData;
class DTTSCand;
class DTConfigTSPhi;
// added DBSM
class DTTrigGeom;
//----------------------
// Base Class Headers --
//----------------------
// added DBSM
#include "L1Trigger/DTUtilities/interface/DTGeomSupplier.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTSM {

  public:

    ///  Constructor
    // SM double TSM 
    DTTSM(int);   

    /// Destructor 
    ~DTTSM();
  
    /// Return identifier
    inline int number() const { return _n; }

    /// Set configuration
    void setConfig(DTConfigTSPhi *config) {  _config=config; }

    /// Add a TSS candidate to the TSM, ifs is first/second track flag
    void addCand(DTTSCand* cand);

    /// Set a flag to skip sort2
    void ignoreSecondTrack() { _ignoreSecondTrack=1; }

    /// Run the TSM algorithm
    void run(int bkmod);

    /// Sort 1
    // added DBSM
    DTTSCand* sortTSM1(int bkmod);

    /// Sort 2
    DTTSCand* sortTSM2(int bkmod);

    /// Clear
    void clear();

    /// Configuration set
    inline DTConfigTSPhi* config() const { return _config; }

    /// Return the number of input tracks (first/second)
    unsigned nCand(int ifs) const;

    /// Return the number of input first tracks
    inline int nFirstT() const { return _incand[0].size(); }

    /// Return the number of input second tracks
    inline int nSecondT() const { return _incand[1].size(); }

    /// Return requested TS candidate
    DTTSCand* getDTTSCand(int ifs, unsigned n) const;

    /// Return requested TRACO trigger
    const DTTracoTrigData* getTracoT(int ifs, unsigned n) const;

    /// Return the number of sorted tracks
    inline int nTracks() const { return _outcand.size(); }

    /// Return the requested track
    DTTSCand* getTrack(int n) const;
 
  private:

    DTConfigTSPhi* _config;


    // SM double TSM
    // identification (as for DTTSS.h)
    int _n;

    // input data
    std::vector<DTTSCand*> _incand[2];

    // output data
    std::vector<DTTSCand*> _outcand;

    // internal use variables
    int _ignoreSecondTrack;

};
#endif











