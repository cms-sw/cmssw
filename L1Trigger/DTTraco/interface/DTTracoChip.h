//-------------------------------------------------
//
/**  \class DTTracoChip
 *
 *   Implementation of TRACO trigger algorithm.
 *   Internally uses DTTracoCand to store BTI triggers
 * 
 * 
 *   $Date: 2008/06/30 13:42:21 $
 *   $Revision: 1.6 $
 * 
 *   \author S. Vanini
 */
//
//--------------------------------------------------
#ifndef DT_TRACO_CHIP_H
#define DT_TRACO_CHIP_H

//-------------------
// Constants file  --
//-------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfig.h"
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h" 
#include "L1Trigger/DTTraco/interface/Lut.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTTracoCard;
class DTBtiTrigData;
class DTTracoCand;
class DTTracoTrig;
class DTTracoTrigData;

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/MuonDetId/interface/DTTracoId.h"
#include "L1Trigger/DTTraco/interface/DTTracoLUTs.h"
#include "L1Trigger/DTUtilities/interface/DTTrigGeom.h"
#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTraco.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTracoChip {

  public:

    /// Constructor
    //DTTracoChip(DTTracoCard* card, int n);

    /// Constructor for passing configuration
    DTTracoChip(DTTracoCard* card, int n, DTConfigTraco* config);

    /// Copy constructor
    DTTracoChip(const DTTracoChip& traco);
  
    /// Destructor 
    ~DTTracoChip();

    /// Assignment operator
    DTTracoChip& operator=(const DTTracoChip& traco);

    /// Add a BTI trigger to the TRACO
    void add_btiT(int step, int pos, const DTBtiTrigData* btitrig);

    /// Add a TRACO trigger
    void addTrig(int step, DTTracoTrig*);

    /// Set the preview values for a TRACO trigger
    void setPV(int step, int ntrk, int code, int K);

    /// Calculate trigger angles
    void calculateAngles(DTTracoTrig*);

    /// Run TRACO algorithm
    void run();

    /// Clear
    void clear();

    /// Return TRACO number
    inline int number() const { return _id.traco(); }

    /// raise overlap flag
    void raiseOverlap(int step);

    /// Return trigger geometry
    inline DTTrigGeom* geom() const { return _geom; }

    /// Return TRACO id
    inline DTTracoId id() const { return _id; }

    /// Return wheel number
    inline int wheel() const { return _geom->wheel(); }

    /// Return station number
    inline int station() const { return _geom->station(); }

    /// Return sector number
    inline int sector() const { return _geom->sector(); }

    /// Configuration set
    //inline DTConfig* config() const { return _geom->config(); }

    /// New Configuration set
    inline DTConfigTraco* config() const { return _config; }

    /// Radial angle of correlator center in mrad referred to plane sl
    float psiRad(int sl=0) const;

    /// K par of the radial angle of corr center referred to plane sl
    int KRad() const;
    //int KRad(int sl=0) const;

    /// BTIC parameter 
    int BTIC() const { return _btic; }
  
    /// IBTIOFF parameter
    int IBTIOFF() const { return _ibtioff; }

    /// DD parameter
    int DD() const { return _dd; }

    /// Return the number of trigger candidates
    int nTrig(int step) const;

    /// Return the requested trigger
    DTTracoTrig* trigger(int step, unsigned n) const;

    /// Return the data part of the requested trigger
    DTTracoTrigData triggerData(int step, unsigned n) const;

    /// a flag for a usable second track
    int useSecondTrack(int step) const;

    /// flags for HTRIG in edge BTI
    int edgeBTI(int step, int io, int lr) const;

    /// Position in chamber frame
    LocalPoint localPosition() const { return _geom->localPosition(_id); }

    /// Position in CMS frame
    GlobalPoint CMSPosition() const { return _geom->CMSPosition(_id); }

    /// Set flags for multiple trigger detection between cons. TRACO's
    void setFlag(int step,int ext=0);

    /// return overlap flag
    inline int ovlFlag(int step) {
      return _flag[step-DTConfigTraco::NSTEPF].element(1);}

  private:

    /// Get the best inner/outer candidate
    DTTracoCand* bestCand(int itk, std::vector<DTTracoCand> & tclist);

    /// Set the preview for a trigger
    DTTracoTrig* setPV(int itk, DTTracoCand* inner, DTTracoCand* outer);

    /// Do suppression of LTRIG on BTI close to selected HTRIG
    void DoAdjBtiLts(DTTracoCand* candidate, std::vector<DTTracoCand> & tclist);

    /// Do suppression of LTRIG on adjacent TRACO
    int AdjBtiLTSuppressed(DTTracoCand* candidate);

    /// Check correlation and store correlated trigger
    int storeCorr(DTTracoTrig* tctrig, DTTracoCand* inner, DTTracoCand* outer, int tkn);

    /// Store uncorrelated trigger
    int storeUncorr(DTTracoTrig* tctrig, DTTracoCand* inner, DTTracoCand* outer, int tkn);

    /// Check if a trigger is inside the angular acceptance window
    int insideAngWindow(DTTracoTrig* ) const;

    /// Compute traco chip acceptances
    void setTracoAcceptances();

  private:
    // identification
    DTTrigGeom* _geom;
    DTTracoId _id;
    // parent card
    DTTracoCard* _card;
    //config
    DTConfigTraco* _config;

    int _krad;
    int _btic;
    int _ibtioff;
    int _dd;

    // input data
    std::vector<DTTracoCand> _innerCand[DTConfigTraco::NSTEPL-DTConfigTraco::NSTEPF+1];
    std::vector<DTTracoCand> _outerCand[DTConfigTraco::NSTEPL-DTConfigTraco::NSTEPF+1];

    // output data
    std::vector<DTTracoTrig*> _tracotrig[DTConfigTraco::NSTEPL-DTConfigTraco::NSTEPF+1];

    // internal use variables: SV 11V04 lts suppression if to bx+1, bx=array index
    BitArray<DTConfigTraco::NSTEPL+2> _bxlts;
    // *** FOR TESTBEAM DATA  *** SV from input data instead from card! 
    // 1: overlap with II track from bx-1)
    // 2: II track rej.
    // 3...8: IL,IR,OL,OR,th,th
    // 9: H present in traco at bx
    BitArray<32> _flag[DTConfigTraco::NSTEPL-DTConfigTraco::NSTEPF+1]; 

    // psi acceptance of correlator MT ports
    int _PSIMIN[4*DTConfig::NBTITC];
    int _PSIMAX[4*DTConfig::NBTITC];

    // LUT file class
    DTTracoLUTs* _luts;
    
    Lut* _lutsCCB;

};

#endif
