//-------------------------------------------------
//
/**  \class DTBtiChip
 *
 *   Implementation of DTBtiChip trigger algorithm
 *   Internally uses DTBtiHit to store muon digis
 *
 *
 *   $Date: 2010/01/21 10:22:12 $
 *   $Revision: 1.8 $
 *
 *   \author S. Vanini
 */
//
//--------------------------------------------------
#ifndef DT_BTI_CHIP_H
#define DT_BTI_CHIP_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTBtiHit;
class DTBtiTrig;
class DTBtiTrigData;
class DTDigi;

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/MuonDetId/interface/DTBtiId.h"
#include "L1Trigger/DTUtilities/interface/DTTrigGeom.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfig.h"
#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigBti.h"
#include "L1Trigger/DTBti/interface/DTBtiCard.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTBtiChip {

  public:

  //! original constructor 
  //DTBtiChip(DTTrigGeom* geom, int supl, int n);

  //! new constructor with configuration 
  DTBtiChip(DTBtiCard* card, DTTrigGeom* geom, int supl, int n, DTConfigBti* _config );

  //! Copy constructor
  DTBtiChip(const DTBtiChip& bti);

  //! Destructor 
  ~DTBtiChip();

  //! Assignment operator
  DTBtiChip& operator=(const DTBtiChip& bti);

  //! Add a digi to the DTBtiChip
  void add_digi(int cell, const DTDigi* digi);

  //! Add a clock digi to the DTBtiChip
  void add_digi_clock(int cell, int clock_digi);

   //! get digi vector - SV 28/XI/02
  std::vector<const DTDigi*> get_CellDigis(int cell) { return _digis[cell];}

  //! Run DTBtiChip algorithm
  void run();

  //! delete hits and triggers
  void clear();

  //! Add a DTBtiChip trigger
  //! (normally used by DTBtiChip itself - may be used for debugging by other classes)
  void addTrig(int step, DTBtiTrig* btitrig);

  // Public const methods

  //! Return DTBtiChip number
  inline int number() const { return _id.bti(); }

  //! Return superlayer
  inline int superlayer() const { return _id.superlayer(); }

  //! Position in chamber frame (x is the one of first traco in slave plane)
  inline LocalPoint localPosition() const { return _geom->localPosition(_id); }

  //! Position in CMS frame
  inline GlobalPoint CMSPosition() const { return _geom->CMSPosition(_id); }

  //! Number of cells with hits
  int nCellHit() const;

  //! Number of triggers found
  int nTrig(int step) const;

  //  // Return the trigger vector
  std::vector<DTBtiTrig*> trigList(int step) const;

  //! Return the requested trigger
  DTBtiTrig* trigger(int step, unsigned n) const;

  //! Return the data part of the requested trigger
  DTBtiTrigData triggerData(int step, unsigned n) const;

  //! Configuration set
  //inline DTConfig* config() const { return _geom->config(); }

  //! testing DTConfigBti
  inline DTConfigBti* config() const { return _config; }


  //! Return trigger geometry
  inline DTTrigGeom* geom() const { return _geom; }

  //! Return the DTBtiChip Id
  inline DTBtiId id() const { return _id; }

  //! Return wheel number
  inline int wheel() const { return _geom->wheel(); }

  //! Return station number
  inline int station() const { return _geom->station(); }

  //! Return sector number
  inline int sector() const { return _geom->sector(); }

  void init_clock();    // initialization from clocks



 private:

  void init(); // initialization
  void tick(); // next step (80 MHz)
  // return current step (40MHz)
  inline int currentStep() const { return (int)(((float)(_curStep)+0.5)/2); }
  inline int currentIntStep() const { return _curStep; } // int. step (80MHz)
  void computeSums();                                 // compute sums and diffs
  void sum(const int s, const int a, const int b);    //   "     a sum and dif
  void computeEqs();                                  // compute X and K equat.
  void findTrig();                                    // find triggers
  int keepTrig(const int eq, const int acp, const int code); // find  a trigger
  int keepTrigPatt(int flag, const int eq, const int pattType, int hlflag);//SV
  bool matchEq(float eqA, float eqB, int AC);
  void acceptMask(BitArray<80> * BitArrPtr,int k,int accep);   
  void doLTS();                                   // adjacent LTRIG suppression
  int store(const int eq, const int code, const int K, const int X, 
             float KeqAB=0., float KeqBC=0., float KeqCD=0., 
             float KeqAC=0., float KeqBD=0., float KeqAD=0.); 
  void eraseTrigger(int step, unsigned n); // Erase the requested trigger
  void setSnap();
  void reSumSet(); //remainder table compute
  int reSum(int a, int b) { return reSumAr[a][b+2];}
  int reSum23(int a, int b) { return reSumAr23[a][b+2];}

 private:

  // parent card
  DTBtiCard* _card;

  DTTrigGeom*  _geom;
  DTConfigBti* _config;

  DTBtiId _id;

  // input data from DTDigis
  std::vector<const DTDigi*> _digis[9];
  // input data from clock digis
  std::vector<int > _digis_clock[9];

  // output data (ordered by step number)
  std::vector<DTBtiTrig*> _trigs[ DTConfig::NSTEPL - DTConfig::NSTEPF+1 ];

  // internal use variables
  int _curStep;                      // current step
  std::vector<DTBtiHit*> _hits[9];    // current hits in cells
  int _thisStepUsedTimes[9];         // current used times in cells (JTRIG)
  DTBtiHit* _thisStepUsedHit[9]; // link to currently used hits
  int _nStepUsedHits;                // number of currently used hits
  float _sums[25], _difs[25];        // time sums and differences 
  float _Keq[32][6];                 // The K equations
  float _Xeq[32][2];                 // The X equations
  int _MinKAcc;                      // min K value accepted by DTBtiChip 
  int _MaxKAcc;                      // max K value accepted by DTBtiChip
  int _MinKleftTraco;		     // K limits for left traco
  int _MaxKleftTraco;
  int _MinKcenterTraco;              // K limits for center traco
  int _MaxKcenterTraco;
  int _MinKrightTraco;               // K limits for right traco
  int _MaxKrightTraco;

  float _XeqAC_patt0, _XeqBD_patt0;  // special pattern 0 X equations
  float _XeqAB_patt0, _XeqCD_patt0;

  float _KTR[32][2];                 //
  float _JTR[32][3];                 //
  int init_done;                     // initialization flag
  int _busyStart_clock[9];            // SV - busy wire flag

  //snap register variables
  int ST43, RE43, ST23, RE23, ST, ST2, ST3, ST4, ST5, ST7;
  //remainder table
  int reSumAr[3][5];
  int reSumAr23[3][5];

};
#endif

