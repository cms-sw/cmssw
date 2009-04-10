#ifndef EESelectiveReadoutTask_H
#define EESelectiveReadoutTask_H

/*
 * \file EESelectiveReadoutTask.h
 *
 * $Date: 2008/12/01 09:29:27 $
 * $Revision: 1.7 $
 * \author P. Gras
 * \author E. Di Marco
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

static const char endcapDccMap[401] = {
  "       777777       "      
  "    666777777888    "      
  "   66667777778888   "      
  "  6666667777888888  "
  " 666666677778888888 "
  " 566666677778888880 "             //    Z          
  " 555666667788888000 "             //     x-----> X 
  "55555566677888000000"             //     |         
  "555555566  880000000"             //     |         
  "55555555    00000000"//_          //     |         
  "55555554    10000000"             //     V Y       
  "554444444  111111100"
  "44444444332211111111"
  " 444444333222111111 "
  " 444443333222211111 "
  " 444433333222221111 "
  "  4443333322222111  "
  "   43333332222221   "
  "    333333222222    "
  "       333222       "};

class EESelectiveReadoutTask: public edm::EDAnalyzer{

public:

/// Constructor
EESelectiveReadoutTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EESelectiveReadoutTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(const edm::EventSetup& c);

/// EndJob
void endJob(void);

/// BeginRun
void beginRun(const edm::Run& r, const edm::EventSetup& c);

/// EndRun
void endRun(const edm::Run& r, const edm::EventSetup& c);

/// Reset
void reset(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

/// Constants
static const int nECALDcc = 54;
static const int nEEDcc = 18;
static const int nEBDcc = 36;
static const int kByte = 1024;

///number of RUs for EE
static const int nEeRus = 2*(34+32+33+33+32+34+33+34+33);

///number of endcaps
static const int nEndcaps = 2;

///EE crystal grid size along X
static const int nEeX = 100;

///EE crystal grid size along Y
static const int nEeY = 100;
 
///Number of crystals along a supercrystal edge
static const int scEdge = 5;

///Number of Trigger Towers along Eta
static const int nTtEta = 56;

///Number of Trigger Towers along Phi
static const int nTtPhi = 72;

///Number of bytes per crystal
static const int bytesPerCrystal = 24;

///To store the readout crystals / tower
int nCryTower[42][18];

///To store the events with full readout
int nEvtFullReadout[42][18];

///To store the events with any readout
int nEvtAnyReadout[42][18];

private:

///distinguishes barral and endcap of ECAL.
enum subdet_t {EB, EE};

/** Accumulates statitics for data volume analysis. To be called for each
 * ECAL digi. See anaDigiInit().
 */
void anaDigi(const EEDataFrame& frame, const EESrFlagCollection& srFlagColl);

/** Initializes statistics accumalator for data volume analysis. To
 * be call at start of each event analysis.
 */
void anaDigiInit();

/** Retrieve the logical number of the DCC reading a given crystal channel.
 * @param xtarId crystal channel identifier
 * @return the DCC logical number starting from 1.
 */
unsigned dccNum(const DetId& xtalId) const;

/** Retrives the readout unit, a trigger tower in the barrel case,
 * and a supercrystal in the endcap case, a given crystal belongs to.
 * @param xtalId identifier of the crystal
 * @return identifer of the supercrystal or of the trigger tower.
 */
EcalScDetId readOutUnitOf(const EEDetId& xtalId) const;
  
/** Converts a std CMSSW crystal x or y index to a c-array index (starting
 * from zero and without hole).
 */
int iXY2cIndex(int iX) const{
  return iX-1;
}

/** converse of iXY2cIndex() method.
 */
int cIndex2iXY(int iX0) const{
  return iX0+1;
}

/** Computes the size of an ECAL endcap event fragment.
 * @param nReadXtals number of read crystal channels
 * @return the event fragment size in bytes
 */
double getEeEventSize(double nReadXtals) const;

/** Gets the size in bytes fixed-size part of a DCC event fragment.
 * @return the fixed size in bytes.
 */
double getDccOverhead(subdet_t subdet) const{
  //  return (subdet==EB?34:25)*8;
  return (subdet==EB?34:52)*8;
}

/** Gets the size of an DCC event fragment.
 * @param iDcc0 the DCC logical number starting from 0.
 * @param nReadXtals number of read crystal channels.
 * @return the DCC event fragment size in bytes.
 */
double getDccEventSize(int iDcc0, double nReadXtals) const{
  subdet_t subdet;
  if(iDcc0<9 || iDcc0>=45){
    subdet = EE;
  } else{
    subdet = EB;
  }
  return getDccOverhead(subdet)+nReadXtals*bytesPerCrystal+nRuPerDcc_[iDcc0]*8;
}

/** Gets the phi index of the DCC reading a RU (SC or TT)
 * @param i iX
 * @param j iY
 * @return DCC phi index between 0 and 8 for EE
 */
int dccPhiIndexOfRU(int i, int j) const;

/** Gets the phi index of the DCC reading a crystal
 * @param i iX
 * @param j iY
 * @return DCC phi index between 0 and 8 for EE
 */
inline int dccPhiIndex(int i, int j) const {
  return dccPhiIndexOfRU(i/5, j/5);
}

/** Gets the index of the DCC reading a crystal
 * @param i iX
 * @param j iY
 * @return DCC phi index between 0 and 8 for EE
 */
int dccIndex(int iDet, int i, int j) const;

/** ECAL endcap read channel count
 */
int nEe_[2];

/** ECAL endcap low interest read channel count
 */
int nEeLI_[2];

/** ECAL endcap high interest read channel count
 */
int nEeHI_[2];

/** ECAL read channel count for each DCC:
 */
int nPerDcc_[nECALDcc];

/** Count for each DCC of RUs with at leat one channel read out:
 */
int nRuPerDcc_[nECALDcc];

/** For book keeping of RU actually read out (not fully zero suppressed)
 */
bool eeRuActive_[nEndcaps][nEeX/scEdge][nEeY/scEdge];

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

edm::InputTag EEDigiCollection_;
edm::InputTag EEUnsuppressedDigiCollection_;
edm::InputTag EcalRecHitCollection_;
edm::InputTag EESRFlagCollection_;
edm::InputTag EcalTrigPrimDigiCollection_;
edm::InputTag FEDRawDataCollection_;

MonitorElement* EEDccEventSize_;
MonitorElement* EETowerSize_[18];
MonitorElement* EETowerFullReadoutFrequency_[18];
MonitorElement* EEReadoutUnitForcedBitMap_[2];
MonitorElement* EEFullReadoutSRFlagMap_[2];
MonitorElement* EEHighInterestTriggerTowerFlagMap_[2];
MonitorElement* EELowInterestTriggerTowerFlagMap_[2];
MonitorElement* EEEventSize_[2];
MonitorElement* EEHighInterestPayload_[2];
MonitorElement* EELowInterestPayload_[2];

bool init_;

};

#endif
