#ifndef EBSelectiveReadoutTask_H
#define EBSelectiveReadoutTask_H

/*
 * \file EBSelectiveReadoutTask.h
 *
 * $Date: 2008/07/30 16:20:33 $
 * $Revision: 1.8 $
 * \author P. Gras
 * \author E. Di Marco
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EBSelectiveReadoutTask: public edm::EDAnalyzer{

public:

/// Constructor
EBSelectiveReadoutTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBSelectiveReadoutTask();

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
static const int nTTEta = 34;
static const int nTTPhi = 72;
static const int nECALDcc = 54;
static const int nEBDcc = 36;
static const int kByte = 1024;

///number of RUs for EB
static const int nEbRus = 36*68;

///number of crystals along Eta in EB
static const int nEbEta = 170;

///number of crystals along Phi in EB
static const int nEbPhi = 360;

///Number of crystals along an EB TT
static const int ebTtEdge = 5;
 
///Number of crystals along a supercrystal edge
static const int scEdge = 5;

///Number of Trigger Towers along Eta
static const int nTtEta = 56;

///Number of Trigger Towers along Phi
static const int nTtPhi = 72;

///Number of bytes per crystal
static const int bytesPerCrystal = 24;

private:

///distinguishes barral and endcap of ECAL.
enum subdet_t {EB, EE};

/** Accumulates statitics for data volume analysis. To be called for each
 * ECAL digi. See anaDigiInit().
 */
void anaDigi(const EBDataFrame& frame, const EBSrFlagCollection& srFlagColl);

/** Initializes statistics accumalator for data volume analysis. To
 * be call at start of each event analysis.
 */
void anaDigiInit();

/** Retrieve the logical number of the DCC reading a given crystal channel.
 * @param xtarId crystal channel identifier
 * @return the DCC logical number starting from 1.
 */
unsigned dccNum(const DetId& xtalId) const;

/** Converts a std CMSSW crystal eta index to a c-array index (starting from
 * zero and without hole).
 */
int iEta2cIndex(int iEta) const{
  return (iEta<0)?iEta+85:iEta+84;
}

/** Converts a std CMSSW crystal phi index to a c-array index (starting from
 * zero and without hole).
 */
int iPhi2cIndex(int iPhi) const{
  return iPhi-1;
}

/** Retrives the readout unit, a trigger tower in the barrel case,
 * and a supercrystal in the endcap case, a given crystal belongs to.
 * @param xtalId identifier of the crystal
 * @return identifer of the supercrystal or of the trigger tower.
 */
EcalTrigTowerDetId readOutUnitOf(const EBDetId& xtalId) const;
  
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

/** converse of iEta2cIndex() method.
 */
int cIndex2iEta(int i) const{
  return (i<85)?i-85:i-84;
}

/** converse of iPhi2cIndex() method.
 */
int cIndex2iPhi(int i) const {
  return i+1;
}

/** Computes the size of an ECAL barrel event fragment.
 * @param nReadXtals number of read crystal channels
 * @return the event fragment size in bytes
 */
double getEbEventSize(double nReadXtals) const;

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
 * @param i iEta 
 * @param j iPhi 
 * @return DCC phi index between 0 and 17 for EB
 */
int dccPhiIndexOfRU(int i, int j) const;

/** Gets the phi index of the DCC reading a crystal
 * @param i iEta
 * @param j iPhi
 * @return DCC phi index between 0 and 17 for EB
 */
int dccPhiIndex(int i, int j) const {
  return dccPhiIndexOfRU(i/5, j/5);
}

/** Gets the index of the DCC reading a crystal
 * @param i iEta
 * @param j iPhi
 * @return DCC index between 0 and 17 
 */
int dccIndex(int i, int j) const;

/** ECAL barrel read channel count
 */
int nEb_;

/** ECAL barrel low interest read channel count
 */
int nEbLI_;

/** ECAL barrel high interest read channel count
 */
int nEbHI_;

/** ECAL read channel count for each DCC:
 */
int nPerDcc_[nECALDcc];

 /** Count for each DCC of RUs with at leat one channel read out:
  */
int nRuPerDcc_[nECALDcc];

/** For book keeping of RU actually read out (not fully zero suppressed)
 */
bool ebRuActive_[nEbEta/ebTtEdge][nEbPhi/ebTtEdge];

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

edm::InputTag EBDigiCollection_;
edm::InputTag EBUnsuppressedDigiCollection_;
edm::InputTag EcalRecHitCollection_;
edm::InputTag EBSRFlagCollection_;
edm::InputTag EcalTrigPrimDigiCollection_;
edm::InputTag FEDRawDataCollection_;

MonitorElement* EBDccEventSize_;
MonitorElement* EBReadoutUnitForcedBitMap_;
MonitorElement* EBFullReadoutSRFlagMap_;
MonitorElement* EBHighInterestTriggerTowerFlagMap_;
MonitorElement* EBLowInterestTriggerTowerFlagMap_;
MonitorElement* EBEventSize_;
MonitorElement* EBHighInterestPayload_;
MonitorElement* EBLowInterestPayload_;

bool init_;

};

#endif
