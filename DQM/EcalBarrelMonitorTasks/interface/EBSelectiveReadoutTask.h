#ifndef EBSelectiveReadoutTask_H
#define EBSelectiveReadoutTask_H

/*
 * \file EBSelectiveReadoutTask.h
 *
 * $Date: 2011/03/03 22:05:50 $
 * $Revision: 1.19 $
 * \author P. Gras
 * \author E. Di Marco
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class MonitorElement;
class DQMStore;
class EcalSRSettings;

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
void beginJob(void);

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

///maximum number of RUs read by a DCC
static const int nDccChs = 68;

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

///To store the readout crystals / tower
int nCryTower[72][34];

///To store the events with full readout 
int nEvtFullReadout[72][34];

///To store the events with RU forced 
int nEvtRUForced[72][34];

///To store the events with any readout
int nEvtAnyReadout[72][34];

///To store the events with ZS1 readout
int nEvtZS1Readout[72][34];

///To store the events with ZS1 or ZS2 readout
int nEvtZSReadout[72][34];

///To store the events with complete readout when ZS is requested
int nEvtCompleteReadoutIfZS[72][34];

///To store the events with 0 channels readout when FR is requested
int nEvtDroppedReadoutIfFR[72][34];

///To store the events with high interest TT
int nEvtHighInterest[72][34];

///To store the events with medium interest TT
int nEvtMediumInterest[72][34];

///To store the events with low interest TT
int nEvtLowInterest[72][34];

///To store the events with any interest
int nEvtAnyInterest[72][34];

private:

///distinguishes barral and endcap of ECAL.
enum subdet_t {EB, EE};

/** Accumulates statitics for data volume analysis. To be called for each
 * ECAL digi. See anaDigiInit().
 */
void anaDigi(const EBDataFrame& frame, const EBSrFlagCollection& srFlagColl, uint16_t statusCode);

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

/** Configure DCC ZS FIR weights. Heuristic is used to determine
 * if input weights are normalized weights or integer weights in
 * the hardware representation.
 * @param weightsForZsFIR weights from configuration file
 */
void configFirWeights(std::vector<double> weightsForZsFIR);

/** Emulates the DCC zero suppression FIR filter. If one of the time sample
 * is not in gain 12, numeric_limits<int>::max() is returned.
 * @param frame data frame
 * @param firWeights TAP weights
 * @param firstFIRSample index (starting from 1) of the first time
 * sample to be used in the filter
 * @param saturated if not null, *saturated is set to true if all the time
 * sample are not in gain 12 and set to false otherwise.
 * @return FIR output or numeric_limits<int>::max().
 */
static int dccZsFIR(const EcalDataFrame& frame,
                    const std::vector<int>& firWeights,
                    int firstFIRSample,
                    bool* saturated = 0);
  

 /** Computes the ZS FIR filter weights from the normalized weights.
  * @param normalizedWeights the normalized weights
  * @return the computed ZS filter weights.
  */
static std::vector<int> getFIRWeights(const std::vector<double>&
                                      normalizedWeights);

/** Retrieves number of crystal channel read out by a DCC channel
 * @param iDcc DCC ID starting from 1
 * @param iDccCh DCC channel starting from 1
 * @return crystal count
 */
int getCrystalCount() { return 25; }


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

/** Number of crystal read for each DCC channel (aka readout unit).
 */
int nPerRu_[nECALDcc][nDccChs];   

 /** Count for each DCC of RUs with at leat one channel read out:
  */
int nRuPerDcc_[nECALDcc];

/** For book keeping of RU actually read out (not fully zero suppressed)
 */
bool ebRuActive_[nEbEta/ebTtEdge][nEbPhi/ebTtEdge];

/** Weights to be used for the ZS FIR filter
 */
std::vector<int> firWeights_;

/** Time position of the first sample to use in zero suppession FIR
 * filter. Numbering starts at 0.
 */
int firstFIRSample_;

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

float xbins[37];
float ybins[89];

MonitorElement* EBDccEventSize_;
MonitorElement* EBDccEventSizeMap_;
MonitorElement* EBTowerSize_;
MonitorElement* EBTTFMismatch_;
MonitorElement* EBReadoutUnitForcedBitMap_;
MonitorElement* EBFullReadoutSRFlagMap_;
MonitorElement* EBFullReadoutSRFlagCount_;
MonitorElement* EBZeroSuppression1SRFlagMap_;
MonitorElement* EBHighInterestTriggerTowerFlagMap_;
MonitorElement* EBMediumInterestTriggerTowerFlagMap_;
MonitorElement* EBLowInterestTriggerTowerFlagMap_;
MonitorElement* EBTTFlags_;
MonitorElement* EBCompleteZSMap_;
MonitorElement* EBCompleteZSCount_;
MonitorElement* EBDroppedFRMap_;
MonitorElement* EBDroppedFRCount_;
MonitorElement* EBEventSize_;
MonitorElement* EBHighInterestPayload_;
MonitorElement* EBLowInterestPayload_;
MonitorElement* EBHighInterestZsFIR_;
MonitorElement* EBLowInterestZsFIR_;

bool init_;

bool useCondDb_;
const EcalSRSettings* settings_;

};

#endif
