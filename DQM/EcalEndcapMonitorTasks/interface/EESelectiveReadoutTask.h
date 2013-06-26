#ifndef EESelectiveReadoutTask_H
#define EESelectiveReadoutTask_H

/*
 * \file EESelectiveReadoutTask.h
 *
 * $Date: 2011/03/03 22:05:50 $
 * $Revision: 1.22 $
 * \author P. Gras
 * \author E. Di Marco
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

class MonitorElement;
class DQMStore;
class EcalSRSettings;

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
static const int nECALDcc = 54;
static const int nEEDcc = 18;
static const int nEBDcc = 36;
static const int kByte = 1024;

///maximum number of RUs read by a DCC
static const int nDccChs = 68;

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

///To store the readout crystals / SC
int nCrySC[20][20][2];

///To store the readout crystals / iTT
/// indexes are [iTTC][iTT] 
int nCryTT[108][41];

///To store the events with full readout
int nEvtFullReadout[20][20][2];

///To store the events with RU forced
int nEvtRUForced[20][20][2];

///To store the events with ZS1 readout
int nEvtZS1Readout[20][20][2];

///To store the events with ZS1 or ZS2 readout
int nEvtZSReadout[20][20][2];

///To store the events with complete readout when ZS is requested
int nEvtCompleteReadoutIfZS[20][20][2];

///To store the events with 0 channels readout when FR is requested
int nEvtDroppedReadoutIfFR[20][20][2];

///To store the events with any readout
int nEvtAnyReadout[20][20][2];

///To store the events with high interest TT
int nEvtHighInterest[100][100][2];

///To store the events with medium interest TT
int nEvtMediumInterest[100][100][2];

///To store the events with low interest TT
int nEvtLowInterest[100][100][2];

///To store the events with any interest
int nEvtAnyInterest[100][100][2];


private:

///distinguishes barral and endcap of ECAL.
enum subdet_t {EB, EE};

/** Accumulates statitics for data volume analysis. To be called for each
 * ECAL digi. See anaDigiInit().
 */
 void anaDigi(const EEDataFrame& frame, const EESrFlagCollection& srFlagColl, uint16_t statusCode);

/** Initializes statistics accumalator for data volume analysis. To
 * be call at start of each event analysis.
 */
void anaDigiInit();

/** Retrieve the logical number of the DCC reading a given crystal channel.
 * @param xtarId crystal channel identifier
 * @return the DCC logical number starting from 1.
 */
unsigned dccNum(const DetId& xtalId) const;

/** Retrieve the logical number of the DCC reading a given SC channel.
 * @param scId SC channel identifier
 * @return the DCC logical number starting from 1.
 */
unsigned dccNumOfRU(const EcalScDetId& scId) const;

/** Retrives the readout unit, a trigger tower in the barrel case,
 * and a supercrystal in the endcap case, a given crystal belongs to.
 * @param xtalId identifier of the crystal
 * @return identifer of the supercrystal or of the trigger tower.
 */
const EcalScDetId readOutUnitOf(const EEDetId& xtalId) const;
  
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
int getCrystalCount(int iDcc, int iDccCh);

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

/** Number of crystal read for each DCC channel (aka readout unit).
 */
int nPerRu_[nECALDcc][nDccChs];   

/** Count for each DCC of RUs with at leat one channel read out:
 */
int nRuPerDcc_[nECALDcc];

/** For book keeping of RU actually read out (not fully zero suppressed)
 */
bool eeRuActive_[nEndcaps][nEeX/scEdge][nEeY/scEdge];

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

edm::InputTag EEDigiCollection_;
edm::InputTag EEUnsuppressedDigiCollection_;
edm::InputTag EcalRecHitCollection_;
edm::InputTag EESRFlagCollection_;
edm::InputTag EcalTrigPrimDigiCollection_;
edm::InputTag FEDRawDataCollection_;

float xbins[19];
float ybins[133];

MonitorElement* EEDccEventSize_;
MonitorElement* EEDccEventSizeMap_;
MonitorElement* EETowerSize_[2];
MonitorElement* EETTFMismatch_[2];
MonitorElement* EEReadoutUnitForcedBitMap_[2];
MonitorElement* EEFullReadoutSRFlagMap_[2];
MonitorElement* EEFullReadoutSRFlagCount_[2];
MonitorElement* EEZeroSuppression1SRFlagMap_[2];
MonitorElement* EEHighInterestTriggerTowerFlagMap_[2];
MonitorElement* EEMediumInterestTriggerTowerFlagMap_[2];
MonitorElement* EELowInterestTriggerTowerFlagMap_[2];
MonitorElement* EETTFlags_[2];
MonitorElement* EECompleteZSMap_[2];
MonitorElement* EECompleteZSCount_[2];
MonitorElement* EEDroppedFRMap_[2];
MonitorElement* EEDroppedFRCount_[2];
MonitorElement* EEEventSize_[2];
MonitorElement* EEHighInterestPayload_[2];
MonitorElement* EELowInterestPayload_[2];
MonitorElement* EEHighInterestZsFIR_[2];
MonitorElement* EELowInterestZsFIR_[2];

bool init_;

bool useCondDb_;
const EcalSRSettings* settings_;

};

#endif
