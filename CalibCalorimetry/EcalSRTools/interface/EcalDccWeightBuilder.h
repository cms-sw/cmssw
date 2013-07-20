/*
 * $Id: EcalDccWeightBuilder.h,v 1.6 2009/10/09 15:55:35 pgras Exp $
 */

#ifndef ECALDCCWEIGHTBUILDER_CC
#define ECALDCCWEIGHTBUILDER_CC


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include <vector>
#include <map>
#include <inttypes.h>

// Forward declarations:
class EcalElectronicsMapping;

/**
 */
class EcalDccWeightBuilder: public edm::EDAnalyzer {
private:
  enum mode_t { WEIGHTS_FROM_CONFIG, COMPUTE_WEIGHTS};
  
  //constructor(s) and destructor(s)
public:
  /** Constructs an EcalDccWeightBuilder
   * @param ps CMSSW mondule configuration
   */
  EcalDccWeightBuilder(edm::ParameterSet const& ps);
  
  /**Destructor
   */
  virtual ~EcalDccWeightBuilder(){};

  //method(s)
public:

  /** Analyze method called by the event loop.
   * @param event CMSSW event
   * @param es event setup
   */
  void analyze(const edm::Event& event, const edm::EventSetup& es);

private:

  /** Weight computation
   * @param shape signal shape to use for weight computation
   * @param binOfMax time bin which shall contain the pulse peak
   * @param timePhase signal phase. First sample corresponds to
   *                  t = ((binOfMax-1)*25+timePhase) ns
   * in the time reference of EcalShapeBase.
   * @param iFirst first sample the weights must be applied to
   * @param nWeights number of weights
   * @param iSkip0 if greater than 0, the corresponding sample will not be
   * used (weights forced to 0).
   * @param result [out] vector filled with computed weights. The vector is
   * resized to the number of weights
   */
  void
  computeWeights(const EcalShapeBase& shape,
		 int binOfMax, double timePhase, 
		 int iFirst0, int nWeights, int iSkip0,
                 std::vector<double>& result);

  void computeAllWeights(bool withIntercalib);
  
  int encodeWeight(double w);

  double decodeWeight(int W);

  void unbiasWeights(std::vector<double>& weights,
                     std::vector<int32_t>* encodedWeigths);

  /** Retrieve intercalibration coefficent for channel detId.
   * @param detId ID of the channel the intercalibration coef. must be
   * retrieved for.
   */
  double intercalib(const DetId& detId);
  
  void writeWeightToAsciiFile();
  void writeWeightToRootFile();   
  void writeWeightToDB();

  //converts DetId to IDs used by DB:
  void dbId(const DetId& detId, int& fedId, int& smId, int& ruId,
           int& xtalId) const;

  /** Vector index sorting.
   * @param vector of values
   * @param [out] sorted index of the vector
   * @param decreasingOrder switch to use decreasing order instead of
   * default increasing order
   */
  template<class T>
  void sort(const std::vector<T>& a,
	    std::vector<int>& s,
	    bool decreasingOrder = false);

  
  //attribute(s)
protected:
private:
  int dcc1stSample_;
  int sampleToSkip_;
  int nDccWeights_;
  /** weights used in weightFromConfig mode.
   */
  std::vector<double> inputWeights_;
  std::string mode_;
  mode_t imode_;
  bool dccWeightsWithIntercalib_;
  bool writeToDB_;
  bool writeToAsciiFile_;
  bool writeToRootFile_;
  std::string asciiOutputFileName_;
  std::string rootOutputFileName_;
  std::string dbSid_;
  std::string dbUser_;
  std::string dbPassword_;
  std::string dbTag_;
  int dbVersion_;
  bool sqlMode_;
  
  edm::ESHandle<CaloGeometry> geom_;
  
  EcalIntercalibConstantMap& calibMap_;
  EcalIntercalibConstantMap emptyCalibMap_;
  std::map<DetId, std::vector<int> > encodedWeights_;

  static const double weightScale_;
  const EcalElectronicsMapping* ecalElectronicsMap_;

  static const int ecalDccFedIdMin = 601;
  static const int ecalDccFedIdMax = 654;
  static const int nDccs = ecalDccFedIdMax-ecalDccFedIdMin+1;
};

#endif //ECALDCCWEIGHTBUILDER_CC not defined
