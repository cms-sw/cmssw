/*
 * ProcessorBase.h
 *
 *  Created on: Jul 28, 2017
 *      Author: kbunkow
 */

#ifndef OMTF_PROCESSORBASE_H_
#define OMTF_PROCESSORBASE_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinput.h"

#include <memory>

class L1TMuonOverlapParams;
class SimTrack;

template <class GoldenPatternType>
class ProcessorBase {
public:
  typedef std::vector<std::shared_ptr<GoldenPatternType> > GoldenPatternVec;

  ProcessorBase(OMTFConfiguration* omtfConfig, const L1TMuonOverlapParams* omtfPatterns) : myOmtfConfig(omtfConfig) {
    configure(omtfConfig, omtfPatterns);
  };

  ProcessorBase(OMTFConfiguration* omtfConfig, const GoldenPatternVec& gps) : myOmtfConfig(omtfConfig) {
    setGPs(gps);
    //initPatternPtRange(true); is called in the setGPs
    omtfConfig->setPatternPtRange(getPatternPtRange());
  };

  virtual ~ProcessorBase() {}

  ///Return vector of GoldenPatterns
  virtual const GoldenPatternVec& getPatterns() const { return theGPs; };

  ///Fill GP vec with patterns from CondFormats object
  virtual bool configure(OMTFConfiguration* omtfParams, const L1TMuonOverlapParams* omtfPatterns);

  ///Add GoldenPattern to pattern vec.
  virtual void addGP(GoldenPatternType* aGP);

  virtual void setGPs(const GoldenPatternVec& gps);

  ///Reset all configuration parameters
  virtual void resetConfiguration();

  virtual void initPatternPtRange(bool firstPatFrom0);

  const std::vector<OMTFConfiguration::PatternPt>& getPatternPtRange() const { return patternPts; }

  virtual void printInfo() const;

protected:
  ///vector holding Golden Patterns
  GoldenPatternVec theGPs;

  ///Configuration of the algorithm. This object
  ///does not contain the patterns data.
  const OMTFConfiguration* myOmtfConfig;

  ///Remove hits whis are outside input range
  ///for given processor and cone
  virtual MuonStubPtrs1D restrictInput(unsigned int iProcessor,
                                       unsigned int iCone,
                                       unsigned int iLayer,
                                       const OMTFinput& input);

  std::vector<OMTFConfiguration::PatternPt> patternPts;
};

#endif /* OMTF_PROCESSORBASE_H_ */
