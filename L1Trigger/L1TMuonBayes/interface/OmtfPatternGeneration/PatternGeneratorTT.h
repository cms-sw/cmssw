/*
 * PatternGeneratorTT.h
 *
 *  Created on: Oct 17, 2018
 *      Author: kbunkow
 */

#ifndef OMTF_PATTERNGENERATORTT_H_
#define OMTF_PATTERNGENERATORTT_H_

#include <L1Trigger/L1TMuonBayes/interface/OmtfPatternGeneration/PatternOptimizerBase.h>

class PatternGeneratorTT: public PatternOptimizerBase {
public:
  PatternGeneratorTT(const edm::ParameterSet& edmCfg, const OMTFConfiguration* omtfConfig, std::vector<std::shared_ptr<GoldenPatternWithStat> >& gps);

  virtual ~PatternGeneratorTT();

  virtual void observeEventEnd(const edm::Event& iEvent);

  void endJob();
protected:
  void updateStat();

  void upadatePdfs();

  virtual void saveHists(TFile& outfile);

  //[charge][iLayer]
  std::vector<std::vector<TH2I*> > ptDeltaPhiHists;
};

#endif /* OMTF_PATTERNGENERATORTT_H_ */
