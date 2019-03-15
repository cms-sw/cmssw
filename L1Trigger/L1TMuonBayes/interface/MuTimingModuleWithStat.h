/*
 * MuTimingModuleWithStat.h
 *
 *  Created on: Mar 8, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef INTERFACE_MUTIMINGMODULEWITHSTAT_H_
#define INTERFACE_MUTIMINGMODULEWITHSTAT_H_

#include <L1Trigger/L1TMuonBayes/interface/MuTimingModule.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH2I.h"

class MuTimingModuleWithStat: public MuTimingModule {
public:
  MuTimingModuleWithStat(const ProcConfigurationBase* config);
  virtual ~MuTimingModuleWithStat();

  void process(AlgoMuonBase* algoMuon) override;

  void generateCoefficients();
private:
  //[layer][wheel_ring][etaBin][timing][1_Beta]
  std::vector<std::vector<std::vector<TH2I*> > >  timigVs1_BetaHists; //gives average 1/beta

  TH1I* betaDist = nullptr;

  edm::Service<TFileService> fileService;
};

#endif /* INTERFACE_MUTIMINGMODULEWITHSTAT_H_ */
