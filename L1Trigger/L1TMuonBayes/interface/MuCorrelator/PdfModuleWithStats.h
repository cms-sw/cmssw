/*
 * PdfModuleWithStats.h
 *
 *  Created on: Feb 4, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef INTERFACE_MUCORRELATOR_PDFMODULEWITHSTATS_H_
#define INTERFACE_MUCORRELATOR_PDFMODULEWITHSTATS_H_

#include "L1Trigger/L1TMuonBayes/interface/MuCorrelator/PdfModule.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH2I.h"

class PdfModuleWithStats: public PdfModule {
public:
  PdfModuleWithStats(MuCorrelatorConfigPtr& config);

  virtual ~PdfModuleWithStats();

  virtual float getPdfVal(unsigned int layer, unsigned int etaBin, unsigned int refLayer, unsigned int ptBin, int pdfBin);

  //writes pdfHists to current root file
  //virtual void write() const;

  virtual void generateCoefficients();

  virtual void generateCoefficients1();
private:
  //[layer][etaBin][refLayer](ptBin, pdfBin)
  std::vector<std::vector<std::vector<TH2I* > > > pdfHists;

  edm::Service<TFileService> fs;

  double sigmaFactor = 1.;
};

#endif /* INTERFACE_MUCORRELATOR_PDFMODULEWITHSTATS_H_ */
