#ifndef HLTriggerOffline_Exotica_HLTExoticaValidator_H
#define HLTriggerOffline_Exotica_HLTExoticaValidator_H

/** \class HLTExoticaValidator
 *  Generate histograms for trigger efficiencies Exotica related
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/EXOTICATriggerValidation
 *
 *  \author  Thiago R. Fernandez Perez Tomei
 *           Based and adapted from:
 *           J. Duarte Campderros code from HLTriggerOffline/Higgs and
 *           J. Klukas, M. Vander Donckt and J. Alcaraz code
 *           from the HLTriggerOffline/Muon package.
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "HLTriggerOffline/Exotica/interface/HLTExoticaSubAnalysis.h"

#include <cstring>
#include <vector>

struct EVTColContainer;

/// The HLTExoticaValidator module is the main module of the
/// package. It books a vector of auxiliary classes
/// (HLTExoticaSubAnalysis), where each of those takes care
/// of one single analysis. Each of those, in turn, books a
/// vector if HLTExoticaPlotters to make plots for each
/// HLT path
class HLTExoticaValidator : public DQMOneEDAnalyzer<> {
public:
  /// Constructor and destructor
  HLTExoticaValidator(const edm::ParameterSet &);
  ~HLTExoticaValidator() override;

protected:
  /// Method called by the framework to book histograms.
  void bookHistograms(DQMStore::IBooker &iBooker, const edm::Run &iRun, const edm::EventSetup &iSetup) override;

private:
  void beginJob() override;
  /// Method called by the framework just before dqmBeginRun()
  void dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) override;
  /// Method called for each event.
  void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) override;
  void dqmEndRun(const edm::Run &iRun, const edm::EventSetup &iSetup) override;
  void endJob() override;

  /// Copy (to be modified) of the input ParameterSet from configuration file.
  edm::ParameterSet _pset;
  /// The names of the subanalyses
  std::vector<std::string> _analysisnames;

  /// The instances of the class which do the real work
  std::vector<HLTExoticaSubAnalysis> _analyzers;

  /// Centralized point of access to all collections used
  EVTColContainer *_collections;
};

#endif
