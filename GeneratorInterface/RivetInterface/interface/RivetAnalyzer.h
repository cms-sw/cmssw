#ifndef GeneratorInterface_RivetInterface_RivetAnalyzer
#define GeneratorInterface_RivetInterface_RivetAnalyzer

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "Rivet/AnalysisHandler.hh"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"

#include "Rivet/Tools/RivetYODA.hh"
//#include "YODA/ROOTCnv.h"

#include <vector>
#include <string>

class RivetAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  RivetAnalyzer(const edm::ParameterSet &);

  ~RivetAnalyzer() override;

  void beginJob() override;

  void endJob() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;

  void beginRun(const edm::Run &, const edm::EventSetup &) override;

  void endRun(const edm::Run &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<edm::HepMCProduct> _hepmcCollection;
  bool _useLHEweights;
  double _weightCap;
  double _NLOSmearing;
  bool _setIgnoreBeams;
  bool _skipMultiWeights;
  std::string _selectMultiWeights;
  std::string _deselectMultiWeights;
  std::string _setNominalWeightName;
  edm::EDGetTokenT<LHEEventProduct> _LHECollection;
  edm::EDGetTokenT<GenEventInfoProduct> _genEventInfoCollection;
  edm::EDGetTokenT<GenLumiInfoHeader> _genLumiInfoToken;
  edm::EDGetTokenT<LHERunInfoProduct> _lheRunInfoToken;
  std::unique_ptr<Rivet::AnalysisHandler> _analysisHandler;
  bool _isFirstEvent;
  std::string _outFileName;
  std::vector<std::string> _analysisNames;
  bool _doFinalize;
  const edm::InputTag _lheLabel;
  double _xsection;
  std::vector<std::string> _weightNames;
  std::vector<std::string> _lheWeightNames;
  std::vector<std::string> _cleanedWeightNames;
};

#endif
