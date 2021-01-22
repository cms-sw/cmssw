//--------------------------------------------------------------------------------------------------
//
// PileupJetIdAlgo
//
// Author: P. Musella, P. Harris
//--------------------------------------------------------------------------------------------------

#ifndef RecoJets_JetProducers_plugins_PileupJetIdAlgo_h
#define RecoJets_JetProducers_plugins_PileupJetIdAlgo_h

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"

// ----------------------------------------------------------------------------------------------------
class PileupJetIdAlgo {
public:
  enum version_t { USER = -1, PHILv0 = 0 };

  class AlgoGBRForestsAndConstants;

  PileupJetIdAlgo(AlgoGBRForestsAndConstants const* cache);
  ~PileupJetIdAlgo();

  PileupJetIdentifier computeIdVariables(
      const reco::Jet* jet, float jec, const reco::Vertex*, const reco::VertexCollection&, double rho, bool usePuppi);

  void set(const PileupJetIdentifier&);
  float getMVAval(const std::vector<std::string>&, const std::unique_ptr<const GBRForest>&);
  PileupJetIdentifier computeMva();
  const std::string method() const { return cache_->tmvaMethod(); }

  std::string dumpVariables() const;

  typedef std::map<std::string, std::pair<float*, float>> variables_list_t;

  std::pair<int, int> getJetIdKey(float jetPt, float jetEta);
  int computeCutIDflag(float betaStarClassic, float dR2Mean, float nvtx, float jetPt, float jetEta);
  int computeIDflag(float mva, float jetPt, float jetEta);
  int computeIDflag(float mva, int ptId, int etaId);

  /// const PileupJetIdentifier::variables_list_t & getVariables() const { return variables_; };
  const variables_list_t& getVariables() const { return variables_; };

  // In multithreaded mode, each PileupIdAlgo object will get duplicated
  // on every stream. Some of the data it contains never changes after
  // construction and can be shared by all streams. This nested class contains
  // the data members that can be shared across streams. In particular
  // the GBRForests take significant time to initialize and can be shared.
  class AlgoGBRForestsAndConstants {
  public:
    AlgoGBRForestsAndConstants(edm::ParameterSet const&, bool runMvas);

    std::unique_ptr<const GBRForest> const& reader() const { return reader_; }
    std::vector<std::unique_ptr<const GBRForest>> const& etaReader() const { return etaReader_; }
    bool cutBased() const { return cutBased_; }
    bool etaBinnedWeights() const { return etaBinnedWeights_; }
    bool runMvas() const { return runMvas_; }
    int nEtaBins() const { return nEtaBins_; }
    std::vector<double> const& jEtaMin() const { return jEtaMin_; }
    std::vector<double> const& jEtaMax() const { return jEtaMax_; }
    std::string const& label() const { return label_; }
    std::string const& tmvaMethod() const { return tmvaMethod_; }
    std::vector<std::string> const& tmvaVariables() const { return tmvaVariables_; }
    std::vector<std::vector<std::string>> const& tmvaEtaVariables() const { return tmvaEtaVariables_; }

    typedef float array_t[3][5][4];
    array_t const& mvacut() const { return mvacut_; }
    array_t const& rmsCut() const { return rmsCut_; }
    array_t const& betaStarCut() const { return betaStarCut_; }

  private:
    std::unique_ptr<const GBRForest> reader_;
    std::vector<std::unique_ptr<const GBRForest>> etaReader_;
    bool cutBased_;
    bool etaBinnedWeights_;
    bool runMvas_;
    int nEtaBins_;
    std::vector<double> jEtaMin_;
    std::vector<double> jEtaMax_;
    std::string label_;
    std::string tmvaMethod_;
    std::vector<std::string> tmvaVariables_;
    std::vector<std::vector<std::string>> tmvaEtaVariables_;

    float mvacut_[3][5][4];       //Keep the array fixed
    float rmsCut_[3][5][4];       //Keep the array fixed
    float betaStarCut_[3][5][4];  //Keep the array fixed

    std::map<std::string, std::string> tmvaNames_;
  };

protected:
  void runMva();
  void resetVariables();
  void initVariables();

  PileupJetIdentifier internalId_;
  variables_list_t variables_;
  AlgoGBRForestsAndConstants const* cache_;
};
#endif
