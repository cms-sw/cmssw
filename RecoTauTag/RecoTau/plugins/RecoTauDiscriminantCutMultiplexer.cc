/*
 * RecoTauDiscriminantCutMultiplexer
 *
 * Author: Evan K. Friis, UW
 *
 * Takes two PFTauDiscriminators.
 *
 * The "key" discriminantor is rounded to the nearest integer.
 *
 * A set of cuts for different keys on the "toMultiplex" discriminantor is
 * provided in the config file.
 *
 * Both the key and toMultiplex discriminators should map to the same PFTau
 * collection.
 *
 */
#include <boost/foreach.hpp>
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TMath.h"
#include "TGraph.h"
#include "TFormula.h"
#include "TFile.h"

class RecoTauDiscriminantCutMultiplexer : public PFTauDiscriminationProducerBase 
{
 public:
  explicit RecoTauDiscriminantCutMultiplexer(const edm::ParameterSet& pset);

  ~RecoTauDiscriminantCutMultiplexer();
  double discriminate(const reco::PFTauRef&);
  void beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup);
  
 private:
  std::string moduleLabel_;

  struct DiscriminantCutEntry
  {
    DiscriminantCutEntry()
      : cutVariable_(0),
	cutFunction_(0),
	mode_(kUndefined)
    {}
    ~DiscriminantCutEntry()
    {
      delete cutVariable_;
      delete cutFunction_;
    }
    double cutValue_;
    StringObjectFunction<reco::PFTau>* cutVariable_;
    const TGraph* cutFunction_;
    enum { kUndefined, kFixedCut, kVariableCut };
    int mode_;
  };
  typedef std::map<int, DiscriminantCutEntry*> DiscriminantCutMap;
  DiscriminantCutMap cuts_;

  edm::InputTag toMultiplex_;
  edm::InputTag key_;
  edm::Handle<reco::PFTauDiscriminator> toMultiplexHandle_;
  edm::Handle<reco::PFTauDiscriminator> keyHandle_;

  const TFormula* mvaOutput_normalization_;
  std::vector<TFile*> inputFilesToDelete_;

  int verbosity_;
};

namespace
{
  template <typename T>
  const T* loadObjectFromFile(const edm::FileInPath& inputFileName, const std::string& objectName, std::vector<TFile*>& inputFilesToDelete)
  {
    if (  inputFileName.location() == edm::FileInPath::Unknown ) throw cms::Exception("RecoTauDiscriminantCutMultiplexer::loadObjectFromFile") 
      << " Failed to find File = " << inputFileName << " !!\n";
    TFile* inputFile = new TFile(inputFileName.fullPath().data());
  
    const T* object = dynamic_cast<T*>(inputFile->Get(objectName.data()));
    if ( !object )
      throw cms::Exception("RecoTauDiscriminantCutMultiplexer::loadObjectFromFile") 
        << " Failed to load Object = " << objectName.data() << " from file = " << inputFileName.fullPath().data() << " !!\n";

    inputFilesToDelete.push_back(inputFile);

    return object;
  }
}

RecoTauDiscriminantCutMultiplexer::RecoTauDiscriminantCutMultiplexer(const edm::ParameterSet& cfg)
  : PFTauDiscriminationProducerBase(cfg),
    moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    mvaOutput_normalization_(0)
{
  toMultiplex_ = cfg.getParameter<edm::InputTag>("toMultiplex");
  key_ = cfg.getParameter<edm::InputTag>("key");

  if ( cfg.exists("mvaOutput_normalization" ) ) {
    edm::FileInPath inputFileName = cfg.getParameter<edm::FileInPath>("inputFileName"); 
    std::string mvaOutput_normalization_string = cfg.getParameter<std::string>("mvaOutput_normalization");
    mvaOutput_normalization_ = loadObjectFromFile<TFormula>(inputFileName, mvaOutput_normalization_string, inputFilesToDelete_);
  }

  // Setup our cut map
  typedef std::vector<edm::ParameterSet> VPSet;
  VPSet mapping = cfg.getParameter<VPSet>("mapping");
  for ( VPSet::const_iterator mappingEntry = mapping.begin();
	mappingEntry != mapping.end(); ++mappingEntry ) {
    unsigned category = mappingEntry->getParameter<uint32_t>("category");
    DiscriminantCutEntry* cut = new DiscriminantCutEntry();
    if ( mappingEntry->existsAs<double>("cut") ) {
      cut->cutValue_ = mappingEntry->getParameter<double>("cut");
      cut->mode_ = DiscriminantCutEntry::kFixedCut;
    } else if ( mappingEntry->existsAs<std::string>("cut") ) {
      std::string cut_string = mappingEntry->getParameter<std::string>("cut");
      edm::FileInPath inputFileName = cfg.getParameter<edm::FileInPath>("inputFileName");      
      cut->cutFunction_ = loadObjectFromFile<TGraph>(inputFileName, cut_string, inputFilesToDelete_);
      std::string cutVariable_string = mappingEntry->getParameter<std::string>("variable");
      cut->cutVariable_ = new StringObjectFunction<reco::PFTau>(cutVariable_string.data());
      cut->mode_ = DiscriminantCutEntry::kVariableCut;
    } else {
      throw cms::Exception("RecoTauDiscriminantCutMultiplexer") 
        << " Undefined Configuration Parameter 'cut' !!\n";
    }
    cuts_[category] = cut;
  }

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

RecoTauDiscriminantCutMultiplexer::~RecoTauDiscriminantCutMultiplexer()
{
  for ( std::map<int, DiscriminantCutEntry*>::iterator it = cuts_.begin();
	it != cuts_.end(); ++it ) {
    delete it->second;
  }
  // CV: all entries in inputFilesToDelete list actually refer to the same file
  //    --> delete the first entry only
  if ( inputFilesToDelete_.size() >= 1 ) {
    delete inputFilesToDelete_.front();
  }
}

void RecoTauDiscriminantCutMultiplexer::beginEvent(const edm::Event& evt, const edm::EventSetup& es) 
{
  evt.getByLabel(toMultiplex_, toMultiplexHandle_);
  evt.getByLabel(key_, keyHandle_);
}

double
RecoTauDiscriminantCutMultiplexer::discriminate(const reco::PFTauRef& tau) 
{
  if ( verbosity_ ) {
    std::cout << "<RecoTauDiscriminantCutMultiplexer::discriminate>:" << std::endl;
    std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
  }

  double disc_result = (*toMultiplexHandle_)[tau];
  if ( verbosity_ ) {
    std::cout << "disc_result = " <<  disc_result << std::endl;
  }
  if ( mvaOutput_normalization_ ) {
    disc_result = mvaOutput_normalization_->Eval(disc_result);
    //if ( disc_result > 1. ) disc_result = 1.;
    //if ( disc_result < 0. ) disc_result = 0.;
    if ( verbosity_ ) {
      std::cout << "disc_result (normalized) = " <<  disc_result << std::endl;
    }
  }
  double key_result = (*keyHandle_)[tau];
  DiscriminantCutMap::const_iterator cutIter = cuts_.find(TMath::Nint(key_result));
  
  // Return null if it doesn't exist
  if ( cutIter == cuts_.end() ) {
    return prediscriminantFailValue_;
  }

  // See if the discriminator passes our cuts
  bool passesCuts = false;
  if ( cutIter->second->mode_ == DiscriminantCutEntry::kFixedCut ) {
    passesCuts = (disc_result > cutIter->second->cutValue_);
    if ( verbosity_ ) {
      std::cout << "cutValue (fixed) = " << cutIter->second->cutValue_ << " --> passesCuts = " << passesCuts << std::endl;
    }
  } else if ( cutIter->second->mode_ == DiscriminantCutEntry::kVariableCut ) {
    double cutVariable = (*cutIter->second->cutVariable_)(*tau);
    double xMin, xMax, dummy;
    cutIter->second->cutFunction_->GetPoint(0, xMin, dummy);
    cutIter->second->cutFunction_->GetPoint(cutIter->second->cutFunction_->GetN() - 1, xMax, dummy);
    const double epsilon = 1.e-3;
    if      ( cutVariable < (xMin + epsilon) ) cutVariable = xMin + epsilon;
    else if ( cutVariable > (xMax - epsilon) ) cutVariable = xMax - epsilon;
    double cutValue = cutIter->second->cutFunction_->Eval(cutVariable);
    passesCuts = (disc_result > cutValue);
    if ( verbosity_ ) {
      std::cout << "cutValue (@" << cutVariable << ") = " << cutValue << " --> passesCuts = " << passesCuts << std::endl;
    }
  } else assert(0);

  return passesCuts;
}

DEFINE_FWK_MODULE(RecoTauDiscriminantCutMultiplexer);
