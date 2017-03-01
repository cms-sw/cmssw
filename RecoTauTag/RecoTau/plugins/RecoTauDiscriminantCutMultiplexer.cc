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

#include "CondFormats/PhysicsToolsObjects/interface/PhysicsTGraphPayload.h"
#include "CondFormats/DataRecord/interface/PhysicsTGraphPayloadRcd.h"
#include "CondFormats/PhysicsToolsObjects/interface/PhysicsTFormulaPayload.h"
#include "CondFormats/DataRecord/interface/PhysicsTFormulaPayloadRcd.h"

#include "TMath.h"
#include "TGraph.h"
#include "TFormula.h"
#include "TFile.h"

class RecoTauDiscriminantCutMultiplexer : public PFTauDiscriminationProducerBase 
{
 public:
  explicit RecoTauDiscriminantCutMultiplexer(const edm::ParameterSet& pset);

  ~RecoTauDiscriminantCutMultiplexer();
  double discriminate(const reco::PFTauRef&) const override;
  void beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup) override;
  
 private:
  std::string moduleLabel_;

  bool loadMVAfromDB_;
  edm::FileInPath inputFileName_;

  struct DiscriminantCutEntry
  {
    DiscriminantCutEntry()
      : cutVariable_(),
	cutFunction_(),
	mode_(kUndefined)
    {}
    ~DiscriminantCutEntry()
    {
    }
    double cutValue_;
    std::string cutName_;
    std::unique_ptr<StringObjectFunction<reco::PFTau>> cutVariable_;
    std::unique_ptr<const TGraph> cutFunction_;
    enum { kUndefined, kFixedCut, kVariableCut };
    int mode_;
  };
  typedef std::map<int, std::unique_ptr<DiscriminantCutEntry>> DiscriminantCutMap;
  DiscriminantCutMap cuts_;

  std::string mvaOutputNormalizationName_;
  std::unique_ptr<const TFormula> mvaOutput_normalization_;

  bool isInitialized_;

  edm::InputTag toMultiplex_;
  edm::InputTag key_;
  edm::Handle<reco::PFTauDiscriminator> toMultiplexHandle_;
  edm::Handle<reco::PFTauDiscriminator> keyHandle_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> toMultiplex_token;
  edm::EDGetTokenT<reco::PFTauDiscriminator> key_token;

  int verbosity_;
};

namespace
{
  std::unique_ptr<TFile> openInputFile(const edm::FileInPath& inputFileName) {
    if ( inputFileName.location() == edm::FileInPath::Unknown){  throw cms::Exception("RecoTauDiscriminantCutMultiplexer::loadObjectFromFile") 
      << " Failed to find File = " << inputFileName << " !!\n";
    }
    return std::unique_ptr<TFile>{ new TFile(inputFileName.fullPath().data()) };
  }

  template <typename T>
  std::unique_ptr<const T> loadObjectFromFile(TFile& inputFile, const std::string& objectName)
  {
    const T* object = dynamic_cast<T*>(inputFile.Get(objectName.data()));
    if ( !object )
      throw cms::Exception("RecoTauDiscriminantCutMultiplexer::loadObjectFromFile") 
        << " Failed to load Object = " << objectName.data() << " from file = " << inputFile.GetName() << " !!\n";
    //Need to use TObject::Clone since the type T might be a base class
    std::unique_ptr<const T> copy{ static_cast<T*>(object->Clone()) };

    return std::move(copy);
  }

  std::unique_ptr<const TGraph> loadTGraphFromDB(const edm::EventSetup& es, const std::string& graphName, const int& verbosity_ = 0)
  {
    if(verbosity_){
      std::cout << "<loadTGraphFromDB>:" << std::endl;
      std::cout << " graphName = " << graphName << std::endl;
    }
    edm::ESHandle<PhysicsTGraphPayload> graphPayload;
    es.get<PhysicsTGraphPayloadRcd>().get(graphName, graphPayload);
    return std::unique_ptr<const TGraph>{ new TGraph(*graphPayload.product()) };
  }  

  std::unique_ptr<TFormula> loadTFormulaFromDB(const edm::EventSetup& es, const std::string& formulaName, const TString& newName, const int& verbosity_ = 0)
  {
    if(verbosity_){
      std::cout << "<loadTFormulaFromDB>:" << std::endl;
      std::cout << " formulaName = " << formulaName << std::endl;
    }
    edm::ESHandle<PhysicsTFormulaPayload> formulaPayload;
    es.get<PhysicsTFormulaPayloadRcd>().get(formulaName, formulaPayload);

    if ( formulaPayload->formulas().size() == 1 && formulaPayload->limits().size() == 1 ) {
      return std::unique_ptr<TFormula> {new TFormula(newName, formulaPayload->formulas().at(0).data()) };
    } else {
      throw cms::Exception("RecoTauDiscriminantCutMultiplexer::loadTFormulaFromDB") 
	<< "Failed to load TFormula = " << formulaName << " from Database !!\n";
    }
    return std::unique_ptr<TFormula>{};
  }  
}

RecoTauDiscriminantCutMultiplexer::RecoTauDiscriminantCutMultiplexer(const edm::ParameterSet& cfg)
  : PFTauDiscriminationProducerBase(cfg),
    moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    mvaOutput_normalization_(),
    isInitialized_(false)
{
  
  toMultiplex_ = cfg.getParameter<edm::InputTag>("toMultiplex");
  toMultiplex_token = consumes<reco::PFTauDiscriminator>(toMultiplex_);
  key_ = cfg.getParameter<edm::InputTag>("key");
  key_token = consumes<reco::PFTauDiscriminator>(key_);

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;


  loadMVAfromDB_ = cfg.exists("loadMVAfromDB") ? cfg.getParameter<bool>("loadMVAfromDB") : false;
  if ( !loadMVAfromDB_ ) {
    if(cfg.exists("inputFileName")){
      inputFileName_ = cfg.getParameter<edm::FileInPath>("inputFileName");
    }else throw cms::Exception("MVA input not defined") << "Requested to load tau MVA input from ROOT file but no file provided in cfg file";
  }
  if(verbosity_)  std::cout << moduleLabel_ << " loadMVA = " << loadMVAfromDB_ << std::endl;
  if ( cfg.exists("mvaOutput_normalization") ) {
    mvaOutputNormalizationName_ = cfg.getParameter<std::string>("mvaOutput_normalization"); 
  }

  // Setup our cut map
  typedef std::vector<edm::ParameterSet> VPSet;
  VPSet mapping = cfg.getParameter<VPSet>("mapping");
  for ( VPSet::const_iterator mappingEntry = mapping.begin();
	mappingEntry != mapping.end(); ++mappingEntry ) {
    unsigned category = mappingEntry->getParameter<uint32_t>("category");
    std::unique_ptr<DiscriminantCutEntry> cut{new DiscriminantCutEntry()};
    if ( mappingEntry->existsAs<double>("cut") ) {
      cut->cutValue_ = mappingEntry->getParameter<double>("cut");
      cut->mode_ = DiscriminantCutEntry::kFixedCut;
    } else if ( mappingEntry->existsAs<std::string>("cut") ) {
      cut->cutName_ = mappingEntry->getParameter<std::string>("cut");
      std::string cutVariable_string = mappingEntry->getParameter<std::string>("variable");
      cut->cutVariable_.reset( new StringObjectFunction<reco::PFTau>(cutVariable_string.data()) );
      cut->mode_ = DiscriminantCutEntry::kVariableCut;
    } else {
      throw cms::Exception("RecoTauDiscriminantCutMultiplexer") 
        << " Undefined Configuration Parameter 'cut' !!\n";
    }
    cuts_[category] = std::move(cut);
  }

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
  if(verbosity_) std::cout << "constructed " << moduleLabel_ << std::endl;
}

RecoTauDiscriminantCutMultiplexer::~RecoTauDiscriminantCutMultiplexer()
{
}

void RecoTauDiscriminantCutMultiplexer::beginEvent(const edm::Event& evt, const edm::EventSetup& es) 
{
  if(verbosity_) std::cout << " begin! " << moduleLabel_ << " " << isInitialized_ << std::endl;
  if ( !isInitialized_ ) {
    //Only open the file once and we can close it when this routine is done
    // since all objects gotten from the file will have been copied
    std::unique_ptr<TFile> inputFile;
    if ( mvaOutputNormalizationName_ != "" ) {
      if ( !loadMVAfromDB_ ) {
	inputFile = openInputFile(inputFileName_);
	mvaOutput_normalization_ = loadObjectFromFile<TFormula>(*inputFile, mvaOutputNormalizationName_);
      } else {
	auto temp = loadTFormulaFromDB(es, mvaOutputNormalizationName_, Form("%s_mvaOutput_normalization", moduleLabel_.data()), verbosity_);
	mvaOutput_normalization_.reset(temp.release());
      }
    }
    for ( DiscriminantCutMap::iterator cut = cuts_.begin();
	  cut != cuts_.end(); ++cut ) {
      if ( cut->second->mode_ == DiscriminantCutEntry::kVariableCut ) {
	if ( !loadMVAfromDB_ ) {
	  if(not inputFile) {
	    inputFile = openInputFile(inputFileName_);
	  }
	  if(verbosity_) std::cout << "Loading from file" << inputFileName_ << std::endl;
	  cut->second->cutFunction_ = loadObjectFromFile<TGraph>(*inputFile, cut->second->cutName_);
	} else {
	  if(verbosity_) std::cout << "Loading from DB" << std::endl;
	  cut->second->cutFunction_ = loadTGraphFromDB(es, cut->second->cutName_, verbosity_);
	}
      }
    }
    isInitialized_ = true;
  }

  evt.getByToken(toMultiplex_token, toMultiplexHandle_);
  evt.getByToken(key_token, keyHandle_);
}

double
RecoTauDiscriminantCutMultiplexer::discriminate(const reco::PFTauRef& tau) const
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
