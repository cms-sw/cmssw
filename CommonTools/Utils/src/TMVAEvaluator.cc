#include "CommonTools/Utils/interface/TMVAEvaluator.h"

#include "CommonTools/Utils/interface/TMVAZipReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TMVA/MethodBDT.h"


TMVAEvaluator::TMVAEvaluator() :
  mIsInitialized(false), mUsingGBRForest(false), mUseAdaBoost(false)
{
}


void TMVAEvaluator::initialize(const std::string & options, const std::string & method, const std::string & weightFile,
                               const std::vector<std::string> & variables, const std::vector<std::string> & spectators, bool useGBRForest, bool useAdaBoost)
{
  // initialize the TMVA reader
  mReader.reset(new TMVA::Reader(options.c_str()));
  mReader->SetVerbose(false);
  mMethod = method;

  // add input variables
  for(std::vector<std::string>::const_iterator it = variables.begin(); it!=variables.end(); ++it)
  {
    mVariables.insert( std::make_pair( *it, std::make_pair( it - variables.begin(), 0. ) ) );
    mReader->AddVariable(it->c_str(), &(mVariables.at(*it).second));
  }

  // add spectator variables
  for(std::vector<std::string>::const_iterator it = spectators.begin(); it!=spectators.end(); ++it)
  {
    mSpectators.insert( std::make_pair( *it, std::make_pair( it - spectators.begin(), 0. ) ) );
    mReader->AddSpectator(it->c_str(), &(mSpectators.at(*it).second));
  }

  // load the TMVA weights
  mIMethod = std::unique_ptr<TMVA::IMethod>( reco::details::loadTMVAWeights(mReader.get(), mMethod.c_str(), weightFile.c_str()) );

  if (useGBRForest)
  {
    mGBRForest.reset( new GBRForest( dynamic_cast<TMVA::MethodBDT*>( mReader->FindMVA(mMethod.c_str()) ) ) );

    // now can free some memory
    mReader.reset(nullptr);
    mIMethod.reset(nullptr);

    mUsingGBRForest = true;
    mUseAdaBoost = useAdaBoost;
  }

  mIsInitialized = true;
}


void TMVAEvaluator::initializeGBRForest(const GBRForest* gbrForest, const std::vector<std::string> & variables,
                                        const std::vector<std::string> & spectators, bool useAdaBoost)
{
  // add input variables
  for(std::vector<std::string>::const_iterator it = variables.begin(); it!=variables.end(); ++it)
    mVariables.insert( std::make_pair( *it, std::make_pair( it - variables.begin(), 0. ) ) );

  // add spectator variables
  for(std::vector<std::string>::const_iterator it = spectators.begin(); it!=spectators.end(); ++it)
    mSpectators.insert( std::make_pair( *it, std::make_pair( it - spectators.begin(), 0. ) ) );

  // do not take ownership if getting GBRForest from an external source
  mGBRForest = std::shared_ptr<const GBRForest>(gbrForest, [](const GBRForest*) {} );

  mIsInitialized = true;
  mUsingGBRForest = true;
  mUseAdaBoost = useAdaBoost;
}


void TMVAEvaluator::initializeGBRForest(const edm::EventSetup &iSetup, const std::string & label,
                                        const std::vector<std::string> & variables, const std::vector<std::string> & spectators, bool useAdaBoost)
{
  edm::ESHandle<GBRForest> gbrForestHandle;

  iSetup.get<GBRWrapperRcd>().get(label.c_str(), gbrForestHandle);

  initializeGBRForest(gbrForestHandle.product(), variables, spectators, useAdaBoost);
}


float TMVAEvaluator::evaluateTMVA(const std::map<std::string,float> & inputs, bool useSpectators) const
{
  // default value
  float value = -99.;

  // TMVA::Reader is not thread safe
  std::lock_guard<std::mutex> lock(m_mutex);

  // set the input variable values
  for(auto it = mVariables.begin(); it!=mVariables.end(); ++it)
  {
    if (inputs.count(it->first)>0)
      it->second.second = inputs.at(it->first);
    else
      edm::LogError("MissingInputVariable") << "Input variable " << it->first << " is missing from the list of inputs. The returned discriminator value might not be sensible.";
  }

  // if using spectator variables
  if(useSpectators)
  {
    // set the spectator variable values
    for(auto it = mSpectators.begin(); it!=mSpectators.end(); ++it)
    {
      if (inputs.count(it->first)>0)
        it->second.second = inputs.at(it->first);
      else
        edm::LogError("MissingSpectatorVariable") << "Spectator variable " << it->first << " is missing from the list of inputs. The returned discriminator value might not be sensible.";
    }
  }

  // evaluate the MVA
  value = mReader->EvaluateMVA(mMethod.c_str());

  return value;
}


float TMVAEvaluator::evaluateGBRForest(const std::map<std::string,float> & inputs) const
{
  // default value
  float value = -99.;

  std::unique_ptr<float[]> vars(new float[mVariables.size()]); // allocate n floats

  // set the input variable values
  for(auto it = mVariables.begin(); it!=mVariables.end(); ++it)
  {
    if (inputs.count(it->first)>0)
      vars[it->second.first] = inputs.at(it->first);
    else
      edm::LogError("MissingInputVariable") << "Input variable " << it->first << " is missing from the list of inputs. The returned discriminator value might not be sensible.";
  }

  // evaluate the MVA
  if (mUseAdaBoost)
    value = mGBRForest->GetAdaBoostClassifier(vars.get());
  else
    value = mGBRForest->GetGradBoostClassifier(vars.get());

  return value;
}

float TMVAEvaluator::evaluate(const std::map<std::string,float> & inputs, bool useSpectators) const
{
  // default value
  float value = -99.;

  if(!mIsInitialized)
  {
    edm::LogError("InitializationError") << "TMVAEvaluator not properly initialized.";
    return value;
  }

  if( useSpectators && inputs.size() < ( mVariables.size() + mSpectators.size() ) )
  {
    edm::LogError("MissingInputs") << "Too few inputs provided (" << inputs.size() << " provided but " << mVariables.size() << " input and " << mSpectators.size() << " spectator variables expected).";
    return value;
  }
  else if( inputs.size() < mVariables.size() )
  {
    edm::LogError("MissingInputVariable(s)") << "Too few input variables provided (" << inputs.size() << " provided but " << mVariables.size() << " expected).";
    return value;
  }

  if (mUsingGBRForest)
  {
    if(useSpectators)
      edm::LogWarning("UnsupportedFunctionality") << "Use of spectator variables with GBRForest is not supported. Spectator variables will be ignored.";
    value = evaluateGBRForest(inputs);
  }
  else
    value = evaluateTMVA(inputs, useSpectators);

  return value;
}
