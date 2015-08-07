#include "CommonTools/Utils/interface/TMVAEvaluator.h"

#include "CommonTools/Utils/interface/TMVAZipReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


TMVAEvaluator::TMVAEvaluator() :
  mIsInitialized(false)
{
}


TMVAEvaluator::~TMVAEvaluator()
{
}


void TMVAEvaluator::initialize(const std::string & options, const std::string & method, const std::string & weightFile,
                               const std::vector<std::string> & variables, const std::vector<std::string> & spectators)
{
  // initialize the TMVA reader
  mReader.reset(new TMVA::Reader(options.c_str()));
  mReader->SetVerbose(false);
  mMethod = method;

  // add input variables
  for(std::vector<std::string>::const_iterator it = variables.begin(); it!=variables.end(); ++it)
  {
    mVariables.insert( std::pair<std::string,float>(*it,0.) );
    mReader->AddVariable(it->c_str(), &mVariables.at(*it));
  }

  // add spectator variables
  for(std::vector<std::string>::const_iterator it = spectators.begin(); it!=spectators.end(); ++it)
  {
    mSpectators.insert( std::pair<std::string,float>(*it,0.) );
    mReader->AddSpectator(it->c_str(), &mSpectators.at(*it));
  }

  // load the TMVA weights
  reco::details::loadTMVAWeights(mReader.get(), mMethod.c_str(), weightFile.c_str());

  mIsInitialized = true;
}


float TMVAEvaluator::evaluate(const std::map<std::string,float> & inputs, const bool useSpectators)
{
  if(!mIsInitialized)
  {
    edm::LogError("InitializationError") << "TMVAEvaluator not properly initialized.";
    return -99.;
  }

  if( useSpectators && inputs.size() < ( mVariables.size() + mSpectators.size() ) )
  {
    edm::LogError("MissingInputs") << "Too few inputs provided (" << inputs.size() << " provided but " << mVariables.size() << " input and " << mSpectators.size() << " spectator variables expected).";
    return -99.;
  }
  else if( inputs.size() < mVariables.size() )
  {
    edm::LogError("MissingInputVariable(s)") << "Too few input variables provided (" << inputs.size() << " provided but " << mVariables.size() << " expected).";
    return -99.;
  }

  // set the input variable values
  for(std::map<std::string,float>::iterator it = mVariables.begin(); it!=mVariables.end(); ++it)
  {
    if (inputs.count(it->first)>0)
      it->second = inputs.at(it->first);
    else
      edm::LogError("MissingInputVariable") << "Input variable " << it->first << " is missing from the list of inputs. The returned discriminator value might not be sensible.";
  }

  // if using spectator variables
  if(useSpectators)
  {
    // set the spectator variable values
    for(std::map<std::string,float>::iterator it = mSpectators.begin(); it!=mSpectators.end(); ++it)
    {
      if (inputs.count(it->first)>0)
        it->second = inputs.at(it->first);
      else
        edm::LogError("MissingSpectatorVariable") << "Spectator variable " << it->first << " is missing from the list of inputs. The returned discriminator value might not be sensible.";
    }
  }

  // evaluate the MVA
  float value = mReader->EvaluateMVA(mMethod.c_str());

  return value;
}
