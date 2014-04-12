#include "JetMETCorrections/Type1MET/interface/METCorrectionAlgorithm.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <TString.h>

#include <string>

METCorrectionAlgorithm::METCorrectionAlgorithm(const edm::ParameterSet& cfg, edm::ConsumesCollector && iConsumesCollector)
{
  applyType1Corrections_ = cfg.getParameter<bool>("applyType1Corrections");
  if ( applyType1Corrections_ )
    {
      vInputTag srcType1Corrections = cfg.getParameter<vInputTag>("srcType1Corrections");
      for (vInputTag::const_iterator inputTag = srcType1Corrections.begin(); inputTag != srcType1Corrections.end(); ++inputTag)
	{
	  type1Tokens_.push_back(iConsumesCollector.consumes<CorrMETData>(*inputTag));
	}
    }
  
  applyType2Corrections_ = cfg.getParameter<bool>("applyType2Corrections");
  if ( applyType2Corrections_ )
    {
      vInputTag srcUnclEnergySums = cfg.getParameter<vInputTag>("srcUnclEnergySums");
    
      if ( cfg.exists("type2Binning") ) 
	{
	  typedef std::vector<edm::ParameterSet> vParameterSet;
	  vParameterSet cfgType2Binning = cfg.getParameter<vParameterSet>("type2Binning");
	  for ( vParameterSet::const_iterator cfgType2BinningEntry = cfgType2Binning.begin();
		cfgType2BinningEntry != cfgType2Binning.end(); ++cfgType2BinningEntry ) {
	    type2Binning_.push_back(new type2BinningEntryType(*cfgType2BinningEntry, srcUnclEnergySums, iConsumesCollector));
	  }
	}
      else
	{
	  std::string type2CorrFormula = cfg.getParameter<std::string>("type2CorrFormula").data();
	  edm::ParameterSet type2CorrParameter = cfg.getParameter<edm::ParameterSet>("type2CorrParameter");
	  type2Binning_.push_back(new type2BinningEntryType(type2CorrFormula, type2CorrParameter, srcUnclEnergySums, iConsumesCollector));
	}
    }

  applyType0Corrections_ = cfg.exists("applyType0Corrections") ? cfg.getParameter<bool>("applyType0Corrections") : false;
  if ( applyType0Corrections_ )
    {
      vInputTag srcCHSSums = cfg.getParameter<vInputTag>("srcCHSSums");
      for (vInputTag::const_iterator inputTag = srcCHSSums.begin(); inputTag != srcCHSSums.end(); ++inputTag)
	{
	  chsSumTokens_.push_back(iConsumesCollector.consumes<CorrMETData>(*inputTag));
	}

      type0Rsoft_ = cfg.getParameter<double>("type0Rsoft");
      type0Cuncl_ = 1.0;
      if (applyType2Corrections_)
	{
	  if (cfg.exists("type2Binning")) throw cms::Exception("Invalid Arg") << "Currently, applyType0Corrections and type2Binning cannot be used together!";
	  std::string type2CorrFormula = cfg.getParameter<std::string>("type2CorrFormula").data();
	  if (!(type2CorrFormula == "A")) throw cms::Exception("Invalid Arg") << "type2CorrFormula must be \"A\" if applyType0Corrections!";
	  edm::ParameterSet type2CorrParameter = cfg.getParameter<edm::ParameterSet>("type2CorrParameter");
	  type0Cuncl_ = type2CorrParameter.getParameter<double>("A");
	}
    }
}

METCorrectionAlgorithm::~METCorrectionAlgorithm()
{
  for ( std::vector<type2BinningEntryType*>::const_iterator it = type2Binning_.begin();
	it != type2Binning_.end(); ++it ) {
    delete (*it);
  }
}

CorrMETData METCorrectionAlgorithm::compMETCorrection(edm::Event& evt, const edm::EventSetup& es)
{
  CorrMETData metCorr;
  metCorr.mex   = 0.;
  metCorr.mey   = 0.;
  metCorr.sumet = 0.;

  if ( applyType0Corrections_ ) {
//--- sum all Type 0 MET correction terms
    edm::Handle<CorrMETData> chsSum;
    for (std::vector<edm::EDGetTokenT<CorrMETData> >::const_iterator corrToken = chsSumTokens_.begin(); corrToken != chsSumTokens_.end(); ++corrToken)
      {
	evt.getByToken(*corrToken, chsSum);

	metCorr.mex   += type0Cuncl_*(1 - type0Rsoft_)*chsSum->mex;
	metCorr.mey   += type0Cuncl_*(1 - type0Rsoft_)*chsSum->mey;
	metCorr.sumet += type0Cuncl_*(1 - type0Rsoft_)*chsSum->sumet;
      }
  }

  if ( applyType1Corrections_ ) {
//--- sum all Type 1 MET correction terms
    edm::Handle<CorrMETData> type1Correction;
    for (std::vector<edm::EDGetTokenT<CorrMETData> >::const_iterator corrToken = type1Tokens_.begin(); corrToken != type1Tokens_.end(); ++corrToken)
      {
	evt.getByToken(*corrToken, type1Correction);

	metCorr.mex   += type1Correction->mex;
	metCorr.mey   += type1Correction->mey;
	metCorr.sumet += type1Correction->sumet;
      }
  }

  if ( applyType2Corrections_ )
    {
//--- compute momentum sum of all "unclustered energy" in the event
//
//    NOTE: calibration factors/formulas for Type 2 MET correction may depend on eta
//         (like the jet energy correction factors do)
//
    
      for ( std::vector<type2BinningEntryType*>::const_iterator type2BinningEntry = type2Binning_.begin();
	  type2BinningEntry != type2Binning_.end(); ++type2BinningEntry )
	{
	  CorrMETData unclEnergySum;
	  edm::Handle<CorrMETData> unclEnergySummand;
	  for (std::vector<edm::EDGetTokenT<CorrMETData> >::const_iterator corrToken = (*type2BinningEntry)->corrTokens_.begin(); corrToken != (*type2BinningEntry)->corrTokens_.end(); ++corrToken)
	    {
	      evt.getByToken(*corrToken, unclEnergySummand);
	
	      unclEnergySum.mex   += unclEnergySummand->mex;
	      unclEnergySum.mey   += unclEnergySummand->mey;
	      unclEnergySum.sumet += unclEnergySummand->sumet;
	    }

//--- calibrate "unclustered energy"
	  double unclEnergySumPt = sqrt(unclEnergySum.mex*unclEnergySum.mex + unclEnergySum.mey*unclEnergySum.mey);
	  double unclEnergyScaleFactor = (*type2BinningEntry)->binCorrFormula_->Eval(unclEnergySumPt);

//--- MET balances momentum of reconstructed particles,
//    hence correction to "unclustered energy" and corresponding Type 2 MET correction are of opposite sign
	  metCorr.mex   -= (unclEnergyScaleFactor - 1.)*unclEnergySum.mex;
	  metCorr.mey   -= (unclEnergyScaleFactor - 1.)*unclEnergySum.mey;
	  metCorr.sumet += (unclEnergyScaleFactor - 1.)*unclEnergySum.sumet;
	}
    }

  return metCorr;
}
