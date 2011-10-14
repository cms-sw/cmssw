#include "JetMETCorrections/Type1MET/interface/METCorrectionAlgorithm.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <TString.h>

#include <string>

METCorrectionAlgorithm::METCorrectionAlgorithm(const edm::ParameterSet& cfg)
{
  applyType1Corrections_ = cfg.getParameter<bool>("applyType1Corrections");
  if ( applyType1Corrections_ ) {
    srcType1Corrections_ = cfg.getParameter<vInputTag>("srcType1Corrections");
  }
  
  applyType2Corrections_ = cfg.getParameter<bool>("applyType2Corrections");
  if ( applyType2Corrections_ ) {
    vInputTag srcUnclEnergySums = cfg.getParameter<vInputTag>("srcUnclEnergySums");
    
    if ( cfg.exists("type2Binning") ) {
      typedef std::vector<edm::ParameterSet> vParameterSet;
      vParameterSet cfgType2Binning = cfg.getParameter<vParameterSet>("type2Binning");
      for ( vParameterSet::const_iterator cfgType2BinningEntry = cfgType2Binning.begin();
	    cfgType2BinningEntry != cfgType2Binning.end(); ++cfgType2BinningEntry ) {
	type2Binning_.push_back(new type2BinningEntryType(*cfgType2BinningEntry, srcUnclEnergySums));
      }
    } else {
      std::string type2CorrFormula = cfg.getParameter<std::string>("type2CorrFormula").data();
      edm::ParameterSet type2CorrParameter = cfg.getParameter<edm::ParameterSet>("type2CorrParameter");
      type2Binning_.push_back(new type2BinningEntryType(type2CorrFormula, type2CorrParameter, srcUnclEnergySums));
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

  if ( applyType1Corrections_ ) {
//--- sum all Type 1 MET correction terms
    for ( vInputTag::const_iterator srcType1Correction = srcType1Corrections_.begin();
	  srcType1Correction != srcType1Corrections_.end(); ++srcType1Correction ) {
      edm::Handle<CorrMETData> type1Correction;
      evt.getByLabel(*srcType1Correction, type1Correction);

      metCorr.mex   += type1Correction->mex;
      metCorr.mey   += type1Correction->mey;
      metCorr.sumet += type1Correction->sumet;
    }
  }

  if ( applyType2Corrections_ ) {
//--- compute momentum sum of all "unclustered energy" in the event
//
//    NOTE: calibration factors/formulas for Type 2 MET correction may depend on eta
//         (like the jet energy correction factors do)
//
    for ( std::vector<type2BinningEntryType*>::const_iterator type2BinningEntry = type2Binning_.begin();
	  type2BinningEntry != type2Binning_.end(); ++type2BinningEntry ) {
      CorrMETData unclEnergySum;
      for ( vInputTag::const_iterator srcUnclEnergySum = (*type2BinningEntry)->srcUnclEnergySums_.begin();
	    srcUnclEnergySum != (*type2BinningEntry)->srcUnclEnergySums_.end(); ++srcUnclEnergySum ) {
	edm::Handle<CorrMETData> unclEnergySummand;
	evt.getByLabel(*srcUnclEnergySum, unclEnergySummand);
	
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
