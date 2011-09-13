#include "JetMETCorrections/Type1MET/interface/METCorrectionAlgorithm.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <TString.h>

#include <string>

typedef std::vector<std::string> vstring;

METCorrectionAlgorithm::METCorrectionAlgorithm(const edm::ParameterSet& cfg)
  : type2CorrFormula_(0)
{
  applyType1Corrections_ = cfg.getParameter<bool>("applyType1Corrections");
  if ( applyType1Corrections_ ) {
    srcType1Corrections_ = cfg.getParameter<vInputTag>("srcType1Corrections");
  }
  
  applyType2Corrections_ = cfg.getParameter<bool>("applyType2Corrections");
  if ( applyType2Corrections_ ) {
    srcUnclEnergySums_ = cfg.getParameter<vInputTag>("srcUnclEnergySums");
    
    TString formula_string = cfg.getParameter<std::string>("type2CorrFormula").data();
    
    edm::ParameterSet cfgParameter = cfg.getParameter<edm::ParameterSet>("type2CorrParameter");
    vstring parNames = cfgParameter.getParameterNamesForType<double>();
    int numParameter = parNames.size();
    type2CorrParameter_.resize(numParameter);    
    for ( int parIndex = 0; parIndex < numParameter; ++parIndex ) {
      const std::string& parName = parNames[parIndex].data();

      double parValue = cfgParameter.getParameter<double>(parName);
      type2CorrParameter_[parIndex] = parValue;

      TString parName_internal = Form("[%i]", parIndex);
      formula_string = formula_string.ReplaceAll(parName.data(), parName_internal);
    }

    type2CorrFormula_ = new TFormula("type2CorrFormula", formula_string);

//--- check that syntax of formula string is valid 
//   (i.e. that TFormula "compiled" without errors)
    if ( type2CorrFormula_->GetNdim() != numParameter ) 
      throw cms::Exception("METCorrectionAlgorithm") 
	<< "Formula for Type 2 correction has invalid syntax = " << formula_string << " !!\n";

    for ( int parIndex = 0; parIndex < numParameter; ++parIndex ) {
      type2CorrFormula_->SetParameter(parIndex, type2CorrParameter_[parIndex]);
      type2CorrFormula_->SetParName(parIndex, parNames[parIndex].data());
    }
  }
}

METCorrectionAlgorithm::~METCorrectionAlgorithm()
{
  delete type2CorrFormula_;
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

      metCorr.mex   += type1Correction->mey;
      metCorr.mey   += type1Correction->mex;
      metCorr.sumet += type1Correction->sumet;
    }
  }

  std::cout << "type I correction: px = " << metCorr.mex << ", py = " << metCorr.mey << std::endl;

  if ( applyType2Corrections_ ) {
//--- compute momentum sum of all "unclustered energy" in the event
    CorrMETData unclEnergySum;
    for ( vInputTag::const_iterator srcUnclEnergySum = srcUnclEnergySums_.begin();
	  srcUnclEnergySum != srcUnclEnergySums_.end(); ++srcUnclEnergySum ) {
      edm::Handle<CorrMETData> unclEnergySummand;
      evt.getByLabel(*srcUnclEnergySum, unclEnergySummand);

      std::cout << srcUnclEnergySum->label() << ": px = " << unclEnergySummand->mex << "," 
		<< " py = " << unclEnergySummand->mey << std::endl;

      unclEnergySum.mex   += unclEnergySummand->mex;
      unclEnergySum.mey   += unclEnergySummand->mey;
      unclEnergySum.sumet += unclEnergySummand->sumet;
    }

    std::cout << " unclEnergySum: px = " << unclEnergySum.mex << ", py = " << unclEnergySum.mey << std::endl;

//--- calibrate "unclustered energy"
    double unclEnergySumPt = sqrt(unclEnergySum.mex*unclEnergySum.mex + unclEnergySum.mey*unclEnergySum.mey);
    std::cout << " unclEnergySumPt = " << unclEnergySumPt << std::endl;
    double unclEnergyScaleFactor = type2CorrFormula_->Eval(unclEnergySumPt);
    std::cout << " unclEnergyScaleFactor = " << unclEnergyScaleFactor << std::endl;

    std::cout << "type II correction: px = " << -(unclEnergyScaleFactor - 1.)*unclEnergySum.mex << ","
	      << " py = " << -(unclEnergyScaleFactor - 1.)*unclEnergySum.mey << std::endl;

//--- MET balances momentum of reconstructed particles,
//    hence correction to "unclustered energy" and corresponding Type 2 MET correction are of opposite sign
    metCorr.mex   -= (unclEnergyScaleFactor - 1.)*unclEnergySum.mex;
    metCorr.mey   -= (unclEnergyScaleFactor - 1.)*unclEnergySum.mey;
    metCorr.sumet += unclEnergySum.sumet;
  }

  return metCorr;

}
