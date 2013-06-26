// -*- C++ -*-
// $Id: Type2CorrectionProducer.cc,v 1.1 2013/01/15 06:49:10 sakuma Exp $

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include <TFormula.h>

#include <iostream>
#include <vector>

//____________________________________________________________________________||
class Type2CorrectionProducer : public edm::EDProducer
{
public:
  explicit Type2CorrectionProducer(const edm::ParameterSet&);
  ~Type2CorrectionProducer() { }

private:

  typedef std::vector<edm::InputTag> vInputTag;

  typedef std::vector<std::string> vstring;
  struct type2BinningEntryType
  {
    type2BinningEntryType(const std::string& binCorrformula, const edm::ParameterSet& binCorrParameter, const vInputTag& srcUnclEnergySums)
      : binLabel_(""),
        srcUnclEnergySums_(srcUnclEnergySums),
        binCorrFormula_(0)
    {
      initialize(binCorrformula, binCorrParameter);
    }
    type2BinningEntryType(const edm::ParameterSet& cfg, const vInputTag& srcUnclEnergySums)
      : binLabel_(cfg.getParameter<std::string>("binLabel")),
        binCorrFormula_(0)
    {
      for ( vInputTag::const_iterator srcUnclEnergySum = srcUnclEnergySums.begin();
	    srcUnclEnergySum != srcUnclEnergySums.end(); ++srcUnclEnergySum ) {
	std::string instanceLabel = srcUnclEnergySum->instance();
	if ( instanceLabel != "" && binLabel_ != "" ) instanceLabel.append("#");
	instanceLabel.append(binLabel_);
	srcUnclEnergySums_.push_back(edm::InputTag(srcUnclEnergySum->label(), instanceLabel));
      }
      
      std::string binCorrFormula = cfg.getParameter<std::string>("binCorrFormula").data();
    
      edm::ParameterSet binCorrParameter = cfg.getParameter<edm::ParameterSet>("binCorrParameter");

      initialize(binCorrFormula, binCorrParameter);
    }
    void initialize(const std::string& binCorrFormula, const edm::ParameterSet& binCorrParameter)
    {
      TString formula = binCorrFormula;
    
      vstring parNames = binCorrParameter.getParameterNamesForType<double>();
      int numParameter = parNames.size();
      binCorrParameter_.resize(numParameter);    
      for ( int parIndex = 0; parIndex < numParameter; ++parIndex ) {
        const std::string& parName = parNames[parIndex].data();

        double parValue = binCorrParameter.getParameter<double>(parName);
        binCorrParameter_[parIndex] = parValue;

        TString parName_internal = Form("[%i]", parIndex);
        formula = formula.ReplaceAll(parName.data(), parName_internal);
      }

      std::string binCorrFormulaName = std::string("binCorrFormula").append("_").append(binLabel_); 
      binCorrFormula_ = new TFormula(binCorrFormulaName.data(), formula);

//--- check that syntax of formula string is valid 
//   (i.e. that TFormula "compiled" without errors)
      if ( !(binCorrFormula_->GetNdim() <= 1 && binCorrFormula_->GetNpar() == numParameter) ) 
	throw cms::Exception("METCorrectionAlgorithm2") 
	  << "Formula for Type 2 correction has invalid syntax = " << formula << " !!\n";

      for ( int parIndex = 0; parIndex < numParameter; ++parIndex ) {
	binCorrFormula_->SetParameter(parIndex, binCorrParameter_[parIndex]);
	binCorrFormula_->SetParName(parIndex, parNames[parIndex].data());
      }
    }
    ~type2BinningEntryType() 
    {
      delete binCorrFormula_;
    }
    std::string binLabel_;
    vInputTag srcUnclEnergySums_;
    TFormula* binCorrFormula_;
    std::vector<double> binCorrParameter_;
  };

  std::vector<type2BinningEntryType*> type2Binning_;

  void produce(edm::Event&, const edm::EventSetup&);

};

//____________________________________________________________________________||
Type2CorrectionProducer::Type2CorrectionProducer(const edm::ParameterSet& cfg)
{
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
  produces<CorrMETData>("");
}

//____________________________________________________________________________||
void Type2CorrectionProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  CorrMETData product;

  for ( std::vector<type2BinningEntryType*>::const_iterator type2BinningEntry = type2Binning_.begin();
	type2BinningEntry != type2Binning_.end(); ++type2BinningEntry ) {
    CorrMETData unclEnergySum;
    for (vInputTag::const_iterator inputTag = (*type2BinningEntry)->srcUnclEnergySums_.begin();
	 inputTag != (*type2BinningEntry)->srcUnclEnergySums_.end(); ++inputTag) {
      edm::Handle<CorrMETData> unclEnergySummand;
      evt.getByLabel(*inputTag, unclEnergySummand);

      unclEnergySum += (*unclEnergySummand);
    }

    double unclEnergySumPt = sqrt(unclEnergySum.mex*unclEnergySum.mex + unclEnergySum.mey*unclEnergySum.mey);
    double unclEnergyScaleFactor = (*type2BinningEntry)->binCorrFormula_->Eval(unclEnergySumPt);

    unclEnergySum.mex = -unclEnergySum.mex;
    unclEnergySum.mey = -unclEnergySum.mey;

    product += (unclEnergyScaleFactor - 1.)*unclEnergySum;
  }

  std::auto_ptr<CorrMETData> pprod(new CorrMETData(product));
  evt.put(pprod, "");
}

//____________________________________________________________________________||

DEFINE_FWK_MODULE(Type2CorrectionProducer);

