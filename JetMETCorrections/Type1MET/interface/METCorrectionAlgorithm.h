#ifndef JetMETCorrections_Type1MET_METCorrectionAlgorithm_h
#define JetMETCorrections_Type1MET_METCorrectionAlgorithm_h

/** \class METCorrectionAlgorithm
 *
 * Algorithm for 
 *  o propagating jet energy corrections to MET (Type 1 MET corrections)
 *  o calibrating momentum of particles not within jets ("unclustered energy")
 *    and propagating those corrections to MET (Type 2 MET corrections)
 *
 * \authors Michael Schmitt, Richard Cavanaugh, The University of Florida
 *          Florent Lacroix, University of Illinois at Chicago
 *          Christian Veelken, LLR
 *          Tai Sakuma, Texas A&M
 *
 *
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include <TFormula.h>

#include <vector>

class METCorrectionAlgorithm {
public:
  explicit METCorrectionAlgorithm(const edm::ParameterSet&, edm::ConsumesCollector&& iConsumesCollector);
  ~METCorrectionAlgorithm();

  CorrMETData compMETCorrection(edm::Event&);

private:
  typedef std::vector<edm::InputTag> vInputTag;

  std::vector<edm::EDGetTokenT<CorrMETData> > chsSumTokens_;
  std::vector<edm::EDGetTokenT<CorrMETData> > type1Tokens_;

  bool applyType0Corrections_;
  bool applyType1Corrections_;
  bool applyType2Corrections_;

  double type0Rsoft_;
  double type0Cuncl_;

  typedef std::vector<std::string> vstring;
  struct type2BinningEntryType {
    type2BinningEntryType(const std::string& binCorrformula,
                          const edm::ParameterSet& binCorrParameter,
                          const vInputTag& srcUnclEnergySums,
                          edm::ConsumesCollector& iConsumesCollector)
        : binLabel_(""), binCorrFormula_(nullptr) {
      for (vInputTag::const_iterator inputTag = srcUnclEnergySums.begin(); inputTag != srcUnclEnergySums.end();
           ++inputTag) {
        corrTokens_.push_back(iConsumesCollector.consumes<CorrMETData>(*inputTag));
      }

      initialize(binCorrformula, binCorrParameter);
    }
    type2BinningEntryType(const edm::ParameterSet& cfg,
                          const vInputTag& srcUnclEnergySums,
                          edm::ConsumesCollector& iConsumesCollector)
        : binLabel_(cfg.getParameter<std::string>("binLabel")), binCorrFormula_(nullptr) {
      for (vInputTag::const_iterator srcUnclEnergySum = srcUnclEnergySums.begin();
           srcUnclEnergySum != srcUnclEnergySums.end();
           ++srcUnclEnergySum) {
        std::string instanceLabel = srcUnclEnergySum->instance();
        if (!instanceLabel.empty() && !binLabel_.empty())
          instanceLabel.append("#");
        instanceLabel.append(binLabel_);
        edm::InputTag inputTag(srcUnclEnergySum->label(), instanceLabel);
        corrTokens_.push_back(iConsumesCollector.consumes<CorrMETData>(inputTag));
      }

      std::string binCorrFormula = cfg.getParameter<std::string>("binCorrFormula");

      edm::ParameterSet binCorrParameter = cfg.getParameter<edm::ParameterSet>("binCorrParameter");

      initialize(binCorrFormula, binCorrParameter);
    }
    void initialize(const std::string& binCorrFormula, const edm::ParameterSet& binCorrParameter) {
      TString formula = binCorrFormula;

      vstring parNames = binCorrParameter.getParameterNamesForType<double>();
      int numParameter = parNames.size();
      binCorrParameter_.resize(numParameter);
      for (int parIndex = 0; parIndex < numParameter; ++parIndex) {
        const std::string& parName = parNames[parIndex];

        double parValue = binCorrParameter.getParameter<double>(parName);
        binCorrParameter_[parIndex] = parValue;

        TString parName_internal = Form("[%i]", parIndex);
        formula = formula.ReplaceAll(parName.data(), parName_internal);
      }

      std::string binCorrFormulaName = std::string("binCorrFormula").append("_").append(binLabel_);
      binCorrFormula_ = new TFormula(binCorrFormulaName.data(), formula);

      //--- check that syntax of formula string is valid
      //   (i.e. that TFormula "compiled" without errors)
      if (!(binCorrFormula_->GetNdim() <= 1 && binCorrFormula_->GetNpar() == numParameter))
        throw cms::Exception("METCorrectionAlgorithm")
            << "Formula for Type 2 correction has invalid syntax = " << formula << " !!\n";

      for (int parIndex = 0; parIndex < numParameter; ++parIndex) {
        binCorrFormula_->SetParameter(parIndex, binCorrParameter_[parIndex]);
        binCorrFormula_->SetParName(parIndex, parNames[parIndex].data());
      }
    }
    ~type2BinningEntryType() { delete binCorrFormula_; }
    std::string binLabel_;
    std::vector<edm::EDGetTokenT<CorrMETData> > corrTokens_;
    TFormula* binCorrFormula_;
    std::vector<double> binCorrParameter_;
  };
  std::vector<type2BinningEntryType*> type2Binning_;
};

#endif
