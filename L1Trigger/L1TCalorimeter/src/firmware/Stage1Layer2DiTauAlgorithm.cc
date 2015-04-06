///
/// \class l1t::Stage1Layer2DiTauAlgorithm
///
/// \authors:
///           R. Alex Barbieri
///
/// Description: DiTau Algorithm

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2HFRingSumAlgorithmImp.h"
//#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
//#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFRingEtSums.h"

l1t::Stage1Layer2DiTauAlgorithm::Stage1Layer2DiTauAlgorithm(CaloParamsStage1* params) : params_(params)
{
}


l1t::Stage1Layer2DiTauAlgorithm::~Stage1Layer2DiTauAlgorithm() {}


void l1t::Stage1Layer2DiTauAlgorithm::processEvent(const std::vector<l1t::CaloRegion> & regions,
						   const std::vector<l1t::CaloEmCand> & EMCands,
						   const std::vector<l1t::Tau> * taus,
						   l1t::CaloSpare * spares) {

  std::vector<l1t::Tau> *isoTaus = new std::vector<l1t::Tau>();
  for(std::vector<l1t::Tau>::const_iterator itTau = taus->begin();
      itTau != taus->end(); ++itTau){
    if( !itTau->hwIso() ) continue;
    isoTaus->push_back( *itTau );
  }

  int isoPtMax=0;
  int diIsoPtMax=0;
  int triIsoPtMax=0;
  int quadIsoPtMax=0;
  if(isoTaus->size()>0) {
    isoPtMax= (*isoTaus).at(0).hwPt();
    if (isoTaus->size()>1) diIsoPtMax  = (*isoTaus).at(1).hwPt();
    if (isoTaus->size()>2) triIsoPtMax = (*isoTaus).at(2).hwPt();
    if (isoTaus->size()>3) quadIsoPtMax= (*isoTaus).at(3).hwPt();
  }

  int rankIso     = 0;
  int rankDiIso   = 0;
  int rankTriIso  = 0;
  int rankQuadIso = 0;

  bool useLut=true;
  // encode the highest pt Iso and DiIso in the HF ET rings
  if (useLut){
    unsigned int MAX_LUT_ADDRESS = params_->tauEtToHFRingEtLUT()->maxSize()-1;
    unsigned int lutAddress = isoPtMax;
    if (lutAddress > MAX_LUT_ADDRESS) lutAddress = MAX_LUT_ADDRESS;
    rankIso=params_->tauEtToHFRingEtLUT()->data(lutAddress);

    lutAddress = diIsoPtMax;
    if (lutAddress > MAX_LUT_ADDRESS) lutAddress = MAX_LUT_ADDRESS;
    rankDiIso=params_->tauEtToHFRingEtLUT()->data(lutAddress);

    lutAddress = triIsoPtMax;
    if (lutAddress > MAX_LUT_ADDRESS) lutAddress = MAX_LUT_ADDRESS;
    rankTriIso=params_->tauEtToHFRingEtLUT()->data(lutAddress);

    lutAddress = quadIsoPtMax;
    if (lutAddress > MAX_LUT_ADDRESS) lutAddress = MAX_LUT_ADDRESS;
    rankQuadIso=params_->tauEtToHFRingEtLUT()->data(lutAddress);

  }else{
    double etIso     = params_->jetScale().et( isoPtMax );  // convert from hwPt to Physical pT
    double etDiIso   = params_->jetScale().et( diIsoPtMax );
    double etTriIso  = params_->jetScale().et( triIsoPtMax );
    double etQuadIso = params_->jetScale().et( quadIsoPtMax );
    rankIso     = params_->HfRingScale().rank( etIso );  //convert to HfRingScale Rank
    rankDiIso   = params_->HfRingScale().rank( etDiIso );
    rankTriIso  = params_->HfRingScale().rank( etTriIso );
    rankQuadIso = params_->HfRingScale().rank( etQuadIso );
  }

  // std::cout << "Max Iso Tau pT: " << isoPtMax << "\t" << etIso << "\t" << rankIso << std::endl;

  L1GctHFRingEtSums s;
  s.setEtSum(0, rankIso);
  s.setEtSum(1, rankDiIso);
  s.setEtSum(2, rankTriIso);
  s.setEtSum(3, rankQuadIso);
  uint16_t raw = s.raw();
  spares->setHwPt(raw);

  delete isoTaus;


  const bool verbose = false;
  if(verbose)
  {
    std::cout << "HF Ring Sums (Isolated Taus)" << std::endl;
    std::cout << bitset<12>(spares->hwPt()).to_string() << std::endl;
  }
}
