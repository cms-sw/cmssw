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

  // encode the highest pt Iso and DiIso in the HF ET rings
  double etIso     = params_->jetScale().et( isoPtMax );  // convert from hwPt to Physical pT
  double etDiIso   = params_->jetScale().et( diIsoPtMax );
  double etTriIso  = params_->jetScale().et( triIsoPtMax );
  double etQuadIso = params_->jetScale().et( quadIsoPtMax );
  int rankIso     = params_->HfRingScale().rank( etIso );  //convert to HfRingScale Rank
  int rankDiIso   = params_->HfRingScale().rank( etDiIso );
  int rankTriIso  = params_->HfRingScale().rank( etTriIso );
  int rankQuadIso = params_->HfRingScale().rank( etQuadIso );


  L1GctHFRingEtSums s;
  s.setEtSum(0, rankIso);
  s.setEtSum(1, rankDiIso);
  s.setEtSum(2, rankTriIso);
  s.setEtSum(3, rankQuadIso);
  uint16_t raw = s.raw();
  spares->setHwPt(raw);

  delete isoTaus;
}
