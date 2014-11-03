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

l1t::Stage1Layer2DiTauAlgorithm::Stage1Layer2DiTauAlgorithm(CaloParamsStage1* params) : params_(params)
{
}


l1t::Stage1Layer2DiTauAlgorithm::~Stage1Layer2DiTauAlgorithm() {}


void l1t::Stage1Layer2DiTauAlgorithm::processEvent(const std::vector<l1t::CaloRegion> & regions,
						   const std::vector<l1t::CaloEmCand> & EMCands,
						   const std::vector<l1t::Tau> * taus,
						   l1t::CaloSpare * spares) {

  const size_t nMaxThresh(12); // we have 12 bits to fill as output
  int outputBits =0;

  // int overThreshold[32] = {0};
  //
  // for(std::vector<l1t::Tau>::const_iterator itTau = taus->begin();
  //     itTau != taus->end(); ++itTau){
  //   if( !itTau->hwIso() ) continue;
  //
  //   for(int i = 0; i < 32; ++i)
  //   {
  //     if( itTau->hwPt() > i ) //taus have 4GeV LSB right now, assumed!!
  // 	overThreshold[i]++;
  //   }
  // }
  //
  // for(int i = 0; i < 32; ++i)
  //   if(overThreshold[i] > 1) outputBits = i;


  //int DiIsoThresholds[6] ={12,56,64,72,80,88};

  std::vector<double> DiIsoThresholds = params_->diIsoTauThresholds();
  std::vector<double> IsoThresholds   = params_->isoTauThresholds();
  int nDiIsoThresh=std::min(DiIsoThresholds.size(),nMaxThresh);
  int nIsoThresh= (nDiIsoThresh+IsoThresholds.size()<=nMaxThresh) ? IsoThresholds.size() : (nDiIsoThresh+IsoThresholds.size())-nMaxThresh;

  std::vector<l1t::Tau> *isoTaus = new std::vector<l1t::Tau>();
  for(std::vector<l1t::Tau>::const_iterator itTau = taus->begin();
      itTau != taus->end(); ++itTau){
    if( !itTau->hwIso() ) continue;
    isoTaus->push_back( *itTau );
  }

  int isoPtMax=0;
  int diIsoPtMax=0;
  if(isoTaus->size()>0) {
    isoPtMax= (*isoTaus).at(0).hwPt();
    if (isoTaus->size()>1) diIsoPtMax= (*isoTaus).at(1).hwPt();
  }

  int isowd=0;
  for (int i= 0; i<nIsoThresh; i++){
    // std::cout << "IsoThresholds: " << IsoThresholds.at(i) << std::endl;
    int thresh=IsoThresholds.at(i)/4;
    if (isoPtMax >= thresh) isowd |= 1 << i;
  }

  int diIsowd=0;
  for (int i= 0; i<nDiIsoThresh; i++){
    // std::cout << "DiIsoThresholds: " << DiIsoThresholds.at(i) << std::endl;
    int thresh=DiIsoThresholds.at(i)/4;
    if (diIsoPtMax >= thresh) diIsowd |= 1 << i;
  }

  outputBits=((isowd & 0x7ff)<< 9) | (diIsowd & 0x7ff);

  // ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > dummy(0,0,0,0);

  // l1t::CaloSpare ditau (*&dummy,CaloSpare::CaloSpareType::Tau,outputBits,0,0,0);

  spares->setHwPt(outputBits);
}
