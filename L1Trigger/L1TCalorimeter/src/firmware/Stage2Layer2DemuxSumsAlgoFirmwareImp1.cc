///
/// \class l1t::Stage2Layer2SumsAlgorithmFirmwareImp1
///
/// \author:
///
/// Description:

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxSumsAlgoFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include <vector>
#include <algorithm>


l1t::Stage2Layer2DemuxSumsAlgoFirmwareImp1::Stage2Layer2DemuxSumsAlgoFirmwareImp1(CaloParamsHelper const* params) :
  params_(params), cordic_(Cordic(144*16,17,8))  // These are the settings in the hardware - should probably make this configurable
{
}


void l1t::Stage2Layer2DemuxSumsAlgoFirmwareImp1::processEvent(const std::vector<l1t::EtSum> & inputSums,
                                                              std::vector<l1t::EtSum> & outputSums) {

  int et(0), etHF(0), etHFOnly(0), etem(0), metx(0), mety(0), metxHF(0), metyHF(0), ht(0), htHF(0), mhtx(0), mhty(0), mhtxHF(0), mhtyHF(0), metPhi(0), metPhiHF(0), mhtPhi(0), mhtPhiHF(0);
  double etPos(0), etNeg(0), etHFPos(0), etHFNeg(0), htPos(0), htNeg(0), htHFPos(0), htHFNeg(0);
  int cent(0);
  unsigned int asymEt(0), asymEtHF(0), asymHt(0), asymHtHF(0);
  bool posEt(false), posEtHF(false), posHt(false), posHtHF(false);
  unsigned int met(0), metHF(0), mht(0), mhtHF(0);
  unsigned int mbp0(0), mbm0(0), mbp1(0), mbm1(0);
  unsigned int ntow(0);

  bool metSat(false), metHFSat(false), mhtSat(false), mhtHFSat(false);

  // Add up the x, y and scalar components
  for (auto&& eSum : inputSums)
  {
      switch (eSum.getType()) {

      case l1t::EtSum::EtSumType::kTotalEt:
	et += eSum.hwPt();
	if(posEt) etPos = eSum.hwPt();
	else {
	  etNeg = eSum.hwPt();
	  posEt = true;
	}
        break;

      case l1t::EtSum::EtSumType::kTotalEtHF:
        etHF += eSum.hwPt();
	if(posEtHF) etHFPos = eSum.hwPt();
	else {
	  etHFNeg = eSum.hwPt();
	  posEtHF = true;
	}
        break;

      case l1t::EtSum::EtSumType::kTotalEtEm:
        etem += eSum.hwPt();
        break;

      case l1t::EtSum::EtSumType::kTotalEtx:
	if(eSum.hwPt()==0x7fffffff) metSat=true;
        else metx += eSum.hwPt();
        break;

      case l1t::EtSum::EtSumType::kTotalEty:
	if(eSum.hwPt()==0x7fffffff) metSat=true;
        else mety += eSum.hwPt();
        break;

      case l1t::EtSum::EtSumType::kTotalHt:
        ht += eSum.hwPt();
	if(posHt) htPos = eSum.hwPt();
	else {
	  htNeg = eSum.hwPt();
	  posHt = true;
	}
        break;

      case l1t::EtSum::EtSumType::kTotalHtHF:
        htHF += eSum.hwPt();
	if(posHtHF) htHFPos = eSum.hwPt();
	else {
	  htHFNeg = eSum.hwPt();
	  posHtHF = true;
	}
        break;

      case l1t::EtSum::EtSumType::kTotalHtx:
	if(eSum.hwPt()==0x7fffffff) mhtSat=true;
        else mhtx += eSum.hwPt();
	break;

      case l1t::EtSum::EtSumType::kTotalHty:
	if(eSum.hwPt()==0x7fffffff) mhtSat=true;
	else mhty += eSum.hwPt();
        break;
	
      case l1t::EtSum::EtSumType::kTotalEtxHF:
	if(eSum.hwPt()==0x7fffffff) metHFSat=true;
        else metxHF += eSum.hwPt();
        break;
	
      case l1t::EtSum::EtSumType::kTotalEtyHF:
	if(eSum.hwPt()==0x7fffffff) metHFSat=true;
        else metyHF += eSum.hwPt();
        break;

      case l1t::EtSum::EtSumType::kTotalHtxHF:
	if(eSum.hwPt()==0x7fffffff) mhtHFSat=true;
	else mhtxHF += eSum.hwPt();
        break;
	
      case l1t::EtSum::EtSumType::kTotalHtyHF:
	if(eSum.hwPt()==0x7fffffff) mhtHFSat=true;
	else mhtyHF += eSum.hwPt();
        break;

      case l1t::EtSum::EtSumType::kMinBiasHFP0:
	mbp0 = eSum.hwPt();
	break;

      case l1t::EtSum::EtSumType::kMinBiasHFM0:
	mbm0 = eSum.hwPt();
	break;

      case l1t::EtSum::EtSumType::kMinBiasHFP1:
	mbp1 = eSum.hwPt();
	break;

      case l1t::EtSum::EtSumType::kMinBiasHFM1:
	mbm1 = eSum.hwPt();
	break;

      case l1t::EtSum::EtSumType::kTowerCount:
	ntow = eSum.hwPt();
	break;

      default:
        continue; // Should throw an exception or something?
      }
  }

  // calculate centrality
  etHFOnly = abs(etHF - et);
  for(uint i=0; i<8; ++i){
    if(etHFOnly >= (params_->etSumCentLower(i)/params_->towerLsbSum())
       && etHFOnly <= (params_->etSumCentUpper(i)/params_->towerLsbSum())){
      cent |= 1 << i;
    }
  }

  // calculate HI imbalance
  asymEt   = l1t::CaloTools::gloriousDivision(abs(etPos-etNeg), et);
  asymEtHF = l1t::CaloTools::gloriousDivision(abs(etHFPos-etHFNeg), etHF);
  asymHt   = l1t::CaloTools::gloriousDivision(abs(htPos-htNeg), ht);
  asymHtHF = l1t::CaloTools::gloriousDivision(abs(htHFPos-htHFNeg), htHF);

  if (et>0xFFF)   et   = 0xFFF;
  if (etHF>0xFFF) etHF = 0xFFF;
  if (etem>0xFFF) etem = 0xFFF;
  if (ht>0xFFF)   ht   = 0xFFF;
  if (htHF>0xFFF) htHF = 0xFFF;

  if(et == 0xFFF) asymEt   = 0xFF;
  if(etHF == 0xFFF) asymEtHF = 0xFF;
  if(ht == 0xFFF) asymHt   = 0xFF;
  if(htHF == 0xFFF) asymHtHF = 0xFF;
  if((etHF == 0xFFF) || (et == 0xFFF)) cent = 0x80;
  
  //if (mhtx>0xFFF) mhtx = 0xFFF;
  //if (mhty>0xFFF) mhty = 0xFFF;


  //mhtPhi = (111 << 4);
  //mhtPhiHF = (111 << 4); // to match hw value if undefined
  
  // Final MET calculation
  if ( (metx != 0 || mety != 0) && !metSat ) cordic_( metx , mety , metPhi , met );
  // sets the met scale back to the original range for output into GT, this corresponds to
  // the previous scaling of sin/cos factors in calculation of metx and mety by 2^10 = 1024
  met >>= 10; 

  // Final METHF calculation
  if ( (metxHF != 0 || metyHF != 0) && !metHFSat ) cordic_( metxHF , metyHF , metPhiHF , metHF );
  metHF >>= 10;


  // Final MHT calculation
  if ( (mhtx != 0 || mhty != 0) && !mhtSat ) cordic_( mhtx , mhty , mhtPhi , mht );
  // sets the mht scale back to the original range for output into GT, the other 4
  // bits are brought back just before the accumulation of ring sum in MP jet sum algorithm
  mht >>= 6; 

  if ( (mhtxHF != 0 || mhtyHF != 0) && !mhtHFSat ) cordic_( mhtxHF , mhtyHF , mhtPhiHF , mhtHF );
  mhtHF >>= 6; 


  if(metSat) met=0xFFF;
  if(metHFSat) metHF=0xFFF;
  if(mhtSat) mht=0xFFF;
  if(mhtHFSat) mhtHF=0xFFF;

  // Make final collection
  math::XYZTLorentzVector p4;

  l1t::EtSum etSumTotalEt(p4,l1t::EtSum::EtSumType::kTotalEt,et,0,0,0);
  l1t::EtSum etSumTotalEtHF(p4,l1t::EtSum::EtSumType::kTotalEtHF,etHF,0,0,0);
  l1t::EtSum etSumTotalEtEm(p4,l1t::EtSum::EtSumType::kTotalEtEm,etem,0,0,0);
  l1t::EtSum etSumMissingEt(p4,l1t::EtSum::EtSumType::kMissingEt,met,0,metPhi>>4,0);
  l1t::EtSum etSumMissingEtHF(p4,l1t::EtSum::EtSumType::kMissingEtHF,metHF,0,metPhiHF>>4,0);
  l1t::EtSum htSumht(p4,l1t::EtSum::EtSumType::kTotalHt,ht,0,0,0);
  l1t::EtSum htSumhtHF(p4,l1t::EtSum::EtSumType::kTotalHtHF,htHF,0,0,0);
  l1t::EtSum htSumMissingHt(p4,l1t::EtSum::EtSumType::kMissingHt,mht,0,mhtPhi>>4,0);
  l1t::EtSum htSumMissingHtHF(p4,l1t::EtSum::EtSumType::kMissingHtHF,mhtHF,0,mhtPhiHF>>4,0);
  l1t::EtSum etSumMinBiasHFP0(p4,l1t::EtSum::EtSumType::kMinBiasHFP0,mbp0,0,0,0);
  l1t::EtSum etSumMinBiasHFM0(p4,l1t::EtSum::EtSumType::kMinBiasHFM0,mbm0,0,0,0);
  l1t::EtSum etSumMinBiasHFP1(p4,l1t::EtSum::EtSumType::kMinBiasHFP1,mbp1,0,0,0);
  l1t::EtSum etSumMinBiasHFM1(p4,l1t::EtSum::EtSumType::kMinBiasHFM1,mbm1,0,0,0);
  l1t::EtSum etSumTowCount(p4,l1t::EtSum::EtSumType::kTowerCount,ntow,0,0,0);
  l1t::EtSum etAsym(p4,l1t::EtSum::EtSumType::kAsymEt,asymEt,0,0,0);
  l1t::EtSum etHFAsym(p4,l1t::EtSum::EtSumType::kAsymEtHF,asymEtHF,0,0,0);
  l1t::EtSum htAsym(p4,l1t::EtSum::EtSumType::kAsymHt,asymHt,0,0,0);
  l1t::EtSum htHFAsym(p4,l1t::EtSum::EtSumType::kAsymHtHF,asymHtHF,0,0,0);
  l1t::EtSum centrality(p4,l1t::EtSum::EtSumType::kCentrality,cent,0,0,0);

  outputSums.push_back(etSumTotalEt);
  outputSums.push_back(etSumTotalEtHF);
  outputSums.push_back(etSumTotalEtEm);
  outputSums.push_back(etSumMinBiasHFP0);
  outputSums.push_back(htSumht);
  outputSums.push_back(htSumhtHF);
  outputSums.push_back(etSumMinBiasHFM0);
  outputSums.push_back(etSumMissingEt);
  outputSums.push_back(etSumMinBiasHFP1);
  outputSums.push_back(htSumMissingHt);
  outputSums.push_back(etSumMinBiasHFM1);
  outputSums.push_back(etSumMissingEtHF);
  outputSums.push_back(htSumMissingHtHF);
  outputSums.push_back(etSumTowCount);
  outputSums.push_back(etAsym);
  outputSums.push_back(etHFAsym);
  outputSums.push_back(htAsym);
  outputSums.push_back(htHFAsym);
  outputSums.push_back(centrality);

}
