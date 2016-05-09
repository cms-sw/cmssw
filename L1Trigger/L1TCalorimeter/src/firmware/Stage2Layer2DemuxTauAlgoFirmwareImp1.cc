///
/// \class l1t::Stage2Layer2TauAlgorithmFirmwareImp1
///
/// \author:
///
/// Description:

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxTauAlgoFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include <vector>
#include <algorithm>
#include "L1Trigger/L1TCalorimeter/interface/BitonicSort.h"

namespace l1t {
  inline bool operator > ( l1t::Tau& a, l1t::Tau& b )
  {
    if ( a.pt() == b.pt() ){
      if( a.hwPhi() == b.hwPhi() ){
	return a.hwEta() > b.hwEta();
      }
      else{
	  return a.hwPhi() > b.hwPhi();	  
	}

    }
    else{
	return a.pt() > b.pt();
      }
  }
}

l1t::Stage2Layer2DemuxTauAlgoFirmwareImp1::Stage2Layer2DemuxTauAlgoFirmwareImp1(CaloParamsHelper* params) :
  params_(params)
{


}


l1t::Stage2Layer2DemuxTauAlgoFirmwareImp1::~Stage2Layer2DemuxTauAlgoFirmwareImp1() {


}


void l1t::Stage2Layer2DemuxTauAlgoFirmwareImp1::processEvent(const std::vector<l1t::Tau> & inputTaus,
    std::vector<l1t::Tau> & outputTaus) {


  vector<pair<int,double> > etaGT;
  for(int i=0;i<115;i++)
    etaGT.push_back( make_pair(i,i*(0.087/2.)) );

  vector<pair<int,double> > phiGT;
  for(int i=0;i<145;i++)
    phiGT.push_back( make_pair(i,i*(M_PI/72.)) );
  phiGT[144] = make_pair(0,2*M_PI); //2pi = 0

  outputTaus = inputTaus;

  for(auto& tau : outputTaus){

    double eta = tau.eta();
    double phi = tau.phi();
    if(phi<0)
      phi+=2*M_PI;
 
    double minDistance = 99999.;
    pair<int, double> closestPoint = make_pair(0,0.);
    
    for(const auto& p : etaGT){
      double distance = abs(abs(eta) - p.second);
      if(distance < minDistance){
	closestPoint = p;
	minDistance = distance;
      }
    }

    int hwEta_GT = (eta>0) ? closestPoint.first : - closestPoint.first;
    double eta_GT = (eta>0) ? closestPoint.second : - closestPoint.second;

    minDistance = 99999.;
    closestPoint = make_pair(0,0.);
    
    for(const auto& p : phiGT){
      double distance = abs(phi - p.second);
      if(distance < minDistance){
	closestPoint = p;
	minDistance = distance;
      }
    }

    int hwPhi_GT = closestPoint.first;
    double phi_GT = closestPoint.second;

    tau.setHwEta(hwEta_GT);
    tau.setHwPhi(hwPhi_GT);

    //9 bits threshold
    if(tau.hwPt()>511)
      tau.setHwPt(511);

    math::PtEtaPhiMLorentzVector tauP4(tau.hwPt()*params_->egLsb(), eta_GT, phi_GT, 0.);
    tau.setP4(tauP4);

  }

  //sorting with descending pT
  std::vector<l1t::Tau>::iterator start_ = outputTaus.begin();  
  std::vector<l1t::Tau>::iterator end_   = outputTaus.end();
  BitonicSort<l1t::Tau>(down, start_, end_);

}
