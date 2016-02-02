///
/// \class l1t::Stage2Layer2EGAlgorithmFirmwareImp1
///
/// \author:
///
/// Description:

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxEGAlgoFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include <vector>
#include <algorithm>

l1t::Stage2Layer2DemuxEGAlgoFirmwareImp1::Stage2Layer2DemuxEGAlgoFirmwareImp1(CaloParamsHelper* params) :
  params_(params)
{


}


l1t::Stage2Layer2DemuxEGAlgoFirmwareImp1::~Stage2Layer2DemuxEGAlgoFirmwareImp1() {


}


void l1t::Stage2Layer2DemuxEGAlgoFirmwareImp1::processEvent(const std::vector<l1t::EGamma> & inputEGammas,
    std::vector<l1t::EGamma> & outputEGammas) {

  vector<pair<int,double> > etaGT;
  for(int i=0;i<115;i++)
    etaGT.push_back( make_pair(i,i*(0.087/2.)) );

  vector<pair<int,double> > phiGT;
  for(int i=0;i<145;i++)
    phiGT.push_back( make_pair(i,i*(M_PI/72.)) );
  phiGT[144] = make_pair(0,2*M_PI); //2pi = 0

  outputEGammas = inputEGammas;

  for(auto& egamma : outputEGammas){

    double eta = egamma.eta();
    double phi = egamma.phi();
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

    egamma.setHwEta(hwEta_GT);
    egamma.setHwPhi(hwPhi_GT);

    //9 bits threshold
    if(egamma.hwPt()>511)
      egamma.setHwPt(511);

  }


}
