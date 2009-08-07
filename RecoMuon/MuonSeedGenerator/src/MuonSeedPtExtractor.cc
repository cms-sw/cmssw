#include "RecoMuon/MuonSeedGenerator/src/MuonSeedPtExtractor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "TMath.h"
#include <sstream>

MuonSeedPtExtractor::MuonSeedPtExtractor(const edm::ParameterSet& par)
: scaleDT_( par.getParameter<bool>("scaleDT") )
{
  // load pT seed parameters
  // DT combinations
  fillParametersForCombo("DT_12", par);
  fillParametersForCombo("DT_13", par);
  fillParametersForCombo("DT_14", par);
  fillParametersForCombo("DT_23", par);
  fillParametersForCombo("DT_24", par);
  fillParametersForCombo("DT_34", par);
  // CSC combinations
  fillParametersForCombo("CSC_01", par);
  fillParametersForCombo("CSC_12", par);
  fillParametersForCombo("CSC_02", par);
  fillParametersForCombo("CSC_13", par);
  fillParametersForCombo("CSC_03", par);
  fillParametersForCombo("CSC_14", par);
  fillParametersForCombo("CSC_23", par);
  fillParametersForCombo("CSC_24", par);
  fillParametersForCombo("CSC_34", par);

  // Overlap combinations
  fillParametersForCombo("OL_1213", par);
  fillParametersForCombo("OL_1222", par);
  fillParametersForCombo("OL_1232", par);
  fillParametersForCombo("OL_2213", par);
  fillParametersForCombo("OL_2222", par);

  // Single segments (CSC)
  fillParametersForCombo("SME_11", par);
  fillParametersForCombo("SME_12", par);
  fillParametersForCombo("SME_13", par);
  fillParametersForCombo("SME_21", par);
  fillParametersForCombo("SME_22", par);
  fillParametersForCombo("SME_31", par);
  fillParametersForCombo("SME_32", par);
  fillParametersForCombo("SME_41", par);
  fillParametersForCombo("SME_42", par);

  // Single segments (DT)
  fillParametersForCombo("SMB_10", par);
  fillParametersForCombo("SMB_11", par);
  fillParametersForCombo("SMB_12", par);
  fillParametersForCombo("SMB_20", par);
  fillParametersForCombo("SMB_21", par);
  fillParametersForCombo("SMB_22", par);
  fillParametersForCombo("SMB_30", par);
  fillParametersForCombo("SMB_31", par);
  fillParametersForCombo("SMB_32", par);

  fillScalesForCombo("CSC_01_1_scale", par);
  fillScalesForCombo("CSC_12_1_scale", par);
  fillScalesForCombo("CSC_12_2_scale", par);
  fillScalesForCombo("CSC_12_3_scale", par);
  fillScalesForCombo("CSC_13_2_scale", par);
  fillScalesForCombo("CSC_13_3_scale", par);
  fillScalesForCombo("CSC_14_3_scale", par);
  fillScalesForCombo("CSC_23_1_scale", par);
  fillScalesForCombo("CSC_23_2_scale", par);
  fillScalesForCombo("CSC_24_1_scale", par);
  fillScalesForCombo("CSC_34_1_scale", par);
  fillScalesForCombo("DT_12_1_scale", par);
  fillScalesForCombo("DT_12_2_scale", par);
  fillScalesForCombo("DT_13_1_scale", par);
  fillScalesForCombo("DT_13_2_scale", par);
  fillScalesForCombo("DT_14_1_scale", par);
  fillScalesForCombo("DT_14_2_scale", par);
  fillScalesForCombo("DT_23_1_scale", par);
  fillScalesForCombo("DT_23_2_scale", par);
  fillScalesForCombo("DT_24_1_scale", par);
  fillScalesForCombo("DT_24_2_scale", par);
  fillScalesForCombo("DT_34_1_scale", par);
  fillScalesForCombo("DT_34_2_scale", par);
  fillScalesForCombo("OL_1213_0_scale", par);
  fillScalesForCombo("OL_1222_0_scale", par);
  fillScalesForCombo("OL_1232_0_scale", par);
  fillScalesForCombo("OL_2213_0_scale", par);
  fillScalesForCombo("OL_2222_0_scale", par);
  fillScalesForCombo("SMB_10_0_scale", par);
  fillScalesForCombo("SMB_11_0_scale", par);
  fillScalesForCombo("SMB_12_0_scale", par);
  fillScalesForCombo("SMB_20_0_scale", par);
  fillScalesForCombo("SMB_21_0_scale", par);
  fillScalesForCombo("SMB_22_0_scale", par);
  fillScalesForCombo("SMB_30_0_scale", par);
  fillScalesForCombo("SMB_31_0_scale", par);
  fillScalesForCombo("SMB_32_0_scale", par);
  fillScalesForCombo("SME_11_0_scale", par);
  fillScalesForCombo("SME_12_0_scale", par);
  fillScalesForCombo("SME_13_0_scale", par);
  fillScalesForCombo("SME_21_0_scale", par);
  fillScalesForCombo("SME_22_0_scale", par);

}


MuonSeedPtExtractor::~MuonSeedPtExtractor(){
}


void MuonSeedPtExtractor::fillParametersForCombo(const std::string & name, const edm::ParameterSet & pset)
{
  theParametersForCombo[name] = pset.getParameter<std::vector<double> >(name);
}


void MuonSeedPtExtractor::fillScalesForCombo(const std::string & name, const edm::ParameterSet & pset)
{
  theScalesForCombo[name] = pset.getParameter<std::vector<double> >(name);
}


std::vector<double> MuonSeedPtExtractor::pT_extract(MuonTransientTrackingRecHit::ConstMuonRecHitPointer firstHit,
                                                    MuonTransientTrackingRecHit::ConstMuonRecHitPointer secondHit) const
{
  GlobalPoint innerPoint = firstHit->globalPosition();
  GlobalPoint outerPoint = secondHit->globalPosition();
  MuonTransientTrackingRecHit::ConstMuonRecHitPointer innerHit = firstHit;
  MuonTransientTrackingRecHit::ConstMuonRecHitPointer outerHit = secondHit;

  // ways in which the hits could be in the wrong order
  if( (outerHit->isDT() && innerHit->isCSC())
   || (outerHit->isDT() && innerHit->isDT() && (innerPoint.perp() > outerPoint.perp())) 
   || (outerHit->isCSC() && innerHit->isCSC() && (fabs(innerPoint.z()) > fabs(outerPoint.z()))) )
  {
    innerHit = secondHit;
    outerHit = firstHit;
    innerPoint = innerHit->globalPosition();
    outerPoint = outerHit->globalPosition();
  } 
  
  double phiInner = innerPoint.phi();
  double phiOuter = outerPoint.phi();

  double etaInner = innerPoint.eta();
  double etaOuter = outerPoint.eta();
  //std::cout<<" inner pos = "<< innerPoint << " phi eta " << phiInner << " " << etaInner << std::endl;
  //std::cout<<" outer pos = "<< outerPoint << " phi eta " << phiOuter << " " << etaOuter << std::endl;
  //double thetaInner = firstHit->globalPosition().theta();
  // if some of the segments is missing r-phi measurement then we should
  // use only the 4D phi estimate (also use 4D eta estimate only)
  // the direction is not so important (it will be corrected) 
  /*
    bool firstOK = (4==allValidSets[iSet][firstMeasurement]->hit->dimension());
    bool lastOK = (4==allValidSets[iSet][lastMeasurement]->hit->dimension());
    if(!(firstOK * lastOK)){
    if(!firstOK){
    }
    if(!firstOK){
    }
    }
  */
  double dPhi =  phiInner - phiOuter;
  if(dPhi < -TMath::Pi()){
    dPhi += 2*TMath::Pi();
  }
  else if(dPhi >  TMath::Pi()){
    dPhi -= 2*TMath::Pi();
  }
  int sign = 1;
  if ( dPhi< 0.) {
    dPhi = -dPhi;
    sign = -1;
  }

  if (dPhi < 1.0e-6){
    dPhi = 1.0e-6;
  }
  double eta = fabs(etaOuter);// what if it is 2D segment? use inner?


  std::vector <int> stationCoded(2);
  stationCoded[0] = stationCoded[1] = 999;

  DetId  detId_inner = innerHit->hit()->geographicalId();
  DetId  detId_outer = outerHit->hit()->geographicalId();

  stationCoded[0] = stationCode(innerHit);
  stationCoded[1] = stationCode(outerHit);

  std::ostringstream os0 ;
  std::ostringstream os1 ;
  os0 << abs(stationCoded[0]);
  os1 << abs(stationCoded[1]);

  //std::cout<<" st1 = "<<stationCoded[0]<<" st2 = "<<stationCoded[1]<<std::endl;
  //std::cout<<" detId_inner = "<<detId_inner.rawId()<<"  detId_outer = "<< detId_outer.rawId()<<std::endl;
  std::string  combination = "0";
  std::string init_combination = combination;
  bool singleSegment = false;
  //if(detId_first == detId_second){
  if( stationCoded[0] == stationCoded[1]){
    // single segment - DT or CSC
    singleSegment = true;
    //eta = innerPoint.eta();
    GlobalVector gv = innerHit->globalDirection();

    // Psi is angle between the segment origin and segment direction
    // Use dot product between two vectors to get Psi in global x-y plane
    double cosDpsi  = (gv.x()*innerPoint.x() + gv.y()*innerPoint.y());
    cosDpsi /= sqrt(innerPoint.x()*innerPoint.x() + innerPoint.y()*innerPoint.y());
    cosDpsi /= sqrt(gv.x()*gv.x() + gv.y()*gv.y());

    double axb = ( innerPoint.x()*gv.y() ) - ( innerPoint.y()*gv.x() ) ;
    sign = (axb < 0.) ? 1 : -1;

    double dpsi = fabs(acos(cosDpsi)) ;
    if ( dpsi > TMath::Pi()/2.) {
      dpsi = TMath::Pi() - dpsi;
    }
    if (fabs(dpsi) < 0.00005) {
      dpsi = 0.00005;
    }
    dPhi = dpsi;

    if(innerHit->isDT())
    {
      DTChamberId dtCh(detId_inner);
      std::ostringstream os;
      os <<  dtCh.station() << abs(dtCh.wheel());
      combination = "SMB_"+os.str();
    }
    else if(innerHit->isCSC())
    {
      CSCDetId cscId(detId_inner);
      std::ostringstream os;
      int ring = cscId.ring();
      if(ring == 4) ring = 1;
      os << cscId.station() << ring;
      combination = "SME_"+os.str();
    }
    else
    {
      throw cms::Exception("MuonSeedPtExtractor") << "Bad hit DetId";
    }
  }
  else{
    if(stationCoded[0]<0){
      if(stationCoded[1]<0){
        // DT-DT
        combination = "DT_" + os0.str() + os1.str();
      }
      else{
        // DT-CSC
        eta = fabs(etaInner);
        if(-1==stationCoded[0]){
          switch (stationCoded[1]){
          case 1:
            combination = "OL_1213";
            break;
          case 2:
            combination = "OL_1222";
            break;
          case 3:
            combination = "OL_1232";
            break;
          default:
            // can not be 
            break;
          }
        }
        else if (-2==stationCoded[0]){
          if(1==stationCoded[1]){
            combination = "OL_2213";
          }
          else{
            // can not be (not coded?)
	    combination = "OL_2222";// in case
          }
        }
        else{
          // can not be
        }
      }
    }
    else{
      if(stationCoded[1]<0){
        // CSC-DT
        // can not be
      }
      else{
        // CSC-CSC
        combination = "CSC_" + os0.str() + os1.str();
        if("CSC_04" == combination){
          combination = "CSC_14";
        }
      }
    }
  }

  std::vector<double> pTestimate(2);// 
  //std::cout<<" combination = "<<combination<<std::endl;
  if(init_combination!=combination){
        //std::cout<<" combination = "<<combination<<" eta = "<<eta<<" dPhi = "<<dPhi<<std::endl;
    ParametersMap::const_iterator parametersItr = theParametersForCombo.find(combination);
    if(parametersItr == theParametersForCombo.end()) {
       edm::LogWarning("RecoMuon|MuonSeedGenerator|MuonSeedPtExtractor") << "Cannot find parameters for combo " << combination;
       pTestimate[0] = pTestimate[1] = 100;
    }
    else {
      if(scaleDT_ && outerHit->isDT() )
      {
        dPhi = scaledPhi(dPhi, combination, detId_outer);
      }
      pTestimate = getPt(parametersItr->second, eta, dPhi);
      if(singleSegment){
        pTestimate[0] = fabs(pTestimate[0]);
        pTestimate[1] = fabs(pTestimate[1]);
      }
      pTestimate[0] *= double(sign);
    }
  }
  else{
    // often a MB3 - ME1/3 seed
    pTestimate[0] = pTestimate[1] = 100;
    // hmm
  }
  // std::cout<<" pTestimate[0] = "<<pTestimate[0]<<" pTestimate[1] = "<<pTestimate[1]<<std::endl;
  /*
                              MuonRecHitContainer_clusters[cluster][iHit+1]->isDT());
          if(specialCase){
            DTChamberId dtCh(detId);
            DTChamberId dtCh_2(detId_2);
            specialCase =  (dtCh.station() == dtCh_2.station());
          }
  */
  //return vPara;
  return pTestimate;
}


int MuonSeedPtExtractor::stationCode(MuonTransientTrackingRecHit::ConstMuonRecHitPointer hit) const
{
  DetId detId(hit->hit()->geographicalId());
  int result = -999;
  if(hit->isDT() ){
    DTChamberId dtCh(detId);
    //std::cout<<"first (DT) St/W/S = "<<dtCh.station()<<"/"<<dtCh.wheel()<<"/"<<dtCh.sector()<<"/"<<std::endl;
    result = -1 * dtCh.station();
  }
  else if( hit->isCSC() ){
    CSCDetId cscID(detId);
    //std::cout<<"first (CSC) E/S/R/C = "<<cscID.endcap()<<"/"<<cscID.station()<<"/"<<cscID.ring()<<"/"<<cscID.chamber()<<std::endl;
    result = cscID.station();
    if(result == 1 && (1 == cscID.ring() ||  4 == cscID.ring()) )
       result = 0;
  }
  else if(hit->isRPC()){
  }
  return result;
}


std::vector<double> MuonSeedPtExtractor::getPt(const std::vector<double> & vPara, double eta, double dPhi ) const {
   //std::cout<<" eta = "<<eta<<" dPhi = "<<dPhi<<" vPara[0] = "<<vPara[0]<<" vPara[1] = "<<vPara[1]<<" vPara[2] = "<<vPara[2]<<std::endl;
  double h  = fabs(eta);
  double estPt  = ( vPara[0] + vPara[1]*h + vPara[2]*h*h ) / dPhi;
  double estSPt = ( vPara[3] + vPara[4]*h + vPara[5]*h*h ) * estPt;
  // std::cout<<"estPt = "<<estPt<<std::endl;
  std::vector<double> paraPt ;
  paraPt.push_back( estPt );
  paraPt.push_back( estSPt ) ;
  return paraPt ;
}


double MuonSeedPtExtractor::scaledPhi( double dphi, const std::string & combination, const DTChamberId & outerDetId) const
{
  int wheel = 0;
  if(combination[0] == 'D') {
    wheel = abs(outerDetId.wheel());
  }

  std::ostringstream os;
  os << combination << "_" << wheel << "_scale";

  ScalesMap::const_iterator scalesItr = theScalesForCombo.find(os.str());
  if (dphi != 0. && scalesItr != theScalesForCombo.end()) {
    double t1 = scalesItr->second[3];
    double oPhi = 1./dphi ;
    double scaleFactor = 1./( 1. + t1/( oPhi + 10. ) ) ;
    dphi *= scaleFactor ;
  }
  return dphi ;
}

