/**
 *  See header file for a description of this class.
 *
 *  \author: Shih-Chuan Kao, Dominique Fortin - UCR
 *
 */


#include "RecoMuon/MuonSeedGenerator/src/MuonSeedCreator.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "gsl/gsl_statistics.h"


/*
 * Constructor
 */
MuonSeedCreator::MuonSeedCreator(const edm::ParameterSet& pset){
  
  theMinMomentum  = pset.getParameter<double>("minimumSeedPt");  
  theMaxMomentum  = pset.getParameter<double>("maximumSeedPt");  
  defaultMomentum = pset.getParameter<double>("defaultSeedPt");
  debug           = pset.getParameter<bool>("DebugMuonSeed");
  sysError        = pset.getParameter<double>("SeedPtSystematics");
  // load seed PT parameters 
  DT12 = pset.getParameter<std::vector<double> >("DT_12");
  DT13 = pset.getParameter<std::vector<double> >("DT_13");
  DT14 = pset.getParameter<std::vector<double> >("DT_14");
  DT23 = pset.getParameter<std::vector<double> >("DT_23");
  DT24 = pset.getParameter<std::vector<double> >("DT_24");
  DT34 = pset.getParameter<std::vector<double> >("DT_34");

  CSC01 = pset.getParameter<std::vector<double> >("CSC_01");
  CSC12 = pset.getParameter<std::vector<double> >("CSC_12");
  CSC02 = pset.getParameter<std::vector<double> >("CSC_02");
  CSC13 = pset.getParameter<std::vector<double> >("CSC_13");
  CSC03 = pset.getParameter<std::vector<double> >("CSC_03");
  CSC14 = pset.getParameter<std::vector<double> >("CSC_14");
  CSC23 = pset.getParameter<std::vector<double> >("CSC_23");
  CSC24 = pset.getParameter<std::vector<double> >("CSC_24");
  CSC34 = pset.getParameter<std::vector<double> >("CSC_34");

  OL1213 = pset.getParameter<std::vector<double> >("OL_1213");
  OL1222 = pset.getParameter<std::vector<double> >("OL_1222");
  OL1232 = pset.getParameter<std::vector<double> >("OL_1232");
  OL1213 = pset.getParameter<std::vector<double> >("OL_1213");
  OL2222 = pset.getParameter<std::vector<double> >("OL_1222");

  SME11 =  pset.getParameter<std::vector<double> >("SME_11");
  SME12 =  pset.getParameter<std::vector<double> >("SME_12");
  SME13 =  pset.getParameter<std::vector<double> >("SME_13");
  SME21 =  pset.getParameter<std::vector<double> >("SME_21");
  SME22 =  pset.getParameter<std::vector<double> >("SME_22");
  SME31 =  pset.getParameter<std::vector<double> >("SME_31");
  SME32 =  pset.getParameter<std::vector<double> >("SME_32");
  SME41 =  pset.getParameter<std::vector<double> >("SME_41");

  SMB10 =  pset.getParameter<std::vector<double> >("SMB_10");
  SMB11 =  pset.getParameter<std::vector<double> >("SMB_11");
  SMB12 =  pset.getParameter<std::vector<double> >("SMB_12");
  SMB20 =  pset.getParameter<std::vector<double> >("SMB_20");
  SMB21 =  pset.getParameter<std::vector<double> >("SMB_21");
  SMB22 =  pset.getParameter<std::vector<double> >("SMB_22");
  SMB30 =  pset.getParameter<std::vector<double> >("SMB_30");
  SMB31 =  pset.getParameter<std::vector<double> >("SMB_31");
  SMB32 =  pset.getParameter<std::vector<double> >("SMB_32");

  // Load dphi scale parameters
  CSC01_1 = pset.getParameter<std::vector<double> >("CSC_01_1_scale");
  CSC12_1 = pset.getParameter<std::vector<double> >("CSC_12_1_scale");
  CSC12_2 = pset.getParameter<std::vector<double> >("CSC_12_2_scale");
  CSC12_3 = pset.getParameter<std::vector<double> >("CSC_12_3_scale");
  CSC13_2 = pset.getParameter<std::vector<double> >("CSC_13_2_scale");
  CSC13_3 = pset.getParameter<std::vector<double> >("CSC_13_3_scale");
  CSC14_3 = pset.getParameter<std::vector<double> >("CSC_14_3_scale");
  CSC23_1 = pset.getParameter<std::vector<double> >("CSC_23_1_scale");
  CSC23_2 = pset.getParameter<std::vector<double> >("CSC_23_2_scale");
  CSC24_1 = pset.getParameter<std::vector<double> >("CSC_24_1_scale");
  CSC34_1 = pset.getParameter<std::vector<double> >("CSC_34_1_scale");

  DT12_1 = pset.getParameter<std::vector<double> >("DT_12_1_scale");
  DT12_2 = pset.getParameter<std::vector<double> >("DT_12_2_scale");
  DT13_1 = pset.getParameter<std::vector<double> >("DT_13_1_scale");
  DT13_2 = pset.getParameter<std::vector<double> >("DT_13_2_scale");
  DT14_1 = pset.getParameter<std::vector<double> >("DT_14_1_scale");
  DT14_2 = pset.getParameter<std::vector<double> >("DT_14_2_scale");
  DT23_1 = pset.getParameter<std::vector<double> >("DT_23_1_scale");
  DT23_2 = pset.getParameter<std::vector<double> >("DT_23_2_scale");
  DT24_1 = pset.getParameter<std::vector<double> >("DT_24_1_scale");
  DT24_2 = pset.getParameter<std::vector<double> >("DT_24_2_scale");
  DT34_1 = pset.getParameter<std::vector<double> >("DT_34_1_scale");
  DT34_2 = pset.getParameter<std::vector<double> >("DT_34_2_scale");

  OL_1213 = pset.getParameter<std::vector<double> >("OL_1213_0_scale");
  OL_1222 = pset.getParameter<std::vector<double> >("OL_1222_0_scale");
  OL_1232 = pset.getParameter<std::vector<double> >("OL_1232_0_scale");
  OL_2213 = pset.getParameter<std::vector<double> >("OL_2213_0_scale");
  OL_2222 = pset.getParameter<std::vector<double> >("OL_2222_0_scale");

  SMB_10S = pset.getParameter<std::vector<double> >("SMB_10_0_scale");
  SMB_11S = pset.getParameter<std::vector<double> >("SMB_11_0_scale");
  SMB_12S = pset.getParameter<std::vector<double> >("SMB_12_0_scale");
  SMB_20S = pset.getParameter<std::vector<double> >("SMB_20_0_scale");
  SMB_21S = pset.getParameter<std::vector<double> >("SMB_21_0_scale");
  SMB_22S = pset.getParameter<std::vector<double> >("SMB_22_0_scale");
  SMB_30S = pset.getParameter<std::vector<double> >("SMB_30_0_scale");
  SMB_31S = pset.getParameter<std::vector<double> >("SMB_31_0_scale");
  SMB_32S = pset.getParameter<std::vector<double> >("SMB_32_0_scale");

  SME_11S = pset.getParameter<std::vector<double> >("SME_11_0_scale");
  SME_12S = pset.getParameter<std::vector<double> >("SME_12_0_scale");
  SME_13S = pset.getParameter<std::vector<double> >("SME_13_0_scale");
  SME_21S = pset.getParameter<std::vector<double> >("SME_21_0_scale");
  SME_22S = pset.getParameter<std::vector<double> >("SME_22_0_scale");

}


/*
 * Destructor
 */
MuonSeedCreator::~MuonSeedCreator(){
  
}


/*
 * createSeed
 *
 * Note type = 1 --> CSC
 *           = 2 --> Overlap
 *           = 3 --> DT
 */
TrajectorySeed MuonSeedCreator::createSeed(int type, const SegmentContainer& seg, const std::vector<int>& layers, int NShowers, int NShowerSegments ) {

  // The index of the station closest to the IP
  int last = 0;

  double ptmean = theMinMomentum;
  double sptmean = theMinMomentum;

  AlgebraicVector t(4);
  AlgebraicSymMatrix mat(5,0) ;
  LocalPoint segPos;

  // Compute the pt according to station types used;
  if (type == 1 ) estimatePtCSC(seg, layers, ptmean, sptmean);
  if (type == 2 ) estimatePtOverlap(seg, layers, ptmean, sptmean);
  if (type == 3 ) estimatePtDT(seg, layers, ptmean, sptmean);
  if (type == 4 ) estimatePtSingle(seg, layers, ptmean, sptmean);
  // type 5 are the seeding for ME1/4
  if (type == 5 ) estimatePtCSC(seg, layers, ptmean, sptmean);

  // for certain clear showering case, set-up the minimum value 
  if ( NShowers > 0 ) estimatePtShowering( NShowers, NShowerSegments, ptmean, sptmean ); 
  //if ( NShowers > 0 ) std::cout<<" Showering happened "<<NShowers<<" times w/ "<< NShowerSegments<<std::endl; ; 


  // Minimal pt
  double charge = 1.0;
  if (ptmean < 0.) charge = -1.0; 
  if ( (charge * ptmean) < theMinMomentum ) {
    ptmean  = theMinMomentum * charge;
    sptmean = theMinMomentum ;
  }
  else if ( (charge * ptmean) > theMaxMomentum ) {
    ptmean  = theMaxMomentum * charge;
    sptmean = theMaxMomentum * 0.25 ;
  }

  LocalTrajectoryParameters param;

  double p_err =0.0;
  // determine the seed layer
  int    best_seg= 0;
  double chi2_dof = 9999.0;
  unsigned int ini_seg = 0;
  // avoid generating seed from  1st layer(ME1/1)
  if ( type == 5 )  ini_seg = 1;
  for (size_t i = ini_seg ; i < seg.size(); i++) {
      double dof = static_cast<double>(seg[i]->degreesOfFreedom());
      if ( chi2_dof  < ( seg[i]->chi2()/dof ) ) continue;
      chi2_dof = seg[i]->chi2() / dof ;
      best_seg = static_cast<int>(i);
  }  
  

  if ( type==1 || type==5 || type== 4) {
     // Fill the LocalTrajectoryParameters
     /// get the Global position
     last = best_seg;
     // last = 0;
     GlobalVector mom = seg[last]->globalPosition()-GlobalPoint();
     segPos = seg[last]->localPosition();
     /// get the Global direction
  
     GlobalVector polar(GlobalVector::Spherical(mom.theta(),seg[last]->globalDirection().phi(),1.));

     /// scale the magnitude of total momentum
     polar *= fabs(ptmean)/polar.perp();
     /// Trasfer into local direction
     LocalVector segDirFromPos = seg[last]->det()->toLocal(polar);
     int chargeI = static_cast<int>(charge);
     LocalTrajectoryParameters param1(segPos, segDirFromPos, chargeI);
     param = param1;
     p_err =  (sptmean*sptmean)/(polar.mag()*polar.mag()*ptmean*ptmean) ;
     mat = seg[last]->parametersError().similarityT( seg[last]->projectionMatrix() );  
     mat[0][0]= p_err;
     if ( type == 5 ) { 
        mat[0][0] = mat[0][0]/fabs( tan(mom.theta()) ); 
        mat[1][1] = mat[1][1]/fabs( tan(mom.theta()) ); 
        mat[3][3] = 2.25*mat[3][3];
        mat[4][4] = 2.25*mat[4][4];
     }
     if ( type == 4 ) { 
        mat[0][0] = mat[0][0]/fabs( tan(mom.theta()) ); 
        mat[1][1] = mat[1][1]/fabs( tan(mom.theta()) ); 
        mat[2][2] = 2.25*mat[2][2];
        mat[3][3] = 2.25*mat[3][3];
        mat[4][4] = 2.25*mat[4][4];
     }
     double dh = fabs( seg[last]->globalPosition().eta() ) - 1.6 ; 
     if ( fabs(dh) < 0.1 && type == 1 ) {
        mat[1][1] = 4.*mat[1][1];
        mat[2][2] = 4.*mat[2][2];
        mat[3][3] = 9.*mat[3][3];
        mat[4][4] = 9.*mat[4][4];
     }

     //if ( !highPt && type != 1 ) mat[1][1]= 2.25*mat[1][1];
     //if (  highPt && type != 1 ) mat[3][3]= 2.25*mat[1][1];
     //mat[2][2]= 3.*mat[2][2];
     //mat[3][3]= 2.*mat[3][3];
     //mat[4][4]= 2.*mat[4][4];
  }
  else {
     // Fill the LocalTrajectoryParameters
     /// get the Global position
     last = 0;
     segPos = seg[last]->localPosition();
     GlobalVector mom = seg[last]->globalPosition()-GlobalPoint();
     /// get the Global direction
     GlobalVector polar(GlobalVector::Spherical(mom.theta(),seg[last]->globalDirection().phi(),1.));
     //GlobalVector polar(GlobalVector::Spherical(seg[last]->globalDirection().theta(),seg[last]->globalDirection().phi(),1.));

     /// count the energy loss - from parameterization
     //double ptRatio = 1.0 - (2.808/(fabs(ptmean) -1)) + (4.546/( (fabs(ptmean)-1)*(fabs(ptmean)-1)) );
     //ptmean = ptmean*ptRatio ;
      
     /// scale the magnitude of total momentum
     polar *= fabs(ptmean)/polar.perp();
     /// Trasfer into local direction
     LocalVector segDirFromPos = seg[last]->det()->toLocal(polar);
     int chargeI = static_cast<int>(charge);
     LocalTrajectoryParameters param1(segPos, segDirFromPos, chargeI);
     param = param1;
     p_err =  (sptmean*sptmean)/(polar.mag()*polar.mag()*ptmean*ptmean) ;
     mat = seg[last]->parametersError().similarityT( seg[last]->projectionMatrix() );  
     //mat[0][0]= 1.44 * p_err;
     mat[0][0]= p_err;
  }

  if ( debug ) {
    GlobalPoint gp = seg[last]->globalPosition();
    float Geta = gp.eta();

    std::cout << "Type "      << type   << " Nsegments " << layers.size() << " ";
    std::cout << "pt "        << ptmean << " errpt    " << sptmean 
              << " eta "      << Geta   << " charge "   << charge 
              << std::endl;
  }

  // this perform H.T() * parErr * H, which is the projection of the 
  // the measurement error (rechit rf) to the state error (TSOS rf)
  // Legend:
  // H => is the 4x5 projection matrix
  // parError the 4x4 parameter error matrix of the RecHit

  
  LocalTrajectoryError error(asSMatrix<5>(mat));
  
  // Create the TrajectoryStateOnSurface
  TrajectoryStateOnSurface tsos(param, error, seg[last]->det()->surface(),&*BField);
  
  // Take the DetLayer on which relies the segment
  DetId id = seg[last]->geographicalId();

  // Transform it in a TrajectoryStateOnSurface
  
  
  PTrajectoryStateOnDet seedTSOS = trajectoryStateTransform::persistentState( tsos, id.rawId());

  edm::OwnVector<TrackingRecHit> container;
  for (unsigned l=0; l<seg.size(); l++) {
      container.push_back( seg[l]->hit()->clone() ); 
      //container.push_back(seg[l]->hit()); 
  }

  TrajectorySeed theSeed(seedTSOS,container,alongMomentum);
  return theSeed;
}


/*
 * estimatePtCSC
 *
 * Look at delta phi to determine pt as:
 * pt = (c_1 * eta + c_2) / dphi
 *
 * Note that only segment pairs with one segment in the ME1 layers give sensitive results
 *
 */
void MuonSeedCreator::estimatePtCSC(const SegmentContainer& seg, const std::vector<int>& layers, double& thePt, double& theSpt ) {

  unsigned size = seg.size();
  if (size < 2) return;
  
  // reverse the segment and layer container first for pure CSC case
  //if ( layers[0] > layers[ layers.size()-1 ] ) {
  //   reverse( layers.begin(), layers.end() );
  //   reverse( seg.begin(), seg.end() );
  //}

  std::vector<double> ptEstimate;
  std::vector<double> sptEstimate;

  thePt  = defaultMomentum;
  theSpt = defaultMomentum;

  double pt = 0.;
  double spt = 0.;   
  GlobalPoint  segPos[2];

  int layer0 = layers[0];
  segPos[0] = seg[0]->globalPosition();
  float eta = fabs( segPos[0].eta() );
  //float corr = fabs( tan(segPos[0].theta()) );
  // use pt from vertex information
  /*
  if ( layer0 == 0 ) {
     SegmentContainer seg0;
     seg0.push_back(seg[0]);
     std::vector<int> lyr0(1,0);
     estimatePtSingle( seg0, lyr0, thePt, theSpt);
     ptEstimate.push_back( thePt );
     sptEstimate.push_back( theSpt );
  }
  */

  //std::cout<<" estimate CSC "<<std::endl;

  unsigned idx1 = size;
  if (size > 1) {
    while ( idx1 > 1 ) {
      idx1--;
      int layer1 = layers[idx1];
      if (layer0 == layer1) continue;
      segPos[1] = seg[idx1]->globalPosition();      

      double dphi = segPos[0].phi() - segPos[1].phi();
      //double temp_dphi = dphi/corr;
      double temp_dphi = dphi;
       
      double sign = 1.0;  
      if (temp_dphi < 0.) {
        temp_dphi = -1.0*temp_dphi;
        sign = -1.0;
      }

      // Ensure that delta phi is not too small to prevent pt from blowing up
      if (temp_dphi < 0.0001 ) { 
         temp_dphi = 0.0001 ;
         pt = theMaxMomentum ;
         spt = theMaxMomentum*0.25 ; 
         ptEstimate.push_back( pt*sign );
         sptEstimate.push_back( spt );
      }
      // ME1 is inner-most
      if ( layer0 == 0 && temp_dphi >= 0.0001 ) {
 
        // ME1/2 is outer-most
        if ( layer1 ==  1 ) {
          //temp_dphi = scaledPhi(temp_dphi, CSC01_1[3] );
          pt  = getPt( CSC01, eta , temp_dphi )[0];
          spt = getPt( CSC01, eta , temp_dphi )[1];
        }  
        // ME2 is outer-most
        else if ( layer1 == 2  ) {
          //temp_dphi = scaledPhi(temp_dphi, CSC12_3[3] );
          pt  = getPt( CSC02, eta , temp_dphi )[0];
          spt = getPt( CSC02, eta , temp_dphi )[1];
        }
        // ME3 is outer-most
        else if ( layer1 == 3 ) {
          //temp_dphi = scaledPhi(temp_dphi, CSC13_3[3] );
          pt  = getPt( CSC03, eta , temp_dphi )[0];
          spt = getPt( CSC03, eta , temp_dphi )[1];
        }
        // ME4 is outer-most
        else {
          //temp_dphi = scaledPhi(temp_dphi, CSC14_3[3]);
          pt  = getPt( CSC14, eta , temp_dphi )[0];
          spt = getPt( CSC14, eta , temp_dphi )[1];
        }
        ptEstimate.push_back( pt*sign );
        sptEstimate.push_back( spt );
      }

      // ME1/2,ME1/3 is inner-most
      if ( layer0 == 1 ) {
        // ME2 is outer-most
        if ( layer1 == 2 ) {

          //if ( eta <= 1.2 )  {   temp_dphi = scaledPhi(temp_dphi, CSC12_1[3]); }
          //if ( eta >  1.2 )  {   temp_dphi = scaledPhi(temp_dphi, CSC12_2[3]); }
          pt  = getPt( CSC12, eta , temp_dphi )[0];
          spt = getPt( CSC12, eta , temp_dphi )[1];
        }
        // ME3 is outer-most
        else if ( layer1 == 3 ) {
          temp_dphi = scaledPhi(temp_dphi, CSC13_2[3]);
          pt  = getPt( CSC13, eta , temp_dphi )[0];
          spt = getPt( CSC13, eta , temp_dphi )[1];
        }
        // ME4 is outer-most
        else {
          temp_dphi = scaledPhi(temp_dphi, CSC14_3[3]);
          pt  = getPt( CSC14, eta , temp_dphi )[0];
          spt = getPt( CSC14, eta , temp_dphi )[1];
        }
        ptEstimate.push_back( pt*sign );
        sptEstimate.push_back( spt );
      }

      // ME2 is inner-most
      if ( layer0 == 2 && temp_dphi > 0.0001 ) {

        // ME4 is outer-most
        bool ME4av =false;
        if ( layer1 == 4 )  {
          temp_dphi = scaledPhi(temp_dphi, CSC24_1[3]);
          pt  = getPt( CSC24, eta , temp_dphi )[0];
          spt = getPt( CSC24, eta , temp_dphi )[1];
          ME4av = true;
        }
        // ME3 is outer-most
        else {
          // if ME2-4 is availabe , discard ME2-3 
          if ( !ME4av ) {
            if ( eta <= 1.7 )  {   temp_dphi = scaledPhi(temp_dphi, CSC23_1[3]); }
            if ( eta >  1.7 )  {   temp_dphi = scaledPhi(temp_dphi, CSC23_2[3]); }
            pt  = getPt( CSC23, eta , temp_dphi )[0];
            spt = getPt( CSC23, eta , temp_dphi )[1];
          }
        }
        ptEstimate.push_back( pt*sign );   
        sptEstimate.push_back( spt );
      }

      // ME3 is inner-most
      if ( layer0 == 3 && temp_dphi > 0.0001 ) {

        temp_dphi = scaledPhi(temp_dphi, CSC34_1[3]);
        pt  = getPt( CSC34, eta , temp_dphi )[0];
        spt = getPt( CSC34, eta , temp_dphi )[1];
        ptEstimate.push_back( pt*sign );   
        sptEstimate.push_back( spt );
      }

    } 
  }

  // Compute weighted average if have more than one estimator
  if ( ptEstimate.size() > 0 ) weightedPt( ptEstimate, sptEstimate, thePt, theSpt);

}


/*
 * estimatePtDT
 *
 * Look at delta phi between segments to determine pt as:
 * pt = (c_1 * eta + c_2) / dphi
 */
void MuonSeedCreator::estimatePtDT(const SegmentContainer& seg, const std::vector<int>& layers, double& thePt, double& theSpt) {

  unsigned size = seg.size();
  if (size < 2) return;
  
  std::vector<double> ptEstimate;
  std::vector<double> sptEstimate;

  thePt  = defaultMomentum;
  theSpt = defaultMomentum;

  double pt = 0.;
  double spt = 0.;   
  GlobalPoint segPos[2];

  int layer0 = layers[0];
  segPos[0] = seg[0]->globalPosition();
  float eta = fabs(segPos[0].eta());

  //std::cout<<" estimate DT "<<std::endl;
  // inner-most layer
  //for ( unsigned idx0 = 0; idx0 < size-1; ++idx0 ) {
  //  layer0 = layers[idx0];
  //  segPos[0]  = seg[idx0]->globalPosition();
    // outer-most layer
    // for ( unsigned idx1 = idx0+1; idx1 < size; ++idx1 ) {
    for ( unsigned idx1 = 1; idx1 <size ;  ++idx1 ) {

      int layer1 = layers[idx1];
      segPos[1] = seg[idx1]->globalPosition();      
 
      //eta = fabs(segPos[1].eta());  
      //if (layer1 == -4) eta = fabs(segPos[0].eta());

      double dphi = segPos[0].phi() - segPos[1].phi();
      double temp_dphi = dphi;

      // Ensure that delta phi is not too small to prevent pt from blowing up
      
      double sign = 1.0;  
      if (temp_dphi < 0.) {
        temp_dphi = -temp_dphi;
        sign = -1.0;
      }
      
      if (temp_dphi < 0.0001 ) { 
         temp_dphi = 0.0001 ;
         pt = theMaxMomentum ;
         spt = theMaxMomentum*0.25 ; 
         ptEstimate.push_back( pt*sign );
         sptEstimate.push_back( spt );
      }

      // MB1 is inner-most
      bool MB23av = false;
      if (layer0 == -1 && temp_dphi > 0.0001 ) {
        // MB2 is outer-most
        if (layer1 == -2) {

          if ( eta <= 0.7 )  {   temp_dphi = scaledPhi(temp_dphi, DT12_1[3]); }
          if ( eta >  0.7 )  {   temp_dphi = scaledPhi(temp_dphi, DT12_2[3]); }
          pt  = getPt( DT12, eta , temp_dphi )[0];
          spt = getPt( DT12, eta , temp_dphi )[1];
          MB23av = true;
        }
        // MB3 is outer-most
        else if (layer1 == -3) {

          if ( eta <= 0.6 )  {   temp_dphi = scaledPhi(temp_dphi, DT13_1[3]); }
          if ( eta >  0.6 )  {   temp_dphi = scaledPhi(temp_dphi, DT13_2[3]); }
          pt  = getPt( DT13, eta , temp_dphi )[0];
          spt = getPt( DT13, eta , temp_dphi )[1];
          MB23av = true;
        }
        // MB4 is outer-most
        else {
          if ( !MB23av ) {
             if ( eta <= 0.52 )  {   temp_dphi = scaledPhi(temp_dphi, DT14_1[3]); }
	     if ( eta >  0.52 )  {   temp_dphi = scaledPhi(temp_dphi, DT14_2[3]); }
	     pt  = getPt( DT14, eta , temp_dphi )[0];
	     spt = getPt( DT14, eta , temp_dphi )[1];
          }
        }
        ptEstimate.push_back( pt*sign );
        sptEstimate.push_back( spt );
      }

      // MB2 is inner-most
      if (layer0 == -2 && temp_dphi > 0.0001 ) {
        // MB3 is outer-most
        if ( layer1 == -3) {

          if ( eta <= 0.6 )  {   temp_dphi = scaledPhi(temp_dphi, DT23_1[3]); }
          if ( eta >  0.6 )  {   temp_dphi = scaledPhi(temp_dphi, DT23_2[3]); }
          pt  = getPt( DT23, eta , temp_dphi )[0];
          spt = getPt( DT23, eta , temp_dphi )[1];
        }
        // MB4 is outer-most
        else {

          if ( eta <= 0.52 )  {   temp_dphi = scaledPhi(temp_dphi, DT24_1[3]); }
          if ( eta >  0.52 )  {   temp_dphi = scaledPhi(temp_dphi, DT24_2[3]); }
          pt  = getPt( DT24, eta , temp_dphi )[0];
          spt = getPt( DT24, eta , temp_dphi )[1];
        }
        ptEstimate.push_back( pt*sign );
        sptEstimate.push_back( spt );
      }

      // MB3 is inner-most    -> only marginally useful to pick up the charge
      if (layer0 == -3 && temp_dphi > 0.0001 ) {
        // MB4 is outer-most

        if ( eta <= 0.51 )  {   temp_dphi = scaledPhi(temp_dphi, DT34_1[3]); }
        if ( eta >  0.51 )  {   temp_dphi = scaledPhi(temp_dphi, DT34_2[3]); }
        pt  = getPt( DT34, eta , temp_dphi )[0];
        spt = getPt( DT34, eta , temp_dphi )[1];
        ptEstimate.push_back( pt*sign );   
        sptEstimate.push_back( spt );
      }
    }   
  //}
  
  
  // Compute weighted average if have more than one estimator
  if (ptEstimate.size() > 0 ) weightedPt( ptEstimate, sptEstimate, thePt, theSpt);

}


/*
 * estimatePtOverlap
 *
 */
void MuonSeedCreator::estimatePtOverlap(const SegmentContainer& seg, const std::vector<int>& layers, double& thePt, double& theSpt) {

  int size = layers.size();

  thePt  = defaultMomentum;
  theSpt = defaultMomentum;

  SegmentContainer segCSC;
  std::vector<int> layersCSC;
  SegmentContainer segDT;
  std::vector<int> layersDT;

  // DT layers are numbered as -4 to -1, whereas CSC layers go from 0 to 4:
  for ( unsigned j = 0; j < layers.size(); ++j ) {
    if ( layers[j] > -1 ) {
      segCSC.push_back(seg[j]);
      layersCSC.push_back(layers[j]);
    } 
    else {
      segDT.push_back(seg[j]);
      layersDT.push_back(layers[j]);
    } 
  }

  std::vector<double> ptEstimate;
  std::vector<double> sptEstimate;

  GlobalPoint segPos[2];
  int layer0 = layers[0];
  segPos[0] = seg[0]->globalPosition();
  float eta = fabs(segPos[0].eta());
  //std::cout<<" estimate OL "<<std::endl;
    
  if ( segDT.size() > 0 && segCSC.size() > 0 ) {
    int layer1 = layers[size-1];
    segPos[1] = seg[size-1]->globalPosition();
  
    double dphi = segPos[0].phi() - segPos[1].phi();
    double temp_dphi = dphi;

    // Ensure that delta phi is not too small to prevent pt from blowing up
    
    double sign = 1.0;
    if (temp_dphi < 0.) {      
      temp_dphi = -temp_dphi;
      sign = -1.0;  
    } 
     
    if (temp_dphi < 0.0001 ) { 
         temp_dphi = 0.0001 ;
         thePt = theMaxMomentum ;
         theSpt = theMaxMomentum*0.25 ; 
         ptEstimate.push_back( thePt*sign );
         sptEstimate.push_back( theSpt );
    }
  
    // MB1 is inner-most
    if ( layer0 == -1 && temp_dphi > 0.0001 ) {
      // ME1/3 is outer-most
      if ( layer1 == 1 ) {
        temp_dphi = scaledPhi(temp_dphi, OL_1213[3]);
        thePt  = getPt( OL1213, eta , temp_dphi )[0];
        theSpt = getPt( OL1213, eta , temp_dphi )[1];
      }
      // ME2 is outer-most
      else if ( layer1 == 2) {
        temp_dphi = scaledPhi(temp_dphi, OL_1222[3]);
        thePt  = getPt( OL1222, eta , temp_dphi )[0];
        theSpt = getPt( OL1222, eta , temp_dphi )[1];
      }
      // ME3 is outer-most
      else {
        temp_dphi = scaledPhi(temp_dphi, OL_1232[3]);
        thePt  = getPt( OL1232, eta , temp_dphi )[0];
        theSpt = getPt( OL1232, eta , temp_dphi )[1];
      }
      ptEstimate.push_back(thePt*sign);
      sptEstimate.push_back(theSpt);
    } 
    // MB2 is inner-most
    if ( layer0 == -2 && temp_dphi > 0.0001 ) {
      // ME1/3 is outer-most
      if ( layer1 == 1 ) {
        temp_dphi = scaledPhi(temp_dphi, OL_2213[3]);
        thePt  = getPt( OL2213, eta , temp_dphi )[0];
        theSpt = getPt( OL2213, eta , temp_dphi )[1];
        ptEstimate.push_back(thePt*sign);
        sptEstimate.push_back(theSpt);
      }
      // ME2 is outer-most
      if ( layer1 == 2) {
        temp_dphi = scaledPhi(temp_dphi, OL_2222[3]);
        thePt  = getPt( OL2222, eta , temp_dphi )[0];
        theSpt = getPt( OL2222, eta , temp_dphi )[1];
      }
    }
  } 

  if ( segDT.size() > 1 ) {
    estimatePtDT(segDT, layersDT, thePt, theSpt);
    ptEstimate.push_back(thePt);
    sptEstimate.push_back(theSpt);
  } 

  /*
  // not useful ....and pt estimation is bad
  if ( segCSC.size() > 1 ) {
    // don't estimate pt without ME1 information
    bool CSCLayer1=false;
    for (unsigned i=0; i< layersCSC.size(); i++) {
        if ( layersCSC[i]==0 || layersCSC[i]==1 ) CSCLayer1 = true;
    }
    if (CSCLayer1) {
       estimatePtCSC(segCSC, layersCSC, thePt, theSpt);
       ptEstimate.push_back(thePt);
       sptEstimate.push_back(theSpt);
    }
  }
  */

  // Compute weighted average if have more than one estimator
  if (ptEstimate.size() > 0 ) weightedPt( ptEstimate, sptEstimate, thePt, theSpt);

}
/*
 *
 *   estimate Pt for single segment events
 *
 */
void MuonSeedCreator::estimatePtSingle(const SegmentContainer& seg, const std::vector<int>& layers, double& thePt, double& theSpt) {

  thePt  = defaultMomentum;
  theSpt = defaultMomentum;

  GlobalPoint segPos = seg[0]->globalPosition();
  double eta = segPos.eta();
  GlobalVector gv = seg[0]->globalDirection();

  // Psi is angle between the segment origin and segment direction
  // Use dot product between two vectors to get Psi in global x-y plane
  double cosDpsi  = (gv.x()*segPos.x() + gv.y()*segPos.y());
  cosDpsi /= sqrt(segPos.x()*segPos.x() + segPos.y()*segPos.y());
  cosDpsi /= sqrt(gv.x()*gv.x() + gv.y()*gv.y());

  double axb = ( segPos.x()*gv.y() ) - ( segPos.y()*gv.x() ) ;
  double sign = (axb < 0.) ? 1.0 : -1.0;

  double dpsi = fabs(acos(cosDpsi)) ;
  if ( dpsi > 1.570796 ) {
      dpsi = 3.141592 - dpsi;
      sign = -1.*sign ;
  }
  if (fabs(dpsi) < 0.00005) {
     dpsi = 0.00005;
  }

  // the 1st layer
  if ( layers[0] == -1 ) {
     // MB10
     if ( fabs(eta) < 0.3 ) {
        dpsi = scaledPhi(dpsi, SMB_10S[3] );
        thePt  = getPt( SMB10, eta , dpsi )[0];
        theSpt = getPt( SMB10, eta , dpsi )[1];
     }
     // MB11
     if ( fabs(eta) >= 0.3 && fabs(eta) < 0.82 ) {
        dpsi = scaledPhi(dpsi, SMB_11S[3] );
        thePt  = getPt( SMB11, eta , dpsi )[0];
        theSpt = getPt( SMB11, eta , dpsi )[1];
     }
     // MB12
     if ( fabs(eta) >= 0.82 && fabs(eta) < 1.2 ) {
        dpsi = scaledPhi(dpsi, SMB_12S[3] );
        thePt  = getPt( SMB12, eta , dpsi )[0];
        theSpt = getPt( SMB12, eta , dpsi )[1];
     }
  }
  if ( layers[0] == 1 ) {
     // ME13
     if ( fabs(eta) > 0.92 && fabs(eta) < 1.16 ) {
        dpsi = scaledPhi(dpsi, SME_13S[3] );
        thePt  = getPt( SME13, eta , dpsi )[0];
        theSpt = getPt( SME13, eta , dpsi )[1];
     }
     // ME12
     if ( fabs(eta) >= 1.16 && fabs(eta) <= 1.6 ) {
        dpsi = scaledPhi(dpsi, SME_12S[3] );
        thePt  = getPt( SME12, eta , dpsi )[0];
        theSpt = getPt( SME12, eta , dpsi )[1];
     }
  }
  if ( layers[0] == 0  ) {
     // ME11
     if ( fabs(eta) > 1.6 ) {
        dpsi = scaledPhi(dpsi, SMB_11S[3] );
        thePt  = getPt( SME11, eta , dpsi )[0];
        theSpt = getPt( SME11, eta , dpsi )[1];
     }
  }
  // the 2nd layer
  if ( layers[0] == -2 ) {
     // MB20
     if ( fabs(eta) < 0.25 ) {
        dpsi = scaledPhi(dpsi, SMB_20S[3] );
        thePt  = getPt( SMB20, eta , dpsi )[0];
        theSpt = getPt( SMB20, eta , dpsi )[1];
     }
     // MB21
     if ( fabs(eta) >= 0.25 && fabs(eta) < 0.72 ) {
        dpsi = scaledPhi(dpsi, SMB_21S[3] );
        thePt  = getPt( SMB21, eta , dpsi )[0];
        theSpt = getPt( SMB21, eta , dpsi )[1];
     }
     // MB22
     if ( fabs(eta) >= 0.72 && fabs(eta) < 1.04 ) {
        dpsi = scaledPhi(dpsi, SMB_22S[3] );
        thePt  = getPt( SMB22, eta , dpsi )[0];
        theSpt = getPt( SMB22, eta , dpsi )[1];
     }
  }
  if ( layers[0] == 2 ) {
     // ME22
     if ( fabs(eta) > 0.95 && fabs(eta) <= 1.6 ) {
        dpsi = scaledPhi(dpsi, SME_22S[3] );
        thePt  = getPt( SME22, eta , dpsi )[0];
        theSpt = getPt( SME22, eta , dpsi )[1];
     }
     // ME21
     if ( fabs(eta) > 1.6 && fabs(eta) < 2.45 ) {
        dpsi = scaledPhi(dpsi, SME_21S[3] );
        thePt  = getPt( SME21, eta , dpsi )[0];
        theSpt = getPt( SME21, eta , dpsi )[1];
     }
  }

  // the 3rd layer
  if ( layers[0] == -3 ) {
     // MB30
     if ( fabs(eta) <= 0.22 ) {
        dpsi = scaledPhi(dpsi, SMB_30S[3] );
        thePt  = getPt( SMB30, eta , dpsi )[0];
        theSpt = getPt( SMB30, eta , dpsi )[1];
     }
     // MB31
     if ( fabs(eta) > 0.22 && fabs(eta) <= 0.6 ) {
        dpsi = scaledPhi(dpsi, SMB_31S[3] );
        thePt  = getPt( SMB31, eta , dpsi )[0];
        theSpt = getPt( SMB31, eta , dpsi )[1];
     }
     // MB32
     if ( fabs(eta) > 0.6 && fabs(eta) < 0.95 ) {
        dpsi = scaledPhi(dpsi, SMB_32S[3] );
        thePt  = getPt( SMB32, eta , dpsi )[0];
        theSpt = getPt( SMB32, eta , dpsi )[1];
     }
  }
  thePt = fabs(thePt)*sign;
  theSpt = fabs(theSpt);

  return;
}

// setup the minimum value for obvious showering cases  
void MuonSeedCreator::estimatePtShowering(int& NShowers, int& NShowerSegments,  double& thePt, double& theSpt) {

  if ( NShowers > 2 && thePt < 300. ) {
     thePt  = 800. ;
     theSpt = 200. ; 
  } 
  if ( NShowers == 2 && NShowerSegments >  11 && thePt < 150. ) {
     thePt = 280. ;
     theSpt = 70. ; 
  }
  if ( NShowers == 2 && NShowerSegments <= 11 && thePt < 50.  ) {
     thePt =  80.;
     theSpt = 40. ;
  }
  if ( NShowers == 1 && NShowerSegments <= 5 && thePt < 10. ) {
     thePt = 16. ;
     theSpt = 8. ; 
  }

}

/*
 * weightedPt
 *
 * Look at delta phi between segments to determine pt as:
 * pt = (c_1 * eta + c_2) / dphi
 */
void MuonSeedCreator::weightedPt(const std::vector<double>& ptEstimate, const std::vector<double>& sptEstimate, double& thePt, double& theSpt) {

 
  int size = ptEstimate.size();

  // If only one element, by-pass computation below
  if (size < 2) {
    thePt = ptEstimate[0];
    theSpt = sptEstimate[0];
    return;
  }

  double charge = 0.;
  // If have more than one pt estimator, first look if any estimator is biased
  // by comparing the charge of each estimator

  for ( unsigned j = 0; j < ptEstimate.size(); j++ ) {
    //std::cout<<" weighting pt: "<< ptEstimate[j] <<std::endl; 
    if ( ptEstimate[j] < 0. ) {
      // To prevent from blowing up, add 0.1 
      charge -= 1. * (ptEstimate[j]*ptEstimate[j]) / (sptEstimate[j]*sptEstimate[j] );  // weight by relative error on pt
    } else {
      charge += 1. * (ptEstimate[j]*ptEstimate[j]) / (sptEstimate[j]*sptEstimate[j] );  // weight by relative error on pt
    }
  }
 
  // No need to normalize as we want to know only sign ( + or - )
  if (charge < 0.) {
    charge = -1.;
  } else {
    charge = 1.;
  }

  //int n = 0;
  double weightPtSum  = 0.;
  double sigmaSqr_sum = 0.;
          
  // Now, we want to compute average Pt using estimators with "correct" charge
  // This is to remove biases
  for ( unsigned j = 0; j < ptEstimate.size(); ++j ) {
    //if ( (minpt_ratio < 0.5) && (fabs(ptEstimate[j]) < 5.0) ) continue;
    //if ( ptEstimate[j] * charge > 0. ) {
      //n++;
      sigmaSqr_sum += 1.0 / (sptEstimate[j]*sptEstimate[j]);
      weightPtSum  += fabs(ptEstimate[j])/(sptEstimate[j]*sptEstimate[j]);
    //}
  }
  /*  
  if (n < 1) {
    thePt  = defaultMomentum*charge;
    theSpt = defaultMomentum; 
    return;
  } 
  */
  // Compute weighted mean and error

  thePt  = (charge*weightPtSum) / sigmaSqr_sum;
  theSpt = sqrt( 1.0 / sigmaSqr_sum ) ;

  //std::cout<<" final weighting : "<< thePt <<" ~ "<< fabs( theSpt/thePt ) <<std::endl;

  return;
}

std::vector<double> MuonSeedCreator::getPt(const std::vector<double>& vPara, double eta, double dPhi ) {

       double h  = fabs(eta);
       double estPt  = ( vPara[0] + vPara[1]*h + vPara[2]*h*h ) / dPhi;
       double estSPt = ( vPara[3] + vPara[4]*h + vPara[5]*h*h ) * estPt;
       std::vector<double> paraPt ;
       paraPt.push_back( estPt );
       paraPt.push_back( estSPt ) ;

       //std::cout<<"      pt:"<<estPt<<" +/-"<< estSPt<<"   h:"<<eta<<"  df:"<<dPhi<<std::endl;
       return paraPt ;
}

double MuonSeedCreator::scaledPhi( double dphi, double t1) {

  if (dphi != 0. ) {

    double oPhi = 1./dphi ;
    dphi = dphi /( 1. + t1/( oPhi + 10. ) ) ;
    return dphi ;

  } else {
    return dphi ;
  } 

}

