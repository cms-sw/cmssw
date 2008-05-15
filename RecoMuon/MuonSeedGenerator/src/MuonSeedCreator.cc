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
  // load seed parameters 
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
TrajectorySeed MuonSeedCreator::createSeed(int type, SegmentContainer seg, std::vector<int> layers, std::vector<int> badSeedLayer ) {

  // The index of the station closest to the IP
  int last = 0;

  double ptmean = theMinMomentum;
  double sptmean = theMinMomentum;

  AlgebraicVector t(4);
  AlgebraicSymMatrix mat(5,0) ;

  //LocalPoint segPos = seg[last]->localPosition();
  LocalPoint segPos;

  // Compute the pt according to station types used;
  if (type == 1 ) estimatePtCSC(seg, layers, ptmean, sptmean);
  if (type == 2 ) estimatePtOverlap(seg, layers, ptmean, sptmean);
  if (type == 3 ) estimatePtDT(seg, layers, ptmean, sptmean);
  if (type == 4 ) estimatePtSingle(seg, layers, ptmean, sptmean);
  // type 5 are the seeding for ME1/4
  if (type == 5 ) estimatePtCSC(seg, layers, ptmean, sptmean);

  // Minimal pt
  double charge = 1.0;
  if (ptmean < 0.) charge = -1.0; 
  if ( (charge * ptmean) < theMinMomentum ) {
    ptmean  = theMinMomentum * charge;
    sptmean = theMinMomentum;
  }
  else if ( (charge * ptmean) > theMaxMomentum ) {
    ptmean  = theMaxMomentum * charge;
    sptmean = theMaxMomentum;
  }

  LocalTrajectoryParameters param;
  double p_err =0.0;
  // check chi2 to determine "inside-out" or "outside-in"
  bool out_in = false;
  bool in_out = false;
  bool expand = false;
  unsigned int best_seg= 0;
  double chi2_dof=-1.0;

  // determine the seed layer
  if ( seg.size() ==1 ) {
     best_seg = 0; 
  } else {
     unsigned int ini_seg = 0;
     // avoid generating seed from  1st layer(ME1/1)
     if ( type == 5 )  ini_seg = 1;
        
     for (unsigned int i = ini_seg ; i < seg.size(); i++) {
         double dof = static_cast<double>(seg[i]->degreesOfFreedom());
         if (chi2_dof < 0.0) {
            chi2_dof = seg[i]->chi2() / dof ;
            best_seg = i;
         }
         if ( chi2_dof > ( seg[i]->chi2()/dof ) ) {
            // avoiding showering chamber
            /*
            bool shower = false;
            for (unsigned int j = 0; j< badSeedLayer.size(); j++ ) {
                if (badSeedLayer[j] == layers[i]) shower = true;
            }
            if (shower) continue;
            */
            chi2_dof = seg[i]->chi2() / dof ;
            best_seg = i;
         }
     }  
  }

  badSeedLayer.clear();
  // determine seed direction
  if ( abs(layers[best_seg])== 1 ) {
     in_out = true;
  } else if ( (abs(layers[best_seg])== 0) && ( seg.size()==1 ) ) {
     in_out = true;
  } else if ( (abs(layers[best_seg]) == 4) && ( type ==5 ) ){
     out_in = true;
  } else if ( (abs(layers[best_seg]) >= 3) && ( type !=5 ) ){
     out_in = true;
  } else {
     expand = true;
  }

  
  if ( type==1 || type==5 ) {
  //if ( type==1 ) {
     // Fill the LocalTrajectoryParameters
     /// get the Global position
     last = best_seg;
     GlobalVector mom = seg[last]->globalPosition()-GlobalPoint();
     segPos = seg[last]->localPosition();
     /// get the Global direction
     GlobalVector polar(GlobalVector::Spherical(mom.theta(),seg[last]->globalDirection().phi(),1.));

     /// count the energy loss - from parameterization
     double ptRatio = 0.994 - (2.14/(ptmean -1)) + (3.46/((ptmean-1)*(ptmean-1)));
     ptmean = ptmean*ptRatio ;

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
     if (in_out) {
        if (type==5) { mat[0][0] = 4.0*mat[0][0]; }
        mat[1][1]= 3.*mat[1][1];
	mat[2][2]= 3.*mat[2][2];
	mat[3][3]= 2.*mat[3][3];
	mat[4][4]= 2.*mat[4][4];
     }
     else {
        mat[0][0]= 2.25*mat[0][0];
        mat[1][1]= 2.25*mat[1][1];
	mat[3][3]= 2.25*mat[3][3];
	mat[4][4]= 2.25*mat[4][4];
     }
  }
  /*else if ( type==5 ) {
     // Fill the LocalTrajectoryParameters
     /// get the Global position
     last = best_seg;
     GlobalVector mom = seg[last]->globalPosition()-GlobalPoint();
     //GlobalVector polar(GlobalVector::Spherical(mom.theta(),seg[last]->globalDirection().phi(),1.));
     //polar *= fabs(ptmean)/polar.perp();
     //LocalVector segDirFromPos = seg[last]->det()->toLocal(polar);

     LocalPoint  segLocalPos = seg[last]->localPosition();
     LocalVector segLocalDir = seg[last]->localDirection();
     double totalP = fabs( ptmean/sin(mom.theta()) );
     double QbP = charge / totalP ;
     double dxdz = segLocalDir.x()/segLocalDir.z();
     double dydz = segLocalDir.y()/segLocalDir.z();
     //double dydz = segDirFromPos.y()/segDirFromPos.z();
     double lx = segLocalPos.x();
     double ly = segLocalPos.y();
     double pz_sign =  segLocalDir.z() > 0.0 ? 1.0:-1.0 ;
     LocalTrajectoryParameters param1(QbP,dxdz,dydz,lx,ly,pz_sign,true);
     param = param1;
     p_err =  (sptmean*sptmean)/(totalP*totalP*ptmean*ptmean) ;
     mat = seg[last]->parametersError().similarityT( seg[last]->projectionMatrix() );
     mat[0][0]= 4.0*p_err;
     mat[3][3]= 4.0*mat[3][3];
     mat[4][4]= 4.0*mat[4][4];
  }*/
  else {
     // Fill the LocalTrajectoryParameters
     /// get the Global position
     last = 0;
     segPos = seg[last]->localPosition();
     GlobalVector mom = seg[last]->globalPosition()-GlobalPoint();
     /// get the Global direction
     //GlobalVector polar(GlobalVector::Spherical(mom.theta(),seg[last]->globalDirection().phi(),1.));
     GlobalVector polar(GlobalVector::Spherical(seg[last]->globalDirection().theta(),seg[last]->globalDirection().phi(),1.));

     /// count the energy loss - from parameterization
     double ptRatio = 1.0 - (2.808/(ptmean -1)) + (4.546/((ptmean-1)*(ptmean-1)));
     ptmean = ptmean*ptRatio ;

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

  
  LocalTrajectoryError error(mat);
  
  // Create the TrajectoryStateOnSurface
  TrajectoryStateOnSurface tsos(param, error, seg[last]->det()->surface(),&*BField);
  
  // Take the DetLayer on which relies the segment
  DetId id = seg[last]->geographicalId();

  // Transform it in a TrajectoryStateOnSurface
  TrajectoryStateTransform tsTransform;
  
  //PTrajectoryStateOnDet *seedTSOS = tsTransform.persistentState( tsos, id.rawId());
  std::auto_ptr<PTrajectoryStateOnDet> seedTSOS(tsTransform.persistentState( tsos, id.rawId()));  

  edm::OwnVector<TrackingRecHit> container;
  for (unsigned l=0; l<seg.size(); l++) {
      container.push_back( seg[l]->hit()->clone() ); 
      //container.push_back(seg[l]->hit()); 
  }

  TrajectorySeed theSeed(*seedTSOS,container,alongMomentum);
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
void MuonSeedCreator::estimatePtCSC(SegmentContainer seg, std::vector<int> layers, double& thePt, double& theSpt ) {

  // reverse the segment and layer container first for pure CSC case
  if ( layers[0] > layers[ layers.size()-1 ] ) {
     reverse( layers.begin(), layers.end() );
     reverse( seg.begin(), seg.end() );
  }

  std::vector<double> ptEstimate;
  std::vector<double> sptEstimate;

  thePt  = defaultMomentum;
  theSpt = defaultMomentum;

  double pt = 0.;
  double spt = 0.;   
  GlobalPoint  segPos[2];
  GlobalVector segVec[2];


  unsigned size = seg.size();

  //if (size < 2) return;
  
  int layer0 = layers[0];
  segPos[0] = seg[0]->globalPosition();
  segVec[0] = seg[0]->globalDirection();
  float eta = fabs(segPos[0].eta());

  unsigned idx1 = size;

  if (size > 1) {
    while ( idx1 > 1 ) {
      idx1--;
      int layer1 = layers[idx1];
      if (layer0 == layer1) continue;
      segPos[1] = seg[idx1]->globalPosition();      
      segVec[1] = seg[idx1]->globalDirection();      
      //eta = fabs(segPos[0].eta());  // Eta is better determined from track closest from IP
      eta = fabs(segPos[1].eta()); 

      double dphi = segPos[0].phi() - segPos[1].phi();
      double temp_dphi = dphi;
      double dphiV = segVec[0].phi() - segVec[1].phi();
      double temp_dphiV = dphiV;
       
      // Ensure that delta phi is not too small to prevent pt from blowing up
      
      double sign = 1.0;  
      if (temp_dphi < 0.) {
        temp_dphi = -1.0*temp_dphi;
        sign = -1.0;
      }
      if (temp_dphiV > 0.) {
        temp_dphiV = -1.0*temp_dphiV;
      }
      

      if (temp_dphi < 1.0e-6) temp_dphi = 1.0e-6;

      // ME1 is inner-most
      if ( layer0 == 0 ) {
        // ME1/2 is outer-most
        if ( layer1 == 1 ) {
          pt  = getPt( CSC01, eta , temp_dphi )[0];
          spt = getPt( CSC01, eta , temp_dphi )[1];
        }  
        // ME2 is outer-most
        else if ( layer1 == 2  ) {
          pt  = getPt( CSC02, eta , temp_dphi )[0];
          spt = getPt( CSC02, eta , temp_dphi )[1];
        }
        // ME3 is outer-most
        else if ( layer1 == 3 ) {
          pt  = getPt( CSC03, eta , temp_dphi )[0];
          spt = getPt( CSC03, eta , temp_dphi )[1];
        }
        // ME4 is outer-most
        else {
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
          pt  = getPt( CSC12, eta , temp_dphi )[0];
          spt = getPt( CSC12, eta , temp_dphi )[1];
        }
        // ME3 is outer-most
        else if ( layer1 == 3 ) {
          pt  = getPt( CSC13, eta , temp_dphi )[0];
          spt = getPt( CSC13, eta , temp_dphi )[1];
        }
        // ME4 is outer-most
        else {
          pt  = getPt( CSC14, eta , temp_dphi )[0];
          spt = getPt( CSC14, eta , temp_dphi )[1];
        }
        ptEstimate.push_back( pt*sign );
        sptEstimate.push_back( spt );
      }

      // ME2 is inner-most
      if ( layer0 == 2 ) {
        // ME3 is outer-most
        if ( layer1 == 3 ) {
          pt  = getPt( CSC23, eta , temp_dphi )[0];
          spt = getPt( CSC23, eta , temp_dphi )[1];
        }
        // ME4 is outer-most
        else {
          pt  = getPt( CSC24, eta , temp_dphi )[0];
          spt = getPt( CSC24, eta , temp_dphi )[1];
        }
        ptEstimate.push_back( pt );   
        sptEstimate.push_back( spt );
      }

      // ME3 is inner-most
      if ( layer0 == 3 ) {
        pt  = getPt( CSC34, eta , temp_dphi )[0];
        spt = getPt( CSC34, eta , temp_dphi )[1];
        ptEstimate.push_back( pt*sign );   
        sptEstimate.push_back( spt );
      }

      /*
      // Estimate pT with dPhi from segment directions
      // ME1 is inner-most
      if ( layer0 == 0 || layer0 == 1 ) {
        // ME2 is outer-most
        if ( (layer1 == 2) && (eta < 1.6) ) {
          pt  = (   2.133 - 3.772*eta - 1.367*eta*eta) / temp_dphiV;
          spt = (  -6.699 - 9.242*eta - 3.336*eta*eta) * pt;
        }
        // ME3 is outer-most
        if ( (layer1 == 3) && (eta < 1.7) ) {
          pt  = (   3.647 - 5.912*eta - 1.978*eta*eta) / temp_dphiV; 
          spt = (  -4.554 + 5.818*eta - 1.973*eta*eta) * pt;
        }

        if ( (spt/pt) < 0.8  ) {
           ptEstimate.push_back( pt*sign );
           sptEstimate.push_back( spt );
        }
      }

      // ME2 is inner-most
      if ( layer0 == 2 ) {
        // ME3 is outer-most
        if ( layer1 == 3 ) {
          pt  = (  1.108 - 1.654*eta + 0.477*eta*eta) / temp_dphi; 
          spt = ( -4.195 - 4.157*eta - 1.141*eta*eta) * pt;
        }
        // ME4 is outer-most
        else {
          pt  = ( -0.1076 - 0.481*eta + 0.1803*eta*eta) / temp_dphi;
          spt = (  5.115  - 5.000*eta + 1.089*eta*eta ) * pt;
        }
        if ( (spt/pt) < 0.8  ) {
           ptEstimate.push_back( pt );   
           sptEstimate.push_back( spt );
        }
      }
      */
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
void MuonSeedCreator::estimatePtDT(SegmentContainer seg, std::vector<int> layers, double& thePt, double& theSpt) {

  std::vector<double> ptEstimate;
  std::vector<double> sptEstimate;

  thePt  = defaultMomentum;
  theSpt = defaultMomentum;

  double pt = 0.;
  double spt = 0.;   
  GlobalPoint segPos[2];
  float eta = 0.;

  unsigned size = seg.size();

  if (size < 2) return;
  
  int layer0 = layers[0];
  //segPos[0] = seg[0]->globalPosition();
  //float eta = fabs(segPos[0].eta());

  // inner-most layer
  for ( unsigned idx0 = 0; idx0 < size-1; ++idx0 ) {
    layer0 = layers[idx0];
    segPos[0]  = seg[idx0]->globalPosition();
    // outer-most layer
    for ( unsigned idx1 = idx0+1; idx1 < size; ++idx1 ) {

      int layer1 = layers[idx1];
      segPos[1] = seg[idx1]->globalPosition();      
 
      // using eta from outer layer because parameterization do so ..
      //eta = fabs(segPos[0].eta());  
      eta = fabs(segPos[1].eta());  
      if (layer1 == -4) eta = fabs(segPos[0].eta());

      double dphi = segPos[0].phi() - segPos[1].phi();
      double temp_dphi = dphi;

      // Ensure that delta phi is not too small to prevent pt from blowing up
      
      double sign = 1.0;  
      if (temp_dphi < 0.) {
        temp_dphi = -temp_dphi;
        sign = -1.0;
      }
      
      if (temp_dphi < 1.0e-6) temp_dphi = 1.0e-6;

      // MB1 is inner-most
      if (layer0 == -1) {
        // MB2 is outer-most
        if (layer1 == -2) {
          pt  = getPt( DT12, eta , temp_dphi )[0];
          spt = getPt( DT12, eta , temp_dphi )[1];
        }
        // MB3 is outer-most
        else if (layer1 == -3) {
          pt  = getPt( DT13, eta , temp_dphi )[0];
          spt = getPt( DT13, eta , temp_dphi )[1];
        }
        // MB4 is outer-most
        else {
          pt  = getPt( DT14, eta , temp_dphi )[0];
          spt = getPt( DT14, eta , temp_dphi )[1];
        }
        ptEstimate.push_back( pt*sign );
        sptEstimate.push_back( spt );
      }

      // MB2 is inner-most
      if (layer0 == -2) {
        // MB3 is outer-most
        if ( layer1 == -3) {
          pt  = getPt( DT23, eta , temp_dphi )[0];
          spt = getPt( DT23, eta , temp_dphi )[1];
        }
        // MB4 is outer-most
        else {
          pt  = getPt( DT24, eta , temp_dphi )[0];
          spt = getPt( DT24, eta , temp_dphi )[1];
        }
        ptEstimate.push_back( pt*sign );
        sptEstimate.push_back( spt );
      }

      // MB3 is inner-most    -> only marginally useful to pick up the charge
      if (layer0 == -3) {
        // MB4 is outer-most
        pt  = getPt( DT34, eta , temp_dphi )[0];
        spt = getPt( DT34, eta , temp_dphi )[1];
        ptEstimate.push_back( pt*sign );   
        sptEstimate.push_back( spt );
      }
    }   
  }
  
  
  // Compute weighted average if have more than one estimator
  if (ptEstimate.size() > 0 ) weightedPt( ptEstimate, sptEstimate, thePt, theSpt);

}


/*
 * estimatePtOverlap
 *
 */
void MuonSeedCreator::estimatePtOverlap(SegmentContainer seg, std::vector<int> layers, double& thePt, double& theSpt) {

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
     
    if (temp_dphi < 1.0e-6) temp_dphi = 1.0e-6;
  
    // MB1 is inner-most
    if ( layer0 == -1 ) {
      // ME1/3 is outer-most
     if ( layer1 == 1 ) {
        thePt  = getPt( OL1213, eta , temp_dphi )[0];
        theSpt = getPt( OL1213, eta , temp_dphi )[1];
      }
      // ME2 is outer-most
     else if ( layer1 == 2) {
        thePt  = getPt( OL1222, eta , temp_dphi )[0];
        theSpt = getPt( OL1222, eta , temp_dphi )[1];
      }
      // ME3 is outer-most
      else {
        thePt  = getPt( OL1232, eta , temp_dphi )[0];
        theSpt = getPt( OL1232, eta , temp_dphi )[1];
      }
      ptEstimate.push_back(thePt*sign);
      sptEstimate.push_back(theSpt);
    } 
    // MB2 is inner-most
    if ( layer0 == -2 ) {
      // ME1/3 is outer-most
      if ( layer1 == 1 ) {
        thePt  = getPt( OL2213, eta , temp_dphi )[0];
        theSpt = getPt( OL2213, eta , temp_dphi )[1];
        ptEstimate.push_back(thePt*sign);
        sptEstimate.push_back(theSpt);
      }
    }
  } 

  if ( segDT.size() > 1 ) {
    estimatePtDT(segDT, layersDT, thePt, theSpt);
    ptEstimate.push_back(thePt);
    sptEstimate.push_back(theSpt);
  } 

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

  // Compute weighted average if have more than one estimator
  if (ptEstimate.size() > 0 ) weightedPt( ptEstimate, sptEstimate, thePt, theSpt);

}
/*
 *
 *   estimate Pt for single segment events
 *
 */
void MuonSeedCreator::estimatePtSingle(SegmentContainer seg, std::vector<int> layers, double& thePt, double& theSpt) {

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
  }
  if (fabs(dpsi) < 0.00005) {
     dpsi = 0.00005;
  }

  // the 1st layer
  if ( layers[0] == -1 ) {
     // MB10
     if ( fabs(eta) < 0.3 ) {
        thePt  = getPt( SMB10, eta , dpsi )[0];
        theSpt = getPt( SMB10, eta , dpsi )[1];
     }
     // MB11
     if ( fabs(eta) >= 0.3 && fabs(eta) < 0.82 ) {
        thePt  = getPt( SMB11, eta , dpsi )[0];
        theSpt = getPt( SMB11, eta , dpsi )[1];
     }
     // MB12
     if ( fabs(eta) >= 0.82 && fabs(eta) < 1.2 ) {
        thePt  = getPt( SMB12, eta , dpsi )[0];
        theSpt = getPt( SMB12, eta , dpsi )[1];
     }
  }
  if ( layers[0] == 1 ) {
     // ME13
     if ( fabs(eta) > 0.92 && fabs(eta) < 1.16 ) {
        thePt  = getPt( SME13, eta , dpsi )[0];
        theSpt = getPt( SME13, eta , dpsi )[1];
     }
     // ME12
     if ( fabs(eta) >= 1.16 && fabs(eta) <= 1.6 ) {
        thePt  = getPt( SME12, eta , dpsi )[0];
        theSpt = getPt( SME12, eta , dpsi )[1];
     }
  }
  if ( layers[0] == 0  ) {
     // ME11
     if ( fabs(eta) > 1.6 && fabs(eta) < 2.45 ) {
        thePt  = getPt( SME11, eta , dpsi )[0];
        theSpt = getPt( SME11, eta , dpsi )[1];
     }
  }
  // the 2nd layer
  if ( layers[0] == -2 ) {
     // MB20
     if ( fabs(eta) < 0.25 ) {
        thePt  = getPt( SMB20, eta , dpsi )[0];
        theSpt = getPt( SMB20, eta , dpsi )[1];
     }
     // MB21
     if ( fabs(eta) >= 0.25 && fabs(eta) < 0.72 ) {
        thePt  = getPt( SMB21, eta , dpsi )[0];
        theSpt = getPt( SMB21, eta , dpsi )[1];
     }
     // MB22
     if ( fabs(eta) >= 0.72 && fabs(eta) < 1.04 ) {
        thePt  = getPt( SMB22, eta , dpsi )[0];
        theSpt = getPt( SMB22, eta , dpsi )[1];
     }
  }
  if ( layers[0] == 2 ) {
     // ME22
     if ( fabs(eta) > 0.95 && fabs(eta) <= 1.6 ) {
        thePt  = getPt( SME22, eta , dpsi )[0];
        theSpt = getPt( SME22, eta , dpsi )[1];
     }
     // ME21
     if ( fabs(eta) > 1.6 && fabs(eta) < 2.45 ) {
        thePt  = getPt( SME21, eta , dpsi )[0];
        theSpt = getPt( SME21, eta , dpsi )[1];
     }
  }

  // the 3rd layer
  if ( layers[0] == -3 ) {
     // MB30
     if ( fabs(eta) <= 0.22 ) {
        thePt  = getPt( SMB30, eta , dpsi )[0];
        theSpt = getPt( SMB30, eta , dpsi )[1];
     }
     // MB31
     if ( fabs(eta) > 0.22 && fabs(eta) <= 0.6 ) {
        thePt  = getPt( SMB31, eta , dpsi )[0];
        theSpt = getPt( SMB31, eta , dpsi )[1];
     }
     // MB32
     if ( fabs(eta) > 0.6 && fabs(eta) < 0.95 ) {
        thePt  = getPt( SMB32, eta , dpsi )[0];
        theSpt = getPt( SMB32, eta , dpsi )[1];
     }
  }
  thePt = fabs(thePt)*sign;
  theSpt = fabs(theSpt);

  return;
}



/*
 * weightedPt
 *
 * Look at delta phi between segments to determine pt as:
 * pt = (c_1 * eta + c_2) / dphi
 */
void MuonSeedCreator::weightedPt(std::vector<double> ptEstimate, std::vector<double> sptEstimate, double& thePt, double& theSpt) {

 
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
    if ( ptEstimate[j] < 0. ) {
      // To prevent from blowing up, add 0.1 
      charge -= 1. * (ptEstimate[j]*ptEstimate[j]) / (sptEstimate[j]*sptEstimate[j] );  // weight by relative error on pt
    } else {
      charge += 1. * (ptEstimate[j]*ptEstimate[j]) / (sptEstimate[j]*sptEstimate[j] );  // weight by relative error on pt
    }
  }
 
  //std::cout <<" Q= "<<charge<<std::endl;   
  // No need to normalize as we want to know only sign ( + or - )
  if (charge < 0.) {
    charge = -1.;
  } else {
    charge = 1.;
  }

  int n = 0;
  double weightPtSum  = 0.;
  double sigmaSqr_sum = 0.;
          
  // Now, we want to compute average Pt using estimators with "correct" charge
  // This is to remove biases
  for ( unsigned j = 0; j < ptEstimate.size(); ++j ) {
    //if ( (minpt_ratio < 0.5) && (fabs(ptEstimate[j]) < 5.0) ) continue;
    if ( ptEstimate[j] * charge > 0. ) {
      n++;
      sigmaSqr_sum += 1.0 / (sptEstimate[j]*sptEstimate[j]);
      weightPtSum  += ptEstimate[j]/(sptEstimate[j]*sptEstimate[j]);
    }
  }  
  if (n < 1) {
    thePt  = defaultMomentum*charge;
    theSpt = defaultMomentum; 
    return;
  } 
  // Compute weighted mean and error

  thePt  = weightPtSum / sigmaSqr_sum;
  //thePt  = 10.0;
  theSpt = sqrt( 1.0 / sigmaSqr_sum ) ;
  //std::cout<<" pt= "<<thePt<<" sPt= "<<theSpt<< std::endl;
  return;
}

std::vector<double> MuonSeedCreator::getPt(std::vector<double> vPara, double eta, double dPhi ) {

       double h  = fabs(eta);
       double estPt  = ( vPara[0] + vPara[1]*h + vPara[2]*h*h ) / dPhi;
       double estSPt = ( vPara[3] + vPara[4]*h + vPara[5]*h*h ) * estPt;
       std::vector<double> paraPt ;
       paraPt.push_back( estPt );
       paraPt.push_back( estSPt ) ;
       return paraPt ;
}

