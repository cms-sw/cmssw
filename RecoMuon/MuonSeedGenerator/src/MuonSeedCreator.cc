/**
 *  See header file for a description of this class.
 *
 *  \author Shih-Chuan Kao, Dominique Fortin - UCR
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


  // At inner-most layer
  if ( out_in ) {
     TrajectorySeed theSeed(*seedTSOS,container,oppositeToMomentum);
     return theSeed;
  } else if ( expand ) {
            TrajectorySeed theSeed(*seedTSOS,container,anyDirection);
            return theSeed;
  }else {
     //if ( abs(layers[last]) < 2 ) {
	    TrajectorySeed theSeed(*seedTSOS,container,alongMomentum);
	    return theSeed;
     //}
     /*
     // At outer-most layer
     else if (abs(layers[last]) >= 3 ) {
	    TrajectorySeed theSeed(*seedTSOS,container,oppositeToMomentum);
	    return theSeed;
     }
     // In between
     else {
	    TrajectorySeed theSeed(*seedTSOS,container,anyDirection);
	    return theSeed;
     }*/
  }

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
          pt  = ( 0.8348 - (0.4091 * eta)) / temp_dphi;
          spt = ( 0.25 ) * pt;
        }  
        // ME2 is outer-most
        else if ( layer1 == 2  ) {
          pt  = ( 0.7782 - 0.3524*eta + 0.0337*eta*eta) / temp_dphi;
          spt = ( 1.7780 - 1.7289*eta + 0.4915*eta*eta) * pt;
        }
        // ME3 is outer-most
        else if ( layer1 == 3 ) {
          pt  = ( 1.0537 - 0.5768*eta + 0.08545*eta*eta) / temp_dphi; 
          spt = (-0.1875 + 0.2202*eta + 0.02222*eta*eta) * pt;
        }
        // ME4 is outer-most
        else {
          pt  = ( 1.0419 - 0.5709*eta + 0.0876*eta*eta) / temp_dphi;
          spt = ( 1.1362 - 1.0226*eta + 0.3227*eta*eta) * pt;
        }
        ptEstimate.push_back( pt*sign );
        sptEstimate.push_back( spt );
      }

      // ME1/2,ME1/3 is inner-most
      if ( layer0 == 1 ) {
        // ME2 is outer-most
        //if ( (layer1 == 2) && (eta > 1.6) ) {
        //  pt  = ( 0.7782 - 0.3524*eta + 0.0337*eta*eta) / temp_dphi;
        //  spt = ( 1.7780 - 1.7289*eta + 0.4915*eta*eta) * pt;
        //}
        if ( layer1 == 2 ) {
          pt  = ( -0.5474 + 0.8620*eta - 0.2794*eta*eta) / temp_dphi;
          spt = (  3.4666 - 4.3546*eta + 1.4666*eta*eta) * pt;
        }
        // ME3 is outer-most
        //else if ( (layer1 == 3) && (eta > 1.6) ) {
        //  pt  = ( 1.0537 - 0.5768*eta + 0.08545*eta*eta) / temp_dphi; 
        //  spt = (-0.1875 + 0.2202*eta + 0.02222*eta*eta) * pt;
        //}
        else if ( layer1 == 3 ) {
          pt  = ( -0.6416 + 0.9726*eta - 0.2973*eta*eta) / temp_dphi; 
          spt = (  2.0256 - 2.0803*eta + 0.6333*eta*eta) * pt;
        }
        // ME4 is outer-most
        else {
          pt  = ( 1.0419 - 0.5709*eta + 0.0876*eta*eta) / temp_dphi;
          spt = ( 1.1362 - 1.0226*eta + 0.3227*eta*eta) * pt;
        }
        ptEstimate.push_back( pt*sign );
        sptEstimate.push_back( spt );
      }

      // ME2 is inner-most
      if ( layer0 == 2 ) {
        // ME3 is outer-most
        if ( layer1 == 3 ) {
          pt  = (-0.0795 + 0.1140*eta - 0.0288*eta*eta) / temp_dphi; 
          spt = ( 6.4577 - 6.0346*eta + 1.5801*eta*eta) * pt;
        }
        // ME4 is outer-most
        else {
          pt  = ( 0.0157 + 0.0156*eta - 0.0015*eta*eta) / temp_dphi;
          spt = ( 1.3456 - 0.056*eta*eta ) * pt;
        }
        ptEstimate.push_back( pt );   
        sptEstimate.push_back( spt );
      }

      // ME3 is inner-most
      if ( layer0 == 3 ) {
        pt  = ( 0.0519 - 0.0537*eta + 0.0156*eta*eta) / temp_dphi;
        spt = ( 23.241 - 15.425*eta + 2.6788*eta*eta) * pt;        
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

  // Want to look at every possible pairs
  // inner-most layer
  for ( unsigned idx0 = 0; idx0 < size-1; ++idx0 ) {
    layer0 = layers[idx0];
    segPos[0]  = seg[idx0]->globalPosition();
    // outer-most layer
    for ( unsigned idx1 = idx0+1; idx1 < size; ++idx1 ) {

      int layer1 = layers[idx1];
      segPos[1] = seg[idx1]->globalPosition();      
 
<<<<<<< MuonSeedCreator.cc
      
      //eta = fabs(segPos[0].eta());  // Eta is better determined from track closest from IP
      // using the eta from outter layer because parameterization do so
      eta = fabs(segPos[1].eta());
=======
      eta = fabs(segPos[0].eta());  // Eta is better determined from track closest from IP
>>>>>>> 1.7

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
 	  pt  = ( 0.1997 + 0.0569*eta - 0.0928*eta*eta) / temp_dphi;
	  spt = ( 0.1195 - 0.0424*eta + 0.0838*eta*eta) * pt;
        }
        // MB3 is outer-most
        else if (layer1 == -3) {
  	  pt  = ( 0.3419 + 0.0659*eta - 0.1345*eta*eta) / temp_dphi;
	  spt = ( 0.1402 - 0.1069*eta + 0.1753*eta*eta) * pt;
        }
        // MB4 is outer-most
        else {
    	  pt  = ( 0.3892 + 0.0502*eta - 0.1180*eta*eta) / temp_dphi;
	  spt = ( 0.1712 -  0.01 *eta - 0.0658*eta*eta) * pt;
        }
        ptEstimate.push_back( pt*sign );
        sptEstimate.push_back( spt );
      }

      // MB2 is inner-most
      if (layer0 == -2) {
        // MB3 is outer-most
        if ( layer1 == -3) {
  	  pt  = ( 0.1398 + 0.0286*eta - 0.0680*eta*eta) / temp_dphi;
	  spt = ( 0.1908 - 0.0914*eta + 0.1851*eta*eta) * pt;
        }
        // MB4 is outer-most
        else {
    	  pt  = ( 0.1864 + 0.0356*eta - 0.0801*eta*eta) / temp_dphi;
	  spt = ( 0.2709 - 0.1385*eta + 0.2557*eta*eta) * pt;
        }
        ptEstimate.push_back( pt*sign );
        sptEstimate.push_back( spt );
      }

      // MB3 is inner-most    -> only marginally useful to pick up the charge
      if (layer0 == -3) {
        // MB4 is outer-most
        pt  = ( 0.0470 +  0.01*eta - 0.0242*eta*eta) / temp_dphi;
        spt = ( 0.5455 - 0.1407*eta + 0.3828*eta*eta) * pt;
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
        thePt  = ( 1.0650 - 0.8274*eta) / temp_dphi;
        theSpt = ( 0.3208 - 0.1192*eta) * thePt;
      }
      // ME2 is outer-most
     else if ( layer1 == 2) {
       thePt  = ( 1.0250 - 0.7387*eta) / temp_dphi;
       theSpt = ( 0.0393 + 0.1814*eta) * thePt;
      }
      // ME3 is outer-most
      else {
        thePt  = ( 0.6929 - 0.4361*eta ) / temp_dphi;
        theSpt = ( 0.1091 + 0.1757*eta ) * thePt;
      }
      ptEstimate.push_back(thePt*sign);
      sptEstimate.push_back(theSpt);
    } 
    // MB2 is inner-most
    if ( layer0 == -2 ) {
      // ME1/3 is outer-most
      if ( layer1 == 1 ) {
        thePt  = ( 0.6283 - 0.5460*eta) / temp_dphi;
        theSpt = ( 0.5499 - 0.2569*eta) * thePt;
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
       thePt  =(1.457  + 0.008*fabs(eta) ) / dpsi;
       theSpt =(0.1043 - 0.00188*fabs(eta) )*thePt;
     }
     // MB11
     if ( fabs(eta) >= 0.3 && fabs(eta) < 0.82 ) {
       thePt  =(1.551  - 0.1719*fabs(eta) ) / dpsi;
       theSpt =(0.105  - 0.0000*fabs(eta) )*thePt;
     }
     // MB12
     if ( fabs(eta) >= 0.82 && fabs(eta) < 1.2 ) {
       thePt  =(2.232  - 1.005*fabs(eta) ) / dpsi;
       theSpt =(0.120  - 0.000*fabs(eta) )*thePt;
     }
  }
  if ( layers[0] == 1 ) {
     // ME13
     if ( fabs(eta) > 0.92 && fabs(eta) < 1.16 ) {
       thePt  =(-1.816  + 2.226*fabs(eta) ) / dpsi;
       theSpt =( 4.522  - 3.753*fabs(eta) )*thePt;
     }
     // ME12
     if ( fabs(eta) >= 1.16 && fabs(eta) <= 1.6 ) {
       thePt  =(0.2128  + 0.5369*fabs(eta) ) / dpsi;
       theSpt =(0.2666  + 0.01795*fabs(eta) )*thePt;
     }
  }
  if ( layers[0] == 0  ) {
     // ME11
     if ( fabs(eta) > 1.6 && fabs(eta) < 2.45 ) {
       thePt  =( 2.552  - 0.9044*fabs(eta) ) / dpsi;
       theSpt =(-1.742  + 1.156*fabs(eta) )*thePt;
     }
  }
  // the 2nd layer
  if ( layers[0] == -2 ) {
     // MB20
     if ( fabs(eta) < 0.25 ) {
       thePt  =(1.064  - 0.032*fabs(eta) ) / dpsi;
       theSpt =(0.1364 - 0.0054*fabs(eta) )*thePt;
     }
     // MB21
     if ( fabs(eta) >= 0.25 && fabs(eta) < 0.72 ) {
       thePt  =(1.131  - 0.2012*fabs(eta) ) / dpsi;
       theSpt =(0.117  - 0.0654*fabs(eta) )*thePt;
     }
     // MB22
     if ( fabs(eta) >= 0.72 && fabs(eta) < 1.04 ) {
       thePt  =(1.567  - 0.809*fabs(eta) ) / dpsi;
       theSpt =(0.0579  + 0.1466*fabs(eta) )*thePt;
     }
  }
  if ( layers[0] == 2 ) {
     // ME22
     if ( fabs(eta) > 0.95 && fabs(eta) <= 1.6 ) {
       thePt  =(-0.5333  + 0.6436*fabs(eta) ) / dpsi;
       theSpt =( 3.522  - 3.333*fabs(eta) )*thePt;
     }
     // ME21
     if ( fabs(eta) > 1.6 && fabs(eta) < 2.45 ) {
       thePt  =(0.8672  - 0.2218*fabs(eta) ) / dpsi;
       theSpt =(-1.322  + 1.320*fabs(eta) )*thePt;
     }
  }

  // the 3rd layer
  if ( layers[0] == -3 ) {
     // MB30
     if ( fabs(eta) <= 0.22 ) {
       thePt  =(0.539  + 0.0466*fabs(eta) ) / dpsi;
       theSpt =(0.325  - 0.000*fabs(eta) )*thePt;
     }
     // MB31
     if ( fabs(eta) > 0.22 && fabs(eta) <= 0.6 ) {
       thePt  =(0.5917  - 0.1479*fabs(eta) ) / dpsi;
       theSpt =(0.2872  + 0.0995*fabs(eta) )*thePt;
     }
     // MB32
     if ( fabs(eta) > 0.6 && fabs(eta) < 0.95 ) {
       thePt  =(0.6712  - 0.285*fabs(eta) ) / dpsi;
       theSpt =(0.232  + 0.273*fabs(eta) )*thePt;
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
  theSpt = sqrt( 1.0 / sigmaSqr_sum ) ;
  //std::cout<<" pt= "<<thePt<<" sPt= "<<theSpt<< std::endl;
  return;
}


