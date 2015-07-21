/**
 * \file GEMCSCSegAlgoRR.cc
 * originally based on CSCSegAlgoDF.cc
 * modified by Piet Verwilligen 
 * to use GEMCSCSegFit class
 *
 *  \authors: Raffaella Radogna
 */

#include "GEMCSCSegAlgoRR.h"
#include "GEMCSCSegFit.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>



/**
 *  Constructor
 */
GEMCSCSegAlgoRR::GEMCSCSegAlgoRR(const edm::ParameterSet& ps) : GEMCSCSegmentAlgorithm(ps), myName("GEMCSCSegAlgoRR"), sfit_(0) 
{	 
  debug                     = ps.getUntrackedParameter<bool>("GEMCSCDebug");
  minHitsPerSegment         = ps.getParameter<unsigned int>("minHitsPerSegment");
  preClustering             = ps.getParameter<bool>("preClustering");
  dXclusBoxMax              = ps.getParameter<double>("dXclusBoxMax");
  dYclusBoxMax              = ps.getParameter<double>("dYclusBoxMax");
  preClustering_useChaining = ps.getParameter<bool>("preClusteringUseChaining");
  dPhiChainBoxMax           = ps.getParameter<double>("dPhiChainBoxMax");
  dThetaChainBoxMax         = ps.getParameter<double>("dThetaChainBoxMax");
  dRChainBoxMax             = ps.getParameter<double>("dRChainBoxMax");
  maxRecHitsInCluster       = ps.getParameter<int>("maxRecHitsInCluster");
}



/** 
 * Destructor
 */
GEMCSCSegAlgoRR::~GEMCSCSegAlgoRR() {}



/**
 * Run the algorithm
 */
std::vector<GEMCSCSegment> GEMCSCSegAlgoRR::run( const std::map<uint32_t, const CSCLayer*>& csclayermap, const std::map<uint32_t, const GEMEtaPartition*>& gemrollmap, 
						 const std::vector<const CSCSegment*>& cscsegments, const std::vector<const GEMRecHit*>& gemrechits) 
{

  // This function is called for each CSC Chamber (ME1/1) associated with a GEM chamber in which GEM Rechits were found
  // This means that the csc segment vector can contain more than one segment and that all combinations with the gemrechits have to be tried


  // assign the maps < detid, geometry object > to the class variables
  theCSCLayers_   = csclayermap;
  theGEMEtaParts_ = gemrollmap;


  // LogDebug info about reading of CSC Chamber map and GEM Eta Partition map
  edm::LogVerbatim("GEMCSCSegAlgoRR") << "[GEMCSCSegAlgoRR::run] cached the csclayermap and the gemrollmap";

  // --- LogDebug for CSC Layer map -----------------------------------------
  std::stringstream csclayermapss; csclayermapss<<"[GEMCSCSegAlgoRR::run] :: csclayermap :: elements ["<<std::endl;
  for(std::map<uint32_t, const CSCLayer*>::const_iterator mapIt = theCSCLayers_.begin(); mapIt != theCSCLayers_.end(); ++mapIt) 
    {
      csclayermapss<<"[CSC DetId "<<mapIt->first<<" ="<<CSCDetId(mapIt->first)<<", CSC Layer "<<mapIt->second<<" ="<<(mapIt->second)->id()<<"],"<<std::endl;
    }
  csclayermapss<<"]"<<std::endl; 
  std::string csclayermapstr = csclayermapss.str();
  edm::LogVerbatim("GEMCSCSegAlgoRR") << csclayermapstr;
  // --- End LogDebug -------------------------------------------------------

  // --- LogDebug for GEM Eta Partition map ---------------------------------
  std::stringstream gemetapartmapss; gemetapartmapss<<"[GEMCSCSegAlgoRR::run] :: gemetapartmap :: elements ["<<std::endl;
  for(std::map<uint32_t, const GEMEtaPartition*>::const_iterator mapIt = theGEMEtaParts_.begin(); mapIt != theGEMEtaParts_.end(); ++mapIt) 
    {
      gemetapartmapss<<"[GEM DetId "<<mapIt->first<<" ="<<GEMDetId(mapIt->first)<<", GEM EtaPart "<<mapIt->second<<"],"<<std::endl;
    }
  gemetapartmapss<<"]"<<std::endl; 
  std::string gemetapartmapstr = gemetapartmapss.str();
  edm::LogVerbatim("GEMCSCSegAlgoRR") << gemetapartmapstr;
  // --- End LogDebug -------------------------------------------------------


  std::vector<GEMCSCSegment>                        segmentvectorfinal;
  std::vector<GEMCSCSegment>                        segmentvectorchamber;
  std::vector<GEMCSCSegment>                        segmentvectorfromfit;
  std::vector<const TrackingRecHit*>                chainedRecHits;
  

  // From the GEMCSCSegmentBuilder we get 
  // - a collection of CSC Segments belonging all to the same chamber
  // - a collection of GEM RecHits belonging to GEM Eta-Partitions close to this CSC
  //
  // Now we have to loop over all CSC Segments 
  // and see to which CSC segments we can match the GEM rechits
  // if matching fails we will still keep those segments in the GEMCSC segment collection
  // but we will assign -1 to the matched parameter --> 2B implemented in the DataFormat
  // 1 for matched, 0 for no GEM chamber available and -1 for not matched

  // empty the temporary gemcsc segments vector
  // segmentvectortmp.clear();

  for(std::vector<const CSCSegment*>::const_iterator cscSegIt = cscsegments.begin(); cscSegIt != cscsegments.end(); ++cscSegIt)
    {

      // chain hits :: make a vector of TrackingRecHits
      //               that contains the CSC and GEM rechits
      //               and that can be given to the fitter
      chainedRecHits = this->chainHitsToSegm(*cscSegIt, gemrechits);
  
      // if gemrechits are associated, run the buildSegments step
      if(chainedRecHits.size() > (*cscSegIt)->specificRecHits().size()) 
	{
	  segmentvectorfromfit = this->buildSegments(*cscSegIt, chainedRecHits);
	}
  
      // if no gemrechits are associated, wrap the existing CSCSegment in a GEMCSCSegment class
      else 
	{
	  std::vector<const GEMRecHit*> gemRecHits_noGEMrh; // empty GEMRecHit vector
	  GEMCSCSegment tmp(*cscSegIt, gemRecHits_noGEMrh, (*cscSegIt)->localPosition(), (*cscSegIt)->localDirection(), (*cscSegIt)->parametersError(), (*cscSegIt)->chi2());
	  segmentvectorfromfit.push_back(tmp);
	}

      segmentvectorchamber.insert( segmentvectorchamber.end(), segmentvectorfromfit.begin(), segmentvectorfromfit.end() );
      segmentvectorfromfit.clear();
    }

  // add the found gemcsc segments to the collection of all gemcsc segments and return
  segmentvectorfinal.insert( segmentvectorfinal.end(), segmentvectorchamber.begin(), segmentvectorchamber.end() );  
  segmentvectorchamber.clear();
  edm::LogVerbatim("GEMCSCSegAlgoRR") << "[GEMCSCSegAlgoRR::run] GEMCSC Segments fitted or wrapped, size of vector = "<<segmentvectorfinal.size();
  return segmentvectorfinal;
}


/**
 * Chain hits :: make a TrackingRecHit vector containing both CSC and GEM rechits 
 *               take the hits from the CSCSegment and add a clone of them
 *               take the hits from the GEM RecHit vector and 
 *               check whether they are compatible with the CSC segment extrapolation
 *               to the GEM plane. If they are compatible, add these GEM rechits
 */
std::vector<const TrackingRecHit*> GEMCSCSegAlgoRR::chainHitsToSegm(const CSCSegment* cscsegment, const std::vector<const GEMRecHit*>& gemrechits) 
{

  std::vector<const TrackingRecHit*> chainedRecHits;

  // working with layers makes it here a bit more difficult:
  // the CSC segment points to the chamber, which we cannot ask from the map
  // the CSC rechits point to the layers, from which we can ask the chamber

  auto segLP                   = cscsegment->localPosition();
  auto segLD                   = cscsegment->localDirection();
  auto cscrhs                  = cscsegment->specificRecHits();

  // Loop over the CSC Segment rechits 
  // and save a copy in the chainedRecHits vector
  for (auto crh = cscrhs.begin(); crh!= cscrhs.end(); crh++)
    {
      chainedRecHits.push_back(crh->clone());
    }

  // now ask the layer id of the first CSC rechit
  std::vector<const TrackingRecHit*>::const_iterator trhIt = chainedRecHits.begin();
  // make sure pointer is valid 
  if(trhIt == chainedRecHits.end()) {
    edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::chainHitsToSegm] CSC segment has zero rechits ... end function here";
    return chainedRecHits;
  }
  const CSCLayer * cscLayer = theCSCLayers_.find((*trhIt)->rawId())->second;
  // now ask the chamber id of the first CSC rechit
  if(!cscLayer) {
    edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::chainHitsToSegm] CSC rechit ID was not found back in the CSCLayerMap ... end function here";
    return chainedRecHits;
  }
  const CSCChamber* cscChamber = cscLayer->chamber();

  // For non-empty GEM rechit vector
  if(gemrechits.size()!=0)
    {
      float Dphi_min_l1 = 999;
      float Dphi_min_l2 = 999;
      float Dtheta_min_l1 = 999;
      float Dtheta_min_l2 = 999;
     
      std::vector<const GEMRecHit*>::const_iterator grhIt = gemrechits.begin();

      const GEMRecHit* gemrh_min_l1= *grhIt;
      const GEMRecHit* gemrh_min_l2= *grhIt;
      
      // Loop over GEM rechits from the EnsembleGEMHitContainer
      for(grhIt = gemrechits.begin(); grhIt != gemrechits.end(); ++grhIt) 
	{
	  // get GEM Rechit Local & Global Position
	  auto rhLP = (*grhIt)->localPosition();
	  const GEMEtaPartition* rhRef  = theGEMEtaParts_.find((*grhIt)->gemId())->second;
	  auto rhGP = rhRef->toGlobal(rhLP);
	  // get GEM Rechit Local position w.r.t. the CSC Chamber
	  // --> we are interessed in the z-coordinate
	  auto rhLP_inSegmRef = cscChamber->toLocal(rhGP);
	  // calculate the extrapolation of the CSC segment to the GEM plane (z-coord)
	  // to get x- and y- coordinate 
	  float xe = 0.0, ye = 0.0, ze = 0.0;
	  if(segLD.z() != 0)
	    {
	      xe = segLP.x()+segLD.x()*rhLP_inSegmRef.z()/segLD.z();
	      ye = segLP.y()+segLD.y()*rhLP_inSegmRef.z()/segLD.z();
	      ze = rhLP_inSegmRef.z();
	    }
	  // 3D extrapolated point in the GEM plane
	  LocalPoint extrPoint(xe,ye,ze);
	  
	  // get GEM Rechit Global Position, but obtained from CSC Chamber
	  // --> this should be the same as rhGP = rhRef->toGlobal(rhLP);
	  auto rhGP_fromSegmRef = cscChamber->toGlobal(rhLP_inSegmRef);
	  float phi_rh          = rhGP_fromSegmRef.phi();
	  float theta_rh        = rhGP_fromSegmRef.theta();
	  // get Extrapolat Global Position, also obtained from CSC Chamber
	  auto extrPoinGP_fromSegmRef = cscChamber->toGlobal(extrPoint);
	  float phi_ext               = extrPoinGP_fromSegmRef.phi();
	  float theta_ext             = extrPoinGP_fromSegmRef.theta();
	  
	  // GEM 1st Layer :: Search for GEM Rechit with smallest Delta Eta and Delta Phi
	  //                  and save it inside gemrh_min_l1 
	  if ((*grhIt)->gemId().layer()==1) 
	    {
	      float Dphi_l1 = fabs(phi_ext-phi_rh);
	      float Dtheta_l1 = fabs(theta_ext-theta_rh);
	      if (Dphi_l1 <= Dphi_min_l1) 
		{
		  Dphi_min_l1   = Dphi_l1;
		  Dtheta_min_l1 = Dtheta_l1;
		  gemrh_min_l1  = *grhIt;
		}	  
	    }
	  // GEM 2nd Layer :: Search for GEM Rechit with smallest Delta Eta and Delta Phi
	  //                  and save it inside gemrh_min_l2 
	  if ((*grhIt)->gemId().layer()==2) 
	    {
	      float Dphi_l2 = fabs(phi_ext-phi_rh);
	      float Dtheta_l2 = fabs(theta_ext-theta_rh);
	      if (Dphi_l2 <= Dphi_min_l2)
		{
		  Dphi_min_l2   = Dphi_l2;
		  Dtheta_min_l2 = Dtheta_l2;
		  gemrh_min_l2  = *grhIt;
		}
	    }
	  
	} // end loop over GEM Rechits
      
      // Check whether GEM rechit with smallest delta eta and delta phi 
      // w.r.t. the extrapolation of the CSC segment is within the 
      // maxima given by the configuration of the algorithm
      bool phiRequirementOK_l1 = Dphi_min_l1 < dPhiChainBoxMax;
      bool thetaRequirementOK_l1 = Dtheta_min_l1 < dThetaChainBoxMax;
      if(phiRequirementOK_l1 && thetaRequirementOK_l1 && gemrh_min_l1!=0) 
	{
	  chainedRecHits.push_back(gemrh_min_l1->clone());  
	}
      bool phiRequirementOK_l2 = Dphi_min_l2 < dPhiChainBoxMax;
      bool thetaRequirementOK_l2 = Dtheta_min_l2 < dThetaChainBoxMax;
      if(phiRequirementOK_l2 && thetaRequirementOK_l2 && gemrh_min_l2!=0) 
	{
	  chainedRecHits.push_back(gemrh_min_l2->clone());
	}
    } // End check > 0 GEM rechits

  return chainedRecHits;
}



/** 
 * This algorithm uses a Minimum Spanning Tree (ST) approach to build
 * endcap muon track segments from the rechits in the 6 layers of a CSC
 * and the 2 layers inside a GE1/1 GEM. Fit is implemented in GEMCSCSegFit.cc
 */
std::vector<GEMCSCSegment> GEMCSCSegAlgoRR::buildSegments(const CSCSegment* cscsegment, const std::vector<const TrackingRecHit*>& rechits) 
{

  std::vector<GEMCSCSegment>    gemcscsegmentvector;
  std::vector<const GEMRecHit*> gemrechits;

  // Leave the function if the amount of rechits in the EnsembleHitContainer < min required
  if (rechits.size() < minHitsPerSegment) 
    {
      return gemcscsegmentvector;
    }

  // Extract the GEMRecHits from the TrackingRecHit vector
  for(std::vector<const TrackingRecHit*>::const_iterator trhIt = rechits.begin(); trhIt!=rechits.end(); ++trhIt) 
    {
      if (DetId((*trhIt)->rawId()).subdetId() == MuonSubdetId::GEM) { gemrechits.push_back( ((GEMRecHit*)*trhIt) ); }
    }

  // The actual fit on all hits of the vector of the selected Tracking RecHits:
  delete sfit_;
  sfit_ = new GEMCSCSegFit(theCSCLayers_, theGEMEtaParts_, rechits);
  sfit_->fit();

  // obtain all information necessary to make the segment:
  LocalPoint protoIntercept      = sfit_->intercept();
  LocalVector protoDirection     = sfit_->localdir();
  AlgebraicSymMatrix protoErrors = sfit_->covarianceMatrix(); 
  double protoChi2               = sfit_->chi2();

  edm::LogVerbatim("GEMCSCSegAlgoRR") << "[GEMCSCSegAlgoRR::buildSegments] fit done, will now try to make GEMCSC Segment from CSC Segment in "<<cscsegment->cscDetId()
				      << " made of "<<cscsegment->specificRecHits().size()<<" rechits and with chi2 = "<<cscsegment->chi2()<<" and with "<<gemrechits.size()<<" GEM RecHits";

  // save all information inside GEMCSCSegment
  GEMCSCSegment tmp(cscsegment, gemrechits, protoIntercept, protoDirection, protoErrors, protoChi2);

  edm::LogVerbatim("GEMCSCSegAlgoRR") << "[GEMCSCSegAlgoRR::buildSegments] GEMCSC Segment made";
  gemcscsegmentvector.push_back(tmp);    
  return gemcscsegmentvector;
}
