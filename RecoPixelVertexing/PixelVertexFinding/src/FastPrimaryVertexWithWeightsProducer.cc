// -*- C++ -*-
//
// Package:    FastPrimaryVertexWithWeightsProducer
// Class:      FastPrimaryVertexWithWeightsProducer
// 
/**\class FastPrimaryVertexWithWeightsProducer FastPrimaryVertexWithWeightsProducer.cc RecoBTag/FastPrimaryVertexWithWeightsProducer/src/FastPrimaryVertexWithWeightsProducer.cc

 Description: The FastPrimaryVertex is an algorithm used to find the primary vertex of an event @HLT. It takes the pixel clusters compabible with a jet and project it to the beamSpot with the eta-angle of the jet. The peak on the z-projected clusters distribution is our FastPrimaryVertex.


 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Silvio DONATO
//         Created:  Wed Dec 18 10:05:40 CET 2013
//
//

// system include files
#include <memory>
#include <vector>
#include <math.h>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "RecoPixelVertexing/PixelVertexFinding/interface/FindPeakFastPV.h"

#include "FWCore/Utilities/interface/InputTag.h"

using namespace std;

class FastPrimaryVertexWithWeightsProducer : public edm::EDProducer {
   public:
      explicit FastPrimaryVertexWithWeightsProducer(const edm::ParameterSet&);
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);



  double m_maxZ;		// Use only pixel clusters with |z| < maxZ
  edm::InputTag m_clusters;	// PixelClusters InputTag
  std::string m_pixelCPE; 	// PixelCPE (PixelClusterParameterEstimator)
  edm::InputTag m_beamSpot;	// BeamSpot InputTag
  edm::InputTag m_jets;		// Jet InputTag
  edm::EDGetTokenT<SiPixelClusterCollectionNew> clustersToken;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken;
  edm::EDGetTokenT<edm::View<reco::Jet> > jetsToken;

// PARAMETERS USED IN THE BARREL PIXEL CLUSTERS PROJECTION
  int m_njets;			// Use only the first njets
  double m_maxJetEta;		// Use only jets with |eta| < maxJetEta
  double m_minJetPt;		// Use only jets with Pt > minJetPt
  bool m_barrel;		// Use clusters from pixel endcap 
  double m_maxSizeX;		// Use only pixel clusters with sizeX <= maxSizeX
  double m_maxDeltaPhi;		// Use only pixel clusters with DeltaPhi(Jet,Cluster) < maxDeltaPhi
  double m_weight_charge_down;	// Use only pixel clusters with ClusterCharge > weight_charge_down
  double m_weight_charge_up;	// Use only pixel clusters with ClusterCharge < weight_charge_up
  double m_PixelCellHeightOverWidth;//It is the ratio between pixel cell height and width along z coordinate about 285µm/150µm=1.9
  double m_minSizeY_q;		// Use only pixel clusters with sizeY > PixelCellHeightOverWidth * |jetZOverRho| + minSizeY_q
  double m_maxSizeY_q;		// Use only pixel clusters with sizeY < PixelCellHeightOverWidth * |jetZOverRho| + maxSizeY_q
  
// PARAMETERS USED TO WEIGHT THE BARREL PIXEL CLUSTERS   
  // The cluster weight is defined as weight = weight_dPhi * weight_sizeY  * weight_rho * weight_sizeX1 * weight_charge

  double m_weight_dPhi;		// used in weight_dPhi = exp(-|DeltaPhi(JetCluster)|/m_weight_dPhi)    
  double  m_weight_SizeX1;	// used in weight_SizeX1 = (ClusterSizeX==2)*1+(ClusterSizeX==1)*m_weight_SizeX1;    
  double m_weight_rho_up; 	// used in weight_rho = (m_weight_rho_up - ClusterRho)/m_weight_rho_up 
  double m_weight_charge_peak; 	// Give the maximum weight_charge for a cluster with Charge = m_weight_charge_peak
  double m_peakSizeY_q;		// Give the maximum weight_sizeY for a cluster with sizeY = PixelCellHeightOverWidth * |jetZOverRho| + peakSizeY_q

// PARAMETERS USED IN THE ENDCAP PIXEL CLUSTERS PROJECTION
  bool m_endCap;		// Use clusters from pixel endcap 
  double m_minJetEta_EC;	// Use only jets with |eta| > minJetEta_EC
  double m_maxJetEta_EC;	// Use only jets with |eta| < maxJetEta_EC
  double m_maxDeltaPhi_EC;	// Use only pixel clusters with DeltaPhi(Jet,Cluster) < maxDeltaPhi_EC

// PARAMETERS USED TO WEIGHT THE ENDCAP PIXEL CLUSTERS
  double m_EC_weight;		// In EndCap the weight is defined as weight = m_EC_weight*(weight_dPhi) 
  double m_weight_dPhi_EC; 	// Used in weight_dPhi = exp(-|DeltaPhi|/m_weight_dPhi_EC )
   
// PARAMETERS USED TO FIND THE FASTPV AS PEAK IN THE Z-PROJECTIONS DISTRIBUTION
  // First Iteration: look for a cluster with a width = m_zClusterWidth_step1
  double m_zClusterWidth_step1;          // cluster width in step1

  // Second Iteration: use only z-projections with weight > weightCut_step2 and look for a cluster with a width = m_zClusterWidth_step2, within of weightCut_step2 of the previous result 
  double m_zClusterWidth_step2; 	// cluster width in step2
  double m_zClusterSearchArea_step2; 	// cluster width in step2
  double m_weightCut_step2;		// minimum z-projections weight required in step2

  // Third Iteration: use only z-projections with weight > weightCut_step3 and look for a cluster with a width = m_zClusterWidth_step3, within of weightCut_step3 of the previous result 
  double m_zClusterWidth_step3; 	// cluster width in step3
  double m_zClusterSearchArea_step3;	// cluster width in step3
  double m_weightCut_step3; 		// minimum z-projections weight required in step3
  
};

FastPrimaryVertexWithWeightsProducer::FastPrimaryVertexWithWeightsProducer(const edm::ParameterSet& iConfig)
{
  m_maxZ	      		= iConfig.getParameter<double>("maxZ");
  m_clusters          		= iConfig.getParameter<edm::InputTag>("clusters");
  clustersToken                 = consumes<SiPixelClusterCollectionNew>(m_clusters);
  m_pixelCPE          		= iConfig.getParameter<std::string>("pixelCPE");
  m_beamSpot          		= iConfig.getParameter<edm::InputTag>("beamSpot");
  beamSpotToken                 = consumes<reco::BeamSpot>(m_beamSpot);
  m_jets              		= iConfig.getParameter<edm::InputTag>("jets");
  jetsToken                     = consumes<edm::View<reco::Jet> >(m_jets);

  m_njets     			= iConfig.getParameter<int>("njets");
  m_maxJetEta     		= iConfig.getParameter<double>("maxJetEta");
  m_minJetPt     		= iConfig.getParameter<double>("minJetPt");

  m_barrel     			= iConfig.getParameter<bool>("barrel");
  m_maxSizeX	      		= iConfig.getParameter<double>("maxSizeX");
  m_maxDeltaPhi       		= iConfig.getParameter<double>("maxDeltaPhi");
  m_PixelCellHeightOverWidth    = iConfig.getParameter<double>("PixelCellHeightOverWidth");
  m_weight_charge_down     	= iConfig.getParameter<double>("weight_charge_down");
  m_weight_charge_up     	= iConfig.getParameter<double>("weight_charge_up");
  m_minSizeY_q     		= iConfig.getParameter<double>("minSizeY_q");
  m_maxSizeY_q     		= iConfig.getParameter<double>("maxSizeY_q");
  
  m_weight_dPhi     		= iConfig.getParameter<double>("weight_dPhi");
  m_weight_SizeX1      		= iConfig.getParameter<double>("weight_SizeX1");
  m_weight_rho_up      		= iConfig.getParameter<double>("weight_rho_up");
  m_weight_charge_peak     	= iConfig.getParameter<double>("weight_charge_peak");
  m_peakSizeY_q     		= iConfig.getParameter<double>("peakSizeY_q");

  m_endCap     			= iConfig.getParameter<bool>("endCap");
  m_minJetEta_EC     		= iConfig.getParameter<double>("minJetEta_EC");
  m_maxJetEta_EC     		= iConfig.getParameter<double>("maxJetEta_EC");
  m_maxDeltaPhi_EC     		= iConfig.getParameter<double>("maxDeltaPhi_EC");
  m_EC_weight     		= iConfig.getParameter<double>("EC_weight");
  m_weight_dPhi_EC     		= iConfig.getParameter<double>("weight_dPhi_EC");

  m_zClusterWidth_step1      	= iConfig.getParameter<double>("zClusterWidth_step1");

  m_zClusterWidth_step2      	= iConfig.getParameter<double>("zClusterWidth_step2");
  m_zClusterSearchArea_step2    = iConfig.getParameter<double>("zClusterSearchArea_step2");
  m_weightCut_step2      	= iConfig.getParameter<double>("weightCut_step2");

  m_zClusterWidth_step3      	= iConfig.getParameter<double>("zClusterWidth_step3");
  m_zClusterSearchArea_step3    = iConfig.getParameter<double>("zClusterSearchArea_step3");
  m_weightCut_step3      	= iConfig.getParameter<double>("weightCut_step3");

  produces<reco::VertexCollection>();
  produces<float>();

}

void
FastPrimaryVertexWithWeightsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag> ("clusters",edm::InputTag("hltSiPixelClusters"));
  desc.add<edm::InputTag> ("beamSpot",edm::InputTag("hltOnlineBeamSpot"));
  desc.add<edm::InputTag> ("jets",edm::InputTag("hltCaloJetL1FastJetCorrected"));
  desc.add<std::string>("pixelCPE","hltESPPixelCPEGeneric");
  desc.add<double>("maxZ",19.0);
  desc.add<int>("njets",999);
  desc.add<double>("maxJetEta",2.6);
  desc.add<double>("minJetPt",40.);
  desc.add<bool>("barrel",true);
  desc.add<double>("maxSizeX",2.1);
  desc.add<double>("maxDeltaPhi",0.21);
  desc.add<double>("PixelCellHeightOverWidth",1.8);
  desc.add<double>("weight_charge_down",11.*1000.);
  desc.add<double>("weight_charge_up",190.*1000.);
  desc.add<double>("maxSizeY_q",2.0);
  desc.add<double>("minSizeY_q",-0.6);
  desc.add<double>("weight_dPhi",0.13888888);
  desc.add<double>("weight_SizeX1",0.88);
  desc.add<double>("weight_rho_up",22.);
  desc.add<double>("weight_charge_peak",22.*1000.);
  desc.add<double>("peakSizeY_q",1.0);
  desc.add<bool>("endCap",true);
  desc.add<double>("minJetEta_EC",1.3);
  desc.add<double>("maxJetEta_EC",2.6);
  desc.add<double>("maxDeltaPhi_EC",0.14);
  desc.add<double>("EC_weight",0.008);
  desc.add<double>("weight_dPhi_EC",0.064516129);
  desc.add<double>("zClusterWidth_step1",2.0);
  desc.add<double>("zClusterWidth_step2",0.65);
  desc.add<double>("zClusterSearchArea_step2",3.0);
  desc.add<double>("weightCut_step2",0.05);
  desc.add<double>("zClusterWidth_step3",0.3);
  desc.add<double>("zClusterSearchArea_step3",0.55);
  desc.add<double>("weightCut_step3",0.1);
  descriptions.add("fastPrimaryVertexWithWeightsProducer",desc);
}

void
FastPrimaryVertexWithWeightsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   using namespace edm;
   using namespace reco;
   using namespace std;

   const float barrel_lenght=30; //half lenght of the pixel barrel 30 cm

   //get pixel cluster
   Handle<SiPixelClusterCollectionNew> cH;
   iEvent.getByToken(clustersToken,cH);
   const SiPixelClusterCollectionNew & pixelClusters = *cH.product();

   //get jets
   Handle<edm::View<reco::Jet> > jH;
   iEvent.getByToken(jetsToken,jH);
   const edm::View<reco::Jet> & jets = *jH.product();

   vector<const reco::Jet*> selectedJets;
   int countjet=0;   
   for(edm::View<reco::Jet>::const_iterator it = jets.begin() ; it != jets.end() && countjet<m_njets ; it++)
   {
   if( //select jets used in barrel or endcap pixel cluster projections
    ((it->pt() >= m_minJetPt) && fabs(it->eta()) <= m_maxJetEta) || //barrel 
    ((it->pt() >= m_minJetPt) && fabs(it->eta()) <= m_maxJetEta_EC && fabs(it->eta()) >= m_minJetEta_EC) //endcap 
    )
    {
      selectedJets.push_back(&(*it));
      countjet++;
    } 
   }
  
   //get PixelClusterParameterEstimator
   edm::ESHandle<PixelClusterParameterEstimator> pe; 
   const PixelClusterParameterEstimator * pp ;
   iSetup.get<TkPixelCPERecord>().get(m_pixelCPE , pe );  
   pp = pe.product();

   //get beamSpot
   edm::Handle<BeamSpot> beamSpot;
   iEvent.getByToken(beamSpotToken,beamSpot);
 
   //get TrackerGeometry
   edm::ESHandle<TrackerGeometry> tracker;
   iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
   const TrackerGeometry * trackerGeometry = tracker.product();

// PART I: get z-projections with z-weights
   std::vector<float> zProjections;
   std::vector<float> zWeights;
   int jet_count=0;
   for(vector<const reco::Jet*>::iterator jit = selectedJets.begin() ; jit != selectedJets.end() ; jit++)
   {//loop on selected jets
     float px=(*jit)->px();
     float py=(*jit)->py();
     float pz=(*jit)->pz();
     float pt=(*jit)->pt();
     float eta=(*jit)->eta();
     float jetZOverRho = (*jit)->momentum().Z()/(*jit)->momentum().Rho();
     for(SiPixelClusterCollectionNew::const_iterator it = pixelClusters.begin() ; it != pixelClusters.end() ; it++) //Loop on pixel modules with clusters
     {//loop on pixel modules
        DetId id = it->detId();
        const edmNew::DetSet<SiPixelCluster> & detset  = (*it);
        Point3DBase<float, GlobalTag> modulepos=trackerGeometry->idToDet(id)->position();
        float zmodule = modulepos.z() - ((modulepos.x()-beamSpot->x0())*px+(modulepos.y()-beamSpot->y0())*py)/pt * pz/pt; 
        if ((fabs(deltaPhi((*jit)->momentum().Phi(),modulepos.phi()))< m_maxDeltaPhi*2)&&(fabs(zmodule)<(m_maxZ+barrel_lenght))){//if it is a compatible module
        for(size_t j = 0 ; j < detset.size() ; j ++) 
        {//loop on pixel clusters on this module
	  const SiPixelCluster & aCluster =  detset[j];
          if(//it is a cluster to project
		  (// barrel
		  m_barrel &&
		  fabs(modulepos.z())<barrel_lenght &&
		  pt>=m_minJetPt &&
		  jet_count<m_njets &&
		  fabs(eta)<=m_maxJetEta &&
		  aCluster.sizeX() <= m_maxSizeX &&
		  aCluster.sizeY()  >= m_PixelCellHeightOverWidth*fabs(jetZOverRho)+m_minSizeY_q &&
		  aCluster.sizeY() <= m_PixelCellHeightOverWidth*fabs(jetZOverRho)+m_maxSizeY_q
		  )
		  ||
		  (// EC
		  m_endCap &&
		  fabs(modulepos.z())>barrel_lenght &&
		  pt > m_minJetPt &&
		  jet_count<m_njets &&
		  fabs(eta) <= m_maxJetEta_EC &&
		  fabs(eta) >= m_minJetEta_EC &&
		  aCluster.sizeX() <= m_maxSizeX 
		  )	  
          ){
            Point3DBase<float, GlobalTag> v = trackerGeometry->idToDet(id)->surface().toGlobal(pp->localParametersV( aCluster,( *trackerGeometry->idToDetUnit(id)))[0].first) ;
            GlobalPoint v_bs(v.x()-beamSpot->x0(),v.y()-beamSpot->y0(),v.z());
            if(//it pass DeltaPhi(Jet,Cluster) requirements
		  (m_barrel && fabs(modulepos.z())<barrel_lenght && fabs(deltaPhi((*jit)->momentum().Phi(),v_bs.phi())) <= m_maxDeltaPhi ) || //barrel
		  (m_endCap && fabs(modulepos.z())>barrel_lenght && fabs(deltaPhi((*jit)->momentum().Phi(),v_bs.phi())) <= m_maxDeltaPhi_EC )  //EC
            )
            {
              float z = v.z() - ((v.x()-beamSpot->x0())*px+(v.y()-beamSpot->y0())*py)/pt * pz/pt;   //calculate z-projection
              if(fabs(z) < m_maxZ)
              {
	        zProjections.push_back(z); //add z-projection in zProjections
		float weight=0;
		//calculate zWeight
	        if(fabs(modulepos.z())<barrel_lenght)
	        { //barrel
	        	//calculate weight_sizeY
			float sizeY=aCluster.sizeY();
			float sizeY_up =  m_PixelCellHeightOverWidth*fabs(jetZOverRho)+m_maxSizeY_q;
			float sizeY_peak = m_PixelCellHeightOverWidth*fabs(jetZOverRho)+m_peakSizeY_q; 
			float sizeY_down = m_PixelCellHeightOverWidth*fabs(jetZOverRho)+m_minSizeY_q;
			float weight_sizeY_up = (sizeY_up-sizeY)/(sizeY_up-sizeY_peak);
			float weight_sizeY_down = (sizeY-sizeY_down)/(sizeY_peak-sizeY_down);
			weight_sizeY_down = weight_sizeY_down *(weight_sizeY_down>0)*(weight_sizeY_down<1);
			weight_sizeY_up = weight_sizeY_up *(weight_sizeY_up>0)*(weight_sizeY_up<1);
			float weight_sizeY = weight_sizeY_up + weight_sizeY_down;			    	

	        	//calculate weight_rho
		    	float rho = sqrt(v_bs.x()*v_bs.x() +  v_bs.y()*v_bs.y());
			float weight_rho = ((m_weight_rho_up - rho)/m_weight_rho_up);
		    	
	        	//calculate weight_dPhi
		    	float weight_dPhi = exp(-  fabs(deltaPhi((*jit)->momentum().Phi(),v_bs.phi()))/m_weight_dPhi );
		    	
	        	//calculate weight_sizeX1
			float weight_sizeX1= (aCluster.sizeX()==2) + (aCluster.sizeX()==1)*m_weight_SizeX1;

	        	//calculate weight_charge
			float charge=aCluster.charge();
			float weightCluster_up = (m_weight_charge_up-charge)/(m_weight_charge_up-m_weight_charge_peak);
			float weightCluster_down = (charge-m_weight_charge_down)/(m_weight_charge_peak-m_weight_charge_down);
			weightCluster_down = weightCluster_down *(weightCluster_down>0)*(weightCluster_down<1);
			weightCluster_up = weightCluster_up *(weightCluster_up>0)*(weightCluster_up<1);
			float weight_charge = weightCluster_up +  weightCluster_down;
		
	        	//calculate the final weight
			weight = weight_dPhi * weight_sizeY  * weight_rho * weight_sizeX1 * weight_charge ;
	        }
	        else if(fabs(modulepos.z())>barrel_lenght) // EC
	        {// EC
	        	//calculate weight_dPhi
		    	float weight_dPhi = exp(-  fabs(deltaPhi((*jit)->momentum().Phi(),v_bs.phi())) /m_weight_dPhi_EC );
	        	//calculate the final weight
			weight=	 m_EC_weight*(weight_dPhi) ;    	        
	        }
	        zWeights.push_back(weight); //add the weight to zWeights
	      }	
  	    }//if it pass DeltaPhi(Jet,Cluster) requirements
          }///if it is a cluster to project 
	}//loop on pixel clusters on this module
      }//if it is a compatible module
    }//loop on pixel modules
   jet_count++;
   }//loop on selected jets
   
  //order zProjections and zWeights by z
  std::multimap<float,float>  zWithW;
  size_t i=0;
  for(i=0;i<zProjections.size();i++) zWithW.insert(std::pair<float,float>(zProjections[i],zWeights[i])); 
  i=0;
  for(std::multimap<float,float>::iterator it=zWithW.begin(); it!=zWithW.end(); it++,i++) { zProjections[i]=it->first; zWeights[i]=it->second; } //order zProjections and zWeights by z
   

  //calculate zWeightsSquared
  std::vector<float> zWeightsSquared;
  for(std::vector<float>::iterator it=zWeights.begin();it!=zWeights.end();it++) {zWeightsSquared.push_back((*it)*(*it));}

  //do multi-step peak searching
  float res_step1 = FindPeakFastPV(  zProjections, zWeights, 0.0, m_zClusterWidth_step1, 999.0, -1.0);    
  float res_step2 = FindPeakFastPV(  zProjections, zWeights, res_step1, m_zClusterWidth_step2, m_zClusterSearchArea_step2, m_weightCut_step2);  
  float res_step3 = FindPeakFastPV(  zProjections, zWeightsSquared, res_step2, m_zClusterWidth_step3, m_zClusterSearchArea_step3, m_weightCut_step3*m_weightCut_step3);  

  float centerWMax=res_step3;
  //End of PART II
  
  //Make the output
  float res=0; 
  if(zProjections.size() > 2) 
  {
     res=centerWMax;
     Vertex::Error e; 
     e(0, 0) = 0.0015 * 0.0015;
     e(1, 1) = 0.0015 * 0.0015;
     e(2, 2) = 1.5 * 1.5;
     Vertex::Point p(beamSpot->x(res), beamSpot->y(res), res);
     Vertex thePV(p, e, 1, 1, 0);
     std::auto_ptr<reco::VertexCollection> pOut(new reco::VertexCollection());
     pOut->push_back(thePV);
     iEvent.put(pOut);
   } else
   {
     Vertex::Error e;
     e(0, 0) = 0.0015 * 0.0015;
     e(1, 1) = 0.0015 * 0.0015;
     e(2, 2) = 1.5 * 1.5;
     Vertex::Point p(beamSpot->x(res), beamSpot->y(res), res);
     Vertex thePV(p, e, 0, 0, 0);
     std::auto_ptr<reco::VertexCollection> pOut(new reco::VertexCollection());
     pOut->push_back(thePV);
     iEvent.put(pOut);
   }

//Finally, calculate the zClusterQuality as Sum(weights near the fastPV)/sqrt(Sum(total weights)) [a kind of zCluster significance]

 

  const float half_width_peak=1;
  float nWeightedTot=0;
  float nWeightedTotPeak=0;
  for(std::vector<float>::iterator it = zProjections.begin();it!=zProjections.end(); it++)
  {
  	nWeightedTot+=zWeights[it-zProjections.begin()]; 
  	if((res-half_width_peak)<=(*it) && (*it)<=(res+half_width_peak))
	{
	  	nWeightedTotPeak+=zWeights[it-zProjections.begin()]; 	
	}
  }

std::auto_ptr<float > zClusterQuality(new float());
*zClusterQuality=-1;
if(nWeightedTot!=0)
{
   *zClusterQuality=nWeightedTotPeak / sqrt(nWeightedTot/(2*half_width_peak)); // where 30 is the beam spot lenght 
   iEvent.put(zClusterQuality);
}
else
   iEvent.put(zClusterQuality);

}


//define this as a plug-in
DEFINE_FWK_MODULE(FastPrimaryVertexWithWeightsProducer);
