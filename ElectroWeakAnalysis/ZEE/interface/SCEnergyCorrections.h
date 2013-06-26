#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

reco::CaloClusterPtrVector CaloClusterVectorCopier(const reco::SuperCluster& sc);

reco::SuperCluster fEtaScCorr(const reco::SuperCluster& sc)
{
	reco::CaloClusterPtrVector bcs = CaloClusterVectorCopier(sc);		
	double ieta = fabs(sc.eta())*(5/0.087);
	double p0 = 40.2198;
	double p1 = -3.03103e-6;
	double newE;
//                      std::cout << "Corrected E = Raw E * (1+ p1*(ieta - p0)*(ieta - p0))"<< std::endl;
	if ( ieta < p0 ) newE = sc.rawEnergy();
	else newE = sc.rawEnergy()/(1 + p1*(ieta - p0)*(ieta - p0));

	reco::SuperCluster corrSc(newE, sc.position(), sc.seed(), bcs, sc.preshowerEnergy(), 0., 0.);
	return corrSc;
}

reco::SuperCluster fBremScCorr(const reco::SuperCluster& sc, const edm::ParameterSet& ps)
{
	std::vector<double> fBrem = ps.getParameter<std::vector<double> >("fBremVec");
	double bremFrLowThr = ps.getParameter<double>("brLinearLowThr");
	double bremFrHighThr = ps.getParameter<double>("brLinearHighThr");
	
	reco::CaloClusterPtrVector bcs = CaloClusterVectorCopier(sc);		
	double bremFrac = sc.phiWidth()/sc.etaWidth();	
	double newE = sc.energy();
	if(fabs(sc.eta()) < 1.479)
	{
		reco::SuperCluster fEtaSC = fEtaScCorr(sc);
		reco::CaloClusterPtrVector bcs = CaloClusterVectorCopier(sc);		
		newE = fEtaSC.energy();
	}

	if(bremFrac < bremFrLowThr)  bremFrac = bremFrLowThr;
	if(bremFrac < bremFrHighThr)  bremFrac = bremFrHighThr;
	
	double p0 = fBrem[0]; 
	double p1 = fBrem[1]; 
	double p2 = fBrem[2]; 
	double p3 = fBrem[3]; 
	double p4 = fBrem[4]; 
	//
	double threshold = p4; 
	  
	double y = p0*threshold*threshold + p1*threshold + p2;
	double yprime = 2*p0*threshold + p1;
	double a = p3;
	double b = yprime - 2*a*threshold;
	double c = y - a*threshold*threshold - b*threshold;
	 
	double fCorr = 1;
	if( bremFrac < threshold ) 
	    fCorr = p0*bremFrac*bremFrac + p1*bremFrac + p2;
	else 
	    fCorr = a*bremFrac*bremFrac + b*bremFrac + c;

	newE /= fCorr;
	reco::SuperCluster corrSc(newE, sc.position(), sc.seed(), bcs, sc.preshowerEnergy(), 0., 0.);
	return corrSc;
}

reco::SuperCluster fEtEtaCorr(const reco::SuperCluster& sc, const edm::ParameterSet& ps)
{
  // et -- Et of the SuperCluster (with respect to (0,0,0))
  // eta -- eta of the SuperCluster
  	std::vector<double> fEtEtaParams = ps.getParameter<std::vector<double> >("fEtEtaParamsVec");

	reco::SuperCluster fBremSC = fBremScCorr(sc, ps);
	reco::CaloClusterPtrVector bcs = CaloClusterVectorCopier(sc);		

	double eta = sc.eta();
	double et = fBremSC.energy()/cosh(eta);
	double fCorr = 0.;

	double p0 = fEtEtaParams[0] + fEtEtaParams[1]/(et + fEtEtaParams[ 2]) + fEtEtaParams[ 3]/(et*et);
	double p1 = fEtEtaParams[4] + fEtEtaParams[5]/(et + fEtEtaParams[ 6]) + fEtEtaParams[ 7]/(et*et);
	double p2 = fEtEtaParams[8] + fEtEtaParams[9]/(et + fEtEtaParams[10]) + fEtEtaParams[11]/(et*et);

	fCorr = 
	p0 + 
	p1 * atan(fEtEtaParams[12]*(fEtEtaParams[13]-fabs(eta))) + fEtEtaParams[14] * fabs(eta) + 
	p1 * fEtEtaParams[15] * fabs(eta) +
	p2 * fEtEtaParams[16] * eta * eta; 

	if ( fCorr < 0.5 ) fCorr = 0.5;

	double newE = et/(fCorr*cosh(eta));
	reco::SuperCluster corrSc(newE, sc.position(), sc.seed(), bcs, sc.preshowerEnergy(), 0., 0.);
	return corrSc;
}

reco::SuperCluster fEAddScCorr(const reco::SuperCluster& sc, double Ecorr)
{
	reco::CaloClusterPtrVector bcs = CaloClusterVectorCopier(sc);		
	
	double newE = sc.rawEnergy()+Ecorr; 
	reco::SuperCluster corrSc(newE, sc.position(), sc.seed(), bcs, sc.preshowerEnergy(), 0., 0.);
	return corrSc;
}
	

reco::CaloClusterPtrVector CaloClusterVectorCopier(const reco::SuperCluster& sc)
{
  	reco::CaloClusterPtrVector clusters_v;

  	for(reco::CaloCluster_iterator cluster = sc.clustersBegin(); cluster != sc.clustersEnd(); cluster ++)
 	{
		clusters_v.push_back(*cluster);
	}

	return clusters_v;
}
