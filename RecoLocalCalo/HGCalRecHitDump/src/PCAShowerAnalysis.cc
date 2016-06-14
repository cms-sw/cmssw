#include "RecoLocalCalo/HGCalRecHitDump/interface/PCAShowerAnalysis.h"

//
PCAShowerAnalysis::PCAShowerAnalysis (bool segmented, bool logweighting, bool debug) : 
  principal_(new TPrincipal(3,"D")), 
  debug_(debug),
  logweighting_(logweighting), 
  segmented_(segmented),
  entryz_(320.38)
{
}

//
PCAShowerAnalysis::~PCAShowerAnalysis ()
{
  delete principal_;
}

//
PCAShowerAnalysis::PCASummary_t PCAShowerAnalysis::computeShowerParameters(const reco::CaloCluster &cl,const SlimmedRecHitCollection &recHits)
{  
  //repare the summary
  PCAShowerAnalysis::PCASummary_t summary;

  principal_->Clear();
  double variables[3] = {0.,0.,0.};
  for (unsigned int ih=0;ih<cl.hitsAndFractions().size();++ih) 
    {
      uint32_t id = (cl.hitsAndFractions())[ih].first.rawId();
      SlimmedRecHitCollection::const_iterator theHit=std::find(recHits.begin(),recHits.end(),SlimmedRecHit(id));
      if(theHit==recHits.end()) continue;

      //fill position variables
      variables[0] = theHit->x_; 
      variables[1] = theHit->y_; 
      variables[2] = segmented_ ? theHit->z_ : entryz_;
      if (!logweighting_) 
	{
	  for (int i=0; i<int(theHit->en_); i++) principal_->AddRow(variables); 
	} 
      else {
	// a log-weighting, energy not in fraction of total
	double w0 = -log(20.); // threshold, could use here JB's thresholds
	double scale = 250.; // to scale the weight so to get ~same nbr of points as for E-weight for the highest hit of ~0.1 GeV
	int nhit = int(scale*(w0+log(theHit->en_)));
	if (nhit<0) nhit=1;
	for (int i=0; i<nhit; i++) principal_->AddRow(variables);	
      }	     
    }
  
  principal_->MakePrincipals();
  TMatrixD matrix = *principal_->GetEigenVectors();
  TVectorD eigenvalues = *principal_->GetEigenValues();
  TVectorD sigmas = *principal_->GetSigmas();

  //fill rest of the summary
  summary.center_x = (*principal_->GetMeanValues())[0];
  summary.center_y = (*principal_->GetMeanValues())[1];
  summary.center_z = (*principal_->GetMeanValues())[2];	
  float axis_sign( matrix(2,0)*summary.center_z < 0 ? -1 : 1);
  summary.axis_x   = axis_sign*matrix(0,0);
  summary.axis_y   = axis_sign*matrix(1,0);
  summary.axis_z   = axis_sign*matrix(2,0);  
  summary.ev_1     = eigenvalues(0);
  summary.ev_2     = eigenvalues(1);
  summary.ev_3     = eigenvalues(2);
  summary.sigma_1  = sigmas(0);
  summary.sigma_2  = sigmas(1);
  summary.sigma_3  = sigmas(2);

  if (debug_)
    {
      std::cout << "*** Principal component analysis  ****" << std::endl
		<< "\t average (x,y,z) = " << "(" << summary.center_x << "," << summary.center_y << "," << summary.center_z << std::endl
		<< "\t main axis (x,y,z) = " << "(" << summary.axis_x << "," << summary.axis_y << "," << summary.axis_z << std::endl
		<< "\t eigenvalues = " << "(" << summary.ev_1 << "," << summary.ev_2 << "," << summary.ev_3 << std::endl
		<< "\t sigmas = " << summary.sigma_1 << "," << summary.sigma_2 << "," << summary.sigma_3 << std::endl;
    }

  return summary;
}

