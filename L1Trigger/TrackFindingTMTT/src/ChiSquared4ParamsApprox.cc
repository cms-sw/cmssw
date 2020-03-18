#include "L1Trigger/TrackFindingTMTT/interface/ChiSquared4ParamsApprox.h"

namespace TMTT {
 
ChiSquared4ParamsApprox::ChiSquared4ParamsApprox(const Settings* settings, const uint nPar) : L1ChiSquared(settings, nPar){
    //parameterStream_ << "4Params_TrackletStyle_MCTruthSeed";
    //configParameters_ = (lsStr_.str());
    largestresid_ = -1.0;
    ilargestresid_  = -1;
 
}
 
std::map<std::string, double> ChiSquared4ParamsApprox::vecToMap(std::vector<double> x){
    // Convert a vector of track parameters to a labelled map for ease of use
    std::map<std::string, double> result;
    result["rInv"] = x[0];
    result["phi0"] = x[1];
    result["t"] = x[2];
    result["z0"] = x[3];
    return result;
}
 
std::vector<double> ChiSquared4ParamsApprox::mapToVec(std::map<std::string, double> x){
    // Conevrt the map of labelled track parameters to a vector (in correct order)
    std::vector<double> result;
    result.resize(4);
    result[0] = x["rInv"];
    result[1] = x["phi0"];
    result[2] = x["t"];
    result[3] = x["z0"];
    return result;
}
 
std::vector<double> ChiSquared4ParamsApprox::seed(const L1track3D& l1track3D){
    /* Cheat by using MC trutth to initialize helix parameters. Useful to check if conevrgence is the problem */
    std::map<std::string, double> x;
    x["rInv"] = getSettings()->invPtToInvR() * l1track3D.qOverPt();
    x["phi0"] = l1track3D.phi0();
    x["z0"] = l1track3D.z0();
    x["t"] = l1track3D.tanLambda();
    return mapToVec(x);
}
 
Matrix<double> ChiSquared4ParamsApprox::D(std::vector<double> x){
    Matrix<double> D(2 * stubs_.size(), nPar_, 0.0); // Empty matrix
    int j = 0;
    std::map<std::string, double> y = vecToMap(x); // Get the track params by label
    double rInv = y["rInv"];
    double phi0 = y["phi0"];
    double t = y["t"];
    for (unsigned i = 0; i < stubs_.size(); i++){
        double ri=stubs_[i]->r();
        if( stubs_[i]->barrel() ){
	  D(j, 0) = -0.5*ri*ri; // Fine for now;
	  D(j, 1) = ri; // Fine
	  //D(j, 2);
	  //D(j, 3);
	  j++;
	  //D(j, 0)
	  //D(j, 1)
	  D(j, 2) = ri; // ri; // Fine for now
	  D(j, 3) = 1; // Fine
	  j++;
        } 
	else {
            //here we handle a disk hit
            //first we have the r position
	  double phii=stubs_[i]->phi();
	  int iphi=stubs_[i]->iphi();
       
	  // N.B. These represent HALF the width and number of strips of sensor.
	  double width = stubs_[i]->width()/2.0;
	  double nstrip = stubs_[i]->nstrip()/2.0;
	  
	  double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...

	  if (stubs_[i]->z()>0.0) Deltai=-Deltai;
	  double DeltaiOverRi = Deltai/ri;
	  double theta0=(DeltaiOverRi)+0.67*(DeltaiOverRi)*(DeltaiOverRi)*(DeltaiOverRi);

	  double phi_track=phi0-0.5*rInv*ri; //Expected phi hit given the track
	  //std::cout << phi_track << "/" << phi0 << "/" << rInv << "/" << t << std::endl;
	  
	  double tInv = 1/t;
	  
	  D(j, 0) = -0.167*ri*ri*ri*rInv; // Tweaking of constant?
	  D(j, 1) = 0; // Exact
	  D(j, 2) = -ri*tInv; // Fine;
	  D(j, 3) = -1*tInv; // Fine
	  j++;
	  //second the rphi position
	  D(j, 0) = -0.5 * ri * ri; // Needs fine tuning, was (phimultiplier*-0.5*(zi-z0)/t+rmultiplier*drdrinv);
	  D(j, 1) = ri; // Fine, originally phimultiplier
	  D(j, 2) = ri*0.5*rInv*ri*tInv - ((phi_track-phii)-theta0)*ri*tInv;
	  D(j, 3) = ri*0.5*rInv*tInv    - ((phi_track-phii)-theta0)*tInv;
	  j++;
        }
    }
    return D;
}
 
Matrix<double> ChiSquared4ParamsApprox::Vinv(){
    Matrix<double> Vinv(2*stubs_.size(), 2*stubs_.size(), 0.0);
    for(unsigned i = 0; i < stubs_.size(); i++){
        if(stubs_[i]->barrel()){
            Vinv(2*i, 2*i) = 1/stubs_[i]->sigmaX();
            Vinv(2*i + 1, 2*i + 1) = 1/stubs_[i]->sigmaZ();
        }else{
            Vinv(2*i, 2*i) = 1/stubs_[i]->sigmaZ();
            Vinv(2*i + 1, 2*i + 1) = 1/stubs_[i]->sigmaX();
        }
 
    }
    return Vinv;
}
 
std::vector<double> ChiSquared4ParamsApprox::residuals(std::vector<double> x) {
 
    unsigned int n=stubs_.size();
   
    std::vector<double> delta;
    delta.resize(2*n);
 
    std::map<std::string, double> trackParams = vecToMap(x); // Get the track params by label
    double rInv = trackParams["rInv"];
    double phi0 = trackParams["phi0"];
    double t = trackParams["t"];
    double z0 = trackParams["z0"];
 
    double chiSq=0.0;
 
    unsigned int j=0;
  
    if (getSettings()->debug()==6) std::cout << "Residuals ("<<chiSq<<") ["<<getSettings()->invPtToInvR()/rInv<<"]: ";
 
    largestresid_ = -1.0;
    ilargestresid_  = -1;
 
   for(unsigned int i=0;i<n;i++) {
        double ri=stubs_[i]->r();
        double zi=stubs_[i]->z();
        double phii=stubs_[i]->phi();
        const double sigmax=stubs_[i]->sigmaX();
        const double sigmaz=stubs_[i]->sigmaZ();
 
        if ( stubs_[i]->barrel() ) {
          //we are dealing with a barrel stub
 
	  double halfRinvRi = 0.5*ri*rInv;
	  double aSinHalfRinvRi = halfRinvRi + 0.67*halfRinvRi*halfRinvRi*halfRinvRi;
	
          double deltaphi=phi0-aSinHalfRinvRi-phii;
          if (deltaphi>M_PI) deltaphi-=2*M_PI;
          if (deltaphi<-M_PI) deltaphi+=2*M_PI;
          delta[j++]=(ri*deltaphi)/sigmax; // TODO this is different from tracklet
          delta[j++]=(z0+(2.0/rInv)*t*aSinHalfRinvRi-zi)/sigmaz;
        }else {
            //we are dealing with a disk hit

	    double tInv = 1/t;
	 
	    double r_track=(zi-z0)*tInv;
            double phi_track=phi0-0.5*rInv*(zi-z0)*tInv;
            int iphi=stubs_[i]->iphi();
 
            // N.B. These represent HALF the width and number of strips of sensor.
            double width = stubs_[i]->width()/2.0;
            double nstrip = stubs_[i]->nstrip()/2.0;
 
            double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...
 
            if (stubs_[i]->z()>0.0) Deltai=-Deltai;

            double DeltaiOverRi = Deltai/ri; 
            double theta0=(DeltaiOverRi)+0.67*(DeltaiOverRi)*(DeltaiOverRi)*(DeltaiOverRi); //+0.125*DeltaiOverRi*DeltaiOverRi*DeltaiOverRi*DeltaiOverRi*DeltaiOverRi;
 
            double Delta=Deltai-r_track*(theta0-(phi_track-phii));
 
            delta[j++]=(r_track-ri)/sigmaz;
            delta[j++]=Delta/sigmax;
        }
 
        if (getSettings()->debug()==6) std::cout << delta[j-2]<<" "<<delta[j-1]<<" ";
 
        chiSq+=delta[j-2]*delta[j-2]+delta[j-1]*delta[j-1];
 
        if (fabs(delta[j-2])>largestresid_) {
          largestresid_=fabs(delta[j-2]);
          ilargestresid_=i;
        }
 
        if (fabs(delta[j-1])>largestresid_) {
          largestresid_=fabs(delta[j-1]);
          ilargestresid_=i;
        }
	if (getSettings()->debug()==6) std::cout << __FILE__ << ":" << __LINE__ << " - Residuals(): delta["<<j-2<<"]/delta["<<j-1<<"]: "<< delta[j-2] << "/" << delta[j-1] << std::endl;
	if (getSettings()->debug()==6) std::cout << __FILE__ << ":" << __LINE__ << " - Residuals(): chisq: " << chiSq  << std::endl;
 
    }
  
    return delta;
 
}
 
std::map<std::string, double> ChiSquared4ParamsApprox::convertParams(std::vector<double> x){
    std::map<std::string, double> y = vecToMap(x); // Get track parameters by label
    std::map<std::string, double> result;
    result["qOverPt"] = y["rInv"] / getSettings()->invPtToInvR();
    result["phi0"] = y["phi0"];
    result["z0"] = y["z0"];
    result["t"] = y["t"];
    return result;
}

}
