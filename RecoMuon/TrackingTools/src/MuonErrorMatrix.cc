#include "RecoMuon/TrackingTools/interface/MuonErrorMatrix.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TROOT.h"
#include "TString.h"
#include "TRandom2.h"
#include "TMath.h"

#include <sstream>
#include <atomic>

using namespace std;

const TString MuonErrorMatrix::vars[5]={"#frac{q}{|p|}","#lambda","#varphi_{0}","X_{T}","Y_{T}"};

MuonErrorMatrix::MuonErrorMatrix(const edm::ParameterSet & iConfig):theD(nullptr){
  theCategory="MuonErrorMatrix";
  std::string action = iConfig.getParameter<std::string>("action");

  bool madeFromCff=iConfig.exists("errorMatrixValuesPSet");
  edm::ParameterSet errorMatrixValuesPSet;

  std::string fileName;
  if (!madeFromCff){ fileName = iConfig.getParameter<std::string>("rootFileName");}
  else { errorMatrixValuesPSet =iConfig.getParameter<edm::ParameterSet>("errorMatrixValuesPSet");}
  MuonErrorMatrix::action a= use;

   int NPt=5;
   std::vector<double> xBins;
   double * xBinsArray = nullptr;
   double minPt=1;
   double maxPt=200;
   int NEta=5;
   std::vector<double> yBins;
   double * yBinsArray =nullptr;
   double minEta=0;
   double maxEta=2.5;
   int NPhi=1;
   double minPhi=-TMath::Pi();
   double maxPhi=TMath::Pi();

  if (action!="use"){
    a = constructor;
    
    NPt = iConfig.getParameter<int>("NPt");
    if (NPt!=0){
      minPt = iConfig.getParameter<double>("minPt");
      maxPt = iConfig.getParameter<double>("maxPt");}
    else{
      xBins = iConfig.getParameter<std::vector<double> >("PtBins");
      if (xBins.empty()){edm::LogError( theCategory)<<"Npt=0 and no entries in the vector. I will do aseg fault soon.";}
      NPt = xBins.size()-1;
      xBinsArray = &(xBins.front());
      minPt = xBins.front();
      maxPt = xBins.back();}

    NEta = iConfig.getParameter<int>("NEta");
    if (NEta!=0){
      minEta = iConfig.getParameter<double>("minEta");
      maxEta = iConfig.getParameter<double>("maxEta");}
    else{
      yBins = iConfig.getParameter<std::vector<double> >("EtaBins");
      if (yBins.empty()){edm::LogError( theCategory)<<"NEta=0 and no entries in the vector. I will do aseg fault soon.";}
      NEta = yBins.size()-1;
      yBinsArray = &(yBins.front());
      minEta = yBins.front();
      maxEta = yBins.back();}

    NPhi = iConfig.getParameter<int>("NPhi");
    std::stringstream get(iConfig.getParameter<std::string>("minPhi"));
    if (get.str() =="-Pi")
      {	minPhi =-TMath::Pi();}
    else if(get.str() =="Pi")
      { minPhi =TMath::Pi();}
    else { get>>minPhi;}
    get.str(iConfig.getParameter<std::string>("maxPhi"));
        if (get.str() =="-Pi")
      {	maxPhi =-TMath::Pi();}
    else if(get.str() =="Pi")
      { maxPhi =TMath::Pi();}
    else { get>>maxPhi;}
  }//action!=use
  else if (madeFromCff){

    xBins = errorMatrixValuesPSet.getParameter<std::vector<double> >("xAxis");
    NPt = xBins.size()-1;
    xBinsArray = &(xBins.front());
    minPt = xBins.front();
    maxPt = xBins.back();
    
    yBins = errorMatrixValuesPSet.getParameter<std::vector<double> >("yAxis");
    NEta = yBins.size()-1;
    yBinsArray = &(yBins.front());
    minEta = yBins.front();
    maxEta = yBins.back();

    std::vector<double> zBins = errorMatrixValuesPSet.getParameter<std::vector<double> >("zAxis");
    NPhi=1;
    if (zBins.size()!=2){
      edm::LogError(theCategory)<<"please specify a zAxis with 2 entries only. more bins not implemented yet.";}
    minPhi=zBins.front();
    maxPhi=zBins.back();

  }

  if (a==use){
    if (!madeFromCff){
      edm::LogInfo(theCategory)<<"using an error matrix object from: "<<fileName;
      edm::FileInPath data(fileName);
      std::string fullpath = data.fullPath();
      gROOT->cd();
      theD = new TFile(fullpath.c_str());
      theD->SetWritable(false);
    }else{
      static std::atomic<unsigned int> neverTheSame{0};
      std::stringstream dirName("MuonErrorMatrixDirectory");
      dirName<<neverTheSame++;
      edm::LogInfo(theCategory)<<"using an error matrix object from configuration file. putting memory histograms to: "<<dirName.str();
      gROOT->cd();
      theD = new TDirectory(dirName.str().c_str(),"transient directory to host MuonErrorMatrix TProfile3D");
      theD->SetWritable(false);
    }
  }
  else{
    edm::LogInfo(theCategory)<<"creating  an error matrix object: "<<fileName;
    theD = new TFile(fileName.c_str(),"recreate");
  }

  if (a==use && !madeFromCff ){gROOT->cd();}
  else {theD->cd();}

  for (int i=0;i!=5;i++){for (int j=i;j!=5;j++){
      TString pfname(Form("pf3_V%1d%1d",i+1,j+1));
      TProfile3D * pf =nullptr;
      if (a==use && !madeFromCff ){
	//read from the rootfile
	edm::LogVerbatim(theCategory)<<"getting "<<pfname<<" from "<<fileName;
	pf = (TProfile3D *)theD->Get(pfname);
	theData[Pindex(i,j)]=pf;
	theData_fast[i][j]=pf;	  theData_fast[j][i]=pf;
      }
      else{
	//	  curvilinear coordinate system
	//need to make some input parameter to be to change the number of bins

	TString pftitle;
	if (i==j){pftitle="#sigma_{"+vars[i]+"}";}
	else{pftitle="#rho("+vars[i]+","+vars[j]+")";}
	edm::LogVerbatim(theCategory)<<"booking "<<pfname<<" into "<<fileName;
	pf = new TProfile3D(pfname,pftitle,NPt,minPt,maxPt,NEta,minEta,maxEta,NPhi,minPhi,maxPhi,"S");	    
	pf->SetXTitle("muon p_{T} [GeV]");
	pf->SetYTitle("muon |#eta|");
	pf->SetZTitle("muon #varphi");

	//set variable size binning
	if (xBinsArray){
	  pf->GetXaxis()->Set(NPt,xBinsArray);}
	if (yBinsArray){
	  pf->GetYaxis()->Set(NEta,yBinsArray);}

	if (madeFromCff){
	  edm::ParameterSet pSet = errorMatrixValuesPSet.getParameter<edm::ParameterSet>(pfname.Data());
	  //set the values from the configuration file itself
	  std::vector<double> values = pSet.getParameter<std::vector<double> >("values");
	  unsigned int iX=pf->GetNbinsX();
	  unsigned int iY=pf->GetNbinsY();
	  unsigned int iZ=pf->GetNbinsZ();
	  unsigned int continuous_i=0;
	  for(unsigned int ix=1;ix<=iX;++ix){
	    for(unsigned int iy=1;iy<=iY;++iy){
	      for(unsigned int iz=1;iz<=iZ;++iz){
		LogTrace(theCategory)<<"filling profile:"
				     <<"\n pt (x)= "<<pf->GetXaxis()->GetBinCenter(ix)
				     <<"\n eta (y)= "<<pf->GetYaxis()->GetBinCenter(iy)
				     <<"\n phi (z)= "<<pf->GetZaxis()->GetBinCenter(iz)
				     <<"\n value= "<<values[continuous_i];
		pf->Fill(pf->GetXaxis()->GetBinCenter(ix),
			 pf->GetYaxis()->GetBinCenter(iy),
			 pf->GetZaxis()->GetBinCenter(iz),
			 values[continuous_i++]);
	      }}}
	  //term action
	  std::string tAction = pSet.getParameter<std::string>("action");
	  if (tAction == "scale")
	    theTermAction[Pindex(i,j)] = scale;
	  else if (tAction == "assign")
	    theTermAction[Pindex(i,j)] = assign;
	  else{
	    edm::LogError(theCategory)<<" wrong configuration: term action: "<<tAction<<" is not recognized.";
	    theTermAction[Pindex(i,j)] = error;
	  }
	  
	}
      }

      LogDebug(theCategory)<<" index "<<i<<":"<<j<<" -> "<<Pindex(i,j);
      theData[Pindex(i,j)]=pf;
      theData_fast[i][j]=pf;	  theData_fast[j][i]=pf;
      if (!pf){
	edm::LogError(theCategory)<<" profile "<<pfname<<" in file "<<fileName<<" is not valid. exiting.";
	exit(1);
      }
    }}
    
  //verify it
  for (int i=0;i!=15;i++){ 
    if (theData[i]) {edm::LogVerbatim(theCategory)<<i<<" :"<<theData[i]->GetName()
						  <<" "<< theData[i]->GetTitle()<<std::endl;}}
 
}

//void MuonErrorMatrix::writeIntoCFF(){}

void MuonErrorMatrix::close(){
  //close the file
  if (theD){
    theD->cd();
    //write to it first if constructor
    if (theD->IsWritable()){
      for (int i=0;i!=15;i++){ if (theData[i]) { theData[i]->Write();}}}
    theD->Close();
  }}

MuonErrorMatrix::~MuonErrorMatrix()  {
  close();
}



CurvilinearTrajectoryError MuonErrorMatrix::get(GlobalVector momentum,bool convolute)  {
  AlgebraicSymMatrix55 V;
  //retrieves a 55 matrix containing (i,i)^2 and (i,j)*(i,i)*(j,j)
  for (int i=0;i!=5;i++){for (int j=i;j!=5;j++){
      V(i,j) = Value(momentum,i,j,convolute);}}
  return CurvilinearTrajectoryError(V);}

CurvilinearTrajectoryError MuonErrorMatrix::getFast(GlobalVector momentum) {
  //will be faster but make assumptions that could be broken at some point
  //  same bining for all TProfile
  AlgebraicSymMatrix55 V;

  double pT = momentum.perp();
  double eta = fabs(momentum.eta());
  double phi = momentum.phi();

  //assume all the same axis in X,Y,Z
  int iBin_x= findBin(theData_fast[0][0]->GetXaxis(),pT);
  int iBin_y= findBin(theData_fast[0][0]->GetYaxis(),eta);
  int iBin_z= findBin(theData_fast[0][0]->GetZaxis(),phi);

  //retreive values
  double values[5][5]; //sigma_i and rho_ij
  for (int i=0;i!=5;i++){for (int j=i;j!=5;j++){
      values[i][j]=theData_fast[i][j]->GetBinContent(iBin_x, iBin_y, iBin_z);
    }}

  for (int i=0;i!=5;i++){for (int j=i;j!=5;j++){
      if (i==j){
        //sigma_i * sigma_i
        V(i,j) = values[i][j];
        V(i,j)*=V(i,j);
      }
      else{
        //sigma_i * sigma_j * rho_ij
        V(i,j) = values[i][i] * values[j][j] * values[i][j];
      }
    }}

  return CurvilinearTrajectoryError(V);}



/*CurvilinearTrajectoryError MuonErrorMatrix::get_random(GlobalVector momentum)  { 
  static TRandom2 rand;
  AlgebraicSymMatrix55 V;//result
  //first proceed with diagonal elements
  for (int i=0;i!=5;i++){
  V(i,i)=rand.Gaus( Value(momentum,i,i), Rms(momentum,i,i));}

  //now proceed with the correlations
  for (int i=0;i!=5;i++){for (int j=i+1;j<5;j++){
      double corr = rand.Gaus( Value(momentum,i,j), Rms(momentum,i,j));
      //assign the covariance from correlation and sigmas
      V(i,j)= corr * sqrt( V[i][i] * V[j][j]);}}
      return CurvilinearTrajectoryError(V);  }
*/

int MuonErrorMatrix::findBin(TAxis * axis, double value){
  //find the proper bin, protecting against under/over flow
  int result = axis->FindBin(value);
  if (result <= 0) result=1; //protect against under flow
  else if (result > axis->GetNbins() ) result = axis->GetNbins();
  return result;}


double MuonErrorMatrix::Value(GlobalVector & momentum, int i, int j,bool convolute)  {
  double result=0;
  TProfile3D * ij = Index(i,j);
  if (!ij) {edm::LogError(theCategory)<<"cannot get the profile ("<<i<<":"<<j<<")"; return result;}
  
  double pT = momentum.perp();
  double eta = fabs(momentum.eta());
  double phi = momentum.phi();

  int iBin_x= findBin(ij->GetXaxis(),pT);
  int iBin_y= findBin(ij->GetYaxis(),eta);
  int iBin_z= findBin(ij->GetZaxis(),phi);

  if (convolute){
  if (i!=j){
    //return the covariance = correlation*sigma_1 *sigma_2;
    TProfile3D * ii = Index(i,i);
    TProfile3D * jj = Index(j,j);
    if (!ii){edm::LogError(theCategory)<<"cannot get the profile ("<<i<<":"<<i<<")"; return result;}
    if (!jj){edm::LogError(theCategory)<<"cannot get the profile ("<<j<<":"<<j<<")"; return result;}


    int iBin_i_x = findBin(ii->GetXaxis(),pT);
    int iBin_i_y = findBin(ii->GetYaxis(),eta);
    int iBin_i_z = findBin(ii->GetZaxis(),phi);
    int iBin_j_x = findBin(jj->GetXaxis(),pT);
    int iBin_j_y = findBin(jj->GetYaxis(),eta);
    int iBin_j_z = findBin(jj->GetZaxis(),phi);

    double corr = ij->GetBinContent(iBin_x,iBin_y,iBin_z);
    double sigma_1 = (ii->GetBinContent(iBin_i_x,iBin_i_y,iBin_i_z));
    double sigma_2 = (jj->GetBinContent(iBin_j_x,iBin_j_y,iBin_j_z));

    
    result=corr*sigma_1*sigma_2;
    LogDebug(theCategory)<<"for: (pT,eta,phi)=("<<pT<<", "<<eta<<", "<<phi<<") nterms are"
			 <<"\nrho["<<i<<","<<j<<"]: "<<corr<<" ["<< iBin_x<<", "<<iBin_y<<", "<<iBin_z<<"]"
			 <<"\nsigma["<<i<<","<<i<<"]: "<< sigma_1
			 <<"\nsigma["<<j<<","<<j<<"]: "<< sigma_2
			 <<"Covariance["<<i<<","<<j<<"] is: "<<result;
    return result;
  }
  else{
    //return the variance = sigma_1 **2
    //    result=ij->GetBinContent(iBin);
    result=ij->GetBinContent(iBin_x,iBin_y,iBin_z);
    result*=result;
    LogDebug(theCategory)<<"for: (pT,eta,phi)=("<<pT<<", "<<eta<<", "<<phi<<") sigma^2["<<i<<","<<j<<"] is: "<<result;
    return result;
  }
  }else{
    //do not convolute
    result=ij->GetBinContent(iBin_x,iBin_y,iBin_z);
    return result;
  }
}

double MuonErrorMatrix::Rms(GlobalVector & momentum, int i, int j)  {
  double result=0;
  TProfile3D * ij = Index(i,j);
  if (!ij){edm::LogError(theCategory)<<"cannot get the profile ("<<i<<":"<<i<<")"; return result;} 
  double pT = momentum.perp();
  double eta = fabs(momentum.eta());
  double phi = momentum.phi();
  
  int iBin_x= ij->GetXaxis()->FindBin(pT);
  int iBin_y= ij->GetYaxis()->FindBin(eta);
  int iBin_z= ij->GetZaxis()->FindBin(phi);
  result=ij->GetBinError(iBin_x,iBin_y,iBin_z);

  LogDebug(theCategory)<<"for: (pT,eta,phi)=("<<pT<<", "<<eta<<", "<<phi<<") error["<<i<<","<<j<<"] is: "<<result;
  return result;
}

double MuonErrorMatrix::Term(const AlgebraicSymMatrix55 & curv, int i, int j){
  //return sigma or correlation factor
  double result=0;
  if (i==j){
    result = curv(i,j);
    if (result<0){
      //check validity of this guy
      edm::LogError("MuonErrorMatrix")<<"invalid term in the error matrix.\n sii: "<< result;
      return 0;}
    return sqrt(result);}
  else{
    double si = curv(i,i);
    double sj = curv(j,j);

    if (si<=0 || sj<=0){
      edm::LogError("MuonErrorMatrix")<<"invalid term in the error matrix.\n si: "
				<<si<<" sj: "<<sj<<". result will be corrupted\n"<<curv;
      return 0;}

    si=sqrt(si);
    sj=sqrt(sj);
    //check validity

    return result = curv(i,j)/(si*sj);}
  //by default
  return 0;
}


void MuonErrorMatrix::multiply(CurvilinearTrajectoryError & initial_error, const CurvilinearTrajectoryError & scale_error){
  //scale term by term the matrix
  const AlgebraicSymMatrix55 & scale_matrix=scale_error.matrix();
  AlgebraicSymMatrix55 revised_matrix = initial_error.matrix();
  // the true type of the matrix is such that [i][j] is the same memory object as [j][i]: looping i:0-5, j:0-5 double multiply the terms
  // need to loop only on i:0-5, j:i-5
  for(int i = 0;i!=5;i++){for(int j = i;j!=5;j++){
      revised_matrix(i,j)*=scale_matrix(i,j);
    }}
  initial_error = CurvilinearTrajectoryError(revised_matrix);
}
bool MuonErrorMatrix::divide(CurvilinearTrajectoryError & num_error, const CurvilinearTrajectoryError & denom_error){
  //divide term by term the matrix
  const AlgebraicSymMatrix55 & denom_matrix=denom_error.matrix();
  AlgebraicSymMatrix55 num_matrix = num_error.matrix();
  // the true type of the matrix is such that [i][j] is the same memory object as [j][i]: looping i:0-5, j:0-5 double multiply the terms
  // need to loop only on i:0-5, j:i-5
  for(int i = 0;i!=5;i++){for(int j = i;j!=5;j++){
      if (denom_matrix(i,j)==0) return false;
      num_matrix(i,j)/=denom_matrix(i,j);
    }}
  num_error = CurvilinearTrajectoryError(num_matrix);
  return true;
}

void MuonErrorMatrix::simpleTerm(const AlgebraicSymMatrix55 & input, AlgebraicSymMatrix55 & output){
  for(int i = 0;i!=5;i++){for(int j = i;j!=5;j++){
      if (i==j)
	output(i,j) = sqrt(input(i,j)); //sigma
      else
	output(i,j) = input(i,j) / sqrt( input(i,i)* input(j,j)); //rho
    }}
}

void MuonErrorMatrix::complicatedTerm(const AlgebraicSymMatrix55 & input, AlgebraicSymMatrix55 & output){

  for(int i = 0;i!=5;i++){for(int j = i;j!=5;j++){
      if (i==j)
	output(i,j) = input(i,j)*input(i,j); //sigma squared
      else
	output(i,j) = input(i,j)*input(i,i)*input(j,j); //rho*sigma*sigma
    }}
}



void MuonErrorMatrix::adjust(FreeTrajectoryState & state){

  LogDebug(theCategory+"|Adjust")<<"state: \n"<<state;
  AlgebraicSymMatrix55 simpleTerms;
  simpleTerm(state.curvilinearError(), simpleTerms);
  //the above contains sigma(i), rho(i,j)
  LogDebug(theCategory+"|Adjust")<<"state sigma(i), rho(i,j): \n"<<simpleTerms;

  AlgebraicSymMatrix55 simpleValues=get(state.momentum(),false).matrix();
  LogDebug(theCategory+"|Adjust")<<"config: (i,i), (i,j): \n"<<simpleValues;

  for(int i = 0;i!=5;i++){for(int j = i;j!=5;j++){
      //check on each term for desired action
      switch(theTermAction[Pindex(i,j)]){
      case scale: 
	{
	  simpleTerms(i,j) *= simpleValues(i,j);
	  break;
	}
      case assign:
	{
	  simpleTerms(i,j) = simpleValues(i,j);
	  break;
	}
      case error:
	{
	  edm::LogError(theCategory+"|Adjust")<<" cannot properly adjust for term: "<<i <<","<<j;
	}
      }
    }}
  LogDebug(theCategory+"|Adjust")<<"updated state sigma(i), rho(i,j): \n"<<simpleTerms;

  AlgebraicSymMatrix55 finalTerms;
  complicatedTerm(simpleTerms, finalTerms);
  LogDebug(theCategory+"|Adjust")<<"updated state COV(i,j): \n"<<finalTerms;  
  
  CurvilinearTrajectoryError oMat(finalTerms);
  state = FreeTrajectoryState(state.parameters(), oMat);
  LogDebug(theCategory+"|Adjust")<<"updated state:\n"<<state;
}



void MuonErrorMatrix::adjust(TrajectoryStateOnSurface & state){

  AlgebraicSymMatrix55 simpleTerms;
  simpleTerm(state.curvilinearError(), simpleTerms);
  LogDebug(theCategory+"|Adjust")<<"state sigma(i), rho(i,j): \n"<<simpleTerms;

  AlgebraicSymMatrix55 simpleValues= get(state.globalMomentum(),false).matrix();
  LogDebug(theCategory+"|Adjust")<<"config: (i,i), (i,j):\n"<<simpleValues;

  for(int i = 0;i!=5;i++){for(int j = i;j!=5;j++){
      //check on each term for desired action
      switch(theTermAction[Pindex(i,j)]){
      case scale: 
	{
	  simpleTerms(i,j) *= simpleValues(i,j);
	  break;
	}
      case assign:
	{
	  simpleTerms(i,j) = simpleValues(i,j);
	  break;
	}
      case error:
	{
	  edm::LogError(theCategory+"|Adjust")<<" cannot properly adjust for term: "<<i <<","<<j;
	}
      }
    }}
  LogDebug(theCategory+"|Adjust")<<"updated state sigma(i), rho(i,j): \n"<<simpleTerms;

  AlgebraicSymMatrix55 finalTerms;
  complicatedTerm(simpleTerms, finalTerms);
  LogDebug(theCategory+"|Adjust")<<"updated state COV(i,j): \n"<<finalTerms;  

  CurvilinearTrajectoryError oMat(finalTerms);
  state = TrajectoryStateOnSurface(state.weight(),
                                   state.globalParameters(),
				   oMat,
				   state.surface(),
				   state.surfaceSide()
				  );
  LogDebug(theCategory+"|Adjust")<<"updated state:\n"<<state;
}
