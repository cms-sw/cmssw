#include "../interface/FitParams.h"

FitParams::FitParams(){
  nb_layers = 4;
  threshold = 1000; 
  principal = NULL;
  nb_principal=0;
  transform = new double*[3*nb_layers];
  for(int i=0;i<3*nb_layers;i++){
    transform[i]=new double[3*nb_layers];
  }

  nb_multidimfit=0;
  pt_fit = NULL;
  pt_fit_data = NULL;
  phi0_fit = NULL;
  phi0_fit_data = NULL;
  d0_fit = NULL;
  d0_fit_data = NULL;
  eta0_fit = NULL;
  eta0_fit_data = NULL;
  z0_fit = NULL;
  z0_fit_data = NULL;
}

FitParams::FitParams(int n_layers, int thresh){
  nb_layers = n_layers;
  principal = NULL;
  nb_principal=0;
  threshold = thresh;
  transform = new double*[3*nb_layers];
  for(int i=0;i<3*nb_layers;i++){
    transform[i]=new double[3*nb_layers];
  }

  nb_multidimfit=0;
  pt_fit = NULL;
  pt_fit_data = NULL;
  phi0_fit = NULL;
  phi0_fit_data = NULL;
  d0_fit = NULL;
  d0_fit_data = NULL;
  eta0_fit = NULL;
  eta0_fit_data = NULL;
  z0_fit = NULL;
  z0_fit_data = NULL;
}

FitParams::FitParams(const FitParams& ref){
  nb_layers = ref.nb_layers;
  principal = NULL;
  nb_principal=ref.nb_principal;
  threshold = ref.threshold;
  transform = new double*[3*nb_layers];
  for(int i=0;i<3*nb_layers;i++){
    transform[i]=new double[3*nb_layers];
    for(int j=0;j<3*nb_layers;j++){
      transform[i][j]=ref.transform[i][j];
    }
  }
  eigen = ref.eigen;
  sig = ref.sig;
  mean = ref.mean;
  
  nb_multidimfit=ref.nb_multidimfit;
  pt_fit = NULL;
  pt_fit_data = new MultiDimFitData(*ref.pt_fit_data);
  phi0_fit = NULL;
  phi0_fit_data = new MultiDimFitData(*ref.phi0_fit_data);
  d0_fit = NULL;
  d0_fit_data = new MultiDimFitData(*ref.d0_fit_data);
  eta0_fit = NULL;
  eta0_fit_data = new MultiDimFitData(*ref.eta0_fit_data);
  z0_fit = NULL;
  z0_fit_data =new MultiDimFitData(*ref.z0_fit_data);
}

void FitParams::init(){
  for(int i=0;i<3*nb_layers;i++){
    delete[] transform[i];
  }
  delete[] transform;

  transform = new double*[3*nb_layers];
  for(int i=0;i<3*nb_layers;i++){
    transform[i]=new double[3*nb_layers];
    eigen.push_back(0);
    mean.push_back(0);
    sig.push_back(0);
  }
}

FitParams::~FitParams(){
  //cout<<"destruction fitparams"<<endl;
  if(principal!=NULL)
     delete principal;
  for(int i=0;i<3*nb_layers;i++){
    delete[] transform[i];
  }
  delete[] transform;

  if(pt_fit!=NULL)
    delete pt_fit;
  if(pt_fit_data!=NULL)
    delete pt_fit_data;
  if(phi0_fit!=NULL)
    delete phi0_fit;
  if(phi0_fit_data!=NULL)
    delete phi0_fit_data;
  if(d0_fit!=NULL)
    delete d0_fit;
  if(d0_fit_data!=NULL)
    delete d0_fit_data;
  if(eta0_fit!=NULL)
    delete eta0_fit;
  if(eta0_fit_data!=NULL)
    delete eta0_fit_data;
  if(z0_fit!=NULL)
    delete z0_fit;
  if(z0_fit_data!=NULL)
    delete z0_fit_data;

  // cout<<"destruction FitParams ok"<<endl;
}

void FitParams::addDataForPrincipal(double* d){
  if(nb_principal<threshold){
    nb_principal++;
    if(principal==NULL)
      principal = new TPrincipal(3*nb_layers);
    principal->AddRow(d);
  }
  
  if(nb_principal==threshold){
    computePrincipalParams();
  }
}

bool FitParams::hasPrincipalParams(){
  return nb_principal>threshold;
}

void FitParams::forcePrincipalParamsComputing(){
  if(nb_principal>1){
    computePrincipalParams();
  }
}

bool FitParams::hasMultiDimFitParams(){
  return nb_multidimfit>threshold;
}

void FitParams::forceMultiDimFitParamsComputing(){
  if(nb_multidimfit>1){
    computeMultiDimFitParams();
  }
}
void FitParams::computePrincipalParams(){
  principal->MakePrincipals();
  
  const TVectorD* m_eigen     = principal->GetEigenValues();
  //  if((*m_eigen)[0]!=(*m_eigen)[0])
  //  cout<<"NaN value in eigen vector ("<<nb_principal<<" tracks used)!!"<<endl;
  const TMatrixD* m_transform = principal->GetEigenVectors();
  const TVectorD* m_mean      = principal->GetMeanValues();
  const TVectorD* m_sig       = principal->GetSigmas();
  
  for(int i=0;i<nb_layers*3;i++){
    eigen.push_back((*m_eigen)[i]);
    mean.push_back((*m_mean)[i]);
    sig.push_back((*m_sig)[i]);
    for(int j=0;j<nb_layers*3;j++){
      transform[i][j] = (*m_transform)[i][j];
    }
  }
  
  //principal->GetEigenVectors()->Print();
  principal->Print("mse");
  delete principal;
  principal=NULL;
  nb_principal=threshold+1;
}

void FitParams::initializeMultiDimFit(TMultiDimFit* f){
  Int_t mPowers1[]   = { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
  //Int_t mPowers1[]   = { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
  f->SetMaxPowers(mPowers1);
  f->SetMaxFunctions(100);
  f->SetMaxStudy(100);
  f->SetMaxTerms(10);
  f->SetPowerLimit(0.3);
  f->SetMinAngle(1);
}

void FitParams::addDataForMultiDimFit(double* d, double* val){
  if(nb_multidimfit<threshold){
    nb_multidimfit++;

    double new_system[nb_layers*3];
    x2p(d,new_system);

    if(pt_fit==NULL){
      pt_fit = new TMultiDimFit(3*(nb_layers-1), TMultiDimFit::kChebyshev,"v");
      initializeMultiDimFit(pt_fit);
    }
    if(phi0_fit==NULL){
      phi0_fit = new TMultiDimFit(3*(nb_layers-1), TMultiDimFit::kChebyshev,"v");
      initializeMultiDimFit(phi0_fit);
    }
    if(d0_fit==NULL){
      d0_fit = new TMultiDimFit(3*(nb_layers-1), TMultiDimFit::kChebyshev,"v");
      initializeMultiDimFit(d0_fit);
    }
    if(eta0_fit==NULL){
      eta0_fit = new TMultiDimFit(3*(nb_layers-1), TMultiDimFit::kChebyshev,"v");
      initializeMultiDimFit(eta0_fit);
    }
    if(z0_fit==NULL){
      z0_fit = new TMultiDimFit(3*(nb_layers-1), TMultiDimFit::kChebyshev,"v");
      initializeMultiDimFit(z0_fit);
    }

    //for(int i=0;i<12;i++){
    //  cout<<d[i]<<" -> "<<new_system[i]<<endl;
    //}
   
    //cout<<"D0 : "<<val[2]<<endl;
    //cout<<endl;

    pt_fit->AddRow(new_system,val[0]);
    phi0_fit->AddRow(new_system,val[1]);
    d0_fit->AddRow(new_system,val[2]);
    eta0_fit->AddRow(new_system,val[3]);
    z0_fit->AddRow(new_system,val[4]);
  }
  
  if(nb_multidimfit==threshold){
    computeMultiDimFitParams();
  }
}

void FitParams::computeMultiDimFitParams(){
  if(pt_fit!=NULL && pt_fit_data==NULL){
    pt_fit->FindParameterization();
    pt_fit_data = new MultiDimFitData(pt_fit, nb_layers);
    delete pt_fit;
    pt_fit=NULL;
    phi0_fit->FindParameterization();
    phi0_fit_data = new MultiDimFitData(phi0_fit, nb_layers);
    delete phi0_fit;
    phi0_fit=NULL;
    d0_fit->FindParameterization();
    d0_fit_data = new MultiDimFitData(d0_fit, nb_layers);
    delete d0_fit;
    d0_fit=NULL;
    eta0_fit->FindParameterization();
    eta0_fit_data = new MultiDimFitData(eta0_fit, nb_layers);
    delete eta0_fit;
    eta0_fit=NULL;
    z0_fit->FindParameterization();
    z0_fit_data = new MultiDimFitData(z0_fit, nb_layers);
    delete z0_fit;
    z0_fit=NULL;
    nb_multidimfit=threshold+1;
  }
}

void FitParams::x2p(double *x, double *p)
{
  // This small method is processing these lines using parameters
  // retrieved from the TPrincipal objects

  for (int i=0; i<nb_layers*3; ++i)
  {
    p[i]  = 0.;

    for (int j=0; j<nb_layers*3; ++j)
    {
      p[i]  += (x[j]-mean[j])*transform[j][i]/sig[j];
    }
  }
}

double FitParams::get_chi_square(double *x, double p)
{  
  double chi2 = 0.;
  int dim = nb_layers*3;


  if (3*p>dim) return -1;

  for (int i=(int)(dim-3*p);i<dim;++i){
    //cout<<pow(x[i],2)<<"/"<<eigen[i]<<endl;
    chi2 += pow(x[i],2)/eigen[i];
  }

  return chi2;
}

double FitParams::getPTFitValue(double* val){
  if(pt_fit_data==NULL)
    return -1000;
  
  return pt_fit_data->getVal(val);
}

double FitParams::getPhi0FitValue(double* val){
  if(phi0_fit_data==NULL)
    return -1000;
  
  return phi0_fit_data->getVal(val);
}

double FitParams::getD0FitValue(double* val){
  if(d0_fit_data==NULL){
    cout<<"D0 params do not exist"<<endl;
    return -1000;
  }
  
  return d0_fit_data->getVal(val);
}

double FitParams::getEta0FitValue(double* val){
  if(eta0_fit_data==NULL)
    return -1000;
  
  return eta0_fit_data->getVal(val);
}

double FitParams::getZ0FitValue(double* val){
  if(z0_fit_data==NULL)
    return -1000;
  
  return z0_fit_data->getVal(val);
}

Track* FitParams::getTrack(double* val){
  double pt = getPTFitValue(val);
  double phi0 = getPhi0FitValue(val);
  double d0 = getD0FitValue(val);
  double eta0 = getEta0FitValue(val);
  double z0 = getZ0FitValue(val);
  
  phi0 = atan(phi0);
  pt   = (0.3*3.833)/(2*pt*pow(cos(phi0),3));
  d0   = d0*cos(phi0);
  eta0 = asinh(eta0*cos(phi0));

  Track* t = new Track(pt,d0,phi0,eta0,z0);
  return t;
}

int  FitParams::getNbPrincipalTracks(){
  return nb_principal;
}

int FitParams:: getNbMultiDimFitTracks(){
  return nb_multidimfit;
}
