#include "RecoLocalCalo/EcalRecAlgos/interface/PulseChiSqSNNLS.h"
#include <math.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

void eigen_solve_submatrix(PulseMatrix& mat, PulseVector& invec, PulseVector& outvec, unsigned NP) {
  using namespace Eigen;
  switch( NP ) { // pulse matrix is always square.
  case 10:
    {
      Matrix<double,10,10> temp = mat;
      outvec.head<10>() = temp.ldlt().solve(invec.head<10>());
    }
    break;
  case 9:
    {
      Matrix<double,9,9> temp = mat.topLeftCorner<9,9>();
      outvec.head<9>() = temp.ldlt().solve(invec.head<9>());
    }
    break;
  case 8:
    {
      Matrix<double,8,8> temp = mat.topLeftCorner<8,8>();
      outvec.head<8>() = temp.ldlt().solve(invec.head<8>());
    }
    break;
  case 7:
    {
      Matrix<double,7,7> temp = mat.topLeftCorner<7,7>();
      outvec.head<7>() = temp.ldlt().solve(invec.head<7>());
    }
    break;
  case 6:
    {
      Matrix<double,6,6> temp = mat.topLeftCorner<6,6>();
      outvec.head<6>() = temp.ldlt().solve(invec.head<6>());
    }
    break;
  case 5:
    {
      Matrix<double,5,5> temp = mat.topLeftCorner<5,5>();
      outvec.head<5>() = temp.ldlt().solve(invec.head<5>());
    }
    break;
  case 4:
    {
      Matrix<double,4,4> temp = mat.topLeftCorner<4,4>();
      outvec.head<4>() = temp.ldlt().solve(invec.head<4>());
    }
    break;
  case 3: 
    {
      Matrix<double,3,3> temp = mat.topLeftCorner<3,3>();
      outvec.head<3>() = temp.ldlt().solve(invec.head<3>());
    }
    break;
  case 2:
    {
      Matrix<double,2,2> temp = mat.topLeftCorner<2,2>();
      outvec.head<2>() = temp.ldlt().solve(invec.head<2>());
    }
    break;
  case 1:
    {
      Matrix<double,1,1> temp = mat.topLeftCorner<1,1>();
      outvec.head<1>() = temp.ldlt().solve(invec.head<1>());
    }
    break;
  default:
    throw cms::Exception("MultFitWeirdState")
      << "Weird number of pulses encountered in multifit, module is configured incorrectly!";
  }
}

PulseChiSqSNNLS::PulseChiSqSNNLS() :
  _chisq(0.),
  _computeErrors(true),
  _maxiters(50),
  _maxiterwarnings(true)
{
  
  Eigen::initParallel();
      
}  

PulseChiSqSNNLS::~PulseChiSqSNNLS() {
  
}

bool PulseChiSqSNNLS::DoFit(const SampleVector &samples, const SampleMatrix &samplecor, double pederr, const BXVector &bxs, const FullSampleVector &fullpulse, const FullSampleMatrix &fullpulsecov) {
 
  //const unsigned int nsample = SampleVector::RowsAtCompileTime;
  _npulsetot = bxs.rows();
  //const unsigned int npulse = bxs.rows();

  _sampvec = samples;
  _bxs = bxs;
  
  //_pulsemat = SamplePulseMatrix::Zero(nsample,npulse);
  _pulsemat.resize(Eigen::NoChange,_npulsetot);
  _ampvec = PulseVector::Zero(_npulsetot);
  _errvec = PulseVector::Zero(_npulsetot);  
  _nP = 0;
  _chisq = 0.;
  
  if (_bxs.rows()==1) {
    _ampvec.coeffRef(0) = _sampvec.coeff(_bxs.coeff(0) + 5);
  }
  
  aTamat.resize(_npulsetot,_npulsetot);
  wvec.resize(_npulsetot);

  //initialize pulse template matrix
  for (unsigned int ipulse=0; ipulse<_npulsetot; ++ipulse) {
    int bx = _bxs.coeff(ipulse);
    //int firstsamplet = std::max(0,bx + 3);
    int offset = 7-3-bx;
    
    //const unsigned int nsamplepulse = nsample-firstsamplet;
    //_pulsemat.col(ipulse).segment(firstsamplet,nsamplepulse) = fullpulse.segment(firstsamplet+offset,nsamplepulse);
    
    _pulsemat.col(ipulse) = fullpulse.segment<SampleVector::RowsAtCompileTime>(offset);
  }

  //do the actual fit
  bool status = Minimize(samplecor,pederr,fullpulsecov);
  _ampvecmin = _ampvec;
  _bxsmin = _bxs;
  
  if (!status) return status;
  
  if(!_computeErrors) return status;

  //compute MINOS-like uncertainties for in-time amplitude
  bool foundintime = false;
  unsigned int ipulseintime = 0;
  for (unsigned int ipulse=0; ipulse<_npulsetot; ++ipulse) {
    if (_bxs.coeff(ipulse)==0) {
      ipulseintime = ipulse;
      foundintime = true;
      break;
    }
  }
  if (!foundintime) return status;
  
  const unsigned int ipulseintimemin = ipulseintime;
  
  double approxerr = ComputeApproxUncertainty(ipulseintime);
  double chisq0 = _chisq;
  double x0 = _ampvecmin[ipulseintime];  
  
  //move in time pulse first to active set if necessary
  if (ipulseintime<_nP) {
    _pulsemat.col(_nP-1).swap(_pulsemat.col(ipulseintime));
    std::swap(_ampvec.coeffRef(_nP-1),_ampvec.coeffRef(ipulseintime));
    std::swap(_bxs.coeffRef(_nP-1),_bxs.coeffRef(ipulseintime));
    ipulseintime = _nP - 1;
    --_nP;    
  }
  
  
  SampleVector pulseintime = _pulsemat.col(ipulseintime);
  _pulsemat.col(ipulseintime).setZero();
  
  //two point interpolation for upper uncertainty when amplitude is away from boundary
  double xplus100 = x0 + approxerr;
  _ampvec.coeffRef(ipulseintime) = xplus100;
  _sampvec = samples - _ampvec.coeff(ipulseintime)*pulseintime;  
  status &= Minimize(samplecor,pederr,fullpulsecov);
  if (!status) return status;
  double chisqplus100 = ComputeChiSq();
  
  double sigmaplus = std::abs(xplus100-x0)/sqrt(chisqplus100-chisq0);
  
  //if amplitude is sufficiently far from the boundary, compute also the lower uncertainty and average them
  if ( (x0/sigmaplus) > 0.5 ) {
    for (unsigned int ipulse=0; ipulse<_npulsetot; ++ipulse) {
      if (_bxs.coeff(ipulse)==0) {
        ipulseintime = ipulse;
        break;
      }
    }    
    double xminus100 = std::max(0.,x0-approxerr);
    _ampvec.coeffRef(ipulseintime) = xminus100;
   _sampvec = samples - _ampvec.coeff(ipulseintime)*pulseintime;
    status &= Minimize(samplecor,pederr,fullpulsecov);
    if (!status) return status;
    double chisqminus100 = ComputeChiSq();
    
    double sigmaminus = std::abs(xminus100-x0)/sqrt(chisqminus100-chisq0);
    _errvec[ipulseintimemin] = 0.5*(sigmaplus + sigmaminus);
    
  }
  else {
    _errvec[ipulseintimemin] = sigmaplus;
  }
            
  _chisq = chisq0;  

  return status;
  
}

bool PulseChiSqSNNLS::Minimize(const SampleMatrix &samplecor, double pederr, const FullSampleMatrix &fullpulsecov) {

  const unsigned int npulse = _bxs.rows();
  
  int iter = 0;
  bool status = false;
  while (true) {    
    
    if (iter>=_maxiters) {
      if (_maxiterwarnings) {
        edm::LogWarning("PulseChiSqSNNLS::Minimize") << "Max Iterations reached at iter " << iter <<  std::endl;
      }
      break;
    }    
    
    status = updateCov(samplecor,pederr,fullpulsecov);    
    if (!status) break; 
    if (npulse>1) {
      status = NNLS();
    }
    else {
      //special case for one pulse fit (performance optimized)
      status = OnePulseMinimize();
    }
    if (!status) break;
        
    double chisqnow = ComputeChiSq();
    double deltachisq = chisqnow-_chisq;
        
    _chisq = chisqnow;
    if (std::abs(deltachisq)<1e-3) {
      break;
    }
    ++iter;    
  }  
  
  return status;  
  
}

bool PulseChiSqSNNLS::updateCov(const SampleMatrix &samplecor, double pederr, const FullSampleMatrix &fullpulsecov) {
 
  const unsigned int nsample = SampleVector::RowsAtCompileTime;
  const unsigned int npulse = _bxs.rows();

  const double pederr2 = pederr*pederr;
  _invcov = pederr2*samplecor; //
  
  for (unsigned int ipulse=0; ipulse<npulse; ++ipulse) {
    if (_ampvec.coeff(ipulse)==0.) continue;
    int bx = _bxs.coeff(ipulse);
    int firstsamplet = std::max(0,bx + 3);
    int offset = 7-3-bx;
    
    const double ampveccoef = _ampvec.coeff(ipulse);
    const double ampsq = ampveccoef*ampveccoef;
    
    const unsigned int nsamplepulse = nsample-firstsamplet;    
    _invcov.block(firstsamplet,firstsamplet,nsamplepulse,nsamplepulse) += 
      ampsq*fullpulsecov.block(firstsamplet+offset,firstsamplet+offset,nsamplepulse,nsamplepulse);   
  }
  
  _covdecomp.compute(_invcov);
  
  bool status = true;
  return status;
    
}

double PulseChiSqSNNLS::ComputeChiSq() {
  
//   SampleVector resvec = _pulsemat*_ampvec - _sampvec;
//   return resvec.transpose()*_covdecomp.solve(resvec);
  
  return _covdecomp.matrixL().solve(_pulsemat*_ampvec - _sampvec).squaredNorm();
  
}

double PulseChiSqSNNLS::ComputeApproxUncertainty(unsigned int ipulse) {
  //compute approximate uncertainties
  //(using 1/second derivative since full Hessian is not meaningful in
  //presence of positive amplitude boundaries.)
      
  return 1./_covdecomp.matrixL().solve(_pulsemat.col(ipulse)).norm();
  
}

bool PulseChiSqSNNLS::NNLS() {
  
  //Fast NNLS (fnnls) algorithm as per http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.9203&rep=rep1&type=pdf
  
  const unsigned int npulse = _bxs.rows();

  invcovp = _covdecomp.matrixL().solve(_pulsemat);
  aTamat = invcovp.transpose()*invcovp; //.triangularView<Eigen::Lower>()
  //aTamat = aTamat.selfadjointView<Eigen::Lower>();  
  aTbvec = invcovp.transpose()*_covdecomp.matrixL().solve(_sampvec);  
  
  int iter = 0;
  Index idxwmax = 0;
  double wmax = 0.0;
  double threshold = 1e-11;
  //work = PulseVector::zeros();
  while (true) {    
    //can only perform this step if solution is guaranteed viable
    if (iter>0 || _nP==0) {
      if ( _nP==npulse ) break;                  
      
      const unsigned int nActive = npulse - _nP;
      
      updatework = aTbvec - aTamat*_ampvec;      
      Index idxwmaxprev = idxwmax;
      double wmaxprev = wmax;
      wmax = updatework.tail(nActive).maxCoeff(&idxwmax);
      
      //convergence
      if (wmax<threshold || (idxwmax==idxwmaxprev && wmax==wmaxprev)) break;
      
      //worst case protection
      if (iter>=500) {
        edm::LogWarning("PulseChiSqSNNLS::NNLS()") << "Max Iterations reached at iter " << iter <<  std::endl;
        break;
      }
      
      //unconstrain parameter
      Index idxp = _nP + idxwmax;
      //printf("adding index %i, orig index %i\n",int(idxp),int(_bxs.coeff(idxp)));
      aTamat.col(_nP).swap(aTamat.col(idxp));
      aTamat.row(_nP).swap(aTamat.row(idxp));
      _pulsemat.col(_nP).swap(_pulsemat.col(idxp));
      std::swap(aTbvec.coeffRef(_nP),aTbvec.coeffRef(idxp));
      std::swap(_ampvec.coeffRef(_nP),_ampvec.coeffRef(idxp));
      std::swap(_bxs.coeffRef(_nP),_bxs.coeffRef(idxp));
      
      // update now that we are done doing work
      wvec.tail(nActive) = updatework.tail(nActive); 
      ++_nP;
    }

    
    while (true) {
      //printf("iter in, idxsP = %i\n",int(_idxsP.size()));
      
      if (_nP==0) break;     
      
      ampvecpermtest = _ampvec;
      
      //solve for unconstrained parameters 
      //need to have specialized function to call optimized versions
      // of matrix solver... this is truly amazing...
      eigen_solve_submatrix(aTamat,aTbvec,ampvecpermtest,_nP);
      
      //check solution
      auto ampvecpermhead = ampvecpermtest.head(_nP);
      if ( ampvecpermhead.minCoeff()>0. ) {
        _ampvec.head(_nP) = ampvecpermhead.head(_nP);
        break;
      }      

      //update parameter vector
      Index minratioidx=0;
      
      // no realizable optimization here (because it autovectorizes!)
      double minratio = std::numeric_limits<double>::max();
      for (unsigned int ipulse=0; ipulse<_nP; ++ipulse) {
        if (ampvecpermtest.coeff(ipulse)<=0.) {
	  const double c_ampvec = _ampvec.coeff(ipulse);
          const double ratio = c_ampvec/(c_ampvec-ampvecpermtest.coeff(ipulse));
          if (ratio<minratio) {
            minratio = ratio;
            minratioidx = ipulse;
          }
        }
      }

      _ampvec.head(_nP) += minratio*(ampvecpermhead - _ampvec.head(_nP));
      
      //avoid numerical problems with later ==0. check
      _ampvec.coeffRef(minratioidx) = 0.;
            
      //printf("removing index %i, orig idx %i\n",int(minratioidx),int(_bxs.coeff(minratioidx)));
      aTamat.col(_nP-1).swap(aTamat.col(minratioidx));
      aTamat.row(_nP-1).swap(aTamat.row(minratioidx));
      _pulsemat.col(_nP-1).swap(_pulsemat.col(minratioidx));
      std::swap(aTbvec.coeffRef(_nP-1),aTbvec.coeffRef(minratioidx));
      std::swap(_ampvec.coeffRef(_nP-1),_ampvec.coeffRef(minratioidx));
      std::swap(_bxs.coeffRef(_nP-1),_bxs.coeffRef(minratioidx));
      --_nP;      
    }
    ++iter;
    
    //adaptive convergence threshold to avoid infinite loops but still
    //ensure best value is used
    if (iter%50==0) {
      threshold *= 10.;
    }
  }
  
  return true;
  
  
}

bool PulseChiSqSNNLS::OnePulseMinimize() {
  
  //Fast NNLS (fnnls) algorithm as per http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.9203&rep=rep1&type=pdf
  
//   const unsigned int npulse = 1;

  invcovp = _covdecomp.matrixL().solve(_pulsemat);
//   aTamat = invcovp.transpose()*invcovp;
//   aTbvec = invcovp.transpose()*_covdecomp.matrixL().solve(_sampvec);

  SingleMatrix aTamatval = invcovp.transpose()*invcovp;
  SingleVector aTbvecval = invcovp.transpose()*_covdecomp.matrixL().solve(_sampvec);
  _ampvec.coeffRef(0) = std::max(0.,aTbvecval.coeff(0)/aTamatval.coeff(0));
  
  return true;
  
}
