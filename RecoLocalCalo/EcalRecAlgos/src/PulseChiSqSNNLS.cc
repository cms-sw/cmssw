#include "RecoLocalCalo/EcalRecAlgos/interface/PulseChiSqSNNLS.h"
#include <math.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

PulseChiSqSNNLS::PulseChiSqSNNLS() :
  _chisq(0.),
  _computeErrors(true)
{
  
  Eigen::initParallel();
      
}  

PulseChiSqSNNLS::~PulseChiSqSNNLS() {
  
}

bool PulseChiSqSNNLS::DoFit(const std::vector<double> &samples, const SampleMatrix &samplecor, double pederr, const std::set<int> &bxs, const FullSampleVector &fullpulse, const FullSampleMatrix &fullpulsecov) {
 
  const unsigned int nsample = SampleVector::RowsAtCompileTime;
  const unsigned int npulse = bxs.size();

  for (unsigned int isample=0; isample<nsample; ++isample) {
    _sampvec(isample) = samples[isample];
  }
  
  _pulsemat = SamplePulseMatrix::Zero(nsample,npulse);
  _ampvec = PulseVector::Zero(npulse);
  _errvec = PulseVector::Zero(npulse);  
  _ampvecmin = PulseVector::Zero(npulse);
  
  //initialize parameters and index index set
  _idxsP.clear();
  _idxsFixed.clear();
  _chisq = 0.;
  
  //initialize pulse template matrix
  for (std::set<int>::const_iterator bxit = bxs.begin(); bxit!=bxs.end(); ++bxit) {
    int ipulse = std::distance(bxs.begin(),bxit);
    int bx = *bxit;
    int firstsamplet = std::max(0,bx + 3);
    int offset = 7-3-bx;
    
    const unsigned int nsamplepulse = nsample-firstsamplet;    
    //_pulsemat.block(firstsamplet,ipulse,nsamplepulse,1) = fullpulse.segment(firstsamplet+offset,nsamplepulse).transpose();
    //_pulsemat.col(ipulse).segment(firstsamplet,nsamplepulse) = fullpulse.segment(firstsamplet+offset,nsamplepulse).transpose();
    _pulsemat.col(ipulse).segment(firstsamplet,nsamplepulse) = fullpulse.segment(firstsamplet+offset,nsamplepulse);
    
  }

  bool status = Minimize(samplecor,pederr,bxs,fullpulsecov);
  _ampvecmin = _ampvec;
  
  if (!status) return status;
  
  if(!_computeErrors) return status;
 
  //compute MINOS-like uncertainties for in-time amplitude
  if (bxs.count(0)) {
    int ipulseintime = std::distance(bxs.begin(), bxs.find(0));
    double approxerr = ComputeApproxUncertainty(ipulseintime);
    
    double chisq0 = _chisq;
    double x0 = _ampvecmin[ipulseintime];

    //fix in time amplitude
    _idxsFixed.insert(ipulseintime);
    std::set<unsigned int>::const_iterator itintime = _idxsP.find(ipulseintime);
    if (itintime != _idxsP.end()) {
      _idxsP.erase(itintime);
    }
    
    //two point interpolation for upper uncertainty when amplitude is away from boundary
    double xplus100 = x0 + approxerr;
    _ampvec[ipulseintime] = xplus100;
    status &= Minimize(samplecor,pederr,bxs,fullpulsecov);
    if (!status) return status;
    double chisqplus100 = ComputeChiSq();
    
    double sigmaplus = std::abs(xplus100-x0)/sqrt(chisqplus100-chisq0);
    
    //if amplitude is sufficiently far from the boundary, compute also the lower uncertainty and average them
    if ( (x0/sigmaplus) > 0.5 ) {
      double xminus100 = std::max(0.,x0-approxerr);
      _ampvec[ipulseintime] = xminus100;
      status &= Minimize(samplecor,pederr,bxs,fullpulsecov);
      if (!status) return status;
      double chisqminus100 = ComputeChiSq();
      
      double sigmaminus = std::abs(xminus100-x0)/sqrt(chisqminus100-chisq0);
      _errvec[ipulseintime] = 0.5*(sigmaplus + sigmaminus);
      
    }
    else {
      _errvec[ipulseintime] = sigmaplus;
    }
              
    _chisq = chisq0;
  }
  
  return status;
  
}

bool PulseChiSqSNNLS::Minimize(const SampleMatrix &samplecor, double pederr, const std::set<int> &bxs, const FullSampleMatrix &fullpulsecov) {

  
  const int maxiter = 50;
  int iter = 0;
  bool status = false;
  while (true) {    
    
    if (iter>=maxiter) {
      edm::LogWarning("PulseChiSqSNNLS::Minimize") << "Max Iterations reached at iter " << iter <<  std::endl;
      break;
    }    
    
    status = updateCov(samplecor,pederr,bxs,fullpulsecov);    
    if (!status) break;    
    status = NNLS();
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

bool PulseChiSqSNNLS::updateCov(const SampleMatrix &samplecor, double pederr, const std::set<int> &bxs, const FullSampleMatrix &fullpulsecov) {
 
  const unsigned int nsample = SampleVector::RowsAtCompileTime;
  
  _invcov = pederr*pederr*samplecor;
  
  for (std::set<int>::const_iterator bxit = bxs.begin(); bxit!=bxs.end(); ++bxit) {
    int ipulse = std::distance(bxs.begin(),bxit);
    if (_ampvec[ipulse]==0.) continue;
    int bx = *bxit;
    int firstsamplet = std::max(0,bx + 3);
    int offset = 7-3-bx;
        
    double ampsq = _ampvec[ipulse]*_ampvec[ipulse];
    
    const unsigned int nsamplepulse = nsample-firstsamplet;    
    _invcov.block(firstsamplet,firstsamplet,nsamplepulse,nsamplepulse) += ampsq*fullpulsecov.block(firstsamplet+offset,firstsamplet+offset,nsamplepulse,nsamplepulse);
    
  }
    
  //_decompP.SetMatrixFast(_invcov,_decompPstorage.data());
  //bool status = _decompP.Invert(_invcov);
  
  _covdecomp.compute(_invcov);
  
  bool status = true;
  return status;
    
}

double PulseChiSqSNNLS::ComputeChiSq() {
 
  //compute chi square after fit
//   _workvec = _pulsemat*_ampvec;
//   _workvec -= _sampvec;
//   _workvec *= -1.;
//   return _invcov.Similarity(_workvec);   
  
  SampleVector resvec = _pulsemat*_ampvec - _sampvec;
  return resvec.transpose()*_covdecomp.solve(resvec);
  
}

double PulseChiSqSNNLS::ComputeApproxUncertainty(unsigned int ipulse) {
  //compute approximate uncertainties
  //(using 1/second derivative since full Hessian is not meaningful in
  //presence of positive amplitude boundaries.)
    
  const auto &pulsevec = _pulsemat.col(ipulse);
  return 1./sqrt(pulsevec.transpose()*_covdecomp.solve(pulsevec));
  
}

bool PulseChiSqSNNLS::NNLS() {
  
  //Fast NNLS (fnnls) algorithm as per http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.9203&rep=rep1&type=pdf
  
  //const unsigned int nsample = SampleVector::RowsAtCompileTime;
  
  SamplePulseMatrix invcovp = _covdecomp.solve(_pulsemat);  
  PulseMatrix aTamat = _pulsemat.transpose()*invcovp;
  PulseVector aTbvec = invcovp.transpose()*_sampvec;
  
  //correct for possible effect of non-zero fixed amplitudes
  SampleVector sampcorvec = _sampvec;
  for (std::set<unsigned int>::const_iterator itidx=_idxsFixed.begin(); itidx!=_idxsFixed.end(); ++itidx) {
    sampcorvec -= _ampvec[*itidx]*_pulsemat.col(*itidx);
  }
  PulseVector aTbcorvec = invcovp.transpose()*sampcorvec;
  
  const unsigned int npulse = _ampvec.rows();
  int iter = 0;
  while (true) {
    //printf("iter out, idxsP = %i\n",int(_idxsP.size()));
    
    //can only perform this step if solution is guaranteed viable
    if (iter>0 || !_idxsP.size()) {
      if ( (_idxsP.size()+_idxsFixed.size())==npulse ) break;
      
      //compute derivatives
      PulseVector wvec = aTbvec - aTamat*_ampvec;
      
      //find wmax in active set
      double wmax = -std::numeric_limits<double>::max();
      unsigned int idxwmax = 0;
      for (unsigned int idx=0; idx<npulse; ++idx) {
        //printf("_ampvec[%i] = %5e, w[%i] = %5e\n",idx,_ampvec[idx],idx,wvec[idx]);
        if (!_idxsP.count(idx) && !_idxsFixed.count(idx) && wvec[idx]>wmax) {
          wmax = wvec[idx];
          idxwmax = idx;
        }
      }
      
      //convergence
      if (wmax<1e-11) break;
      
      //unconstrain parameter
      _idxsP.insert(idxwmax);
    }

    
    while (true) {
      //printf("iter in, idxsP = %i\n",int(_idxsP.size()));
      
      if (_idxsP.size()==0) break;
      
      //trick: resize matrices without reallocating memory
      const unsigned int npulseP = _idxsP.size();
      PulseMatrix aPmat(npulseP,npulseP);
      PulseVector sPvec(npulseP);
      
      //fill reduced matrix AP
      for (std::set<unsigned int>::const_iterator itidx=_idxsP.begin(); itidx!=_idxsP.end(); ++itidx) {
        unsigned int iidx = std::distance(_idxsP.begin(),itidx);
        sPvec(iidx) = aTbcorvec(*itidx);        
        for (std::set<unsigned int>::const_iterator jtidx=_idxsP.begin(); jtidx!=_idxsP.end(); ++jtidx) {
          unsigned int jidx = std::distance(_idxsP.begin(),jtidx);
          aPmat(iidx,jidx) = aTamat(*itidx,*jtidx);
        }
      }
      
      //solve for unconstrained parameters      
      sPvec = aPmat.ldlt().solve(sPvec);
      
      //check solution
      if (sPvec.minCoeff()>0.) {
        for (std::set<unsigned int>::const_iterator itidx=_idxsP.begin(); itidx!=_idxsP.end(); ++itidx) {
          unsigned int iidx = std::distance(_idxsP.begin(),itidx);
          _ampvec[*itidx] = sPvec[iidx];
        }
              
        break;
      }      
      
      //update parameter vector
      double minratio = std::numeric_limits<double>::max();
      unsigned int minratioidx = 0;
      for (std::set<unsigned int>::const_iterator itidx=_idxsP.begin(); itidx!=_idxsP.end(); ++itidx) {
        unsigned int iidx = std::distance(_idxsP.begin(),itidx);
        double ratio = _ampvec[*itidx]/(_ampvec[*itidx]-sPvec[iidx]);
        if (sPvec[iidx]<=0. && ratio<minratio) {
          minratio = ratio;
          minratioidx = *itidx;
        }
      }
      
      //re-constraint parameters at the boundary
      for (std::set<unsigned int>::const_iterator itidx=_idxsP.begin(); itidx!=_idxsP.end(); ++itidx) {
        unsigned int iidx = std::distance(_idxsP.begin(),itidx);
        _ampvec[*itidx] += minratio*(sPvec[iidx] - _ampvec[*itidx]);
      }
      
      
      //printf("fixing indexes\n");
      for (unsigned int ipulse = 0; ipulse<npulse; ++ipulse) {
        if (_ampvec[ipulse]==0. || ipulse==minratioidx) {
          std::set<unsigned int>::const_iterator itpulse = _idxsP.find(ipulse);
          if (itpulse!=_idxsP.end()) {
            _ampvec[ipulse] = 0.;
            _idxsP.erase(itpulse);
          }
        }
      }          
      
    }
    ++iter;
  }

  return true;
  
  
}
