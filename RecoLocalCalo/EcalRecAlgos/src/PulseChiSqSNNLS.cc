#include "RecoLocalCalo/EcalRecAlgos/interface/PulseChiSqSNNLS.h"
#include <math.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

PulseChiSqSNNLS::PulseChiSqSNNLS() :
  _sampvec(10),
  _invcov(10),
  _workvec(10),
  _chisq(0.),
  _computeErrors(true)
{
      
}  

PulseChiSqSNNLS::~PulseChiSqSNNLS() {
  
}

bool PulseChiSqSNNLS::DoFit(const std::vector<double> &samples, const TMatrixDSym &samplecor, double pederr, const std::set<int> &bxs, const TVectorD &fullpulse, const TMatrixDSym &fullpulsecov) {
 
  const unsigned int nsample = samples.size();
  const unsigned int npulse = bxs.size();
    
  //resize matrices using reserved memory on the stack
  _pulsemat.Use(nsample,npulse,_pulsematstorage.data());
  _ampvec.Use(npulse,_ampvecstorage.data());
  _ampvecmin.Use(npulse,_ampvecminstorage.data());
  _errvec.Use(npulse,_errvecstorage.data());
  _workmat.Use(npulse,nsample,_workmatstorage.data());
  _aTamat.Use(npulse,npulse,_aTamatstorage.data());
  _wvec.Use(npulse,_wvecstorage.data());
  _aTbvec.Use(npulse,_aTbvecstorage.data());
  _aTbcorvec.Use(npulse,_aTbcorvecstorage.data());
                
  //initialize parameters and index index set
  _sampvec.SetElements(samples.data());
  _ampvec.Zero();
  _idxsP.clear();
  _idxsFixed.clear();
  _chisq = 0.;
  
  //initialize pulse template matrix
  for (std::set<int>::const_iterator bxit = bxs.begin(); bxit!=bxs.end(); ++bxit) {
    int ipulse = std::distance(bxs.begin(),bxit);
    int bx = *bxit;
    int firstsamplet = std::max(0,bx + 3);
    int offset = -3-bx;
        
    for (unsigned int isample = firstsamplet; isample<nsample; ++isample) {
      _pulsemat(isample,ipulse) = fullpulse(isample+offset);
    }
  }

  bool status = Minimize(samplecor,pederr,bxs,fullpulsecov);
  _ampvecmin = _ampvec;
  _errvec = _wvec;
  if (!status) return status;
  
  if(!_computeErrors) return status;
 
  //compute MINOS-like uncertainties for in-time amplitude
  if (bxs.count(0)) {
    int ipulseintime = std::distance(bxs.begin(), bxs.find(0));
    double chisq0 = _chisq;
    double x0 = _ampvecmin[ipulseintime];

    //fix in time amplitude
    _idxsFixed.insert(ipulseintime);
    std::set<unsigned int>::const_iterator itintime = _idxsP.find(ipulseintime);
    if (itintime != _idxsP.end()) {
      _idxsP.erase(itintime);
    }
    
    //two point interpolation for upper uncertainty when amplitude is away from boundary
    double xplus100 = x0 + _errvec[ipulseintime];
    _ampvec[ipulseintime] = xplus100;
    status &= Minimize(samplecor,pederr,bxs,fullpulsecov);
    if (!status) return status;
    double chisqplus100 = ComputeChiSq();
    
    double sigmaplus = std::abs(xplus100-x0)/sqrt(chisqplus100-chisq0);
    
    //if amplitude is sufficiently far from the boundary, compute also the lower uncertainty and average them
    if ( (x0/sigmaplus) > 0.5 ) {
      double xminus100 = std::max(0.,x0-_errvec[ipulseintime]);
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

bool PulseChiSqSNNLS::Minimize(const TMatrixDSym &samplecor, double pederr, const std::set<int> &bxs, const TMatrixDSym &fullpulsecov) {

  
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

bool PulseChiSqSNNLS::updateCov(const TMatrixDSym &samplecor, double pederr, const std::set<int> &bxs, const TMatrixDSym &fullpulsecov) {
 
  const unsigned int nsample = _sampvec.GetNrows();
  
  _invcov = samplecor;
  _invcov *= pederr*pederr;
  
  for (std::set<int>::const_iterator bxit = bxs.begin(); bxit!=bxs.end(); ++bxit) {
    int ipulse = std::distance(bxs.begin(),bxit);
    if (_ampvec[ipulse]==0.) continue;
    int bx = *bxit;
    int firstsamplet = std::max(0,bx + 3);
    int offset = -3-bx;
        
    double ampsq = _ampvec[ipulse]*_ampvec[ipulse];
    for (unsigned int isample = firstsamplet; isample<nsample; ++isample) {
      for (unsigned int jsample = firstsamplet; jsample<nsample; ++jsample) {
        _invcov(isample,jsample) += ampsq*fullpulsecov(isample+offset,jsample+offset);
      }
    }
  }
    
  _decompP.SetMatrixFast(_invcov,_decompPstorage.data());
  bool status = _decompP.Invert(_invcov);
  
  return status;
    
}

double PulseChiSqSNNLS::ComputeChiSq() {
 
  //compute chi square after fit
  _workvec = _pulsemat*_ampvec;
  _workvec -= _sampvec;
  _workvec *= -1.;
  return _invcov.Similarity(_workvec);   
  
}

bool PulseChiSqSNNLS::NNLS() {
  
  //Fast NNLS (fnnls) algorithm as per http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.9203&rep=rep1&type=pdf
  
  unsigned int nsample = _sampvec.GetNrows();
  
  _workmat.TMult(_pulsemat,_invcov);
  _aTamat.Mult(_workmat,_pulsemat);
  _aTbvec = _workmat*_sampvec;
  
  //correct for possible effect of non-zero fixed amplitudes
  _workvec = _sampvec;
  for (std::set<unsigned int>::const_iterator itidx=_idxsFixed.begin(); itidx!=_idxsFixed.end(); ++itidx) {
    for (unsigned int isample=0; isample<nsample; ++isample) {
      _workvec[isample] -= _ampvec[*itidx]*_pulsemat(isample,*itidx);
    }
  }
  _aTbcorvec = _workmat*_workvec;
  
  const unsigned int npulse = _ampvec.GetNrows();
  int iter = 0;
  while (true) {
    //printf("iter out, idxsP = %i\n",int(_idxsP.size()));
    
    //can only perform this step if solution is guaranteed viable
    if (iter>0 || !_idxsP.size()) {
      if ( (_idxsP.size()+_idxsFixed.size())==npulse ) break;
      
      //compute derivatives
      _wvec = _ampvec;
      _wvec *= _aTamat;
      _wvec -= _aTbvec;
      _wvec *= -1.0;
      
      //find wmax in active set
      double wmax = -std::numeric_limits<double>::max();
      unsigned int idxwmax = 0;
      for (unsigned int idx=0; idx<npulse; ++idx) {
        //printf("_ampvec[%i] = %5e, w[%i] = %5e\n",idx,_ampvec[idx],idx,_wvec[idx]);
        if (!_idxsP.count(idx) && !_idxsFixed.count(idx) && _wvec[idx]>wmax) {
          wmax = _wvec[idx];
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
      _aPmat.Use(npulseP,_aPstorage.data());
      _sPvec.Use(npulseP,_sPstorage.data()); 
      
      //fill reduced matrix AP
      for (std::set<unsigned int>::const_iterator itidx=_idxsP.begin(); itidx!=_idxsP.end(); ++itidx) {
        unsigned int iidx = std::distance(_idxsP.begin(),itidx);
        _sPvec(iidx) = _aTbcorvec(*itidx);        
        for (std::set<unsigned int>::const_iterator jtidx=_idxsP.begin(); jtidx!=_idxsP.end(); ++jtidx) {
          unsigned int jidx = std::distance(_idxsP.begin(),jtidx);
          _aPmat(iidx,jidx) = _aTamat(*itidx,*jtidx);
        }
      }
      
      //solve for unconstrained parameters
      _decompP.SetMatrixFast(_aPmat,_decompPstorage.data());
      bool status = _decompP.Solve(_sPvec);
      if (!status) return false;
      
      //check solution
      if (_sPvec.Min()>0.) {
        //_ampvec.Zero();
        for (std::set<unsigned int>::const_iterator itidx=_idxsP.begin(); itidx!=_idxsP.end(); ++itidx) {
          unsigned int iidx = std::distance(_idxsP.begin(),itidx);
          _ampvec[*itidx] = _sPvec[iidx];
        }
              
        break;
      }      
      
      //update parameter vector
      double minratio = std::numeric_limits<double>::max();
      unsigned int minratioidx = 0;
      for (std::set<unsigned int>::const_iterator itidx=_idxsP.begin(); itidx!=_idxsP.end(); ++itidx) {
        unsigned int iidx = std::distance(_idxsP.begin(),itidx);
        double ratio = _ampvec[*itidx]/(_ampvec[*itidx]-_sPvec[iidx]);
        if (_sPvec[iidx]<=0. && ratio<minratio) {
          minratio = ratio;
          minratioidx = *itidx;
        }
      }
      
      //re-constraint parameters at the boundary
      for (std::set<unsigned int>::const_iterator itidx=_idxsP.begin(); itidx!=_idxsP.end(); ++itidx) {
        unsigned int iidx = std::distance(_idxsP.begin(),itidx);
        _ampvec[*itidx] += minratio*(_sPvec[iidx] - _ampvec[*itidx]);
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
  
  //compute approximate uncertainties
  //(using 1/second derivative since full Hessian is not meaningful in
  //presence of positive amplitude boundaries.)
  for (unsigned int ipulse=0; ipulse<npulse; ++ipulse) {
    for (unsigned int isample=0; isample<nsample; ++isample) {
      _workvec[isample] = _pulsemat(isample,ipulse);
    }
    _wvec[ipulse] = 1./sqrt(_invcov.Similarity(_workvec));
  }

  return true;
  
  
}
