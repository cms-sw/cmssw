#include "RecoLocalCalo/EcalRecAlgos/interface/PulseChiSqSNNLS.h"
#include <math.h>
#include "vdt/vdtMath.h"

PulseChiSqSNNLS::PulseChiSqSNNLS(const std::vector<double> &samples, const TMatrixDSym &samplecov, const std::set<int> &bxs, const TVectorD &fullpulse, const TMatrixDSym &fullpulsecov) :
  _sampvec(samples.size(),samples.data()),
  _pulsemat(samples.size(),bxs.size()),
  _invcov(samplecov),
  _ampvec(bxs.size()),
  _workvec(samples.size()),
  _workmat(bxs.size(),samples.size()),
  _aTamat(bxs.size(),bxs.size()),
  _wvec(bxs.size()),
  _aTbvec(bxs.size())
{
  
  _decompP.SetMatrixFast(_invcov,_decompPstorage.data());
  _decompP.Invert(_invcov);
  
  const unsigned int nsample = _sampvec.GetNrows();
    
  for (std::set<int>::const_iterator bxit = bxs.begin(); bxit!=bxs.end(); ++bxit) {
    int ipulse = std::distance(bxs.begin(),bxit);
    int bx = *bxit;
    int firstsamplet = std::max(0,bx + 3);
    int offset = -3-bx;
        
    for (unsigned int isample = firstsamplet; isample<nsample; ++isample) {
      _pulsemat(isample,ipulse) = fullpulse(isample+offset);
    }
  }
    
}  

PulseChiSqSNNLS::~PulseChiSqSNNLS() {
  
}


bool PulseChiSqSNNLS::updateCov(const double *invals, const TMatrixDSym &samplecov, const std::set<int> &bxs, const TMatrixDSym &fullpulsecov) {
 
  const unsigned int nsample = _sampvec.GetNrows();
  
  
  _invcov = samplecov;
  
  for (std::set<int>::const_iterator bxit = bxs.begin(); bxit!=bxs.end(); ++bxit) {
    int ipulse = std::distance(bxs.begin(),bxit);
    int bx = *bxit;
    int firstsamplet = std::max(0,bx + 3);
    int offset = -3-bx;
        
    double ampsq = invals[ipulse]*invals[ipulse];
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


double PulseChiSqSNNLS::ChiSq() {
  
  _workvec = _sampvec - _pulsemat*_ampvec;
  return _invcov.Similarity(_workvec);
  
}

bool PulseChiSqSNNLS::Minimize() {
  
  //Fast NNLS (fnnls) algorithm as per http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.9203&rep=rep1&type=pdf
  
  _workmat.TMult(_pulsemat,_invcov);
  _aTamat.Mult(_workmat,_pulsemat);
  _aTbvec = _sampvec;
  _aTbvec *= _workmat;
  
  const unsigned int npulse = _ampvec.GetNrows();
  int iter = 0;
  while (true) {
    //printf("iter out, idxsP = %i\n",int(_idxsP.size()));
    
    //can only perform this step if solution is guaranteed viable
    if (iter>0 || !_idxsP.size()) {
      if (_idxsP.size()==npulse) break;
      
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
        if (!_idxsP.count(idx) && _wvec[idx]>wmax) {
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
        _sPvec(iidx) = _aTbvec(*itidx);        
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
        _ampvec.Zero();
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
            _idxsP.erase(itpulse);
          }
        }
      }          
      
    }
    ++iter;
  }
  
  return true;
  
  
}
