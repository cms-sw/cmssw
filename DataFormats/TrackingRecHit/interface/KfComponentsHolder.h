#ifndef DataFormats_TrackingRecHit_interface_KfComponentsHolder_h_
#define DataFormats_TrackingRecHit_interface_KfComponentsHolder_h_


#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/Math/interface/ProjectMatrix.h"
#include <stdint.h>

class TrackingRecHit;

#define Debug_KfComponentsHolder

class KfComponentsHolder {
    public:
  KfComponentsHolder() : params_(0), errors_(0), projection_(0), useProjFunc_(false)
        {
#ifdef Debug_KfComponentsHolder
            size_ = 0;
#endif
        }

        template <unsigned int D>
        void setup(
            typename AlgebraicROOTObject<D>::Vector       *params,
            typename AlgebraicROOTObject<D,D>::SymMatrix  *errors,
            typename AlgebraicROOTObject<D,5>::Matrix     *projection,
	    ProjectMatrix<double,5,D>                     *projFunc, 
            typename AlgebraicROOTObject<D>::Vector       *measuredParams,
            typename AlgebraicROOTObject<D,D>::SymMatrix  *measuredErrors,
            const AlgebraicVector5 & tsosLocalParameters, 
            const AlgebraicSymMatrix55 & tsosLocalErrors 
        ) ;


  // backward compatible call
        template <unsigned int D>
        void setup(
            typename AlgebraicROOTObject<D>::Vector       *params,
            typename AlgebraicROOTObject<D,D>::SymMatrix  *errors,
            typename AlgebraicROOTObject<D,5>::Matrix     *projection,
            typename AlgebraicROOTObject<D>::Vector       *measuredParams,
            typename AlgebraicROOTObject<D,D>::SymMatrix  *measuredErrors,
            const AlgebraicVector5 & tsosLocalParameters, 
            const AlgebraicSymMatrix55 & tsosLocalErrors 
        ) ;


        template <unsigned int D>
        typename AlgebraicROOTObject<D>::Vector & params() { 
#ifdef Debug_KfComponentsHolder
            assert(size_ == D);
#endif
            return  * reinterpret_cast<typename AlgebraicROOTObject<D>::Vector *>(params_);
        }

        template <unsigned int D>
        typename AlgebraicROOTObject<D,D>::SymMatrix & errors() { 
#ifdef Debug_KfComponentsHolder
            assert(size_ == D);
#endif
            return  * reinterpret_cast<typename AlgebraicROOTObject<D,D>::SymMatrix *>(errors_);
        }

        template <unsigned int D>
        typename AlgebraicROOTObject<D,5>::Matrix & projection() { 
#ifdef Debug_KfComponentsHolder
            assert(size_ == D);
#endif
            return  * reinterpret_cast<typename AlgebraicROOTObject<D,5>::Matrix *>(projection_);
        }

        template <unsigned int D>
        ProjectMatrix<double,5,D> & projFunc() { 
#ifdef Debug_KfComponentsHolder
            assert(size_ == D);
#endif
            return  * reinterpret_cast< ProjectMatrix<double,5,D> *>(projFunc_);
        }

        /// Fill in datamembers from a generic TrackingRecHit using the CLHEP matrices
        void genericFill(const TrackingRecHit &hit); 

        template <unsigned int D>
        typename AlgebraicROOTObject<D>::Vector & measuredParams() { 
#ifdef Debug_KfComponentsHolder
            assert(size_ == D);
#endif
            return  * reinterpret_cast<typename AlgebraicROOTObject<D>::Vector *>(measuredParams_);
        }

        template <unsigned int D>
        typename AlgebraicROOTObject<D,D>::SymMatrix & measuredErrors() { 
#ifdef Debug_KfComponentsHolder
            assert(size_ == D);
#endif
            return  * reinterpret_cast<typename AlgebraicROOTObject<D,D>::SymMatrix *>(measuredErrors_);
        }

        const AlgebraicVector5     & tsosLocalParameters() const { return *tsosLocalParameters_; }
        const AlgebraicSymMatrix55 & tsosLocalErrors()     const { return *tsosLocalErrors_;     }

        bool useProjFunc() const { return useProjFunc_;}
        void doUseProjFunc() {  useProjFunc_ = true; }

        template<unsigned int D> void dump() ;
    private:
#ifdef Debug_KfComponentsHolder
        uint16_t size_;
#endif
  void *params_, *errors_, *projection_, *projFunc_, *measuredParams_, *measuredErrors_;
  const AlgebraicVector5 * tsosLocalParameters_;
  const AlgebraicSymMatrix55 * tsosLocalErrors_;

  bool useProjFunc_;

  template<unsigned int D>
  void genericFill_(const TrackingRecHit &hit);

        
};


template<unsigned int D>
void KfComponentsHolder::setup(
        typename AlgebraicROOTObject<D>::Vector       *params,
        typename AlgebraicROOTObject<D,D>::SymMatrix  *errors,
        typename AlgebraicROOTObject<D,5>::Matrix     *projection,
	ProjectMatrix<double,5,D>                     *projFunc, 
        typename AlgebraicROOTObject<D>::Vector       *measuredParams,
        typename AlgebraicROOTObject<D,D>::SymMatrix  *measuredErrors,
        const AlgebraicVector5     & tsosLocalParameters, 
        const AlgebraicSymMatrix55 & tsosLocalErrors)
{
#ifdef Debug_KfComponentsHolder
    assert(size_ == 0); // which means it was uninitialized
    size_ = D;
#endif
    params_     = params;
    errors_     = errors;
    projection_ = projection;
    projFunc_ = projFunc;
    measuredParams_ = measuredParams;
    measuredErrors_ = measuredErrors;
    tsosLocalParameters_ = & tsosLocalParameters;
    tsosLocalErrors_     = & tsosLocalErrors;
}

template<unsigned int D>
void KfComponentsHolder::dump() {
    using namespace std;
    cout << "Params  my: " << params<D>() << endl;
    cout << "      tsos: " << tsosLocalParameters() << endl;
    cout << "      meas: " << measuredParams<D>() << endl;
    cout << "Errors  my:\n" << errors<D>() << endl;
    cout << "      tsos:\n" << tsosLocalErrors() << endl;
    cout << "      meas:\n" << measuredErrors<D>() << endl;
    cout << "Projection:\n" << projection<D>() << endl;
}

#ifdef Debug_KfComponentsHolder
// undef it so we don't pollute someone else's code.
#undef Debug_KfComponentsHolder
#endif 

#endif
