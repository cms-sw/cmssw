#include <iostream>
#include <limits>

#include "cuda.h"

#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"

#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "inplace_fnnls.h"
#include "AmplitudeComputationKernelsV1.h"
#include "AmplitudeComputationCommonKernels.h"

namespace ecal { namespace multifit {

void eigen_solve_submatrix(SampleMatrix& mat, 
                           SampleVector& invec, 
                           SampleVector& outvec, unsigned NP) {
    using namespace Eigen;
    switch( NP ) { // pulse matrix is always square.
    case 10: {   
        Matrix<SampleMatrix::Scalar,10,10> temp = mat.topLeftCorner<10,10>();
        outvec.head<10>() = temp.ldlt().solve(invec.head<10>());
        break;
    }   
    case 9: {
        Matrix<SampleMatrix::Scalar,9,9> temp = mat.topLeftCorner<9,9>();
        outvec.head<9>() = temp.ldlt().solve(invec.head<9>());
        break;
    }   
    case 8: {   
        Matrix<SampleMatrix::Scalar,8,8> temp = mat.topLeftCorner<8,8>();
        outvec.head<8>() = temp.ldlt().solve(invec.head<8>());
        break;
    }   
    case 7: {   
        Matrix<SampleMatrix::Scalar,7,7> temp = mat.topLeftCorner<7,7>();
        outvec.head<7>() = temp.ldlt().solve(invec.head<7>());
        break;
    }   
    case 6: {   
        Matrix<SampleMatrix::Scalar,6,6> temp = mat.topLeftCorner<6,6>();
        outvec.head<6>() = temp.ldlt().solve(invec.head<6>());
        break;
    }   
    case 5: {   
        Matrix<SampleMatrix::Scalar,5,5> temp = mat.topLeftCorner<5,5>();
        outvec.head<5>() = temp.ldlt().solve(invec.head<5>());
        break;
    }   
    case 4: {   
        Matrix<SampleMatrix::Scalar,4,4> temp = mat.topLeftCorner<4,4>();
        outvec.head<4>() = temp.ldlt().solve(invec.head<4>());
        break;
    }   
    case 3: {   
        Matrix<SampleMatrix::Scalar,3,3> temp = mat.topLeftCorner<3,3>();
        outvec.head<3>() = temp.ldlt().solve(invec.head<3>());
        break;
    }   
    case 2: {   
        Matrix<SampleMatrix::Scalar,2,2> temp = mat.topLeftCorner<2,2>();
        outvec.head<2>() = temp.ldlt().solve(invec.head<2>());
        break;
    }   
    case 1: {   
        Matrix<SampleMatrix::Scalar,1,1> temp = mat.topLeftCorner<1,1>();
        outvec.head<1>() = temp.ldlt().solve(invec.head<1>());
        break;
    }    
    default:
        return;
    }
}

#define PRINT_MATRIX_10x10(M)\
            printf("%f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f\n", \
                M(0, 0), M(1, 0), M(2, 0), M(3, 0), M(4, 0), \
                M(5, 0), M(6, 0), M(7, 0), M(8, 0), M(9, 0), \
                M(0, 1), M(1, 1), M(2, 1), M(3, 1), M(4, 1), \
                M(5, 1), M(6, 1), M(7, 1), M(8, 1), M(9, 1), \
                M(0, 2), M(1, 2), M(2, 2), M(3, 2), M(4, 2), \
                M(5, 2), M(6, 2), M(7, 2), M(8, 2), M(9, 2), \
                M(0, 3), M(1, 3), M(2, 3), M(3, 3), M(4, 3), \
                M(5, 3), M(6, 3), M(7, 3), M(8, 3), M(9, 3), \
                M(0, 4), M(1, 4), M(2, 4), M(3, 4), M(4, 4), \
                M(5, 4), M(6, 4), M(7, 4), M(8, 4), M(9, 4), \
                M(0, 5), M(1, 5), M(2, 5), M(3, 5), M(4, 5), \
                M(5, 5), M(6, 5), M(7, 5), M(8, 5), M(9, 5), \
                M(0, 6), M(1, 6), M(2, 6), M(3, 6), M(4, 6), \
                M(5, 6), M(6, 6), M(7, 6), M(8, 6), M(9, 6), \
                M(0, 7), M(1, 7), M(2, 7), M(3, 7), M(4, 7), \
                M(5, 7), M(6, 7), M(7, 7), M(8, 7), M(9, 7), \
                M(0, 8), M(1, 8), M(2, 8), M(3, 8), M(4, 8), \
                M(5, 8), M(6, 8), M(7, 8), M(8, 8), M(9, 8), \
                M(0, 9), M(1, 9), M(2, 9), M(3, 9), M(4, 9), \
                M(5, 9), M(6, 9), M(7, 9), M(8, 9), M(9, 9) \
            )

__device__
__forceinline__
bool update_covariance(SampleMatrix const& noisecov,
                       FullSampleMatrix const& full_pulse_cov,
                       SampleMatrix& inverse_cov,
                       BXVectorType const& bxs,
                       SampleDecompLLT& covariance_decomposition,
                       SampleVector const& amplitudes) {
    constexpr int nsamples = SampleVector::RowsAtCompileTime;
    constexpr int npulses = BXVectorType::RowsAtCompileTime;

    inverse_cov = noisecov;

    for (unsigned int ipulse=0; ipulse<npulses; ipulse++) {
        if (amplitudes.coeff(ipulse) == 0) 
            continue;

        int bx = bxs.coeff(ipulse);
        int first_sample_t = std::max(0, bx+3);
        int offset = 7 - 3 - bx;

        auto const value = amplitudes.coeff(ipulse);
        auto const value_sq = value*value;

        unsigned int nsample_pulse = nsamples - first_sample_t;
        inverse_cov.block(first_sample_t, first_sample_t, 
                          nsample_pulse, nsample_pulse)
            += value_sq * full_pulse_cov.block(first_sample_t + offset,
                                               first_sample_t + offset,
                                               nsample_pulse,
                                               nsample_pulse);
    }

    return true;
}

__device__
__forceinline__
SampleVector::Scalar compute_chi2(SampleDecompLLT& covariance_decomposition,
                   PulseMatrixType const& pulse_matrix,
                   SampleVector const& amplitudes,
                   SampleVector const& samples) {
    return covariance_decomposition.matrixL()
        .solve(pulse_matrix * amplitudes - samples)
        .squaredNorm();
}

///
/// launch ctx parameters are (nchannels / block, blocks)
/// TODO: trivial impl for now, there must be a way to improve
///
/// Conventions:
///   - amplitudes -> solution vector, what we are fitting for
///   - samples -> raw detector responses
///   - passive constraint - satisfied constraint
///   - active constraint - unsatisfied (yet) constraint
///
__global__
void kernel_minimize(SampleMatrix const* noisecov,
                     FullSampleMatrix const* full_pulse_cov,
                     BXVectorType *bxs,
                     SampleVector const* samples,
                     SampleVector* amplitudes,
                     PulseMatrixType* pulse_matrix, 
                     ::ecal::reco::StorageScalarType* chi2s,
                     char *acState,
                     int nchannels,
                     int max_iterations) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < nchannels) {
        if (static_cast<MinimizationState>(acState[idx]) == 
            MinimizationState::Precomputed)
            return;

        // inits
        int iter = 0;
        int npassive = 0;
        
        // inits
        SampleDecompLLT covariance_decomposition;
        SampleMatrix inverse_cov;
        SampleVector::Scalar chi2 = 0, chi2_now = 0;

#ifdef ECAL_MULTIFIT_KERNEL_MINIMIZE_V1
//    PRINT_MATRIX_10x10(noisecov[idx]);
#endif

        // loop until ocnverge
        while (true) {
            if (iter >= max_iterations)
                break;

            update_covariance(
                noisecov[idx], 
                full_pulse_cov[idx],
                inverse_cov,
                bxs[idx],
                covariance_decomposition,
                amplitudes[idx]);

            // compute actual covariance decomposition
            covariance_decomposition.compute(inverse_cov);

            // prepare input matrices for fnnls
            SampleMatrix A = covariance_decomposition.matrixL()
                .solve(pulse_matrix[idx]);
            SampleVector b = covariance_decomposition.matrixL()
                .solve(samples[idx]);
            
            inplace_fnnls(
                A, b, amplitudes[idx],
                npassive, bxs[idx], pulse_matrix[idx]);
                
            chi2_now = compute_chi2(
                covariance_decomposition,
                pulse_matrix[idx],
                amplitudes[idx],
                samples[idx]);
            auto deltachi2 = chi2_now - chi2;


#ifdef ECAL_MULTIFIT_KERNEL_MINIMIZE_V1
            if (iter > 10) {
                printf("idx = %d iter = %d chi2 = %f chi2old = %f\n",
                    idx, iter, chi2_now, chi2);
                
                printf("noisecov(0, i): %f %f %f %f %f %f %f %f %f %f\n",
                    noisecov[idx](0, 0),
                    noisecov[idx](0, 1),
                    noisecov[idx](0, 2),
                    noisecov[idx](0, 3),
                    noisecov[idx](0, 4),
                    noisecov[idx](0, 5),
                    noisecov[idx](0, 6),
                    noisecov[idx](0, 7),
                    noisecov[idx](0, 8),
                    noisecov[idx](0, 9));

                printf("ampls: %f %f %f %f %f %f %f %f %f %f\n",
                    amplitudes[idx](0),
                    amplitudes[idx](1),
                    amplitudes[idx](2),
                    amplitudes[idx](3),
                    amplitudes[idx](4),
                    amplitudes[idx](5),
                    amplitudes[idx](6),
                    amplitudes[idx](7),
                    amplitudes[idx](8),
                    amplitudes[idx](9));
            }
#endif

            chi2 = chi2_now;

            if (ecal::abs(deltachi2) < 1e-3)
                break;

            //---- AM: TEST
            //---- it was 3 lines above, now here as in the CPU version
            ++iter;
        }

        // the rest will be set later
        chi2s[idx] = chi2;
    }
}

namespace v1 {

void minimization_procedure(
        EventInputDataCPU const& eventInputCPU, EventInputDataGPU& eventInputGPU,
        EventOutputDataGPU& eventOutputGPU, EventDataForScratchGPU& scratch,
        ConditionsProducts const& conditions,
        ConfigurationParameters const& configParameters,
        cuda::stream_t<>& cudaStream) {
    unsigned int totalChannels = eventInputCPU.ebDigis.size() 
        + eventInputCPU.eeDigis.size();
//    unsigned int threads_min = conf.threads.x;
    // TODO: configure from python
    unsigned int threads_min = configParameters.kernelMinimizeThreads[0];
    unsigned int blocks_min = threads_min > totalChannels
        ? 1
        : (totalChannels + threads_min - 1) / threads_min;
    kernel_minimize<<<blocks_min, threads_min, 0, cudaStream.id()>>>(
        scratch.noisecov,
        scratch.pulse_covariances,
        scratch.activeBXs,
        scratch.samples,
        (SampleVector*)eventOutputGPU.amplitudesAll,
        scratch.pulse_matrix,
        eventOutputGPU.chi2,
        scratch.acState,
        totalChannels,
        50);
    cudaCheck(cudaGetLastError());

    //
    // permute computed amplitudes
    // and assign the final uncalibared energy value
    //
    unsigned int threadsPermute = 32 * EcalDataFrame::MAXSAMPLES; // 32 * 10
    unsigned int blocksPermute = threadsPermute > 10 * totalChannels
        ? 1
        : (10 * totalChannels + threadsPermute - 1) / threadsPermute;
    int bytesPermute = threadsPermute * sizeof(SampleVector::Scalar);
    kernel_permute_results<<<blocksPermute, threadsPermute, 
                             bytesPermute, cudaStream.id()>>>(
        (SampleVector*)eventOutputGPU.amplitudesAll,
        scratch.activeBXs,
        eventOutputGPU.amplitude,
        scratch.acState,
        totalChannels);
    cudaCheck(cudaGetLastError());
}

}

}}
