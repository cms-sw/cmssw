#ifndef RecoLocalCalo_EcalRecAlgos_EigenMatrixTypes_h
#define RecoLocalCalo_EcalRecAlgos_EigenMatrixTypes_h

#include <Eigen/Dense>
#include <array>

constexpr int SampleVectorSize = 10;
constexpr int FullSampleVectorSize = 19;
constexpr int PulseVectorSize = 12;
constexpr int NGains = 3;

typedef Eigen::Matrix<double, SampleVectorSize, 1> SampleVector;
typedef Eigen::Matrix<double, FullSampleVectorSize, 1> FullSampleVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, 0, PulseVectorSize, 1> PulseVector;
typedef Eigen::Matrix<char, Eigen::Dynamic, 1, 0, PulseVectorSize, 1> BXVector;
typedef Eigen::Matrix<char, SampleVectorSize, 1> SampleGainVector;
typedef Eigen::Matrix<double, SampleVectorSize, SampleVectorSize> SampleMatrix;
typedef Eigen::Matrix<double, FullSampleVectorSize, FullSampleVectorSize> FullSampleMatrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, PulseVectorSize, PulseVectorSize> PulseMatrix;
typedef Eigen::Matrix<double, SampleVectorSize, Eigen::Dynamic, 0, SampleVectorSize, PulseVectorSize> SamplePulseMatrix;
typedef Eigen::LLT<SampleMatrix> SampleDecompLLT;
typedef Eigen::LLT<PulseMatrix> PulseDecompLLT;
typedef Eigen::LDLT<PulseMatrix> PulseDecompLDLT;

typedef Eigen::Matrix<double, 1, 1> SingleMatrix;
typedef Eigen::Matrix<double, 1, 1> SingleVector;

typedef std::array<SampleMatrix, NGains> SampleMatrixGainArray;

#endif
