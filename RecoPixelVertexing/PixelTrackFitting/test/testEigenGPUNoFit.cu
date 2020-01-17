#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "test_common.h"

using namespace Eigen;

using Matrix5d = Matrix<double, 5, 5>;

__host__ __device__ void eigenValues(Matrix3d *m, Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret) {
#if TEST_DEBUG
  printf("Matrix(0,0): %f\n", (*m)(0, 0));
  printf("Matrix(1,1): %f\n", (*m)(1, 1));
  printf("Matrix(2,2): %f\n", (*m)(2, 2));
#endif
  SelfAdjointEigenSolver<Matrix3d> es;
  es.computeDirect(*m);
  (*ret) = es.eigenvalues();
  return;
}

__global__ void kernel(Matrix3d *m, Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret) {
  eigenValues(m, ret);
}

__global__ void kernelInverse3x3(Matrix3d *in, Matrix3d *out) { (*out) = in->inverse(); }

__global__ void kernelInverse4x4(Matrix4d *in, Matrix4d *out) { (*out) = in->inverse(); }

__global__ void kernelInverse5x5(Matrix5d *in, Matrix5d *out) { (*out) = in->inverse(); }

template <typename M1, typename M2, typename M3>
__global__ void kernelMultiply(M1 *J, M2 *C, M3 *result) {
//  Map<M3> res(result->data());
#if TEST_DEBUG
  printf("*** GPU IN ***\n");
#endif
  printIt(J);
  printIt(C);
  //  res.noalias() = (*J) * (*C);
  //  printIt(&res);
  (*result) = (*J) * (*C);
#if TEST_DEBUG
  printf("*** GPU OUT ***\n");
#endif
  return;
}

template <int row1, int col1, int row2, int col2>
void testMultiply() {
  std::cout << "TEST MULTIPLY" << std::endl;
  std::cout << "Product of type " << row1 << "x" << col1 << " * " << row2 << "x" << col2 << std::endl;
  Eigen::Matrix<double, row1, col1> J;
  fillMatrix(J);
  Eigen::Matrix<double, row2, col2> C;
  fillMatrix(C);
  Eigen::Matrix<double, row1, col2> multiply_result = J * C;
#if TEST_DEBUG
  std::cout << "Input J:" << std::endl;
  printIt(&J);
  std::cout << "Input C:" << std::endl;
  printIt(&C);
  std::cout << "Output:" << std::endl;
  printIt(&multiply_result);
#endif
  // GPU
  Eigen::Matrix<double, row1, col1> *JGPU = nullptr;
  Eigen::Matrix<double, row2, col2> *CGPU = nullptr;
  Eigen::Matrix<double, row1, col2> *multiply_resultGPU = nullptr;
  Eigen::Matrix<double, row1, col2> *multiply_resultGPUret = new Eigen::Matrix<double, row1, col2>();

  cudaCheck(cudaMalloc((void **)&JGPU, sizeof(Eigen::Matrix<double, row1, col1>)));
  cudaCheck(cudaMalloc((void **)&CGPU, sizeof(Eigen::Matrix<double, row2, col2>)));
  cudaCheck(cudaMalloc((void **)&multiply_resultGPU, sizeof(Eigen::Matrix<double, row1, col2>)));
  cudaCheck(cudaMemcpy(JGPU, &J, sizeof(Eigen::Matrix<double, row1, col1>), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(CGPU, &C, sizeof(Eigen::Matrix<double, row2, col2>), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
      multiply_resultGPU, &multiply_result, sizeof(Eigen::Matrix<double, row1, col2>), cudaMemcpyHostToDevice));

  kernelMultiply<<<1, 1>>>(JGPU, CGPU, multiply_resultGPU);
  cudaDeviceSynchronize();

  cudaCheck(cudaMemcpy(
      multiply_resultGPUret, multiply_resultGPU, sizeof(Eigen::Matrix<double, row1, col2>), cudaMemcpyDeviceToHost));
  printIt(multiply_resultGPUret);
  assert(isEqualFuzzy(multiply_result, (*multiply_resultGPUret)));
}

void testInverse3x3() {
  std::cout << "TEST INVERSE 3x3" << std::endl;
  Matrix3d m;
  fillMatrix(m);
  m += m.transpose().eval();

  Matrix3d m_inv = m.inverse();
  Matrix3d *mGPU = nullptr;
  Matrix3d *mGPUret = nullptr;
  Matrix3d *mCPUret = new Matrix3d();

#if TEST_DEBUG
  std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
  std::cout << "Its inverse is:" << std::endl << m.inverse() << std::endl;
#endif
  cudaCheck(cudaMalloc((void **)&mGPU, sizeof(Matrix3d)));
  cudaCheck(cudaMalloc((void **)&mGPUret, sizeof(Matrix3d)));
  cudaCheck(cudaMemcpy(mGPU, &m, sizeof(Matrix3d), cudaMemcpyHostToDevice));

  kernelInverse3x3<<<1, 1>>>(mGPU, mGPUret);
  cudaDeviceSynchronize();

  cudaCheck(cudaMemcpy(mCPUret, mGPUret, sizeof(Matrix3d), cudaMemcpyDeviceToHost));
#if TEST_DEBUG
  std::cout << "Its GPU inverse is:" << std::endl << (*mCPUret) << std::endl;
#endif
  assert(isEqualFuzzy(m_inv, *mCPUret));
}

void testInverse4x4() {
  std::cout << "TEST INVERSE 4x4" << std::endl;
  Matrix4d m;
  fillMatrix(m);
  m += m.transpose().eval();

  Matrix4d m_inv = m.inverse();
  Matrix4d *mGPU = nullptr;
  Matrix4d *mGPUret = nullptr;
  Matrix4d *mCPUret = new Matrix4d();

#if TEST_DEBUG
  std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
  std::cout << "Its inverse is:" << std::endl << m.inverse() << std::endl;
#endif
  cudaCheck(cudaMalloc((void **)&mGPU, sizeof(Matrix4d)));
  cudaCheck(cudaMalloc((void **)&mGPUret, sizeof(Matrix4d)));
  cudaCheck(cudaMemcpy(mGPU, &m, sizeof(Matrix4d), cudaMemcpyHostToDevice));

  kernelInverse4x4<<<1, 1>>>(mGPU, mGPUret);
  cudaDeviceSynchronize();

  cudaCheck(cudaMemcpy(mCPUret, mGPUret, sizeof(Matrix4d), cudaMemcpyDeviceToHost));
#if TEST_DEBUG
  std::cout << "Its GPU inverse is:" << std::endl << (*mCPUret) << std::endl;
#endif
  assert(isEqualFuzzy(m_inv, *mCPUret));
}

void testInverse5x5() {
  std::cout << "TEST INVERSE 5x5" << std::endl;
  Matrix5d m;
  fillMatrix(m);
  m += m.transpose().eval();

  Matrix5d m_inv = m.inverse();
  Matrix5d *mGPU = nullptr;
  Matrix5d *mGPUret = nullptr;
  Matrix5d *mCPUret = new Matrix5d();

#if TEST_DEBUG
  std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
  std::cout << "Its inverse is:" << std::endl << m.inverse() << std::endl;
#endif
  cudaCheck(cudaMalloc((void **)&mGPU, sizeof(Matrix5d)));
  cudaCheck(cudaMalloc((void **)&mGPUret, sizeof(Matrix5d)));
  cudaCheck(cudaMemcpy(mGPU, &m, sizeof(Matrix5d), cudaMemcpyHostToDevice));

  kernelInverse5x5<<<1, 1>>>(mGPU, mGPUret);
  cudaDeviceSynchronize();

  cudaCheck(cudaMemcpy(mCPUret, mGPUret, sizeof(Matrix5d), cudaMemcpyDeviceToHost));
#if TEST_DEBUG
  std::cout << "Its GPU inverse is:" << std::endl << (*mCPUret) << std::endl;
#endif
  assert(isEqualFuzzy(m_inv, *mCPUret));
}

void testEigenvalues() {
  std::cout << "TEST EIGENVALUES" << std::endl;
  Matrix3d m;
  fillMatrix(m);
  m += m.transpose().eval();

  Matrix3d *m_gpu = nullptr;
  Matrix3d *mgpudebug = new Matrix3d();
  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret =
      new Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType;
  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret1 =
      new Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType;
  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret_gpu = nullptr;
  eigenValues(&m, ret);
#if TEST_DEBUG
  std::cout << "Generated Matrix M 3x3:\n" << m << std::endl;
  std::cout << "The eigenvalues of M are:" << std::endl << (*ret) << std::endl;
  std::cout << "*************************\n\n" << std::endl;
#endif
  cudaCheck(cudaMalloc((void **)&m_gpu, sizeof(Matrix3d)));
  cudaCheck(cudaMalloc((void **)&ret_gpu, sizeof(Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType)));
  cudaCheck(cudaMemcpy(m_gpu, &m, sizeof(Matrix3d), cudaMemcpyHostToDevice));

  kernel<<<1, 1>>>(m_gpu, ret_gpu);
  cudaDeviceSynchronize();

  cudaCheck(cudaMemcpy(mgpudebug, m_gpu, sizeof(Matrix3d), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(
      ret1, ret_gpu, sizeof(Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType), cudaMemcpyDeviceToHost));
#if TEST_DEBUG
  std::cout << "GPU Generated Matrix M 3x3:\n" << (*mgpudebug) << std::endl;
  std::cout << "GPU The eigenvalues of M are:" << std::endl << (*ret1) << std::endl;
  std::cout << "*************************\n\n" << std::endl;
#endif
  assert(isEqualFuzzy(*ret, *ret1));
}

int main(int argc, char *argv[]) {
  cms::cudatest::requireDevices();

  testEigenvalues();
  testInverse3x3();
  testInverse4x4();
  testInverse5x5();

  testMultiply<1, 2, 2, 1>();
  testMultiply<1, 2, 2, 2>();
  testMultiply<1, 2, 2, 3>();
  testMultiply<1, 2, 2, 4>();
  testMultiply<1, 2, 2, 5>();
  testMultiply<2, 1, 1, 2>();
  testMultiply<2, 1, 1, 3>();
  testMultiply<2, 1, 1, 4>();
  testMultiply<2, 1, 1, 5>();
  testMultiply<2, 2, 2, 2>();
  testMultiply<2, 3, 3, 1>();
  testMultiply<2, 3, 3, 2>();
  testMultiply<2, 3, 3, 4>();
  testMultiply<2, 3, 3, 5>();
  testMultiply<3, 2, 2, 3>();
  testMultiply<2, 3, 3, 3>();  // DOES NOT COMPILE W/O PATCHING EIGEN
  testMultiply<3, 3, 3, 3>();
  testMultiply<8, 8, 8, 8>();
  testMultiply<3, 4, 4, 3>();
  testMultiply<2, 4, 4, 2>();
  testMultiply<3, 4, 4, 2>();  // DOES NOT COMPILE W/O PATCHING EIGEN

  return 0;
}
