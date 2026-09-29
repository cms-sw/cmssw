#include <ap_int.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "L1Trigger/L1CaloTrigger/interface/GCTSum_h.h"
#include "L1Trigger/L1CaloTrigger/interface/bitonicSort32_GCT_h.h"
#include "L1Trigger/L1CaloTrigger/interface/bitonicSort32_GCT_cpp.h"
#include "L1Trigger/L1CaloTrigger/interface/GCTSum_cpp.h"
#include "L1Trigger/L1CaloTrigger/interface/GCTSumToGT_h.h"
#include "L1Trigger/L1CaloTrigger/interface/GCTSumToGT_cpp.h"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

ap_uint<576> sourceSums(
    int ex, int ey, int htx, int hty, unsigned int sumEt, unsigned int ht, unsigned int count) {
  ap_uint<576> out = 0;
  out.range(11, 0) = ap_int<12>(ex);
  out.range(23, 12) = ap_int<12>(ey);
  out.range(35, 24) = ap_int<12>(htx);
  out.range(47, 36) = ap_int<12>(hty);
  out.range(75, 64) = ap_uint<12>(sumEt);
  out.range(87, 76) = ap_uint<12>(ht);
  out.range(99, 88) = ap_uint<12>(count);
  return out;
}

ap_uint<576> sideSums(
    int ex, int ey, int htx, int hty, unsigned int sumEt, unsigned int ht, unsigned int count) {
  ap_uint<576> out = 0;
  out.range(15, 0) = ap_int<16>(ex);
  out.range(31, 16) = ap_int<16>(ey);
  out.range(47, 32) = ap_int<16>(htx);
  out.range(63, 48) = ap_int<16>(hty);
  out.range(79, 64) = ap_uint<16>(sumEt);
  out.range(95, 80) = ap_uint<16>(ht);
  out.range(111, 96) = ap_uint<16>(count);
  return out;
}

std::uint64_t outputWord(const ap_uint<576>& link, unsigned int word) {
  return link.range(word * 64 + 63, word * 64).to_uint64();
}

std::uint64_t referenceVectorSum(int x, int y, unsigned int scalar) {
  const double pi = std::acos(-1.0);
  const auto magnitude = static_cast<unsigned int>(
      std::min(65535.0, std::floor(std::hypot(static_cast<double>(x), static_cast<double>(y)) * 16.0)));
  const double angle = (x == 0 && y == 0) ? 0.0 : std::atan2(-static_cast<double>(y), -static_cast<double>(x));
  const int phi = static_cast<int>(std::round(angle * 4096.0 / pi));
  const unsigned int scaledScalar = std::min(65535U, scalar * 16U);
  return 1ULL | (std::uint64_t(magnitude) << 1) | (std::uint64_t(phi & 8191) << 17) |
         (std::uint64_t(scaledScalar) << 30);
}

void testSideReduction() {
  std::array<ap_uint<576>, p2gctsum::N_INPUT_LINKS> input{};
  std::array<ap_uint<576>, p2gctsum::N_OUTPUT_LINKS> output{};

  input[2] = sourceSums(1, -1, -3, 7, 20, 10, 1);
  input[5] = sourceSums(2, 3, 4, -2, 21, 11, 2);
  input[8] = sourceSums(4, 5, 8, 1, 22, 12, 3);
  input[11] = sourceSums(6, 7, -2, 3, 23, 13, 4);

  p2gctsum::algo_top(input.data(), output.data());

  require(output[0] == 0, "zero gamma inputs must remain invalid zero words");
  require(output[1] == 0, "zero hadron inputs must remain invalid zero words");
  require(output[2] == sideSums(13, 14, 7, 9, 86, 46, 10), "SUM ABI v2 side reduction is incorrect");
}

void testGtPacketSums() {
  std::array<ap_uint<576>, p2gctsumGT::N_INPUT_LINKS_GT> input{};
  std::array<ap_uint<576>, p2gctsumGT::N_OUTPUT_LINKS_GT> output{};

  input[2] = sideSums(5, -2, 8, -3, 40, 30, 3);
  input[5] = sideSums(-1, 4, -2, -5, 60, 50, 5);
  p2gctsumGT::algo_top_GT(input.data(), output.data());

  for (unsigned int link = 0; link < 5; ++link)
    require(output[link] == 0, "empty object links must remain invalid zero words");
  require(outputWord(output[5], 3) == referenceVectorSum(4, 2, 100), "GT MET + scalar ET word is incorrect");
  require(outputWord(output[5], 4) == referenceVectorSum(6, -8, 80), "GT MHT + scalar HT word is incorrect");
  require(outputWord(output[5], 5) == (1ULL | (8ULL << 1)), "GT object-count word is incorrect");
  require(outputWord(output[5], 6) == 0, "GT reserved sum word must be zero");
  require(outputWord(output[5], 7) == 0 && outputWord(output[5], 8) == 0,
          "GT reserved tail words must be zero");
}

void testExtendedEtaAndZeroValidity() {
  p2gctsum::GCTvar gamma;
  gamma.ET = 23;
  gamma.Eta = 212;
  gamma.Phi = 10;
  gamma.isBarrel = 0;
  const ap_uint<48> packed = p2gctsum::pack_gamma_output(gamma);
  require(packed.range(18, 12) == 84 && packed[28] == 1, "extended endcap eta was not preserved");

  p2gctsumGT::GTInputVar input;
  input.unpack(packed, false);
  require(input.Eta == 212 && input.Phi == 10, "TO_GT did not recover the extended eta without corrupting phi");

  p2gctsumGT::GTvar output;
  output.convertAndPack(input);
  require(output.isValid == 1 && output.Eta == ap_uint<14>(212 * 23), "extended eta conversion is incorrect");

  gamma.ET = 0;
  gamma.Eta = 212;
  gamma.Phi = 10;
  require(p2gctsum::pack_gamma_output(gamma) == 0, "zero-ET gamma must be an invalid zero word");
}

}  // namespace

#ifdef GCTSUM_SYSTEMC_AP_INT
int sc_main(int, char**) {
#else
int main() {
#endif
  try {
    testSideReduction();
    testGtPacketSums();
    testExtendedEtaAndZeroValidity();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "PASS: GCT Sum clean emulator matches sums ABI v2 and the six-link GT packet\n";
  return 0;
}
