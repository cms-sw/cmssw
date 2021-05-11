#ifndef _ANLZ4CMSSW_H_
#define _ANLZ4CMSSW_H_
#include <vector>
#include <cstdint>
#include <cstdlib>

void anlz4cmssw_init();
void anlz4cmssw_load_trained_model(std::vector<std::uint8_t> &model);
std::size_t anlz4cmssw_compress(std::vector<std::vector<std::uint8_t>> &data, std::vector<std::uint8_t> &compressed);
std::size_t anlz4cmssw_decompress(std::vector<std::uint8_t> &compressed, std::vector<std::vector<double>> &data);
#endif