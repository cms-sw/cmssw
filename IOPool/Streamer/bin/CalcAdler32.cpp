#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "IOPool/Streamer/interface/MsgTools.h"

#include <memory>

#include <fstream>
#include <iostream>
#include <cstdint>

int main(int argc, char* argv[]) {
  using namespace edm::streamer;

  if (argc < 2) {
    std::cerr << "No command line argument given, expected path/filename.\n";
    return 1;
  }

  std::string filename(argv[1]);
  std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "File " << filename << " could not be opened.\n";
    return 1;
  }

  std::ifstream::pos_type size = file.tellg();
  file.seekg(0, std::ios::beg);

  auto ptr = std::make_unique<char[]>(1024 * 1024);
  uint32_t adler = cms::adler32_init;

  std::ifstream::pos_type rsize = 0;
  while (rsize < size) {
    file.read(ptr.get(), 1024 * 1024);
    rsize += 1024 * 1024;
    //std::cout << file.tellg() << " " << rsize << " " << size << " - " << file.gcount() << std::endl;
    if (file.gcount()) {
      adler = cms::adler32(ptr.get(), file.gcount(), adler);
    } else {
      break;
    }
  }

  std::cout << std::hex << adler << std::dec << " " << filename << std::endl;

  file.close();
  return 0;
}
