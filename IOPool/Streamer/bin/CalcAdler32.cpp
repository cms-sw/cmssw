#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "IOPool/Streamer/interface/MsgTools.h"

#include <memory>

#include <fstream>
#include <iostream>
#include <cstdint>

int main(int argc, char* argv[]) {
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
  uint32_t a = 1, b = 0;

  std::ifstream::pos_type rsize = 0;
  while (rsize < size) {
    file.read(ptr.get(), 1024 * 1024);
    rsize += 1024 * 1024;
    //std::cout << file.tellg() << " " << rsize << " " << size << " - " << file.gcount() << std::endl;
    if (file.gcount()) {
      cms::Adler32(ptr.get(), file.gcount(), a, b);
    } else {
      break;
    }
  }

  uint32 adler = (b << 16) | a;
  std::cout << std::hex << adler << std::dec << " " << filename << std::endl;

  file.close();
  return 0;
}
