#include <iostream>

#include <filesystem>
#include <fstream>

int main(int argc, char** argv) {
  if (argc != 3) {
    return 0;
  }

  std::string directoryName = argv[1];

  std::string baseName = argv[2];

  const std::filesystem::path readDirectory{directoryName};

  if (not is_directory(readDirectory)) {
    std::cerr << "Error: " << readDirectory << " is not a directory";
    return 1;
  }

  std::ofstream mergedFile(baseName + ".bin", std::ios::binary);
  std::ofstream indexFile(baseName + ".index");

  unsigned long int offset = 0;
  for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{readDirectory}) {
    if (dir_entry.is_regular_file()) {
      std::string fileName = relative(dir_entry.path(), readDirectory);
      //std::cout <<" "<<relative(dir_entry.path(), readDirectory)<<"  "<<offset<<"\n";
      indexFile << fileName << " " << offset << "\n";
      offset += dir_entry.file_size();

      std::ifstream toRead(dir_entry.path(), std::ios::binary);

      std::array<char, 1024> buffer;
      while (toRead.read(buffer.data(), buffer.size())) {
        mergedFile.write(buffer.data(), buffer.size());
      }
      if (toRead.bad()) {
        std::cerr << " problem copying file " << dir_entry.path();
        return 1;
      }
      if (not toRead.eof()) {
        std::cerr << " failed to read end of file " << dir_entry.path();
        return 1;
      }
      int lastRead = toRead.gcount();
      if (lastRead != 1024) {
        mergedFile.write(buffer.data(), lastRead);
      }
    }
  }
  return 0;
}
