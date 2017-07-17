/*
 * FileIO.cc
 *
 *  Created on: Sep 25, 2012
 *      Author: aspataru
 */

#include "EventFilter/Utilities/interface/FileIO.h"

#include <iostream>
#include <fstream>
#include <streambuf>
#include <cstdlib>
#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <cstring>

using namespace jsoncollector;

void FileIO::writeStringToFile(std::string const& filename, std::string& content)
{
  std::ofstream outputFile;
  outputFile.open(filename.c_str());
  outputFile << content;
  outputFile.close();
}

bool FileIO::readStringFromFile(std::string const& filename, std::string& content)
{
  if (!fileExists(filename))
    return false;

  std::ifstream inputFile(filename.c_str());
  inputFile.seekg(0, std::ios::end);
  content.reserve(inputFile.tellg());
  inputFile.seekg(0, std::ios::beg);
  content.assign((std::istreambuf_iterator<char>(inputFile)), std::istreambuf_iterator<char>());
  inputFile.close();
  return true;
}

bool FileIO::fileExists(std::string const& path)
{
  std::ifstream ifile(path.c_str());
  return !ifile.fail();
}

