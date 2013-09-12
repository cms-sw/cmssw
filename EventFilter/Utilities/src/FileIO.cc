/*
 * FileIO.cc
 *
 *  Created on: Sep 25, 2012
 *      Author: aspataru
 */

#include "../interface/FileIO.h"

#include <iostream>
#include <fstream>
#include <streambuf>
#include <cstdlib>
#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <cstring>

using namespace jsoncollector;
using std::string;
using std::ofstream;
using std::vector;
using std::ifstream;
using std::strlen;

void FileIO::writeStringToFile(string& filename, string& content) {
	ofstream outputFile;
	outputFile.open(filename.c_str());
	outputFile << content;
	outputFile.close();
}

bool FileIO::readStringFromFile(string& filename, string& content) {
	if (!fileExists(filename))
		return false;

	std::ifstream inputFile(filename.c_str());

	inputFile.seekg(0, std::ios::end);
	content.reserve(inputFile.tellg());
	inputFile.seekg(0, std::ios::beg);

	content.assign((std::istreambuf_iterator<char>(inputFile)),
			std::istreambuf_iterator<char>());

	inputFile.close();
	return true;
}

bool FileIO::fileExists(string& path) {
	ifstream ifile(path.c_str());
	return ifile;
}
