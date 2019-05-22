#include "LzmaFile.interface.h"
#include "LzmaFile.h"

#include <iostream>
using namespace std;

LzmaFile global_LzmaFile;

extern "C" {
void lzmaopenfile_(char* name) { global_LzmaFile.Open(name); }
}

extern "C" {
void lzmafillarray_(double* data, const int& length) { global_LzmaFile.FillArray(data, length); }
}

extern "C" {
int lzmanextnumber_(double& data) { return global_LzmaFile.ReadNextNumber(data); }
}

extern "C" {
void lzmaclosefile_() { global_LzmaFile.Close(); }
}
