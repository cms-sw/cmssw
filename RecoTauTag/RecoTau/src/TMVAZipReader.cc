#include "RecoTauTag/RecoTau/interface/TMVAZipReader.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include "zlib.h"

using namespace std;

// From http://stackoverflow.com/questions/874134/find-if-string-endswith-another-string-in-c
bool reco::details::hasEnding(std::string const &fullString, std::string const &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(
          fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

char* reco::details::readGzipFile(const std::string& weightFile)
{
  FILE *f = fopen(weightFile.c_str(), "r");
  int magic;
  int size;
  fread(&magic, 4, 1, f);
  fseek(f, -4, SEEK_END);
  fread(&size, 4, 1, f);
  fclose(f);
  //printf("%x, %i\n", magic, size);

  gzFile  file = gzopen (weightFile.c_str(), "r");

  int bytes_read;
  char *buffer = (char*)malloc(size);
  bytes_read = gzread (file, buffer, size - 1);
  buffer[bytes_read] = '\0';
  if (!gzeof (file)) {
    int err;
    const char * error_string;
    error_string = gzerror (file, & err);
    if (err) {
      cout<<"Error while reading gzipped file:  "<<error_string;
    }
  }
  gzclose (file);
  return buffer;
}

void reco::details::loadTMVAWeights(TMVA::Reader* reader, const std::string& method,
    const std::string& weightFile, bool verbose) {
  verbose = false;
  if (verbose)
    std::cout << "Booking TMVA Reader with " << method << " and weight file: " << weightFile
      << std::endl;

  if (reco::details::hasEnding(weightFile, ".xml")) {
    if (verbose)
      std::cout << "Weight file is pure xml." << std::endl;
    // Let TMVA read the file
    reader->BookMVA(method, weightFile);
  } else if (reco::details::hasEnding(weightFile, ".gz") || reco::details::hasEnding(weightFile, ".gzip")) {
    if (verbose)
      std::cout << "Unzipping file." << std::endl;
    char* c = readGzipFile(weightFile);

    // We can't use tmpnam, gcc emits a warning about security.
    // This is also technically insecure in the same way, since we append
    // a suffix and then open another file.
    char tmpFilename[] = "/tmp/tmva.XXXXXX";
    int fdToUselessFile = mkstemp(tmpFilename);
    std::string weight_file_name(tmpFilename);
    weight_file_name += ".xml";
    FILE *theActualFile = fopen(weight_file_name.c_str(), "w");
    // write xml
    fputs(c, theActualFile);
    fputs("\n", theActualFile);
    fclose(theActualFile);
    close(fdToUselessFile);
    if (verbose)
      std::cout << "Booking MvA" << std::endl;
    reader->BookMVA(method, weight_file_name);
    if (verbose)
      std::cout << "Cleaning up" << std::endl;
    remove(weight_file_name.c_str());
    remove(tmpFilename);

    // Someday this will work.
    //reader->BookMVA(TMVA::Types::Instance().GetMethodType(TString(method)), c);
    if (verbose) {
      std::cout << "Reader booked" << std::endl;
    }
    delete c;
  } else {
    throw cms::Exception("BadTMVAWeightFilename")
      << "I don't understand the extension on the filename: "
      << weightFile << ", it should be .xml, .gz, or .gzip" << std::endl;
  }
}
