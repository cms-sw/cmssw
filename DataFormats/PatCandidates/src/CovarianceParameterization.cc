//#include <iostream>

#include <fmt/printf.h>

#include <TParameter.h>
#include <TVector.h>
#include <TFolder.h>

#include "DataFormats/Math/interface/liblogintpack.h"
#include "DataFormats/Math/interface/libminifloat.h"
#include "DataFormats/PatCandidates/interface/CovarianceParameterization.h"
#include "FWCore/Utilities/interface/FileInPath.h"

uint16_t CompressionElement::pack(float value, float ref) const {
  float toCompress = 0;
  switch (target) {
    case (realValue):
      toCompress = value;
      break;
    case (ratioToRef):
      toCompress = value / ref;
      break;
    case (differenceToRef):
      toCompress = value - ref;
      break;
  }
  switch (method) {
    case (float16):
      return MiniFloatConverter::float32to16(toCompress * params[0]);
      break;
    case (reduceMantissa):
      return MiniFloatConverter::reduceMantissaToNbits(toCompress, params[0]);
      break;
    case (zero):
      return 0;
      break;
    case (one):
      return 1.0;
      break;
    case (tanLogPack):
      return 0;  //FIXME: should be implemented
      break;
    case (logPack):
      int16_t r = logintpack::pack16log(toCompress, params[0], params[1], bits);
      return *reinterpret_cast<uint16_t *>(&r);
      break;
  }
  return 0;
}
float CompressionElement::unpack(uint16_t packed, float ref) const {
  float unpacked = 0;
  switch (method) {
    case (float16):
      unpacked = MiniFloatConverter::float16to32(packed) / params[0];
      break;
    case (reduceMantissa):
      unpacked = packed;
      break;
    case (logPack):
      unpacked = logintpack::unpack16log(*reinterpret_cast<int16_t *>(&packed), params[0], params[1], bits);
      break;
    case (zero):
      unpacked = 0;
      break;
    case (one):
    case (tanLogPack):
      unpacked = 1;  //FIXME: should be implemented
  }
  switch (target) {
    case (realValue):
      return unpacked;
    case (ratioToRef):
      return unpacked * ref;
    case (differenceToRef):
      return unpacked + ref;
  }

  return ref;
}

void CovarianceParameterization::load(int version) {
  edm::FileInPath fip(
      fmt::sprintf("DataFormats/PatCandidates/data/CovarianceParameterization_version%d.root", version));
  fileToRead_ = TFile::Open(fip.fullPath().c_str());
  TFile &fileToRead = *fileToRead_;
  //Read files from here fip.fullPath().c_str();
  if (fileToRead.IsOpen()) {
    readFile(fileToRead);

    TIter next(((TDirectoryFile *)fileToRead.Get("schemas"))->GetListOfKeys());
    TKey *key;
    while ((key = (TKey *)next())) {
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (!cl->InheritsFrom("TDirectoryFile"))
        continue;
      std::string schemaNumber = key->ReadObj()->GetName();
      uint16_t schemaN = std::stoi(schemaNumber);
      //for (int folderNumber = 0; folderNumber < 6 ; folderNumber++) {
      CompressionSchema schema;
      for (int i = 0; i < 5; i++) {
        for (int j = i; j < 5; j++) {  //FILLING ONLY THE SCHEMA OF SOME ELEMENTS
          std::string folder = "schemas/" + schemaNumber + "/" + char(48 + i) + char(48 + j);
          std::string methodString = folder + "/method";
          std::string targetString = folder + "/target";
          std::string bitString = folder + "/bit";
          std::vector<float> vParams;
          TVector *p = (TVector *)fileToRead.Get((folder + "/param").c_str());
          vParams.reserve(p->GetNoElements());
          for (int k = 0; k < p->GetNoElements(); k++) {
            vParams.push_back((*p)[k]);
          }

          schema(i, j) = CompressionElement(
              (CompressionElement::Method)((TParameter<int> *)fileToRead.Get(methodString.c_str()))->GetVal(),
              (CompressionElement::Target)((TParameter<int> *)fileToRead.Get(targetString.c_str()))->GetVal(),
              (int)((TParameter<int> *)fileToRead.Get(bitString.c_str()))->GetVal(),
              vParams);
        }
      }
      schemas[schemaN] = schema;
    }

    loadedVersion_ = version;
  } else {
    loadedVersion_ = -1;
  }
}

void CovarianceParameterization::readFile(TFile &f) {
  for (int i = 0; i < 5; i++) {
    for (int j = i; j < 5; j++) {
      std::string String_first_positive = "_pixel_";
      std::string String_second_positive = "_noPixel_";

      addTheHistogram(&cov_elements_pixelHit, String_first_positive, i, j, f);
      addTheHistogram(&cov_elements_noPixelHit, String_second_positive, i, j, f);
    }
  }
}

void CovarianceParameterization::addTheHistogram(
    std::vector<TH3D *> *HistoVector, std::string StringToAddInTheName, int i, int j, TFile &fileToRead) {
  std::string List_covName[5] = {"qoverp", "lambda", "phi", "dxy", "dsz"};

  std::string histoNameString = "covariance_" + List_covName[i] + "_" + List_covName[j] + StringToAddInTheName +
                                "parametrization";  // + "_entries";
  TH3D *matrixElememtHistogramm = (TH3D *)fileToRead.Get(histoNameString.c_str());
  HistoVector->push_back(matrixElememtHistogramm);
}

float CovarianceParameterization::meanValue(
    int i, int j, int sign, float pt, float eta, int nHits, int pixelHits, float cii, float cjj) const {
  int hitNumberToUse = nHits;
  if (hitNumberToUse < 2)
    hitNumberToUse = 2;
  if (hitNumberToUse > 32)
    hitNumberToUse = 32;
  int ptBin = cov_elements_pixelHit[0]->GetXaxis()->FindBin(pt);
  int etaBin = cov_elements_pixelHit[0]->GetYaxis()->FindBin(std::abs(eta));
  int hitBin = cov_elements_pixelHit[0]->GetZaxis()->FindBin(hitNumberToUse);
  int min_idx = i;
  int max_idx = j;

  if (i > j) {
    min_idx = j;
    max_idx = i;
  }

  int indexOfTheHitogramInTheList = ((9 - min_idx) * min_idx) / 2 + max_idx;

  double meanValue = 0.;
  if (pixelHits > 0) {
    meanValue = sign * cov_elements_pixelHit[indexOfTheHitogramInTheList]->GetBinContent(ptBin, etaBin, hitBin);
  } else {
    meanValue = sign * cov_elements_noPixelHit[indexOfTheHitogramInTheList]->GetBinContent(ptBin, etaBin, hitBin);
  }
  return meanValue;
}

float CovarianceParameterization::pack(
    float value, int schema, int i, int j, float pt, float eta, int nHits, int pixelHits, float cii, float cjj) const {
  if (i > j)
    std::swap(i, j);
  float ref = meanValue(i, j, 1., pt, eta, nHits, pixelHits, cii, cjj);
  if (ref == 0) {
    schema = 0;
  }
  if (schema == 0 && i == j && (i == 2 || i == 0))
    ref = 1. / (pt * pt);
  /*  //Used for debugging, to be later removed  
    uint16_t p=(*schemas.find(schema)).second(i,j).pack(value,ref);
    float up=(*schemas.find(schema)).second(i,j).unpack(p,ref);
    std::cout << "check " << i << " " << j << " " << value << " " << up << " " << p << " " << ref << " " << schema<< std::endl;*/
  return (*schemas.find(schema)).second(i, j).pack(value, ref);
}
float CovarianceParameterization::unpack(
    uint16_t packed, int schema, int i, int j, float pt, float eta, int nHits, int pixelHits, float cii, float cjj)
    const {
  if (i > j)
    std::swap(i, j);
  float ref = meanValue(i, j, 1., pt, eta, nHits, pixelHits, cii, cjj);
  if (ref == 0) {
    schema = 0;
  }
  if (schema == 0 && i == j && (i == 2 || i == 0))
    ref = 1. / (pt * pt);
  if (i == j && (*schemas.find(schema)).second(i, j).unpack(packed, ref) == 0)
    return 1e-9;
  else
    return (*schemas.find(schema)).second(i, j).unpack(packed, ref);
}
