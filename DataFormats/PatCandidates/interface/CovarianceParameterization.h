#ifndef _DataFormats_PatCandidates_CovarianceParameterization_h_
#define _DataFormats_PatCandidates_CovarianceParameterization_h_
#include <TFile.h>
#include <TH3D.h>
#include <iostream>
#include <unordered_map>
#include <array>
#include <TKey.h>
class CompressionElement {
    public:
      enum Method {float16=0,reduceMantissa=1,logPack=2,tanLogPack=3,zero=4,one=5};
      enum Target {realValue=0,ratioToRef=1,differenceToRef=2};
      CompressionElement():method(zero),target(realValue){}
      CompressionElement(Method m, Target t, int bitsUsed, std::vector<float> p): method(m),target(t),bits(bitsUsed),params(p){}
      Method method;
      Target target;
      int bits;
      std::vector<float> params;
      uint16_t pack(float value, float ref=0.) const ;
      float unpack(uint16_t packed, float ref=0.) const;

};
  

class CovarianceParameterization {
    public:
        static int index(int i, int j) {if(i>=j) return j+i*(i+1)/2; else return i+j*(j+1)/2 ; }
        struct CompressionSchema {
             CompressionSchema() {}
             std::array<CompressionElement,15> elements;
             CompressionElement & operator()(int i,int j) {return elements[index(i,j)];}
             const CompressionElement & operator()(int i,int j) const {return elements[index(i,j)];}
        };
        CovarianceParameterization() : loadedVersion_(-1) 
        {
        }
        bool isValid() const {return loadedVersion_!=-1; }
        int loadedVersion() const {return loadedVersion_; }
        void load(int version);
        float  meanValue(int i,int j,int sign,float pt, float eta, int nHits,int pixelHits,  float cii=1.,float cjj=1.) const ;
        float  pack(float value,int schema, int i,int j,float pt, float eta, int nHits,int pixelHits,  float cii=1.,float cjj=1.) const;
        float  unpack(uint16_t packed,int schema, int i,int j,float pt, float eta, int nHits,int pixelHits,  float cii=1.,float cjj=1.) const;
    private:
        void readFile( TFile &);
        void  addTheHistogram(std::vector<TH3D *> * HistoVector, std::string StringToAddInTheName, int i, int j, TFile & fileToRead);
        int loadedVersion_;
	TFile * fileToRead_;
        std::unordered_map<uint16_t,CompressionSchema> schemas; 
        std::vector<TH3D *>  cov_elements_pixelHit;
        std::vector<TH3D *>  cov_elements_noPixelHit;
};

#endif

