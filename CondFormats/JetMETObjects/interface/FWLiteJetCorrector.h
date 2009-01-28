#ifndef FWLITEJETCORRECTOR_H
#define FWLITEJETCORRECTOR_H
#include <vector>
#include <string>
using namespace std;

class FWLiteJetCorrector
{
  public:
    FWLiteJetCorrector();
    FWLiteJetCorrector(std::vector<std::string> Levels, std::vector<std::string> CorrectionTags);
    FWLiteJetCorrector(std::vector<std::string> Levels, std::vector<std::string> CorrectionTags, std::string FlavorOption, std::string PartonOption);
    double getCorrection(double pt, double eta);
    double getCorrection(double pt, double eta, double emf);
    std::vector<double> getSubCorrections(double pt, double eta);
    vector<double> getSubCorrections(double pt, double eta, double emf);
    ~FWLiteJetCorrector();
    
  private:
    std::vector<std::string> mLevels;
    std::vector<std::string> mDataFiles;
    std::string mFlavorOption;
    std::string mPartonOption;  
};
#endif
