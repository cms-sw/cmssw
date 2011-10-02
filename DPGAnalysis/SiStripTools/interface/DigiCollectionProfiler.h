#ifndef DPGAnalysis_SiStripTools_DigiCollectionProfile_H
#define DPGAnalysis_SiStripTools_DigiCollectionProfile_H

#include <vector>

class TProfile;
class TH2F;
class DetId;

class SiStripDigi;

class DigiCollectionProfiler {

 public:
  DigiCollectionProfiler();
  DigiCollectionProfiler(TProfile* tibprof,
			 TProfile* tobprof,
			 TProfile* tecpprof,
			 TProfile* tecmprof,
			 TH2F* tib2d,
			 TH2F* tob2d,
			 TH2F* tecp2d,
			 TH2F* tecm2d);

  ~DigiCollectionProfiler() {};

  void analyze(edm::Handle<edm::DetSetVector<SiStripDigi> > digis);
  void setMaskedModules(std::vector<unsigned int> maskedmod);

 private:

  int ismasked(const DetId& mod) const;

  unsigned int m_nevent;

  TProfile* m_tibprof;
  TProfile* m_tobprof;
  TProfile* m_tecpprof;
  TProfile* m_tecmprof;
  TH2F* m_tib2d;
  TH2F* m_tob2d;
  TH2F* m_tecp2d;
  TH2F* m_tecm2d;

  std::vector<unsigned int> m_maskedmod;

};

#endif // DPGAnalysis_SiStripTools_DigiCollectionProfile_H
