#ifndef CommonTools_PileupAlgos_PuppiCandidate_h
#define CommonTools_PileupAlgos_PuppiCandidate_h

struct PuppiCandidate {
  double pt{0};
  double eta{0};
  double phi{0};
  double m{0};
  double rapidity{0};
  double px{0};
  double py{0};
  double pz{0};
  double e{0};
  int id{0};
  int puppi_register{0};
};

#endif
