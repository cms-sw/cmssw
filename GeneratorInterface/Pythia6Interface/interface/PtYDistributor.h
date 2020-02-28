#ifndef gen_PTYDISTRIBUTOR_H
#define gen_PTYDISTRIBUTOR_H

namespace edm {
  class FileInPath;
}

namespace CLHEP {
  class RandGeneral;
  class HepRandomEngine;
}  // namespace CLHEP

namespace gen {
  class PtYDistributor {
  public:
    PtYDistributor(){};
    //PtYDistributor(std::string inputfile, CLHEP::HepRandomEngine& fRandomEngine, double ptmax, double ptmin, double ymax, double ymin, int ptbins, int ybins);
    PtYDistributor(
        const edm::FileInPath& fip, double ptmax, double ptmin, double ymax, double ymin, int ptbins, int ybins);
    virtual ~PtYDistributor(){};

    double fireY(CLHEP::HepRandomEngine*);
    double firePt(CLHEP::HepRandomEngine*);
    double fireY(double ymin, double ymax, CLHEP::HepRandomEngine*);
    double firePt(double ptmin, double ptmax, CLHEP::HepRandomEngine*);

  private:
    double ptmax_;
    double ptmin_;
    double ymax_;
    double ymin_;

    int ptbins_;
    int ybins_;

    CLHEP::RandGeneral* fYGenerator;
    CLHEP::RandGeneral* fPtGenerator;
  };
}  // namespace gen
#endif
