#ifndef GeneratorInterface_LHEInterface_LHEEvent_h
#define GeneratorInterface_LHEInterface_LHEEvent_h

#include <iostream>
#include <utility>
#include <memory>
#include <vector>
#include <string>

#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "HepMC/PdfInfo.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

namespace lhef {

  class LHEEvent {
  public:
    LHEEvent(const std::shared_ptr<LHERunInfo> &runInfo, std::istream &in);
    LHEEvent(const std::shared_ptr<LHERunInfo> &runInfo, const HEPEUP &hepeup);
    LHEEvent(const std::shared_ptr<LHERunInfo> &runInfo,
             const HEPEUP &hepeup,
             const LHEEventProduct::PDF *pdf,
             const std::vector<std::string> &comments);
    LHEEvent(const std::shared_ptr<LHERunInfo> &runInfo, const LHEEventProduct &product);
    ~LHEEvent();

    typedef LHEEventProduct::PDF PDF;
    typedef LHEEventProduct::WGT WGT;

    const std::shared_ptr<LHERunInfo> &getRunInfo() const { return runInfo; }
    const HEPEUP *getHEPEUP() const { return &hepeup; }
    const HEPRUP *getHEPRUP() const { return runInfo->getHEPRUP(); }
    const PDF *getPDF() const { return pdf.get(); }
    const std::vector<std::string> &getComments() const { return comments; }
    const int getReadAttempts() { return readAttemptCounter; }

    void addWeight(const WGT &wgt) { weights_.push_back(wgt); }
    void setPDF(std::unique_ptr<PDF> pdf) { this->pdf = std::move(pdf); }

    double originalXWGTUP() const { return originalXWGTUP_; }
    const std::vector<WGT> &weights() const { return weights_; }

    const std::vector<float> &scales() const { return scales_; }
    void setScales(const std::vector<float> &scales) { scales_ = scales; }

    int npLO() const { return npLO_; }
    int npNLO() const { return npNLO_; }
    int evtnum() const { return evtnum_; }

    void setNpLO(int n) { npLO_ = n; }
    void setNpNLO(int n) { npNLO_ = n; }
    void setEvtNum(int n) { evtnum_ = n; }

    void addComment(const std::string &line) { comments.push_back(line); }

    static void removeParticle(lhef::HEPEUP &hepeup, int index);
    void removeResonances(const std::vector<int> &ids);

    void count(LHERunInfo::CountMode count, double weight = 1.0, double matchWeight = 1.0);

    void attempted() {
      readAttemptCounter++;
      return;
    }

    void fillPdfInfo(HepMC::PdfInfo *info) const;
    void fillEventInfo(HepMC::GenEvent *hepmc) const;

    std::unique_ptr<HepMC::GenEvent> asHepMCEvent() const;

    static const HepMC::GenVertex *findSignalVertex(const HepMC::GenEvent *event, bool status3 = true);

    static void fixHepMCEventTimeOrdering(HepMC::GenEvent *event);

  private:
    static bool checkHepMCTree(const HepMC::GenEvent *event);
    HepMC::GenParticle *makeHepMCParticle(unsigned int i) const;

    const std::shared_ptr<LHERunInfo> runInfo;

    HEPEUP hepeup;
    std::unique_ptr<PDF> pdf;
    std::vector<WGT> weights_;
    std::vector<std::string> comments;
    bool counted;
    int readAttemptCounter;
    double originalXWGTUP_;
    std::vector<float> scales_;  //scale value used to exclude EWK-produced partons from matching
    int npLO_;                   //number of partons for LO process (used to steer matching/merging)
    int npNLO_;                  //number of partons for NLO process (used to steer matching/merging)
    int evtnum_;  //The number of the event (needed to ensure the correct LHE events are saved for MG +Herwig)
  };

}  // namespace lhef

#endif  // GeneratorEvent_LHEInterface_LHEEvent_h
