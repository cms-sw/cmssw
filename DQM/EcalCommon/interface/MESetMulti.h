#ifndef MESetMulti_H
#define MESetMulti_H

#include "MESet.h"

namespace ecaldqm {
  /* class MESetMulti
   wrapper for a set of MESets
   limit() filters out unused MESets
   use() method sets the MESet to be used
*/

  class MESetMulti : public MESet {
  public:
    typedef std::map<std::string, std::vector<std::string>> ReplCandidates;

    MESetMulti(MESet const &, ReplCandidates const &);
    MESetMulti(MESetMulti const &);
    ~MESetMulti() override;

    MESet &operator=(MESet const &) override;

    MESet *clone(std::string const & = "") const override;

    void book(DQMStore::IBooker &, EcalElectronicsMapping const *) override;
    bool retrieve(EcalElectronicsMapping const *, DQMStore::IGetter &, std::string * = nullptr) const override;
    void clear() const override;

    void fill(
        EcalDQMSetupObjects const edso, DetId const &_id, double _xyw = 1., double _yw = 1., double _w = 1.) override {
      current_->fill(edso, _id, _xyw, _yw, _w);
    }
    void fill(EcalDQMSetupObjects const edso,
              EcalElectronicsId const &_id,
              double _xyw = 1.,
              double _yw = 1.,
              double _w = 1.) override {
      current_->fill(edso, _id, _xyw, _yw, _w);
    }
    void fill(
        EcalDQMSetupObjects const edso, int _dcctccid, double _xyw = 1., double _yw = 1., double _w = 1.) override {
      current_->fill(edso, _dcctccid, _xyw, _yw, _w);
    }
    void fill(EcalDQMSetupObjects const edso, double _x, double _yw = 1., double _w = 1.) override {
      current_->fill(edso, _x, _yw, _w);
    }

    void setBinContent(EcalDQMSetupObjects const edso, DetId const &_id, double _content) override {
      current_->setBinContent(edso, _id, _content);
    }
    void setBinContent(EcalDQMSetupObjects const edso, EcalElectronicsId const &_id, double _content) override {
      current_->setBinContent(edso, _id, _content);
    }
    void setBinContent(EcalDQMSetupObjects const edso, int _dcctccid, double _content) override {
      current_->setBinContent(edso, _dcctccid, _content);
    }
    void setBinContent(EcalDQMSetupObjects const edso, DetId const &_id, int _bin, double _content) override {
      current_->setBinContent(edso, _id, _bin, _content);
    }
    void setBinContent(EcalDQMSetupObjects const edso,
                       EcalElectronicsId const &_id,
                       int _bin,
                       double _content) override {
      current_->setBinContent(edso, _id, _bin, _content);
    }
    void setBinContent(EcalDQMSetupObjects const edso, int _dcctccid, int _bin, double _content) override {
      current_->setBinContent(edso, _dcctccid, _bin, _content);
    }

    void setBinError(EcalDQMSetupObjects const edso, DetId const &_id, double _error) override {
      current_->setBinError(edso, _id, _error);
    }
    void setBinError(EcalDQMSetupObjects const edso, EcalElectronicsId const &_id, double _error) override {
      current_->setBinError(edso, _id, _error);
    }
    void setBinError(EcalDQMSetupObjects const edso, int _dcctccid, double _error) override {
      current_->setBinError(edso, _dcctccid, _error);
    }
    void setBinError(EcalDQMSetupObjects const edso, DetId const &_id, int _bin, double _error) override {
      current_->setBinError(edso, _id, _bin, _error);
    }
    void setBinError(EcalDQMSetupObjects const edso, EcalElectronicsId const &_id, int _bin, double _error) override {
      current_->setBinError(edso, _id, _bin, _error);
    }
    void setBinError(EcalDQMSetupObjects const edso, int _dcctccid, int _bin, double _error) override {
      current_->setBinError(edso, _dcctccid, _bin, _error);
    }

    void setBinEntries(EcalDQMSetupObjects const edso, DetId const &_id, double _entries) override {
      current_->setBinEntries(edso, _id, _entries);
    }
    void setBinEntries(EcalDQMSetupObjects const edso, EcalElectronicsId const &_id, double _entries) override {
      current_->setBinEntries(edso, _id, _entries);
    }
    void setBinEntries(EcalDQMSetupObjects const edso, int _dcctccid, double _entries) override {
      current_->setBinEntries(edso, _dcctccid, _entries);
    }
    void setBinEntries(EcalDQMSetupObjects const edso, DetId const &_id, int _bin, double _entries) override {
      current_->setBinEntries(edso, _id, _bin, _entries);
    }
    void setBinEntries(EcalDQMSetupObjects const edso,
                       EcalElectronicsId const &_id,
                       int _bin,
                       double _entries) override {
      current_->setBinEntries(edso, _id, _bin, _entries);
    }
    void setBinEntries(EcalDQMSetupObjects const edso, int _dcctccid, int _bin, double _entries) override {
      current_->setBinEntries(edso, _dcctccid, _bin, _entries);
    }

    double getBinContent(EcalDQMSetupObjects const edso, DetId const &_id, int _bin = 0) const override {
      return current_->getBinContent(edso, _id, _bin);
    }
    double getBinContent(EcalDQMSetupObjects const edso, EcalElectronicsId const &_id, int _bin = 0) const override {
      return current_->getBinContent(edso, _id, _bin);
    }
    double getBinContent(EcalDQMSetupObjects const edso, int _dcctccid, int _bin = 0) const override {
      return current_->getBinContent(edso, _dcctccid, _bin);
    }

    double getBinError(EcalDQMSetupObjects const edso, DetId const &_id, int _bin = 0) const override {
      return current_->getBinError(edso, _id, _bin);
    }
    double getBinError(EcalDQMSetupObjects const edso, EcalElectronicsId const &_id, int _bin = 0) const override {
      return current_->getBinError(edso, _id, _bin);
    }
    double getBinError(EcalDQMSetupObjects const edso, int _dcctccid, int _bin = 0) const override {
      return current_->getBinError(edso, _dcctccid, _bin);
    }

    double getBinEntries(EcalDQMSetupObjects const edso, DetId const &_id, int _bin = 0) const override {
      return current_->getBinEntries(edso, _id, _bin);
    }
    double getBinEntries(EcalDQMSetupObjects const edso, EcalElectronicsId const &_id, int _bin = 0) const override {
      return current_->getBinEntries(edso, _id, _bin);
    }
    double getBinEntries(EcalDQMSetupObjects const edso, int _dcctccid, int _bin = 0) const override {
      return current_->getBinEntries(edso, _dcctccid, _bin);
    }

    void reset(EcalElectronicsMapping const *, double = 0., double = 0., double = 0.) override;
    void resetAll(double = 0., double = 0., double = 0.) override;

    bool maskMatches(DetId const &_id,
                     uint32_t _mask,
                     StatusManager const *_statusManager,
                     EcalTrigTowerConstituentsMap const *trigTowerMap) const override {
      return current_ && current_->maskMatches(_id, _mask, _statusManager, trigTowerMap);
    }

    bool isVariableBinning() const override { return current_->isVariableBinning(); }

    std::string const &getPath() const override { return current_->getPath(); }
    MonitorElement const *getME(unsigned _iME) const override { return current_->getME(_iME); }
    MonitorElement *getME(unsigned _iME) override { return current_->getME(_iME); }

    void use(unsigned) const;
    MESet *getCurrent() const { return current_; }
    unsigned getMultiplicity() const { return sets_.size(); }
    unsigned getIndex(PathReplacements const &) const;

    const_iterator begin(EcalElectronicsMapping const *electronicsMap) const override {
      return const_iterator(electronicsMap, *current_);
    }
    const_iterator end(EcalElectronicsMapping const *electronicsMap) const override {
      return const_iterator(electronicsMap, *current_, -1, -1);
    }
    const_iterator beginChannel(EcalElectronicsMapping const *electronicsMap) const override {
      return current_->beginChannel(electronicsMap);
    }
    iterator begin(EcalElectronicsMapping const *electronicsMap) override {
      return iterator(electronicsMap, *current_);
    }
    iterator end(EcalElectronicsMapping const *electronicsMap) override {
      return iterator(electronicsMap, *current_, -1, -1);
    }
    iterator beginChannel(EcalElectronicsMapping const *electronicsMap) override {
      return current_->beginChannel(electronicsMap);
    }

  protected:
    mutable MESet *current_;
    std::vector<MESet *> sets_;
    ReplCandidates replCandidates_;
  };
}  // namespace ecaldqm

#endif
