#ifndef MESetMulti_H
#define MESetMulti_H

#include "MESet.h"

namespace ecaldqm
{
  /* class MESetMulti
     wrapper for a set of MESets
     limit() filters out unused MESets
     use() method sets the MESet to be used
  */

  class MESetMulti : public MESet {
  public:
    typedef std::map<std::string, std::vector<std::string> > ReplCandidates;

    MESetMulti(MESet const&, ReplCandidates const&);
    MESetMulti(MESetMulti const&);
    ~MESetMulti();

    MESet& operator=(MESet const&) override;

    MESet* clone(std::string const& = "") const override;

    void book(DQMStore&) override;
    void book(DQMStore::IBooker&) override;
    bool retrieve(DQMStore const&, std::string* = 0) const override;
    void clear() const override;

    void fill(DetId const& _id, double _xyw = 1., double _yw = 1., double _w = 1.) override { current_->fill(_id, _xyw, _yw, _w); }
    void fill(EcalElectronicsId const& _id, double _xyw = 1., double _yw = 1., double _w = 1.) override { current_->fill(_id, _xyw, _yw, _w); }
    void fill(int _dcctccid, double _xyw = 1., double _yw = 1., double _w = 1.) override { current_->fill(_dcctccid, _xyw, _yw, _w); }
    void fill(double _x, double _yw = 1., double _w = 1.) override { current_->fill(_x, _yw, _w); }

    void setBinContent(DetId const& _id, double _content) override { current_->setBinContent(_id, _content); }
    void setBinContent(EcalElectronicsId const& _id, double _content) override { current_->setBinContent(_id, _content); }
    void setBinContent(int _dcctccid, double _content) override { current_->setBinContent(_dcctccid, _content); }
    void setBinContent(DetId const& _id, int _bin, double _content) override { current_->setBinContent(_id, _bin, _content); }
    void setBinContent(EcalElectronicsId const& _id, int _bin, double _content) override { current_->setBinContent(_id, _bin, _content); }
    void setBinContent(int _dcctccid, int _bin, double _content) override { current_->setBinContent(_dcctccid, _bin, _content); }

    void setBinError(DetId const& _id, double _error) override { current_->setBinError(_id, _error); }
    void setBinError(EcalElectronicsId const& _id, double _error) override { current_->setBinError(_id, _error); }
    void setBinError(int _dcctccid, double _error) override { current_->setBinError(_dcctccid, _error); }
    void setBinError(DetId const& _id, int _bin, double _error) override { current_->setBinError(_id, _bin, _error); }
    void setBinError(EcalElectronicsId const& _id, int _bin, double _error) override { current_->setBinError(_id, _bin, _error); }
    void setBinError(int _dcctccid, int _bin, double _error) override { current_->setBinError(_dcctccid, _bin, _error); }

    void setBinEntries(DetId const& _id, double _entries) override { current_->setBinEntries(_id, _entries); }
    void setBinEntries(EcalElectronicsId const& _id, double _entries) override { current_->setBinEntries(_id, _entries); }
    void setBinEntries(int _dcctccid, double _entries) override { current_->setBinEntries(_dcctccid, _entries); }
    void setBinEntries(DetId const& _id, int _bin, double _entries) override { current_->setBinEntries(_id, _bin, _entries); }
    void setBinEntries(EcalElectronicsId const& _id, int _bin, double _entries) override { current_->setBinEntries(_id, _bin, _entries); }
    void setBinEntries(int _dcctccid, int _bin, double _entries) override { current_->setBinEntries(_dcctccid, _bin, _entries); }

    double getBinContent(DetId const& _id, int _bin = 0) const override { return current_->getBinContent(_id, _bin); }
    double getBinContent(EcalElectronicsId const& _id, int _bin = 0) const override { return current_->getBinContent(_id, _bin); }
    double getBinContent(int _dcctccid, int _bin = 0) const override { return current_->getBinContent(_dcctccid, _bin); }

    double getBinError(DetId const& _id, int _bin = 0) const override { return current_->getBinError(_id, _bin); }
    double getBinError(EcalElectronicsId const& _id, int _bin = 0) const override { return current_->getBinError(_id, _bin); }
    double getBinError(int _dcctccid, int _bin = 0) const override { return current_->getBinError(_dcctccid, _bin); }

    double getBinEntries(DetId const& _id, int _bin = 0) const override { return current_->getBinEntries(_id, _bin); }
    double getBinEntries(EcalElectronicsId const& _id, int _bin = 0) const override { return current_->getBinEntries(_id, _bin); }
    double getBinEntries(int _dcctccid, int _bin = 0) const override { return current_->getBinEntries(_dcctccid, _bin); }

    void reset(double = 0., double = 0., double = 0.) override;
    void resetAll(double = 0., double = 0., double = 0.) override;

    bool maskMatches(DetId const& _id, uint32_t _mask, StatusManager const* _statusManager) const override { return current_ && current_->maskMatches(_id, _mask, _statusManager); }

    bool isVariableBinning() const override { return current_->isVariableBinning(); }

    std::string const& getPath() const override { return current_->getPath(); }
    MonitorElement const* getME(unsigned _iME) const override { return current_->getME(_iME); }
    MonitorElement* getME(unsigned _iME) override { return current_->getME(_iME); }

    void use(unsigned) const;
    MESet* getCurrent() const { return current_; }
    unsigned getMultiplicity() const { return sets_.size(); }
    unsigned getIndex(PathReplacements const&) const;

    const_iterator begin() const override { return const_iterator(*current_); }
    const_iterator end() const override { return const_iterator(*current_, -1, -1); }
    const_iterator beginChannel() const override { return current_->beginChannel(); }
    iterator begin() override { return iterator(*current_); }
    iterator end() override { return iterator(*current_, -1, -1); }
    iterator beginChannel() override { return current_->beginChannel(); }

  protected:
    mutable MESet* current_;
    std::vector<MESet*> sets_;
    ReplCandidates replCandidates_;
  };
}

#endif
