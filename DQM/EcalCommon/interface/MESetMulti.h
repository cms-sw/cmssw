#ifndef MESetMulti_H
#define MESetMulti_H

#include "MESet.h"

namespace ecaldqm
{
  /* class MESetMulti
     wrapper for a set of MESets
     use() method sets the MESet to be used
  */

  class MESetMulti : public MESet {
  public:
    MESetMulti(MESet const&, unsigned);
    MESetMulti(MESetMulti const&);
    ~MESetMulti();

    MESet& operator=(MESet const&);

    MESet* clone() const;

    void book();
    bool retrieve() const;
    void clear() const;

    void fill(DetId const& _id, double _xyw = 1., double _yw = 1., double _w = 1.) { current_->fill(_id, _xyw, _yw, _w); }
    void fill(EcalElectronicsId const& _id, double _xyw = 1., double _yw = 1., double _w = 1.) { current_->fill(_id, _xyw, _yw, _w); }
    void fill(unsigned _dcctccid, double _xyw = 1., double _yw = 1., double _w = 1.) { current_->fill(_dcctccid, _xyw, _yw, _w); }
    void fill(double _x, double _yw = 1., double _w = 1.) { current_->fill(_x, _yw, _w); }

    void setBinContent(DetId const& _id, double _content) { current_->setBinContent(_id, _content); }
    void setBinContent(EcalElectronicsId const& _id, double _content) { current_->setBinContent(_id, _content); }
    void setBinContent(unsigned _dcctccid, double _content) { current_->setBinContent(_dcctccid, _content); }
    void setBinContent(DetId const& _id, int _bin, double _content) { current_->setBinContent(_id, _bin, _content); }
    void setBinContent(EcalElectronicsId const& _id, int _bin, double _content) { current_->setBinContent(_id, _bin, _content); }
    void setBinContent(unsigned _dcctccid, int _bin, double _content) { current_->setBinContent(_dcctccid, _bin, _content); }
    void setBinContent(int _bin, double _content) { current_->setBinContent(_bin, _content); }

    void setBinError(DetId const& _id, double _error) { current_->setBinError(_id, _error); }
    void setBinError(EcalElectronicsId const& _id, double _error) { current_->setBinError(_id, _error); }
    void setBinError(unsigned _dcctccid, double _error) { current_->setBinError(_dcctccid, _error); }
    void setBinError(DetId const& _id, int _bin, double _error) { current_->setBinError(_id, _bin, _error); }
    void setBinError(EcalElectronicsId const& _id, int _bin, double _error) { current_->setBinError(_id, _bin, _error); }
    void setBinError(unsigned _dcctccid, int _bin, double _error) { current_->setBinError(_dcctccid, _bin, _error); }
    void setBinError(int _bin, double _error) { current_->setBinError(_bin, _error); }

    void setBinEntries(DetId const& _id, double _entries) { current_->setBinEntries(_id, _entries); }
    void setBinEntries(EcalElectronicsId const& _id, double _entries) { current_->setBinEntries(_id, _entries); }
    void setBinEntries(unsigned _dcctccid, double _entries) { current_->setBinEntries(_dcctccid, _entries); }
    void setBinEntries(DetId const& _id, int _bin, double _entries) { current_->setBinEntries(_id, _bin, _entries); }
    void setBinEntries(EcalElectronicsId const& _id, int _bin, double _entries) { current_->setBinEntries(_id, _bin, _entries); }
    void setBinEntries(unsigned _dcctccid, int _bin, double _entries) { current_->setBinEntries(_dcctccid, _bin, _entries); }
    void setBinEntries(int _bin, double _entries) { current_->setBinEntries(_bin, _entries); }

    double getBinContent(DetId const& _id, int _bin = 0) const { return current_->getBinContent(_id, _bin); }
    double getBinContent(EcalElectronicsId const& _id, int _bin = 0) const { return current_->getBinContent(_id, _bin); }
    double getBinContent(unsigned _dcctccid, int _bin = 0) const { return current_->getBinContent(_dcctccid, _bin); }
    double getBinContent(int _bin) const { return current_->getBinContent(_bin); }

    double getBinError(DetId const& _id, int _bin = 0) const { return current_->getBinError(_id, _bin); }
    double getBinError(EcalElectronicsId const& _id, int _bin = 0) const { return current_->getBinError(_id, _bin); }
    double getBinError(unsigned _dcctccid, int _bin = 0) const { return current_->getBinError(_dcctccid, _bin); }
    double getBinError(int _bin) const { return current_->getBinError(_bin); }

    double getBinEntries(DetId const& _id, int _bin = 0) const { return current_->getBinEntries(_id, _bin); }
    double getBinEntries(EcalElectronicsId const& _id, int _bin = 0) const { return current_->getBinEntries(_id, _bin); }
    double getBinEntries(unsigned _dcctccid, int _bin = 0) const { return current_->getBinEntries(_dcctccid, _bin); }
    double getBinEntries(int _bin) const { return current_->getBinEntries(_bin); }

    int findBin(DetId const& _id) const { return current_->findBin(_id); }
    int findBin(EcalElectronicsId const& _id) const { return current_->findBin(_id); }
    int findBin(unsigned _dcctccid) const { return current_->findBin(_dcctccid); }
    int findBin(DetId const& _id, double _x, double _y = 0.) const { return current_->findBin(_id, _x, _y); }
    int findBin(EcalElectronicsId const& _id, double _x, double _y = 0.) const { return current_->findBin(_id, _x, _y); }
    int findBin(unsigned _dcctccid, double _x, double _y = 0.) const { return current_->findBin(_dcctccid, _x, _y); }
    int findBin(double _x, double _y = 0.) const { return current_->findBin(_x, _y); }

    void reset(double _content = 0., double _err = 0., double _entries = 0.) { current_->reset(_content, _err, _entries); }
    void resetAll(double _content = 0., double _err = 0., double _entries = 0.) { current_->reset(_content, _err, _entries); }

    void formPath(std::map<std::string, std::string> const& _replacements) const { current_->formPath(_replacements); }

    std::string const& getDir() const { return current_->getDir(); }
    void setDir(std::string const& _dir) { current_->setDir(_dir); }
    std::string const& getName() const { return current_->getName(); }
    void setName(std::string const& _name) { current_->setName(_name); }
    MonitorElement const* getME(unsigned _iME) const { return current_->getME(_iME); }

    void use(unsigned) const;
    MESet* getCurrent() const { return current_; }

    const_iterator begin() const { return const_iterator(current_); }
    const_iterator end() const { return const_iterator(current_, -1, -1); }
    const_iterator beginChannel() const { return current_->beginChannel(); }
    iterator begin() { return iterator(current_); }
    iterator end() { return iterator(current_, -1, -1); }
    iterator beginChannel() { return current_->beginChannel(); }

  protected:
    mutable MESet* current_;
    std::vector<MESet*> sets_;
  };
}

#endif
