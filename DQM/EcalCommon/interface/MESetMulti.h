#ifndef MESetMulti_H
#define MESetMulti_H

#include "MESet.h"

namespace ecaldqm
{
  /* class MESetMulti
     wrapper for a set of MESets
     use() method sets the MESet to be used
     MESet will not be booked / retrieved unless it is used once
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

    void fill(DetId const& _id, double _xyw = 1., double _yw = 1., double _w = 1.) { if(current_) current_->fill(_id, _xyw, _yw, _w); }
    void fill(EcalElectronicsId const& _id, double _xyw = 1., double _yw = 1., double _w = 1.) { if(current_) current_->fill(_id, _xyw, _yw, _w); }
    void fill(unsigned _dcctccid, double _xyw = 1., double _yw = 1., double _w = 1.) { if(current_) current_->fill(_dcctccid, _xyw, _yw, _w); }
    void fill(double _x, double _yw = 1., double _w = 1.) { if(current_) current_->fill(_x, _yw, _w); }

    void setBinContent(DetId const& _id, double _content) { if(current_) current_->setBinContent(_id, _content); }
    void setBinContent(EcalElectronicsId const& _id, double _content) { if(current_) current_->setBinContent(_id, _content); }
    void setBinContent(unsigned _dcctccid, double _content) { if(current_) current_->setBinContent(_dcctccid, _content); }
    void setBinContent(DetId const& _id, int _bin, double _content) { if(current_) current_->setBinContent(_id, _bin, _content); }
    void setBinContent(EcalElectronicsId const& _id, int _bin, double _content) { if(current_) current_->setBinContent(_id, _bin, _content); }
    void setBinContent(unsigned _dcctccid, int _bin, double _content) { if(current_) current_->setBinContent(_dcctccid, _bin, _content); }
    void setBinContent(int _bin, double _content) { if(current_) current_->setBinContent(_bin, _content); }

    void setBinError(DetId const& _id, double _error) { if(current_) current_->setBinError(_id, _error); }
    void setBinError(EcalElectronicsId const& _id, double _error) { if(current_) current_->setBinError(_id, _error); }
    void setBinError(unsigned _dcctccid, double _error) { if(current_) current_->setBinError(_dcctccid, _error); }
    void setBinError(DetId const& _id, int _bin, double _error) { if(current_) current_->setBinError(_id, _bin, _error); }
    void setBinError(EcalElectronicsId const& _id, int _bin, double _error) { if(current_) current_->setBinError(_id, _bin, _error); }
    void setBinError(unsigned _dcctccid, int _bin, double _error) { if(current_) current_->setBinError(_dcctccid, _bin, _error); }
    void setBinError(int _bin, double _error) { if(current_) current_->setBinError(_bin, _error); }

    void setBinEntries(DetId const& _id, double _entries) { if(current_) current_->setBinEntries(_id, _entries); }
    void setBinEntries(EcalElectronicsId const& _id, double _entries) { if(current_) current_->setBinEntries(_id, _entries); }
    void setBinEntries(unsigned _dcctccid, double _entries) { if(current_) current_->setBinEntries(_dcctccid, _entries); }
    void setBinEntries(DetId const& _id, int _bin, double _entries) { if(current_) current_->setBinEntries(_id, _bin, _entries); }
    void setBinEntries(EcalElectronicsId const& _id, int _bin, double _entries) { if(current_) current_->setBinEntries(_id, _bin, _entries); }
    void setBinEntries(unsigned _dcctccid, int _bin, double _entries) { if(current_) current_->setBinEntries(_dcctccid, _bin, _entries); }
    void setBinEntries(int _bin, double _entries) { if(current_) current_->setBinEntries(_bin, _entries); }

    double getBinContent(DetId const& _id, int _bin = 0) const { return current_ ? current_->getBinContent(_id, _bin) : 0.; }
    double getBinContent(EcalElectronicsId const& _id, int _bin = 0) const { return current_ ? current_->getBinContent(_id, _bin) : 0.; }
    double getBinContent(unsigned _dcctccid, int _bin = 0) const { return current_ ? current_->getBinContent(_dcctccid, _bin) : 0.; }
    double getBinContent(int _bin) const { return current_ ? current_->getBinContent(_bin) : 0.; }

    double getBinError(DetId const& _id, int _bin = 0) const { return current_ ? current_->getBinError(_id, _bin) : 0.; }
    double getBinError(EcalElectronicsId const& _id, int _bin = 0) const { return current_ ? current_->getBinError(_id, _bin) : 0.; }
    double getBinError(unsigned _dcctccid, int _bin = 0) const { return current_ ? current_->getBinError(_dcctccid, _bin) : 0.; }
    double getBinError(int _bin) const { return current_ ? current_->getBinError(_bin) : 0.; }

    double getBinEntries(DetId const& _id, int _bin = 0) const { return current_ ? current_->getBinEntries(_id, _bin) : 0.; }
    double getBinEntries(EcalElectronicsId const& _id, int _bin = 0) const { return current_ ? current_->getBinEntries(_id, _bin) : 0.; }
    double getBinEntries(unsigned _dcctccid, int _bin = 0) const { return current_ ? current_->getBinEntries(_dcctccid, _bin) : 0.; }
    double getBinEntries(int _bin) const { return current_ ? current_->getBinEntries(_bin) : 0.; }

    void reset(double _content = 0., double _err = 0., double _entries = 0.) { if(current_) current_->reset(_content, _err, _entries); }
    void resetAll(double _content = 0., double _err = 0., double _entries = 0.) { if(current_) current_->resetAll(_content, _err, _entries); }

    void formPath(std::map<std::string, std::string> const& _replacements) const { if(current_) current_->formPath(_replacements); }

    std::string const& getPath() const { return current_ ? current_->getPath() : path_; }
    MonitorElement const* getME(unsigned _iME) const { return current_ ? current_->getME(_iME) : 0; }
    MonitorElement* getME(unsigned _iME) { return current_ ? current_->getME(_iME) : 0; }

    void use(unsigned) const;
    MESet* getCurrent() const { return current_; }
    unsigned getMultiplicity() const { return sets_.size(); }

    const_iterator begin() const { return current_ ? const_iterator(current_) : end(); }
    const_iterator end() const { return const_iterator(current_, -1, -1); }
    const_iterator beginChannel() const { return current_ ? current_->beginChannel() : end(); }
    iterator begin() { return current_ ? iterator(current_) : end(); }
    iterator end() { return iterator(current_, -1, -1); }
    iterator beginChannel() { return current_ ? current_->beginChannel() : end(); }

  protected:
    mutable MESet* current_;
    std::vector<MESet*> sets_;
    mutable std::vector<bool> use_;
  };
}

#endif
