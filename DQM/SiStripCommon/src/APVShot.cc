#include <vector>
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DQM/SiStripCommon/interface/APVShot.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>

APVShot::APVShot(): 
  _zs(true), _apv(-1), _nstrips(0), _median(-1), _detid() { }

APVShot::APVShot(const bool zs): 
  _zs(zs), _apv(-1), _nstrips(0), _median(-1), _detid() { }

APVShot::APVShot(const std::vector<SiStripDigi>& digis, const DetId& detid, const bool zs):
  _zs(zs), _apv(-1), _nstrips(0), _median(-1), _detid() 
{ 
  computeShot(digis,detid,zs);
}

void APVShot::computeShot(const std::vector<SiStripDigi>& digis, const DetId& detid, const bool zs) {

  _zs = zs;
  _detid = detid;

  _nstrips = 0;
  _apv = -1;
  _median = -1;

  std::vector<unsigned int> charge;
  for(std::vector<SiStripDigi>::const_iterator digi=digis.begin();digi!=digis.end();++digi) {

    if(!_zs || digi->adc()>0) {
      int oldapv = _apv;
      _apv = digi->strip()/128;
      if(oldapv>=0 && oldapv!=_apv) throw cms::Exception("WrongDigiVector") << "Digis from Different APVs" ;
      
      charge.push_back(digi->adc());
      ++_nstrips;
    }
  }
  
  // charge to be sorted in descending order

  std::sort(charge.begin(),charge.end()); 
  std::reverse(charge.begin(),charge.end());

  if(charge.size()> 64) { _median = float(charge[64]); }

}

const bool APVShot::isGenuine() const { return (_nstrips > _threshold); }

const int APVShot::apvNumber() const { return _apv; }

const int APVShot::nStrips() const { return _nstrips; }

const float APVShot::median() const { return _median; }

const int APVShot::subDet() const { return _detid.subdetId(); }

const unsigned int APVShot::detId() const { return _detid.rawId(); }

const int APVShot::_threshold = 64;
