#include "DQM/EcalCommon/interface/EcalDQMBinningService.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include <algorithm>

#include "TMath.h"

using namespace ecaldqm;

std::vector<EcalDQMBinningService::AxisSpecs>
EcalDQMBinningService::getBinningEB_(BinningType _btype, bool _isMap) const
{
  std::vector<AxisSpecs> axes(0);
  AxisSpecs xaxis, yaxis;

  if(!_isMap){

    switch(_btype){
//     case kTriggerTower:
//     case kSuperCrystal:
//       xaxis.nbins = 2448;
//       xaxis.low = 0.;
//       xaxis.high = 2448.;
//       xaxis.title = "iTT";
//       break;
    case kTCC:
      xaxis.nbins = 36;
      xaxis.low = 9.;
      xaxis.high = 45.;
      xaxis.title = "iTCC";
      break;
    case kDCC:
      xaxis.nbins = 36;
      xaxis.low = 9.;
      xaxis.high = 45.;
      break;
    case kProjEta:
      xaxis.nbins = nEBEtaBins;
      xaxis.low = -etaBound_;
      xaxis.high = etaBound_;
      xaxis.title = "eta";
      break;
    case kProjPhi:
      xaxis.nbins = nPhiBins;
      xaxis.low = -TMath::Pi() / 18.;
      xaxis.high = TMath::Pi() * 35./18.;
      xaxis.title = "phi";
      break;
    default:
      return axes;
    }

    axes.push_back(xaxis);

  }
  else{

    switch(_btype){
    case kCrystal:
      xaxis.nbins = 360;
      yaxis.nbins = 170;
      break;
    case kSuperCrystal:
    case kTriggerTower:
      xaxis.nbins = 72;
      yaxis.nbins = 34;
      break;
    default:
      return axes;
    }

    xaxis.low = 0.;
    xaxis.high = 360.;
    xaxis.title = "iphi";
    yaxis.low = -85.;
    yaxis.high = 85.;
    yaxis.title = "ieta";

    axes.push_back(xaxis);
    axes.push_back(yaxis);

  }

  return axes;
}

std::vector<EcalDQMBinningService::AxisSpecs>
EcalDQMBinningService::getBinningEBMEM_(BinningType _btype, bool _isMap) const
{ 
  std::vector<AxisSpecs> axes(0);
  AxisSpecs xaxis, yaxis;

  if(_btype != kCrystal || !_isMap) return axes;

  xaxis.nbins = 18;
  xaxis.low = 0.;
  xaxis.high = 18.;
  xaxis.title = "channel";

  yaxis.nbins = 20;
  yaxis.low = -10.;
  yaxis.high = 10.;
  yaxis.title = "pseudo-strip";

  axes.push_back(xaxis);
  axes.push_back(yaxis);

  return axes;
}

std::vector<EcalDQMBinningService::AxisSpecs>
EcalDQMBinningService::getBinningEE_(BinningType _btype, bool _isMap, int _zside) const
{
  std::vector<AxisSpecs> axes(0);
  AxisSpecs xaxis, yaxis;

  if(!_isMap){

    switch(_btype){
//     case kTriggerTower:
//       xaxis.nbins = _zside ? 720 : 1440;
//       xaxis.low = 0.;
//       xaxis.high = _zside ? 720. : 1440.;
//       xaxis.title = "iTT";
//       break;
//     case kSuperCrystal:
//       xaxis.nbins = _zside ? 312 : 624;
//       xaxis.low = 0.;
//       xaxis.high = _zside ? 312. : 624.;
//       xaxis.title = "iSC";
//       break;
    case kTCC:
      xaxis.nbins = _zside ? 36 : 72;
      xaxis.low = 0.;
      xaxis.high = _zside ? 36. : 72.;
      xaxis.title = "iTCC";
      break;
    case kDCC:
      xaxis.nbins = _zside ? 9 : 18;
      xaxis.low = 0.;
      xaxis.high = _zside ? 9. : 18.;
      break;
    case kProjEta:
      if(!_zside) return axes;
      xaxis.nbins = nEEEtaBins;
      xaxis.low = _zside < 0 ? -3. : etaBound_;
      xaxis.high = _zside < 0 ? -etaBound_ : 3.;
      xaxis.title = "eta";
      break;
    case kProjPhi:
      xaxis.nbins = nPhiBins;
      xaxis.low = -TMath::Pi() / 18.;
      xaxis.high = TMath::Pi() * 35./18.;
      xaxis.title = "phi";
      break;
    default:
      return axes;
    }

    axes.push_back(xaxis);

  }else{

    switch(_btype){
    case kCrystal:
    case kTriggerTower:
      xaxis.nbins = _zside ? 100 : 200;
      yaxis.nbins = 100;
      break;
    case kSuperCrystal:
      xaxis.nbins = _zside ? 20 : 40;
      yaxis.nbins = 20;
      break;
    default:
      return axes;
    }

    xaxis.low = 0.;
    xaxis.high = _zside ? 100. : 200.;
    xaxis.title = "ix";
    yaxis.low = 0.;
    yaxis.high = 100.;
    yaxis.title = "iy";

    axes.push_back(xaxis);
    axes.push_back(yaxis);

  }

  return axes;
}

std::vector<EcalDQMBinningService::AxisSpecs>
EcalDQMBinningService::getBinningEEMEM_(BinningType _btype, bool _isMap) const
{
  std::vector<AxisSpecs> axes(0);
  AxisSpecs xaxis, yaxis;

  if(_btype != kCrystal || !_isMap) return axes;

  xaxis.nbins = 4;
  xaxis.low = 0.;
  xaxis.high = 4.;
  xaxis.title = "channel";

  yaxis.nbins = 20;
  yaxis.low = -10.;
  yaxis.high = 10.;
  yaxis.title = "pseudo-strip";

  axes.push_back(xaxis);
  axes.push_back(yaxis);

  return axes;
}

std::vector<EcalDQMBinningService::AxisSpecs>
EcalDQMBinningService::getBinningSM_(BinningType _btype, bool _isMap, unsigned _offset) const
{
  const bool isBarrel(_offset >= kEBmLow && _offset <= kEBpHigh);

  std::vector<AxisSpecs> axes(0);
  AxisSpecs xaxis, yaxis;

  if(!_isMap){

    switch(_btype){
    case kCrystal:
      xaxis.nbins = isBarrel ? 1700 : getElectronicsMap()->dccConstituents(_offset + 1).size();
      xaxis.low = 0.;
      xaxis.high = xaxis.nbins;
      xaxis.title = "channel";
      break;
    case kTriggerTower:
      xaxis.nbins = isBarrel ? 68 : 80;
      xaxis.low = 0.;
      xaxis.high = xaxis.nbins;
      xaxis.title = "tower";
      break;
    case kSuperCrystal:
      xaxis.nbins = isBarrel ? 68 : getNSuperCrystals(_offset + 1);
      xaxis.low = 0.;
      xaxis.high = xaxis.nbins;
      xaxis.title = "tower";
      break;
    default:
      return axes;
    }

    axes.push_back(xaxis);
    
  }else{

    int nEEX(nEESMX);
    if(_offset == kEEm02 || _offset == kEEm08 || _offset == kEEp02 || _offset == kEEp08) nEEX = nEESMXExt;

    switch(_btype){
    case kCrystal:
      xaxis.nbins = isBarrel ? nEBSMEta : nEEX;
      yaxis.nbins = isBarrel ? nEBSMPhi : nEESMY;
      break;
    case kTriggerTower:
      xaxis.nbins = isBarrel ? nEBSMEta / 5 : nEEX;
      yaxis.nbins = isBarrel ? nEBSMPhi / 5 : nEESMY;
      break;
    case kSuperCrystal:
      xaxis.nbins = isBarrel ? nEBSMEta / 5 : nEEX / 5;
      yaxis.nbins = isBarrel ? nEBSMPhi / 5 : nEESMY / 5;
      break;
    default:
      return axes;
    }
    xaxis.low = xlow(_offset);
    xaxis.high = xaxis.low + (isBarrel ? nEBSMEta : nEEX);
    xaxis.title = isBarrel ? (_offset < kEBpLow ? "-ieta" : "ieta") : "ix";
    yaxis.low = ylow(_offset);
    yaxis.high = yaxis.low + (isBarrel ? nEBSMPhi : nEESMY);
    yaxis.title = isBarrel ? "iphi" : "iy";

    axes.push_back(xaxis);
    axes.push_back(yaxis);

  }

  return axes;
}

std::vector<EcalDQMBinningService::AxisSpecs>
EcalDQMBinningService::getBinningSMMEM_(BinningType _btype, bool _isMap, unsigned _idcc) const
{
  std::vector<AxisSpecs> axes(0);
  AxisSpecs xaxis, yaxis;

  if(dccNoMEM.find(_idcc) != dccNoMEM.end()) return axes;
  if(_btype != kCrystal) return axes;

  xaxis.nbins = 10;
  xaxis.low = _idcc >= kEBpLow ? 0. : -10.;
  xaxis.high = _idcc >= kEBpLow ? 10. : 0.;
  xaxis.title = "pseudo-strip";

  axes.push_back(xaxis);

  if(_isMap){
    yaxis.nbins = 1;
    yaxis.low = 0.;
    yaxis.high = 5.;
    yaxis.title = "channel";
    axes.push_back(yaxis);
  }

  return axes;
}

std::vector<EcalDQMBinningService::AxisSpecs>
EcalDQMBinningService::getBinningEcal_(BinningType _btype, bool _isMap) const
{
  std::vector<AxisSpecs> axes(0);
  AxisSpecs xaxis, yaxis;

  if(!_isMap){

    switch(_btype){
    case kTCC:
      xaxis.nbins = 108;
      xaxis.low = 0.;
      xaxis.high = 108.;
      xaxis.title = "iTCC";
      break;
    case kDCC:
      xaxis.nbins = 54;
      xaxis.low = 0.;
      xaxis.high = 54.;
      break;
    case kProjEta:
      xaxis.nbins = nEBEtaBins + 2 * nEEEtaBins;
      xaxis.edges = new double[xaxis.nbins + 1];
      for(int i(0); i <= nEEEtaBins; i++)
	xaxis.edges[i] = -3. + (3. - etaBound_) / nEEEtaBins * i;
      for(int i(1); i <= nEBEtaBins; i++)
	xaxis.edges[i + nEEEtaBins] = -etaBound_ + 2. * etaBound_ / nEBEtaBins * i;
      for(int i(1); i <= nEEEtaBins; i++)
	xaxis.edges[i + nEEEtaBins + nEBEtaBins] = etaBound_ + (3. - etaBound_) / nEEEtaBins * i;
      xaxis.title = "eta";
      break;
    case kProjPhi:
      xaxis.nbins = nPhiBins;
      xaxis.low = -TMath::Pi() / 18.;
      xaxis.high = TMath::Pi() * 35./18.;
      xaxis.title = "phi";
      break;
    default:
      return axes;
    }

    axes.push_back(xaxis);
  }
  else{

    switch(_btype){
    case kCrystal:
      xaxis.nbins = 360;
      yaxis.nbins = 270;
      break;
    case kSuperCrystal:
      xaxis.nbins = 72;
      yaxis.nbins = 54;
      break;
    default:
      return axes;
    }

    xaxis.low = 0.;
    xaxis.high = 360.;
    xaxis.title = "iphi/ix/ix+100";
    yaxis.low = 0.;
    yaxis.high = 270.;
    yaxis.title = "ieta/iy";

    axes.push_back(xaxis);
    axes.push_back(yaxis);

  }

  return axes;
}


const std::vector<int>*
EcalDQMBinningService::getBinMapEB_(BinningType _bkey) const
{
  std::vector<int>& binMap(binMaps_[kEB][_bkey]);

  switch(_bkey){
  case kCrystal:
    binMap.resize(EBDetId::kSizeForDenseIndexing); // EBDetId -> bin
    for(int ix = 1; ix <= 360; ix++){
      for(int iy = 1; iy <= 170; iy++){
	uint32_t dIndex(EBDetId(iy < 86 ? iy - 86 : iy - 85, ix).hashedIndex());
	binMap[dIndex] = 360 * (iy - 1) + ix ;
      }
    }
    return &binMap;

  case kSuperCrystal:
    binMap.resize(EcalTrigTowerDetId::kEBTotalTowers); // EcalTrigTowerDetId -> bin
    for(int ix = 1; ix <= 72; ix++){
      for(int iy = 1; iy <= 34; iy++){
	int ieta(iy < 18 ? (iy - 18) * 5 : (iy - 17) * 5);
	int iphi(ix * 5);
	uint32_t dIndex(EBDetId(ieta, iphi).tower().hashedIndex());
	binMap[dIndex] = 72 * (iy - 1) + ix;
      }
    }
    return &binMap;

  case kDCC:
    {
      int nEBDCC(kEBpHigh - kEBmLow + 1);
      binMap.resize(nEBDCC); // DCCId (shifted) -> bin
      for(int ix = 1; ix <= nEBDCC; ix++)
	binMap[ix - 1] = ix;
    }
    return &binMap;

  case kTCC:
    {
      int nEBTCC(kEBTCCHigh - kEBTCCLow + 1);
      binMap.resize(nEBTCC); // TCCId (shifted) -> bin
      for(int ix = 1; ix <= nEBTCC; ix++)
	binMap[ix - 1] = ix;
    }
    return &binMap;

  case kProjEta:
    {
      float binEdges[nEBEtaBins + 1];
      for(int i(0); i <= nEBEtaBins; i++)
	binEdges[i] = -etaBound_ + 2. * etaBound_ / nEBEtaBins * i;
      binMap.resize(170); // ieta -> bin
      for(int ieta(-85); ieta <= 85; ieta++){
	if(ieta == 0) continue;
	EBDetId ebid(ieta, 1);
	float eta(geometry_->getGeometry(ebid)->getPosition().eta());
	float* pBin(std::upper_bound(binEdges, binEdges + nEBEtaBins + 1, eta));
	uint32_t dIndex(ieta < 0 ? ieta + 85 : ieta + 84);
	binMap[dIndex] = static_cast<int>(pBin - binEdges);
      }
    }
    return &binMap;

  case kProjPhi:
    {
      float binEdges[nPhiBins + 1];
      for(int i(0); i <= nPhiBins; i++)
	binEdges[i] = TMath::Pi() * (-1./18. + 2. / nPhiBins * i);
      binMap.resize(360); // iphi -> bin
      for(int iphi(1); iphi <= 360; iphi++){
	EBDetId ebid(1, iphi);
	float phi(geometry_->getGeometry(ebid)->getPosition().phi());
	if(phi < -TMath::Pi() * 1./18.) phi += 2. * TMath::Pi();
	float* pBin(std::upper_bound(binEdges, binEdges + nPhiBins + 1, phi));
	uint32_t dIndex(iphi - 1);
	binMap[dIndex] = static_cast<int>(pBin - binEdges);
      }
    }
    return &binMap;

  default:
    return 0;
  }
}

const std::vector<int>*
EcalDQMBinningService::getBinMapEBMEM_(BinningType _bkey) const
{
  if(_bkey != kCrystal) return 0;

  int nEBMEM((kEBpHigh - kEBmLow + 1) * 10);
  std::vector<int>& binMap(binMaps_[kEBMEM][kCrystal]);

  binMap.resize(nEBMEM); // EcalPnDiodeDetId -> bin; original hashing (DCCId * 10 + PnId)
  for(int ix = 1; ix <= 18; ix++){
    for(int iy = 1; iy <= 20; iy++){
      int idcc((iy < 11 ? kEBmLow : kEBpLow) + ix - 1);
      int pnId(iy < 11 ? 11 - iy : iy - 10);
      uint32_t dIndex((idcc - kEBmLow) * 10 + pnId - 1);
      binMap[dIndex] = 18 * (iy - 1) + ix;
    }
  }
  return &binMap;
}

const std::vector<int>*
EcalDQMBinningService::getBinMapEE_(BinningType _bkey, int _zside) const
{
  unsigned okey(0);
  switch(_zside){
  case -1: okey = kEEm; break;
  case 0: okey = kEE; break;
  case 1: okey = kEEp; break;
  default: return 0;
  }

  std::vector<int>& binMap(binMaps_[okey][_bkey]);

  int ixmax(_zside == 0 ? 200 : 100);
  int zside(_zside);

  switch(_bkey){
  case kCrystal: // EEDetId -> bin
    binMap.resize(_zside == 0 ? EEDetId::kSizeForDenseIndexing : EEDetId::kEEhalf);
    for(int ix = 1; ix <= ixmax; ix++){
      if(_zside == 0) zside = (ix <= 100 ? -1 : 1);
      int iix(_zside == 0 && ix > 100 ? ix - 100 : ix);
      for(int iy = 1; iy <= 100; iy++){
	if(!EEDetId::validDetId(iix, iy, zside)) continue;
	uint32_t dIndex(EEDetId(iix, iy, zside).hashedIndex());
	if(_zside == 1) dIndex -= EEDetId::kEEhalf;
	binMap[dIndex] = ixmax * (iy - 1) + ix;
      }
    }
    return &binMap;

  case kSuperCrystal: // EcalScDetId -> bin
    binMap.resize(_zside == 0 ? EcalScDetId::kSizeForDenseIndexing : EcalScDetId::SC_PER_EE_CNT);
    for(int ix = 1; ix <= ixmax / 5; ix++){
      if(_zside == 0) zside = (ix <= 20 ? -1 : 1);
      int iix(_zside == 0 && ix > 20 ? ix - 20 : ix);
      for(int iy = 1; iy <= 20; iy++){
	if(!EcalScDetId::validDetId(iix, iy, zside)) continue;
	uint32_t dIndex(EcalScDetId(iix, iy, zside).hashedIndex());
	if(_zside == 1) dIndex -= EcalScDetId::SC_PER_EE_CNT;
	binMap[dIndex] = ixmax / 5 * (iy - 1) + ix;
      }
    }
    return &binMap;

  case kDCC:
    {
      int nEEDCC(kEEmHigh - kEEmLow + kEEpHigh - kEEpLow + 2);
      if(_zside != 0) nEEDCC /= 2;
      binMap.resize(nEEDCC); // DCCId (shifted) -> bin
      for(int ix = 1; ix <= nEEDCC; ix++)
	binMap[ix - 1] = (ix + 5) % 9 + 1 + 9 * ((ix - 1) / 9);
    }
    return &binMap;

  case kTCC:
    {
      int nEETCC(kEEmTCCHigh - kEEmTCCLow + kEEpTCCHigh - kEEpTCCLow + 2);
      if(_zside != 0) nEETCC /= 2;
      binMap.resize(nEETCC); // TCCId (shifted) -> bin
      for(int ix = 1; ix <= nEETCC; ix++)
	binMap[ix - 1] = ix;
    }
    return &binMap;

  case kProjEta:
    {
      if(_zside == 0) return 0;

      float binEdges[nEEEtaBins + 1];
      if(_zside < 0){
	for(int i(0); i <= nEEEtaBins; i++)
	  binEdges[i] = -3. + (3. - etaBound_) / nEEEtaBins * i;
      }
      else{
	for(int i(0); i <= nEEEtaBins; i++)
	  binEdges[i] = etaBound_ + (3. - etaBound_) / nEEEtaBins * i;
      }
      binMap.resize(EEDetId::kEEhalf / 2); // EEDetId (half) -> bin
      // Only a quadrant is really necessary, but the hashed index cannot be resolved for quadrants
      for(int ix = 1; ix <= 100; ix++){
	for(int iy = 1; iy <= 50; iy++){
	  if(!EEDetId::validDetId(ix, iy, _zside)) continue;
	  EEDetId eeid(ix, iy, _zside);
	  float eta(geometry_->getGeometry(eeid)->getPosition().eta());
	  float* pBin(std::upper_bound(binEdges, binEdges + nEEEtaBins + 1, eta));
	  uint32_t dIndex(eeid.hashedIndex());
	  if(_zside == 1) dIndex -= EEDetId::kEEhalf;
	  binMap[dIndex] = static_cast<int>(pBin - binEdges);
	}    
      }
    }
    return &binMap;

  case kProjPhi:
    {
      if(_zside != 0) return 0;

      float binEdges[nPhiBins + 1];
      for(int i(0); i <= nPhiBins; i++)
	binEdges[i] = TMath::Pi() * (-1./18. + 2. / nPhiBins * i);
      binMap.resize(EEDetId::kEEhalf); // EEDetId(-) -> bin
      for(int ix = 1; ix <= 100; ix++){
	for(int iy = 1; iy <= 100; iy++){
	  if(!EEDetId::validDetId(ix, iy, -1)) continue;
	  EEDetId eeid(ix, iy, -1);
	  float phi(geometry_->getGeometry(eeid)->getPosition().phi());
	  if(phi < -TMath::Pi() * 1./18.) phi += 2. * TMath::Pi();
	  float* pBin(std::upper_bound(binEdges, binEdges + nPhiBins + 1, phi));
	  uint32_t dIndex(eeid.hashedIndex());
	  binMap[dIndex] = static_cast<int>(pBin - binEdges);
	}    
      }
    }
    return &binMap;

  default:
    return 0;
  }
}

const std::vector<int>*
EcalDQMBinningService::getBinMapEEMEM_(BinningType _bkey) const
{
  if(_bkey != kCrystal) return 0;

  std::vector<int>& binMap(binMaps_[kEEMEM][kCrystal]);

  binMap.resize((kEEmHigh - kEEmLow + kEEpHigh - kEEpLow + 2) * 10); // EcalPnDiodeDetId -> bin (see above)
  int memIx(1);
  int iEEDCC(-1);
  int offset(0);
  for(int iSM(kEEmLow); iSM <= kEEpHigh; iSM++){
    iEEDCC++;
    if (dccNoMEM.find(iSM) != dccNoMEM.end()) {
      for (int ich(0); ich < 10; ich++)
	binMap[iEEDCC * 10 + ich] = 0;
      continue;
    }
    if (iSM == kEBmLow) {
      iSM = kEEpLow;
      offset = 10;
      memIx = 1;
    }
    for(int iy = 1 + offset; iy <= 10 + offset; iy++){
      int pnId(iy < 11 ? 11 - iy : iy - 10);
      uint32_t dIndex(iEEDCC * 10 + pnId - 1);
      binMap[dIndex] = 4 * (iy - 1) + memIx;
    }
    memIx++;
  }

  return &binMap;
}

const std::vector<int>*
EcalDQMBinningService::getBinMapSM_(BinningType _bkey) const
{
  int totalBins(0);

  std::vector<int>& binMap(binMaps_[kSM][_bkey]);

  switch(_bkey){
  case kCrystal:
    binMap.resize(EBDetId::kSizeForDenseIndexing + EEDetId::kSizeForDenseIndexing);
    for (int iDCC(0); iDCC < nDCC; iDCC++) {
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh){
	for(int ix = 1; ix <= nEBSMEta; ix++){
	  for(int iy = 1; iy <= nEBSMPhi; iy++){
	    int ieta(ix + xlow(iDCC));
	    int iphi(iy + ylow(iDCC));
	    if (iDCC >= kEBpLow){
	      iphi -= 1;
	      iphi *= -1;
	    }
	    else
	      ieta *= -1;

	    uint32_t dIndex(EBDetId(ieta, iphi).hashedIndex() + EEDetId::kEEhalf);
	    binMap[dIndex] = totalBins + nEBSMEta * (iy - 1) + ix;
	  }
	}
	totalBins += nEBSMBins;
      }
      else{
	std::vector<DetId> crystals(getElectronicsMap()->dccConstituents(iDCC + 1));
	int nEEX(nEESMX), nEEBins(nEESMBins);
	if(iDCC == kEEm02 || iDCC == kEEm08 || iDCC == kEEp02 || iDCC == kEEp08){
	  nEEX = nEESMXExt;
	  nEEBins = nEESMBinsExt;
	}
	for(std::vector<DetId>::iterator idItr(crystals.begin()); idItr != crystals.end(); ++idItr){
	  EEDetId id(*idItr);
	  uint32_t dIndex(id.hashedIndex());
	  int ix(id.ix() - xlow(iDCC));
	  int iy(id.iy() - ylow(iDCC));
	  if (id.zside() > 0)
	    dIndex += EBDetId::kSizeForDenseIndexing;

	  binMap[dIndex] = totalBins + nEEX * (iy - 1) + ix;
	}
	totalBins += nEEBins;
      }
    }
    return &binMap;

  case kSuperCrystal:
    binMap.resize(EcalTrigTowerDetId::kEBTotalTowers + EcalScDetId::kSizeForDenseIndexing); // EcalTrigTowerDetId / EcalScDetId -> bin

    for(int iSM(0); iSM < nDCC; iSM++){
      if(iSM >= kEBmLow && iSM <= kEBpHigh){
	for(int ix = 1; ix <= nEBSMEta; ix += 5){
	  for(int iy = 1; iy <= nEBSMPhi; iy += 5){
	    int ieta(ix + xlow(iSM));
	    int iphi(iy + ylow(iSM));
	    if(iSM >= kEBpLow){
	      iphi -= 1;
	      iphi *= -1;
	    }
	    else
	      ieta *= -1;
	    uint32_t dIndex(EBDetId(ieta, iphi).tower().hashedIndex());
	    binMap[dIndex + EcalScDetId::SC_PER_EE_CNT] = totalBins + nEBSMEta / 5 * (iy - 1) / 5 + (ix - 1) / 5 + 1;
	  }
	}
	totalBins += nEBSMBins / 25;
      }
      else{
	int nEEX(nEESMX), nEEBins(nEESMBins);
	if(iSM == kEEm02 || iSM == kEEm08 || iSM == kEEp02 || iSM == kEEp08){
	  nEEX = nEESMXExt;
	  nEEBins = nEESMBinsExt;
	}
	for (int ix(1); ix <= nEEX / 5; ix++) {
	  for (int iy(1); iy <= nEESMY / 5; iy++) {
	    int sciz(1);
	    if (iSM <= kEEmHigh) sciz = -1;
	    int scix(ix + xlow(iSM) / 5);
	    int sciy(iy + ylow(iSM) / 5);
	    if(scix <= 0 || scix > 20 || sciy <= 0 || sciy > 20) continue;
	    if(!EcalScDetId::validDetId(scix, sciy, sciz)) continue;
	    uint32_t dIndex(EcalScDetId(scix, sciy, sciz).hashedIndex());
	    if(sciz > 0) dIndex += EcalTrigTowerDetId::kEBTotalTowers;
	    binMap[dIndex] = totalBins + nEEX / 5 * (iy - 1) + ix;
	  }
	}
	totalBins += nEEBins / 25;
      }
    }
    return &binMap;

  case kTriggerTower:
    binMap.resize(EcalTrigTowerDetId::kEBTotalTowers + EEDetId::kSizeForDenseIndexing); // EcalTrigTowerDetId / EEDetId -> bin

    for(int iSM(0); iSM < nDCC; iSM++){
      if(iSM >= kEBmLow && iSM <= kEBpHigh){
	for(int ix = 1; ix <= nEBSMEta; ix += 5){
	  for(int iy = 1; iy <= nEBSMPhi; iy += 5){
	    int ieta(ix + xlow(iSM));
	    int iphi(iy + ylow(iSM));
	    if(iSM >= kEBpLow){
	      iphi -= 1;
	      iphi *= -1;
	    }
	    else
	      ieta *= -1;
	    uint32_t dIndex(EBDetId(ieta, iphi).tower().hashedIndex());
	    binMap[dIndex + EEDetId::kEEhalf] = totalBins + nEBSMEta / 5 * (iy - 1) / 5 + (ix - 1) / 5 + 1;
	  }
	}
	totalBins += nEBSMBins / 25;
      }
      else{
	int nEEX(nEESMX), nEEBins(nEESMBins);
	if(iSM == kEEm02 || iSM == kEEm08 || iSM == kEEp02 || iSM == kEEp08){
	  nEEX = nEESMXExt;
	  nEEBins = nEESMBinsExt;
	}

	int tccid[4];
	tccid[0] = ((iSM % 9) * 2 + 17) % 18 + 1;
	tccid[1] = tccid[0] % 18 + 1;
	tccid[2] = tccid[0] + 18;
	tccid[3] = tccid[1] + 18;
	if(iSM >= int(kEEpLow))
	  for(int i(0); i < 4; i++) tccid[i] += kEEpTCCLow;
	for(int it(0); it < 4; it++){
	  std::vector<DetId> crystals(getElectronicsMap()->tccConstituents(tccid[it]));
	  for(std::vector<DetId>::iterator idItr(crystals.begin()); idItr != crystals.end(); ++idItr){
	    EEDetId id(*idItr);
	    uint32_t dIndex(id.hashedIndex());
	    int ix(id.ix() - xlow(iSM));
	    int iy(id.iy() - ylow(iSM));
	    if (id.zside() > 0)
	      dIndex += EcalTrigTowerDetId::kEBTotalTowers;

	    binMap[dIndex] = totalBins + nEEX * (iy - 1) + ix;
	  }
	}
	totalBins += nEEBins;
      }
    }
    return &binMap;

  default:
    return 0;
  }
}

const std::vector<int>*
EcalDQMBinningService::getBinMapSMMEM_(BinningType _bkey) const
{
  if(_bkey != kCrystal) return 0;

  int totalBins(0);

  std::vector<int>& binMap(binMaps_[kSMMEM][_bkey]);

  binMap.resize(nDCC * 10); // EcalPnDiodeDetId -> bin (see above)
  for(int iSM(0); iSM < nDCC; iSM++){
    if (dccNoMEM.find(iSM) != dccNoMEM.end()) {
      for (int ich(0); ich < 10; ich++)
	binMap[iSM * 10 + ich] = 0;
      continue;
    }
    for(int ix = 1; ix <= 10; ix++){
      int pnId(iSM <= kEBmHigh ? 11 - ix : ix);
      uint32_t dIndex(iSM * 10 + pnId - 1);
      binMap[dIndex] = totalBins + ix;
    }

    totalBins += 10;
  }

  return &binMap;
}

const std::vector<int>*
EcalDQMBinningService::getBinMapEcal_(BinningType _bkey) const
{
  std::vector<int>& binMap(binMaps_[kEcal][_bkey]);

  switch(_bkey){
  case kCrystal:
    binMap.resize(EBDetId::kSizeForDenseIndexing + EEDetId::kSizeForDenseIndexing); // DetId -> bin
    for(int ix = 1; ix <= 360; ix++){
      for(int iy = 101; iy <= 270; iy++){
	uint32_t dIndex(EBDetId(iy < 186 ? iy - 186 : iy - 185, ix).hashedIndex() + EEDetId::kEEhalf);
	binMap[dIndex] = 360 * (iy - 1) + ix ;
      }
    }
    for(int ix = 1; ix <= 200; ix++){
      int iix(ix > 100 ? ix - 100 : ix);
      int zside(ix > 100 ? 1 : -1);
      for(int iy = 1; iy <= 100; iy++){
	if(!EEDetId::validDetId(iix, iy, zside)) continue;
	uint32_t dIndex(EEDetId(iix, iy, zside).hashedIndex());
	if(zside == 1) dIndex += EBDetId::kSizeForDenseIndexing;
	binMap[dIndex] = 360 * (iy - 1) + ix;
      }
    }
    return &binMap;

  case kSuperCrystal:
    binMap.resize(EcalTrigTowerDetId::kEBTotalTowers + EcalScDetId::kSizeForDenseIndexing); // EcalTrigTowerDetId -> bin
    for(int ix = 1; ix <= 72; ix++){
      for(int iy = 21; iy <= 54; iy++){
	int ieta(iy < 38 ? (iy - 38) * 5 : (iy - 37) * 5);
	int iphi(ix * 5);
	uint32_t dIndex(EBDetId(ieta, iphi).tower().hashedIndex() + EcalScDetId::SC_PER_EE_CNT);
	binMap[dIndex] = 72 * (iy - 1) + ix;
      }
    }
    for(int ix = 1; ix <= 40; ix++){
      int iix((ix - 1) % 20 + 1);
      int zside(ix > 20 ? 1 : -1);
      for(int iy = 1; iy <= 20; iy++){
	if(!EcalScDetId::validDetId(iix, iy, zside)) continue;
	uint32_t dIndex(EcalScDetId(iix, iy, zside).hashedIndex());
	if(zside == 1) dIndex += EcalTrigTowerDetId::kEBTotalTowers;
	binMap[dIndex] = 72 * (iy - 1) + ix;
      }
    }
    return &binMap;

  case kProjEta:
    {
      float binEdges[nEBEtaBins + 2 * nEEEtaBins + 1];
      for(int i(0); i <= nEEEtaBins; i++)
	binEdges[i] = -3. + (3. - etaBound_) / nEEEtaBins * i;
      for(int i(1); i <= nEBEtaBins; i++)
	binEdges[i + nEEEtaBins] = -etaBound_ + 2. * etaBound_ / nEBEtaBins * i;
      for(int i(1); i <= nEEEtaBins; i++)
	binEdges[i + nEEEtaBins + nEBEtaBins] = etaBound_ + (3. - etaBound_) / nEEEtaBins * i;

      float* lowEdge(binEdges);
      binMap.resize(170 + EEDetId::kEEhalf); // EEDetId (half) -> bin
      for(int ix = 1; ix <= 100; ix++){
	for(int iy = 1; iy <= 50; iy++){
	  if(!EEDetId::validDetId(ix, iy, -1)) continue;
	  EEDetId eeid(ix, iy, -1);
	  float eta(geometry_->getGeometry(eeid)->getPosition().eta());
	  float* pBin(std::upper_bound(lowEdge, lowEdge + nEEEtaBins + 1, eta));
	  uint32_t dIndex(eeid.hashedIndex());
	  binMap[dIndex] = static_cast<int>(pBin - binEdges);
	}    
      }
      lowEdge += nEEEtaBins;
      for(int ieta(-85); ieta <= 85; ieta++){
	if(ieta == 0) continue;
	EBDetId ebid(ieta, 1);
	float eta(geometry_->getGeometry(ebid)->getPosition().eta());
	float* pBin(std::upper_bound(lowEdge, lowEdge + nEBEtaBins + 1, eta));
	uint32_t dIndex(ieta < 0 ? ieta + 85 : ieta + 84);
	dIndex += EEDetId::kEEhalf / 2;
	binMap[dIndex] = static_cast<int>(pBin - binEdges);
      }
      lowEdge += nEBEtaBins;
      for(int ix = 1; ix <= 100; ix++){
	for(int iy = 1; iy <= 50; iy++){
	  if(!EEDetId::validDetId(ix, iy, 1)) continue;
	  EEDetId eeid(ix, iy, 1);
	  float eta(geometry_->getGeometry(eeid)->getPosition().eta());
	  float* pBin(std::upper_bound(lowEdge, lowEdge + nEEEtaBins + 1, eta));
	  uint32_t dIndex(eeid.hashedIndex() - EEDetId::kEEhalf / 2 + 170);
	  binMap[dIndex] = static_cast<int>(pBin - binEdges);
	}    
      }
    }
    return &binMap;

  case kProjPhi:
    {
      float binEdges[nPhiBins + 1];
      for(int i(0); i <= nPhiBins; i++)
	binEdges[i] = TMath::Pi() * (-1./18. + 2. / nPhiBins * i);
      binMap.resize(360 + EEDetId::kEEhalf); // EEDetId(-) -> bin
      for(int ix = 1; ix <= 100; ix++){
	for(int iy = 1; iy <= 100; iy++){
	  if(!EEDetId::validDetId(ix, iy, -1)) continue;
	  EEDetId eeid(ix, iy, -1);
	  float phi(geometry_->getGeometry(eeid)->getPosition().phi());
	  if(phi < -TMath::Pi() * 1./18.) phi += 2. * TMath::Pi();
	  float* pBin(std::upper_bound(binEdges, binEdges + nPhiBins + 1, phi));
	  uint32_t dIndex(eeid.hashedIndex());
	  binMap[dIndex] = static_cast<int>(pBin - binEdges);
	}    
      }
      for(int iphi(1); iphi <= 360; iphi++){
	EBDetId ebid(1, iphi);
	float phi(geometry_->getGeometry(ebid)->getPosition().phi());
	if(phi < -TMath::Pi() * 1./18.) phi += 2. * TMath::Pi();
	float* pBin(std::upper_bound(binEdges, binEdges + nPhiBins + 1, phi));
	uint32_t dIndex(iphi - 1 + EEDetId::kEEhalf);
	binMap[dIndex] = static_cast<int>(pBin - binEdges);
      }
    }
    return &binMap;

  case kTCC:
    binMap.resize(nTCC);
    for(int ix = 1; ix <= nTCC; ix++)
      binMap[ix - 1] = ix;
    return &binMap;

  case kDCC:
    binMap.resize(nDCC);
    for(int ix = kEEmLow; ix <= kEEmHigh + 1; ix++)
      binMap[ix - 1] = (ix + 5) % 9 + 1;
    for(int ix = kEBpLow; ix <= kEBpHigh + 1; ix++)
      binMap[ix - 1] = ix;
    for(int ix = kEEpLow; ix <= kEEpHigh + 1; ix++)
      binMap[ix - 1] = (ix + 5) % 9 + 1 + kEEpLow;
    return &binMap;

  default:
    return 0;
  }
}

void
EcalDQMBinningService::findBinsCrystal_(const DetId& _id, ObjectType _okey, const std::vector<int>& _binMap, std::vector<int>& _bins) const
{
  using namespace std;

  bool fullRange(_okey == kSM || _okey == kEcal);

  switch(_id.subdetId()){
  case EcalBarrel:
    {
      unsigned index(EBDetId(_id).denseIndex());
      if(fullRange) index += EEDetId::kEEhalf;
      if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      break;
    }
  case EcalEndcap:
    if(isEcalScDetId(_id)){
      pair<int, int> dccSc(getElectronicsMap()->getDCCandSC(EcalScDetId(_id)));
      vector<DetId> detIds(getElectronicsMap()->dccTowerConstituents(dccSc.first, dccSc.second));
      for(vector<DetId>::iterator idItr(detIds.begin()); idItr != detIds.end(); ++idItr) {
	EEDetId eeid(*idItr);
	unsigned index(eeid.denseIndex());
	if(eeid.zside() > 0) {
	  if(fullRange) index += EBDetId::kSizeForDenseIndexing;
	  else if(_okey == kEEp) index -= EEDetId::kEEhalf;
	}
	if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      }
    }
    else{
      EEDetId eeid(_id);
      unsigned index(eeid.denseIndex());
      if(eeid.zside() > 0) {
	if(fullRange) index += EBDetId::kSizeForDenseIndexing;
	else if(_okey == kEEp) index -= EEDetId::kEEhalf;
      }
      if(index < _binMap.size()) _bins.push_back(_binMap[index]);
    }
    break;
  case EcalTriggerTower:
    {
      EcalTrigTowerDetId ttid(_id);
      vector<DetId> detIds(getTrigTowerMap()->constituentsOf(ttid));
      for(vector<DetId>::iterator idItr(detIds.begin()); idItr != detIds.end(); ++idItr) {
	if(idItr->subdetId() == EcalBarrel) {
	  unsigned index(EBDetId(*idItr).denseIndex());
	  if(fullRange) index += EEDetId::kEEhalf;
	  if(index < _binMap.size()) _bins.push_back(_binMap[index]);
	}
	else {
	  EEDetId eeid(*idItr);
	  unsigned index(eeid.denseIndex());
	  if(eeid.zside() > 0) {
	    if(fullRange) index += EBDetId::kSizeForDenseIndexing;
	    else if(_okey == kEEp) index -= EEDetId::kEEhalf;
	  }
	  if (index < _binMap.size()) _bins.push_back(_binMap[index]);
	}
      }
      break;
    }
  case EcalLaserPnDiode:
    {
      EcalPnDiodeDetId pnid(_id);
      int iDCC(pnid.iDCCId() - 1);
      unsigned index(iDCC * 10 + pnid.iPnId());
      if(_okey == kEBMEM) index -= kEEmHigh * 10;
      else if(_okey == kEEMEM && iDCC >= kEEpLow) index -= (kEEpLow - kEBmLow) * 10;
      if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      break;
    }
  default:
    break;
  }
}

void
EcalDQMBinningService::findBinsTriggerTower_(const DetId& _id, ObjectType _okey, const std::vector<int>& _binMap, std::vector<int>& _bins) const
{
  using namespace std;

  switch(_id.subdetId()){
  case EcalBarrel:
    {
      EcalTrigTowerDetId ttid(EBDetId(_id).tower());
      unsigned index(ttid.hashedIndex() + EEDetId::kEEhalf);
      if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      break;
    }
  case EcalEndcap:
    if(!isEcalScDetId(_id)){
      vector<DetId> detIds(getTrigTowerMap()->constituentsOf(getTrigTowerMap()->towerOf(_id)));
      for(vector<DetId>::iterator idItr(detIds.begin()); idItr != detIds.end(); ++idItr){
	EEDetId eeid(*idItr);
	unsigned index(eeid.denseIndex());
	if(eeid.zside() > 0) index += EcalTrigTowerDetId::kEBTotalTowers;
	if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      }
      break;
    }
  case EcalTriggerTower:
    {
      EcalTrigTowerDetId ttid(_id);
      if(ttid.subDet() == EcalBarrel){
	unsigned index(ttid.hashedIndex() + EEDetId::kEEhalf);
	if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      }
      else{
	vector<DetId> detIds(getTrigTowerMap()->constituentsOf(ttid));
	for(vector<DetId>::iterator idItr(detIds.begin()); idItr != detIds.end(); ++idItr){
	  EEDetId eeid(*idItr);
	  unsigned index(eeid.denseIndex());
	  if(eeid.zside() > 0) index += EcalTrigTowerDetId::kEBTotalTowers;
	  if(index < _binMap.size()) _bins.push_back(_binMap[index]);
	}
      }
      break;
    }
  default:
    break;
  }

}

void
EcalDQMBinningService::findBinsSuperCrystal_(const DetId& _id, ObjectType _okey, const std::vector<int>& _binMap, std::vector<int>& _bins) const
{
  bool fullRange(_okey == kSM || _okey == kEcal);

  switch(_id.subdetId()){
  case EcalBarrel:
    {
      unsigned index(EBDetId(_id).tower().denseIndex());
      if(fullRange) index += EcalScDetId::SC_PER_EE_CNT;
      if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      break;
    }
  case EcalEndcap:
    {
      int zside(0);
      unsigned index;
      if (isEcalScDetId(_id)) {
	EcalScDetId scid(_id);
	zside = scid.zside();
	index = scid.denseIndex();
      }
      else {
	EEDetId eeid(_id);
	zside = eeid.zside();
	index = eeid.sc().denseIndex();
      }
      if(zside > 0) {
	if(fullRange) index += EcalTrigTowerDetId::kEBTotalTowers;
	else if(_okey == kEEp) index -= EcalScDetId::SC_PER_EE_CNT;
      }
      if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      break;
    }
  case EcalTriggerTower: // btype == kSuperCrystal && subdet == EcalTriggerTower => only happens for EB
    {
      EcalTrigTowerDetId ttid(_id);
      unsigned index(ttid.denseIndex());
      if(fullRange) index += EcalScDetId::SC_PER_EE_CNT;
      if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      break;
    }
  default:
    break;
  }
}
void
EcalDQMBinningService::findBinsDCC_(const DetId& _id, ObjectType _okey, const std::vector<int>& _binMap, std::vector<int>& _bins) const
{
  unsigned index(dccId(_id) - 1);
  if(_okey == kEB) index -= kEBmLow;
  else if(_okey == kEE && index >= kEEpLow) index -= (kEBpHigh - kEEmHigh);
  else if(_okey == kEEp) index -= kEEpLow;
  if(index < _binMap.size()) _bins.push_back(_binMap[index]);
}

void
EcalDQMBinningService::findBinsTCC_(const DetId& _id, ObjectType _okey, const std::vector<int>& _binMap, std::vector<int>& _bins) const
{
  unsigned index(tccId(_id) - 1);
  if(_okey == kEB) index -= kEBTCCLow;
  else if(_okey == kEE && index >= kEEpLow) index -= (kEBTCCHigh - kEEmTCCHigh);
  else if(_okey == kEEp) index -= kEEpTCCLow;
  if(index < _binMap.size()) _bins.push_back(_binMap[index]);
}

void
EcalDQMBinningService::findBinsProjEta_(const DetId& _id, ObjectType _okey, const std::vector<int>& _binMap, std::vector<int>& _bins) const
{
  using namespace std;

  switch(_id.subdetId()){
  case EcalBarrel:
    {
      EBDetId ebid(_id);
      unsigned index(ebid.ieta() < 0 ? ebid.ieta() + 85 : ebid.ieta() + 84);
      if(_okey == kEcal) index += EEDetId::kEEhalf / 2;
      if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      break;
    }
  case EcalEndcap:
    if(!isEcalScDetId(_id)){
      EEDetId eeid(_id);
      if(eeid.iquadrant() < 3)
	eeid = EEDetId(eeid.ix(), 101 - eeid.iy(), eeid.zside());
      unsigned index(eeid.denseIndex());
      if(_okey == kEEp) index -= EEDetId::kEEhalf;
      else if(_okey == kEcal && eeid.zside() > 0) index += 170 - EEDetId::kEEhalf / 2;
      if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      break;
    }
  case EcalTriggerTower:
    {
      EcalTrigTowerDetId ttid(_id);
      vector<DetId> detIds(getTrigTowerMap()->constituentsOf(ttid));
      set<int> binset;
      if(ttid.subDet() == EcalBarrel){
	for(vector<DetId>::iterator idItr(detIds.begin()); idItr != detIds.end(); ++idItr){
	  EBDetId ebid(*idItr);
	  unsigned index(ebid.ieta() < 0 ? ebid.ieta() + 85 : ebid.ieta() + 84);
	  if(_okey == kEcal) index += EEDetId::kEEhalf / 2;
	  if(index < _binMap.size()) binset.insert(_binMap[index]);
	}
      }
      else{
	for(vector<DetId>::iterator idItr(detIds.begin()); idItr != detIds.end(); ++idItr){
	  EEDetId eeid(*idItr);
	  if(eeid.iquadrant() < 3)
	    eeid = EEDetId(eeid.ix(), eeid.iy() <= 50 ? eeid.iy() : 101 - eeid.iy(), eeid.zside());
	  unsigned index(eeid.denseIndex());
	  if(_okey == kEEp) index -= EEDetId::kEEhalf;
	  else if(_okey == kEcal && eeid.zside() > 0) index += 170 - EEDetId::kEEhalf / 2;
	  if(index < _binMap.size()) binset.insert(_binMap[index]);
	}
      }
      for(set<int>::iterator binItr(binset.begin()); binItr != binset.end(); ++binItr)
	_bins.push_back(*binItr);
      break;
    }
  default:
    break;
  }
}

void
EcalDQMBinningService::findBinsProjPhi_(const DetId& _id, ObjectType _okey, const std::vector<int>& _binMap, std::vector<int>& _bins) const
{
  using namespace std;

  switch(_id.subdetId()){
  case EcalBarrel:
    {
      EBDetId ebid(_id);
      unsigned index(ebid.iphi() - 1);
      if(_okey == kEcal) index += EEDetId::kEEhalf;
      if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      break;
    }
  case EcalEndcap:
    if(!isEcalScDetId(_id)){
      EEDetId eeid(_id);
      unsigned index(eeid.denseIndex());
      if(eeid.zside() > 0) index -= EEDetId::kEEhalf;
      if(index < _binMap.size()) _bins.push_back(_binMap[index]);
      break;
    }
  case EcalTriggerTower:
    {
      EcalTrigTowerDetId ttid(_id);
      vector<DetId> detIds(getTrigTowerMap()->constituentsOf(ttid));
      set<int> binset;
      if(ttid.subDet() == EcalBarrel){
	for(vector<DetId>::iterator idItr(detIds.begin()); idItr != detIds.end(); ++idItr){
	  EBDetId ebid(*idItr);
	  unsigned index(ebid.iphi() - 1);
	  if(_okey == kEcal) index += EEDetId::kEEhalf;
	  if(index < _binMap.size()) binset.insert(_binMap[index]);
	}
      }
      else{
	for(vector<DetId>::iterator idItr(detIds.begin()); idItr != detIds.end(); ++idItr){
	  EEDetId eeid(*idItr);
	  unsigned index(eeid.denseIndex());
	  if(eeid.zside() > 0) index -= EEDetId::kEEhalf;
	  if(index < _binMap.size()) binset.insert(_binMap[index]);
	}
      }
      for(set<int>::iterator binItr(binset.begin()); binItr != binset.end(); ++binItr)
	_bins.push_back(*binItr);
      break;
    }
  default:
    break;
  }
}
