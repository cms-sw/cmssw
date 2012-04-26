#include "DQM/EcalCommon/interface/MESetEcal.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <limits>

namespace ecaldqm
{

  MESetEcal::MESetEcal(std::string const& _fullpath, MEData const& _data, int _logicalDimensions, bool _readOnly/* = false*/) :
    MESet(_fullpath, _data, _readOnly),
    logicalDimensions_(_logicalDimensions),
    cacheId_(0),
    cache_(std::make_pair(-1, std::vector<int>(0)))
  {
  }

  MESetEcal::~MESetEcal()
  {
  }

  void
  MESetEcal::book()
  {
    using namespace std;

    clear();

    dqmStore_->setCurrentFolder(dir_);

    if(data_->btype == BinService::kUser && ((logicalDimensions_ > 0 && !data_->xaxis) || (logicalDimensions_ > 1 && !data_->yaxis)))
      throw cms::Exception("InvalidCall") << "Need axis specifications" << std::endl;

    if(data_->btype == BinService::kReport && name_ == "")
      name_ = dir_.substr(0, dir_.find_first_of('/'));

    std::vector<std::string> meNames(generateNames());

    for(unsigned iME(0); iME < meNames.size(); iME++){
      unsigned iObj(iME);

      BinService::ObjectType actualObject(binService_->objectFromOffset(data_->otype, iObj));

      BinService::AxisSpecs xaxis, yaxis, zaxis;

      if(logicalDimensions_ > 0){
	if(data_->xaxis){
	  xaxis = *data_->xaxis;
	}
	else{ // uses preset
	  bool isMap(logicalDimensions_ > 1);
	  vector<BinService::AxisSpecs> presetAxes(binService_->getBinning(actualObject, data_->btype, isMap, iObj));
	  if(presetAxes.size() != logicalDimensions_)
	    throw cms::Exception("InvalidCall") << "Dimensionality mismatch " << data_->otype << " " << data_->btype << " " << iObj << std::endl;

	  xaxis = presetAxes[0];
	  if(isMap) yaxis = presetAxes[1];
	}

	if(data_->yaxis){
	  yaxis = *data_->yaxis;
	}
	if(logicalDimensions_ == 1 && yaxis.high - yaxis.low < 0.0001){
	  yaxis.low = -numeric_limits<double>::max();
	  yaxis.high = numeric_limits<double>::max();
	}

	if(data_->zaxis){
	  zaxis = *data_->zaxis;
	}
	if(logicalDimensions_ > 1 && zaxis.high - zaxis.low < 0.0001){
	  zaxis.low = -numeric_limits<double>::max();
	  zaxis.high = numeric_limits<double>::max();
	}
      }

      MonitorElement* me(0);

      switch(data_->kind) {
      case MonitorElement::DQM_KIND_REAL :
	me = dqmStore_->bookFloat(meNames[iME]);

	break;

      case MonitorElement::DQM_KIND_TH1F :
	if(xaxis.edges)
	  me = dqmStore_->book1D(meNames[iME], meNames[iME], xaxis.nbins, xaxis.edges);
	else
	  me = dqmStore_->book1D(meNames[iME], meNames[iME], xaxis.nbins, xaxis.low, xaxis.high);

	break;

      case MonitorElement::DQM_KIND_TPROFILE :
	if(xaxis.edges) {
	  double* xedges(new double[xaxis.nbins + 1]);
	  for(int i = 0; i <= xaxis.nbins; i++) xedges[i] = xaxis.edges[i];
	  me = dqmStore_->bookProfile(meNames[iME], meNames[iME], xaxis.nbins, xedges, yaxis.low, yaxis.high, "");
	  delete [] xedges;
	}
	else
	  me = dqmStore_->bookProfile(meNames[iME], meNames[iME], xaxis.nbins, xaxis.low, xaxis.high, yaxis.low, yaxis.high, "");

	break;

      case MonitorElement::DQM_KIND_TH2F :
	if(xaxis.edges || yaxis.edges) {
	  BinService::AxisSpecs* specs[] = {&xaxis, &yaxis};
	  for(int iSpec(0); iSpec < 2; iSpec++){
	    if(!specs[iSpec]->edges) {
	      int nbins(specs[iSpec]->nbins);
	      float low(specs[iSpec]->low), high(specs[iSpec]->high);
	      specs[iSpec]->edges = new float[nbins + 1];
	      for(int i(0); i < nbins + 1; i++)
		specs[iSpec]->edges[i] = low + (high - low) / nbins * i;
	    }
	  }
	  me = dqmStore_->book2D(meNames[iME], meNames[iME], xaxis.nbins, xaxis.edges, yaxis.nbins, yaxis.edges);
	}
	else
	  me = dqmStore_->book2D(meNames[iME], meNames[iME], xaxis.nbins, xaxis.low, xaxis.high, yaxis.nbins, yaxis.low, yaxis.high);

	break;

      case MonitorElement::DQM_KIND_TPROFILE2D :
	if(zaxis.edges) {
	  zaxis.low = zaxis.edges[0];
	  zaxis.high = zaxis.edges[zaxis.nbins];
	}
	if(xaxis.edges || yaxis.edges)
	  throw cms::Exception("InvalidCall") << "Variable bin size for 2D profile not implemented" << std::endl;
	me = dqmStore_->bookProfile2D(meNames[iME], meNames[iME], xaxis.nbins, xaxis.low, xaxis.high, yaxis.nbins, yaxis.low, yaxis.high, zaxis.low, zaxis.high, "");

	break;

      default :
	break;
      }

      if(!me)
	throw cms::Exception("InvalidCall") << "ME could not be booked" << std::endl;

      if(logicalDimensions_ > 0){
	me->setAxisTitle(xaxis.title, 1);
	me->setAxisTitle(yaxis.title, 2);
	// For plot tagging in RenderPlugin; default values are 1 for both
	me->getTH1()->SetMarkerStyle(actualObject + 2);
	me->getTH1()->SetMarkerStyle(data_->btype + 2);
      }

      if(logicalDimensions_ == 1 && data_->btype == BinService::kDCC){
	if(actualObject == BinService::kEB){
	  for(int iBin(1); iBin <= me->getNbinsX(); iBin++)
	    me->setBinLabel(iBin, binService_->channelName(iBin + kEBmLow));
	}
	else if(actualObject == BinService::kEE){
	  for(int iBin(1); iBin <= me->getNbinsX() / 2; iBin++){
	    unsigned dccid((iBin + 2) % 9 + 1);
	    me->setBinLabel(iBin, binService_->channelName(dccid));
	  }
	  for(int iBin(1); iBin <= me->getNbinsX() / 2; iBin++){
	    unsigned dccid((iBin + 2) % 9 + 46);
	    me->setBinLabel(iBin + me->getNbinsX() / 2, binService_->channelName(dccid));
	  }
	}
	else if(actualObject == BinService::kEEm){
	  for(int iBin(1); iBin <= me->getNbinsX(); iBin++){
	    unsigned dccid((iBin + 2) % 9 + 1);
	    me->setBinLabel(iBin, binService_->channelName(dccid));
	  }
	}
	else if(actualObject == BinService::kEEp){
	  for(int iBin(1); iBin <= me->getNbinsX(); iBin++){
	    unsigned dccid((iBin + 2) % 9 + 46);
	    me->setBinLabel(iBin, binService_->channelName(dccid));
	  }
	}
      }
      
      mes_.push_back(me);
    }

    // To avoid the ambiguity between "content == 0 because the mean is 0" and "content == 0 because the entry is 0"
    // RenderPlugin must be configured accordingly
    if(data_->kind == MonitorElement::DQM_KIND_TPROFILE2D)
      resetAll(std::numeric_limits<double>::max(), 0., -1.);

    active_ = true;
  }

  bool
  MESetEcal::retrieve() const
  {
    clear();

    std::vector<std::string> meNames(generateNames());
    if(meNames.size() == 0) return false;

    for(unsigned iME(0); iME < meNames.size(); iME++){
      MonitorElement* me(dqmStore_->get(dir_ + "/" + meNames[iME]));
      if(me) mes_.push_back(me);
      else{
	clear();
	return false;
      }
    }

    active_ = true;
    return true;
  }

  void
  MESetEcal::fill(DetId const& _id, float _x/* = 1.*/, float _wy/* = 1.*/, float _w/* = 1.*/)
  {
    unsigned offset(binService_->findOffset(data_->otype, _id));
    if(offset >= mes_.size() || !mes_[offset])
      throw cms::Exception("InvalidCall") << "ME array index overflow" << std::endl;

    fill_(offset, _x, _wy, _w);
  }

  void
  MESetEcal::fill(unsigned _dcctccid, float _x/* = 1.*/, float _wy/* = 1.*/, float _w/* = 1.*/)
  {
    unsigned offset(binService_->findOffset(data_->otype, data_->btype, _dcctccid));

    if(offset >= mes_.size() || !mes_[offset])
      throw cms::Exception("InvalidCall") << "ME array index overflow" << offset << std::endl;

    fill_(offset, _x, _wy, _w);
  }

  void
  MESetEcal::fill(float _x, float _wy/* = 1.*/, float _w/* = 1.*/)
  {
    if(mes_.size() != 1)
      throw cms::Exception("InvalidCall") << "MESet type incompatible" << std::endl;

    fill_(0, _x, _wy, _w);
  }

  void
  MESetEcal::setBinContent(DetId const& _id, float _content, float _err/* = 0.*/)
  {
    find_(_id);

    std::vector<int>& bins(cache_.second);
    for(std::vector<int>::iterator binItr(bins.begin()); binItr != bins.end(); ++binItr)
      setBinContent_(cache_.first, *binItr, _content, _err);
  }

  void
  MESetEcal::setBinContent(unsigned _dcctccid, float _content, float _err/* = 0.*/)
  {
    find_(_dcctccid);

    std::vector<int>& bins(cache_.second);
    for(std::vector<int>::iterator binItr(bins.begin()); binItr != bins.end(); ++binItr)
      setBinContent_(cache_.first, *binItr, _content, _err);
  }

  void
  MESetEcal::setBinEntries(DetId const& _id, float _entries)
  {
    find_(_id);

    std::vector<int>& bins(cache_.second);
    for(std::vector<int>::iterator binItr(bins.begin()); binItr != bins.end(); ++binItr)
      setBinEntries_(cache_.first, *binItr, _entries);
  }

  void
  MESetEcal::setBinEntries(unsigned _dcctccid, float _entries)
  {
    find_(_dcctccid);

    std::vector<int>& bins(cache_.second);
    for(std::vector<int>::iterator binItr(bins.begin()); binItr != bins.end(); ++binItr)
      setBinEntries_(cache_.first, *binItr, _entries);
  }

  float
  MESetEcal::getBinContent(DetId const& _id, int) const
  {
    find_(_id);

    if(cache_.second.size() == 0) return 0.;

    return getBinContent_(cache_.first, cache_.second[0]);
  }

  float
  MESetEcal::getBinContent(unsigned _dcctccid, int) const
  {
    find_(_dcctccid);

    if(cache_.second.size() == 0) return 0.;

    return getBinContent_(cache_.first, cache_.second[0]);
  }

  float
  MESetEcal::getBinError(DetId const& _id, int) const
  {
    find_(_id);

    if(cache_.second.size() == 0) return 0.;

    return getBinError_(cache_.first, cache_.second[0]);
  }

  float
  MESetEcal::getBinError(unsigned _dcctccid, int) const
  {
    find_(_dcctccid);

    if(cache_.second.size() == 0) return 0.;
    
    return getBinError_(cache_.first, cache_.second[0]);
  }

  float
  MESetEcal::getBinEntries(DetId const& _id, int) const
  {
    find_(_id);

    if(cache_.second.size() == 0) return 0.;

    return getBinEntries_(cache_.first, cache_.second[0]);
  }

  float
  MESetEcal::getBinEntries(unsigned _dcctccid, int) const
  {
    find_(_dcctccid);

    if(cache_.second.size() == 0) return 0.;
    
    return getBinEntries_(cache_.first, cache_.second[0]);
  }

  void
  MESetEcal::reset(float _content/* = 0.*/, float _err/* = 0.*/, float _entries/* = 0.*/)
  {
    using namespace std;

    if(data_->btype >= unsigned(BinService::nPresetBinnings)){
      MESet::reset(_content, _err, _entries);
      return;
    }

    unsigned nME(1);
    switch(data_->otype){
    case BinService::kSM:
      nME = BinService::nDCC;
      break;
    case BinService::kSMMEM:
      nME = BinService::nDCCMEM;
      break;
    case BinService::kEcal2P:
      nME = 2;
      break;
    case BinService::kEcal3P:
      nME = 3;
      break;
    case BinService::kEcalMEM2P:
      nME = 2;
      break;
    default:
      break;
    }

    for(unsigned iME(0); iME < nME; iME++) {
      unsigned iObj(iME);

      BinService::ObjectType okey(binService_->objectFromOffset(data_->otype, iObj));
      BinService::BinningType bkey(data_->btype);
      if(okey == BinService::nObjType)
	throw cms::Exception("InvalidCall") << "Undefined object & bin type";

      std::vector<int> const* binMap(binService_->getBinMap(okey, bkey));
      if(!binMap)
	throw cms::Exception("InvalidCall") << "Cannot retrieve bin map";

      for(std::vector<int>::const_iterator binItr(binMap->begin()); binItr != binMap->end(); ++binItr){
	int bin(*binItr - binService_->smOffsetBins(okey, bkey, iObj));
	setBinContent_(iME, bin, _content, _err);
	setBinEntries_(iME, bin, 0.);
      }
    }
  }

  std::vector<std::string>
  MESetEcal::generateNames() const
  {
    using namespace std;

    unsigned nME(1);
    switch(data_->otype){
    case BinService::kSM:
      nME = BinService::nDCC;
      break;
    case BinService::kSMMEM:
      nME = BinService::nDCCMEM;
      break;
    case BinService::kEcal2P:
      nME = 2;
      break;
    case BinService::kEcal3P:
      nME = 3;
      break;
    case BinService::kEcalMEM2P:
      nME = 2;
      break;
    default:
      break;
    }

    std::vector<std::string> names(0);

    for(unsigned iME(0); iME < nME; iME++) {

      unsigned iObj(iME);

      BinService::ObjectType actualObject(binService_->objectFromOffset(data_->otype, iObj));

      string name(name_);
      string spacer(" ");

      if(data_->btype == BinService::kProjEta) name += " eta";
      else if(data_->btype == BinService::kProjPhi) name += " phi";
      else if(data_->btype == BinService::kReport) spacer = "_";

      switch(actualObject){
      case BinService::kEB:
      case BinService::kEBMEM:
	name += spacer + "EB"; break;
      case BinService::kEE:
      case BinService::kEEMEM:
	name += spacer + "EE"; break;
      case BinService::kEEm:
	name += spacer + "EE-"; break;
      case BinService::kEEp:
	name += spacer + "EE+"; break;
      case BinService::kSM:
	name += spacer + binService_->channelName(iObj + 1); break;
      case BinService::kSMMEM:
	//dccId(unsigned) skips DCCs without MEM
	iObj = dccId(iME) - 1;
	name += spacer + binService_->channelName(iObj + 1); break;
      default:
	break;
      }

      names.push_back(name);
    }

    return names;
  }

  void
  MESetEcal::find_(uint32_t _id) const
  {
    if(_id == cacheId_) return;

    DetId id(_id);
    if(id.det() == DetId::Ecal)
      cache_ = binService_->findBins(data_->otype, data_->btype, id);
    else
      cache_ = binService_->findBins(data_->otype, data_->btype, unsigned(_id));

    if(cache_.first >= mes_.size() || !mes_[cache_.first])
      throw cms::Exception("InvalidCall") << "ME array index overflow" << std::endl;

    // some TTs are apparently empty..!
//     if(cache_.second.size() == 0)
//       throw cms::Exception("InvalidCall") << "No bins to get content from" << std::endl;

    cacheId_ = _id;
  }

  void
  MESetEcal::fill_(unsigned _offset, float _x, float _wy, float _w)
  {
    if(data_->kind == MonitorElement::DQM_KIND_REAL)
      mes_[_offset]->Fill(_x);
    else if(data_->kind < MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE)
      mes_[_offset]->Fill(_x, _wy);
    else
      mes_[_offset]->Fill(_x, _wy, _w);
  }

  void
  MESetEcal::fill_(float _w)
  {
    for(unsigned iBin(0); iBin < cache_.second.size(); iBin++)
      MESet::fill_(cache_.first, cache_.second[iBin], _w);
  }

}
