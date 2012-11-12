#include "DQM/EcalCommon/interface/MESetEcal.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetUtils.h"

#include <limits>
#include <sstream>

namespace ecaldqm
{

  MESetEcal::MESetEcal(std::string const& _fullPath, BinService::ObjectType _otype, BinService::BinningType _btype, MonitorElement::Kind _kind, unsigned _logicalDimensions, BinService::AxisSpecs const* _xaxis/* = 0*/, BinService::AxisSpecs const* _yaxis/* = 0*/, BinService::AxisSpecs const* _zaxis/* = 0*/) :
    MESet(_fullPath, _otype, _btype, _kind),
    logicalDimensions_(_logicalDimensions),
    xaxis_(_xaxis),
    yaxis_(_yaxis),
    zaxis_(_zaxis)
  {
    if(btype_ == BinService::kUser && ((logicalDimensions_ > 0 && !xaxis_) || (logicalDimensions_ > 1 && !yaxis_)))
      throw_("Need axis specifications");
  }

  MESetEcal::MESetEcal(MESetEcal const& _orig) :
    MESet(_orig),
    logicalDimensions_(_orig.logicalDimensions_),
    xaxis_(_orig.xaxis_ ? new BinService::AxisSpecs(*_orig.xaxis_) : 0),
    yaxis_(_orig.yaxis_ ? new BinService::AxisSpecs(*_orig.yaxis_) : 0),
    zaxis_(_orig.zaxis_ ? new BinService::AxisSpecs(*_orig.zaxis_) : 0)
  {
  }

  MESetEcal::~MESetEcal()
  {
    delete xaxis_;
    delete yaxis_;
    delete zaxis_;
  }

  MESet&
  MESetEcal::operator=(MESet const& _rhs)
  {
    delete xaxis_;
    delete yaxis_;
    delete zaxis_;

    MESetEcal const* pRhs(dynamic_cast<MESetEcal const*>(&_rhs));
    if(pRhs){
      logicalDimensions_ = pRhs->logicalDimensions_;
      if(pRhs->xaxis_) xaxis_ = new BinService::AxisSpecs(*pRhs->xaxis_);
      if(pRhs->yaxis_) yaxis_ = new BinService::AxisSpecs(*pRhs->yaxis_);
      if(pRhs->zaxis_) zaxis_ = new BinService::AxisSpecs(*pRhs->zaxis_);
    }
    return MESet::operator=(_rhs);
  }

  MESet*
  MESetEcal::clone() const
  {
    return new MESetEcal(*this);
  }

  void
  MESetEcal::book()
  {
    using namespace std;

    clear();

    vector<string> mePaths(generatePaths());

    for(unsigned iME(0); iME < mePaths.size(); iME++){
      string& path(mePaths[iME]);
      if(path.find('%') != string::npos)
        throw_("book() called with incompletely formed path");

      BinService::ObjectType actualObject(binService_->getObject(otype_, iME));

      BinService::AxisSpecs xaxis, yaxis, zaxis;

      bool isHistogram(logicalDimensions_ > 0);
      bool isMap(logicalDimensions_ > 1);

      if(isHistogram){

	if(xaxis_){
	  xaxis = *xaxis_;
	}
	else{ // uses preset
	  xaxis = binService_->getBinning(actualObject, btype_, isMap, 1, iME);
          if(isMap) yaxis = binService_->getBinning(actualObject, btype_, true, 2, iME);
	}

	if(yaxis_){
	  yaxis = *yaxis_;
	}
        if(yaxis.high - yaxis.low < 1.e-10){
          yaxis.low = -numeric_limits<double>::max();
          yaxis.high = numeric_limits<double>::max();
        }
	if(zaxis_){
	  zaxis = *zaxis_;
	}
        if(zaxis.high - zaxis.low < 1.e-10){
          zaxis.low = -numeric_limits<double>::max();
          zaxis.high = numeric_limits<double>::max();
        }
      }

      size_t slashPos(path.find_last_of('/'));
      string name(path.substr(slashPos + 1));
      dqmStore_->setCurrentFolder(path.substr(0, slashPos));

      MonitorElement* me(0);

      switch(kind_) {
      case MonitorElement::DQM_KIND_REAL :
	me = dqmStore_->bookFloat(name);

	break;

      case MonitorElement::DQM_KIND_TH1F :
	if(xaxis.edges){
	  float* edges(new float[xaxis.nbins + 1]);
	  for(int i(0); i < xaxis.nbins + 1; i++)
	    edges[i] = xaxis.edges[i];
	  me = dqmStore_->book1D(name, name, xaxis.nbins, edges);
	  delete [] edges;
	}
	else
	  me = dqmStore_->book1D(name, name, xaxis.nbins, xaxis.low, xaxis.high);

	break;

      case MonitorElement::DQM_KIND_TPROFILE :
	if(xaxis.edges) {
	  me = dqmStore_->bookProfile(name, name, xaxis.nbins, xaxis.edges, yaxis.low, yaxis.high, "");
	}
	else
	  me = dqmStore_->bookProfile(name, name, xaxis.nbins, xaxis.low, xaxis.high, yaxis.low, yaxis.high, "");

	break;

      case MonitorElement::DQM_KIND_TH2F :
	if(xaxis.edges || yaxis.edges) {
	  BinService::AxisSpecs* specs[] = {&xaxis, &yaxis};
	  float* edges[] = {new float[xaxis.nbins + 1], new float[yaxis.nbins + 1]};
	  for(int iSpec(0); iSpec < 2; iSpec++){
	    if(specs[iSpec]->edges){
	      for(int i(0); i < specs[iSpec]->nbins + 1; i++)
		edges[iSpec][i] = specs[iSpec]->edges[i];
	    }
	    else{
	      int nbins(specs[iSpec]->nbins);
	      double low(specs[iSpec]->low), high(specs[iSpec]->high);
	      for(int i(0); i < nbins + 1; i++)
		edges[iSpec][i] = low + (high - low) / nbins * i;
	    }
	  }
	  me = dqmStore_->book2D(name, name, xaxis.nbins, edges[0], yaxis.nbins, edges[1]);
	  for(int iSpec(0); iSpec < 2; iSpec++)
	    delete [] edges[iSpec];
	}
	else
	  me = dqmStore_->book2D(name, name, xaxis.nbins, xaxis.low, xaxis.high, yaxis.nbins, yaxis.low, yaxis.high);

	break;

      case MonitorElement::DQM_KIND_TPROFILE2D :
	if(zaxis.edges) {
	  zaxis.low = zaxis.edges[0];
	  zaxis.high = zaxis.edges[zaxis.nbins];
	}
	if(xaxis.edges || yaxis.edges)
	  throw_("Variable bin size for 2D profile not implemented");
	me = dqmStore_->bookProfile2D(name, name, xaxis.nbins, xaxis.low, xaxis.high, yaxis.nbins, yaxis.low, yaxis.high, zaxis.low, zaxis.high, "");

	break;

      default :
	break;
      }

      if(!me)
	throw_("ME could not be booked");

      if(isHistogram){
	me->setAxisTitle(xaxis.title, 1);
	me->setAxisTitle(yaxis.title, 2);
        if(isMap) me->setAxisTitle(zaxis.title, 3);

        // For plot tagging in RenderPlugin; default values are 1 for both
        // bits 19 - 23 are free in TH1::fBits
        // can only pack object + logical dimensions into 5 bits (4 bits for object, 1 bit for dim (1 -> dim >= 2))
        me->getTH1()->SetBit(uint32_t(actualObject + 1) << 20);
        if(isMap) me->getTH1()->SetBit(0x1 << 19);
      }

      mes_.push_back(me);
    }

    active_ = true;
  }

  bool
  MESetEcal::retrieve() const
  {
    clear();

    std::vector<std::string> mePaths(generatePaths());
    if(mePaths.size() == 0) return false;

    for(unsigned iME(0); iME < mePaths.size(); iME++){
      std::string& path(mePaths[iME]);
      if(path.find('%') != std::string::npos)
        throw_("retrieve() called with incompletely formed path");

      MonitorElement* me(dqmStore_->get(path));
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
  MESetEcal::fill(DetId const& _id, double _x/* = 1.*/, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    fill_(iME, _x, _wy, _w);
  }

  void
  MESetEcal::fill(EcalElectronicsId const& _id, double _x/* = 1.*/, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    fill_(iME, _x, _wy, _w);
  }

  void
  MESetEcal::fill(unsigned _dcctccid, double _x/* = 1.*/, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _dcctccid, btype_));
    checkME_(iME);

    fill_(iME, _x, _wy, _w);
  }

  void
  MESetEcal::fill(double _x, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    if(mes_.size() != 1) return;

    fill_(0, _x, _wy, _w);
  }

  void
  MESetEcal::setBinContent(DetId const& _id, int _bin, double _content)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    mes_[iME]->setBinContent(_bin, _content);
  }

  void
  MESetEcal::setBinContent(EcalElectronicsId const& _id, int _bin, double _content)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    mes_[iME]->setBinContent(_bin, _content);
  }

  void
  MESetEcal::setBinContent(unsigned _dcctccid, int _bin, double _content)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _dcctccid, btype_));
    checkME_(iME);

    mes_[iME]->setBinContent(_bin, _content);
  }

  void
  MESetEcal::setBinError(DetId const& _id, int _bin, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    mes_[iME]->setBinError(_bin, _error);
  }

  void
  MESetEcal::setBinError(EcalElectronicsId const& _id, int _bin, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    mes_[iME]->setBinError(_bin, _error);
  }

  void
  MESetEcal::setBinError(unsigned _dcctccid, int _bin, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _dcctccid, btype_));
    checkME_(iME);

    mes_[iME]->setBinError(_bin, _error);
  }

  void
  MESetEcal::setBinEntries(DetId const& _id, int _bin, double _entries)
  {
    if(!active_) return;
    if(kind_ != MonitorElement::DQM_KIND_TPROFILE && kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    mes_[iME]->setBinEntries(_bin, _entries);
  }

  void
  MESetEcal::setBinEntries(EcalElectronicsId const& _id, int _bin, double _entries)
  {
    if(!active_) return;
    if(kind_ != MonitorElement::DQM_KIND_TPROFILE && kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    mes_[iME]->setBinEntries(_bin, _entries);
  }

  void
  MESetEcal::setBinEntries(unsigned _dcctccid, int _bin, double _entries)
  {
    if(!active_) return;
    if(kind_ != MonitorElement::DQM_KIND_TPROFILE && kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(otype_, _dcctccid, btype_));
    checkME_(iME);

    mes_[iME]->setBinEntries(_bin, _entries);
  }

  double
  MESetEcal::getBinContent(DetId const& _id, int _bin) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getBinContent(_bin);
  }

  double
  MESetEcal::getBinContent(EcalElectronicsId const& _id, int _bin) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getBinContent(_bin);
  }

  double
  MESetEcal::getBinContent(unsigned _dcctccid, int _bin) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _dcctccid, btype_));
    checkME_(iME);

    return mes_[iME]->getBinContent(_bin);
  }

  double
  MESetEcal::getBinError(DetId const& _id, int _bin) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getBinError(_bin);
  }

  double
  MESetEcal::getBinError(EcalElectronicsId const& _id, int _bin) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getBinError(_bin);
  }

  double
  MESetEcal::getBinError(unsigned _dcctccid, int _bin) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _dcctccid, btype_));
    checkME_(iME);

    return mes_[iME]->getBinError(_bin);
  }

  double
  MESetEcal::getBinEntries(DetId const& _id, int _bin) const
  {
    if(!active_) return 0.;
    if(kind_ != MonitorElement::DQM_KIND_TPROFILE && kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getBinEntries(_bin);
  }

  double
  MESetEcal::getBinEntries(EcalElectronicsId const& _id, int _bin) const
  {
    if(!active_) return 0.;
    if(kind_ != MonitorElement::DQM_KIND_TPROFILE && kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getBinEntries(_bin);
  }

  double
  MESetEcal::getBinEntries(unsigned _dcctccid, int _bin) const
  {
    if(!active_) return 0.;
    if(kind_ != MonitorElement::DQM_KIND_TPROFILE && kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    unsigned iME(binService_->findPlot(otype_, _dcctccid, btype_));
    checkME_(iME);

    return mes_[iME]->getBinEntries(_bin);
  }

  int
  MESetEcal::findBin(DetId const& _id, double _x, double _y/* = 0.*/) const
  {
    if(!active_) return -1;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_x, _y);
  }

  int
  MESetEcal::findBin(EcalElectronicsId const& _id, double _x, double _y/* = 0.*/) const
  {
    if(!active_) return -1;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_x, _y);
  }

  int
  MESetEcal::findBin(unsigned _dcctccid, double _x, double _y/* = 0.*/) const
  {
    if(!active_) return -1;

    unsigned iME(binService_->findPlot(otype_, _dcctccid, btype_));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_x, _y);
  }

  bool
  MESetEcal::isVariableBinning() const
  {
    return (xaxis_ && xaxis_->edges) || (yaxis_ && yaxis_->edges) || (zaxis_ && zaxis_->edges);
  }

  std::vector<std::string>
  MESetEcal::generatePaths() const
  {
    using namespace std;

    vector<string> paths(0);

    unsigned nME(binService_->getNObjects(otype_));
    map<string, string> replacements;

    for(unsigned iME(0); iME < nME; iME++) {
      BinService::ObjectType obj(binService_->getObject(otype_, iME));

      string path(path_);

      switch(obj){
      case BinService::kEB:
      case BinService::kEBMEM:
        replacements["subdet"] = "EcalBarrel";
        replacements["prefix"] = "EB";
        replacements["suffix"] = "";
        break;
      case BinService::kEE:
      case BinService::kEEMEM:
        replacements["subdet"] = "EcalEndcap";
        replacements["prefix"] = "EE";
        break;
      case BinService::kEEm:
        replacements["subdet"] = "EcalEndcap";
        replacements["prefix"] = "EE";
        replacements["suffix"] = " EE -";
        break;
      case BinService::kEEp:
        replacements["subdet"] = "EcalEndcap";
        replacements["prefix"] = "EE";
        replacements["suffix"] = " EE +";
        break;
      case BinService::kSM:
        if(iME <= kEEmHigh || iME >= kEEpLow){
          replacements["subdet"] = "EcalEndcap";
          replacements["prefix"] = "EE";
        }
        else{
          replacements["subdet"] = "EcalBarrel";
          replacements["prefix"] = "EB";
        }
	replacements["sm"] = binService_->channelName(iME + 1);
        break;
      case BinService::kEBSM:
        replacements["subdet"] = "EcalBarrel";
        replacements["prefix"] = "EB";
	replacements["sm"] = binService_->channelName(iME + kEBmLow + 1);
        break;
      case BinService::kEESM:
        replacements["subdet"] = "EcalEndcap";
        replacements["prefix"] = "EE";
	replacements["sm"] = binService_->channelName(iME <= kEEmHigh ? iME + 1 : iME + 37);
        break;
      case BinService::kSMMEM:
        {
          unsigned iDCC(memDCCId(iME) - 1);
          //dccId(unsigned) skips DCCs without MEM
          if(iDCC <= kEEmHigh || iDCC >= kEEpLow){
            replacements["subdet"] = "EcalEndcap";
            replacements["prefix"] = "EE";
          }
          else{
            replacements["subdet"] = "EcalBarrel";
            replacements["prefix"] = "EB";
          }
          replacements["sm"] = binService_->channelName(iDCC + 1);
        }
        break;
      case BinService::kEBSMMEM:
        {
          unsigned iDCC(memDCCId(iME + 4) - 1);
          replacements["subdet"] = "EcalBarrel";
          replacements["prefix"] = "EB";
          replacements["sm"] = binService_->channelName(iDCC + 1);
        }
        break;
      case BinService::kEESMMEM:
        {
          unsigned iDCC(memDCCId(iME < 4 ? iME : iME + 36) - 1);
          replacements["subdet"] = "EcalEndcap";
          replacements["prefix"] = "EE";
          replacements["sm"] = binService_->channelName(iDCC + 1);
        }
      default:
	break;
      }

      ecaldqm::formPath(path, replacements);

      paths.push_back(path);
    }

    return paths;
  }

}
