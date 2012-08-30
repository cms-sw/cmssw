#include "DQM/EcalCommon/interface/MESetEcal.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

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

    dqmStore_->setCurrentFolder(dir_);

    if(btype_ == BinService::kReport && name_ == "")
      name_ = dir_.substr(0, dir_.find_first_of('/'));

    std::vector<std::string> meNames(generateNames());

    for(unsigned iME(0); iME < meNames.size(); iME++){
      BinService::ObjectType actualObject(binService_->getObject(otype_, iME));

      BinService::AxisSpecs xaxis, yaxis, zaxis;

      if(logicalDimensions_ > 0){

	if(xaxis_){
	  xaxis = *xaxis_;
	}
	else{ // uses preset
	  bool isMap(logicalDimensions_ > 1);
	  xaxis = binService_->getBinning(actualObject, btype_, isMap, 1, iME);
          if(isMap) yaxis = binService_->getBinning(actualObject, btype_, true, 2, iME);
	}

	if(yaxis_){
	  yaxis = *yaxis_;
	}
        if(yaxis.high - yaxis.low < 1.e-10){
          yaxis.low = -std::numeric_limits<double>::max();
          yaxis.high = std::numeric_limits<double>::max();
        }
	if(zaxis_){
	  zaxis = *zaxis_;
	}
        if(zaxis.high - zaxis.low < 1.e-10){
          zaxis.low = -std::numeric_limits<double>::max();
          zaxis.high = std::numeric_limits<double>::max();
        }
      }

      MonitorElement* me(0);

      switch(kind_) {
      case MonitorElement::DQM_KIND_REAL :
	me = dqmStore_->bookFloat(meNames[iME]);

	break;

      case MonitorElement::DQM_KIND_TH1F :
	if(xaxis.edges){
	  float* edges(new float[xaxis.nbins + 1]);
	  for(int i(0); i < xaxis.nbins + 1; i++)
	    edges[i] = xaxis.edges[i];
	  me = dqmStore_->book1D(meNames[iME], meNames[iME], xaxis.nbins, edges);
	  delete [] edges;
	}
	else
	  me = dqmStore_->book1D(meNames[iME], meNames[iME], xaxis.nbins, xaxis.low, xaxis.high);

	break;

      case MonitorElement::DQM_KIND_TPROFILE :
	if(xaxis.edges) {
	  me = dqmStore_->bookProfile(meNames[iME], meNames[iME], xaxis.nbins, xaxis.edges, yaxis.low, yaxis.high, "");
	}
	else
	  me = dqmStore_->bookProfile(meNames[iME], meNames[iME], xaxis.nbins, xaxis.low, xaxis.high, yaxis.low, yaxis.high, "");

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
	  me = dqmStore_->book2D(meNames[iME], meNames[iME], xaxis.nbins, edges[0], yaxis.nbins, edges[1]);
	  for(int iSpec(0); iSpec < 2; iSpec++)
	    delete [] edges[iSpec];
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
	  throw_("Variable bin size for 2D profile not implemented");
	me = dqmStore_->bookProfile2D(meNames[iME], meNames[iME], xaxis.nbins, xaxis.low, xaxis.high, yaxis.nbins, yaxis.low, yaxis.high, zaxis.low, zaxis.high, "");

	break;

      default :
	break;
      }

      if(!me)
	throw_("ME could not be booked");

      if(logicalDimensions_ > 0){
	me->setAxisTitle(xaxis.title, 1);
	me->setAxisTitle(yaxis.title, 2);
        if(logicalDimensions_ > 1) me->setAxisTitle(zaxis.title, 3);
	// For plot tagging in RenderPlugin; default values are 1 for both
        // Does this method work?
	me->getTH1()->SetMarkerStyle(actualObject + 2);
	me->getTH1()->SetMarkerStyle(btype_ + 2);
      }

      if(logicalDimensions_ == 1 && btype_ == BinService::kDCC){
	if(actualObject == BinService::kEB){
	  for(int iBin(1); iBin <= me->getNbinsX(); iBin++)
	    me->setBinLabel(iBin, binService_->channelName(iBin + kEBmLow));
	}
	else if(actualObject == BinService::kEE){
	  for(int iBin(1); iBin <= me->getNbinsX() / 2; iBin++){
	    me->setBinLabel(iBin, binService_->channelName(iBin));
	    me->setBinLabel(iBin + me->getNbinsX() / 2, binService_->channelName(iBin + 45));
	  }
	}
	else if(actualObject == BinService::kEEm){
	  for(int iBin(1); iBin <= me->getNbinsX(); iBin++)
	    me->setBinLabel(iBin, binService_->channelName(iBin));
	}
	else if(actualObject == BinService::kEEp){
	  for(int iBin(1); iBin <= me->getNbinsX(); iBin++)
	    me->setBinLabel(iBin, binService_->channelName(iBin + 45));
	}
      }
      else if(logicalDimensions_ == 1 && btype_ == BinService::kTriggerTower){
        unsigned dccid(0);
        if(actualObject == BinService::kSM && (iME <= kEEmHigh || iME >= kEEpLow)) dccid = iME + 1;
        else if(actualObject == BinService::kEESM) dccid = iME <= kEEmHigh ? iME + 1 : iME + 37;

        if(dccid > 0){
          std::stringstream ss;
          std::pair<unsigned, unsigned> inner(innerTCCs(iME + 1));
          std::pair<unsigned, unsigned> outer(outerTCCs(iME + 1));
          ss << "TCC" << inner.first << " TT1";
          me->setBinLabel(1, ss.str());
          ss.str("");
          ss << "TCC" << inner.second << " TT1";
          me->setBinLabel(25, ss.str());
          ss.str("");
          ss << "TCC" << outer.first << " TT1";
          me->setBinLabel(49, ss.str());
          ss.str("");
          ss << "TCC" << outer.second << " TT1";
          me->setBinLabel(65, ss.str());
          int offset(0);
          for(int iBin(4); iBin <= 80; iBin += 4){
            if(iBin == 28) offset = 24;
            else if(iBin == 52) offset = 48;
            else if(iBin == 68) offset = 64;
            ss.str("");
            ss << iBin - offset;
            me->setBinLabel(iBin, ss.str());
          }
        }
      }
      else if(logicalDimensions_ == 2 && btype_ == BinService::kCrystal){
        if(actualObject == BinService::kMEM){
          for(int iBin(1); iBin <= me->getNbinsX(); ++iBin)
            me->setBinLabel(iBin, binService_->channelName(memDCCId(iBin - 1)));
        }
        if(actualObject == BinService::kEBMEM){
          for(int iBin(1); iBin <= me->getNbinsX(); ++iBin)
            me->setBinLabel(iBin, binService_->channelName(memDCCId(iBin - 5)));
        }
        if(actualObject == BinService::kEEMEM){
          for(int iBin(1); iBin <= me->getNbinsX() / 2; ++iBin){
            me->setBinLabel(iBin, binService_->channelName(memDCCId(iBin - 1)));
            me->setBinLabel(iBin + me->getNbinsX() / 2, binService_->channelName(memDCCId(iBin + 39)));
          }
        }
      }
      else if(logicalDimensions_ == 2 && btype_ == BinService::kDCC){
        if(actualObject == BinService::kEcal){
          me->setBinLabel(1, "EE", 2);
          me->setBinLabel(6, "EE", 2);
          me->setBinLabel(3, "EB", 2);
          me->setBinLabel(5, "EB", 2);
        }
      }
      
      mes_.push_back(me);
    }

    // To avoid the ambiguity between "content == 0 because the mean is 0" and "content == 0 because the entry is 0"
    // RenderPlugin must be configured accordingly
    if(kind_ == MonitorElement::DQM_KIND_TPROFILE2D && logicalDimensions_ == 2)
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

  std::vector<std::string>
  MESetEcal::generateNames() const
  {
    using namespace std;

    std::vector<std::string> names(0);

    unsigned nME(binService_->getNObjects(otype_));

    for(unsigned iME(0); iME < nME; iME++) {
      BinService::ObjectType obj(binService_->getObject(otype_, iME));

      string name(name_);
      string spacer(" ");

      if(btype_ == BinService::kProjEta) name += " eta";
      else if(btype_ == BinService::kProjPhi) name += " phi";
      else if(btype_ == BinService::kReport) spacer = "_";

      switch(obj){
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
	name += spacer + binService_->channelName(iME + 1); break;
      case BinService::kEBSM:
	name += spacer + binService_->channelName(iME + 10); break;
      case BinService::kEESM:
	name += spacer + binService_->channelName(iME <= kEEmHigh ? iME + 1 : iME + 37); break;
      case BinService::kSMMEM:
	//dccId(unsigned) skips DCCs without MEM
	name += spacer + binService_->channelName(memDCCId(iME)); break;
      case BinService::kEBSMMEM:
	name += spacer + binService_->channelName(memDCCId(iME + 9)); break;
      case BinService::kEESMMEM:
	name += spacer + binService_->channelName(memDCCId(iME <= kEEmHigh ? iME : iME + 36)); break;
      default:
	break;
      }

      names.push_back(name);
    }

    return names;
  }

}
