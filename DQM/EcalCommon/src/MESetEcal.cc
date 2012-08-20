#include "DQM/EcalCommon/interface/MESetEcal.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include <limits>

namespace ecaldqm
{

  MESetEcal::MESetEcal(MEData const& _data, int _logicalDimensions) :
    MESet(_data),
    logicalDimensions_(_logicalDimensions)
  {
    if(data_->btype == BinService::kUser && ((logicalDimensions_ > 0 && !data_->xaxis) || (logicalDimensions_ > 1 && !data_->yaxis)))
      throw_("Need axis specifications");
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

    if(data_->btype == BinService::kReport && name_ == "")
      name_ = dir_.substr(0, dir_.find_first_of('/'));

    std::vector<std::string> meNames(generateNames());

    for(unsigned iME(0); iME < meNames.size(); iME++){
      BinService::ObjectType actualObject(binService_->getObject(data_->otype, iME));

      BinService::AxisSpecs xaxis, yaxis, zaxis;

      if(logicalDimensions_ > 0){
	if(data_->xaxis){
	  xaxis = *data_->xaxis;
	}
	else{ // uses preset
	  bool isMap(logicalDimensions_ > 1);
	  xaxis = binService_->getBinning(actualObject, data_->btype, isMap, 1, iME);
	  if(isMap) yaxis = binService_->getBinning(actualObject, data_->btype, true, 2, iME);
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
	// For plot tagging in RenderPlugin; default values are 1 for both
        // Does this method work?
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
  MESetEcal::fill(DetId const& _id, double _x, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    fill_(iME, _x, _wy, _w);
  }

  void
  MESetEcal::fill(EcalElectronicsId const& _id, double _x, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    fill_(iME, _x, _wy, _w);
  }

  void
  MESetEcal::fill(unsigned _dcctccid, double _x, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    fill_(iME, _x, _wy, _w);
  }

  void
  MESetEcal::setBinContent(DetId const& _id, int _bin, double _content)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    mes_[iME]->setBinContent(_bin, _content);
  }

  void
  MESetEcal::setBinContent(EcalElectronicsId const& _id, int _bin, double _content)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    mes_[iME]->setBinContent(_bin, _content);
  }

  void
  MESetEcal::setBinContent(unsigned _dcctccid, int _bin, double _content)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    mes_[iME]->setBinContent(_bin, _content);
  }

  void
  MESetEcal::setBinError(DetId const& _id, int _bin, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    mes_[iME]->setBinError(_bin, _error);
  }

  void
  MESetEcal::setBinError(EcalElectronicsId const& _id, int _bin, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    mes_[iME]->setBinError(_bin, _error);
  }

  void
  MESetEcal::setBinError(unsigned _dcctccid, int _bin, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    mes_[iME]->setBinError(_bin, _error);
  }

  void
  MESetEcal::setBinEntries(DetId const& _id, int _bin, double _entries)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    mes_[iME]->setBinEntries(_bin, _entries);
  }

  void
  MESetEcal::setBinEntries(EcalElectronicsId const& _id, int _bin, double _entries)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    mes_[iME]->setBinEntries(_bin, _entries);
  }

  void
  MESetEcal::setBinEntries(unsigned _dcctccid, int _bin, double _entries)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    mes_[iME]->setBinEntries(_bin, _entries);
  }

  double
  MESetEcal::getBinContent(DetId const& _id, int _bin) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    return mes_[iME]->getBinContent(_bin);
  }

  double
  MESetEcal::getBinContent(EcalElectronicsId const& _id, int _bin) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    return mes_[iME]->getBinContent(_bin);
  }

  double
  MESetEcal::getBinContent(unsigned _dcctccid, int _bin) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    return mes_[iME]->getBinContent(_bin);
  }

  double
  MESetEcal::getBinError(DetId const& _id, int _bin) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    return mes_[iME]->getBinError(_bin);
  }

  double
  MESetEcal::getBinError(EcalElectronicsId const& _id, int _bin) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    return mes_[iME]->getBinError(_bin);
  }

  double
  MESetEcal::getBinError(unsigned _dcctccid, int _bin) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    return mes_[iME]->getBinError(_bin);
  }

  double
  MESetEcal::getBinEntries(DetId const& _id, int _bin) const
  {
    if(!active_) return 0.;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    return mes_[iME]->getBinEntries(_bin);
  }

  double
  MESetEcal::getBinEntries(EcalElectronicsId const& _id, int _bin) const
  {
    if(!active_) return 0.;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    return mes_[iME]->getBinEntries(_bin);
  }

  double
  MESetEcal::getBinEntries(unsigned _dcctccid, int _bin) const
  {
    if(!active_) return 0.;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    return mes_[iME]->getBinEntries(_bin);
  }

  int
  MESetEcal::findBin(DetId const& _id, double _x, double _y/* = 0.*/) const
  {
    if(!active_) return -1;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_x, _y);
  }

  int
  MESetEcal::findBin(EcalElectronicsId const& _id, double _x, double _y/* = 0.*/) const
  {
    if(!active_) return -1;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_x, _y);
  }

  int
  MESetEcal::findBin(unsigned _dcctccid, double _x, double _y/* = 0.*/) const
  {
    if(!active_) return -1;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_x, _y);
  }

  std::vector<std::string>
  MESetEcal::generateNames() const
  {
    using namespace std;

    std::vector<std::string> names(0);

    unsigned nME(binService_->getNObjects(data_->otype));

    for(unsigned iME(0); iME < nME; iME++) {
      BinService::ObjectType obj(binService_->getObject(data_->otype, iME));

      string name(name_);
      string spacer(" ");

      if(data_->btype == BinService::kProjEta) name += " eta";
      else if(data_->btype == BinService::kProjPhi) name += " phi";
      else if(data_->btype == BinService::kReport) spacer = "_";

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
