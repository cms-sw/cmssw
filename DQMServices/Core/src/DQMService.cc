#include "DQMServices/Core/src/DQMService.h"
#include "DQMServices/Core/src/DQMRootBuffer.h"
#include "DQMServices/Core/interface/DQMNet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElementRootT.h"
#include "DQMServices/Core/src/DQMTagHelper.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <iostream>
#include <string>
#include <memory>

DQMService::DQMService(const edm::ParameterSet &pset, edm::ActivityRegistry &ar)
  : bei_(&*edm::Service<DaqMonitorBEInterface>()),
    net_(0)
{
  ar.watchPostProcessEvent(this, &DQMService::flush);
  ar.watchPostEndJob(this, &DQMService::shutdown);

  std::string host = pset.getUntrackedParameter<std::string>("collectorHost", ""); 
  int port = pset.getUntrackedParameter<int>("collectorPort", 9090);
  bool verbose = pset.getUntrackedParameter<bool>("verbose", false);

  if (host != "" && port > 0)
  {
    net_ = new DQMBasicNet;
    net_->debug(verbose);
    net_->updateToCollector(host, port);
    net_->start();
  }
}

DQMService::~DQMService(void)
{
  shutdown();
}

// Flush updates to the network layer at the end of each event.  This
// is the only point at which the main application and the network
// layer interact outside initialisation and exit.
void
DQMService::flush(const edm::Event &, const edm::EventSetup &)
{
  typedef MonitorElementT<TNamed> ROOTObj;

  if (net_)
  {
    // Lock the network layer so we can modify the data.
    net_->lock();
    uint64_t version = lat::Time::current().ns();
    bool updated = false;

    // Find updated contents and update the network cache.
    dqm::me_util::dir_it i, e;
    dqm::me_util::cME_it ci, ce;
    for (i = bei_->Own.paths.begin(), e = bei_->Own.paths.end(); i != e; ++i)
    {
      MonitorElementRootFolder *folder = i->second;
      if (! folder)
	continue;

      std::string path = folder->getPathname();
      for (ci = folder->objects_.begin(), ce = folder->objects_.end(); ci != ce; ++ci)
      {
	MonitorElement *me = ci->second;
	if (me && me->wasUpdated())
	{
	  DQMNet::Object o;
	  const std::string &name = ci->first;

	  o.version = version;
	  o.name.reserve(path.size() + name.size() + 2);
	  o.name += path;
	  o.name += '/';
	  o.name += name;
	  o.object = 0;
	  o.reference = 0;
	  o.flags = DQMNet::DQM_FLAG_NEW;
	  o.lastreq = 0;

	  dqm::me_util::dirt_it idir = bei_->allTags.find(path);
	  if (idir != bei_->allTags.end())
	  {
	    dqm::me_util::tags_it itag = idir->second.find(name);
	    if (itag != idir->second.end())
	    {
	      o.tags.resize(itag->second.size());
	      std::copy(itag->second.begin(), itag->second.end(), o.tags.begin());
	    }
	  }

	  DQMRootBuffer buffer (DQMRootBuffer::kWrite);

	  if (me->hasError())
	    o.flags |= DQMNet::DQM_FLAG_REPORT_ERROR;
	  if (me->hasWarning())
	    o.flags |= DQMNet::DQM_FLAG_REPORT_WARNING;
	  if (me->hasOtherReport())
	    o.flags |= DQMNet::DQM_FLAG_REPORT_OTHER;
    
	  // Save the ROOT object.  This is either a genuine ROOT object,
	  // or a scalar one that stores its value as TObjString.
	  if (ROOTObj *ob = dynamic_cast<ROOTObj *>(me))
	  {
	    if (TObject *tobj = ob->operator->())
	    {
	      buffer.WriteObject(tobj);
	      if (dynamic_cast<TObjString *> (tobj))
		o.flags |= DQMNet::DQM_FLAG_SCALAR;
	    }
	  }
	  else if (FoldableMonitor *ob = dynamic_cast<FoldableMonitor *>(me))
	  {
	    if (TObject *tobj = ob->getTagObject())
	    {
	      buffer.WriteObject(tobj);
	      if (dynamic_cast<TObjString *> (tobj))
		o.flags |= DQMNet::DQM_FLAG_SCALAR;
	    }
	  }

	  if (! buffer.Length())
	  {
	    std::cerr
	      << "ERROR: The DQM object '" << o.name
	      << "' is neither a ROOT object nor a recognised simple object.\n";
	    return;
	  }

	  // Get the reference object.
	  if (MonitorElement *meref = bei_->getReferenceME(me))
	  {
	    FoldableMonitor	*fob;
	    ROOTObj		*rob;
	    TObject		*tobj;

	    if ((rob = dynamic_cast<ROOTObj *>(meref)) && (tobj = rob->operator->()))
	      buffer.WriteObject(tobj);
	    else if ((fob = dynamic_cast<FoldableMonitor *>(meref)) && (tobj = fob->getTagObject()))
	      buffer.WriteObject(tobj);
	    else
	    {
	      std::cerr
		<< "ERROR: The DQM reference object for '" << o.name
		<< "' is neither a ROOT object nor a recognised simple object.\n";
	      buffer.WriteObjectAny(0, 0);
	    }
	  }
	  else
	    buffer.WriteObjectAny(0, 0);

	  // Save the quality test results.
	  typedef dqm::qtests::QR_map QRMap;
	  typedef dqm::qtests::qr_it QRIter;
	  QRMap qreports = me->getQReports();
	  for (QRIter i = qreports.begin(), e = qreports.end(); i != e; ++i)
	    if (FoldableMonitor *qr = dynamic_cast<FoldableMonitor *>(i->second))
	      if (TObject *tobj = qr->getTagObject())
		buffer.WriteObject(tobj);

	  // Save this ensemble to the buffer.
	  o.rawdata.resize(buffer.Length());
	  memcpy(&o.rawdata[0], buffer.Buffer(), buffer.Length());
	  net_->updateLocalObject(o);
	  updated = true;
	}
      }
    }

    // Find removed contents and clear the network cache.
    dqm::me_util::monit_it ri, re;
    dqm::me_util::csIt rmi, rme;
    for (ri = bei_->removedContents.begin(), re = bei_->removedContents.end(); ri != re; ++ri)
      for (rmi = ri->second.begin(), rme = ri->second.end(); rmi != rme; ++rmi, updated = true)
	net_->removeLocalObject(ri->first + '/' + *rmi);

    // Tell network to flush if we updated something.
    if (updated)
      net_->sendLocalChanges();

    // Unlock the network layer.
    net_->unlock();
  }
}

// Disengage the network service.
void
DQMService::shutdown(void)
{
  if (net_)
    net_->shutdown();
}
