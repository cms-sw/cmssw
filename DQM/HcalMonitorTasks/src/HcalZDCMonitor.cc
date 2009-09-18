#include "DQM/HcalMonitorTasks/interface/HcalZDCMonitor.h"

// --------------------------- constructor/destructor ---///
HcalZDCMonitor::HcalZDCMonitor(){}
HcalZDCMonitor::~HcalZDCMonitor(){}
void HcalZDCMonitor::reset() {}

void HcalZDCMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe ) 
{
  // Do basic setup procedures here -- get cfg variables, create histograms, etc.
  
  // Setup base class
  HcalBaseMonitor::setup(ps,dbe);


  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if (fVerbosity>0)
    std::cout <<"<HcalZDCMonitor::setup>  Setting up histograms"<<std::endl;

  baseFolder_ = rootFolder_+"ZDCMonitor_Hcal";

  if (fVerbosity>1)
    std::cout <<"<HcalZDCMonitor::setup> Getting variable values from cfg files"<<std::endl;

  // Specify maximum occupancy rate above which a cell is not considered dead 
  deadthresh_ = ps.getUntrackedParameter<double>("ZDCMonitor_deadthresholdrate",0.);
  zdc_checkNevents_ = ps.getUntrackedParameter<int>("ZDCMonitor_checkNevents",checkNevents_);

  //zeroCounters(); // do we need to do that here?  ZDC has small number of channels/histogram bins -- fill on each event?

  if (m_dbe)
    {
      if (fVerbosity>1)
	std::cout <<"<HcalZDCMonitor::setup>  Setting up Histograms"<<std::endl;
      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("ZDC Event Number");
      meEVT_->Fill(ievt_);
      meTOTALEVT_ = m_dbe->bookInt("ZDC Total Events Processed");
      meTOTALEVT_->Fill(tevt_);

      ProblemZDC_=m_dbe->book2D("ProblemZDC", 
				"Problem Rate in ZDCs",
				18,0,18,
				1,0,1);
      (ProblemZDC_->getTH2F())->SetMinimum(0);
      setZDClabels(ProblemZDC_);

      m_dbe->setCurrentFolder(baseFolder_+"/occupancy");
      avgoccZDC_= m_dbe->book1D("ZDC_occupancy", 
				"ZDC Occupancy Rate",
				18,0,18);
      setZDClabels(avgoccZDC_);

      m_dbe->setCurrentFolder(baseFolder_+"/time");
      // make this a TProfile?  How does a TProfile work when combining multiple sources in the client?

      stringstream name;
      avgtimeZDC_= m_dbe->bookProfile("ZDC_time", 
				      "ZDC Average Time",
				      18,0,18,
				      500,-100,400);
      setZDClabels(avgtimeZDC_);
      for (unsigned int side=0;side<2;++side)
	{
	  // side 0 = ZDC+, side 1 = ZDC -
	  for (unsigned int section=0;section<2;++section)
	    {
	      // section 0 = EM, section 1 = HAD
	      for (unsigned int channel=0;channel<5;++channel)
		{
		  // 5 channels for EM, 4 for HAD
		  if (channel==4 && section==1) continue;
		  if (side==0)
		    name <<"ZDC Plus Time ";
		  else
		    name <<"ZDC Minus Time ";
		  if (section==0)
		    name <<"EM ";
		  else
		    name <<"HAD ";
		  name <<"Channel "<<(channel+1);
		  timeZDC_.push_back(m_dbe->book1D(name.str().c_str(),
						   name.str().c_str(),
						   500,-100,400));
		  name.str("");
		} // loop over channels
	    } // loop over sections
	} // loop over sides
      
      m_dbe->setCurrentFolder(baseFolder_+"/energy");
      avgenergyZDC_= m_dbe->bookProfile("ZDC_energy", 
					"ZDC Average Energy",
					18,0,18,
					500,-2,18);
      setZDClabels(avgenergyZDC_);
      for (unsigned int side=0;side<2;++side)
	{
	  // side 1 = ZDC+, side -1 = ZDC -
	  for (unsigned int section=0;section<2;++section)
	    {
	      // section 0 = EM, section 1 = HAD
	      for (unsigned int channel=0;channel<5;++channel)
		{
		  // 5 channels for EM, 4 for HAD
		  if (channel==4 && section==1) continue;
		  if (side==0)
		    name <<"ZDC Plus Energy ";
		  else
		    name <<"ZDC Minus Energy ";
		  if (section==0)
		    name <<"EM ";
		  else
		    name <<"HAD ";
		  name <<"Channel "<<(channel+1);

		  energyZDC_.push_back(m_dbe->book1D(name.str().c_str(),
						     name.str().c_str(),
						     500,-2,18));
		  name.str("");
		} // loop over channels
	    } // loop over sections
	} // loop over sides

      m_dbe->setCurrentFolder(baseFolder_+"/averageX");
      avgXplus_ = m_dbe->book1D("AverageXPlus"," ZDC EM+ <X> (  #sum E_{i}x_{i}/#sum E_{i}  )",
			       100,1,6);
      avgXminus_ = m_dbe->book1D("AverageXMinus"," ZDC EM- <X> (  #sum E_{i}x_{i}/#sum E_{i}  )",
				 100,1,6);
    } // if (m_dbe)


  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalZDCMonitor SETUP -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalZDCMonitor::setup(...)


void HcalZDCMonitor::processEvent(const ZDCDigiCollection& digi,
				  const ZDCRecHitCollection& rechit)
{
  if (fVerbosity>0)
    std::cout <<"<HcalZDCMonitor::processEvent> Processing Event..."<<std::endl;
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  HcalBaseMonitor::processEvent();

  int histindex=-1;
  double EMSumP=0;
  double WeightedSumP=0;
  double EMSumM=0;
  double WeightedSumM=0;

  for (ZDCRecHitCollection::const_iterator iter=rechit.begin();
       iter!=rechit.end();++iter)
    {
      HcalZDCDetId id(iter->id());
      histindex = 9*((id.zside()>0) ? 0: 1); // first 9 index values are for ZDC+, then ZDC-
      histindex+=5*(id.section()-1)+(id.channel()-1); // first 5 channels for EM, then next 4 for HAD
      if (id.section()==1) // found EM
	{
	  // could put in absolute values to protect against summing with negative energy.
	  // not sure if we want these protections or not
	  // for now, don't include them -- negative energies may tell us something useful
	  if (id.zside()>0)
	    {
	      EMSumP+=(iter->energy());
	      WeightedSumP+=(iter->energy())*id.channel();
	    }
	  else
	    {
	      EMSumM+=(iter->energy());
	      WeightedSumM+=(iter->energy())*id.channel();
	    }
	} // if (id.section()==1)
      //std::cout <<"histindex = "<<histindex<<"\ttime = "<<iter->time()<<"\tenergy = "<<iter->energy()<<std::endl;
      avgoccZDC_->Fill(histindex);
      avgtimeZDC_->Fill(histindex,iter->time());
      avgenergyZDC_->Fill(histindex,iter->energy());
      // fill counters keeping track of number of events
      avgoccZDC_->Fill(-1,1);
      if (histindex>=0)
	{
	  if (histindex<(int)timeZDC_.size())
	    timeZDC_[histindex]->Fill(iter->time());
	  if (histindex<(int)energyZDC_.size())
	    energyZDC_[histindex]->Fill(iter->energy());
	}
    } // loop over rechits
  
  if (EMSumP!=0)
    avgXplus_->Fill(WeightedSumP/EMSumP);
  if (EMSumM!=0)
    avgXminus_->Fill(WeightedSumM/EMSumM);

  if (ievt_%zdc_checkNevents_==0)
    fillHistos();

  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalZDCMonitor PROCESS_EVENT -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  
 return;
} // void HcalZDCMonitor::processEvent


void HcalZDCMonitor::fillHistos(void)
{
  if (fVerbosity>0)
    std::cout <<"<HcalZDCMonitor::fillHistos> Filling Histograms..."<<std::endl;

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  // problem cell is bad if occupancy is less than some threshold
  ProblemZDC_->setBinContent(0,ievt_); // event counter
  for (int i=0;i<18;++i)
    {
      ProblemZDC_->setBinContent(i+1,0); // start by assuming no problem
      // each ZDC cell problem is 100% if occupancy rate too low, 0% otherwise
      if ( 1.*avgoccZDC_->getBinContent(i+1)/ievt_ <= deadthresh_) // occupancy rate too low
	ProblemZDC_->setBinContent(i+1,1,ievt_); // fill with total number of problems found
    }


  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalZDCMonitor FILLHISTOS -> "<<cpu_timer.cpuTime()<<std::endl;
    }
 return;
} // void HcalZDCMonitor::fillHistos(void)


void HcalZDCMonitor::done()
{
  return;
}//void HcalZDCMonitor::done()


void HcalZDCMonitor::setZDClabels(MonitorElement* h)
{
  stringstream name;
  for (int side=0;side<2;++side)
    {
      for (int section=0;section<2;++section)
	{
	  for (int channel=1;channel<6;++channel)
	    {
	      if (section==1 && channel==5) continue;
	      if (section==0)
		name<<"EM";
	      else
		name<<"HAD";
	      if (side==0)
		name<<"+";
	      else 
		name<<"-";
	      name<<channel;
	      h->setBinLabel(9*side+5*section+channel,name.str().c_str());
	      name.str("");
	    }
	}
    }
  return;
} //void HcalZDCMonitor::setZDClabels(MonitorElement* h)

