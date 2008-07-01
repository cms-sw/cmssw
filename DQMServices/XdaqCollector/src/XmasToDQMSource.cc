/*
*/

#include "DQMServices/XdaqCollector/interface/XmasToDQMSource.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TRandom.h" // this is just the random number generator
#include <math.h>

#include <boost/tokenizer.hpp>
//#include <set>

#include <time.h>

using namespace std;
using namespace edm;


#define BXBIN 3564
#define WCBIN 257


typedef boost::tokenizer<boost::char_separator<char> > tokenizer;


//
// constructors and destructor
//
XmasToDQMSource::XmasToDQMSource( const edm::ParameterSet& ps ) :
counterEvt_(0)
{
    cout << "Constructor of XmasToDQMSource called...." << endl;
     
     dbe_ = Service<DQMStore>().operator->();
     parameters_ = ps;
     monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
     
     /*NBINS = parameters_.getUntrackedParameter<string>("NBINS","YourSubsystemName");
     XMIN = parameters_.getUntrackedParameter<string>("XMIN","YourSubsystemName");
     XMAX = parameters_.getUntrackedParameter<string>("XMAX","YourSubsystemName");*/
     
    cout << "NBINS = " << NBINS << endl;
    cout << "XMIN = " << XMIN << endl;
    cout << "XMAX = " << XMAX << endl;
     
     cout << "Monitor name = " << monitorName_ << endl;
     if (monitorName_ != "" ) 
     	monitorName_ = monitorName_+"/" ;
     
     prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
     cout << "===> DQM event prescale = " << prescaleEvt_ << " events "<< endl;
 
     if(NBINS == "")
     	NBINS = "50";
	
     if(XMIN == "")
     	XMIN = "0";
	
     if(XMAX == "")
     	XMAX = "2000";
 
/// book some histograms here
  //const int NBINS = 50; XMIN = 0; XMAX = 20000;
  
    // create and cd into new folder
  //dbe_->setCurrentFolder(/*monitorName_+*/"wse");
  //h1 = dbe_->book1D("histo", "Example 1D histogram.", NBINS, XMIN, XMAX);
  //h1->setAxisTitle("x-axis title", 1);
  //h1->setAxisTitle("y-axis title", 2);
  
  // assign tag to MEs h1, h2 and h7
  //const unsigned int detector_id = 17;
  //dbe_->tag(h1, detector_id);
}


XmasToDQMSource::~XmasToDQMSource()
{
   
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void XmasToDQMSource::beginJob(const EventSetup& context){

}

//--------------------------------------------------------
void XmasToDQMSource::beginRun(const edm::Run& r, const EventSetup& context) {

}

//--------------------------------------------------------
void XmasToDQMSource::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
     const EventSetup& context) {
  
}

// ----------------------------------------------------------
void XmasToDQMSource::analyze(const Event& iEvent, 
			       const EventSetup& iSetup )
{  
	time_t start,end;
	static int times_called=0;
	
	times_called++;
	
	
	std::cout << "inside Analyze.... " << std::endl;
	
	std::map<std::string, std::string, std::less<std::string> >::iterator i;
	
	//cout << "DQMSourceExample::analyze before BSem_.takeO()" << endl;
	
	//xmas2dqm::wse::ToDqm::instance()->BSem_.take();

	std::cout << "inside DQMSource::Analyze...ready to lock the data mutex" << std::endl;
	//protect access to the queue
	pthread_mutex_lock(&xmas2dqm::wse::ToDqm::instance()->mymutex_);
    	
	
	std::cout << "inside DQMSource::Analyze...check (...and possible wait) if data queue is empty" << std::endl;
	
	//check if the queue is empty and wait (a signal that informs that an element has been pushed)
	while(xmas2dqm::wse::ToDqm::instance()->/*QTable_*/MemoryTable_.size() <= 0)
	{
        	pthread_cond_wait(&xmas2dqm::wse::ToDqm::instance()->more_, &xmas2dqm::wse::ToDqm::instance()->mymutex_);
	}
	
	
	std::cout << "inside DQMSource::Analyze...data queue has elements...proceeding..." << std::endl;
	
	start = time(NULL);
	
	//xdata::Table::Reference ref_table;
	xdata::Table *ref_table = NULL;
	
	
	if(xmas2dqm::wse::ToDqm::instance()->/*QTable_*/MemoryTable_.size() > 0)
	{
		//cout << " DQMSourceExample::analyze : Queue size  > 0 " << xmas2dqm::wse::ToDqm::instance()->QTable_.size() << endl;
		
		//pop an element from the queue of the LAS data
		ref_table = xmas2dqm::wse::ToDqm::instance()->/*QTable_*/ MemoryTable_.front();
		//xmas2dqm::wse::ToDqm::instance()->QTable_.pop();		
		
	}
	
	//Insert data to histograms transfered to DQM GUI servers (print the table)
	if(ref_table != NULL)
	{
		size_t row = ref_table->getRowCount();

		for ( size_t r = 0; r < ref_table->numberOfRows_; r++ )
		{
			/* remove prints for benchmarking*/
			/*
			std::cout << "********* Printing table inside DQMSourceExample ***************" << std::endl;
			std:: cout << ref_table->columnData_["context"]->elementAt(r)->toString() << std::endl;
			std:: cout << ref_table->columnData_["slotNumber"]->elementAt(r)->toString() << std::endl;*/
			
			if(ref_table->columnData_["bxHistogram"]->elementAt(r)->toString() == "[]")
			{
				/* remove prints for benchmarking*/
				/*std::cout << ref_table->columnData_["context"]->elementAt(r)->toString() << " has empty bxHistogram" << std::endl;*/
				continue;
			}
			
			//boost::tokenizer<> Context_tokens(ref_table->columnData_["Context"]->elementAt(r)->toString());
			
			boost::char_separator<char> sep(":/.");
    			tokenizer Context_tokens(ref_table->columnData_["Context"]->elementAt(r)->toString(), sep);
		
		
			//check if the combination Host + slotNumber exists already in the set of hosts + slotNumbers
			//if not book a new histogram with correspondent name and push data, else push data to existent histogram
			//std::string host_slot = *(++Context_tokens.begin()) + "_" + ref_table->columnData_["slotNumber"]->elementAt(r)->toString();
			
			std::string host_slot;

			//host_slot = *(++ Context_tokens.begin()) + "-" + *(++ ++ Context_tokens.begin()) + "-" + *(++ ++ ++Context_tokens.begin()) + "_" + ref_table->columnData_["slotNumber"]->elementAt(r)->toString();
			host_slot = *(++ Context_tokens.begin()) + "_" + ref_table->columnData_["slotNumber"]->elementAt(r)->toString();
		
			//host_slot = host_slot + "_" + ref_table->columnData_["slotNumber"]->elementAt(r)->toString();
			
		
			if( HostSlotMap.find(host_slot) == HostSlotMap.end())
			{
				/* remove prints for benchmarking*/
				/*std::cout << "booking new histogram..." << host_slot << std::endl;*/
			
				HostSlotMap[host_slot] = new /*struct*/ Data();
				
			
				HostSlotMap[host_slot]->lastTimestamp = ref_table->columnData_["timestamp"]->elementAt(r)->toString();
				
				// create and cd into new folder for bxHistograms
  				dbe_->setCurrentFolder(/*monitorName_+*/"bxHisto");
				HostSlotMap[host_slot]->bxHistogram1D = dbe_->book1D("bx_"+ host_slot, "FRL histo", BXBIN/*atoi(NBINS.c_str())*/, 0/*atoi(XMIN.c_str())*/, BXBIN-1/*atoi(XMAX.c_str())*/);
				
	  			HostSlotMap[host_slot]->bxHistogram1D->setAxisTitle("BX"/*"x-axis title"*/, 1);
  				HostSlotMap[host_slot]->bxHistogram1D->setAxisTitle("Events"/*"y-axis title"*/, 2);
			
				/* remove prints for benchmarking*/
				/*std::cout << "booked histogram = " << host_slot << std::endl;*/
				boost::tokenizer<> bxHistogram_values(ref_table->columnData_["bxHistogram"]->elementAt(r)->toString());
   	    			
				int ibx=0; //bx counter - bin counter
				
				for(boost::tokenizer<>::iterator itok=bxHistogram_values.begin(); itok!=bxHistogram_values.end();++itok)
	    			{
					ibx++;
					//remove for benchmarking
       					/*std::cout << *itok << std::endl;*/
					string s = *itok;
					//HostSlotMap[host_slot]->Fill(atoi(s.c_str()));
					
					HostSlotMap[host_slot]->bxHistogram1D->setBinContent(ibx-1,atoi(s.c_str()));					
					if(ibx >= BXBIN)
						break;
   				}
				
				
				// create and cd into new folder for wcHistograms
  				dbe_->setCurrentFolder(/*monitorName_+*/"wcHisto");
				HostSlotMap[host_slot]->wcHistogram1D = dbe_->book1D("wc_"+ host_slot, "FRL histo", WCBIN/*atoi(NBINS.c_str())*/, 0/*atoi(XMIN.c_str())*/, WCBIN-1/*atoi(XMAX.c_str())*/);
				
	  			HostSlotMap[host_slot]->wcHistogram1D->setAxisTitle("WC"/*"x-axis title"*/, 1);
  				HostSlotMap[host_slot]->wcHistogram1D->setAxisTitle("Words"/*"y-axis title"*/, 2);
			
				/* remove prints for benchmarking*/
				/*std::cout << "booked histogram = " << host_slot << std::endl;*/
				boost::tokenizer<> wcHistogram_values(ref_table->columnData_["wcHistogram"]->elementAt(r)->toString());
   	    			
				int iwc=0; //wc counter - bin counter
				
				for(boost::tokenizer<>::iterator itok=wcHistogram_values.begin(); itok!=wcHistogram_values.end();++itok)
	    			{
					iwc++;
					//remove for benchmarking
       					/*std::cout << *itok << std::endl;*/
					string s = *itok;
					//HostSlotMap[host_slot]->Fill(atoi(s.c_str()));
					
					HostSlotMap[host_slot]->wcHistogram1D->setBinContent(iwc-1,atoi(s.c_str()));					
					if(iwc >= WCBIN)
						break;
   				}
				
						
			}
			else
			{
			
				//check if the timestamp has changed and proceed adding data only if timestamp has changed
				if(HostSlotMap[host_slot]->lastTimestamp == ref_table->columnData_["timestamp"]->elementAt(r)->toString())
				{
					std::cout << host_slot << " same timestamp found..." << std::endl;
					continue;
				}
				else
				{
					std::cout << host_slot << " different timestamp found..." << std::endl;
					HostSlotMap[host_slot]->lastTimestamp == ref_table->columnData_["timestamp"]->elementAt(r)->toString();
				}
				
				
				
				/* remove prints for benchmarking*/
				/*std::cout << "histogram..." << host_slot << " already booked..." << std::endl;*/
				//insert bxHistogram values
				boost::tokenizer<> bxHistogram_values(ref_table->columnData_["bxHistogram"]->elementAt(r)->toString());
   	    			
				int ibx=0; //bx counter
				for(boost::tokenizer<>::iterator itok=bxHistogram_values.begin(); itok!=bxHistogram_values.end();++itok)
	    			{
					ibx++;
       					/*remove for benchmarking */
					//std::cout << *itok << std::endl;
					string s = *itok;
					//HostSlotMap[host_slot]->Fill(atoi(s.c_str()));
					
					HostSlotMap[host_slot]->bxHistogram1D->setBinContent(ibx-1,atoi(s.c_str()));					
					if(ibx >= BXBIN)
						break;
   				}
				
				
				//insert wcHistogram values
				boost::tokenizer<> wcHistogram_values(ref_table->columnData_["wcHistogram"]->elementAt(r)->toString());
   	    			
				int iwc=0; //wc counter - bin counter
				
				for(boost::tokenizer<>::iterator itok=wcHistogram_values.begin(); itok!=wcHistogram_values.end();++itok)
	    			{
					iwc++;
					//remove for benchmarking
       					/*std::cout << *itok << std::endl;*/
					string s = *itok;
					//HostSlotMap[host_slot]->Fill(atoi(s.c_str()));
					
					HostSlotMap[host_slot]->wcHistogram1D->setBinContent(iwc-1,atoi(s.c_str()));					
					if(iwc >= WCBIN)
						break;
   				}
				
			}
		}
	}	
	
	
	xmas2dqm::wse::ToDqm::instance()->MemoryTable_.pop();
	
	std::cout << "after poping from the data Queue...."<< std::endl;
	
	if(ref_table !=NULL)
	{
		ref_table->~Table();
	}
	
	std::cout << "after calling xdata::Table::Reference destructor...."<< std::endl;
	
	delete ref_table ;
	
	std::cout << "after call of delete...."<< std::endl;
	
	
	end = time(NULL);
	
	std::cout << "time called = " << times_called << " time in seconds needed = " << (end - start) << std::endl;
	
	//cout << "DQMSourceExample::analyze before BSem_.give()" << endl;
	
	//signal that a new element has been inserted
	pthread_cond_signal(&xmas2dqm::wse::ToDqm::instance()->less_);
	
	std::cout << "after signaligng less...." << std::endl;
	
	//allow access to the queue
    	pthread_mutex_unlock(&xmas2dqm::wse::ToDqm::instance()->mymutex_);
	
	std::cout << "after unlocking the mutex...." << std::endl;
	//xmas2dqm::wse::ToDqm::instance()->BSem_.give();
	
	

  	counterEvt_++;
  	if (prescaleEvt_ > 0 && counterEvt_%prescaleEvt_!=0) return;
  	// cout << " processing conterEvt_: " << counterEvt_ <<endl;
  
  	/*if(counterEvt_%100 == 0)
  	{
    		cout << " # of events = " << counterEvt_ << endl;
		dbe_->save("/tmp/thehisto.root","/wse");
  	}*/
  	
	std::cout << "returning from XmasToDQMSource::analyze...." << std::endl;
	//usleep(100);
	//sleep(1000);

}




//--------------------------------------------------------
void XmasToDQMSource::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
                                          const EventSetup& context) {
}
//--------------------------------------------------------
void XmasToDQMSource::endRun(const Run& r, const EventSetup& context){

  
  //dbe_->setCurrentFolder(/*monitorName_+*/"wse");

}
//--------------------------------------------------------
void XmasToDQMSource::endJob(){
}


