/*
*/

#include "DQMServices/XdaqCollector/interface/XmasToDQMSource.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TRandom.h" // this is just the random number generator
#include <math.h>

#include <boost/tokenizer.hpp>
//#include <set>
#include <sstream>

#include <time.h>



using namespace std;
using namespace edm;


#define BXBIN 3564
#define WCBIN 257

//minimun event fragment size = header (8 bytes) + trailer (8 bytes) + zero payload = 16
#define MIN_EVENT_FRAGMENT_SIZE 16


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
     monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","DAQ");
     
     cout << "Monitor name = " << monitorName_ << endl;
     if (monitorName_ != "" ) 
     	monitorName_ = monitorName_+"/" ;
     
     prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
     cout << "===> DQM event prescale = " << prescaleEvt_ << " events "<< endl;
 
 
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
	/*time_t start,end;
	static int times_called=0;
	
	times_called++;*/
	
	
	//std::cout << "inside Analyze.... " << std::endl;
	
	std::map<std::string, std::string, std::less<std::string> >::iterator i;
	
	//cout << "DQMSourceExample::analyze before BSem_.takeO()" << endl;
	
	//xmas2dqm::wse::ToDqm::instance()->BSem_.take();

	std::cout << "inside DQMSource::Analyze...ready to lock the data mutex" << std::endl;
	//protect access to the queue
	pthread_mutex_lock(&xmas2dqm::wse::ToDqm::instance()->LASmutex_);
    	
	
	std::cout << "inside DQMSource::Analyze...check (...and possible wait) if data queue is empty" << std::endl;
	
	//check if the queue is empty and wait (a signal that informs that an element has been pushed)
	while(xmas2dqm::wse::ToDqm::instance()->/*QTable_*/MemoryTable_.size() <= 0)
	{
        	pthread_cond_wait(&xmas2dqm::wse::ToDqm::instance()->more_, &xmas2dqm::wse::ToDqm::instance()->LASmutex_);
	}
	
	
	std::cout << "inside DQMSource::Analyze...data queue has elements...proceeding..." << std::endl;
	
	//start = time(NULL);
	
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

		for ( size_t r = 0; r < ref_table->numberOfRows_; r++ )
		{
			
			//check if the  flashlist contains the element we want to monitor
			if(!ref_table->columnData_[xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element])
			{
				break;
			}
		
			/* remove prints for benchmarking*/
			/*
			std::cout << "********* Printing table inside DQMSourceExample ***************" << std::endl;
			std:: cout << ref_table->columnData_["context"]->elementAt(r)->toString() << std::endl;
			std:: cout << ref_table->columnData_["slotNumber"]->elementAt(r)->toString() << std::endl;*/
			
			//if(ref_table->columnData_["wcHistogram"]->elementAt(r)->toString() == "[]")
			if(ref_table->columnData_[xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString()]->elementAt(r)->toString() == "[]")
			{
				/* remove prints for benchmarking*/
				/*std::cout << ref_table->columnData_["context"]->elementAt(r)->toString() << " has empty bxHistogram" << std::endl;*/
				continue;
			}
			
			
			//check if there is a column runNumber in the LAS table
			if(ref_table->columnData_["runNumber"])
			{
				/* remove prints for benchmarking*/
				
				
				//xmas2dqm::wse::ToDqm::instance()->BSem_.take();
				
				//if runNumber in LAS record different than currnet runNumber go to next LAS record
				if (xmas2dqm::wse::ToDqm::instance()->runNumber_.toString() != ref_table->columnData_["runNumber"]->elementAt(r)->toString())
				{
					continue;
				}
				
				
				//xmas2dqm::wse::ToDqm::instance()->BSem_.give();
				//std::cout << "runNumber ... = " << ref_table->columnData_["runNumber"]->elementAt(r)->toString() << std::endl;
			}	
			
			//boost::tokenizer<> Context_tokens(ref_table->columnData_["Context"]->elementAt(r)->toString());
			
			boost::char_separator<char> sep(":/.");
    			tokenizer Context_tokens(ref_table->columnData_["Context"]->elementAt(r)->toString(), sep);
		
		
			//check if the combination Host + slotNumber exists already in the set of hosts + slotNumbers
			//if not book a new histogram with correspondent name and push data, else push data to existent histogram
			//std::string host_slot = *(++Context_tokens.begin()) + "_" + ref_table->columnData_["slotNumber"]->elementAt(r)->toString();
			
			std::string host_slot;
			host_slot = *(++ Context_tokens.begin());

			//check if there is a column slotNumber in the LAS table in order to use as key for the histogram map the combination of host+slot
			//useful mostly for bxHistogram, wcHistogram of frlHisto, where the histograms (flashlist elements) refer to host+slot
			if(ref_table->columnData_["slotNumber"])
			{
				//host_slot = *(++ Context_tokens.begin()) + "-" + *(++ ++ Context_tokens.begin()) + "-" + *(++ ++ ++Context_tokens.begin()) + "_" + ref_table->columnData_["slotNumber"]->elementAt(r)->toString();
				host_slot = host_slot + "_" + ref_table->columnData_["slotNumber"]->elementAt(r)->toString();
			}
		
			//host_slot = host_slot + "_" + ref_table->columnData_["slotNumber"]->elementAt(r)->toString();
			
		
			//check if there is no entry in the map for this host (+slot in case of wcHistogram, bxHistogram)
			if( HostSlotMap.find(host_slot) == HostSlotMap.end())
			{
				/* remove prints for benchmarking*/
				std::cout << "booking new histogram..." << host_slot << std::endl;
			
				HostSlotMap[host_slot] = new /*struct*/ Data();
				
			
				HostSlotMap[host_slot]->lastTimestamp = ref_table->columnData_["timestamp"]->elementAt(r)->toString();
				
				// create and cd into new folder 
  				//dbe_->setCurrentFolder(monitorName_ + "wcHisto");
				dbe_->setCurrentFolder(monitorName_ + xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString());
				
				//the wcHistogramResolution equals the value of the register Histogram of the FRL, not the value of the bytes resolution for the bin
				// the value of the register multiplied by 16 gives the byte resolution - range of a wcHistogram bin
				// if(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString() == "wcHistogram")
// 				{
// 					HostSlotMap[host_slot]->Histogram1D = dbe_->book1D("wc_"+ host_slot, "FRL wcHisto", WCBIN, 
// 					MIN_EVENT_FRAGMENT_SIZE, MIN_EVENT_FRAGMENT_SIZE + WCBIN*16*atoi(ref_table->columnData_["wcHistogramResolution"]->elementAt(r)->toString().c_str()));
// 					HostSlotMap[host_slot]->Histogram1D->setAxisTitle("Event fragment size (Bytes)"/*"x-axis title"*/, 1);
//   					HostSlotMap[host_slot]->Histogram1D->setAxisTitle("Events"/*"y-axis title"*/, 2);
// 				}
// 				else if(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString() == "bxHistogram")
// 				{
//  					HostSlotMap[host_slot]->Histogram1D = dbe_->book1D("bx_"+ host_slot, "FRL bxHisto", BXBIN, 1, BXBIN);
//  				
//  	  				HostSlotMap[host_slot]->Histogram1D->setAxisTitle("LHC orbit Bunch"/*"x-axis title"*/, 1);
//    					HostSlotMap[host_slot]->Histogram1D->setAxisTitle("Events"/*"y-axis title"*/, 2);
// 				}
				
				std::istringstream str2num;
  				int nbins;
				double xmin,xmax;

				str2num.str(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.bins.toString());
				str2num >> nbins; // now stream is in end of file state
				str2num.clear(); // clear end of file state

				str2num.str(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.xmin.toString());
				str2num >> xmin; // now stream is in end of file state
				str2num.clear(); // clear end of file state

				str2num.str(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.xmax.toString());
				str2num >> xmax; // now stream is in end of file state
				str2num.clear(); // clear end of file state

				HostSlotMap[host_slot]->Histogram1D = dbe_->book1D(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString() + "_" + host_slot, "", nbins, xmin, xmax);
				HostSlotMap[host_slot]->Histogram1D->setAxisTitle(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.xtitle.toString()/*"x-axis title"*/, 1);
  				HostSlotMap[host_slot]->Histogram1D->setAxisTitle(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.ytitle.toString()/*"y-axis title"*/, 2);
				
			
				/* remove prints for benchmarking*/
				/*std::cout << "booked histogram = " << host_slot << std::endl;*/
				
				
				boost::char_separator<char> histo_sep("[,]");
				tokenizer Histogram_values(ref_table->columnData_[xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString()]->elementAt(r)->toString(), histo_sep);
   	    			
				int iwc=0; //bin counter
				
				for(tokenizer::iterator itok=Histogram_values.begin(); itok!=Histogram_values.end();++itok)
	    			{
					iwc++;
					//remove for benchmarking
       					//std::cout << "iwc = "<< iwc << " *itok = " << *itok << std::endl;
					string s = *itok;
					//std::cout << "iwc = "<< iwc << " s = " << s << std::endl;
					//HostSlotMap[host_slot]->Fill(atoi(s.c_str()));
					
					std::istringstream istrfloat(s);
   					float bin_value;
   					istrfloat >> bin_value;
					
					//std::cout << "iwc = "<< iwc << " bin_value = " << bin_value << std::endl;
					
					if(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString() == "wcHistogram" || xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString() == "bxHistogram")
					{
						HostSlotMap[host_slot]->Histogram1D->setBinContent(iwc-1, bin_value/*atoi(s.c_str())*/);
					}
					else
					{
						HostSlotMap[host_slot]->Histogram1D->Fill(bin_value);
					}					
					
					if(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString() == "wcHistogram" && iwc >= nbins /*WCBIN*/)
						break;
						
					if(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString() == "bxHistogram" && iwc >= nbins /*BXBIN*/)
						break;
   				}
				
						
			}
			else
			{
			
				std::istringstream str2num;
  				int nbins;

				str2num.str(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.bins.toString());
				str2num >> nbins; // now stream is in end of file state
				str2num.clear(); // clear end of file state
				
				
				//check if the timestamp has changed and proceed adding data only if timestamp has changed
				if(HostSlotMap[host_slot]->lastTimestamp == ref_table->columnData_["timestamp"]->elementAt(r)->toString())
				{
					//std::cout << host_slot << " same timestamp found..." << std::endl;
					continue;
				}
				else
				{
					//std::cout << host_slot << " different timestamp found..." << std::endl;
					HostSlotMap[host_slot]->lastTimestamp == ref_table->columnData_["timestamp"]->elementAt(r)->toString();
				}
				
				
				
				//insert wcHistogram values
				boost::char_separator<char> histo_sep("[,]");
				tokenizer Histogram_values(ref_table->columnData_[xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString()]->elementAt(r)->toString(), histo_sep);
   	    			
				int iwc=0; //bin counter
				
				for(tokenizer::iterator itok=Histogram_values.begin(); itok!=Histogram_values.end();++itok)
	    			{
					iwc++;
					//remove for benchmarking
       					//std::cout << "fill booked histogram iwc = "<< iwc << " *itok = " << *itok << std::endl;
					
					string s = *itok;
					//HostSlotMap[host_slot]->Fill(atoi(s.c_str()));
					
					std::istringstream istrfloat(s);
   					float bin_value;
   					istrfloat >> bin_value;
					
					if(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString() == "wcHistogram" || xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString() == "bxHistogram")
					{
						HostSlotMap[host_slot]->Histogram1D->setBinContent(iwc-1, bin_value/*atoi(s.c_str())*/);
					}
					else
					{
						HostSlotMap[host_slot]->Histogram1D->Fill(bin_value);
					}					

					if(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString() == "wcHistogram" && iwc >= nbins/*WCBIN*/)
						break;
						
					if(xmas2dqm::wse::ToDqm::instance()->flashlistMonitor_.bag.element.toString() == "bxHistogram" && iwc >= nbins /*BXBIN*/)
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
	
	//std::cout << "after calling xdata::Table::Reference destructor...."<< std::endl;
	
	delete ref_table ;
	
	//std::cout << "after call of delete...."<< std::endl;
	
	
	/*end = time(NULL);
	
	std::cout << "time called = " << times_called << " time in seconds needed = " << (end - start) << std::endl;*/
	
	//cout << "DQMSourceExample::analyze before BSem_.give()" << endl;
	
	//signal that a new element has been inserted
	pthread_cond_signal(&xmas2dqm::wse::ToDqm::instance()->less_);
	
	//std::cout << "after signaligng less...." << std::endl;
	
	//allow access to the queue
    	pthread_mutex_unlock(&xmas2dqm::wse::ToDqm::instance()->LASmutex_);
	
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
  	
	//std::cout << "returning from XmasToDQMSource::analyze...." << std::endl;
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


