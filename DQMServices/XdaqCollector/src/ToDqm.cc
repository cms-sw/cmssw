#include "DQMServices/XdaqCollector/interface/ToDqm.h"
//#include "FWCore/ServiceRegistry/interface/Service.h"


xmas2dqm::wse::ToDqm * xmas2dqm::wse::ToDqm::instance_ = 0;
xmas2dqm::wse::ToDqm * xmas2dqm::wse::ToDqm::instance()
{
	if(instance_ == 0) instance_ = new xmas2dqm::wse::ToDqm(); return instance_;
}

xmas2dqm::wse::ToDqm::ToDqm() : messageCount_(0),BSem_(toolbox::BSem::FULL)
{
	//std::cout << "ToDqm constructor called.... " << std::endl;
	pthread_mutex_init(&LASmutex_,NULL);
	pthread_cond_init(&more_,NULL);
	pthread_cond_init(&less_,NULL);
 
  
}

xmas2dqm::wse::ToDqm::~ToDqm()
{

	//The implementation has detected an attempt to destroy the object referenced by mutex while 
	//it is locked or referenced (for example, while being used in a pthread_cond_timedwait() or pthread_cond_wait()) by another thread.
	
	//pthread_mutex_unlock(&mymutex_);
	 //pthread_mutex_destroy(&mymutex_);
}


void  xmas2dqm::wse::ToDqm::digest
	(
		const std::string& flashListName, 
		const std::string& originator, 
		const std::string& tag, 
		/*xdata::Table::Reference table*/
		xdata::Table *table
	) 
	throw (xmas2dqm::wse::exception::Exception)
{

	std::cout << "inside digest...." << std::endl;
	
	std::map<std::string, std::string, std::less<std::string> >::iterator i;
	
	//std::cout<< "ToDQM->digest : before BSem_.take();" << std::endl;
	//BSem_.take();
	//acquire the mutex - protect access to the queue
	//pthread_mutex_lock(&mymutex_);
   
	//check if the queue is full and wait (a signal that informs that an element has been poped)
	// until there is 'space' in the queue    
    	//while (QTable_.size() >= Qsize_max)
	//{
        	//pthread_cond_wait(&less_, &mymutex_);
	//}
	
	//push new element to the queue of LAS data
	//QTable_.push(table);
	MemoryTable_.push(table);
	
	//MemoryTable_.push(mtable);
	
	//std::cout << "ToDQM->digest : Queue size = " << QTable_.size() << std::endl;
	
	//std::cout<< "ToDQM->digest : before BSem_.give();" << std::endl;
	
	//signal that a new element has been inserted
	//pthread_cond_signal(&more_);

	//allow access to the queue
    	//pthread_mutex_unlock(&mymutex_);
	//BSem_.give();
	
	messageCount_++;
	 
}

void xmas2dqm::wse::ToDqm::free_memory()
{	
	std::cout << "free_memory: before MemoryTable.front()" << std::endl;
	
	xdata::Table * temp = MemoryTable_.front();
	delete temp;
	
	//MemoryTable_.front()->xdata::Table::~Table() ;
	
	//MemoryTable_.front()->clear();
	
	
	std::cout << "free_memory: before MemoryTable.pop()" << std::endl;
	MemoryTable_.pop();
	
	std::cout << "free_memory: returning...." << std::endl;
}

