#ifndef __BU_h__
#define __BU_h__

#include "interface/evb/include/i2oEVBMsgs.h"
#include "interface/shared/include/i2oXFunctionCodes.h"
#include "interface/shared/include/frl_header.h"
#include "interface/shared/include/fed_header.h"
#include "interface/shared/include/fed_trailer.h"

#include "toolbox/include/toolbox/mem/HeapAllocator.h"
#include "toolbox/include/toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/include/toolbox/net/URN.h"

#include "i2o/include/i2o/Method.h"
#include "i2o/utils/include/i2o/utils/AddressMap.h"
#include "xdata/include/xdata/SimpleType.h"
#include "xdata/include/xdata/UnsignedInteger32.h"
#include "xdata/include/xdata/Boolean.h"
#include "xdata/include/xdata/String.h"

#include "xdaq/include/xdaq/WebApplication.h"

#include "CLHEP/Random/RandGauss.h"
#include "extern/i2o/include/i2o/i2oDdmLib.h"

/**
 * \ingroup xdaqApps
 * \brief Builder unit (BU)
 */
class BU :
public xdaq::WebApplication
{
public:

  XDAQ_INSTANTIATOR();
    /**
     * Constructor.
     */
  BU(xdaq::ApplicationStub *s) : xdaq::WebApplication(s),
				 eventHandle_(0),
				 dataBufSize_(4096),
				 fragSize_(1024), 
				 //means average size of fed fragment...
				 nbRUs_(64),
				 buCapacity_(64),
                                 nbEventsSent_(0),
				 useSimulInput_(false),  
				 simulInputFile_("input.raw"),  
				 useFixedSize_(false),
    /*                                 eventFactory_(0),*/
				 pool_(0)
  {

    

    xdata::InfoSpace *is = getApplicationInfoSpace();
    is->fireItemAvailable("dataBufSize" ,   &dataBufSize_ );
    is->fireItemAvailable("fragSize"    ,   &fragSize_    );
    is->fireItemAvailable("nbRUs"       ,   &nbRUs_       );
    is->fireItemAvailable("buCapacity"  ,   &buCapacity_  );
    is->fireItemAvailable("nbEventsSent",   &nbEventsSent_);
    is->fireItemAvailable("useSimulInput",  &useSimulInput_);
    is->fireItemAvailable("simulInputFile", &simulInputFile_);
    is->fireItemAvailable("useFixedSize",   &useFixedSize_);
    
    i2o::bind
      (
       this,
       &BU::buAllocateNMsg,
       I2O_BU_ALLOCATE,
       XDAQ_ORGANIZATION_ID
       );
    
    i2o::bind
        (
	 this,
	 &BU::buCollectMsg,
	 I2O_BU_COLLECT,
	 XDAQ_ORGANIZATION_ID
	 );

    i2o::bind
      (
       this,
       &BU::buDiscardNMsg,
       I2O_BU_DISCARD,
       XDAQ_ORGANIZATION_ID
       );

    try
      {
	toolbox::mem::HeapAllocator *allocator = new toolbox::mem::HeapAllocator();
	toolbox::net::URN urn("toolbox-mem-pool", "ABU");
	toolbox::mem::MemoryPoolFactory *poolFactory =
	  toolbox::mem::getMemoryPoolFactory();
	
	pool_ = poolFactory->createPool(urn, allocator);

	LOG4CPLUS_INFO(getApplicationLogger(),
		       "Created memory pool: " << "ABU");
      }
    catch (toolbox::mem::exception::Exception& e)
      {
	string s = "Failed to create pool: ABU ";
	
	LOG4CPLUS_FATAL(getApplicationLogger(), s);
	XCEPT_RETHROW(xcept::Exception, s, e);
      }
    
  }

private:


    unsigned char payloadNb_;

    U32 eventHandle_;

    xdata::UnsignedInteger32 dataBufSize_;   ///< Exported parameter
    xdata::UnsignedInteger32 fragSize_;      ///< Exported parameter
    xdata::UnsignedInteger32 nbRUs_;         ///< Exported parameter
    xdata::UnsignedInteger32 buCapacity_;    ///< Exported parameter
    xdata::UnsignedInteger32 nbEventsSent_;  ///< Exported parameter
    xdata::Boolean      useSimulInput_; ///< Exported parameter
    xdata::String       simulInputFile_;///< Exported parameter
    xdata::Boolean      useFixedSize_;  ///< Exported parameter

    /**
     * I2O callback routine invoked when a request for N resources has been
     * received from a FU.  Generates a dummy event for each resource.
     */
  void buAllocateNMsg(toolbox::mem::Reference *bufRef)
  {
    LOG4CPLUS_DEBUG(getApplicationLogger(),
		   "Received buAllocate:simulated Input " << useSimulInput_);

    // If using simulated input, initialize factory
    if(useSimulInput_ /*&& (eventFactory_==0)*/)
	{
	}

        I2O_MESSAGE_FRAME *stdMsg = (I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
        I2O_BU_ALLOCATE_MESSAGE_FRAME *msg =
            (I2O_BU_ALLOCATE_MESSAGE_FRAME*)stdMsg;
	toolbox::mem::Reference  *superFrag  = 0;
        unsigned int             rqstIndex   = 0;
        U8                ruInstNb    = 0;
        BU_ALLOCATE       rqst;
        // For each allocate request from the FU
        for(rqstIndex = 0; rqstIndex < msg->n; rqstIndex++)
        {
	  rqst = msg->allocate[rqstIndex];
	  /*
	    DaqEvent *ev_ = 0;

	    // if reading from simulated input, get a new event
	    if(useSimulInput_)
	      {	      ev_ = eventFactory_->rqstEvent();}
	    while(ev_==0)
	      {	     
		eventFactory_->initializeSource(simulInputFile_); // for the moment
		ev_ = eventFactory_->rqstEvent();
	      }

	  */
	    // For each super fragment to be in the dummy event
            for(ruInstNb=0; ruInstNb<nbRUs_; ruInstNb++)
            {
                // Create and send the super fragment

                superFrag = createSuperFrag
                (
                    dataBufSize_,             // dataBufSize
                    fragSize_,                // fragSize
                    stdMsg->InitiatorAddress, // fuTid
                    eventHandle_,             // eventHandle
                    rqst.fuTransactionId,     // fuTransaction
                    ruInstNb,                 // currentFragment
                    nbRUs_                   // totalFragments
		    //		    ev_
                );

		debug(superFrag/*,ev_*/);
		I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *frame =
		  (I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)(superFrag->getDataLocation());
		superFrag->setDataSize(frame->PvtMessageFrame.StdMessageFrame.MessageSize << 2);
		xdaq::ApplicationDescriptor *d = 
		  i2o::utils::getAddressMap()->getApplicationDescriptor(stdMsg->InitiatorAddress);
		getApplicationContext()->postFrame(superFrag, this->getApplicationDescriptor(), d);

            }

            eventHandle_ = (eventHandle_ + 1) % buCapacity_;
            nbEventsSent_.value_++;
        }

        // Free the request message from the FU
        bufRef->release();
    }


    /**
     * I2O callback routine invoked when a request for "more" of an event has
     * been received from a FU.
     */
    void buCollectMsg(toolbox::mem::Reference *bufRef)
    {
        LOG4CPLUS_FATAL(this->getApplicationLogger(), "buCollectMsg() NOT IMPLEMENTED");
        exit(-1);
    }


    /**
     * I2O callback routine invoked when a request to discard N BU resources
     * has been received from a FU.
     */
    void buDiscardNMsg(toolbox::mem::Reference *bufRef)
    {
        // Does nothing but free the incoming I2O message
        bufRef->release();
    }


    /**
     * Creates and returns a chain of blocks/messages representing a dummy
     * super fragment.
     */
    toolbox::mem::Reference *createSuperFrag
      (
       const size_t  dataBufSize,
       const U32     fragSize,
       const I2O_TID fuTid,
       const U32     eventHandle,
       const U32     fuTransaction,
       const U8      currentFragment,
       const U8      totalFragments
       //       DaqEvent *ev_ = 0
       );
    /** extract fed sizes from a log-normal distribution of given mean and width 
     */
  void generateNFedFragments(float aves, float width, 
			     vector<int> &ffed)
  {
    for(unsigned int i = 0; i < ffed.size(); i++)
      {
	float logsiz = RandGauss::shoot(log(aves),log(aves)-log(width/2.));
	float siz = exp(logsiz);
	int isiz = int(siz);
	isiz -= isiz % 8; // all blocks aligned to 64 bit words
	ffed[i] = isiz;
      }
  }

  /** estimate number of blocks needed for a superfragment
   */
  int estimateNBlocks(vector<int> &sfed, size_t fullBlockPayload);


  /** debug a chain (superfragment
   */
  void debug(toolbox::mem::Reference* ref/*, DaqEvent *ev*/)
  {

    vector<toolbox::mem::Reference*> chain;
    toolbox::mem::Reference *nn = ref;
    chain.push_back(ref);
    int ind = 1;
    while((nn=nn->getNextReference())!=0)
      {
	chain.push_back(nn);
	ind++;
      }
	  
    unsigned long blocks_adrs[chain.size()];
    for(unsigned int i = 0; i< chain.size(); i++)
      {
	blocks_adrs[i] = (unsigned long) chain[i]->getDataLocation()+
	  sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
      }
    // call method to unwind data structure and check H/T content 
    int ierr = check_event_data(blocks_adrs,chain.size()/*,ev*/);
    if(ierr!=0) cerr << "ERROR::check_event_data, code = " << ierr << endl;
  }

  int check_event_data (unsigned long blocks_adrs[], int nmb_blocks/*,
								     DaqEvent *ev*/); 
  //  int check_fed(char *, int, int/*, DaqEvent */); 
  toolbox::mem::Pool *pool_;

  void dumpFrame(unsigned char* data, unsigned int len)
  {

  
    //      PI2O_MESSAGE_FRAME  ptrFrame = (PI2O_MESSAGE_FRAME)data;
    //      printf ("\nMessageSize: %d Function %d\n",ptrFrame->MessageSize, ptrFrame->Function);
      
    char left1[20];
    char left2[20];
    char right1[20];
    char right2[20];
      
    //  LOG4CPLUS_ERROR(adapter_->getApplicationLogger(),toolbox::toString("Byte  0  1  2  3  4  5  6  7\n"));
    printf ("Byte  0  1  2  3  4  5  6  7\n");
      
    int c = 0;
    int pos = 0;
      
      
    for (unsigned int i = 0; i < (len/8); i++) {
      int rpos = 0;
      int off = 3;
      for (pos = 0; pos < 12; pos += 3) {
	sprintf (&left1[pos],"%2.2x ", ((unsigned char*)data)[c+off]);
	sprintf (&right1[rpos],"%1c", ((data[c+off] > 32) && (data[c+off] < 127)) ? data[c+off]: '.' );
	sprintf (&left2[pos],"%2.2x ", ((unsigned char*)data)[c+off+4]);
	sprintf (&right2[rpos],"%1c", ((data[c+off+4] > 32) && (data[c+off+4] < 127)) ? data[c+off+4]: '.' );
	rpos++;
	off--;
      }
      c+=8;
      //    LOG4CPLUS_ERROR(adapter_->getApplicationLogger(),toolbox::toString("%4d: %s  ||  %s \n", c-8, left, right));
      printf ("%4d: %s%s ||  %s%s  %x\n", c-8, left1, left2, right1, right2, (int)&data[c-8]);
	
    }
      
    fflush(stdout);	
      
      
  }


  int compareFrame(char* data1, int len1, char* data2, int len2)
  {

    int retval = 0;
    //      PI2O_MESSAGE_FRAME  ptrFrame = (PI2O_MESSAGE_FRAME)data;
    //      printf ("\nMessageSize: %d Function %d\n",ptrFrame->MessageSize, ptrFrame->Function);
      
    char left[40];
    char right[40];
      
    //  LOG4CPLUS_ERROR(adapter_->getApplicationLogger(),toolbox::toString("Byte  0  1  2  3  4  5  6  7\n"));
      
    int c = 0;
    int pos = 0;
      
      
    for (int i = 0; i < (len1/8); i++) {
      int rpos = 0;
      for (pos = 0; pos < 8*3; pos += 3) {
	sprintf (&left[pos],"%2.2x ", ((unsigned char*)data1)[c]);
	sprintf (&right[pos],"%2.2x ", ((unsigned char*)data2)[c]);
	rpos += 1;
	c++;
      }
	
      //    LOG4CPLUS_ERROR(adapter_->getApplicationLogger(),toolbox::toString("%4d: %s  ||  %s \n", c-8, left, right));
      if(strcmp(left,right)!=0)
	{
	  printf ("Byte  0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7\n");
	  printf ("%4d: %s  ||  %s *****\n", c-8, left, right);
	  retval++;
	}
    }
      
    fflush(stdout);	
    return retval;
      
  }



};


#endif
