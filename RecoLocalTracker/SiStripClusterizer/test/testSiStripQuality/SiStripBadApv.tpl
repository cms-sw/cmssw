process ICALIB = {
    
    service = MessageLogger {
	untracked vstring destinations = { "cout" }
	untracked PSet cout = { untracked string threshold = "INFO" }
    }

    source = EmptyIOVSource {
	string timetype = "runnumber"
        untracked uint32 firstRun = 1	untracked uint32 lastRun = 1
        uint32 interval = 1
    }
    untracked PSet maxEvents = {untracked int32 input = 1}
  
    
    service = PoolDBOutputService{
	string connect = "sqlite_file:dbfile.db"    
	string timetype = "runnumber"    
	untracked string BlobStreamerName="TBufferBlobStreamingService"
	PSet DBParameters = {
	    untracked string authenticationPath="/afs/cern.ch/cms/DB/conddb"
	}
	
        VPSet toPut={ 
	    { string record = "SiStripBadStrip" string tag = "SiStripBadFiber_v1"} 
	}
    }
        
  module fiber =  SiStripBadFiberBuilder {

	untracked bool   printDebug         = true
	untracked FileInPath file = "CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"

	untracked VPSet BadComponentList = {insert_BadApvList}

	bool SinceAppendMode = true
	string IOVMode	     = "Run"
	string Record        = "SiStripBadStrip"
	bool doStoreOnDB     = true
    }


    path p = {fiber}
        
    module print = AsciiOutputModule {}
    endpath ep = { print }
}

