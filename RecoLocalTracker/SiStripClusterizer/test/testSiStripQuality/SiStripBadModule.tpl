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
	    { string record = "SiStripBadStrip" string tag = "SiStripBadModule_v1"} 
	}
    }
        
    module mod =  SiStripBadModuleByHandBuilder {

	untracked bool   printDebug         = true
	untracked FileInPath file = "CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"

	untracked vuint32 BadModuleList = {  insert_BadModuleList }
	
	bool SinceAppendMode = true
	string IOVMode	     = "Run"
	string Record        = "SiStripBadStrip"
	bool doStoreOnDB     = true
    }

    path p = { mod }
        
    module print = AsciiOutputModule {}
    endpath ep = { print }
}

